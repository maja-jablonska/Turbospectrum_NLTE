#!/usr/bin/env python3
"""Verify that parameters stay consistent across every pipeline stage.

Reads a pipeline JSON config (the same file passed to
`pipeline_from_config.py` / the PBS wrappers), resolves the artifact
locations it implies, then spot-checks that parameters line up across:

  1. Grid zarr                       (parameter_grid.zarr)
  2. Shard zarr files                (spectra_shard_*.zarr)
  3. Merged zarr                     (synthesized_spectra.zarr) if present
  4. Turbospectrum intermediates     (.opac / .spec / .log on disk) if present

Each check is independent — missing artifacts are reported as "skipped"
rather than a failure, so this script can run at any point in the
pipeline's lifecycle (post-grid, post-shard, post-merge).

The most important invariant it guards is the one introduced by the
turbvel <-> t_value rebind: for every synthesized row,
`t_value == nearest(turbvel, available atmosphere labels)`.

Exit code is 0 if no check failed, 1 otherwise.  Use `--json` for a
machine-readable summary.

Usage:
    python3 scripts/verify_pipeline_consistency.py \
        --config configs/pipeline/config_regular_grid_nlte_fe_intensity_5000.json
    python3 scripts/verify_pipeline_consistency.py --config ... --sample 200 --json
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for path in (REPO_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_turbospectrum import (  # noqa: E402
    _coerce_microturbulence_value,
    _nearest_turb_label,
    _scan_available_turb_labels,
    get_synthesis_output_stem_from_params,
)


################################################################################
# Result reporting
################################################################################

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    status: str
    summary: str = ""
    issues: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != STATUS_FAIL


def _skip(name: str, reason: str) -> CheckResult:
    return CheckResult(name=name, status=STATUS_SKIP, summary=reason)


def _fail(name: str, summary: str, issues: Sequence[str], stats: Optional[Dict[str, Any]] = None) -> CheckResult:
    return CheckResult(
        name=name,
        status=STATUS_FAIL,
        summary=summary,
        issues=list(issues),
        stats=dict(stats or {}),
    )


def _pass(name: str, summary: str, stats: Optional[Dict[str, Any]] = None) -> CheckResult:
    return CheckResult(name=name, status=STATUS_PASS, summary=summary, stats=dict(stats or {}))


################################################################################
# Config resolution (mirrors pipeline_from_config.py / PBS wrappers)
################################################################################


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _abs_from(base_dir: str, raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return raw if os.path.isabs(raw) else os.path.abspath(os.path.join(base_dir, raw))


def _abs_output_from(run_root: str, cfg_dir: str, raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    if os.path.isabs(raw):
        return raw
    if not raw.startswith((".", "..")):
        base = run_root or cfg_dir
        return os.path.abspath(os.path.join(base, raw))
    return os.path.abspath(os.path.join(cfg_dir, raw))


@dataclass
class PipelinePaths:
    config_path: str
    config_dir: str
    run_root: str
    grid_zarr: Optional[str]
    shards_dir: Optional[str]
    shard_template: Optional[str]
    merged_zarr: Optional[str]
    synthesis_config_path: Optional[str]
    model_atmosphere_path: Optional[str]
    output_dir: Optional[str]
    log_dir: Optional[str]
    model_opac_dir: Optional[str]
    project_root: Optional[str]


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged.get(key), value) if key in merged else copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _resolve_paths(config_path: str) -> PipelinePaths:
    config_path = os.path.abspath(config_path)
    cfg = _load_json(config_path)
    cfg_dir = os.path.dirname(config_path)
    outputs = dict(cfg.get("outputs") or {})

    run_root = _abs_from(cfg_dir, outputs.get("run_root")) or ""
    grid_zarr = _abs_output_from(run_root, cfg_dir, outputs.get("grid_zarr"))
    merged_zarr = _abs_output_from(run_root, cfg_dir, outputs.get("spectra_zarr"))
    shard_template = _abs_output_from(run_root, cfg_dir, outputs.get("spectra_shard_template"))
    shards_dir = os.path.dirname(shard_template) if shard_template else None

    ts_section = dict(cfg.get("turbospectrum") or {})
    overrides = ts_section.get("overrides")
    base_synth_path_raw = ts_section.get("config")
    base_synth_path = _abs_from(cfg_dir, base_synth_path_raw) if base_synth_path_raw else None
    synth_cfg: Dict[str, Any] = {}
    if base_synth_path and os.path.isfile(base_synth_path):
        synth_cfg = _load_json(base_synth_path)
    if isinstance(overrides, dict):
        synth_cfg = _deep_merge(synth_cfg, overrides)

    def _synth_path(key: str) -> Optional[str]:
        paths = synth_cfg.get("paths") or {}
        raw = paths.get(key) if isinstance(paths, dict) else None
        if not raw:
            raw = synth_cfg.get(key)
        return _abs_from(cfg_dir, raw) if raw else None

    project_root = _abs_from(cfg_dir, synth_cfg.get("project_root")) or REPO_ROOT
    model_atmosphere_path = _synth_path("model_atmosphere_path")
    output_dir = _synth_path("output_dir")
    log_dir = _synth_path("log_dir")
    model_opac_dir = synth_cfg.get("model_opac_dir") or "COM/contopac"
    if not os.path.isabs(model_opac_dir):
        model_opac_dir = os.path.abspath(os.path.join(project_root, model_opac_dir))

    return PipelinePaths(
        config_path=config_path,
        config_dir=cfg_dir,
        run_root=run_root,
        grid_zarr=grid_zarr,
        shards_dir=shards_dir,
        shard_template=shard_template,
        merged_zarr=merged_zarr,
        synthesis_config_path=base_synth_path,
        model_atmosphere_path=model_atmosphere_path,
        output_dir=output_dir,
        log_dir=log_dir,
        model_opac_dir=model_opac_dir,
        project_root=project_root,
    )


################################################################################
# Zarr helpers
################################################################################


def _open_zarr(path: str):
    import zarr

    return zarr.open_group(path, mode="r")


def _is_valid_zarr(path: Optional[str]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    try:
        _open_zarr(path)
        return True
    except Exception:
        return False


def _load_columns(root, names: Sequence[str], indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in names:
        if name not in root:
            continue
        arr = root[name]
        data = np.asarray(arr[:] if indices is None else arr[indices])
        out[name] = data
    return out


def _row_count(root) -> int:
    for key in ("teff", "global_index", "flux"):
        if key in root:
            return int(np.asarray(root[key]).shape[0])
    raise ValueError("zarr store has no recognizable row-keyed array")


def _sample_indices(n: int, k: int, seed: int = 0) -> np.ndarray:
    if n <= k:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


################################################################################
# Check: grid zarr matches config axes (best-effort, regular grids only)
################################################################################


def _parse_numeric_axis_spec(raw: Any) -> Optional[np.ndarray]:
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        if len(parts) != 3:
            return None
        try:
            start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return None
        if step <= 0 or stop < start:
            return None
        return np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    try:
        values = np.asarray(sorted({float(t) for t in tokens}), dtype=np.float64)
    except ValueError:
        return None
    return values


def _parse_abundance_spec(raw: Any) -> Optional[List[str]]:
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    if ":" in text:
        values = _parse_numeric_axis_spec(text)
        if values is None:
            return None
        return [f"{v:+.2f}" for v in values]
    try:
        return [f"{float(text):+.2f}"]
    except ValueError:
        return [text]


def check_grid_matches_config(paths: PipelinePaths, cfg: Mapping[str, Any]) -> CheckResult:
    name = "grid zarr vs. config axes"
    if not _is_valid_zarr(paths.grid_zarr):
        return _skip(name, f"grid zarr not present: {paths.grid_zarr}")

    root = _open_zarr(paths.grid_zarr)
    grid_section = cfg.get("grid") or {}
    axes = (grid_section.get("axes") or {}) if isinstance(grid_section, dict) else {}
    abundances_cfg = (grid_section.get("abundances") or {}) if isinstance(grid_section, dict) else {}
    synthesis_cfg = (grid_section.get("synthesis") or {}) if isinstance(grid_section, dict) else {}

    issues: List[str] = []
    stats: Dict[str, Any] = {"rows": _row_count(root)}

    # Numeric axes
    for axis_name in ("teff", "logg", "feh"):
        spec = axes.get(axis_name)
        expected = _parse_numeric_axis_spec(spec)
        if expected is None or axis_name not in root:
            continue
        actual = np.unique(np.asarray(root[axis_name][:]).astype(np.float64))
        diff = np.setdiff1d(np.round(actual, 6), np.round(expected, 6))
        if diff.size:
            issues.append(
                f"{axis_name}: grid has {diff[:5].tolist()} not covered by config spec {spec!r}"
            )

    # Abundances
    for key, raw in abundances_cfg.items():
        expected = _parse_abundance_spec(raw)
        if expected is None or key not in root:
            continue
        actual = sorted({str(v) for v in np.asarray(root[key][:]).tolist()})
        if actual != sorted(set(expected)):
            issues.append(f"abundance {key!r}: grid={actual} vs config={sorted(set(expected))}")

    # Synthesis scalars / mu axis
    for key in ("lam_min", "lam_max", "lam_step"):
        expected = synthesis_cfg.get(key)
        if expected is None or key not in root:
            continue
        actual = np.asarray(root[key][:]).astype(np.float64)
        if not np.allclose(actual, float(expected)):
            issues.append(f"{key}: grid has variation {np.unique(actual)[:5].tolist()}, config={expected}")

    mu_range = synthesis_cfg.get("mu_range")
    if mu_range is not None:
        expected_mu = _parse_numeric_axis_spec(mu_range)
        if "mu" not in root:
            issues.append(
                f"mu_range={mu_range!r} in config but grid zarr has no 'mu' column — "
                "Intensity rows will collapse to duplicate filenames."
            )
        elif expected_mu is not None:
            actual_mu = np.unique(np.asarray(root["mu"][:]).astype(np.float64))
            diff = np.setdiff1d(np.round(actual_mu, 6), np.round(expected_mu, 6))
            if diff.size:
                issues.append(f"mu: grid has {diff[:5].tolist()} not covered by config {mu_range!r}")

    for mode_key, column_key in (("output_mode", "output_mode"), ("calculation_mode", "calculation_mode"), ("mode", "mode")):
        expected = synthesis_cfg.get(mode_key)
        if expected is None or column_key not in root:
            continue
        actual = {str(v) for v in np.asarray(root[column_key][:]).tolist() if str(v)}
        if actual and {str(expected)} != actual:
            issues.append(f"{mode_key}: grid={sorted(actual)} vs config={expected!r}")

    if issues:
        return _fail(name, f"{len(issues)} inconsistency(ies) between config and grid zarr", issues, stats)
    return _pass(name, f"grid zarr {stats['rows']:,} rows consistent with config axes/abundances/synthesis", stats)


################################################################################
# Check: shard metadata matches grid at global_index
################################################################################


def _discover_shards(paths: PipelinePaths) -> List[str]:
    if not paths.shards_dir or not os.path.isdir(paths.shards_dir):
        return []
    return sorted(
        os.path.join(paths.shards_dir, entry)
        for entry in os.listdir(paths.shards_dir)
        if entry.startswith("spectra_shard_") and entry.endswith(".zarr")
        and os.path.isdir(os.path.join(paths.shards_dir, entry))
    )


def _value_equal(grid_value: Any, shard_value: Any) -> bool:
    if isinstance(grid_value, (bytes, bytearray)):
        grid_value = grid_value.decode("utf-8", errors="ignore")
    if isinstance(shard_value, (bytes, bytearray)):
        shard_value = shard_value.decode("utf-8", errors="ignore")
    try:
        return bool(np.isclose(float(grid_value), float(shard_value), rtol=1e-6, atol=1e-8))
    except (TypeError, ValueError):
        return str(grid_value).strip() == str(shard_value).strip()


_TURB_SENSITIVE_COLS = {"turbvel", "t_value"}


def check_shards_match_grid(paths: PipelinePaths, *, sample_size: int) -> CheckResult:
    name = "shard rows vs. grid"
    if not _is_valid_zarr(paths.grid_zarr):
        return _skip(name, "grid zarr missing")
    shard_paths = _discover_shards(paths)
    if not shard_paths:
        return _skip(name, f"no shard zarr stores under {paths.shards_dir}")

    grid_root = _open_zarr(paths.grid_zarr)
    n_rows = _row_count(grid_root)
    compare_keys = [
        k for k in grid_root.array_keys()
        if k not in {"wavelength", "flux", "continuum", "status", "message", "mu_selected", "mu_selected_index", "base_name", "global_index"}
        and k not in _TURB_SENSITIVE_COLS
    ]
    issues: List[str] = []
    mismatch_count = 0
    sampled = 0
    shards_ok = 0
    total_shards = 0

    for shard_path in shard_paths:
        total_shards += 1
        try:
            shard_root = _open_zarr(shard_path)
        except Exception as exc:
            issues.append(f"{os.path.basename(shard_path)}: cannot open ({exc})")
            continue
        if "global_index" not in shard_root:
            issues.append(f"{os.path.basename(shard_path)}: missing global_index")
            continue
        gidx = np.asarray(shard_root["global_index"][:], dtype=np.int64)
        if gidx.size == 0:
            continue
        out_of_range = gidx[(gidx < 0) | (gidx >= n_rows)]
        if out_of_range.size:
            issues.append(
                f"{os.path.basename(shard_path)}: {out_of_range.size} global_index values out of [0,{n_rows})"
            )
            continue

        pick_local = _sample_indices(gidx.size, sample_size, seed=hash(shard_path) & 0xFFFFFFFF)
        pick_global = gidx[pick_local]
        grid_sample = _load_columns(grid_root, compare_keys, indices=pick_global)
        shard_sample = _load_columns(shard_root, compare_keys, indices=pick_local)

        shard_ok = True
        for key in compare_keys:
            if key not in grid_sample or key not in shard_sample:
                continue
            g_arr = grid_sample[key]
            s_arr = shard_sample[key]
            if g_arr.shape != s_arr.shape:
                issues.append(f"{os.path.basename(shard_path)}: column {key!r} shape mismatch")
                shard_ok = False
                continue
            for i in range(g_arr.size):
                if not _value_equal(g_arr[i], s_arr[i]):
                    mismatch_count += 1
                    if len(issues) < 20:
                        issues.append(
                            f"{os.path.basename(shard_path)} row global={int(pick_global[i])}: "
                            f"{key}={s_arr[i]!r} (grid: {g_arr[i]!r})"
                        )
                    shard_ok = False
        sampled += pick_local.size
        if shard_ok:
            shards_ok += 1

    stats = {
        "shards_total": total_shards,
        "shards_clean": shards_ok,
        "rows_sampled": sampled,
        "row_mismatches": mismatch_count,
    }
    if issues:
        return _fail(name, f"{shards_ok}/{total_shards} shards clean; {mismatch_count} row mismatches", issues, stats)
    return _pass(name, f"all {total_shards} shards consistent with grid ({sampled:,} rows sampled)", stats)


################################################################################
# Check: t_value is the nearest atmosphere label to turbvel
################################################################################


def check_t_value_matches_turbvel(paths: PipelinePaths) -> CheckResult:
    name = "t_value == nearest(turbvel, atmospheres)"
    if not paths.model_atmosphere_path or not os.path.isdir(paths.model_atmosphere_path):
        return _skip(name, f"atmosphere directory unreadable: {paths.model_atmosphere_path}")
    labels = _scan_available_turb_labels(paths.model_atmosphere_path)
    if not labels:
        return _skip(name, f"no atmosphere labels found in {paths.model_atmosphere_path}")

    targets: List[Tuple[str, Any]] = []
    for shard_path in _discover_shards(paths):
        try:
            targets.append((shard_path, _open_zarr(shard_path)))
        except Exception:
            continue
    if not targets and _is_valid_zarr(paths.merged_zarr):
        targets.append((paths.merged_zarr or "", _open_zarr(paths.merged_zarr)))
    if not targets:
        return _skip(name, "no shard or merged zarr available")

    mismatches: List[str] = []
    mismatch_count = 0
    rows_checked = 0

    for label, root in targets:
        if "turbvel" not in root or "t_value" not in root:
            continue
        turbvel = np.asarray(root["turbvel"][:])
        t_value = np.asarray(root["t_value"][:])
        if turbvel.shape != t_value.shape:
            mismatches.append(f"{os.path.basename(label) or label}: turbvel/t_value shape mismatch")
            continue
        for i in range(turbvel.size):
            rows_checked += 1
            raw = turbvel[i]
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")
            if raw in (None, "", b""):
                continue
            vmicro = _coerce_microturbulence_value(raw, fallback="01")
            expected = _nearest_turb_label(vmicro, labels)
            actual = str(t_value[i]).strip() if not isinstance(t_value[i], (bytes, bytearray)) else t_value[i].decode("utf-8", errors="ignore").strip()
            if expected and actual != expected:
                mismatch_count += 1
                if len(mismatches) < 20:
                    mismatches.append(
                        f"{os.path.basename(label) or label} row={i}: "
                        f"turbvel={raw!r} (vmicro={vmicro:.3f}) t_value={actual!r} expected={expected!r}"
                    )

    stats = {
        "atmospheres_found": len(labels),
        "atmosphere_labels": list(labels),
        "rows_checked": rows_checked,
        "mismatches": mismatch_count,
    }
    if mismatches:
        return _fail(
            name,
            f"{mismatch_count} rows do not match nearest atmosphere label (labels found: {list(labels)})",
            mismatches,
            stats,
        )
    return _pass(name, f"{rows_checked:,} rows match nearest(turbvel, atmospheres); labels={list(labels)}", stats)


################################################################################
# Check: merged zarr row-count and param alignment with shards
################################################################################


def check_merged_consistency(paths: PipelinePaths) -> CheckResult:
    name = "merged zarr vs. shards"
    if not _is_valid_zarr(paths.merged_zarr):
        return _skip(name, f"merged zarr missing: {paths.merged_zarr}")
    shard_paths = _discover_shards(paths)
    if not shard_paths:
        return _skip(name, "no shards available (may have been cleaned up post-merge)")

    merged_root = _open_zarr(paths.merged_zarr)
    merged_flux = merged_root.get("flux") if "flux" in merged_root else None
    if merged_flux is None:
        return _fail(name, "merged zarr has no 'flux' array", ["missing 'flux'"])
    merged_rows = int(merged_flux.shape[0])

    expected_rows = 0
    for shard_path in shard_paths:
        try:
            shard_root = _open_zarr(shard_path)
        except Exception:
            continue
        if "global_index" in shard_root:
            expected_rows += int(np.asarray(shard_root["global_index"][:]).size)

    issues: List[str] = []
    if expected_rows and merged_rows < expected_rows:
        issues.append(f"merged rows={merged_rows} < sum(shard rows)={expected_rows}")

    # Wavelength grid must match between shards and merge.
    if "wavelength" in merged_root and shard_paths:
        merged_wl = np.asarray(merged_root["wavelength"][:])
        for shard_path in shard_paths[:3]:
            try:
                shard_root = _open_zarr(shard_path)
            except Exception:
                continue
            if "wavelength" in shard_root:
                shard_wl = np.asarray(shard_root["wavelength"][:])
                if shard_wl.shape != merged_wl.shape or not np.allclose(shard_wl, merged_wl):
                    issues.append(f"wavelength mismatch: {os.path.basename(shard_path)} vs merged")
                    break

    stats = {"merged_rows": merged_rows, "shard_rows_sum": expected_rows, "shards": len(shard_paths)}
    if issues:
        return _fail(name, f"{len(issues)} issue(s) between shards and merged zarr", issues, stats)
    return _pass(name, f"merged zarr ({merged_rows:,} rows) consistent with {len(shard_paths)} shards", stats)


################################################################################
# Check: Turbospectrum intermediate artifacts (best-effort; may be cleaned up)
################################################################################


def check_intermediate_artifacts(paths: PipelinePaths, *, sample_size: int) -> CheckResult:
    name = "Turbospectrum intermediate files"
    if not paths.output_dir or not os.path.isdir(paths.output_dir):
        return _skip(name, f"synthesis output_dir not present: {paths.output_dir}")
    if not _is_valid_zarr(paths.grid_zarr):
        return _skip(name, "grid zarr missing")

    atmosphere_labels = (
        _scan_available_turb_labels(paths.model_atmosphere_path)
        if paths.model_atmosphere_path and os.path.isdir(paths.model_atmosphere_path)
        else ()
    )

    shards = _discover_shards(paths)
    if not shards:
        return _skip(name, "no shard zarr stores to sample from")

    grid_root = _open_zarr(paths.grid_zarr)
    grid_keys = list(grid_root.array_keys())

    missing_specs: List[str] = []
    missing_logs: List[str] = []
    stem_mismatches: List[str] = []
    sampled = 0
    checked_shards = 0

    for shard_path in shards[:10]:
        try:
            shard_root = _open_zarr(shard_path)
        except Exception:
            continue
        if "global_index" not in shard_root or "status" not in shard_root:
            continue
        gidx = np.asarray(shard_root["global_index"][:], dtype=np.int64)
        statuses = np.asarray(shard_root["status"][:])
        if gidx.size == 0:
            continue
        success_mask = np.array(
            [
                str(s if not isinstance(s, (bytes, bytearray)) else s.decode("utf-8", errors="ignore")).strip().lower() in {"success", "skipped"}
                for s in statuses.tolist()
            ],
            dtype=bool,
        )
        eligible = np.where(success_mask)[0]
        if eligible.size == 0:
            continue

        pick_local = _sample_indices(eligible.size, max(1, sample_size // max(1, len(shards))))
        pick_local = eligible[pick_local]
        pick_global = gidx[pick_local]
        row_columns = _load_columns(grid_root, grid_keys, indices=pick_global)
        shard_base_names = None
        if "base_name" in shard_root:
            shard_base_names = np.asarray(shard_root["base_name"][:])

        checked_shards += 1
        for row_idx_in_shard, row_idx_in_grid in zip(pick_local, pick_global):
            row_values = {k: row_columns[k][np.where(pick_global == row_idx_in_grid)[0][0]] for k in row_columns}
            expected_stem = get_synthesis_output_stem_from_params(
                row_values,
                default_output_mode=str(row_values.get("output_mode", "Flux")),
                default_calculation_mode=str(row_values.get("calculation_mode", "LTE")),
                atmosphere_turb_labels=atmosphere_labels or None,
            )
            is_intensity = str(row_values.get("output_mode", "Flux")).lower() == "intensity"
            suffix = ".intensity.spec" if is_intensity else ".spec"
            spec_candidate = os.path.join(paths.output_dir, f"{expected_stem}{suffix}")
            log_candidate = (
                os.path.join(paths.log_dir, f"{expected_stem}.log")
                if paths.log_dir else None
            )

            sampled += 1
            if shard_base_names is not None and row_idx_in_shard < shard_base_names.size:
                shard_base = str(shard_base_names[int(row_idx_in_shard)]).strip()
                if shard_base and shard_base != expected_stem and not shard_base.startswith(expected_stem[:16]):
                    if len(stem_mismatches) < 10:
                        stem_mismatches.append(
                            f"row global={int(row_idx_in_grid)}: shard base_name={shard_base!r} "
                            f"expected={expected_stem!r}"
                        )
            if not os.path.exists(spec_candidate):
                # Shortened-stem fallback: accept any file whose base matches the first 16 chars.
                prefix = expected_stem[:16]
                if not any(
                    entry.startswith(prefix) and entry.endswith(suffix)
                    for entry in os.listdir(paths.output_dir)
                ):
                    if len(missing_specs) < 10:
                        missing_specs.append(spec_candidate)
            if log_candidate and not os.path.exists(log_candidate):
                if len(missing_logs) < 10:
                    missing_logs.append(log_candidate)

    stats = {
        "shards_inspected": checked_shards,
        "rows_sampled": sampled,
        "missing_spec_files": len(missing_specs),
        "missing_log_files": len(missing_logs),
        "stem_mismatches": len(stem_mismatches),
    }
    issues: List[str] = []
    issues.extend(f"missing spec: {p}" for p in missing_specs)
    issues.extend(f"missing log: {p}" for p in missing_logs)
    issues.extend(stem_mismatches)

    if missing_specs and checked_shards:
        # Intermediate files are routinely cleaned up; downgrade to skip when
        # the output directory is effectively empty of .spec files at all.
        any_spec = any(
            entry.endswith(".spec") or entry.endswith(".intensity.spec")
            for entry in os.listdir(paths.output_dir)
        )
        if not any_spec:
            return _skip(name, f"no .spec files under {paths.output_dir} (likely cleaned up post-merge)")

    if issues:
        return _fail(name, f"{len(missing_specs)} missing specs, {len(missing_logs)} missing logs, {len(stem_mismatches)} stem mismatches", issues, stats)
    return _pass(name, f"{sampled:,} sampled rows have matching intermediate artifacts", stats)


################################################################################
# Driver
################################################################################


def _print_report(results: Sequence[CheckResult], paths: PipelinePaths, *, as_json: bool, verbose: bool) -> None:
    if as_json:
        payload = {
            "config": paths.config_path,
            "paths": dataclasses.asdict(paths),
            "results": [dataclasses.asdict(r) for r in results],
        }
        print(json.dumps(payload, indent=2, default=str))
        return

    print(f"config: {paths.config_path}")
    print(f"  run_root:               {paths.run_root or '<unset>'}")
    print(f"  grid_zarr:              {paths.grid_zarr or '<unset>'}")
    print(f"  shards_dir:             {paths.shards_dir or '<unset>'}")
    print(f"  merged_zarr:            {paths.merged_zarr or '<unset>'}")
    print(f"  model_atmosphere_path:  {paths.model_atmosphere_path or '<unset>'}")
    print()

    pass_count = sum(1 for r in results if r.status == STATUS_PASS)
    fail_count = sum(1 for r in results if r.status == STATUS_FAIL)
    skip_count = sum(1 for r in results if r.status == STATUS_SKIP)

    for r in results:
        mark = {"pass": "OK  ", "fail": "FAIL", "skip": "SKIP"}[r.status]
        print(f"[{mark}] {r.name}: {r.summary}")
        if r.stats:
            keys = [k for k in r.stats if k != "atmosphere_labels"]
            stats_str = ", ".join(f"{k}={r.stats[k]}" for k in keys)
            if stats_str:
                print(f"        {stats_str}")
        if verbose or r.status == STATUS_FAIL:
            for issue in r.issues[:10]:
                print(f"        - {issue}")
            if len(r.issues) > 10:
                print(f"        ... and {len(r.issues) - 10} more")

    print()
    print(f"summary: {pass_count} pass, {fail_count} fail, {skip_count} skip")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pipeline JSON config")
    parser.add_argument("--sample", type=int, default=64, help="Rows to sample per shard for spot checks")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a human report")
    parser.add_argument("--verbose", action="store_true", help="Show issue lists for passing/skipped checks too")
    parser.add_argument(
        "--skip-intermediate",
        action="store_true",
        help="Skip the Turbospectrum intermediate-artifact check (useful when the run has been merged and cleaned up)",
    )
    args = parser.parse_args()

    paths = _resolve_paths(args.config)
    cfg = _load_json(args.config)

    results: List[CheckResult] = []
    results.append(check_grid_matches_config(paths, cfg))
    results.append(check_shards_match_grid(paths, sample_size=args.sample))
    results.append(check_t_value_matches_turbvel(paths))
    results.append(check_merged_consistency(paths))
    if not args.skip_intermediate:
        results.append(check_intermediate_artifacts(paths, sample_size=args.sample))

    _print_report(results, paths, as_json=bool(args.json), verbose=bool(args.verbose))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
