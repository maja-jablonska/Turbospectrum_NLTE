#!/usr/bin/env python3
"""Regular (Cartesian) grid sampler: ``resolve_regular_sampling`` builds the
uniformly spaced (teff, logg, feh, turbvel[, abundances, mu]) columns that
``generate_grid.resolve_grid_columns`` uses for ``grid.sampling: "grid"``.

Running this file is a thin compatibility shim: grid generation and synthesis
go through ``pipeline_from_config.py`` (one engine for every grid type)::

    python scripts/synthesize_regular_grid.py --config-json CONFIG [--skip-synthesis]
    # == python scripts/pipeline_from_config.py --config CONFIG [--skip-synthesis]
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

try:
    from .nlte_ascii_departures import build_nlte_ascii_selector_columns, normalize_nlte_ascii_selector
except ImportError:
    from nlte_ascii_departures import build_nlte_ascii_selector_columns, normalize_nlte_ascii_selector

try:
    from .run_turbospectrum import _nearest_turb_label, _normalize_turbulence_id
except ImportError:
    from run_turbospectrum import _nearest_turb_label, _normalize_turbulence_id

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def nearest_t_value_for_turbvel(turbvel_values: Sequence[Any], t_value_options: Sequence[Any]) -> np.ndarray:
    """Snap each row's atmosphere t_value label to the closest entry in
    ``t_value_options`` to its synthesis ``turbvel``.

    The atmosphere file's microturbulence label only takes a few discrete
    values (e.g. "00","01","02","05"), while turbvel may be any of the
    sampled options. To minimize the physical mismatch between the chosen
    atmosphere and the synthesis microturbulence, pick the nearest label
    per row. Mirrors `run_turbospectrum._nearest_turb_label`, which already
    does the same for atmosphere selection at synthesis time — keeping the
    stored t_value column consistent with what synthesis actually uses.
    """
    pool = [str(tok).strip() for tok in (t_value_options or []) if str(tok).strip()]
    if not pool:
        pool = ["01"]
    fallback = pool[0]
    out = np.empty(len(turbvel_values), dtype=object)
    for i, raw in enumerate(turbvel_values):
        try:
            target = float(_normalize_turbulence_id(raw, default=fallback))
        except (TypeError, ValueError):
            out[i] = fallback
            continue
        nearest = _nearest_turb_label(target, pool)
        out[i] = nearest if nearest is not None else fallback
    return out

# Mirror generate_grid.py / sample_machine_learning_grid.py so the regular and
# LHS paths emit the same param_names ordering for downstream ML code.
LEGACY_ABUNDANCE_KEYS = ("a", "c", "n", "o", "r", "s")


def _ordered_abundance_keys(abund_cfg: Mapping[str, Any]) -> List[str]:
    configured = {str(key).strip() for key in (abund_cfg or {}).keys() if str(key).strip()}
    ordered: List[str] = [key for key in LEGACY_ABUNDANCE_KEYS if key in configured]
    ordered.extend(sorted(configured.difference(ordered)))
    return ordered














def _parse_numeric_axis(raw: str, name: str, *, integer: bool) -> np.ndarray:
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{name} axis spec cannot be empty")

    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        if len(parts) != 3:
            raise ValueError(f"{name} axis range must be start:end:step, got {text!r}")
        start = float(parts[0])
        stop = float(parts[1])
        step = float(parts[2])
        if step <= 0:
            raise ValueError(f"{name} axis step must be positive, got {step}")
        if stop < start:
            raise ValueError(f"{name} axis stop must be >= start, got {start}..{stop}")
        values = np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
    else:
        tokens = [t.strip() for t in text.split(",") if t.strip()]
        if not tokens:
            raise ValueError(f"{name} axis list is empty")
        values = np.asarray([float(t) for t in tokens], dtype=np.float64)
        values = np.sort(np.unique(values))

    if values.size == 0:
        raise ValueError(f"{name} axis must contain at least one value")

    if values.size >= 2:
        diffs = np.diff(values)
        ref = float(diffs[0])
        tol = max(1e-10, abs(ref) * 1e-8)
        if not np.allclose(diffs, ref, rtol=0.0, atol=tol):
            raise ValueError(
                f"{name} axis is not uniformly spaced; got steps like {diffs[:5].tolist()}"
            )

    if integer:
        ints = np.rint(values).astype(np.int64)
        if not np.allclose(values, ints.astype(np.float64), rtol=0.0, atol=1e-8):
            raise ValueError(f"{name} axis requires integer values; got {values[:5].tolist()}")
        return ints
    return values.astype(np.float64)


def _parse_turbvel_values(raw: str) -> np.ndarray:
    tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
    if not tokens:
        raise ValueError("turbvel axis cannot be empty")
    out: List[str] = []
    seen = set()
    for token in tokens:
        if token.lstrip("+-").isdigit():
            token = f"{int(token):02d}"
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return np.asarray(out, dtype=object)


def _resolve_turbvel_axis(vmicro_spec: Any, turbvel_spec: Any) -> np.ndarray:
    """Resolve the regular-grid microturbulence axis.

    A numeric ``vmicro`` axis (a ``start:end:step`` range or a uniform comma list,
    in km/s) is treated like the teff/logg/feh axes and stored as floats in the
    ``turbvel`` column — the Cartesian-grid analogue of the continuous LHS path.
    The legacy discrete ``turbvel`` axis (label list such as ``"01,02,05"``) is
    kept for back-compat. Supplying both is an error. When neither is given, the
    discrete default ``"01,02,03"`` is used.
    """
    has_vmicro = vmicro_spec not in (None, "")
    has_turbvel = turbvel_spec not in (None, "")
    if has_vmicro and has_turbvel:
        raise ValueError(
            "Use either axes.vmicro (continuous) or axes.turbvel (discrete), not both"
        )
    if has_vmicro:
        return _parse_numeric_axis(str(vmicro_spec), "vmicro", integer=False)
    return _parse_turbvel_values(str(turbvel_spec) if has_turbvel else "01,02,03")




def _parse_abundance_spec(raw: Any, name: str) -> np.ndarray:
    """Parse an abundance value: a single float/string or a 'start:end:step' range.

    Returns an array of formatted abundance strings like '+0.00'.
    """
    text = str(raw).strip()
    if not text:
        return np.asarray(["+0.00"], dtype=object)

    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        if len(parts) != 3:
            raise ValueError(f"abundance '{name}' range must be start:end:step, got {text!r}")
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        if step <= 0:
            raise ValueError(f"abundance '{name}' step must be positive, got {step}")
        if stop < start:
            raise ValueError(f"abundance '{name}' stop must be >= start, got {start}..{stop}")
        values = np.arange(start, stop + 0.5 * step, step)
        return np.asarray([f"{v:+.2f}" for v in values], dtype=object)

    try:
        v = float(text)
        return np.asarray([f"{v:+.2f}"], dtype=object)
    except ValueError:
        # Already a formatted string like "+0.00"
        return np.asarray([text], dtype=object)


def _build_regular_columns(
    *,
    teff_axis: np.ndarray,
    logg_axis: np.ndarray,
    feh_axis: np.ndarray,
    turbvel_axis: np.ndarray,
    mu_axis: np.ndarray | None,
    grid_version: str,
    lam_min: float,
    lam_max: float,
    lam_step: float,
    output_mode: str,
    mode: str,
    calculation_mode: str,
    abundances: Dict[str, np.ndarray],
    max_rows: int,
    t_value_options: Sequence[Any],
    nlte_ascii_selector: Any = None,
) -> Dict[str, np.ndarray]:
    # Axes that always participate in the cross-product.
    nt = int(teff_axis.size)
    ng = int(logg_axis.size)
    nf = int(feh_axis.size)
    nu = int(turbvel_axis.size)
    nm = int(mu_axis.size) if mu_axis is not None else 1

    # Abundance axes — only emit configured ones, matching LHS / generate_grid.py
    # (sample_machine_learning_grid.py:_ordered_abundance_keys). Axes with zero
    # configured values are dropped so param_names stays consistent across paths.
    abund_keys = tuple(k for k in _ordered_abundance_keys(abundances) if int(abundances[k].size) > 0)
    abund_sizes = {k: int(abundances[k].size) for k in abund_keys}
    abund_total = 1
    for sz in abund_sizes.values():
        abund_total *= sz

    base_count = nt * ng * nf * nu * abund_total
    row_count = base_count * nm
    if row_count <= 0:
        raise ValueError("Computed row_count is zero")
    if row_count > max_rows:
        raise ValueError(
            f"Grid would create {row_count:,} rows (> max_rows={max_rows:,}). "
            "Lower axis resolutions or increase --max-rows."
        )

    # Build the cross-product using the standard repeat/tile pattern.
    # Axis order: teff (slowest) > logg > feh > turbvel > a > c > n > o > r > s (fastest before mu).
    # suffix_count = product of all axis sizes that come *after* the current axis.
    suffix_after_turbvel = abund_total
    teff_base = np.repeat(teff_axis.astype(np.int64), ng * nf * nu * suffix_after_turbvel)
    logg_base = np.tile(np.repeat(logg_axis.astype(np.float64), nf * nu * suffix_after_turbvel), nt)
    feh_base = np.tile(np.repeat(feh_axis.astype(np.float64), nu * suffix_after_turbvel), nt * ng)
    # Preserve the axis dtype: discrete labels stay object strings ("01"),
    # a numeric vmicro axis stays float64 — matching the continuous LHS path.
    turbvel_base = np.tile(np.repeat(turbvel_axis, suffix_after_turbvel), nt * ng * nf)

    # Abundance columns via the same repeat/tile pattern.
    prefix = nt * ng * nf * nu
    abund_columns: Dict[str, np.ndarray] = {}
    suffix = abund_total  # running product of sizes to the right
    for k in abund_keys:
        sz = abund_sizes[k]
        suffix //= sz
        col = np.tile(np.repeat(abundances[k], suffix), prefix * (abund_total // (sz * suffix)))
        abund_columns[k] = col

    if mu_axis is not None:
        teff = np.repeat(teff_base, nm)
        logg = np.repeat(logg_base, nm)
        feh = np.repeat(feh_base, nm)
        turbvel = np.repeat(turbvel_base, nm)
        for k in abund_keys:
            abund_columns[k] = np.repeat(abund_columns[k], nm)
        mu = np.tile(mu_axis.astype(np.float64), base_count)
    else:
        teff = teff_base
        logg = logg_base
        feh = feh_base
        turbvel = turbvel_base
        mu = None

    # Snap t_value to the available label closest to each row's turbvel. The
    # atmosphere library only ships a discrete set of microturbulence labels
    # (t_value_options, e.g. ["00","01","02","05"]); using the nearest one
    # minimizes the physical mismatch between the chosen atmosphere and the
    # synthesis microturbulence (and matches what run_turbospectrum does at
    # synthesis time via _nearest_turb_label).
    t_value = nearest_t_value_for_turbvel(turbvel, t_value_options)

    columns: Dict[str, np.ndarray] = {
        "grid_version": np.full(row_count, str(grid_version), dtype=object),
        "teff": teff,
        "logg": logg,
        "feh": feh,
        "lam_min": np.full(row_count, float(lam_min), dtype=np.float64),
        "lam_max": np.full(row_count, float(lam_max), dtype=np.float64),
        "lam_step": np.full(row_count, float(lam_step), dtype=np.float64),
        "turbvel": turbvel,
        "t_value": t_value,
        "output_mode": np.full(row_count, output_mode, dtype=object),
        "mode": np.full(row_count, mode, dtype=object),
        "calculation_mode": np.full(row_count, calculation_mode, dtype=object),
    }
    for k in abund_keys:
        columns[k] = abund_columns[k]
    if mu is not None:
        columns["mu"] = mu
    columns.update(build_nlte_ascii_selector_columns(row_count, nlte_ascii_selector))
    return columns


def resolve_regular_sampling(config: Mapping[str, Any], rng: Any = None, base_dir: str | None = None) -> Dict[str, np.ndarray]:
    """Build regular Cartesian-grid columns from a grid config (no CLI).

    Mirrors ``generate_grid._resolve_ml_sampling``: takes the ``grid`` sub-config
    and returns the same column schema, so ``generate_grid.resolve_grid_columns``
    can dispatch to either sampling mode and downstream synthesis is identical.
    ``rng`` is accepted for a uniform interface but unused (the grid is
    deterministic). Reuses the same parsers and ``_build_regular_columns`` as the
    CLI path so there is a single regular-grid implementation.
    """
    grid = dict(config or {})
    axes = grid.get("axes") or {}
    synth = grid.get("synthesis") or {}

    teff_axis = _parse_numeric_axis(str(axes.get("teff", "4000:7000:250")), "teff", integer=True)
    logg_axis = _parse_numeric_axis(str(axes.get("logg", "0.0:5.0:0.5")), "logg", integer=False)
    feh_axis = _parse_numeric_axis(str(axes.get("feh", "-2.5:0.5:0.25")), "feh", integer=False)
    turbvel_axis = _resolve_turbvel_axis(axes.get("vmicro"), axes.get("turbvel"))

    # t_value pool (grid.axes.t_value_options or top-level), normalized to labels.
    tvo_raw = axes.get("t_value_options")
    if tvo_raw is None:
        tvo_raw = grid.get("t_value_options", ["01", "02", "05"])
    if isinstance(tvo_raw, str):
        t_value_options = [tok.strip() for tok in tvo_raw.split(",") if tok.strip()]
    elif isinstance(tvo_raw, Sequence) and not isinstance(tvo_raw, (bytes, bytearray)):
        t_value_options = [str(tok).strip() for tok in tvo_raw if str(tok).strip()]
    else:
        t_value_options = ["01", "02", "05"]
    if not t_value_options:
        t_value_options = ["01", "02", "05"]
    t_value_options = [
        f"{int(tok):02d}" if tok.lstrip("+-").isdigit() else tok for tok in t_value_options
    ]

    output_mode = str(synth.get("output_mode", "Flux"))
    if output_mode not in {"Flux", "Intensity"}:
        raise ValueError(f"output_mode must be Flux or Intensity, got {output_mode!r}")
    mu_range_raw = synth.get("mu_range")
    mu_axis = None
    if mu_range_raw not in (None, ""):
        mu_axis = _parse_numeric_axis(str(mu_range_raw), "mu", integer=False)
        if output_mode != "Intensity":
            raise ValueError("grid.synthesis.mu_range requires output_mode='Intensity'")

    abund_section = grid.get("abundances") if isinstance(grid.get("abundances"), Mapping) else {}
    abundances = {
        k: _parse_abundance_spec(abund_section.get(k, "+0.00"), k)
        for k in _ordered_abundance_keys(abund_section)
    }

    nlte_ascii_cfg = grid.get("nlte_ascii_departures") if isinstance(grid.get("nlte_ascii_departures"), Mapping) else {}
    nlte_ascii_selector = normalize_nlte_ascii_selector(
        directory=nlte_ascii_cfg.get("directory"),
        species=nlte_ascii_cfg.get("species"),
        abundance_column=nlte_ascii_cfg.get("abundance_column"),
        abundance_scale=nlte_ascii_cfg.get("abundance_scale"),
        solar_abundance=nlte_ascii_cfg.get("solar_abundance"),
        match=nlte_ascii_cfg.get("match"),
        base_dir=base_dir,
    )

    limits = grid.get("limits") if isinstance(grid.get("limits"), Mapping) else {}
    max_rows = int(limits.get("max_rows", 2_000_000))

    return _build_regular_columns(
        teff_axis=teff_axis,
        logg_axis=logg_axis,
        feh_axis=feh_axis,
        turbvel_axis=turbvel_axis,
        mu_axis=mu_axis,
        grid_version=str(grid.get("grid_version", "regular-linear-v1")),
        lam_min=float(synth.get("lam_min", 6000.0)),
        lam_max=float(synth.get("lam_max", 6100.0)),
        lam_step=float(synth.get("lam_step", 0.01)),
        output_mode=output_mode,
        mode=str(synth.get("mode", grid.get("mode", "1D"))),
        calculation_mode=str(synth.get("calculation_mode", grid.get("calculation_mode", "LTE"))),
        abundances=abundances,
        t_value_options=t_value_options,
        nlte_ascii_selector=nlte_ascii_selector,
        max_rows=max_rows,
    )


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-json", required=True, help="Pipeline config (grid.sampling 'grid' / grid.axes)")
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--skip-nlte-preflight", action="store_true")
    parser.add_argument("--no-scratch", action="store_true")
    parser.add_argument("--scratch", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(
            f"unsupported arguments {unknown}: grid axes, wavelengths and output paths now come only "
            "from the config file (grid.axes / grid.synthesis / outputs)."
        )
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "pipeline_from_config.py"), "--config", args.config_json]
    for flag in ("skip_synthesis", "skip_nlte_preflight", "no_scratch"):
        if getattr(args, flag):
            cmd.append("--" + flag.replace("_", "-"))
    if args.scratch:
        cmd += ["--scratch", args.scratch]
    if args.workers:
        cmd += ["--workers", str(args.workers)]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
