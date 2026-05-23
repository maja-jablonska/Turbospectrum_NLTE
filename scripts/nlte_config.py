"""Shared helpers that make NLTE configuration easier and consistent.

The pipeline supports two NLTE workflows, and this module gives them one mental
model so callers (``pipeline_from_config.py`` and ``synthesize_regular_grid.py``)
behave identically:

* **Binary departure grids** — downloaded via ``scripts/download_data.sh
  --nlte-atoms`` into ``input_files/nlte_data/`` and selected through an
  ``nlte_info_file`` (defaults to ``DATA/SPECIES_LTE_NLTE.dat``).
* **ASCII per-abundance departures** — configured through an
  ``nlte_ascii_departures`` block.

The **single switch** is the grid's ``calculation_mode`` (``LTE``/``NLTE``):
when it is ``NLTE`` the synthesis config's ``nlte.enabled`` is turned on
automatically, so a config no longer has to keep two flags in sync.

This module is import-light (stdlib only) so it can be reused from any script
and unit-tested without touching Turbospectrum.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Mapping

# Shown whenever NLTE inputs are missing, so the user knows the exact next step.
NLTE_DOWNLOAD_HINT = (
    "Download NLTE data with:\n"
    "    ./scripts/download_data.sh --nlte-atoms all\n"
    "Model atoms + binary departure grids land under input_files/nlte_data/; "
    "the LTE/NLTE species map is DATA/SPECIES_LTE_NLTE.dat."
)


def _noop_warn(_msg: str) -> None:  # pragma: no cover - trivial default
    return None


def is_nlte_calculation_mode(calculation_mode: Any) -> bool:
    """True when ``calculation_mode`` requests NLTE (case-insensitive)."""
    return str(calculation_mode if calculation_mode is not None else "").strip().upper() == "NLTE"


def apply_single_nlte_switch(
    synthesis_cfg: Dict[str, Any],
    calculation_mode: Any,
    *,
    warn: Callable[[str], None] = _noop_warn,
) -> Dict[str, Any]:
    """Derive ``nlte.enabled`` from the grid ``calculation_mode``, in place.

    * ``calculation_mode == 'NLTE'`` turns ``nlte.enabled`` on — this is the
      single switch, so ``calculation_mode: NLTE`` alone is enough.
    * Other modes leave an explicitly-enabled ``nlte`` block untouched (so older
      configs keep working) but ``warn`` about the contradiction, because the
      stored grid metadata would then disagree with what was synthesized.

    Mutates and returns ``synthesis_cfg`` (the dict that owns the ``nlte`` block,
    e.g. the materialized Turbospectrum config or the override block).
    """
    nlte = synthesis_cfg.get("nlte")
    enabled = bool(nlte.get("enabled", False)) if isinstance(nlte, dict) else False
    if is_nlte_calculation_mode(calculation_mode):
        # Only touch the block when enabling NLTE, so LTE runs are left untouched.
        if not isinstance(nlte, dict):
            nlte = {}
            synthesis_cfg["nlte"] = nlte
        if not enabled:
            nlte["enabled"] = True
    elif enabled:
        warn(
            "calculation_mode is not 'NLTE' but nlte.enabled=true — running NLTE "
            "anyway. Set grid.synthesis.calculation_mode='NLTE' so the stored grid "
            "metadata matches the synthesis."
        )
    return synthesis_cfg


def resolve_nlte_ascii_cfg(*candidates: Any) -> Dict[str, Any]:
    """Return the first non-empty ``nlte_ascii_departures`` mapping.

    Lets every entry point accept the block either at the top level of the config
    or nested under ``grid`` — pass the candidate locations in priority order.
    """
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate:
            return dict(candidate)
    return {}


def nlte_is_enabled(synthesis_cfg: Mapping[str, Any], calculation_mode: Any = None) -> bool:
    """Whether NLTE is on, considering both the single switch and nlte.enabled."""
    if is_nlte_calculation_mode(calculation_mode):
        return True
    nlte = synthesis_cfg.get("nlte") if isinstance(synthesis_cfg, Mapping) else None
    return bool(nlte.get("enabled", False)) if isinstance(nlte, Mapping) else False


def default_nlte_info_file(project_root: str) -> str:
    """The default binary-departure species map, matching run_turbospectrum."""
    return os.path.join(project_root, "DATA", "SPECIES_LTE_NLTE.dat")


def _resolve_against(path: str, base_dir: Any) -> str:
    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    if base_dir and not os.path.isabs(expanded):
        expanded = os.path.join(str(base_dir), expanded)
    return os.path.abspath(expanded)


def preflight_nlte(
    *,
    project_root: str,
    enabled: bool,
    nlte_info_file: Any = "",
    nlte_ascii_cfg: Mapping[str, Any] | None = None,
    nlte_ascii_base_dir: Any = None,
) -> list[str]:
    """Validate that the inputs an NLTE run needs are actually present.

    Returns a list of human-readable problems; an empty list means the NLTE
    setup looks runnable. Catches the misconfigurations that otherwise surface
    deep inside the Fortran (missing departure grids, an info file that still
    points at an HPC path, an empty ASCII departure directory, …) and ends with
    the download hint when data is missing.

    The check is best-effort and read-only: it never imports Turbospectrum, and
    it lazily imports the ASCII/info-file parser so the module stays light.
    """
    problems: list[str] = []
    if not enabled:
        return problems

    ascii_cfg = dict(nlte_ascii_cfg) if isinstance(nlte_ascii_cfg, Mapping) else {}

    if ascii_cfg.get("directory"):
        # --- ASCII per-abundance departures ------------------------------------
        directory = _resolve_against(str(ascii_cfg["directory"]), nlte_ascii_base_dir)
        if not os.path.isdir(directory):
            problems.append(f"NLTE ASCII departure directory not found: {directory}")
        else:
            entries = [e for e in os.listdir(directory) if not e.startswith(".")]
            if not entries:
                problems.append(f"NLTE ASCII departure directory is empty: {directory}")
    else:
        # --- Binary departure grids via the info file --------------------------
        info = str(nlte_info_file or "").strip()
        info = _resolve_against(info, project_root) if info else default_nlte_info_file(project_root)
        if not os.path.isfile(info):
            problems.append(f"NLTE info file not found: {info}")
        else:
            try:
                try:
                    from nlte_ascii_departures import parse_nlte_info_file
                except ImportError:
                    from .nlte_ascii_departures import parse_nlte_info_file  # type: ignore
                path_model, path_depart, info_entries = parse_nlte_info_file(info)
            except Exception as exc:  # noqa: BLE001 - surface any parse failure
                problems.append(f"Could not parse NLTE info file {info}: {exc}")
            else:
                name = os.path.basename(info)
                active = [e for e in info_entries if str(e.nlte_flag).strip().lower() == "nlte"]
                if not active:
                    problems.append(
                        f"No species are set to 'nlte' in {name} (every line is LTE or "
                        "commented out) — NLTE is enabled but nothing will run in NLTE."
                    )
                if not os.path.isdir(path_model):
                    problems.append(
                        f"Model-atom path referenced by {name} does not exist: {path_model} "
                        "(the shipped file points at an HPC path; update it to your local "
                        "input_files/nlte_data/model_atoms)."
                    )
                if not os.path.isdir(path_depart):
                    problems.append(
                        f"Departure path referenced by {name} does not exist: {path_depart} "
                        "(update it to your local input_files/nlte_data/departure_grids)."
                    )
                for entry in active:
                    atom = os.path.join(path_model, str(entry.model_atom_file))
                    dep = os.path.join(path_depart, str(entry.departure_file))
                    if os.path.isdir(path_model) and not os.path.isfile(atom):
                        problems.append(f"Missing model atom for {entry.element}: {atom}")
                    if os.path.isdir(path_depart) and not os.path.isfile(dep):
                        problems.append(f"Missing departure file for {entry.element}: {dep}")

    if problems:
        problems.append("")
        problems.append(NLTE_DOWNLOAD_HINT)
    return problems


def _cfg_get(cfg: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def synthesis_nlte_block(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """The synthesis ``nlte`` block, wherever the entry points accept it."""
    for candidate in (
        _cfg_get(cfg, "turbospectrum", "overrides", "nlte"),
        _cfg_get(cfg, "turbospectrum", "nlte"),
        _cfg_get(cfg, "nlte"),
    ):
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {}


def preflight_nlte_from_config(
    cfg: Mapping[str, Any],
    *,
    config_dir: str,
    project_root: str,
) -> Dict[str, Any]:
    """Gather NLTE settings from a whole pipeline/regular-grid config and check.

    One place for the CLI and both pipeline entry points to decide whether NLTE
    is on, which workflow it uses, and whether its inputs are present. Returns a
    dict with ``enabled``, ``calculation_mode``, ``workflow`` and ``problems``.
    """
    calc_mode = _cfg_get(cfg, "grid", "synthesis", "calculation_mode")
    nlte_block = synthesis_nlte_block(cfg)
    enabled = is_nlte_calculation_mode(calc_mode) or bool(nlte_block.get("enabled", False))
    ascii_cfg = resolve_nlte_ascii_cfg(
        cfg.get("nlte_ascii_departures"),
        _cfg_get(cfg, "grid", "nlte_ascii_departures"),
    )
    workflow = "ascii" if ascii_cfg.get("directory") else "binary"
    problems = preflight_nlte(
        project_root=project_root,
        enabled=enabled,
        nlte_info_file=nlte_block.get("nlte_info_file", ""),
        nlte_ascii_cfg=ascii_cfg,
        nlte_ascii_base_dir=config_dir,
    )
    return {
        "enabled": enabled,
        "calculation_mode": calc_mode,
        "workflow": workflow,
        "problems": problems,
    }
