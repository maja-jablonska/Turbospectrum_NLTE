#!/usr/bin/env python3
"""Preflight-check the NLTE setup of a pipeline / regular-grid config.

Run this before a (possibly long) synthesis to catch NLTE misconfigurations up
front instead of deep inside the Fortran:

    python3 scripts/validate_nlte_config.py --config configs/pipeline/config_nlte_minimal.example.json

It reports problems such as a missing departure grid, an info file that still
points at an HPC path, or an empty ASCII departure directory, and prints the
exact download command when data is missing. Exit code is 0 when the NLTE setup
looks runnable (or NLTE is off), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Mapping

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from nlte_config import preflight_nlte_from_config  # noqa: E402


def _strip_private(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def _resolve_project_root(cfg: Mapping[str, Any]) -> str:
    cur: Any
    for keys in (("turbospectrum", "overrides", "project_root"), ("turbospectrum", "project_root"), ("project_root",)):
        cur = cfg
        for key in keys:
            cur = cur.get(key) if isinstance(cur, Mapping) else None
        if cur and os.path.isdir(str(cur)):
            return str(cur)
    return REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a pipeline or regular-grid JSON config")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = _strip_private(json.load(handle))

    result = preflight_nlte_from_config(
        cfg, config_dir=config_dir, project_root=_resolve_project_root(cfg)
    )

    print(f"Config:           {config_path}")
    print(f"calculation_mode: {result['calculation_mode']!r}")
    print(f"NLTE enabled:     {result['enabled']}")
    print(f"Workflow:         {'ASCII departures' if result['workflow'] == 'ascii' else 'binary departure grids'}")

    if not result["enabled"]:
        print("\nNLTE is not enabled for this config — nothing to validate.")
        return 0

    problems = result["problems"]
    if not problems:
        print("\nOK: NLTE inputs look runnable.")
        return 0

    print("\nNLTE preflight found problems:")
    for line in problems:
        print(f"  {line}" if line else "")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
