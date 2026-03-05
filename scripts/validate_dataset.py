#!/usr/bin/env python3

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr
from provenance_contract import (
    REQUIRED_PROVENANCE_FIELDS,
    is_meaningful_provenance_value,
)


############################################
# Helpers
############################################

SHARD_REGEX = re.compile(r"spectra_shard_(\d+)\.zarr")
PROVENANCE_FILENAMES = (
    "canonical_config.yaml",
    "synthesis_config.yaml",
    "linelist_manifest.json",
    "atmosphere_manifest.json",
    "software_manifest.json",
    "environment.txt",
)


def extract_shard_id(path: Path):
    m = SHARD_REGEX.match(path.name)
    return int(m.group(1)) if m else None


def is_valid_zarr(path: Path):
    try:
        zarr.open_group(path, mode="r")
        return True
    except Exception:
        return False


def _contract_fields_from_attrs(root) -> dict[str, str]:
    attrs = dict(root.attrs)
    out = {
        "config_hash": str(attrs.get("config_hash", "")),
        "grid_definition_hash": str(attrs.get("grid_definition_hash", "")),
        "git_commit": str(attrs.get("git_commit") or attrs.get("git_sha") or ""),
        "turbospectrum_version": str(attrs.get("turbospectrum_version") or attrs.get("synthesis_code_version") or ""),
        "linelist_identifier": str(attrs.get("linelist_identifier") or attrs.get("linelist_path") or attrs.get("linelist_files") or ""),
        "linelist_version": str(attrs.get("linelist_version", "")),
        "atmosphere_model_identifier": str(attrs.get("atmosphere_model_identifier") or attrs.get("model_atmosphere_path") or ""),
        "synthesis_timestamp": str(attrs.get("synthesis_timestamp") or attrs.get("created_utc") or ""),
        "pipeline_version": str(attrs.get("pipeline_version") or attrs.get("spice_version") or ""),
    }
    return out


def _missing_contract_fields(fields: dict[str, str]) -> list[str]:
    return [key for key in REQUIRED_PROVENANCE_FIELDS if not is_meaningful_provenance_value(fields.get(key))]


############################################
# Shard Validation
############################################

def validate_shard(path: Path, reference_wavelength=None, *, enforce_provenance: bool = True):

    info = {
        "spectra": 0,
        "nan": False,
        "inf": False,
        "empty": False,
        "wavelength_mismatch": False,
    }

    if not is_valid_zarr(path):
        return False, {"corrupt_zarr": True}

    try:
        root = zarr.open_group(path, mode="r")

        if "flux" not in root or "wavelength" not in root:
            return False, {"missing_arrays": True}

        flux = root["flux"]
        wavelength = root["wavelength"]

        if flux.shape[0] == 0:
            info["empty"] = True
            return False, info

        info["spectra"] = flux.shape[0]

        ########################################
        # Stream rows safely
        ########################################

        for i in range(flux.shape[0]):

            row = flux[i]

            if not np.isfinite(row).all():
                if np.isnan(row).any():
                    info["nan"] = True
                if np.isinf(row).any():
                    info["inf"] = True
                return False, info

            if np.allclose(row, 0):
                info["empty"] = True
                return False, info

        ########################################
        # Wavelength consistency
        ########################################

        if reference_wavelength is not None:
            if wavelength.shape != reference_wavelength.shape:
                return False, {"wavelength_shape_mismatch": True}

            if not np.array_equal(wavelength[:], reference_wavelength):
                return False, {"wavelength_mismatch": True}

        if enforce_provenance:
            contract_fields = _contract_fields_from_attrs(root)
            missing_contract = _missing_contract_fields(contract_fields)
            if missing_contract:
                return False, {"missing_provenance_fields": missing_contract}

            if "provenance" not in root:
                return False, {"missing_provenance_group": True}
            provenance = root["provenance"]
            missing_files = [name for name in PROVENANCE_FILENAMES if name not in provenance]
            if missing_files:
                return False, {"missing_provenance_files": missing_files}

            # Basic payload sanity: manifests should be non-empty and JSON parseable.
            for name in ("canonical_config.yaml", "synthesis_config.yaml", "linelist_manifest.json", "atmosphere_manifest.json", "software_manifest.json"):
                try:
                    raw = provenance[name][...]
                    text = str(raw.item() if hasattr(raw, "item") else raw).strip()
                    if not text:
                        return False, {"empty_provenance_file": name}
                    if name.endswith(".json"):
                        json.loads(text)
                except Exception as exc:  # noqa: BLE001
                    return False, {"invalid_provenance_file": {name: str(exc)}}

        return True, info

    except Exception as e:
        return False, {"exception": str(e)}


def _validate_one_shard(path: Path, reference_wavelength, *, enforce_provenance: bool):
    ok, info = validate_shard(
        path,
        reference_wavelength,
        enforce_provenance=enforce_provenance,
    )
    return path.name, ok, info


############################################
# Completeness Check
############################################

def check_completeness(shards, expected_count):

    ids = []

    for s in shards:
        sid = extract_shard_id(s)
        if sid is not None:
            ids.append(sid)

    ids = sorted(ids)

    id_set = set(ids)

    expected = set(range(expected_count))

    missing = sorted(expected - id_set)
    unexpected = sorted(id_set - expected)

    duplicates = len(ids) != len(id_set)

    return {
        "found": len(ids),
        "expected": expected_count,
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
    }


############################################
# Main Validator
############################################

def validate_dataset(
    shard_dir: Path,
    expected_shards: int,
    *,
    allow_incomplete_provenance: bool,
    workers: int = 1,
):

    shards = sorted(shard_dir.glob("spectra_shard_*.zarr"))

    if not shards:
        print("❌ No shards found.")
        sys.exit(1)

    print(f"Found {len(shards)} shard directories\n")

    ##################################################
    # COMPLETENESS FIRST (fail fast)
    ##################################################

    completeness = check_completeness(shards, expected_shards)

    print("========== COMPLETENESS ==========")
    print(f"Expected shards: {completeness['expected']}")
    print(f"Found shards: {completeness['found']}")

    if completeness["duplicates"]:
        print("❌ Duplicate shard IDs detected.")
        sys.exit(2)

    if completeness["unexpected"]:
        print(f"❌ Unexpected shard IDs: {completeness['unexpected']}")
        sys.exit(2)

    if completeness["missing"]:
        print(f"❌ Missing shards: {completeness['missing'][:20]}")
        if len(completeness["missing"]) > 20:
            print("... (truncated)")
        sys.exit(2)

    print("✅ Completeness verified.\n")

    ##################################################
    # Deep validation
    ##################################################

    workers = max(int(workers), 1)
    reference_wavelength = None
    for shard in shards:
        if not is_valid_zarr(shard):
            continue
        try:
            root = zarr.open_group(shard, mode="r")
            if "wavelength" in root:
                reference_wavelength = root["wavelength"][:]
                break
        except Exception:
            continue

    total_spectra = 0
    failures = []
    results_by_name = {}
    enforce_provenance = not allow_incomplete_provenance

    if workers == 1:
        for shard in shards:
            name, ok, info = _validate_one_shard(
                shard,
                reference_wavelength,
                enforce_provenance=enforce_provenance,
            )
            results_by_name[name] = (ok, info)
    else:
        print(f"Deep validation using {workers} worker threads")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    _validate_one_shard,
                    shard,
                    reference_wavelength,
                    enforce_provenance=enforce_provenance,
                ): shard
                for shard in shards
            }
            for future in as_completed(future_map):
                name, ok, info = future.result()
                results_by_name[name] = (ok, info)

    for shard in shards:
        ok, info = results_by_name[shard.name]
        if not ok:
            failures.append((shard.name, info))
            continue
        total_spectra += info["spectra"]

    ##################################################
    # Report
    ##################################################

    print("\n========== VALIDATION ==========")
    print(f"Valid shards: {len(shards) - len(failures)}")
    print(f"Failed shards: {len(failures)}")
    print(f"Total spectra: {total_spectra}")

    if failures:
        print("\nFailures:\n")
        for name, reason in failures:
            print(f"{name} -> {reason}")

        print("\n❌ Dataset NOT SAFE for merge or ML.")
        sys.exit(3)

    print("\n✅ Dataset fully validated.")
    print("Safe for merge and downstream ML.")


############################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shard-dir",
        required=True,
        help="Directory containing shard zarrs"
    )

    parser.add_argument(
        "--expected-shards",
        type=int,
        required=True,
        help="Total shard count from grid definition"
    )
    parser.add_argument(
        "--allow-incomplete-provenance",
        action="store_true",
        help="Allow shards that do not satisfy provenance/hash contract fields.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker threads for deep shard validation.",
    )

    args = parser.parse_args()

    validate_dataset(
        Path(args.shard_dir),
        args.expected_shards,
        allow_incomplete_provenance=bool(args.allow_incomplete_provenance),
        workers=int(args.workers),
    )
