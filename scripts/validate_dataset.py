#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import zarr


############################################
# Helpers
############################################

SHARD_REGEX = re.compile(r"spectra_shard_(\d+)\.zarr")


def extract_shard_id(path: Path):
    m = SHARD_REGEX.match(path.name)
    return int(m.group(1)) if m else None


def is_valid_zarr(path: Path):
    try:
        zarr.open_group(path, mode="r")
        return True
    except Exception:
        return False


############################################
# Shard Validation
############################################

def validate_shard(path: Path, reference_wavelength=None):

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

        return True, info

    except Exception as e:
        return False, {"exception": str(e)}


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

def validate_dataset(shard_dir: Path, expected_shards: int):

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

    reference_wavelength = None
    total_spectra = 0
    failures = []

    for shard in shards:

        ok, info = validate_shard(shard, reference_wavelength)

        if not ok:
            failures.append((shard.name, info))
            continue

        if reference_wavelength is None:
            root = zarr.open_group(shard, mode="r")
            reference_wavelength = root["wavelength"][:]

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

    args = parser.parse_args()

    validate_dataset(
        Path(args.shard_dir),
        args.expected_shards
    )
