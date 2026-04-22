"""Tests for scripts/verify_pipeline_consistency.py

Builds a synthetic pipeline on disk (grid zarr + shard zarrs + optional
merged zarr + fake atmosphere directory) and exercises each of the
verifier's checks in isolation — confirming that clean artifacts pass,
and that each planted inconsistency is surfaced.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import zarr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for p in (REPO_ROOT, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.zarr_compat import write_string_array  # noqa: E402
from scripts.verify_pipeline_consistency import (  # noqa: E402
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIP,
    _resolve_paths,
    check_grid_matches_config,
    check_intermediate_artifacts,
    check_merged_consistency,
    check_shards_match_grid,
    check_t_value_matches_turbvel,
)


################################################################################
# Fixture helpers
################################################################################


def _write_zarr(path: str, arrays: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    root = zarr.open_group(path, mode="w")
    for name, arr in arrays.items():
        if arr.dtype.kind in ("O", "U", "S"):
            write_string_array(
                root, name, ["" if v is None else str(v) for v in arr.tolist()], chunks=max(1, arr.size)
            )
        else:
            root.create_array(name, shape=arr.shape, dtype=arr.dtype, chunks=arr.shape or (1,))[...] = arr


def _make_atmospheres(directory: str, labels: Iterable[str]) -> None:
    os.makedirs(directory, exist_ok=True)
    for label in labels:
        for teff, logg, feh in [(5000, "+4.0", "+0.00"), (5250, "+4.5", "-0.50")]:
            name = (
                f"p{teff}_g{logg}_m0.0_t{label}_st_z{feh}_a+0.00_"
                f"c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod"
            )
            open(os.path.join(directory, name), "w").close()


class _Fixture:
    def __init__(self, tmp: str) -> None:
        self.tmp = tmp
        self.run_root = os.path.join(tmp, "run")
        self.grid_zarr = os.path.join(self.run_root, "outputs", "grids", "grid.zarr")
        self.shards_dir = os.path.join(self.run_root, "outputs", "shards")
        self.merged_zarr = os.path.join(self.run_root, "outputs", "zarr", "merged.zarr")
        self.atmosphere_dir = os.path.join(tmp, "atmospheres")
        self.output_dir = os.path.join(self.run_root, "outputs", "spectra")
        self.log_dir = os.path.join(self.run_root, "logs", "shards")
        self.synthesis_config_path = os.path.join(tmp, "synthesis.json")
        self.pipeline_config_path = os.path.join(tmp, "pipeline.json")

    def write_synthesis_config(self) -> None:
        cfg = {
            "project_root": self.tmp,
            "paths": {
                "model_atmosphere_path": self.atmosphere_dir,
                "output_dir": self.output_dir,
                "log_dir": self.log_dir,
            },
        }
        with open(self.synthesis_config_path, "w") as h:
            json.dump(cfg, h)

    def write_pipeline_config(
        self,
        *,
        axes: Optional[Dict[str, str]] = None,
        abundances: Optional[Dict[str, str]] = None,
        synthesis: Optional[Dict[str, object]] = None,
    ) -> str:
        axes = axes or {"teff": "5000,5250", "logg": "4.0,4.5", "feh": "-0.5,0.0"}
        abundances = abundances or {"a": "+0.00"}
        synthesis = synthesis or {"lam_min": 5000.0, "lam_max": 5020.0, "lam_step": 0.01}
        cfg = {
            "grid": {"axes": axes, "abundances": abundances, "synthesis": synthesis},
            "outputs": {
                "run_root": self.run_root,
                "grid_zarr": "outputs/grids/grid.zarr",
                "spectra_zarr": "outputs/zarr/merged.zarr",
                "spectra_shard_template": "outputs/shards/spectra_shard_{shard_index}.zarr",
            },
            "turbospectrum": {
                "config": os.path.relpath(self.synthesis_config_path, os.path.dirname(self.pipeline_config_path))
            },
        }
        with open(self.pipeline_config_path, "w") as h:
            json.dump(cfg, h)
        return self.pipeline_config_path

    # ------------------------------------------------------------------ grids

    def write_grid(
        self,
        *,
        teff=(5000, 5000, 5250, 5250),
        logg=(4.0, 4.5, 4.0, 4.5),
        feh=(-0.5, 0.0, -0.5, 0.0),
        turbvel=(2.0, 2.0, 2.5, 2.5),
        t_value=("02", "02", "02", "02"),
        extras: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        n = len(teff)
        arrays: Dict[str, np.ndarray] = {
            "teff": np.asarray(teff, dtype=np.int64),
            "logg": np.asarray(logg, dtype=np.float64),
            "feh": np.asarray(feh, dtype=np.float64),
            "lam_min": np.full(n, 5000.0, dtype=np.float64),
            "lam_max": np.full(n, 5020.0, dtype=np.float64),
            "lam_step": np.full(n, 0.01, dtype=np.float64),
            "turbvel": np.asarray([str(v) for v in turbvel], dtype=object),
            "t_value": np.asarray(t_value, dtype=object),
            "a": np.asarray(["+0.00"] * n, dtype=object),
            "output_mode": np.asarray(["Flux"] * n, dtype=object),
            "calculation_mode": np.asarray(["LTE"] * n, dtype=object),
        }
        if extras:
            arrays.update(extras)
        _write_zarr(self.grid_zarr, arrays)

    def write_shard(
        self,
        shard_index: int,
        global_indices: Iterable[int],
        *,
        overrides: Optional[Dict[str, np.ndarray]] = None,
        include_flux: bool = True,
    ) -> str:
        gidx = np.asarray(list(global_indices), dtype=np.int64)
        grid_root = zarr.open_group(self.grid_zarr, mode="r")
        cols = {}
        for name in grid_root.array_keys():
            cols[name] = np.asarray(grid_root[name][gidx])
        if overrides:
            for k, v in overrides.items():
                cols[k] = np.asarray(v)
        shard_arrays: Dict[str, np.ndarray] = {
            "global_index": gidx,
            "wavelength": np.linspace(5000.0, 5020.0, 10, dtype=np.float64),
            "status": np.asarray(["success"] * gidx.size, dtype=object),
            "message": np.asarray([""] * gidx.size, dtype=object),
        }
        if include_flux:
            shard_arrays["flux"] = np.ones((gidx.size, 10), dtype=np.float32)
            shard_arrays["continuum"] = np.ones((gidx.size, 10), dtype=np.float32)
        shard_arrays.update(cols)
        shard_path = os.path.join(self.shards_dir, f"spectra_shard_{shard_index:04d}.zarr")
        _write_zarr(shard_path, shard_arrays)
        return shard_path

    def write_merged(self, row_count: int) -> None:
        arrays = {
            "flux": np.ones((row_count, 10), dtype=np.float32),
            "continuum": np.ones((row_count, 10), dtype=np.float32),
            "wavelength": np.linspace(5000.0, 5020.0, 10, dtype=np.float64),
            "global_index": np.arange(row_count, dtype=np.int64),
        }
        _write_zarr(self.merged_zarr, arrays)


################################################################################
# Tests
################################################################################


class PathResolutionTests(unittest.TestCase):
    def test_resolves_all_artifact_locations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.write_synthesis_config()
            fx.write_pipeline_config()
            paths = _resolve_paths(fx.pipeline_config_path)
            self.assertEqual(paths.grid_zarr, fx.grid_zarr)
            self.assertEqual(paths.shards_dir, fx.shards_dir)
            self.assertEqual(paths.merged_zarr, fx.merged_zarr)
            self.assertEqual(paths.model_atmosphere_path, fx.atmosphere_dir)
            self.assertEqual(paths.output_dir, fx.output_dir)


class GridMatchesConfigTests(unittest.TestCase):
    def _clean(self, fx: _Fixture) -> None:
        fx.write_synthesis_config()
        fx.write_pipeline_config()
        fx.write_grid()

    def test_clean_grid_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            self._clean(fx)
            paths = _resolve_paths(fx.pipeline_config_path)
            cfg = json.load(open(fx.pipeline_config_path))
            result = check_grid_matches_config(paths, cfg)
            self.assertEqual(result.status, STATUS_PASS)

    def test_teff_axis_mismatch_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.write_synthesis_config()
            fx.write_pipeline_config()
            fx.write_grid(teff=(4000, 4000, 4250, 4250))  # config says 5000/5250
            paths = _resolve_paths(fx.pipeline_config_path)
            cfg = json.load(open(fx.pipeline_config_path))
            result = check_grid_matches_config(paths, cfg)
            self.assertEqual(result.status, STATUS_FAIL)
            self.assertTrue(any("teff" in issue for issue in result.issues))

    def test_mu_range_without_mu_column_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.write_synthesis_config()
            fx.write_pipeline_config(
                synthesis={
                    "lam_min": 5000.0,
                    "lam_max": 5020.0,
                    "lam_step": 0.01,
                    "output_mode": "Intensity",
                    "mu_range": "0.1:1.0:0.1",
                }
            )
            fx.write_grid()  # no mu column
            paths = _resolve_paths(fx.pipeline_config_path)
            cfg = json.load(open(fx.pipeline_config_path))
            result = check_grid_matches_config(paths, cfg)
            self.assertEqual(result.status, STATUS_FAIL)
            self.assertTrue(any("mu" in issue and "column" in issue for issue in result.issues))

    def test_missing_grid_zarr_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.write_synthesis_config()
            fx.write_pipeline_config()
            paths = _resolve_paths(fx.pipeline_config_path)
            cfg = json.load(open(fx.pipeline_config_path))
            result = check_grid_matches_config(paths, cfg)
            self.assertEqual(result.status, STATUS_SKIP)


class ShardsMatchGridTests(unittest.TestCase):
    def _setup(self, fx: _Fixture) -> None:
        fx.write_synthesis_config()
        fx.write_pipeline_config()
        fx.write_grid()

    def test_clean_shards_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            self._setup(fx)
            fx.write_shard(0, [0, 1])
            fx.write_shard(1, [2, 3])
            paths = _resolve_paths(fx.pipeline_config_path)
            result = check_shards_match_grid(paths, sample_size=8)
            self.assertEqual(result.status, STATUS_PASS, result.issues)
            self.assertEqual(result.stats["shards_total"], 2)
            self.assertEqual(result.stats["row_mismatches"], 0)

    def test_shard_with_corrupted_teff_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            self._setup(fx)
            fx.write_shard(0, [0, 1])
            # Corrupt teff in shard 1 — shard disagrees with grid at those rows.
            fx.write_shard(
                1,
                [2, 3],
                overrides={"teff": np.asarray([9999, 9999], dtype=np.int64)},
            )
            paths = _resolve_paths(fx.pipeline_config_path)
            result = check_shards_match_grid(paths, sample_size=8)
            self.assertEqual(result.status, STATUS_FAIL)
            self.assertGreater(result.stats["row_mismatches"], 0)
            self.assertTrue(any("teff" in issue for issue in result.issues))

    def test_shard_global_index_out_of_range_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            self._setup(fx)
            fx.write_shard(0, [0, 1])
            # Craft a shard with an index beyond the grid row count by
            # building it from in-range rows and then rewriting global_index.
            shard_path = fx.write_shard(1, [2, 3])
            bad = zarr.open_group(shard_path, mode="r+")
            bad["global_index"][:] = np.asarray([2, 999], dtype=np.int64)
            paths = _resolve_paths(fx.pipeline_config_path)
            result = check_shards_match_grid(paths, sample_size=8)
            self.assertEqual(result.status, STATUS_FAIL)
            self.assertTrue(any("out of" in issue for issue in result.issues))

    def test_turbvel_divergence_is_not_flagged(self) -> None:
        """Shard metadata may legitimately differ from the grid on t_value /
        turbvel (the shard stores the *resolved* values). The check ignores
        these two columns — regression guard for that carve-out.
        """
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            self._setup(fx)
            fx.write_shard(
                0,
                [0, 1],
                overrides={"t_value": np.asarray(["02", "02"], dtype=object)},
            )
            paths = _resolve_paths(fx.pipeline_config_path)
            result = check_shards_match_grid(paths, sample_size=8)
            self.assertEqual(result.status, STATUS_PASS, result.issues)


class TValueNearestTurbvelTests(unittest.TestCase):
    def test_clean_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.write_synthesis_config()
            fx.write_pipeline_config()
            _make_atmospheres(fx.atmosphere_dir, ("01", "02", "05"))
            fx.write_grid(
                turbvel=(2.0, 2.5, 1.5, 4.5),
                # Expected nearest for (2.0, 2.5, 1.5, 4.5) given {01,02,05}:
                #   2.0 -> 02; 2.5 -> 02; 1.5 -> 01 (tie -> smaller); 4.5 -> 05.
                t_value=("02", "02", "01", "05"),
            )
            fx.write_shard(0, [0, 1])
            fx.write_shard(1, [2, 3])
            paths = _resolve_paths(fx.pipeline_config_path)
            result = check_t_value_matches_turbvel(paths)
            self.assertEqual(result.status, STATUS_PASS, result.issues)
            self.assertEqual(result.stats["mismatches"], 0)

    def test_detects_wrong_t_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.write_synthesis_config()
            fx.write_pipeline_config()
            _make_atmospheres(fx.atmosphere_dir, ("01", "02", "05"))
            fx.write_grid(
                turbvel=(2.5, 2.5, 2.5, 2.5),
                # Deliberately set to "05" even though nearest is "02" — the
                # original bug's symptom if the rebind isn't applied.
                t_value=("05", "05", "05", "05"),
            )
            fx.write_shard(0, [0, 1, 2, 3])
            paths = _resolve_paths(fx.pipeline_config_path)
            result = check_t_value_matches_turbvel(paths)
            self.assertEqual(result.status, STATUS_FAIL)
            self.assertEqual(result.stats["mismatches"], 4)
            self.assertTrue(
                all("expected='02'" in issue for issue in result.issues[:4])
            )

    def test_missing_atmosphere_dir_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.write_synthesis_config()
            fx.write_pipeline_config()
            # Atmosphere dir deliberately not created.
            fx.write_grid()
            fx.write_shard(0, [0, 1])
            paths = _resolve_paths(fx.pipeline_config_path)
            result = check_t_value_matches_turbvel(paths)
            self.assertEqual(result.status, STATUS_SKIP)


class MergedConsistencyTests(unittest.TestCase):
    def test_merged_matches_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.write_synthesis_config()
            fx.write_pipeline_config()
            fx.write_grid()
            fx.write_shard(0, [0, 1])
            fx.write_shard(1, [2, 3])
            fx.write_merged(row_count=4)
            paths = _resolve_paths(fx.pipeline_config_path)
            result = check_merged_consistency(paths)
            self.assertEqual(result.status, STATUS_PASS, result.issues)

    def test_merged_fewer_rows_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.write_synthesis_config()
            fx.write_pipeline_config()
            fx.write_grid()
            fx.write_shard(0, [0, 1])
            fx.write_shard(1, [2, 3])
            fx.write_merged(row_count=2)  # missing rows
            paths = _resolve_paths(fx.pipeline_config_path)
            result = check_merged_consistency(paths)
            self.assertEqual(result.status, STATUS_FAIL)
            self.assertTrue(any("<" in issue for issue in result.issues))

    def test_no_merged_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.write_synthesis_config()
            fx.write_pipeline_config()
            fx.write_grid()
            fx.write_shard(0, [0, 1])
            paths = _resolve_paths(fx.pipeline_config_path)
            result = check_merged_consistency(paths)
            self.assertEqual(result.status, STATUS_SKIP)


class IntermediateArtifactsTests(unittest.TestCase):
    def test_skips_when_output_dir_empty_post_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.write_synthesis_config()
            fx.write_pipeline_config()
            fx.write_grid()
            fx.write_shard(0, [0, 1])
            os.makedirs(fx.output_dir, exist_ok=True)  # empty → cleaned-up state
            paths = _resolve_paths(fx.pipeline_config_path)
            result = check_intermediate_artifacts(paths, sample_size=2)
            self.assertEqual(result.status, STATUS_SKIP)

    def test_detects_missing_specs_when_some_present(self) -> None:
        """If the output dir has some .spec files, missing-for-sampled-rows
        should be reported as a failure — not downgraded to skip.
        """
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            fx.write_synthesis_config()
            fx.write_pipeline_config()
            _make_atmospheres(fx.atmosphere_dir, ("02",))
            fx.write_grid()
            fx.write_shard(0, [0, 1])
            os.makedirs(fx.output_dir, exist_ok=True)
            # Place an unrelated .spec to prevent the "empty dir" shortcut.
            open(os.path.join(fx.output_dir, "some_unrelated_spec.spec"), "w").close()
            paths = _resolve_paths(fx.pipeline_config_path)
            result = check_intermediate_artifacts(paths, sample_size=2)
            self.assertEqual(result.status, STATUS_FAIL)
            self.assertGreater(result.stats["missing_spec_files"], 0)


if __name__ == "__main__":
    unittest.main()
