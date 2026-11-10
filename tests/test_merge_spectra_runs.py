"""Tests for cross-run merging (scripts/merge_spectra_runs.py).

Builds small, self-contained final-schema spectra.zarr stores and exercises the
combiner: param-name union, continuum union, per-row lineage, model_id recompute,
dedup, and the wavelength / strict-params guard rails.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import zarr


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGE_SCRIPT = os.path.join(REPO_ROOT, "scripts", "merge_spectra_runs.py")

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from zarr_compat import (  # noqa: E402
    zarr_store,
    create_root_group,
    create_array,
    write_string_array,
    write_string_scalar,
    write_fixed_string_scalar,
)

_WL = np.linspace(5000.0, 5010.0, 8).astype(np.float32)
_PROV_FILES = (
    "canonical_config.yaml",
    "synthesis_config.yaml",
    "input_config.json",
    "linelist_manifest.json",
    "atmosphere_manifest.json",
    "software_manifest.json",
    "environment.txt",
)


class MergeSpectraRunsTests(unittest.TestCase):
    @staticmethod
    def _write_run(
        path: str,
        *,
        n_rows: int,
        param_names: list[str],
        physics_hash: str,
        with_continuum: bool = True,
        base: float = 0.0,
        wavelengths: np.ndarray | None = None,
        params_matrix: np.ndarray | None = None,
    ) -> None:
        wl = _WL if wavelengths is None else np.asarray(wavelengths, dtype=np.float32)
        root = create_root_group(zarr_store(path), overwrite=True)
        create_array(root, "wavelength", data=wl, chunks=wl.shape)

        flux = (base + np.arange(n_rows)[:, None] + np.arange(wl.size)[None, :]).astype(np.float32)
        create_array(root, "flux", data=flux, chunks=(min(64, n_rows), wl.size))
        if with_continuum:
            create_array(root, "continuum", data=np.ones_like(flux), chunks=(min(64, n_rows), wl.size))

        if params_matrix is not None:
            params = np.asarray(params_matrix, dtype=np.float32)
        else:
            params = (
                base + np.arange(n_rows)[:, None] * 0.1 + np.arange(len(param_names))[None, :]
            ).astype(np.float32)
        create_array(root, "params", data=params, chunks=(min(64, n_rows), len(param_names)))
        write_string_array(root, "param_names", param_names, chunks=len(param_names))
        create_array(root, "model_id", data=np.arange(n_rows, dtype=np.uint64), chunks=(max(1, n_rows),))
        write_fixed_string_scalar(root, "physics_hash", physics_hash, min_width=64)
        write_fixed_string_scalar(root, "schema_version", "1.0.0", min_width=16)

        prov = root.create_group("provenance")
        for name in _PROV_FILES:
            write_string_scalar(prov, name, json.dumps({"run": physics_hash, "file": name}))

        root.attrs.update(
            {
                "linelist_identifier": "gaiaeso-6.1",
                "linelist_version": "6.1",
                "atmosphere_model_identifier": "MARCS",
                "turbospectrum_version": "20.1",
                "config_hash": "cfg" + physics_hash[:6],
                "grid_definition_hash": "grid" + physics_hash[:6],
                "output_mode_values": ["flux"],
                "flux_definition": "unnormalized_flux",
            }
        )

    @staticmethod
    def _run_merge(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, MERGE_SCRIPT, *args],
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _open(path: str):
        return zarr.open_group(store=zarr_store(path), mode="r")

    # -- safety ----------------------------------------------------------------

    def test_refuses_output_that_is_an_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r1 = os.path.join(tmp, "run1.zarr")
            self._write_run(r1, n_rows=3, param_names=["teff"], physics_hash="a" * 64)
            proc = self._run_merge("--run", r1, "--output-zarr", r1)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("also an input run", proc.stderr)
            self.assertEqual(tuple(self._open(r1)["flux"].shape), (3, 8))

    def test_mixed_configs_are_not_labelled_as_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r1 = os.path.join(tmp, "run1.zarr")
            r2 = os.path.join(tmp, "run2.zarr")
            out = os.path.join(tmp, "combined.zarr")
            self._write_run(r1, n_rows=3, param_names=["teff"], physics_hash="a" * 64)
            self._write_run(r2, n_rows=2, param_names=["teff"], physics_hash="b" * 64, base=10.0)
            proc = self._run_merge("--run", r1, "--run", r2, "--output-zarr", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            attrs = self._open(out).attrs
            self.assertTrue(attrs["config_hash"].startswith("multi:"))
            self.assertEqual(attrs["source_config_hashes"], ["cfg" + "a" * 6, "cfg" + "b" * 6])

    # -- happy path ----------------------------------------------------------

    def test_union_merge_concatenates_and_aligns_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r1 = os.path.join(tmp, "run1.zarr")
            r2 = os.path.join(tmp, "run2.zarr")
            out = os.path.join(tmp, "combined.zarr")
            self._write_run(r1, n_rows=5, param_names=["teff", "logg", "feh"], physics_hash="a" * 64, with_continuum=True)
            self._write_run(
                r2, n_rows=3, param_names=["teff", "logg", "feh", "vmicro"],
                physics_hash="b" * 64, with_continuum=False, base=100.0,
            )

            proc = self._run_merge("--run", r1, "--run", r2, "--output-zarr", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            root = self._open(out)
            self.assertEqual(tuple(root["flux"].shape), (8, 8))

            names = [str(x) for x in root["param_names"][:].tolist()]
            self.assertEqual(names, ["teff", "logg", "feh", "vmicro"])

            params = np.asarray(root["params"][:])
            # run1 lacks vmicro -> NaN; run2 has it -> finite.
            self.assertTrue(np.all(np.isnan(params[:5, 3])))
            self.assertTrue(np.all(np.isfinite(params[5:, 3])))

            # continuum present because run1 had it; run2 rows filled with NaN.
            cont = np.asarray(root["continuum"][:])
            self.assertTrue(np.all(cont[:5] == 1.0))
            self.assertTrue(np.all(np.isnan(cont[5:])))

            # per-row lineage
            self.assertEqual(np.asarray(root["run_index"][:]).tolist(), [0, 0, 0, 0, 0, 1, 1, 1])
            src = [str(x) for x in root["source_physics_hash"][:].tolist()]
            self.assertTrue(src[0].startswith("a"))
            self.assertTrue(src[-1].startswith("b"))

            self.assertEqual(root.attrs["n_models"], 8)
            self.assertEqual(root.attrs["runs_merged"], 2)
            self.assertEqual(root.attrs["run_rows_kept"], [5, 3])
            for name in _PROV_FILES:
                self.assertIn(name, root["provenance"])
            self.assertIn("merge_runs_manifest.json", root["provenance"])

    def test_model_id_recomputed_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r1 = os.path.join(tmp, "run1.zarr")
            out = os.path.join(tmp, "combined.zarr")
            self._write_run(r1, n_rows=4, param_names=["teff", "logg", "feh"], physics_hash="c" * 64)

            proc = self._run_merge("--run", r1, "--output-zarr", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            root = self._open(out)
            ids = np.asarray(root["model_id"][:])
            self.assertEqual(ids.shape, (4,))
            # ids are recomputed from params, not carried from the source model_id (0..3)
            self.assertGreater(int(ids.max()), 3)

    # -- dedup ---------------------------------------------------------------

    def test_dedup_by_model_id_drops_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r1 = os.path.join(tmp, "run1.zarr")
            r2 = os.path.join(tmp, "run2.zarr")
            out = os.path.join(tmp, "combined.zarr")
            # Identical params in both runs -> every run2 row is a duplicate.
            self._write_run(r1, n_rows=4, param_names=["teff", "logg", "feh"], physics_hash="a" * 64)
            self._write_run(r2, n_rows=4, param_names=["teff", "logg", "feh"], physics_hash="b" * 64)

            proc = self._run_merge("--run", r1, "--run", r2, "--output-zarr", out, "--dedup-by-model-id")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            root = self._open(out)
            self.assertEqual(int(root["flux"].shape[0]), 4)  # run2 fully deduplicated
            self.assertEqual(root.attrs["run_rows_kept"], [4, 0])

    def test_block_copy_matches_concatenation_with_drops_and_missing_continuum(self) -> None:
        # Uneven run lengths, rows dropped mid-run (dedup) and a run without continuum, with an
        # output chunk that divides neither: every io block size must give the plain concatenation.
        names = ["teff", "logg", "feh"]
        with tempfile.TemporaryDirectory() as tmp:
            r1, r2, r3 = (os.path.join(tmp, f"run{i}.zarr") for i in (1, 2, 3))
            self._write_run(r1, n_rows=37, param_names=names, physics_hash="a" * 64)
            self._write_run(r2, n_rows=50, param_names=names, physics_hash="b" * 64, with_continuum=False, base=500.0)
            p1 = (np.arange(37)[:, None] * 0.1 + np.arange(3)[None, :]).astype(np.float32)
            p3 = (1000.0 + np.arange(23)[:, None] * 0.1 + np.arange(3)[None, :]).astype(np.float32)
            dup = [2, 9, 10, 17]          # run3 rows that repeat run1 params -> dropped by dedup
            p3[dup] = p1[[0, 5, 6, 30]]
            self._write_run(r3, n_rows=23, param_names=names, physics_hash="c" * 64, base=2000.0, params_matrix=p3)

            def flux(n, base):
                return (base + np.arange(n)[:, None] + np.arange(_WL.size)[None, :]).astype(np.float32)

            keep3 = np.setdiff1d(np.arange(23), dup)
            want_flux = np.concatenate([flux(37, 0.0), flux(50, 500.0), flux(23, 2000.0)[keep3]])
            want_cont = np.concatenate([np.ones((37, _WL.size)), np.full((50, _WL.size), np.nan), np.ones((len(keep3), _WL.size))])
            for io_chunks in ("1", "3", "64"):
                out = os.path.join(tmp, f"combined_{io_chunks}.zarr")
                proc = self._run_merge("--run", r1, "--run", r2, "--run", r3, "--output-zarr", out,
                                       "--dedup-by-model-id", "--chunk-rows", "8", "--io-chunks", io_chunks)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                root = self._open(out)
                np.testing.assert_array_equal(root["flux"][...], want_flux)
                np.testing.assert_array_equal(root["continuum"][...], want_cont.astype(np.float32))
                self.assertEqual(root.attrs["run_rows_kept"], [37, 50, len(keep3)])

    def test_filter_bad_snaps_drops_wrong_atmosphere_rows(self) -> None:
        cols = ["teff", "logg", "feh", "t_value", "marcs_teff", "marcs_logg", "marcs_fe_h"]
        with tempfile.TemporaryDirectory() as tmp:
            atmo = os.path.join(tmp, "marcs")
            os.makedirs(atmo)
            for name in (
                "p5000_g+4.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
                "p5000_g+4.0_m0.0_t02_st_z-1.00_a+0.40_c+0.00_n+0.00_o+0.40_r+0.00_s+0.00.mod",
                "s4500_g+1.5_m1.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
            ):
                with open(os.path.join(atmo, name), "w", encoding="utf-8"):
                    pass

            r1 = os.path.join(tmp, "run1.zarr")
            r2 = os.path.join(tmp, "run2.zarr")
            out = os.path.join(tmp, "combined.zarr")
            # Row 0 of each run is correctly snapped; row 1 carries a wrong
            # recorded atmosphere (giant clamped to logg 4.0; t05 solar trap).
            self._write_run(r1, n_rows=2, param_names=cols, physics_hash="a" * 64, params_matrix=np.array([
                [5000.0, 4.0, 0.0, 1.0, 5000.0, 4.0, 0.0],
                [4500.0, 1.5, 0.0, 1.0, 5000.0, 4.0, 0.0],
            ]))
            self._write_run(r2, n_rows=2, param_names=cols, physics_hash="b" * 64, base=100.0, params_matrix=np.array([
                [5000.0, 4.0, 0.0, 1.0, 5000.0, 4.0, 0.0],
                [5000.0, 4.0, -1.0, 5.0, 5000.0, 4.0, 0.0],
            ]))

            proc = self._run_merge(
                "--run", r1, "--run", r2, "--output-zarr", out,
                "--filter-bad-snaps", "--atmosphere-path", atmo,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            root = self._open(out)
            self.assertEqual(int(root["flux"].shape[0]), 2)
            self.assertEqual(root.attrs["run_rows_kept"], [1, 1])
            self.assertEqual(root.attrs["run_rows_wrong_snap"], [1, 1])
            self.assertTrue(root.attrs["snap_filter_applied"])
            np.testing.assert_array_equal(root["run_index"][:], [0, 1])
            # The kept rows are row 0 of each run: flux row 0 starts at base.
            np.testing.assert_allclose(root["flux"][0, 0], 0.0)
            np.testing.assert_allclose(root["flux"][1, 0], 100.0)
            manifest_arr = np.asarray(root["provenance"]["merge_runs_manifest.json"][...]).ravel()
            manifest = json.loads(str(manifest_arr[0]))
            self.assertEqual(manifest["runs"][0]["snap_filter"]["n_wrong_snap"], 1)

    def test_filter_bad_snaps_requires_atmosphere_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r1 = os.path.join(tmp, "run1.zarr")
            self._write_run(r1, n_rows=2, param_names=["teff", "logg", "feh"], physics_hash="a" * 64)
            proc = self._run_merge(
                "--run", r1, "--output-zarr", os.path.join(tmp, "out.zarr"), "--filter-bad-snaps",
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("atmosphere-path", proc.stderr)

    # -- guard rails ---------------------------------------------------------

    def test_strict_params_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r1 = os.path.join(tmp, "run1.zarr")
            r2 = os.path.join(tmp, "run2.zarr")
            out = os.path.join(tmp, "combined.zarr")
            self._write_run(r1, n_rows=2, param_names=["teff", "logg", "feh"], physics_hash="a" * 64)
            self._write_run(r2, n_rows=2, param_names=["teff", "logg", "feh", "vmicro"], physics_hash="b" * 64)

            proc = self._run_merge("--run", r1, "--run", r2, "--output-zarr", out, "--strict-params")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("strict-params", proc.stderr)

    def test_wavelength_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r1 = os.path.join(tmp, "run1.zarr")
            r2 = os.path.join(tmp, "run2.zarr")
            out = os.path.join(tmp, "combined.zarr")
            self._write_run(r1, n_rows=2, param_names=["teff", "logg", "feh"], physics_hash="a" * 64)
            self._write_run(
                r2, n_rows=2, param_names=["teff", "logg", "feh"], physics_hash="b" * 64,
                wavelengths=np.linspace(6000.0, 6010.0, 8).astype(np.float32),
            )

            proc = self._run_merge("--run", r1, "--run", r2, "--output-zarr", out)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Wavelength grids diverge", proc.stderr)

    def test_run_dir_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "runs")
            os.makedirs(run_dir)
            self._write_run(os.path.join(run_dir, "a.zarr"), n_rows=2, param_names=["teff", "logg", "feh"], physics_hash="a" * 64)
            self._write_run(os.path.join(run_dir, "b.zarr"), n_rows=3, param_names=["teff", "logg", "feh"], physics_hash="b" * 64)
            out = os.path.join(tmp, "combined.zarr")

            proc = self._run_merge("--run-dir", run_dir, "--output-zarr", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(int(self._open(out)["flux"].shape[0]), 5)


if __name__ == "__main__":
    unittest.main()
