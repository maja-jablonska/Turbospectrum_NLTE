import os
import tempfile
import unittest
import json

import numpy as np
import zarr

from scripts.synthesize_regular_grid import (
    _build_regular_columns,
    _materialize_synthesis_config,
    _verify_continuum_saved,
)
from scripts.synthesize_spectra_from_zarr import _validate_synthesis_results


class RegularGridValidationTests(unittest.TestCase):
    @staticmethod
    def _create_array(group, name: str, data: np.ndarray) -> None:
        if hasattr(group, "create_array"):
            group.create_array(name, data=data)
        else:
            group.create_dataset(name, data=data)

    def test_validate_synthesis_results_accepts_finite_successful_rows(self) -> None:
        fluxes = np.array([[1.0, 0.9], [0.8, 0.7]], dtype=np.float32)
        continua = np.array([[1.0, 1.0], [0.9, 0.9]], dtype=np.float32)

        counts = _validate_synthesis_results(
            statuses=["success", "skipped"],
            messages=["ok", "cached"],
            fluxes=fluxes,
            continua=continua,
        )

        self.assertEqual(counts, {"success": 1, "skipped": 1})

    def test_validate_synthesis_results_rejects_failed_rows(self) -> None:
        fluxes = np.array([[1.0, 0.9], [0.8, 0.7]], dtype=np.float32)
        continua = np.array([[1.0, 1.0], [0.9, 0.9]], dtype=np.float32)

        with self.assertRaises(RuntimeError) as exc_info:
            _validate_synthesis_results(
                statuses=["success", "error"],
                messages=["ok", "bsyn failed"],
                fluxes=fluxes,
                continua=continua,
            )

        self.assertIn("failed rows=1", str(exc_info.exception))
        self.assertIn("bsyn failed", str(exc_info.exception))

    def test_validate_synthesis_results_rejects_nonfinite_flux(self) -> None:
        fluxes = np.array([[1.0, 0.9], [np.nan, 0.7]], dtype=np.float32)
        continua = np.array([[1.0, 1.0], [0.9, 0.9]], dtype=np.float32)

        with self.assertRaises(RuntimeError) as exc_info:
            _validate_synthesis_results(
                statuses=["success", "success"],
                messages=["ok", "missing spectrum"],
                fluxes=fluxes,
                continua=continua,
            )

        self.assertIn("rows with non-finite flux=1", str(exc_info.exception))

    def test_verify_continuum_saved_rejects_failed_status_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = zarr.open_group(tmpdir, mode="w")
            self._create_array(root, "wavelength", np.array([1.0, 2.0], dtype=np.float32))
            self._create_array(root, "flux", np.array([[1.0, 0.9]], dtype=np.float32))
            self._create_array(root, "continuum", np.array([[1.0, 1.0]], dtype=np.float32))
            root.attrs["status_counts"] = {"success": 1, "error": 2}

            with self.assertRaises(ValueError) as exc_info:
                _verify_continuum_saved(tmpdir)

            self.assertIn("status_counts", str(exc_info.exception))

    def test_verify_continuum_saved_accepts_success_only_status_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = zarr.open_group(tmpdir, mode="w")
            self._create_array(root, "wavelength", np.array([1.0, 2.0], dtype=np.float32))
            self._create_array(root, "flux", np.array([[1.0, 0.9]], dtype=np.float32))
            self._create_array(root, "continuum", np.array([[1.0, 1.0]], dtype=np.float32))
            root.attrs["status_counts"] = {"success": 1}

            _verify_continuum_saved(tmpdir)

    def test_materialize_synthesis_config_applies_inline_linelist_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg_path = os.path.join(tmpdir, "base.json")
            run_root = os.path.join(tmpdir, "run")
            cfg_dir = os.path.join(tmpdir, "configs", "pipeline")
            linelist_dir = os.path.join(tmpdir, "input_files", "linelists")
            os.makedirs(cfg_dir, exist_ok=True)
            os.makedirs(linelist_dir, exist_ok=True)

            with open(base_cfg_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "project_root": "",
                        "paths": {
                            "linelist_path": "",
                            "linelist_files": ["base_list"],
                        },
                        "synthesis_parameters": {
                            "output_mode": "Flux",
                        },
                    },
                    handle,
                )

            materialized = _materialize_synthesis_config(
                base_config_path=base_cfg_path,
                run_root=run_root,
                overrides={
                    "paths": {
                        "linelist_path": "../../input_files/linelists",
                        "linelist_files": ["custom_list"],
                    }
                },
                overrides_base_dir=cfg_dir,
            )

            self.assertTrue(materialized.startswith(os.path.join(run_root, "config")))
            with open(materialized, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)

            self.assertEqual(cfg["paths"]["linelist_path"], linelist_dir)
            self.assertEqual(cfg["paths"]["linelist_files"], ["custom_list"])

    def test_materialize_synthesis_config_applies_mu_range_to_written_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg_path = os.path.join(tmpdir, "base.json")
            run_root = os.path.join(tmpdir, "run")
            with open(base_cfg_path, "w", encoding="utf-8") as handle:
                json.dump({"synthesis_parameters": {}}, handle)

            materialized = _materialize_synthesis_config(
                base_config_path=base_cfg_path,
                run_root=run_root,
                mu_axis=np.asarray([0.2, 0.5, 0.8], dtype=np.float64),
            )

            with open(materialized, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)

            mu_sampling = cfg["synthesis_parameters"]["mu_sampling"]
            self.assertEqual(mu_sampling["mode"], "nearest")
            self.assertEqual(mu_sampling["count"], 1)
            self.assertEqual(mu_sampling["min"], 0.2)
            self.assertEqual(mu_sampling["max"], 0.8)

    def test_build_regular_columns_expands_uniform_mu_axis(self) -> None:
        columns = _build_regular_columns(
            teff_axis=np.asarray([4000, 4250], dtype=np.int64),
            logg_axis=np.asarray([1.0], dtype=np.float64),
            feh_axis=np.asarray([-0.5], dtype=np.float64),
            turbvel_axis=np.asarray(["01"], dtype=object),
            mu_axis=np.asarray([0.2, 0.8], dtype=np.float64),
            grid_version="regular-linear-v1",
            lam_min=8400.0,
            lam_max=8800.0,
            lam_step=0.01,
            output_mode="Intensity",
            mode="1D",
            calculation_mode="LTE",
            abundances={"a": "+0.00", "c": "+0.00", "n": "+0.00", "o": "+0.00", "r": "+0.00", "s": "+0.00"},
            max_rows=100,
        )

        np.testing.assert_array_equal(columns["teff"], np.asarray([4000, 4000, 4250, 4250], dtype=np.int64))
        np.testing.assert_allclose(columns["mu"], np.asarray([0.2, 0.8, 0.2, 0.8], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
