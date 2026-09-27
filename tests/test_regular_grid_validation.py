import os
import tempfile
import unittest
import json

import numpy as np
import zarr

from scripts.synthesize_regular_grid import _build_regular_columns
from scripts.pipeline_from_config import _normalize_synthesis_overrides, _write_temp_turbospectrum_config
from scripts.synthesis_validation import validate_synthesis_results as _validate_synthesis_results


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

        # Returns (so the shard is still written) and logs an ERROR; the
        # caller exits non-zero based on the counts.
        with self.assertLogs("scripts.synthesis_validation", level="ERROR") as logs:
            counts = _validate_synthesis_results(
                statuses=["success", "error"],
                messages=["ok", "bsyn failed"],
                fluxes=fluxes,
                continua=continua,
            )

        self.assertEqual(counts, {"success": 1, "error": 1})
        self.assertIn("failed rows=1", logs.output[0])
        self.assertIn("bsyn failed", logs.output[0])

    def test_validate_synthesis_results_rejects_nonfinite_flux(self) -> None:
        fluxes = np.array([[1.0, 0.9], [np.nan, 0.7]], dtype=np.float32)
        continua = np.array([[1.0, 1.0], [0.9, 0.9]], dtype=np.float32)

        with self.assertLogs("scripts.synthesis_validation", level="ERROR") as logs:
            _validate_synthesis_results(
                statuses=["success", "success"],
                messages=["ok", "missing spectrum"],
                fluxes=fluxes,
                continua=continua,
            )

        self.assertIn("rows with non-finite flux=1", logs.output[0])

    def test_absolute_linelist_override_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            linelist_dir = os.path.join(tmpdir, "external", "linelists")
            os.makedirs(linelist_dir)
            out = _normalize_synthesis_overrides(
                {"paths": {"linelist_path": linelist_dir, "linelist_files": ["custom_list"]}},
                os.path.join(tmpdir, "configs", "pipeline"),
            )
            self.assertEqual(out["paths"]["linelist_path"], linelist_dir)
            self.assertEqual(out["paths"]["linelist_files"], ["custom_list"])

    def test_build_regular_columns_expands_uniform_mu_axis(self) -> None:
        abund = {k: np.asarray(["+0.00"], dtype=object) for k in ("a", "c", "n", "o", "r", "s")}
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
            abundances=abund,
            max_rows=100,
            t_value_options=["01"],
        )

        np.testing.assert_array_equal(columns["teff"], np.asarray([4000, 4000, 4250, 4250], dtype=np.int64))
        np.testing.assert_allclose(columns["mu"], np.asarray([0.2, 0.8, 0.2, 0.8], dtype=np.float64))

    def test_build_regular_columns_snaps_t_value_to_nearest_turbvel(self) -> None:
        abund = {"a": np.asarray(["+0.00"], dtype=object)}
        columns = _build_regular_columns(
            teff_axis=np.asarray([5000], dtype=np.int64),
            logg_axis=np.asarray([4.0], dtype=np.float64),
            feh_axis=np.asarray([0.0], dtype=np.float64),
            turbvel_axis=np.asarray(["01", "02", "03", "05"], dtype=object),
            mu_axis=None,
            grid_version="regular-linear-v1",
            lam_min=6000.0,
            lam_max=6100.0,
            lam_step=0.01,
            output_mode="Flux",
            mode="1D",
            calculation_mode="LTE",
            abundances=abund,
            max_rows=100,
            t_value_options=["00", "01", "02", "05"],
        )

        # turbvel=03 has equal distance to 02 and 05 — _nearest_turb_label
        # tie-breaks toward the smaller integer, so 02.
        np.testing.assert_array_equal(
            columns["t_value"],
            np.asarray(["01", "02", "02", "05"], dtype=object),
        )

    def test_pipeline_temp_turbospectrum_config_applies_base_config_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_dir = os.path.join(tmpdir, "configs", "pipeline")
            synth_dir = os.path.join(tmpdir, "configs", "synthesis")
            linelist_dir = os.path.join(tmpdir, "input_files", "linelists")
            nlte_info_path = os.path.join(tmpdir, "DATA", "SPECIES_LTE_NLTE.dat")
            os.makedirs(cfg_dir, exist_ok=True)
            os.makedirs(synth_dir, exist_ok=True)
            os.makedirs(linelist_dir, exist_ok=True)
            os.makedirs(os.path.dirname(nlte_info_path), exist_ok=True)

            with open(nlte_info_path, "w", encoding="utf-8") as handle:
                handle.write("fake nlte info\n")

            base_cfg_path = os.path.join(synth_dir, "config_sample_comprehensive.json")
            with open(base_cfg_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "project_root": "",
                        "paths": {
                            "linelist_path": "",
                            "linelist_files": ["base_list"]
                        },
                        "nlte": {
                            "enabled": False,
                            "nlte_info_file": ""
                        },
                        "synthesis_parameters": {
                            "output_mode": "Flux"
                        }
                    },
                    handle,
                )

            pipeline_cfg = {
                "turbospectrum": {
                    "config": "../synthesis/config_sample_comprehensive.json",
                    "overrides": {
                        "paths": {
                            "linelist_path": "../../input_files/linelists",
                            "linelist_files": ["custom_list"]
                        },
                        "nlte": {
                            "enabled": True,
                            "nlte_info_file": "../../DATA/SPECIES_LTE_NLTE.dat"
                        }
                    }
                }
            }

            materialized = _write_temp_turbospectrum_config(
                pipeline_cfg,
                config_dir=cfg_dir,
                scratch=None,
            )

            with open(materialized, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)

            self.assertEqual(cfg["paths"]["linelist_path"], linelist_dir)
            self.assertEqual(cfg["paths"]["linelist_files"], ["custom_list"])
            self.assertTrue(cfg["nlte"]["enabled"])
            self.assertEqual(cfg["nlte"]["nlte_info_file"], nlte_info_path)


if __name__ == "__main__":
    unittest.main()
