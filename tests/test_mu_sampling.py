import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.generate_grid import _resolve_ml_sampling
from scripts.run_turbospectrum import TurbospectrumConfig
from scripts.synthesize_spectra_from_zarr import _choose_mu_indices
import scripts.synthesize_spectra_from_zarr as synth_zarr


class MuSamplingTests(unittest.TestCase):
    def test_resolve_ml_sampling_adds_lhs_mu_column_for_nearest_mode(self) -> None:
        columns = _resolve_ml_sampling(
            {
                "num_samples": 8,
                "bounds": {
                    "teff": {"min": 4000, "max": 4500},
                    "logg": {"min": 1.0, "max": 2.0},
                    "feh": {"min": -1.0, "max": 0.0},
                },
                "turbvel": "01",
                "t_value_options": ["01"],
                "abundances": {
                    "a": "+0.00",
                    "c": "+0.00",
                    "n": "+0.00",
                    "o": "+0.00",
                    "r": "+0.00",
                    "s": "+0.00",
                },
                "synthesis": {
                    "lam_min": 4000.0,
                    "lam_max": 5000.0,
                    "lam_step": 0.1,
                    "output_mode": "Intensity",
                    "mu_sampling": {
                        "mode": "nearest",
                        "count": 1,
                        "min": 0.2,
                        "max": 0.8,
                    },
                },
            },
            rng=np.random.default_rng(1234),
        )

        self.assertIn("mu", columns)
        self.assertEqual(len(columns["mu"]), 8)
        self.assertGreaterEqual(float(np.min(columns["mu"])), 0.2)
        self.assertLessEqual(float(np.max(columns["mu"])), 0.8)
        self.assertGreater(len(np.unique(columns["mu"])), 1)

    def test_choose_mu_indices_uses_nearest_target_mu(self) -> None:
        cfg = TurbospectrumConfig(
            project_root="",
            mu_sampling={"mode": "nearest", "count": 1, "min": 0.0, "max": 1.0},
        )
        mu_points = np.asarray([0.05, 0.3, 0.55, 0.85], dtype=np.float32)

        chosen, mu_summary = _choose_mu_indices(
            mu_points,
            row_index=7,
            cfg=cfg,
            target_mu=0.6,
        )

        np.testing.assert_array_equal(chosen, np.asarray([2], dtype=np.int64))
        self.assertAlmostEqual(mu_summary, 0.55, places=6)


    def test_synthesis_task_fallback_uses_target_mu_when_header_missing(self) -> None:
        """When _read_mu_points returns empty (no mu-points header), the fallback
        for mode='nearest' must set mu_selected = target_mu, not NaN.

        Two rows with different target_mu must produce different mu_selected even
        when the spectrum file has no mu-points header line.
        """
        # Build a fake intensity spectrum: 3 header columns + 2*n_mu data columns.
        # Use 4 mu angles → 11 columns total. No "# mu-points" header line.
        n_mu = 4
        n_wl = 3
        n_cols = 3 + 2 * n_mu  # = 11
        data_rows = "\n".join(
            "  ".join(str(float(c)) for c in range(n_cols)) for _ in range(n_wl)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, "test.intensity.spec")
            with open(spec_path, "w") as fh:
                fh.write(data_rows + "\n")

            config = TurbospectrumConfig(
                project_root=tmpdir,
                output_mode="Intensity",
                nlte=False,
                output_dir=tmpdir,
                log_dir=tmpdir,
                tmp_dir=tmpdir,
                mu_sampling={"mode": "nearest", "count": 1, "min": 0.0, "max": 1.0},
            )
            synth_zarr._init_worker(config)

            results = []
            for target_mu in (0.2, 0.8):
                row_values = {
                    "teff": 5000,
                    "logg": 4.0,
                    "feh": 0.0,
                    "lam_min": 0.0,
                    "lam_max": float(n_wl - 1),
                    "lam_step": 1.0,
                    "turbvel": "01",
                    "t_value": "01",
                    "output_mode": "Intensity",
                    "calculation_mode": "LTE",
                    "mu": target_mu,
                }
                with mock.patch.object(
                    synth_zarr,
                    "run_single_synthesis",
                    return_value={
                        "status": "success",
                        "message": "ok",
                        "output_path": spec_path,
                        "base_name": "test",
                        "log_path": "",
                    },
                ):
                    result = synth_zarr._synthesis_task((0, row_values))
                results.append(result)

            mu_sel_02 = results[0]["mu_selected"]
            mu_sel_08 = results[1]["mu_selected"]

            # Both should be finite (not NaN) since target_mu is the fallback estimate.
            self.assertFalse(
                np.isnan(mu_sel_02),
                f"mu_selected should not be NaN for target_mu=0.2, got {mu_sel_02}",
            )
            self.assertFalse(
                np.isnan(mu_sel_08),
                f"mu_selected should not be NaN for target_mu=0.8, got {mu_sel_08}",
            )
            # The two rows should have different mu_selected.
            self.assertNotAlmostEqual(
                mu_sel_02,
                mu_sel_08,
                places=4,
                msg=f"mu_selected should differ between target_mu=0.2 ({mu_sel_02}) and 0.8 ({mu_sel_08})",
            )
            # mu_selected_index must be a valid column index.
            self.assertGreaterEqual(results[0]["mu_selected_index"], 0)
            self.assertGreaterEqual(results[1]["mu_selected_index"], 0)


if __name__ == "__main__":
    unittest.main()
