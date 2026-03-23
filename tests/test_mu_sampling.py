import unittest

import numpy as np

from scripts.generate_grid import _resolve_ml_sampling
from scripts.run_turbospectrum import TurbospectrumConfig
from scripts.synthesize_spectra_from_zarr import _choose_mu_indices


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


if __name__ == "__main__":
    unittest.main()
