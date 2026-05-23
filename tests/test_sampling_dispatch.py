import unittest

import numpy as np

from scripts.generate_grid import resolve_grid_columns
from scripts.synthesize_regular_grid import resolve_regular_sampling


# Shared synthesis block + one fixed abundance so the LHS and regular paths
# declare the same columns and can be compared directly.
_SYNTH = {"lam_min": 5000.0, "lam_max": 5010.0, "lam_step": 0.05, "output_mode": "Flux"}
_ABUND = {"a": "+0.00"}

_LHS_CFG = {
    "grid_version": "lhs-test",
    "num_samples": 16,
    "seed": 7,
    "bounds": {
        "teff": {"min": 4000, "max": 6000},
        "logg": {"min": 1.0, "max": 4.0},
        "feh": {"min": -1.0, "max": 0.0},
    },
    "abundances": _ABUND,
    "synthesis": _SYNTH,
}

_GRID_CFG = {
    "grid_version": "grid-test",
    "axes": {
        "teff": "4000:6000:1000",   # 3 nodes
        "logg": "1.0:2.0:1.0",      # 2 nodes
        "feh": "-1.0:0.0:1.0",      # 2 nodes
        "turbvel": "01,02",          # 2 nodes
    },
    "abundances": _ABUND,
    "synthesis": _SYNTH,
    "limits": {"max_rows": 10_000},
}

_CORE_COLUMNS = {
    "grid_version", "teff", "logg", "feh", "lam_min", "lam_max", "lam_step",
    "turbvel", "t_value", "output_mode", "mode", "calculation_mode",
}


class SamplingDispatchTests(unittest.TestCase):
    def _rng(self):
        return np.random.default_rng(0)

    def test_lhs_mode_row_count(self) -> None:
        cols = resolve_grid_columns({**_LHS_CFG, "sampling": "lhs"}, self._rng())
        self.assertEqual(len(cols["teff"]), 16)

    def test_grid_mode_row_count(self) -> None:
        # 3 * 2 * 2 * 2 = 24 rows.
        cols = resolve_grid_columns({**_GRID_CFG, "sampling": "grid"}, self._rng())
        self.assertEqual(len(cols["teff"]), 24)

    def test_auto_detect_lhs_when_no_axes(self) -> None:
        cols = resolve_grid_columns(_LHS_CFG, self._rng())  # no sampling, no axes
        self.assertEqual(len(cols["teff"]), 16)

    def test_auto_detect_grid_when_axes_present(self) -> None:
        cols = resolve_grid_columns(_GRID_CFG, self._rng())  # no sampling, axes present
        self.assertEqual(len(cols["teff"]), 24)

    def test_both_modes_share_column_schema(self) -> None:
        lhs = resolve_grid_columns({**_LHS_CFG, "sampling": "lhs"}, self._rng())
        grid = resolve_grid_columns({**_GRID_CFG, "sampling": "grid"}, self._rng())
        self.assertEqual(set(lhs), set(grid))
        self.assertTrue(_CORE_COLUMNS.issubset(set(grid)))
        self.assertIn("a", grid)  # the shared configured abundance

    def test_aliases_accepted(self) -> None:
        self.assertEqual(
            len(resolve_grid_columns({**_GRID_CFG, "sampling": "cartesian"}, self._rng())["teff"]), 24
        )
        self.assertEqual(
            len(resolve_grid_columns({**_LHS_CFG, "sampling": "latin_hypercube"}, self._rng())["teff"]), 16
        )

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_grid_columns({**_LHS_CFG, "sampling": "bogus"}, self._rng())

    def test_continuous_vmicro_sampled_like_teff(self) -> None:
        cfg = {
            **_LHS_CFG,
            "sampling": "lhs",
            "bounds": {**_LHS_CFG["bounds"], "vmicro": {"min": 0.5, "max": 3.0}},
            "t_value_options": ["00", "01", "02", "05"],
        }
        cols = resolve_grid_columns(cfg, self._rng())
        turbvel = np.asarray(cols["turbvel"], dtype=float)
        # Continuous float in km/s, within the configured bounds, and not just
        # the integer labels the discrete path would have produced.
        self.assertEqual(turbvel.shape, (16,))
        self.assertTrue(np.all(turbvel >= 0.5) and np.all(turbvel <= 3.0))
        self.assertTrue(np.any(turbvel != np.round(turbvel)))
        # t_value is snapped to the nearest available atmosphere label.
        self.assertTrue(set(map(str, cols["t_value"])).issubset({"00", "01", "02", "05"}))

    def test_vmicro_bounds_and_sample_turbvel_conflict(self) -> None:
        cfg = {
            **_LHS_CFG,
            "sampling": "lhs",
            "bounds": {**_LHS_CFG["bounds"], "vmicro": {"min": 0.5, "max": 3.0}},
            "sample_turbvel": True,
        }
        with self.assertRaises(ValueError):
            resolve_grid_columns(cfg, self._rng())

    def test_warns_when_teff_outside_marcs(self) -> None:
        cfg = {
            **_LHS_CFG,
            "sampling": "lhs",
            "bounds": {**_LHS_CFG["bounds"], "teff": {"min": 7000, "max": 9000}},
        }
        with self.assertLogs("generate_grid", level="WARNING") as cm:
            resolve_grid_columns(cfg, self._rng())
        self.assertTrue(any("teff outside the MARCS grid envelope" in m for m in cm.output))

    def test_no_warning_when_within_marcs(self) -> None:
        with self.assertNoLogs("generate_grid", level="WARNING"):
            resolve_grid_columns({**_LHS_CFG, "sampling": "lhs"}, self._rng())

    def test_marcs_warning_can_be_disabled(self) -> None:
        cfg = {
            **_LHS_CFG,
            "sampling": "lhs",
            "bounds": {**_LHS_CFG["bounds"], "teff": {"min": 7000, "max": 9000}},
            "warn_outside_marcs_bounds": False,
        }
        with self.assertNoLogs("generate_grid", level="WARNING"):
            resolve_grid_columns(cfg, self._rng())

    def test_marcs_bounds_override_tightens_envelope(self) -> None:
        # An otherwise in-range feh trips the warning once the envelope is narrowed.
        cfg = {**_LHS_CFG, "sampling": "lhs", "marcs_bounds": {"feh": {"min": -0.5, "max": 0.0}}}
        with self.assertLogs("generate_grid", level="WARNING") as cm:
            resolve_grid_columns(cfg, self._rng())
        self.assertTrue(any("feh outside the MARCS grid envelope" in m for m in cm.output))

    def test_marcs_check_covers_regular_grid_path(self) -> None:
        cfg = {**_GRID_CFG, "sampling": "grid", "axes": {**_GRID_CFG["axes"], "teff": "7000:9000:1000"}}
        with self.assertLogs("generate_grid", level="WARNING") as cm:
            resolve_grid_columns(cfg, self._rng())
        self.assertTrue(any("teff outside the MARCS grid envelope" in m for m in cm.output))

    def test_regular_grid_continuous_vmicro_axis(self) -> None:
        cfg = {
            "sampling": "grid",
            "axes": {
                "teff": "5000:6000:1000",   # 2 nodes
                "logg": "4.0",              # 1 node
                "feh": "0.0",               # 1 node
                "vmicro": "0.5:2.5:0.5",    # 5 nodes, continuous km/s
                "t_value_options": ["00", "01", "02", "05"],
            },
            "abundances": _ABUND,
            "synthesis": _SYNTH,
            "limits": {"max_rows": 10_000},
        }
        cols = resolve_grid_columns(cfg, self._rng())
        turbvel = np.asarray(cols["turbvel"])
        # 2 teff * 1 logg * 1 feh * 5 vmicro = 10 rows, stored as floats (not labels).
        self.assertEqual(turbvel.shape, (10,))
        self.assertTrue(np.issubdtype(turbvel.dtype, np.floating))
        self.assertEqual(sorted(set(np.round(turbvel.astype(float), 3))), [0.5, 1.0, 1.5, 2.0, 2.5])
        self.assertTrue(set(map(str, cols["t_value"])).issubset({"00", "01", "02", "05"}))

    def test_regular_grid_vmicro_and_turbvel_conflict(self) -> None:
        cfg = {
            "sampling": "grid",
            "axes": {
                "teff": "5000", "logg": "4.0", "feh": "0.0",
                "vmicro": "0.5:2.5:0.5", "turbvel": "01,02",
            },
            "abundances": _ABUND,
            "synthesis": _SYNTH,
        }
        with self.assertRaises(ValueError):
            resolve_grid_columns(cfg, self._rng())

    def test_regular_intensity_tiles_mu_axis(self) -> None:
        cfg = {
            **_GRID_CFG,
            "synthesis": {**_SYNTH, "output_mode": "Intensity", "mu_range": "0.2:1.0:0.2"},
        }
        cols = resolve_regular_sampling(cfg)
        # 24 base rows * 5 mu nodes (0.2,0.4,0.6,0.8,1.0) = 120.
        self.assertEqual(len(cols["teff"]), 120)
        self.assertIn("mu", cols)
        self.assertEqual(sorted(set(np.round(cols["mu"], 3))), [0.2, 0.4, 0.6, 0.8, 1.0])

    def test_regular_mu_range_requires_intensity(self) -> None:
        cfg = {**_GRID_CFG, "synthesis": {**_SYNTH, "mu_range": "0.2:1.0:0.2"}}  # Flux
        with self.assertRaises(ValueError):
            resolve_regular_sampling(cfg)


class StandaloneMLSamplerTests(unittest.TestCase):
    """The standalone CLI must delegate to the single shared LHS sampler
    (`generate_grid.resolve_grid_columns`) rather than duplicate it."""

    def setUp(self) -> None:
        try:
            import scripts.sample_machine_learning_grid as smlg
            import zarr  # noqa: F401
        except Exception as exc:  # optional deps (polars/zarr) missing
            self.skipTest(f"sample_machine_learning_grid unavailable: {exc}")
        self._smlg = smlg

    def test_no_duplicate_sampler_helpers(self) -> None:
        # The bespoke LHS helpers must be gone — there is one sampler now.
        for gone in ("_latin_hypercube", "_resolve_sampling_dimensions", "_resolve_bounds"):
            self.assertFalse(hasattr(self._smlg, gone), f"{gone} should be removed")

    def test_standalone_delegates_and_samples_continuous_vmicro(self) -> None:
        import json
        import os
        import sys
        import tempfile

        import zarr

        smlg = self._smlg
        cfg = {
            "grid_version": "smlg-test",
            "num_samples": 12,
            "seed": 3,
            "bounds": {
                "teff": {"min": 4000, "max": 6000},
                "logg": {"min": 1.0, "max": 4.0},
                "feh": {"min": -1.0, "max": 0.0},
                "vmicro": {"min": 0.5, "max": 3.0},
            },
            "t_value_options": ["00", "01", "02", "05"],
            "abundances": {"a": "+0.00"},
            "synthesis": {"lam_min": 5000, "lam_max": 5010, "lam_step": 0.05, "output_mode": "Flux"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "cfg.json")
            zarr_path = os.path.join(tmp, "grid.zarr")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            argv = [
                "prog", "--config", cfg_path, "--zarr-output", zarr_path,
                "--index-parquet-output", os.path.join(tmp, "index.parquet"),
                "--no-progress",
            ]
            old_argv = sys.argv
            sys.argv = argv
            try:
                smlg.main()
            finally:
                sys.argv = old_argv

            root = zarr.open_group(store=smlg._zarr_store(zarr_path), mode="r")
            turbvel = np.asarray(root["turbvel"][:], dtype=float)
            self.assertEqual(turbvel.shape, (12,))
            self.assertTrue(np.all((turbvel >= 0.5) & (turbvel <= 3.0)))
            self.assertTrue(np.any(turbvel != np.round(turbvel)))  # genuinely continuous
            t_values = {str(v) for v in np.asarray(root["t_value"][:])}
            self.assertTrue(t_values.issubset({"00", "01", "02", "05"}))


if __name__ == "__main__":
    unittest.main()
