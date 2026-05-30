import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.generate_grid import _resolve_ml_sampling
from scripts.run_turbospectrum import (
    TurbospectrumConfig,
    _coerce_mu_points,
    _normalize_config_dict,
    _write_mu_points_file,
)
from scripts.synthesize_regular_grid import _materialize_synthesis_config
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

    def test_choose_mu_indices_reduce_first_with_full_count(self) -> None:
        # Regression: count >= len(mu_points) previously averaged all chosen
        # points into mu_summary, making it identical for every target_mu even
        # though reduce="first" uses only chosen[0]. mu_summary must reflect
        # the mu actually used.
        cfg = TurbospectrumConfig(
            project_root="",
            mu_sampling={
                "mode": "nearest",
                "count": 10,
                "min": 0.1,
                "max": 1.0,
                "reduce": "first",
            },
        )
        mu_points = np.asarray(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32
        )

        _, mu_low = _choose_mu_indices(
            mu_points, row_index=0, cfg=cfg, target_mu=0.25, reduce_mode="first"
        )
        _, mu_high = _choose_mu_indices(
            mu_points, row_index=1, cfg=cfg, target_mu=0.82, reduce_mode="first"
        )

        self.assertAlmostEqual(mu_low, 0.2, places=5)
        self.assertAlmostEqual(mu_high, 0.8, places=5)
        self.assertNotAlmostEqual(mu_low, mu_high, places=3)

    def test_choose_mu_indices_accepts_exact_mode_alias(self) -> None:
        # "exact" is a clearer label for the same behavior as "nearest"/"target"
        # when a per-row target_mu is supplied (bsyn already gets MU-POINTS=[mu]
        # so the spectrum has one column and selection is an identity).
        cfg = TurbospectrumConfig(
            project_root="",
            mu_sampling={"mode": "exact", "count": 1, "min": 0.0, "max": 1.0},
        )
        mu_points = np.asarray([0.05, 0.3, 0.55, 0.85], dtype=np.float32)
        chosen, mu_summary = _choose_mu_indices(
            mu_points, row_index=0, cfg=cfg, target_mu=0.6
        )
        np.testing.assert_array_equal(chosen, np.asarray([2], dtype=np.int64))
        self.assertAlmostEqual(mu_summary, 0.55, places=6)

    def test_resolve_ml_sampling_adds_lhs_mu_column_for_exact_mode(self) -> None:
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
                "abundances": {k: "+0.00" for k in "acnors"},
                "synthesis": {
                    "lam_min": 4000.0,
                    "lam_max": 5000.0,
                    "lam_step": 0.1,
                    "output_mode": "Intensity",
                    "mu_sampling": {
                        "mode": "exact",
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

    def test_choose_mu_indices_reduce_mean_still_averages(self) -> None:
        cfg = TurbospectrumConfig(
            project_root="",
            mu_sampling={
                "mode": "nearest",
                "count": 3,
                "min": 0.0,
                "max": 1.0,
                "reduce": "mean",
            },
        )
        mu_points = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)

        _, mu_mean = _choose_mu_indices(
            mu_points, row_index=0, cfg=cfg, target_mu=0.45, reduce_mode="mean"
        )
        # Three nearest to 0.45: 0.5, 0.3, 0.7 → mean = 0.5.
        self.assertAlmostEqual(mu_mean, 0.5, places=5)


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


class ExactMuPointsTests(unittest.TestCase):
    """bsyn should synthesise the requested mu exactly (via a MU-POINTS file)
    rather than snapping to its 12 Gauss-Radau defaults."""

    def test_coerce_mu_points_dedups_sorts_and_clips(self) -> None:
        self.assertEqual(_coerce_mu_points([0.8, 0.2, 0.8, 0.5]), [0.2, 0.5, 0.8])
        # Tiny float overshoot within tolerance is clipped back into the domain.
        self.assertEqual(_coerce_mu_points([1.0 + 1e-9, -1e-9]), [0.0, 1.0])
        # Blank / None entries are ignored.
        self.assertEqual(_coerce_mu_points([None, "", 0.3]), [0.3])

    def test_coerce_mu_points_rejects_out_of_domain(self) -> None:
        for bad in ([1.5], [-0.5], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                _coerce_mu_points(bad)

    def test_coerce_mu_points_enforces_bsyn_limit(self) -> None:
        # 31 distinct values exceeds bsyn's muoutp(30).
        too_many = [i / 40.0 for i in range(1, 32)]
        with self.assertRaises(ValueError):
            _coerce_mu_points(too_many)
        # Exactly 30 is allowed.
        ok = [i / 40.0 for i in range(1, 31)]
        self.assertEqual(len(_coerce_mu_points(ok)), 30)

    def test_write_mu_points_file_matches_bsyn_format(self) -> None:
        # bsyn reads `nangles` from line 1 then the values from line 2
        # (source/input.f MU-POINTS branch).
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_mu_points_file([0.2, 0.55, 1.0], tmp, "p5000_g+4.0")
            self.assertTrue(path.endswith("p5000_g+4.0.mupoints.dat"))
            with open(path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            self.assertEqual(int(lines[0]), 3)
            vals = [float(x) for x in lines[1].replace(",", " ").split()]
            np.testing.assert_allclose(vals, [0.2, 0.55, 1.0], atol=1e-6)

    def _write_base_config(self, tmp: str) -> str:
        base = os.path.join(tmp, "base.json")
        with open(base, "w", encoding="utf-8") as fh:
            json.dump({"synthesis_parameters": {"output_mode": "Intensity"}}, fh)
        return base

    def test_regular_grid_sets_exact_mu_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = _materialize_synthesis_config(
                base_config_path=self._write_base_config(tmp),
                run_root=tmp,
                mu_axis=np.array([0.8, 0.2, 0.8, 0.5]),
            )
            with open(out, encoding="utf-8") as fh:
                cfg = json.load(fh)
            ms = cfg["synthesis_parameters"]["mu_sampling"]
            self.assertEqual(ms["points"], [0.2, 0.5, 0.8])
            self.assertAlmostEqual(ms["min"], 0.2)
            self.assertAlmostEqual(ms["max"], 0.8)

    def test_regular_grid_rejects_more_than_30_mu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _materialize_synthesis_config(
                    base_config_path=self._write_base_config(tmp),
                    run_root=tmp,
                    mu_axis=np.linspace(0.02, 1.0, 31),
                )

    def test_regular_grid_sets_nearest_mode_for_mu_axis(self) -> None:
        # mu_range rows are explicit per-mu targets; the runtime config must
        # select "nearest" so each row keeps its own mu (not "random"/"none").
        with tempfile.TemporaryDirectory() as tmp:
            out = _materialize_synthesis_config(
                base_config_path=self._write_base_config(tmp),
                run_root=tmp,
                mu_axis=np.array([0.15, 0.45, 0.75]),
            )
            with open(out, encoding="utf-8") as fh:
                cfg = json.load(fh)
            self.assertEqual(cfg["synthesis_parameters"]["mu_sampling"]["mode"], "nearest")


class SynthesisDedupTests(unittest.TestCase):
    """Intensity rows that share a mu axis must collapse to one synthesis per
    physical point (one bsyn call computes all mu), not one per mu-row."""

    @staticmethod
    def _row(teff, mu):
        return {
            "teff": teff, "logg": 4.5, "feh": 0.0, "turbvel": "01", "t_value": "01",
            "lam_min": 5000.0, "lam_max": 5010.0, "lam_step": 0.01,
            "output_mode": "Intensity", "calculation_mode": "LTE", "mode": "1D", "mu": mu,
        }

    def test_group_tasks_dedups_shared_mu_rows(self) -> None:
        cfg = TurbospectrumConfig(
            project_root="/tmp",
            output_mode="Intensity",
            mu_sampling={"mode": "nearest", "points": [0.15, 0.45, 0.75], "min": 0.15, "max": 0.75},
        )
        tasks = []
        i = 0
        for teff in (5500, 6000):
            for mu in (0.15, 0.45, 0.75):
                tasks.append((i, self._row(teff, mu)))
                i += 1
        groups = synth_zarr._group_tasks(tasks, cfg)
        self.assertEqual(len(groups), 2)  # 6 rows -> 2 physical points
        self.assertTrue(all(len(g) == 3 for g in groups))
        for g in groups:
            self.assertEqual(len({t[1]["teff"] for t in g}), 1)  # one star per group

    def test_group_tasks_singletons_without_shared_axis(self) -> None:
        # No shared mu axis (e.g. Flux, or per-row LHS mu) => one group per row.
        cfg = TurbospectrumConfig(project_root="/tmp", output_mode="Flux", mu_sampling={})
        tasks = [
            (0, {"teff": 5500, "logg": 4.5, "feh": 0.0, "turbvel": "01", "t_value": "01",
                 "lam_min": 5000.0, "lam_max": 5010.0, "lam_step": 0.01,
                 "output_mode": "Flux", "calculation_mode": "LTE", "mode": "1D"}),
            (1, {"teff": 6000, "logg": 4.5, "feh": 0.0, "turbvel": "01", "t_value": "01",
                 "lam_min": 5000.0, "lam_max": 5010.0, "lam_step": 0.01,
                 "output_mode": "Flux", "calculation_mode": "LTE", "mode": "1D"}),
        ]
        groups = synth_zarr._group_tasks(tasks, cfg)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(g) == 1 for g in groups))


class IntensityModeLoadTests(unittest.TestCase):
    """Regression: loading an Intensity config must not force mu_sampling.mode to
    'random'. Doing so at load time defeats the per-row 'nearest' selection that
    keeps each target-mu row's own angle (bug: rows got an arbitrary mu)."""

    def test_nested_intensity_config_does_not_force_random(self) -> None:
        flat = _normalize_config_dict(
            {
                "synthesis_parameters": {
                    "output_mode": "Intensity",
                    "mu_sampling": {"points": [0.15, 0.45, 0.75], "min": 0.15, "max": 0.75},
                }
            },
            default_project_root="/tmp",
        )
        self.assertNotEqual(str(flat["mu_sampling"].get("mode", "none")).lower(), "random")

    def test_flat_intensity_config_does_not_force_random(self) -> None:
        out = _normalize_config_dict(
            {"output_mode": "Intensity", "mu_sampling": {"points": [0.2, 0.6]}},
            default_project_root="/tmp",
        )
        self.assertNotEqual(str(out["mu_sampling"].get("mode", "none")).lower(), "random")

    def test_explicit_user_mode_is_preserved(self) -> None:
        out = _normalize_config_dict(
            {"output_mode": "Intensity", "mu_sampling": {"mode": "random"}},
            default_project_root="/tmp",
        )
        self.assertEqual(str(out["mu_sampling"]["mode"]).lower(), "random")


class GridIoResolverTests(unittest.TestCase):
    """Compression/chunk settings must resolve uniformly across grid paths."""

    def _resolve(self, grid_cfg):
        from scripts.generate_grid import resolve_grid_io
        return resolve_grid_io(grid_cfg)

    def test_canonical_grid_io_block(self) -> None:
        io = self._resolve({
            "io": {
                "csv_compression": "gzip",
                "csv_compression_level": 3,
                "zarr_chunks": 8192,
                "zarr_compressor": {"cname": "zstd", "clevel": 9, "shuffle": False},
            }
        })
        self.assertEqual(io["csv_compression"], "gzip")
        self.assertEqual(io["csv_compression_level"], 3)
        self.assertEqual(io["zarr_chunks"], 8192)
        self.assertEqual(io["zarr_compressor"]["clevel"], 9)

    def test_legacy_flat_keys_still_honoured(self) -> None:
        io = self._resolve({
            "csv_compression": "gzip",
            "csv_compression_level": 7,
            "zarr": {"chunks": 1024, "compressor": {"cname": "lz4"}},
        })
        self.assertEqual(io["csv_compression"], "gzip")
        self.assertEqual(io["csv_compression_level"], 7)
        self.assertEqual(io["zarr_chunks"], 1024)
        self.assertEqual(io["zarr_compressor"]["cname"], "lz4")

    def test_canonical_wins_over_legacy(self) -> None:
        io = self._resolve({
            "io": {"zarr_chunks": 256},
            "zarr": {"chunks": 1024},
        })
        self.assertEqual(io["zarr_chunks"], 256)

    def test_uniform_defaults_when_omitted(self) -> None:
        # Both an empty config and a path-only zarr block fall back to the same
        # zstd defaults, so every grid path behaves identically.
        for cfg in ({}, {"zarr": {"path": "x.zarr"}}):
            io = self._resolve(cfg)
            self.assertEqual(io["csv_compression"], "zstd")
            self.assertEqual(io["csv_compression_level"], 5)
            self.assertEqual(io["zarr_chunks"], 2048)
            self.assertEqual(io["zarr_compressor"], {"cname": "zstd", "clevel": 5, "shuffle": True})


if __name__ == "__main__":
    unittest.main()
