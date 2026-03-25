import os
import tempfile
import unittest

import numpy as np

from scripts.generate_grid import _resolve_ml_sampling
from scripts.nlte_ascii_departures import (
    build_nlte_ascii_selector_columns,
    materialize_nlte_info_with_departure_override,
    normalize_nlte_ascii_selector,
    resolve_absolute_abundance,
    select_departure_file,
)
from scripts.run_turbospectrum import get_synthesis_output_stem_from_params
from scripts.synthesize_regular_grid import _build_regular_columns


class NlteAsciiDepartureTests(unittest.TestCase):
    _MODEL_STEM = "p5000_g+4.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00"

    def test_build_selector_columns_returns_constant_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = normalize_nlte_ascii_selector(directory=tmpdir, species="Fe")
            columns = build_nlte_ascii_selector_columns(3, selector)

            self.assertEqual(columns["nlte_ascii_departure_dir"].tolist(), [tmpdir, tmpdir, tmpdir])
            self.assertEqual(columns["nlte_ascii_departure_species"].tolist(), ["Fe", "Fe", "Fe"])
            self.assertEqual(columns["nlte_ascii_abundance_column"].tolist(), ["auto", "auto", "auto"])
            self.assertEqual(columns["nlte_ascii_abundance_scale"].tolist(), ["relative", "relative", "relative"])
            self.assertEqual(columns["nlte_ascii_match"].tolist(), ["nearest", "nearest", "nearest"])

    def test_resolve_absolute_abundance_uses_feh_for_fe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = normalize_nlte_ascii_selector(directory=tmpdir, species="Fe")
            abundance, column = resolve_absolute_abundance({"feh": "-0.50"}, selector)

            self.assertEqual(column, "feh")
            self.assertAlmostEqual(abundance, 7.00, places=6)

    def test_select_departure_file_picks_nearest_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path_lo = os.path.join(tmpdir, f"000001_{self._MODEL_STEM}_abu+7.250.dat")
            path_hi = os.path.join(tmpdir, f"000002_{self._MODEL_STEM}_abu+7.500.dat")
            for path in (path_lo, path_hi):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("dummy\n")

            candidate = select_departure_file(
                directory=tmpdir,
                model_stem=self._MODEL_STEM,
                abundance=7.41,
                match="nearest",
            )

            self.assertEqual(candidate.path, path_hi)
            self.assertAlmostEqual(candidate.abundance, 7.5, places=6)

    def test_materialize_nlte_info_with_departure_override_rewrites_target_species(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "atoms")
            dep_dir = os.path.join(tmpdir, "dep")
            ascii_dir = os.path.join(tmpdir, "ascii")
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(dep_dir, exist_ok=True)
            os.makedirs(ascii_dir, exist_ok=True)

            base_ca = os.path.join(dep_dir, "base_ca.dat")
            base_fe = os.path.join(dep_dir, "base_fe.bin")
            override_fe = os.path.join(ascii_dir, f"000001_{self._MODEL_STEM}_abu+7.500.dat")
            for path in (base_ca, base_fe, override_fe):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("dummy\n")

            nlte_info = os.path.join(tmpdir, "SPECIES_LTE_NLTE.dat")
            with open(nlte_info, "w", encoding="utf-8") as handle:
                handle.write("# path for model atom files     ! don't change this line !\n")
                handle.write("atoms/\n")
                handle.write("#\n")
                handle.write("# path for departure files      ! don't change this line !\n")
                handle.write("dep/\n")
                handle.write("#\n")
                handle.write("# atomic (N)LTE setup\n")
                handle.write("20 'Ca' 'lte' 'atom.ca105b' 'base_ca.dat' 'ascii'\n")
                handle.write("26 'Fe' 'nlte' 'atom.fe607a' 'base_fe.bin' 'binary'\n")

            selector = normalize_nlte_ascii_selector(directory=ascii_dir, species="Fe")
            runtime_info = materialize_nlte_info_with_departure_override(
                base_info_path=nlte_info,
                selector=selector,
                departure_file_path=override_fe,
                output_root=os.path.join(tmpdir, "runtime"),
            )

            with open(runtime_info, "r", encoding="utf-8") as handle:
                contents = handle.read()

            self.assertIn(os.path.basename(override_fe), contents)
            self.assertIn("'ascii'", contents)
            self.assertIn("'base_ca.dat'", contents)

        self.assertTrue(True)

    def test_output_stem_ignores_nlte_ascii_control_keys(self) -> None:
        base = {
            "teff": 5000,
            "logg": 4.0,
            "feh": -0.3,
            "turbvel": "01",
            "t_value": "01",
            "a": "+0.00",
            "lam_min": 6000.0,
            "lam_max": 6100.0,
            "lam_step": 0.01,
            "output_mode": "Flux",
            "calculation_mode": "LTE",
        }

        stem_without = get_synthesis_output_stem_from_params(base)
        stem_with = get_synthesis_output_stem_from_params(
            {
                **base,
                "nlte_ascii_departure_dir": "/tmp/example",
                "nlte_ascii_departure_species": "Fe",
                "nlte_ascii_abundance_column": "feh",
                "nlte_ascii_abundance_scale": "relative",
                "nlte_ascii_match": "nearest",
            }
        )

        self.assertEqual(stem_without, stem_with)

    def test_resolve_ml_sampling_adds_nlte_ascii_selector_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            columns = _resolve_ml_sampling(
                {
                    "num_samples": 4,
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
                        "output_mode": "Flux",
                    },
                    "nlte_ascii_departures": {
                        "directory": tmpdir,
                        "species": "Fe",
                    },
                },
                rng=np.random.default_rng(1234),
            )

            self.assertIn("nlte_ascii_departure_dir", columns)
            self.assertEqual(columns["nlte_ascii_departure_dir"].shape[0], 4)

    def test_build_regular_columns_adds_nlte_ascii_selector_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = normalize_nlte_ascii_selector(directory=tmpdir, species="Fe")
            columns = _build_regular_columns(
                teff_axis=np.asarray([4000, 4250], dtype=np.int64),
                logg_axis=np.asarray([1.0], dtype=np.float64),
                feh_axis=np.asarray([-0.5], dtype=np.float64),
                turbvel_axis=np.asarray(["01"], dtype=object),
                mu_axis=None,
                grid_version="regular-linear-v1",
                lam_min=8400.0,
                lam_max=8800.0,
                lam_step=0.01,
                output_mode="Flux",
                mode="1D",
                calculation_mode="LTE",
                abundances={
                    "a": "+0.00",
                    "c": "+0.00",
                    "n": "+0.00",
                    "o": "+0.00",
                    "r": "+0.00",
                    "s": "+0.00",
                },
                nlte_ascii_selector=selector,
                max_rows=100,
            )

            self.assertIn("nlte_ascii_departure_dir", columns)
            self.assertEqual(columns["nlte_ascii_departure_species"].tolist(), ["Fe", "Fe"])


if __name__ == "__main__":
    unittest.main()
