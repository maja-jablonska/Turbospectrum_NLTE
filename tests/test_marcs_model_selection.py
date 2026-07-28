"""MARCS atmosphere selection: exact-filename alpha coupling and the
nearest-neighbour snap (geometry coverage, turbulence weighting, tie-breaks).
"""
import os
import tempfile
import unittest
from types import SimpleNamespace

from scripts.run_turbospectrum import (
    ModelInterpolator,
    get_model_filename,
    standard_composition_alpha,
)


def _model_name(geom, teff, logg, mass, turb, feh, alpha):
    return (
        f"{geom}{teff}_g{logg:+.1f}_m{mass}_t{turb}_st_z{feh:+.2f}"
        f"_a{alpha:+.2f}_c+0.00_n+0.00_o{alpha:+.2f}_r+0.00_s+0.00.mod"
    )


def _make_store(tmpdir, names):
    for name in names:
        with open(os.path.join(tmpdir, name), "w", encoding="utf-8"):
            pass
    return SimpleNamespace(model_atmosphere_path=tmpdir)


class TestStandardCompositionAlpha(unittest.TestCase):
    def test_ramp_matches_marcs_standard_grid(self) -> None:
        for feh, alpha in [
            (1.0, 0.0), (0.5, 0.0), (0.0, 0.0),
            (-0.25, 0.1), (-0.5, 0.2), (-0.75, 0.3),
            (-1.0, 0.4), (-2.5, 0.4), (-5.0, 0.4),
        ]:
            self.assertAlmostEqual(standard_composition_alpha(feh), alpha, places=6)

    def test_exact_filename_for_subsolar_point(self) -> None:
        # The a/o fields must carry the standard alpha, or every subsolar grid
        # point misses its on-disk file and falls through to the snap.
        self.assertEqual(
            get_model_filename(5000, 4.0, -1.0, "01"),
            "p5000_g+4.0_m0.0_t01_st_z-1.00_a+0.40_c+0.00_n+0.00_o+0.40_r+0.00_s+0.00.mod",
        )
        self.assertEqual(
            get_model_filename(5000, 4.0, -0.5, "01"),
            "p5000_g+4.0_m0.0_t01_st_z-0.50_a+0.20_c+0.00_n+0.00_o+0.20_r+0.00_s+0.00.mod",
        )
        self.assertEqual(
            get_model_filename(5000, 4.0, 0.0, "01"),
            "p5000_g+4.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
        )


class TestNearestModelSelection(unittest.TestCase):
    def test_scan_includes_spherical_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = ModelInterpolator(_make_store(tmpdir, [
                _model_name("p", 5000, 4.0, "0.0", "01", 0.0, 0.0),
                _model_name("s", 4500, 1.5, "1.0", "01", 0.0, 0.0),
            ]))
            geoms = sorted(m["geom"] for m in lib.available_models)
            self.assertEqual(geoms, ["p", "s"])

    def test_giant_snaps_to_spherical_grid_point_not_logg_3(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = ModelInterpolator(_make_store(tmpdir, [
                _model_name("p", 4500, 3.0, "0.0", "01", 0.0, 0.0),
                _model_name("s", 4500, 1.5, "1.0", "01", 0.0, 0.0),
            ]))
            best, msg = lib.find_nearest_model(4500, 1.5, 0.0, "01")
            self.assertEqual((best["geom"], best["logg"]), ("s", 1.5))
            self.assertIn("spherical", msg)

    def test_turb_mismatch_does_not_absorb_large_feh_error(self) -> None:
        # A hard same-turb pre-filter would snap a metal-poor t05 request onto
        # the solar-metallicity t05 subset; the weak vmicro axis must prefer
        # the metal-poor t02 model at the requested stellar point instead.
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = ModelInterpolator(_make_store(tmpdir, [
                _model_name("p", 5000, 4.5, "0.0", "05", 0.0, 0.0),
                _model_name("p", 5000, 4.5, "0.0", "02", -3.0, 0.4),
            ]))
            best, msg = lib.find_nearest_model(5000, 4.5, -3.0, "05")
            self.assertEqual((best["feh"], best["turb"]), (-3.0, "02"))
            self.assertIn("substituted", msg)

    def test_matching_turb_wins_when_stellar_params_tie(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = ModelInterpolator(_make_store(tmpdir, [
                _model_name("p", 5000, 4.5, "0.0", "01", 0.0, 0.0),
                _model_name("p", 5000, 4.5, "0.0", "02", 0.0, 0.0),
            ]))
            best, _ = lib.find_nearest_model(5000, 4.5, 0.0, "02")
            self.assertEqual(best["turb"], "02")

    def test_tie_prefers_plane_parallel_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = ModelInterpolator(_make_store(tmpdir, [
                _model_name("s", 5000, 3.0, "1.0", "01", 0.0, 0.0),
                _model_name("p", 5000, 3.0, "0.0", "01", 0.0, 0.0),
            ]))
            best, _ = lib.find_nearest_model(5000, 3.0, 0.0, "01")
            self.assertEqual(best["geom"], "p")

    def test_spherical_mass_tie_prefers_solar_mass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = ModelInterpolator(_make_store(tmpdir, [
                _model_name("s", 4000, 1.0, "5.0", "02", 0.0, 0.0),
                _model_name("s", 4000, 1.0, "1.0", "02", 0.0, 0.0),
                _model_name("s", 4000, 1.0, "0.5", "02", 0.0, 0.0),
            ]))
            best, _ = lib.find_nearest_model(4000, 1.0, 0.0, "02")
            self.assertEqual(best["mass"], 1.0)

    def test_scan_is_cached_per_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_store(tmpdir, [
                _model_name("p", 5000, 4.0, "0.0", "01", 0.0, 0.0),
            ])
            first = ModelInterpolator(store).available_models
            # A file added after the first scan is not seen: the store is
            # treated as immutable for the lifetime of the process.
            with open(os.path.join(tmpdir, _model_name("p", 5250, 4.0, "0.0", "01", 0.0, 0.0)), "w"):
                pass
            second = ModelInterpolator(store).available_models
            self.assertIs(second, first)


if __name__ == "__main__":
    unittest.main()
