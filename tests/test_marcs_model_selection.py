"""MARCS atmosphere selection: exact-filename alpha coupling and the
nearest-neighbour snap (geometry coverage, turbulence weighting, tie-breaks).
"""
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.generate_grid import apply_bracket_prefilter
from scripts.run_turbospectrum import (
    ModelInterpolator,
    TurbospectrumConfig,
    _normalize_config_dict,
    get_model_filename,
    interpolated_model_has_nan,
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


def _cube(geom, mass, turb, teffs, loggs, fehs, skip=()):
    return [
        _model_name(geom, t, g, mass, turb, z, standard_composition_alpha(z))
        for t in teffs for g in loggs for z in fehs if (t, g, z) not in skip
    ]


class TestInterpolationBracket(unittest.TestCase):
    TEFFS = (5500, 5750, 6000, 6250, 6500)
    PP_LOGGS = (3.0, 3.5, 4.0, 4.5)
    SPH_LOGGS = (1.0, 1.5, 2.0, 2.5, 3.0)
    FEHS = (-0.5, -0.25, 0.0, 0.25)

    def _lib(self, tmpdir, names):
        return ModelInterpolator(_make_store(tmpdir, names))

    def test_interior_point_uses_adjacent_nodes_in_interpol_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = self._lib(tmpdir, _cube("p", "0.0", "02", self.TEFFS, self.PP_LOGGS, self.FEHS))
            b = lib.find_interpolation_bracket(5830, 4.2, -0.1, "02")
            self.assertEqual((b["geom"], b["widen"], b["node_path"]), ("p", 0, None))
            self.assertEqual((b["teff"], b["logg"], b["feh"]), ((5750, 6000), (4.0, 4.5), (-0.25, 0.0)))
            # Teff-major, then logg, then [Fe/H] (what interpol_modeles expects).
            names = [os.path.basename(p) for p in b["paths"]]
            self.assertEqual(names[0], _model_name("p", 5750, 4.0, "0.0", "02", -0.25, 0.1))
            self.assertEqual(names[1], _model_name("p", 5750, 4.0, "0.0", "02", 0.0, 0.0))
            self.assertEqual(names[2], _model_name("p", 5750, 4.5, "0.0", "02", -0.25, 0.1))
            self.assertEqual(names[4], _model_name("p", 6000, 4.0, "0.0", "02", -0.25, 0.1))

    def test_node_point_uses_the_file_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = self._lib(tmpdir, _cube("p", "0.0", "02", self.TEFFS, self.PP_LOGGS, self.FEHS))
            b = lib.find_interpolation_bracket(6000, 4.0, 0.0, "02")
            self.assertEqual(os.path.basename(b["node_path"]),
                             _model_name("p", 6000, 4.0, "0.0", "02", 0.0, 0.0))

    def test_geometry_follows_logg_and_ignores_other_masses_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            names = (_cube("p", "0.0", "02", self.TEFFS, self.PP_LOGGS, self.FEHS)
                     + _cube("s", "1.0", "02", self.TEFFS, self.SPH_LOGGS, self.FEHS)
                     + [_model_name("s", 5750, 2.0, "5.0", "02", 0.0, 0.0),
                        _model_name("s", 5750, 2.0, "1.0", "05", 0.0, 0.0)])
            lib = self._lib(tmpdir, names)
            giant = lib.find_interpolation_bracket(5900, 2.2, -0.1, "02")
            self.assertEqual(giant["geom"], "s")
            self.assertTrue(all("_m1.0_t02_" in os.path.basename(p) for p in giant["paths"]))
            # Exactly at the switch: plane-parallel.
            self.assertEqual(lib.find_interpolation_bracket(5900, 3.0, -0.1, "02")["geom"], "p")
            self.assertEqual(lib.find_interpolation_bracket(5900, 2.99, -0.1, "02")["geom"], "s")

    def test_missing_corner_widens_by_one_node_on_the_cheapest_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = self._lib(tmpdir, _cube("s", "1.0", "02", self.TEFFS, self.SPH_LOGGS, self.FEHS,
                                          skip={(6000, 1.5, 0.0)}))
            b = lib.find_interpolation_bracket(5900, 1.7, -0.1, "02")
            self.assertEqual(b["widen"], 1)
            # Stepping over the missing [Fe/H] = 0 node (0.5 dex) is cheaper than
            # over Teff 6000 (500 K = 2 scale units).
            self.assertEqual((b["teff"], b["logg"], b["feh"]), ((5750, 6000), (1.5, 2.0), (-0.25, 0.25)))
            self.assertTrue(all(os.path.exists(p) for p in b["paths"]))

    def test_no_complete_bracket_within_max_widen_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = self._lib(tmpdir, _cube("s", "1.0", "02", self.TEFFS, self.SPH_LOGGS, self.FEHS,
                                          skip={(6000, 1.5, 0.0)}))
            with self.assertRaises(ValueError):
                lib.find_interpolation_bracket(5900, 1.7, -0.1, "02", max_widen=0)

    def test_outside_node_range_raises_instead_of_extrapolating(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = self._lib(tmpdir, _cube("p", "0.0", "02", self.TEFFS, self.PP_LOGGS, self.FEHS))
            with self.assertRaises(ValueError):
                lib.find_interpolation_bracket(6600, 4.2, -0.1, "02")
            with self.assertRaises(ValueError):
                lib.find_interpolation_bracket(6000, 4.2, 0.4, "02")

    def test_config_block_reaches_turbospectrum_config(self) -> None:
        flat = _normalize_config_dict(
            {"paths": {}, "model_atmosphere": {"selection": "Interpolate", "turb_label": 2, "max_widen": 0}},
            default_project_root=".",
        )
        cfg = TurbospectrumConfig(**flat)
        self.assertEqual((cfg.atmosphere_selection, cfg.atmosphere_turb_label, cfg.atmosphere_max_widen),
                         ("interpolate", "02", 0))
        with self.assertRaises(ValueError):
            TurbospectrumConfig(project_root=".", atmosphere_selection="cubic")

    def test_grid_prefilter_drops_rows_without_a_bracket(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_store(tmpdir, _cube("p", "0.0", "02", self.TEFFS, self.PP_LOGGS, self.FEHS,
                                      skip={(6000, g, 0.25) for g in self.PP_LOGGS}))
            columns = {
                "teff": np.array([5830.0, 5900.0, 6700.0]),
                "logg": np.array([4.2, 4.2, 4.2]),
                "feh": np.array([-0.1, 0.1, -0.1]),
                "row": np.array([0, 1, 2]),
            }
            spec = {"atmosphere_path": tmpdir, "turb_label": "02", "max_widen": 0}
            kept = apply_bracket_prefilter(columns, {"bracket_prefilter": spec})
            # Row 1 needs the missing [Fe/H]=+0.25 corners at 6000 K (no widening
            # allowed); row 2 is above the Teff node range.
            self.assertEqual(kept["row"].tolist(), [0])
            spec["max_widen"] = 1
            kept = apply_bracket_prefilter(columns, {"bracket_prefilter": spec})
            self.assertEqual(kept["row"].tolist(), [0, 1])


class TestInterpolatedModelNaN(unittest.TestCase):
    def _model(self, tmpdir, rows):
        path = os.path.join(tmpdir, "m.interp.mod")
        with open(path, "w") as fh:
            fh.write(f"'sphINTERPOL'  {len(rows)}   5000.  2.50 0 0.00\n")
            fh.writelines(r + "\n" for r in rows)
            fh.write("/path/to/corner/model.mod\n")
        return path

    def test_detects_nan_depth_rows(self) -> None:
        # interpol_modeles output with the target exactly on a log g bracket bound.
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = self._model(tmpdir, [" -4.7994      NaN      NaN      NaN   2.0000    0.657155E+12  -5.0000"] * 3)
            self.assertTrue(interpolated_model_has_nan(bad))

    def test_clean_model_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            good = self._model(tmpdir, [" -4.7995  4742.51  -2.5029   0.7744   2.0000    0.656576E+12  -5.0000"] * 3)
            self.assertFalse(interpolated_model_has_nan(good))


if __name__ == "__main__":
    unittest.main()
