"""Tests for the NLTE configuration ergonomics in scripts/nlte_config.py:

* the single switch (calculation_mode -> nlte.enabled),
* the top-level/grid block-location resolver, and
* the pre-synthesis preflight checks.
"""
import os
import tempfile
import unittest

from scripts.nlte_config import (
    apply_single_nlte_switch,
    is_nlte_calculation_mode,
    nlte_is_enabled,
    preflight_nlte,
    preflight_nlte_from_config,
    resolve_nlte_ascii_cfg,
)


def _write_info_file(directory: str, *, model_dir: str, dep_dir: str, body: str) -> str:
    """Write a minimal SPECIES_LTE_NLTE.dat in the format parse_nlte_info_file expects."""
    path = os.path.join(directory, "SPECIES_LTE_NLTE.dat")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# path for model atom files     ! don't change this line !\n")
        handle.write(f"{model_dir}\n")
        handle.write("#\n")
        handle.write("# path for departure files      ! don't change this line !\n")
        handle.write(f"{dep_dir}\n")
        handle.write("#\n")
        handle.write(body)
    return path


class SingleSwitchTests(unittest.TestCase):
    def test_calculation_mode_nlte_enables(self) -> None:
        cfg = {"nlte": {"enabled": False}}
        apply_single_nlte_switch(cfg, "NLTE")
        self.assertTrue(cfg["nlte"]["enabled"])

    def test_calculation_mode_is_case_insensitive_and_creates_block(self) -> None:
        cfg = {}
        apply_single_nlte_switch(cfg, "nlte")
        self.assertTrue(cfg["nlte"]["enabled"])

    def test_lte_leaves_explicit_enabled_but_warns(self) -> None:
        warnings = []
        cfg = {"nlte": {"enabled": True}}
        apply_single_nlte_switch(cfg, "LTE", warn=warnings.append)
        self.assertTrue(cfg["nlte"]["enabled"])
        self.assertEqual(len(warnings), 1)

    def test_lte_default_stays_off_without_warning(self) -> None:
        warnings = []
        cfg = {"nlte": {"enabled": False}}
        apply_single_nlte_switch(cfg, "LTE", warn=warnings.append)
        self.assertFalse(cfg["nlte"]["enabled"])
        self.assertEqual(warnings, [])

    def test_is_nlte_calculation_mode(self) -> None:
        self.assertTrue(is_nlte_calculation_mode("NLTE"))
        self.assertTrue(is_nlte_calculation_mode("  nlte "))
        self.assertFalse(is_nlte_calculation_mode("LTE"))
        self.assertFalse(is_nlte_calculation_mode(None))

    def test_nlte_is_enabled_considers_both_signals(self) -> None:
        self.assertTrue(nlte_is_enabled({"nlte": {"enabled": False}}, "NLTE"))
        self.assertTrue(nlte_is_enabled({"nlte": {"enabled": True}}, "LTE"))
        self.assertFalse(nlte_is_enabled({"nlte": {"enabled": False}}, "LTE"))


class BlockLocationTests(unittest.TestCase):
    def test_top_level_wins_over_grid(self) -> None:
        self.assertEqual(resolve_nlte_ascii_cfg({"directory": "a"}, {"directory": "b"}), {"directory": "a"})

    def test_falls_back_to_grid(self) -> None:
        self.assertEqual(resolve_nlte_ascii_cfg(None, {"directory": "b"}), {"directory": "b"})

    def test_empty_when_absent(self) -> None:
        self.assertEqual(resolve_nlte_ascii_cfg(None, {}), {})


class PreflightTests(unittest.TestCase):
    def test_disabled_returns_no_problems(self) -> None:
        self.assertEqual(preflight_nlte(project_root="/nope", enabled=False), [])

    def test_ascii_missing_directory(self) -> None:
        problems = preflight_nlte(
            project_root="/nope",
            enabled=True,
            nlte_ascii_cfg={"directory": "does_not_exist"},
            nlte_ascii_base_dir="/tmp",
        )
        self.assertTrue(any("ASCII departure directory not found" in p for p in problems))
        self.assertTrue(any("download_data.sh" in p for p in problems))

    def test_ascii_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            problems = preflight_nlte(
                project_root="/nope", enabled=True, nlte_ascii_cfg={"directory": tmp}
            )
        self.assertTrue(any("empty" in p for p in problems))

    def test_ascii_present_directory_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "dep_abu+0.000.dat"), "w").close()
            problems = preflight_nlte(
                project_root="/nope", enabled=True, nlte_ascii_cfg={"directory": tmp}
            )
        self.assertEqual(problems, [])

    def test_binary_missing_info_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            problems = preflight_nlte(project_root=tmp, enabled=True)
        self.assertTrue(any("NLTE info file not found" in p for p in problems))

    def test_binary_complete_setup_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "model_atoms")
            dep_dir = os.path.join(tmp, "departure_grids")
            os.makedirs(model_dir)
            os.makedirs(dep_dir)
            open(os.path.join(model_dir, "atom.fe607a"), "w").close()
            open(os.path.join(dep_dir, "NLTEgrid_Fe.ascii"), "w").close()
            info = _write_info_file(
                tmp,
                model_dir=model_dir,
                dep_dir=dep_dir,
                body="26\t'Fe'\t'nlte'\t'atom.fe607a'\t'NLTEgrid_Fe.ascii'\t'ascii'\n",
            )
            problems = preflight_nlte(project_root=tmp, enabled=True, nlte_info_file=info)
        self.assertEqual(problems, [])

    def test_binary_missing_departure_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "model_atoms")
            dep_dir = os.path.join(tmp, "departure_grids")
            os.makedirs(model_dir)
            os.makedirs(dep_dir)
            open(os.path.join(model_dir, "atom.fe607a"), "w").close()
            # departure file intentionally absent
            info = _write_info_file(
                tmp,
                model_dir=model_dir,
                dep_dir=dep_dir,
                body="26\t'Fe'\t'nlte'\t'atom.fe607a'\t'NLTEgrid_Fe.ascii'\t'ascii'\n",
            )
            problems = preflight_nlte(project_root=tmp, enabled=True, nlte_info_file=info)
        self.assertTrue(any("Missing departure file for Fe" in p for p in problems))

    def test_binary_all_lte_entries_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "model_atoms")
            dep_dir = os.path.join(tmp, "departure_grids")
            os.makedirs(model_dir)
            os.makedirs(dep_dir)
            info = _write_info_file(
                tmp,
                model_dir=model_dir,
                dep_dir=dep_dir,
                body="26\t'Fe'\t'lte'\t'atom.fe607a'\t'NLTEgrid_Fe.ascii'\t'ascii'\n",
            )
            problems = preflight_nlte(project_root=tmp, enabled=True, nlte_info_file=info)
        self.assertTrue(any("No species are set to 'nlte'" in p for p in problems))


class PreflightFromConfigTests(unittest.TestCase):
    def test_lte_config_not_enabled(self) -> None:
        cfg = {"grid": {"synthesis": {"calculation_mode": "LTE"}}}
        result = preflight_nlte_from_config(cfg, config_dir="/tmp", project_root="/nope")
        self.assertFalse(result["enabled"])
        self.assertEqual(result["problems"], [])

    def test_single_switch_enables_binary_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "model_atoms")
            dep_dir = os.path.join(tmp, "departure_grids")
            os.makedirs(model_dir)
            os.makedirs(dep_dir)
            open(os.path.join(model_dir, "atom.fe607a"), "w").close()
            open(os.path.join(dep_dir, "NLTEgrid_Fe.ascii"), "w").close()
            info = _write_info_file(
                tmp,
                model_dir=model_dir,
                dep_dir=dep_dir,
                body="26\t'Fe'\t'nlte'\t'atom.fe607a'\t'NLTEgrid_Fe.ascii'\t'ascii'\n",
            )
            cfg = {
                "grid": {"synthesis": {"calculation_mode": "NLTE"}},
                "turbospectrum": {"overrides": {"nlte": {"nlte_info_file": info}}},
            }
            result = preflight_nlte_from_config(cfg, config_dir=tmp, project_root=tmp)
        self.assertTrue(result["enabled"])
        self.assertEqual(result["workflow"], "binary")
        self.assertEqual(result["problems"], [])

    def test_ascii_block_at_top_level_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "dep_abu+0.000.dat"), "w").close()
            cfg = {
                "grid": {"synthesis": {"calculation_mode": "NLTE"}},
                "nlte_ascii_departures": {"directory": tmp},
            }
            result = preflight_nlte_from_config(cfg, config_dir="/tmp", project_root="/nope")
        self.assertEqual(result["workflow"], "ascii")
        self.assertEqual(result["problems"], [])


class GeneratorTests(unittest.TestCase):
    def test_build_config_binary_single_switch(self) -> None:
        from argparse import Namespace

        from scripts.init_nlte_config import _build_config

        args = Namespace(
            workflow="binary", ascii_dir=None, species="Fe", run_root="/tmp/run",
            teff="5000,5500", logg="4.0,4.5", feh="-0.5,0.0",
            lam_min=5000.0, lam_max=5200.0, lam_step=0.01, compiler="gf",
        )
        cfg = _build_config(args)
        self.assertEqual(cfg["grid"]["synthesis"]["calculation_mode"], "NLTE")
        self.assertNotIn("nlte_ascii_departures", cfg)  # binary -> no ASCII block

    def test_build_config_ascii_adds_block(self) -> None:
        from argparse import Namespace

        from scripts.init_nlte_config import _build_config

        args = Namespace(
            workflow="ascii", ascii_dir="some/dep/dir", species="Fe", run_root="/tmp/run",
            teff="5000", logg="4.0", feh="0.0",
            lam_min=5000.0, lam_max=5050.0, lam_step=0.01, compiler="gf",
        )
        cfg = _build_config(args)
        self.assertEqual(cfg["nlte_ascii_departures"]["species"], "Fe")
        self.assertTrue(os.path.isabs(cfg["nlte_ascii_departures"]["directory"]))

    def test_patch_species_map_repoints_and_backs_up(self) -> None:
        from scripts.init_nlte_config import _patch_species_map

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_info_file(
                tmp,
                model_dir="/g/data/y89/OLD/ATOMS",
                dep_dir="/g/data/y89/OLD/DEP",
                body="26\t'Fe'\t'nlte'\t'atom.fe607a'\t'NLTEgrid_Fe.ascii'\t'ascii'\n",
            )
            changes, backup = _patch_species_map(path, "/new/model_atoms", "/new/departure_grids")

            self.assertTrue(os.path.isfile(backup))
            self.assertEqual(len(changes), 2)
            from scripts.nlte_ascii_departures import parse_nlte_info_file

            pm, pd, _ = parse_nlte_info_file(path)
            self.assertTrue(pm.rstrip("/").endswith("model_atoms"))
            self.assertTrue(pd.rstrip("/").endswith("departure_grids"))
            # backup keeps the originals
            pm0, pd0, _ = parse_nlte_info_file(backup)
            self.assertIn("OLD", pm0)


class SetupNlteTests(unittest.TestCase):
    def test_element_from_atom_filename(self) -> None:
        from scripts.setup_nlte import _element_from_atom

        self.assertEqual(_element_from_atom("atom.fe607a"), "Fe")
        self.assertEqual(_element_from_atom("atom.h20"), "H")
        self.assertEqual(_element_from_atom("atom.ca105b"), "Ca")
        self.assertEqual(_element_from_atom("atom.co607"), "Co")  # 2-letter wins over C
        self.assertEqual(_element_from_atom("atom.o97"), "O")
        self.assertIsNone(_element_from_atom("atom.zz9"))

    def test_element_from_grid_filename(self) -> None:
        from scripts.setup_nlte import _element_from_grid

        self.assertEqual(_element_from_grid("NLTEgrid_Fe_Sun.ascii"), "Fe")
        self.assertEqual(_element_from_grid("NLTEgrid_Mg_Sun.bin"), "Mg")
        self.assertIsNone(_element_from_grid("NLTEgrid_Sun.ascii"))

    def test_scan_pairs_atoms_and_grids_ignoring_auxdata(self) -> None:
        from scripts.setup_nlte import _scan

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "model_atoms")
            dep_dir = os.path.join(tmp, "departure_grids")
            os.makedirs(model_dir)
            os.makedirs(dep_dir)
            for f in ("atom.fe607a", "atom.h20"):
                open(os.path.join(model_dir, f), "w").close()
            for f in ("NLTEgrid_Fe_Sun.ascii", "auxData_Fe.dat"):
                open(os.path.join(dep_dir, f), "w").close()
            atoms, deps = _scan(model_dir, dep_dir)
        self.assertEqual(atoms, {"Fe": "atom.fe607a", "H": "atom.h20"})
        self.assertEqual(deps, {"Fe": "NLTEgrid_Fe_Sun.ascii"})  # auxData ignored

    def test_build_species_map_flags_and_format(self) -> None:
        from scripts.setup_nlte import _build_species_map

        atoms = {"Fe": "atom.fe607a", "Mg": "atom.mg85", "H": "atom.h20"}
        deps = {"Fe": "NLTEgrid_Fe_Sun.ascii", "Mg": "NLTEgrid_Mg_Sun.bin"}

        # default: every species with a departure grid -> nlte
        text, summary = _build_species_map("/m", "/d", atoms, deps, None)
        flags = {el: flag for _, el, flag in summary}
        self.assertEqual(flags, {"H": "lte", "Mg": "nlte", "Fe": "nlte"})
        self.assertIn("'NLTEgrid_Mg_Sun.bin'\t'binary'", text)   # .bin -> binary
        self.assertIn("'NLTEgrid_Fe_Sun.ascii'\t'ascii'", text)  # .ascii -> ascii
        # LTE rows are commented out (LTE is the default anyway)
        self.assertIn("#1\t'H'", text)

        # restricting NLTE to Fe leaves Mg as lte
        _, summary2 = _build_species_map("/m", "/d", atoms, deps, ["Fe"])
        flags2 = {el: flag for _, el, flag in summary2}
        self.assertEqual(flags2["Mg"], "lte")
        self.assertEqual(flags2["Fe"], "nlte")

    def test_pair_grids_with_aux(self) -> None:
        from scripts.setup_nlte import _pair_grids_with_aux

        with tempfile.TemporaryDirectory() as tmp:
            for f in (
                "NLTEgrid_Fe_Sun.bin", "auxData_Fe_Sun.dat",
                "NLTEgrid_Mg_Sun.bin", "auxData_Mg_Sun.dat",
                "NLTEgrid_Ca_Sun.ascii",  # ascii -> skipped (not a binary grid)
                "NLTEgrid_Ni_Sun.bin",    # no aux -> unpaired
            ):
                open(os.path.join(tmp, f), "w").close()
            pairs, unpaired = _pair_grids_with_aux(tmp)
        self.assertEqual(
            sorted(pairs),
            [("NLTEgrid_Fe_Sun.bin", "auxData_Fe_Sun.dat"),
             ("NLTEgrid_Mg_Sun.bin", "auxData_Mg_Sun.dat")],
        )
        self.assertEqual(unpaired, ["NLTEgrid_Ni_Sun.bin"])


class AsciiExportEndToEndTests(unittest.TestCase):
    """Fabricate a minimal binary grid + aux and run the real export chain."""

    @staticmethod
    def _fake_grid(path: str, *, model_id: str, ndep: int, nk: int) -> int:
        import struct

        import numpy as np

        header = b" " * 1000
        ident = model_id.ljust(500).encode("utf-8")  # records start with a 500-byte id
        meta = struct.pack("<ii", ndep, nk)
        tau = np.linspace(0.1, 10.0, ndep).astype("<f8").tobytes()
        dep = (np.arange(nk * ndep, dtype="<f8").reshape(nk, ndep) + 1.0).tobytes()
        with open(path, "wb") as handle:
            handle.write(header + ident + meta + tau + dep)
        return 1001  # 1-based pointer to the first record (after the 1000-byte header)

    def test_export_produces_matchable_ascii_file(self) -> None:
        from scripts.nlte_ascii_departures import read_departure_file_abundance
        from scripts.setup_nlte import _ascii_export

        with tempfile.TemporaryDirectory() as tmp:
            dep_dir = os.path.join(tmp, "departure_grids")
            os.makedirs(dep_dir)
            pointer = self._fake_grid(
                os.path.join(dep_dir, "NLTEgrid_Fe_Sun.bin"),
                model_id="testmodel", ndep=3, nk=2,
            )
            with open(os.path.join(dep_dir, "auxData_Fe_Sun.dat"), "w", encoding="utf-8") as handle:
                handle.write("# id Teff logg [M/H] alpha mass vturb abundance pointer\n")
                handle.write(f"'testmodel' 5000 4.0 0.0 0.0 1.0 1.0 7.50 {pointer}\n")

            out_dir = os.path.join(tmp, "ascii")
            rc = _ascii_export(dep_dir, out_dir, dry_run=False)
            self.assertEqual(rc, 0)

            fe_dir = os.path.join(out_dir, "Fe")  # per-species subdir
            files = os.listdir(fe_dir)
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].endswith("_abu+7.500.dat"), files[0])
            # the abundance the ASCII matcher will read back matches what we put in
            self.assertAlmostEqual(read_departure_file_abundance(os.path.join(fe_dir, files[0])), 7.5, places=3)


if __name__ == "__main__":
    unittest.main()
