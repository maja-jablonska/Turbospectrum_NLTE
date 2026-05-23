import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.generate_grid import _resolve_ml_sampling
from scripts.nlte_ascii_departures import (
    build_nlte_ascii_selector_columns,
    ensure_nlte_info_paths_slash_terminated,
    materialize_nlte_info_with_departure_override,
    normalize_nlte_ascii_selector,
    parse_nlte_info_file,
    read_departure_file_abundance,
    resolve_absolute_abundance,
    select_departure_file,
)
from scripts.run_turbospectrum import (
    TurbospectrumConfig,
    _build_abundance_controls,
    _resolve_nlte_ascii_runtime_info,
    get_synthesis_output_stem_from_params,
)
from scripts.synthesize_spectra_from_zarr import _build_tasks
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
            self.assertAlmostEqual(abundance, 7.01, places=6)

    def test_normalize_selector_resolves_relative_dir_against_base_dir(self) -> None:
        # Mirrors the example configs: a "../../DATA/..." directory must resolve
        # relative to the config file's directory, not the process cwd.
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_dir = os.path.join(tmpdir, "configs", "pipeline")
            deps_dir = os.path.join(tmpdir, "DATA", "DEP", "nlte_departures_ascii")
            os.makedirs(cfg_dir)
            os.makedirs(deps_dir)
            rel = os.path.join("..", "..", "DATA", "DEP", "nlte_departures_ascii")

            selector = normalize_nlte_ascii_selector(
                directory=rel, species="Fe", base_dir=cfg_dir
            )

            self.assertEqual(selector.directory, os.path.abspath(deps_dir))

    def test_normalize_selector_base_dir_ignored_for_absolute_dir(self) -> None:
        # An absolute directory must be unaffected by base_dir.
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = normalize_nlte_ascii_selector(
                directory=tmpdir, species="Fe", base_dir="/nonexistent/base"
            )

            self.assertEqual(selector.directory, os.path.abspath(tmpdir))

    def test_read_departure_file_abundance_uses_file_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"000001_{self._MODEL_STEM}_abu+7.500.dat")
            with open(path, "w", encoding="utf-8") as handle:
                for _ in range(8):
                    handle.write("# parameter 1.0 1.0\n")
                handle.write("7.510\n")
                handle.write("2\n")
                handle.write("1\n")
                handle.write("-5.0\n")
                handle.write("-4.0\n")
                handle.write("1.0\n")
                handle.write("1.0\n")

            self.assertAlmostEqual(read_departure_file_abundance(path), 7.51, places=6)

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
            _, runtime_departure_dir, entries = parse_nlte_info_file(runtime_info)
            fe_entry = next(entry for entry in entries if entry.atomic_number == 26)
            staged_override = os.path.join(runtime_departure_dir, fe_entry.departure_file)

            self.assertNotEqual(fe_entry.departure_file, os.path.basename(override_fe))
            self.assertLess(len(fe_entry.departure_file), len(os.path.basename(override_fe)))
            self.assertIn("_abu+7.500", fe_entry.departure_file)
            self.assertIn(fe_entry.departure_file, contents)
            self.assertIn("'ascii'", contents)
            self.assertIn("'base_ca.dat'", contents)
            self.assertTrue(os.path.isfile(staged_override))

        self.assertTrue(True)

    def test_materialize_nlte_info_with_departure_override_repairs_missing_staged_file(self) -> None:
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
            runtime_root = os.path.join(tmpdir, "runtime")
            runtime_info = materialize_nlte_info_with_departure_override(
                base_info_path=nlte_info,
                selector=selector,
                departure_file_path=override_fe,
                output_root=runtime_root,
            )

            _, runtime_departure_dir, entries = parse_nlte_info_file(runtime_info)
            fe_entry = next(entry for entry in entries if entry.atomic_number == 26)
            staged_override = os.path.join(runtime_departure_dir, fe_entry.departure_file)
            os.unlink(staged_override)
            self.assertFalse(os.path.exists(staged_override))

            repaired_info = materialize_nlte_info_with_departure_override(
                base_info_path=nlte_info,
                selector=selector,
                departure_file_path=override_fe,
                output_root=runtime_root,
            )

            self.assertEqual(repaired_info, runtime_info)
            self.assertTrue(os.path.isfile(staged_override))

    def test_runtime_info_uses_departure_header_abundance_for_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "DATA")
            model_dir = os.path.join(data_dir, "ATOM")
            dep_dir = os.path.join(data_dir, "DEP")
            ascii_dir = os.path.join(tmpdir, "ascii")
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(dep_dir, exist_ok=True)
            os.makedirs(ascii_dir, exist_ok=True)

            base_fe = os.path.join(dep_dir, "base_fe.bin")
            with open(base_fe, "w", encoding="utf-8") as handle:
                handle.write("dummy\n")

            override_fe = os.path.join(ascii_dir, f"000001_{self._MODEL_STEM}_abu+7.500.dat")
            with open(override_fe, "w", encoding="utf-8") as handle:
                for _ in range(8):
                    handle.write("# parameter 1.0 1.0\n")
                handle.write("7.510\n")
                handle.write("2\n")
                handle.write("1\n")
                handle.write("-5.0\n")
                handle.write("-4.0\n")
                handle.write("1.0\n")
                handle.write("1.0\n")

            nlte_info = os.path.join(data_dir, "SPECIES_LTE_NLTE.dat")
            with open(nlte_info, "w", encoding="utf-8") as handle:
                handle.write("# path for model atom files     ! don't change this line !\n")
                handle.write("/Users/mjablons/Documents/Turbospectrum_NLTE/DATA/ATOMS\n")
                handle.write("#\n")
                handle.write("# path for departure files      ! don't change this line !\n")
                handle.write("/Users/mjablons/Documents/Turbospectrum_NLTE/DATA/DEP\n")
                handle.write("#\n")
                handle.write("# atomic (N)LTE setup\n")
                handle.write("26 'Fe' 'nlte' 'atom.fe607a' 'base_fe.bin' 'binary'\n")

            config = TurbospectrumConfig(
                project_root=tmpdir,
                nlte=True,
                nlte_info_file=nlte_info,
                tmp_dir=os.path.join(tmpdir, "runtime"),
            )
            model_path = os.path.join(tmpdir, f"{self._MODEL_STEM}.mod")
            with open(model_path, "w", encoding="utf-8") as handle:
                handle.write("MARCS\n")

            info = _resolve_nlte_ascii_runtime_info(
                params={
                    "feh": 0.0,
                    "nlte_ascii_departure_dir": ascii_dir,
                    "nlte_ascii_departure_species": "Fe",
                    "nlte_ascii_abundance_column": "feh",
                    "nlte_ascii_abundance_scale": "relative",
                    "nlte_ascii_solar_abundance": 7.50,
                },
                config=config,
                model_input_path=model_path,
            )

            self.assertIsNotNone(info)
            assert info is not None
            self.assertAlmostEqual(info["matched_abundance"], 7.50, places=6)
            self.assertAlmostEqual(info["departure_file_abundance"], 7.51, places=6)
            runtime_model_path, _, _ = parse_nlte_info_file(info["nlte_info_file"])
            self.assertEqual(runtime_model_path, model_dir + os.sep)

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

    def test_build_tasks_preserves_nlte_ascii_control_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = normalize_nlte_ascii_selector(
                directory=tmpdir,
                species="Fe",
                solar_abundance=7.50,
            )
            column_data = {
                "teff": np.asarray([5000], dtype=np.int64),
                "logg": np.asarray([4.0], dtype=np.float64),
                "feh": np.asarray([-0.5], dtype=np.float64),
                "lam_min": np.asarray([5000.0], dtype=np.float64),
                "lam_max": np.asarray([5020.0], dtype=np.float64),
                "lam_step": np.asarray([0.01], dtype=np.float64),
                "turbvel": np.asarray(["01"], dtype=object),
                **build_nlte_ascii_selector_columns(1, selector),
            }

            tasks = _build_tasks(
                1,
                column_data,
                SimpleNamespace(output_mode="Flux", nlte=True),
            )
            row_values = tasks[0][1]

            self.assertEqual(row_values["nlte_ascii_departure_dir"], tmpdir)
            self.assertEqual(row_values["nlte_ascii_departure_species"], "Fe")
            self.assertEqual(row_values["nlte_ascii_abundance_column"], "auto")
            self.assertEqual(row_values["nlte_ascii_abundance_scale"], "relative")
            self.assertEqual(row_values["nlte_ascii_match"], "nearest")
            self.assertAlmostEqual(float(row_values["nlte_ascii_solar_abundance"]), 7.5, places=6)

    def test_ensure_nlte_info_paths_slash_terminated_rewrites_unslashed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "SPECIES_LTE_NLTE.dat")
            with open(base_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "# comment\n"
                    "# path for model atom files     ! don't change this line !\n"
                    "/abs/DATA/ATOMS\n"
                    "#\n"
                    "# path for departure files      ! don't change this line !\n"
                    "/abs/DATA/DEP\n"
                    "#\n"
                    "26\t'Fe'\t'nlte'\t'atom.fe607a'\t'NLTEgrid_Fe_Sun.ascii'\t'ascii'\n"
                )

            output_root = os.path.join(tmpdir, "tmp")
            result = ensure_nlte_info_paths_slash_terminated(
                base_info_path=base_path,
                output_root=output_root,
            )

            self.assertNotEqual(result, base_path)
            with open(result, "r", encoding="utf-8") as handle:
                rewritten = handle.readlines()
            model_idx = next(
                i for i, line in enumerate(rewritten)
                if line.strip().startswith("# path for model atom files")
            )
            depart_idx = next(
                i for i, line in enumerate(rewritten)
                if line.strip().startswith("# path for departure files")
            )
            self.assertEqual(rewritten[model_idx + 1].rstrip(), "/abs/DATA/ATOMS/")
            self.assertEqual(rewritten[depart_idx + 1].rstrip(), "/abs/DATA/DEP/")
            # Non-path content must survive.
            self.assertTrue(any("atom.fe607a" in line for line in rewritten))

            # Idempotent: second call reuses the cached rewrite.
            again = ensure_nlte_info_paths_slash_terminated(
                base_info_path=base_path,
                output_root=output_root,
            )
            self.assertEqual(again, result)

    def test_ensure_nlte_info_paths_slash_terminated_is_noop_when_already_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "SPECIES_LTE_NLTE.dat")
            with open(base_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "# path for model atom files     ! don't change this line !\n"
                    "/abs/DATA/ATOMS/\n"
                    "#\n"
                    "# path for departure files      ! don't change this line !\n"
                    "/abs/DATA/DEP/\n"
                    "#\n"
                )

            result = ensure_nlte_info_paths_slash_terminated(
                base_info_path=base_path,
                output_root=os.path.join(tmpdir, "tmp"),
            )
            self.assertEqual(result, os.path.abspath(base_path))

    def test_build_abundance_controls_preserves_exact_numeric_strings(self) -> None:
        alpha, r_proc, s_proc, block = _build_abundance_controls(
            {
                "a": "+0.00",
                "fe": "+7.000",
            }
        )

        self.assertEqual(alpha, "+0.00")
        self.assertEqual(r_proc, "+0.00")
        self.assertEqual(s_proc, "+0.00")
        self.assertIn(" 26  +7.000", block)


if __name__ == "__main__":
    unittest.main()
