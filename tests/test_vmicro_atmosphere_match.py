"""End-to-end verification of the vmicro -> nearest t_value atmosphere match.

These tests cover every stage of the flow:

- `_nearest_turb_label` and `_scan_available_turb_labels` helpers.
- `_resolve_synthesis_request` rebinds `model_turb_str` to the closest
  available atmosphere label when given a numeric vmicro.
- `get_synthesis_output_stem_from_params` uses the resolved label and
  still emits the `xi{vmicro}` extra so both values stay in the filename.
- `_build_task_batches` (shard entry point) does not collide on filenames
  for Intensity grids that vary mu, and resolves t_value from atmospheres.
- `_merge_resolved_turbulence_columns` overlays resolved per-row values
  on top of the grid's metadata columns, falling back cleanly when the
  worker produced no resolution (e.g. early failure).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for path in (REPO_ROOT, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.run_turbospectrum import (  # noqa: E402
    TurbospectrumConfig,
    _nearest_turb_label,
    _resolve_synthesis_request,
    _scan_available_turb_labels,
    get_synthesis_output_stem_from_params,
)
from scripts.run_turbospectrum_shard import (  # noqa: E402
    _build_task_batches,
    _merge_resolved_turbulence_columns,
)


def _touch_atmosphere_files(directory: str, labels) -> None:
    """Create minimal `.mod` filenames that the scan regex will recognize."""
    teff_logg_feh = [
        (5000, "+4.0", "+0.00"),
        (5250, "+4.5", "-0.50"),
    ]
    for label in labels:
        for teff, logg, feh in teff_logg_feh:
            name = (
                f"p{teff}_g{logg}_m0.0_t{label}_st_z{feh}_a+0.00_"
                f"c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod"
            )
            open(os.path.join(directory, name), "w", encoding="utf-8").close()


class NearestTurbLabelTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertEqual(_nearest_turb_label(2.0, ("01", "02", "05")), "02")

    def test_between_labels_rounds_to_closer(self) -> None:
        self.assertEqual(_nearest_turb_label(2.5, ("01", "02", "05")), "02")
        self.assertEqual(_nearest_turb_label(3.6, ("01", "02", "05")), "05")

    def test_ties_break_to_smaller_label(self) -> None:
        # 1.5 is equidistant from 01 and 02; tiebreaker picks 01.
        self.assertEqual(_nearest_turb_label(1.5, ("01", "02", "05")), "01")

    def test_below_minimum_returns_smallest(self) -> None:
        self.assertEqual(_nearest_turb_label(0.1, ("02", "03", "05")), "02")

    def test_above_maximum_returns_largest(self) -> None:
        self.assertEqual(_nearest_turb_label(25.0, ("01", "02", "05")), "05")

    def test_empty_labels_returns_none(self) -> None:
        self.assertIsNone(_nearest_turb_label(2.0, ()))

    def test_non_numeric_vmicro_returns_none(self) -> None:
        self.assertIsNone(_nearest_turb_label("abc", ("01", "02")))
        self.assertIsNone(_nearest_turb_label(None, ("01", "02")))


class ScanAvailableTurbLabelsTests(unittest.TestCase):
    def test_nonexistent_path_returns_empty(self) -> None:
        self.assertEqual(_scan_available_turb_labels("/does/not/exist"), ())

    def test_empty_path_returns_empty(self) -> None:
        self.assertEqual(_scan_available_turb_labels(""), ())

    def test_finds_unique_labels_in_mod_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch_atmosphere_files(tmp, ("01", "02", "05"))
            # A stray non-.mod file must not contribute a label.
            open(os.path.join(tmp, "README.txt"), "w").close()
            labels = _scan_available_turb_labels(tmp)
            self.assertEqual(labels, ("01", "02", "05"))

    def test_labels_sorted_numerically_not_lexically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch_atmosphere_files(tmp, ("10", "02", "05"))
            labels = _scan_available_turb_labels(tmp)
            self.assertEqual(labels, ("02", "05", "10"))


class ResolveSynthesisRequestTests(unittest.TestCase):
    _BASE_ROW = {
        "teff": 5000,
        "logg": 4.5,
        "feh": -0.5,
        "a": "+0.00",
    }

    def test_no_labels_leaves_model_turb_str_from_params(self) -> None:
        row = {**self._BASE_ROW, "t_value": "01", "turbvel": 2.5}
        request = _resolve_synthesis_request(row)
        self.assertEqual(request["model_turb_str"], "01")
        self.assertEqual(request["synthesis_turb_raw"], 2.5)

    def test_regular_grid_discrete_label_unchanged(self) -> None:
        row = {**self._BASE_ROW, "t_value": "02", "turbvel": "02"}
        request = _resolve_synthesis_request(
            row, atmosphere_turb_labels=("01", "02", "05")
        )
        self.assertEqual(request["model_turb_str"], "02")
        self.assertEqual(request["synthesis_turb_raw"], "02")

    def test_ml_mismatch_rebinds_model_turb_str(self) -> None:
        row = {**self._BASE_ROW, "t_value": "01", "turbvel": 2.5}
        request = _resolve_synthesis_request(
            row, atmosphere_turb_labels=("01", "02", "05")
        )
        self.assertEqual(request["model_turb_str"], "02")
        # Synthesis vmicro must be preserved exactly — it flows to babsma.
        self.assertEqual(request["synthesis_turb_raw"], 2.5)

    def test_missing_turbulence_uses_default_fallback(self) -> None:
        row = dict(self._BASE_ROW)
        request = _resolve_synthesis_request(row)
        self.assertEqual(request["model_turb_str"], "01")

    def test_atmosphere_rebind_ignores_non_numeric_synthesis_value(self) -> None:
        # synthesis_turb_raw ends up as model_turb_str ("02") — coerces to 2.0.
        row = {**self._BASE_ROW, "t_value": "02"}
        request = _resolve_synthesis_request(
            row, atmosphere_turb_labels=("01", "02", "05")
        )
        self.assertEqual(request["model_turb_str"], "02")


class SynthesisOutputStemTests(unittest.TestCase):
    _BASE_ROW = {
        "teff": 5000,
        "logg": 4.5,
        "feh": -0.5,
        "lam_min": 5000.0,
        "lam_max": 5020.0,
        "lam_step": 0.01,
        "output_mode": "Flux",
        "calculation_mode": "LTE",
        "a": "+0.00",
        "c": "+0.00",
        "n": "+0.00",
        "o": "+0.00",
        "r": "+0.00",
        "s": "+0.00",
    }

    def test_stem_uses_resolved_t_value(self) -> None:
        row = {**self._BASE_ROW, "t_value": "01", "turbvel": 2.5}
        stem = get_synthesis_output_stem_from_params(
            row, atmosphere_turb_labels=("01", "02", "05")
        )
        # Atmosphere label rebound to "02"; synthesis vmicro shows up as xi2.5.
        self.assertIn("_t02_", stem)
        self.assertIn("_xi2.5_", stem)

    def test_stem_without_labels_reflects_row_as_given(self) -> None:
        row = {**self._BASE_ROW, "t_value": "01", "turbvel": 2.5}
        stem = get_synthesis_output_stem_from_params(row)
        self.assertIn("_t01_", stem)

    def test_mu_appears_in_intensity_stem(self) -> None:
        row = {
            **self._BASE_ROW,
            "t_value": "02",
            "turbvel": "02",
            "output_mode": "Intensity",
            "calculation_mode": "NLTE",
            "mode": "1D",
            "mu": 0.3,
        }
        stem = get_synthesis_output_stem_from_params(row)
        self.assertIn("_mu0.3", stem)

    def test_intensity_rows_differing_only_in_mu_produce_unique_stems(self) -> None:
        row = {
            **self._BASE_ROW,
            "t_value": "02",
            "turbvel": "02",
            "output_mode": "Intensity",
            "calculation_mode": "NLTE",
            "mode": "1D",
        }
        stems = {
            get_synthesis_output_stem_from_params({**row, "mu": mu})
            for mu in (0.1, 0.2, 0.5, 1.0)
        }
        self.assertEqual(len(stems), 4)


class BuildTaskBatchesTests(unittest.TestCase):
    def _config(self, tmpdir: str, atmosphere_path: str = "") -> TurbospectrumConfig:
        return TurbospectrumConfig(
            project_root=tmpdir,
            output_mode="Flux",
            nlte=False,
            model_atmosphere_path=atmosphere_path,
        )

    def _intensity_columns(self, *, mu_values):
        n = len(mu_values)
        return {
            "teff": np.full(n, 5000, dtype=np.int64),
            "logg": np.full(n, 4.5, dtype=np.float64),
            "feh": np.full(n, -0.5, dtype=np.float64),
            "lam_min": np.full(n, 5000.0, dtype=np.float64),
            "lam_max": np.full(n, 5020.0, dtype=np.float64),
            "lam_step": np.full(n, 0.01, dtype=np.float64),
            "turbvel": np.asarray(["02"] * n, dtype=object),
            "t_value": np.asarray(["02"] * n, dtype=object),
            "output_mode": np.asarray(["Intensity"] * n, dtype=object),
            "calculation_mode": np.asarray(["NLTE"] * n, dtype=object),
            "mode": np.asarray(["1D"] * n, dtype=object),
            "a": np.asarray(["+0.00"] * n, dtype=object),
            "c": np.asarray(["+0.00"] * n, dtype=object),
            "n": np.asarray(["+0.00"] * n, dtype=object),
            "o": np.asarray(["+0.00"] * n, dtype=object),
            "r": np.asarray(["+0.00"] * n, dtype=object),
            "s": np.asarray(["+0.00"] * n, dtype=object),
            "mu": np.asarray(mu_values, dtype=np.float64),
        }

    def test_intensity_with_mu_axis_does_not_collide(self) -> None:
        """Regression test for the 'Grid rows collapse to duplicate filenames'
        crash that hit regular grids with Intensity + mu_range.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            columns = self._intensity_columns(mu_values=[0.1, 0.3, 0.5, 1.0])
            indices = np.arange(len(columns["mu"]), dtype=np.int64)
            # Intentionally no atmosphere dir — we should still disambiguate by mu.
            tasks, _ = _build_task_batches(
                indices,
                columns,
                batch_size=16,
                config=self._config(tmpdir),
            )
            # All rows should make it into the batch; mu flows through row_values.
            flat_rows = [row for batch in tasks for (_gi, row) in batch]
            self.assertEqual(len(flat_rows), 4)
            self.assertEqual(
                sorted(float(r["mu"]) for r in flat_rows),
                [0.1, 0.3, 0.5, 1.0],
            )

    def test_resolved_t_value_used_in_dup_detection(self) -> None:
        """Two ML rows with different sampled t_value but the same turbvel
        should collapse to the same atmosphere once rebound — the duplicate
        guard must then fire.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            atmospheres = os.path.join(tmpdir, "atmospheres")
            os.makedirs(atmospheres)
            _touch_atmosphere_files(atmospheres, ("01", "02", "05"))

            columns = {
                "teff": np.asarray([5000, 5000], dtype=np.int64),
                "logg": np.asarray([4.5, 4.5], dtype=np.float64),
                "feh": np.asarray([-0.5, -0.5], dtype=np.float64),
                "lam_min": np.asarray([5000.0, 5000.0], dtype=np.float64),
                "lam_max": np.asarray([5020.0, 5020.0], dtype=np.float64),
                "lam_step": np.asarray([0.01, 0.01], dtype=np.float64),
                "turbvel": np.asarray([2.5, 2.5], dtype=np.float64),
                # Two different nominal t_values — both resolve to "02".
                "t_value": np.asarray(["01", "05"], dtype=object),
                "output_mode": np.asarray(["Flux", "Flux"], dtype=object),
                "calculation_mode": np.asarray(["LTE", "LTE"], dtype=object),
                "a": np.asarray(["+0.00", "+0.00"], dtype=object),
                "c": np.asarray(["+0.00", "+0.00"], dtype=object),
                "n": np.asarray(["+0.00", "+0.00"], dtype=object),
                "o": np.asarray(["+0.00", "+0.00"], dtype=object),
                "r": np.asarray(["+0.00", "+0.00"], dtype=object),
                "s": np.asarray(["+0.00", "+0.00"], dtype=object),
            }
            indices = np.arange(2, dtype=np.int64)
            with self.assertRaises(ValueError) as cm:
                _build_task_batches(
                    indices,
                    columns,
                    batch_size=16,
                    config=self._config(tmpdir, atmosphere_path=atmospheres),
                )
            self.assertIn("duplicate Turbospectrum output filenames", str(cm.exception))

    def test_ml_rows_with_distinguishing_abundance_stay_unique(self) -> None:
        """Rows that rebind to the same atmosphere but differ in abundance
        must not trigger the duplicate guard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            atmospheres = os.path.join(tmpdir, "atmospheres")
            os.makedirs(atmospheres)
            _touch_atmosphere_files(atmospheres, ("01", "02", "05"))

            n = 3
            columns = {
                "teff": np.full(n, 5000, dtype=np.int64),
                "logg": np.full(n, 4.5, dtype=np.float64),
                "feh": np.full(n, -0.5, dtype=np.float64),
                "lam_min": np.full(n, 5000.0, dtype=np.float64),
                "lam_max": np.full(n, 5020.0, dtype=np.float64),
                "lam_step": np.full(n, 0.01, dtype=np.float64),
                "turbvel": np.full(n, 2.5, dtype=np.float64),
                "t_value": np.asarray(["01"] * n, dtype=object),
                "output_mode": np.asarray(["Flux"] * n, dtype=object),
                "calculation_mode": np.asarray(["LTE"] * n, dtype=object),
                "a": np.asarray(["+0.00", "+0.10", "+0.20"], dtype=object),
                "c": np.asarray(["+0.00"] * n, dtype=object),
                "n": np.asarray(["+0.00"] * n, dtype=object),
                "o": np.asarray(["+0.00"] * n, dtype=object),
                "r": np.asarray(["+0.00"] * n, dtype=object),
                "s": np.asarray(["+0.00"] * n, dtype=object),
            }
            indices = np.arange(n, dtype=np.int64)
            tasks, _ = _build_task_batches(
                indices,
                columns,
                batch_size=16,
                config=self._config(tmpdir, atmosphere_path=atmospheres),
            )
            self.assertEqual(sum(len(b) for b in tasks), n)


class MergeResolvedTurbulenceColumnsTests(unittest.TestCase):
    def _base_columns(self):
        return {
            "teff": np.asarray([5000, 5250], dtype=np.int64),
            "turbvel": np.asarray(["2.5", "3.0"], dtype=object),
            "t_value": np.asarray(["01", "05"], dtype=object),
        }

    def test_overlays_resolved_values_onto_grid_columns(self) -> None:
        merged = _merge_resolved_turbulence_columns(
            self._base_columns(),
            resolved_t_values=["02", "02"],
            resolved_turbvels=["2.5", "3.0"],
        )
        self.assertEqual(list(merged["t_value"]), ["02", "02"])
        self.assertEqual(list(merged["turbvel"]), ["2.5", "3.0"])

    def test_empty_resolved_value_falls_back_to_grid_value(self) -> None:
        merged = _merge_resolved_turbulence_columns(
            self._base_columns(),
            resolved_t_values=["02", ""],
            resolved_turbvels=["", "3.0"],
        )
        # Row 1 had no resolved t_value -> keep grid's "05";
        # row 0 had no resolved turbvel -> keep grid's "2.5".
        self.assertEqual(list(merged["t_value"]), ["02", "05"])
        self.assertEqual(list(merged["turbvel"]), ["2.5", "3.0"])

    def test_all_empty_results_leave_columns_untouched(self) -> None:
        original = self._base_columns()
        merged = _merge_resolved_turbulence_columns(
            original,
            resolved_t_values=["", ""],
            resolved_turbvels=["", ""],
        )
        self.assertEqual(list(merged["t_value"]), ["01", "05"])
        self.assertEqual(list(merged["turbvel"]), ["2.5", "3.0"])
        # Must not mutate the caller's column dict.
        self.assertIsNot(merged, original)

    def test_missing_columns_are_created_when_resolutions_exist(self) -> None:
        columns = {"teff": np.asarray([5000, 5250], dtype=np.int64)}
        merged = _merge_resolved_turbulence_columns(
            columns,
            resolved_t_values=["02", "05"],
            resolved_turbvels=["2.5", "5.0"],
        )
        self.assertEqual(list(merged["t_value"]), ["02", "05"])
        self.assertEqual(list(merged["turbvel"]), ["2.5", "5.0"])


if __name__ == "__main__":
    unittest.main()
