"""filter_bad_snap_rows: detect rows whose recorded MARCS atmosphere is not the
correct nearest snap, and write a filtered store."""
import os
import tempfile
import unittest

import numpy as np

import json

from scripts.filter_bad_snap_rows import (
    classify_rows,
    collect_rejects,
    write_filtered,
    write_rejects,
)
from scripts.zarr_compat import (
    zarr_store,
    open_root_group,
    create_root_group,
    create_array,
    write_string_array,
)


def _model_name(geom, teff, logg, mass, turb, feh, alpha):
    return (
        f"{geom}{teff}_g{logg:+.1f}_m{mass}_t{turb}_st_z{feh:+.2f}"
        f"_a{alpha:+.2f}_c+0.00_n+0.00_o{alpha:+.2f}_r+0.00_s+0.00.mod"
    )


_PARAM_NAMES = ["teff", "logg", "feh", "t_value", "marcs_teff", "marcs_logg", "marcs_fe_h"]

# Fixture rows: (requested teff/logg/feh/t_value, recorded marcs teff/logg/fe_h)
_ROWS = np.array([
    #  teff  logg  feh  t   m_teff m_logg m_feh
    [5000.0, 4.0,  0.0, 1.0, 5000.0, 4.0,  0.0],   # correct snap -> keep
    [4500.0, 1.5,  0.0, 1.0, 5000.0, 4.0,  0.0],   # old giant clamp -> drop
    [5000.0, 4.0, -1.0, 5.0, 5000.0, 4.0,  0.0],   # old t05 solar trap -> drop
    [5000.0, 4.0,  0.0, 1.0, np.nan, np.nan, np.nan],  # unresolved row -> drop
    [4500.0, 2.7,  0.0, 1.0, 4500.0, 1.5,  0.0],   # correct snap but 1.2 dex away
], dtype=np.float64)


class FilterBadSnapRowsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = self._tmp.name
        self.atmo_dir = os.path.join(base, "marcs")
        os.makedirs(self.atmo_dir)
        for name in [
            _model_name("p", 5000, 4.0, "0.0", "01", 0.0, 0.0),
            _model_name("p", 5000, 4.0, "0.0", "02", -1.0, 0.4),
            _model_name("s", 4500, 1.5, "1.0", "01", 0.0, 0.0),
        ]:
            with open(os.path.join(self.atmo_dir, name), "w", encoding="utf-8"):
                pass

        self.store_path = os.path.join(base, "spectra.zarr")
        root = create_root_group(zarr_store(self.store_path), overwrite=True)
        root.attrs["title"] = "fixture"
        n = len(_ROWS)
        create_array(root, "params", data=_ROWS.astype(np.float32), chunks=(n, len(_PARAM_NAMES)))
        write_string_array(root, "param_names", _PARAM_NAMES)
        flux = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
        create_array(root, "flux", data=flux, chunks=(2, 4))
        create_array(root, "wavelength", data=np.linspace(5000, 5003, 4).astype(np.float32), chunks=(4,))
        create_array(root, "global_index", data=np.arange(n, dtype=np.int64), chunks=(n,))
        prov = root.create_group("provenance")
        prov.attrs["note"] = "fixture provenance"
        self.src = open_root_group(self.store_path, mode="r")

    def tearDown(self):
        self._tmp.cleanup()

    def test_classify_flags_wrong_snaps_and_nans(self):
        keep, stats, _details = classify_rows(self.src, self.atmo_dir)
        self.assertEqual(keep.tolist(), [True, False, False, False, True])
        self.assertEqual(stats["n_wrong_snap"], 2)
        self.assertEqual(stats["n_nan"], 1)
        self.assertEqual(stats["n_kept"], 2)

    def test_tolerance_drops_far_but_correct_snaps(self):
        keep, stats, _details = classify_rows(self.src, self.atmo_dir, max_dlogg=0.5)
        self.assertEqual(keep.tolist(), [True, False, False, False, False])
        self.assertEqual(stats["n_out_of_tolerance"], 1)

    def test_write_filtered_store(self):
        keep, stats, _details = classify_rows(self.src, self.atmo_dir)
        out_path = os.path.join(self._tmp.name, "clean.zarr")
        write_filtered(
            self.src, out_path, out_path + ".tmp", keep, stats,
            self.atmo_dir, {"max_dlogg": None},
        )
        self.assertFalse(os.path.exists(out_path + ".tmp"))
        dst = open_root_group(out_path, mode="r")
        self.assertEqual(dst["flux"].shape, (2, 4))
        np.testing.assert_array_equal(dst["flux"][0], np.arange(4, dtype=np.float32))
        np.testing.assert_array_equal(dst["flux"][1], np.arange(16, 20, dtype=np.float32))
        np.testing.assert_array_equal(dst["source_row_index"][...], [0, 4])
        np.testing.assert_array_equal(dst["wavelength"][...], self.src["wavelength"][...])
        self.assertEqual([str(v) for v in dst["param_names"][...]], _PARAM_NAMES)
        self.assertEqual(dst.attrs["title"], "fixture")
        self.assertEqual(dst["provenance"].attrs["note"], "fixture provenance")
        self.assertEqual(dst["snap_filter"].attrs["n_kept"], 2)

    def test_rejects_report_records_reason_and_correct_snap(self):
        keep, stats, details = classify_rows(self.src, self.atmo_dir)
        rejects = collect_rejects(self.src, self.store_path, keep, details)
        self.assertEqual(len(rejects), 3)
        by_row = {r["source_row"]: r for r in rejects}
        # Giant clamp: correct snap is the spherical (4500, 1.5) grid point.
        self.assertEqual(by_row[1]["reason"], "wrong_snap")
        self.assertEqual(by_row[1]["correct_marcs_logg"], 1.5)
        # t05 solar trap: correct snap is the metal-poor t02 model.
        self.assertEqual(by_row[2]["reason"], "wrong_snap")
        self.assertEqual(by_row[2]["correct_marcs_feh"], -1.0)
        self.assertEqual(by_row[3]["reason"], "nan_marcs")
        # Requested params travel with each reject for re-run planning.
        self.assertEqual(by_row[1]["teff"], 4500.0)
        self.assertEqual(by_row[1]["logg"], 1.5)

        csv_path = os.path.join(self._tmp.name, "rejects.csv")
        summary_path = write_rejects(csv_path, rejects, stats["n_rows"])
        with open(csv_path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        self.assertEqual(len(lines), 4)  # header + 3 rejects
        self.assertTrue(lines[0].startswith("run_path,source_row,reason,"))
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        self.assertEqual(summary["n_rejected"], 3)
        self.assertEqual(summary["rejects_by_reason"], {"wrong_snap": 2, "nan_marcs": 1})
        self.assertEqual(summary["n_unique_rejected_points_excl_mu"], 3)
        self.assertEqual(summary["rejected_request_axes"]["logg"]["min"], 1.5)

    def test_plane_parallel_only_pool_cannot_flag_giants_and_warns(self):
        # With no spherical models in the pool, the logg-3-clamped giant's
        # recorded atmosphere IS the pool's true nearest, so it survives the
        # wrong-snap check. The classifier must surface this blind spot loudly.
        import contextlib
        import io

        p_only = os.path.join(self._tmp.name, "marcs_p_only")
        os.makedirs(p_only)
        for name in [
            _model_name("p", 5000, 4.0, "0.0", "01", 0.0, 0.0),
            _model_name("p", 5000, 4.0, "0.0", "02", -1.0, 0.4),
        ]:
            with open(os.path.join(p_only, name), "w", encoding="utf-8"):
                pass

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            keep, stats, _details = classify_rows(self.src, p_only)
        # The clamped giant (row 1) is KEPT — undetectable with this pool —
        # while the t05 trap row and the NaN row are still dropped. Row 4's
        # recorded spherical atmosphere is not in the pool -> wrong_snap.
        self.assertEqual(keep.tolist(), [True, True, False, False, False])
        self.assertEqual(stats["n_below_pool_logg_floor"], 2)
        self.assertFalse(stats["pool_has_spherical"])
        self.assertEqual(stats["pool_logg_floor"], 4.0)
        self.assertIn("NO spherical", stderr.getvalue())
        self.assertIn("--max-dlogg", stderr.getvalue())

        # Tolerance flags are the remedy: the clamped giant goes too.
        with contextlib.redirect_stderr(io.StringIO()):
            keep_tol, _, _ = classify_rows(self.src, p_only, max_dlogg=0.25)
        self.assertEqual(keep_tol.tolist(), [True, False, False, False, False])

    def test_refuses_existing_output(self):
        keep, stats, _details = classify_rows(self.src, self.atmo_dir)
        out_path = os.path.join(self._tmp.name, "exists.zarr")
        os.makedirs(out_path)
        with self.assertRaises(FileExistsError):
            write_filtered(self.src, out_path, out_path + ".tmp", keep, stats, self.atmo_dir, {})


if __name__ == "__main__":
    unittest.main()
