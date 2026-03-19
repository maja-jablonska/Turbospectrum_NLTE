import os
import tempfile
import unittest

import numpy as np
import zarr

from scripts.synthesize_regular_grid import _verify_continuum_saved
from scripts.synthesize_spectra_from_zarr import _validate_synthesis_results


class RegularGridValidationTests(unittest.TestCase):
    @staticmethod
    def _create_array(group, name: str, data: np.ndarray) -> None:
        if hasattr(group, "create_array"):
            group.create_array(name, data=data)
        else:
            group.create_dataset(name, data=data)

    def test_validate_synthesis_results_accepts_finite_successful_rows(self) -> None:
        fluxes = np.array([[1.0, 0.9], [0.8, 0.7]], dtype=np.float32)
        continua = np.array([[1.0, 1.0], [0.9, 0.9]], dtype=np.float32)

        counts = _validate_synthesis_results(
            statuses=["success", "skipped"],
            messages=["ok", "cached"],
            fluxes=fluxes,
            continua=continua,
        )

        self.assertEqual(counts, {"success": 1, "skipped": 1})

    def test_validate_synthesis_results_rejects_failed_rows(self) -> None:
        fluxes = np.array([[1.0, 0.9], [0.8, 0.7]], dtype=np.float32)
        continua = np.array([[1.0, 1.0], [0.9, 0.9]], dtype=np.float32)

        with self.assertRaises(RuntimeError) as exc_info:
            _validate_synthesis_results(
                statuses=["success", "error"],
                messages=["ok", "bsyn failed"],
                fluxes=fluxes,
                continua=continua,
            )

        self.assertIn("failed rows=1", str(exc_info.exception))
        self.assertIn("bsyn failed", str(exc_info.exception))

    def test_validate_synthesis_results_rejects_nonfinite_flux(self) -> None:
        fluxes = np.array([[1.0, 0.9], [np.nan, 0.7]], dtype=np.float32)
        continua = np.array([[1.0, 1.0], [0.9, 0.9]], dtype=np.float32)

        with self.assertRaises(RuntimeError) as exc_info:
            _validate_synthesis_results(
                statuses=["success", "success"],
                messages=["ok", "missing spectrum"],
                fluxes=fluxes,
                continua=continua,
            )

        self.assertIn("rows with non-finite flux=1", str(exc_info.exception))

    def test_verify_continuum_saved_rejects_failed_status_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = zarr.open_group(tmpdir, mode="w")
            self._create_array(root, "wavelength", np.array([1.0, 2.0], dtype=np.float32))
            self._create_array(root, "flux", np.array([[1.0, 0.9]], dtype=np.float32))
            self._create_array(root, "continuum", np.array([[1.0, 1.0]], dtype=np.float32))
            root.attrs["status_counts"] = {"success": 1, "error": 2}

            with self.assertRaises(ValueError) as exc_info:
                _verify_continuum_saved(tmpdir)

            self.assertIn("status_counts", str(exc_info.exception))

    def test_verify_continuum_saved_accepts_success_only_status_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = zarr.open_group(tmpdir, mode="w")
            self._create_array(root, "wavelength", np.array([1.0, 2.0], dtype=np.float32))
            self._create_array(root, "flux", np.array([[1.0, 0.9]], dtype=np.float32))
            self._create_array(root, "continuum", np.array([[1.0, 1.0]], dtype=np.float32))
            root.attrs["status_counts"] = {"success": 1}

            _verify_continuum_saved(tmpdir)


if __name__ == "__main__":
    unittest.main()
