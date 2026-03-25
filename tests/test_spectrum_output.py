import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import zarr

from scripts.spectrum_output import extract_flux_and_continuum, infer_flux_metadata
from scripts.run_turbospectrum import TurbospectrumConfig
import scripts.synthesize_spectra_from_zarr as synth_from_zarr


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGE_SCRIPT = os.path.join(REPO_ROOT, "scripts", "merge_spectra_shards.py")


class SpectrumOutputTests(unittest.TestCase):
    @staticmethod
    def _create_array(group, name: str, data: np.ndarray) -> None:
        if hasattr(group, "create_array"):
            group.create_array(name, data=data)
        else:
            group.create_dataset(name, data=data)

    @staticmethod
    def _create_string_array(group, name: str, values: list[str]) -> None:
        vals = [str(value) for value in values]
        if hasattr(group, "create_array"):
            import zarr.codecs as zc  # type: ignore
            from zarr.core.dtype.npy.string import VariableLengthUTF8  # type: ignore

            array = group.create_array(
                name,
                shape=(len(vals),),
                dtype=VariableLengthUTF8(),
                serializer=zc.VLenUTF8Codec(),
                chunks=min(len(vals), 128) if vals else 1,
            )
            array[:] = vals
            return

        try:
            from numcodecs import VLenUTF8  # type: ignore

            group.array(
                name,
                vals,
                dtype=object,
                object_codec=VLenUTF8(),
                chunks=min(len(vals), 128) if vals else 1,
            )
        except Exception:
            group.create_dataset(name, data=np.asarray(vals, dtype="U256"))

    def test_flux_mode_uses_unnormalized_flux_column(self) -> None:
        data = np.asarray(
            [
                [5000.0, 0.50, 2.00],
                [5000.1, 0.25, 1.00],
            ],
            dtype=np.float32,
        )

        flux, continuum = extract_flux_and_continuum(data, is_intensity=False)

        np.testing.assert_allclose(flux, np.asarray([2.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(continuum, np.asarray([4.0, 4.0], dtype=np.float32))

    def test_synthesis_task_reads_flux_spectrum_without_unbound_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, "test.spec")
            with open(spec_path, "w", encoding="utf-8") as handle:
                handle.write("5000.0 0.50 2.00\n")
                handle.write("5000.1 0.25 1.00\n")

            config = TurbospectrumConfig(
                project_root=tmpdir,
                output_mode="Flux",
                nlte=False,
                output_dir=tmpdir,
                log_dir=tmpdir,
                tmp_dir=tmpdir,
            )
            synth_from_zarr._init_worker(config)
            row_values = {
                "teff": 5000,
                "logg": 4.0,
                "feh": 0.0,
                "lam_min": 5000.0,
                "lam_max": 5000.1,
                "lam_step": 0.1,
                "turbvel": "01",
                "t_value": "01",
                "output_mode": "Flux",
                "calculation_mode": "LTE",
            }

            with mock.patch.object(
                synth_from_zarr,
                "run_single_synthesis",
                return_value={
                    "status": "success",
                    "message": "ok",
                    "output_path": spec_path,
                    "base_name": "test",
                    "log_path": "",
                },
            ):
                result = synth_from_zarr._synthesis_task((0, row_values))

            self.assertEqual(result["status"], "success")
            self.assertIsNotNone(result["spectrum"])
            flux, continuum = result["spectrum"]
            np.testing.assert_allclose(flux, np.asarray([2.0, 1.0], dtype=np.float32))
            np.testing.assert_allclose(continuum, np.asarray([4.0, 4.0], dtype=np.float32))

    def test_flux_metadata_infers_unnormalized_flux(self) -> None:
        self.assertEqual(
            infer_flux_metadata(["Flux"]),
            ("unnormalized_flux", "relative"),
        )

    def test_merge_auto_labels_flux_mode_as_unnormalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            grid_path = os.path.join(tmpdir, "grid.zarr")
            shard_path = os.path.join(tmpdir, "spectra_shard_0.zarr")
            output_path = os.path.join(tmpdir, "merged.zarr")

            grid = zarr.open_group(grid_path, mode="w")
            self._create_array(grid, "teff", np.asarray([4500.0], dtype=np.float32))
            self._create_array(grid, "logg", np.asarray([2.0], dtype=np.float32))
            self._create_array(grid, "feh", np.asarray([0.0], dtype=np.float32))
            self._create_array(grid, "turbvel", np.asarray([1.5], dtype=np.float32))
            self._create_string_array(grid, "output_mode", ["Flux"])

            shard = zarr.open_group(shard_path, mode="w")
            self._create_array(shard, "wavelength", np.asarray([5000.0, 5000.2], dtype=np.float32))
            self._create_array(shard, "global_index", np.asarray([0], dtype=np.int64))
            self._create_array(shard, "flux", np.asarray([[2.0, 1.8]], dtype=np.float32))
            self._create_array(shard, "continuum", np.asarray([[4.0, 4.0]], dtype=np.float32))
            self._create_string_array(shard, "status", ["success"])
            self._create_string_array(shard, "message", ["ok"])

            result = subprocess.run(
                [
                    sys.executable,
                    MERGE_SCRIPT,
                    "--output-zarr",
                    output_path,
                    "--grid-zarr",
                    grid_path,
                    "--shard",
                    shard_path,
                    "--allow-incomplete-provenance",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            merged = zarr.open_group(output_path, mode="r")
            self.assertEqual(str(merged.attrs["flux_definition"]), "unnormalized_flux")
            self.assertEqual(str(merged.attrs["flux_unit"]), "relative")
            np.testing.assert_allclose(
                np.asarray(merged["flux"][:], dtype=np.float32),
                np.asarray([[2.0, 1.8]], dtype=np.float32),
            )
            np.testing.assert_allclose(
                np.asarray(merged["continuum"][:], dtype=np.float32),
                np.asarray([[4.0, 4.0]], dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
