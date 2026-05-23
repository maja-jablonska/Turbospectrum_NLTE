import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import zarr


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGE_SCRIPT = os.path.join(REPO_ROOT, "scripts", "merge_spectra_shards.py")

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from merge_spectra_shards import (  # noqa: E402
    PROVENANCE_FILENAMES,
    _read_string_scalar_optional,
)
from zarr_compat import write_string_scalar  # noqa: E402


class MergeSpectraShardsTests(unittest.TestCase):
    @staticmethod
    def _create_array(group, name: str, data: np.ndarray) -> None:
        if hasattr(group, "create_array"):
            group.create_array(name, data=data)
        else:
            group.create_dataset(name, data=data)

    def _write_grid(self, path: str, row_count: int) -> None:
        root = zarr.open_group(path, mode="w")
        self._create_array(root, "teff", np.linspace(4500.0, 4700.0, row_count, dtype=np.float32))
        self._create_array(root, "logg", np.full(row_count, 2.0, dtype=np.float32))
        self._create_array(root, "feh", np.zeros(row_count, dtype=np.float32))
        self._create_array(root, "turbvel", np.full(row_count, 1.5, dtype=np.float32))

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

    def _write_shard(
        self,
        path: str,
        *,
        wavelengths: np.ndarray | None,
        global_index: np.ndarray,
        flux: np.ndarray,
        statuses: list[str],
        messages: list[str],
    ) -> None:
        root = zarr.open_group(path, mode="w")
        if wavelengths is not None:
            self._create_array(root, "wavelength", np.asarray(wavelengths, dtype=np.float32))
        self._create_array(root, "global_index", np.asarray(global_index, dtype=np.int64))
        self._create_array(root, "flux", np.asarray(flux, dtype=np.float32))
        self._create_string_array(root, "status", statuses)
        self._create_string_array(root, "message", messages)

    def test_partial_merge_skips_invalid_existing_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            grid_path = os.path.join(tmpdir, "grid.zarr")
            output_path = os.path.join(tmpdir, "merged.zarr")
            valid_shard = os.path.join(tmpdir, "spectra_shard_0.zarr")
            invalid_shard = os.path.join(tmpdir, "spectra_shard_1.zarr")
            missing_shard = os.path.join(tmpdir, "spectra_shard_2.zarr")

            self._write_grid(grid_path, row_count=3)
            self._write_shard(
                valid_shard,
                wavelengths=np.asarray([5000.0, 5000.2], dtype=np.float32),
                global_index=np.asarray([0], dtype=np.int64),
                flux=np.asarray([[1.0, 0.95]], dtype=np.float32),
                statuses=["success"],
                messages=["ok"],
            )
            self._write_shard(
                invalid_shard,
                wavelengths=None,
                global_index=np.asarray([1], dtype=np.int64),
                flux=np.asarray([[0.8, 0.75]], dtype=np.float32),
                statuses=["success"],
                messages=["missing wavelength"],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    MERGE_SCRIPT,
                    "--output-zarr",
                    output_path,
                    "--grid-zarr",
                    grid_path,
                    "--shard",
                    valid_shard,
                    "--shard",
                    invalid_shard,
                    "--shard",
                    missing_shard,
                    "--allow-missing",
                    "--skip-nonexistent-shards",
                    "--allow-incomplete-provenance",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            merged = zarr.open_group(output_path, mode="r")
            np.testing.assert_array_equal(
                np.asarray(merged["global_index"][:], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
            )
            np.testing.assert_allclose(
                np.asarray(merged["wavelength"][:], dtype=np.float32),
                np.asarray([5000.0, 5000.2], dtype=np.float32),
            )
            np.testing.assert_allclose(
                np.asarray(merged["flux"][:], dtype=np.float32),
                np.asarray([[1.0, 0.95]], dtype=np.float32),
            )
            self.assertEqual(int(merged.attrs["skipped_nonexistent_shards"]), 1)
            self.assertEqual(int(merged.attrs["skipped_invalid_shards"]), 1)
            self.assertTrue(bool(merged.attrs["skip_invalid_shards_effective"]))

    def test_partial_merge_skips_nan_flux_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            grid_path = os.path.join(tmpdir, "grid.zarr")
            output_path = os.path.join(tmpdir, "merged.zarr")
            valid_shard = os.path.join(tmpdir, "spectra_shard_0.zarr")
            nan_shard = os.path.join(tmpdir, "spectra_shard_1.zarr")
            missing_shard = os.path.join(tmpdir, "spectra_shard_2.zarr")

            self._write_grid(grid_path, row_count=3)
            self._write_shard(
                valid_shard,
                wavelengths=np.asarray([5000.0, 5000.2], dtype=np.float32),
                global_index=np.asarray([0], dtype=np.int64),
                flux=np.asarray([[1.0, 0.95]], dtype=np.float32),
                statuses=["success"],
                messages=["ok"],
            )
            self._write_shard(
                nan_shard,
                wavelengths=np.asarray([5000.0, 5000.2], dtype=np.float32),
                global_index=np.asarray([1], dtype=np.int64),
                flux=np.asarray([[np.nan, np.nan]], dtype=np.float32),
                statuses=["success"],
                messages=["nan flux"],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    MERGE_SCRIPT,
                    "--output-zarr",
                    output_path,
                    "--grid-zarr",
                    grid_path,
                    "--shard",
                    valid_shard,
                    "--shard",
                    nan_shard,
                    "--shard",
                    missing_shard,
                    "--allow-missing",
                    "--skip-nonexistent-shards",
                    "--allow-incomplete-provenance",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            merged = zarr.open_group(output_path, mode="r")
            np.testing.assert_array_equal(
                np.asarray(merged["global_index"][:], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
            )
            np.testing.assert_allclose(
                np.asarray(merged["flux"][:], dtype=np.float32),
                np.asarray([[1.0, 0.95]], dtype=np.float32),
            )
            self.assertEqual(int(merged.attrs["skipped_nonexistent_shards"]), 1)
            self.assertEqual(int(merged.attrs["skipped_invalid_shards"]), 1)
            self.assertTrue(bool(merged.attrs["skip_invalid_shards_effective"]))

    def _write_shard_provenance(self, path: str, input_config_text: str) -> None:
        """Attach a full provenance group to an existing shard zarr."""
        root = zarr.open_group(path, mode="a")
        prov = root.require_group("provenance")
        payload = {
            "canonical_config.yaml": "config: true",
            "synthesis_config.yaml": "synth: true",
            "input_config.json": input_config_text,
            "linelist_manifest.json": json.dumps({"lines": 1}),
            "atmosphere_manifest.json": json.dumps({"atm": 1}),
            "software_manifest.json": json.dumps({"sw": 1}),
            "environment.txt": "python==3.11",
        }
        for name, text in payload.items():
            write_string_scalar(prov, name, text)

    def test_merge_carries_forward_total_input_config(self) -> None:
        self.assertIn("input_config.json", PROVENANCE_FILENAMES)
        with tempfile.TemporaryDirectory() as tmpdir:
            grid_path = os.path.join(tmpdir, "grid.zarr")
            output_path = os.path.join(tmpdir, "merged.zarr")
            shard = os.path.join(tmpdir, "spectra_shard_0.zarr")

            # The total input config carries a key the TurbospectrumConfig dataclass
            # filter would drop; provenance must preserve it verbatim.
            input_config = {
                "lambda_min": 5000.0,
                "extra_key_not_in_dataclass": "must-survive",
            }
            input_config_text = json.dumps(input_config, indent=2, sort_keys=True)

            self._write_grid(grid_path, row_count=1)
            self._write_shard(
                shard,
                wavelengths=np.asarray([5000.0, 5000.2], dtype=np.float32),
                global_index=np.asarray([0], dtype=np.int64),
                flux=np.asarray([[1.0, 0.95]], dtype=np.float32),
                statuses=["success"],
                messages=["ok"],
            )
            self._write_shard_provenance(shard, input_config_text)

            result = subprocess.run(
                [
                    sys.executable,
                    MERGE_SCRIPT,
                    "--output-zarr",
                    output_path,
                    "--grid-zarr",
                    grid_path,
                    "--shard",
                    shard,
                    "--allow-incomplete-provenance",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            merged = zarr.open_group(output_path, mode="r")
            self.assertIn("provenance", merged)
            carried = _read_string_scalar_optional(merged["provenance"], "input_config.json")
            self.assertEqual(carried, input_config_text)
            self.assertIn("extra_key_not_in_dataclass", carried)


if __name__ == "__main__":
    unittest.main()
