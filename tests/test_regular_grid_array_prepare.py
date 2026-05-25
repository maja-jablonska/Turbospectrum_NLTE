import json
import os
import subprocess
import sys
import tempfile
import unittest

import zarr


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PBS_SCRIPT = os.path.join(REPO_ROOT, "turbospectrum_pipeline_example.pbs")
SYNTHESIS_CONFIG = os.path.join(REPO_ROOT, "configs", "synthesis", "config_sample_comprehensive.json")


class RegularGridArrayPrepareTests(unittest.TestCase):
    def test_nonzero_array_task_can_prepare_grid_without_task_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = os.path.join(tmpdir, "run")
            grid_dir = os.path.join(run_root, "outputs", "grids")
            shard_dir = os.path.join(run_root, "outputs", "shards")
            config_path = os.path.join(tmpdir, "config_regular_grid.json")

            config = {
                "grid": {
                    "grid_version": "test-array-prepare",
                    "axes": {
                        "teff": "5000,5250",
                        "logg": "4.0,4.5",
                        "feh": "-0.5,0.0",
                        "turbvel": "01",
                    },
                    "synthesis": {
                        "lam_min": 5000.0,
                        "lam_max": 5001.0,
                        "lam_step": 0.1,
                        "output_mode": "Flux",
                        "mode": "1D",
                        "calculation_mode": "LTE",
                    },
                    "limits": {"max_rows": 128},
                },
                "turbospectrum": {
                    "config": SYNTHESIS_CONFIG,
                },
                "outputs": {
                    "run_root": run_root,
                    "grid_csv": os.path.join(grid_dir, "regular_parameter_grid.csv"),
                    "grid_zarr": os.path.join(grid_dir, "regular_parameter_grid.zarr"),
                    "grid_index_parquet": os.path.join(grid_dir, "index.parquet"),
                    "spectra_zarr": os.path.join(run_root, "outputs", "zarr", "regular_synthesized_spectra.zarr"),
                    "spectra_shard_template": os.path.join(shard_dir, "spectra_shard_{shard_index}.zarr"),
                },
                "runtime": {
                    "workers": 1,
                    "rows_per_shard": 4,
                    "skip_synthesis": True,
                },
            }
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)

            env = os.environ.copy()
            env.update(
                {
                    "PBS_O_WORKDIR": REPO_ROOT,
                    "PBS_ARRAY_INDEX": "5",
                    "PBS_JOBID": "123456[5].gadi-pbs",
                    "CONFIG_PATH": config_path,
                    "PYTHON_BIN": sys.executable,
                    "RUN_ROOT": run_root,
                    "SKIP_SYNTHESIS": "1",
                    "ARRAY_PREP_TIMEOUT": "15",
                }
            )

            proc = subprocess.run(
                ["bash", PBS_SCRIPT],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                proc.returncode,
                0,
                msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}",
            )

            grid_zarr = config["outputs"]["grid_zarr"]
            # The unified pipeline PBS writes a grid-prep marker named
            # .grid_prepare_<jobid>.done where <jobid> is PBS_JOBID truncated at
            # the first '[' (123456[5].gadi-pbs -> 123456). It records a timestamp.
            marker_path = os.path.join(run_root, ".grid_prepare_123456.done")

            self.assertTrue(os.path.isdir(grid_zarr))
            self.assertTrue(os.path.isfile(marker_path))
            with open(marker_path, "r", encoding="utf-8") as handle:
                marker_text = handle.read().strip()
            self.assertTrue(
                marker_text,
                msg="grid-prep marker should record a non-empty timestamp",
            )

            root = zarr.open_group(grid_zarr, mode="r")
            self.assertEqual(int(root["teff"].shape[0]), 8)


if __name__ == "__main__":
    unittest.main()
