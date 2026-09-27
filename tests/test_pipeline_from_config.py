import subprocess
import unittest
from unittest import mock

from scripts import pipeline_from_config


class PipelineFromConfigRunTests(unittest.TestCase):
    def test_run_retries_with_fewer_workers_after_sigkill(self) -> None:
        calls = []

        def fake_run(cmd, check):  # type: ignore[no-untyped-def]
            self.assertTrue(check)
            calls.append(list(cmd))
            workers = int(cmd[cmd.index("--workers") + 1])
            if workers >= 96:
                raise subprocess.CalledProcessError(-9, cmd)
            return None

        cmd = ["python", "child.py", "--workers", "96"]
        with mock.patch.object(pipeline_from_config.subprocess, "run", side_effect=fake_run):
            pipeline_from_config._run(cmd)

        self.assertEqual(
            calls,
            [
                ["python", "child.py", "--workers", "96"],
                ["python", "child.py", "--workers", "48"],
            ],
        )


if __name__ == "__main__":
    unittest.main()


class SynthesisPassthroughTests(unittest.TestCase):
    def test_runtime_limits_are_forwarded(self) -> None:
        runtime = {"max_failures": 5, "fail_fast_after": 20, "batch_size": 8}
        self.assertEqual(
            pipeline_from_config._synthesis_passthrough_args(runtime, True, sharded=True),
            ["--max-failures", "5", "--fail-fast-after", "20", "--batch-size", "8", "--allow-incomplete-provenance"],
        )
        # The full-mode engine has no --batch-size.
        self.assertEqual(
            pipeline_from_config._synthesis_passthrough_args(runtime, False, sharded=False),
            ["--max-failures", "5", "--fail-fast-after", "20"],
        )
        self.assertEqual(pipeline_from_config._synthesis_passthrough_args({}, False, sharded=True), [])

    def test_shard_runner_sigkill_code_triggers_retry(self) -> None:
        from scripts import run_turbospectrum_shard

        self.assertTrue(pipeline_from_config._is_sigkill_returncode(run_turbospectrum_shard.SIGKILL_EXIT_CODE))


class GridGenerationSafetyTests(unittest.TestCase):
    def test_lhs_requires_seed(self) -> None:
        import numpy as np

        from scripts import generate_grid

        with self.assertRaisesRegex(ValueError, "grid.seed is required"):
            generate_grid.resolve_grid_columns({"sampling": "lhs", "num_samples": 4}, rng=np.random.default_rng())

    def test_grid_zarr_write_is_atomic(self) -> None:
        import os
        import tempfile

        import numpy as np

        from scripts import generate_grid

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "grids", "grid.zarr")
            generate_grid._write_zarr_from_columns({"teff": np.arange(3.0)}, path, chunks=2, compressor_cfg=None)
            self.assertTrue(os.path.isdir(path))
            import zarr_compat

            real_create = zarr_compat.create_array

            def create_then_die(group, name, **kw):  # simulate a job killed mid-write
                if name == "logg":
                    raise RuntimeError("killed")
                return real_create(group, name, **kw)

            cols = {"teff": np.arange(3.0) + 10, "logg": np.arange(3.0)}
            with mock.patch.object(zarr_compat, "create_array", side_effect=create_then_die):
                with self.assertRaisesRegex(RuntimeError, "killed"):
                    generate_grid._write_zarr_from_columns(cols, path, chunks=2, compressor_cfg=None)
            import zarr

            root = zarr.open_group(path, mode="r")
            np.testing.assert_array_equal(root["teff"][:], np.arange(3.0))  # previous grid intact
            self.assertEqual(sorted(os.listdir(os.path.dirname(path))), ["grid.zarr"])


class SharedMuPointsTests(unittest.TestCase):
    def _grid(self, tmpdir, mu, mode="Intensity"):
        import os

        import numpy as np
        import zarr

        from scripts.zarr_compat import create_array, write_string_array

        path = os.path.join(tmpdir, "grid.zarr")
        root = zarr.open_group(path, mode="w")
        create_array(root, "mu", data=np.asarray(mu, dtype=np.float64))
        write_string_array(root, "output_mode", [mode] * len(mu))
        return path

    def test_regular_intensity_grid_gets_shared_axis(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._grid(tmpdir, [0.15, 0.45, 0.75, 0.15, 0.45, 0.75])
            self.assertEqual(pipeline_from_config._shared_mu_points(path), [0.15, 0.45, 0.75])

    def test_flux_or_continuous_mu_keeps_per_row_path(self) -> None:
        import tempfile

        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(pipeline_from_config._shared_mu_points(self._grid(tmpdir, [0.5, 0.6], mode="Flux")))
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(pipeline_from_config._shared_mu_points(self._grid(tmpdir, np.linspace(0.1, 1, 31))))
        self.assertIsNone(pipeline_from_config._shared_mu_points(None))


class FullModeFailedRowsTests(unittest.TestCase):
    def _run_full(self, tmpdir, shard_written: bool):
        import json
        import os

        cfg = {
            "grid": {},
            "turbospectrum": {},
            "outputs": {"grid_zarr": os.path.join(tmpdir, "grid.zarr"),
                        "spectra_zarr": os.path.join(tmpdir, "out", "spectra.zarr")},
        }
        cfg_path = os.path.join(tmpdir, "cfg.json")
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh)
        merge_calls = []

        def fake_run(cmd):
            if shard_written:
                os.makedirs(cmd[cmd.index("--output-zarr") + 1], exist_ok=True)
            raise subprocess.CalledProcessError(1, cmd)

        def fake_subprocess_run(cmd, check):
            merge_calls.append(cmd)

        with mock.patch.object(pipeline_from_config, "_run", side_effect=fake_run), \
             mock.patch.object(pipeline_from_config.subprocess, "run", side_effect=fake_subprocess_run), \
             mock.patch.object(pipeline_from_config, "_write_temp_turbospectrum_config", return_value="ts.json"), \
             mock.patch.object(pipeline_from_config, "_nlte_preflight_or_exit"), \
             mock.patch("sys.argv", ["pipeline_from_config.py", "--config", cfg_path, "--skip-grid"]):
            pipeline_from_config.main()
        return merge_calls

    def test_failed_rows_still_merge_when_shard_was_written(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            calls = self._run_full(tmpdir, shard_written=True)
            self.assertEqual(len(calls), 1)
            self.assertIn("merge_spectra_shards.py", calls[0][1])
            self.assertNotIn("--fail-on-failed-rows", calls[0])

    def test_no_shard_is_fatal(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(subprocess.CalledProcessError):
                self._run_full(tmpdir, shard_written=False)
