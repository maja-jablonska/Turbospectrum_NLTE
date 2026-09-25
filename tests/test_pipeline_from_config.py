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
