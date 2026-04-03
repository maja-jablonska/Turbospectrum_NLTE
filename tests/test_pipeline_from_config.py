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
