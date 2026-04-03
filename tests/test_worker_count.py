import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

from scripts.run_turbospectrum import TurbospectrumConfig, determine_worker_count


class WorkerCountTests(unittest.TestCase):
    def test_requested_workers_are_clamped_by_detected_memory_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                config = TurbospectrumConfig(
                    project_root=tmpdir,
                    output_mode="Intensity",
                    nlte=False,
                )

            with mock.patch.dict(os.environ, {"PBS_NP": "96", "PBS_RESOURCE_LIST_MEM": "192GB"}, clear=False):
                with mock.patch("scripts.run_turbospectrum.multiprocessing.cpu_count", return_value=128):
                    with mock.patch("scripts.run_turbospectrum.os.sched_getaffinity", return_value=set(range(96)), create=True):
                        with contextlib.redirect_stdout(io.StringIO()):
                            workers = determine_worker_count(config, requested_workers=96)

        self.assertEqual(workers, 30)


if __name__ == "__main__":
    unittest.main()
