import tempfile
import unittest

import numpy as np

from scripts.run_turbospectrum import TurbospectrumConfig, get_synthesis_output_stem_from_params
from scripts.run_turbospectrum_shard import _build_task_batches
from scripts.synthesize_spectra_from_zarr import _build_tasks


class SynthesisTaskIdentityTests(unittest.TestCase):
    def _make_config(self, project_root: str) -> TurbospectrumConfig:
        return TurbospectrumConfig(
            project_root=project_root,
            output_mode="Flux",
            nlte=False,
        )

    def test_output_stem_distinguishes_t_value_turbvel_and_abundances(self) -> None:
        base = {
            "teff": 5000,
            "logg": 4.0,
            "feh": -0.3,
            "turbvel": "01",
            "t_value": "00",
            "a": "+0.10",
            "c": "+0.00",
            "n": "+0.00",
            "o": "+0.00",
            "r": "+0.00",
            "s": "+0.00",
            "lam_min": 6000.0,
            "lam_max": 6100.0,
            "lam_step": 0.01,
            "output_mode": "Flux",
            "calculation_mode": "LTE",
        }

        self.assertNotEqual(
            get_synthesis_output_stem_from_params(base),
            get_synthesis_output_stem_from_params({**base, "t_value": "01"}),
        )
        self.assertNotEqual(
            get_synthesis_output_stem_from_params(base),
            get_synthesis_output_stem_from_params({**base, "turbvel": "02"}),
        )
        self.assertNotEqual(
            get_synthesis_output_stem_from_params(base),
            get_synthesis_output_stem_from_params({**base, "a": "+0.20"}),
        )

    def test_build_tasks_preserves_turbulence_and_abundance_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            tasks = _build_tasks(
                1,
                {
                    "teff": np.asarray([5000]),
                    "logg": np.asarray([4.0]),
                    "feh": np.asarray([-0.3]),
                    "lam_min": np.asarray([6000.0]),
                    "lam_max": np.asarray([6100.0]),
                    "lam_step": np.asarray([0.01]),
                    "turbvel": np.asarray(["01"], dtype=object),
                    "t_value": np.asarray(["00"], dtype=object),
                    "a": np.asarray(["+0.10"], dtype=object),
                    "c": np.asarray(["+0.02"], dtype=object),
                    "output_mode": np.asarray(["Flux"], dtype=object),
                    "calculation_mode": np.asarray(["LTE"], dtype=object),
                },
                config,
            )

            _, row_values = tasks[0]
            self.assertEqual(str(row_values["turbvel"]), "01")
            self.assertEqual(str(row_values["t_value"]), "00")
            self.assertEqual(str(row_values["a"]), "+0.10")
            self.assertEqual(str(row_values["c"]), "+0.02")

    def test_build_shard_task_batches_keeps_rows_distinct_when_abundances_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            indices = np.asarray([0, 1], dtype=np.int64)
            columns = {
                "teff": np.asarray([5000, 5000]),
                "logg": np.asarray([5.0, 5.0]),
                "feh": np.asarray([-0.36, -0.36]),
                "lam_min": np.asarray([6000.0, 6000.0]),
                "lam_max": np.asarray([6100.0, 6100.0]),
                "lam_step": np.asarray([0.01, 0.01]),
                "turbvel": np.asarray(["01", "01"], dtype=object),
                "t_value": np.asarray(["00", "02"], dtype=object),
                "a": np.asarray(["+0.00", "+0.10"], dtype=object),
                "c": np.asarray(["-0.04", "+0.02"], dtype=object),
                "output_mode": np.asarray(["Flux", "Flux"], dtype=object),
                "calculation_mode": np.asarray(["LTE", "LTE"], dtype=object),
            }

            tasks, global_to_local = _build_task_batches(
                indices,
                columns,
                batch_size=8,
                config=config,
            )

            self.assertEqual(global_to_local, {0: 0, 1: 1})
            self.assertEqual(len(tasks), 1)
            self.assertEqual(len(tasks[0]), 2)
            self.assertEqual(str(tasks[0][0][1]["t_value"]), "00")
            self.assertEqual(str(tasks[0][1][1]["t_value"]), "02")
            self.assertEqual(str(tasks[0][0][1]["a"]), "+0.00")
            self.assertEqual(str(tasks[0][1][1]["a"]), "+0.10")


if __name__ == "__main__":
    unittest.main()
