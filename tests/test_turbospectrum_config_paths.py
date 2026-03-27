import contextlib
import dataclasses
import io
import os
import stat
import tempfile
import unittest

from scripts.run_turbospectrum import (
    TurbospectrumConfig,
    _normalize_config_dict,
    validate_runtime_environment,
)


class TurbospectrumConfigPathTests(unittest.TestCase):
    @staticmethod
    def _write_executable(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\nexit 0\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)

    def test_nested_absolute_paths_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = os.path.join(tmpdir, "project")
            model_dir = os.path.join(tmpdir, "external", "models")
            linelist_dir = os.path.join(tmpdir, "external", "linelists")
            output_dir = os.path.join(tmpdir, "external", "outputs")
            log_dir = os.path.join(tmpdir, "external", "logs")
            tmp_work_dir = os.path.join(tmpdir, "external", "tmp")
            babsma_path = os.path.join(tmpdir, "external", "bin", "babsma_lu")
            bsyn_path = os.path.join(tmpdir, "external", "bin", "bsyn_lu")
            interpol_path = os.path.join(tmpdir, "external", "interp", "interpol_modeles")

            os.makedirs(project_root, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(linelist_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(tmp_work_dir, exist_ok=True)
            with open(os.path.join(model_dir, "sample.mod"), "w", encoding="utf-8") as handle:
                handle.write("model\n")

            self._write_executable(babsma_path)
            self._write_executable(bsyn_path)
            self._write_executable(interpol_path)

            cfg_data = {
                "project_root": project_root,
                "executables": {
                    "babsma_path": babsma_path,
                    "bsyn_path": bsyn_path,
                    "interpol_path": interpol_path,
                },
                "paths": {
                    "model_atmosphere_path": model_dir,
                    "linelist_path": linelist_dir,
                    "output_dir": output_dir,
                    "log_dir": log_dir,
                    "tmp_dir": tmp_work_dir,
                },
                "synthesis_parameters": {
                    "output_mode": "Flux",
                },
            }

            normalized = _normalize_config_dict(cfg_data, default_project_root=project_root)
            accepted = {field.name for field in dataclasses.fields(TurbospectrumConfig)}
            normalized = {key: value for key, value in normalized.items() if key in accepted}

            with contextlib.redirect_stdout(io.StringIO()):
                config = TurbospectrumConfig(**normalized)

            self.assertEqual(config.babsma_path, babsma_path)
            self.assertEqual(config.bsyn_path, bsyn_path)
            self.assertEqual(config.interpol_path, interpol_path)
            self.assertEqual(config.model_atmosphere_path, model_dir)
            self.assertEqual(config.linelist_path, linelist_dir)
            self.assertEqual(config.output_dir, output_dir)
            self.assertEqual(config.log_dir, log_dir)
            self.assertEqual(config.tmp_dir, tmp_work_dir)

            validate_runtime_environment(config)


if __name__ == "__main__":
    unittest.main()
