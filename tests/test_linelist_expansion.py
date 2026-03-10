import contextlib
import io
import os
import tempfile
import unittest

from scripts.run_turbospectrum import (
    LinelistValidationError,
    TurbospectrumConfig,
    create_linelist_file,
    resolve_linelist_paths,
    validate_linelist_files,
)


class LinelistExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.project_root = self.tempdir.name
        self.linelist_root = os.path.join(self.project_root, "input_files", "linelists")
        self.molecule_dir = os.path.join(self.linelist_root, "molecules")
        os.makedirs(self.molecule_dir, exist_ok=True)

        self.alpha = os.path.join(self.molecule_dir, "alpha.bsyn")
        self.beta = os.path.join(self.molecule_dir, "beta.bsyn")
        for path in (self.beta, self.alpha):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(os.path.basename(path))

        os.makedirs(os.path.join(self.molecule_dir, "nested"), exist_ok=True)
        with open(os.path.join(self.molecule_dir, ".DS_Store"), "w", encoding="utf-8") as handle:
            handle.write("ignore me")

    def test_resolve_linelist_paths_expands_glob_patterns(self) -> None:
        resolved = resolve_linelist_paths(self.linelist_root, ["molecules/*"])

        self.assertEqual(resolved, [self.alpha, self.beta])

    def test_resolve_linelist_paths_expands_directories(self) -> None:
        resolved = resolve_linelist_paths(self.linelist_root, ["molecules"])

        self.assertEqual(resolved, [self.alpha, self.beta])

    def test_resolve_linelist_paths_deduplicates_entries(self) -> None:
        resolved = resolve_linelist_paths(
            self.linelist_root,
            ["molecules", "molecules/*", self.alpha],
        )

        self.assertEqual(resolved, [self.alpha, self.beta])

    def test_create_linelist_file_writes_expanded_paths(self) -> None:
        tmp_dir = os.path.join(self.project_root, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        with contextlib.redirect_stdout(io.StringIO()):
            config = TurbospectrumConfig(
                project_root=self.project_root,
                linelist_path=self.linelist_root,
                linelist_files=["molecules/*"],
                tmp_dir=tmp_dir,
            )

        list_file = create_linelist_file(config)

        with open(list_file, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        self.assertEqual(lines, [self.alpha, self.beta])

    def test_validate_linelist_files_rejects_missing_paths(self) -> None:
        with self.assertRaises(LinelistValidationError) as exc_info:
            validate_linelist_files(self.linelist_root, ["missing_file.bsyn"])

        self.assertIn("file does not exist", str(exc_info.exception))

    def test_validate_linelist_files_rejects_corrupted_files(self) -> None:
        broken = os.path.join(self.molecule_dir, "broken.bsyn")
        with open(broken, "wb") as handle:
            handle.write(b"bad\x00payload")

        with self.assertRaises(LinelistValidationError) as exc_info:
            validate_linelist_files(self.linelist_root, ["molecules/broken.bsyn"])

        self.assertIn("appears corrupted", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
