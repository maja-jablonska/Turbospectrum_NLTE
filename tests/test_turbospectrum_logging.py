import os
import tempfile
import unittest

from scripts.run_turbospectrum import (
    _extract_turbospectrum_log_excerpt,
    _with_turbospectrum_log_context,
)


class TurbospectrumLoggingTests(unittest.TestCase):
    def test_extract_turbospectrum_log_excerpt_prefers_error_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "failed.log")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("Starting synthesis\n")
                handle.write("Some setup message\n")
                handle.write("STOP in bsyn: line list missing\n")
                handle.write("forrtl: severe (24): end-of-file during read\n")
                handle.write("Last line\n")

            excerpt = _extract_turbospectrum_log_excerpt(log_path)

            self.assertIn("STOP in bsyn: line list missing", excerpt)
            self.assertIn("forrtl: severe (24): end-of-file during read", excerpt)
            self.assertNotIn("Starting synthesis", excerpt)

    def test_with_turbospectrum_log_context_falls_back_to_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "tail.log")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("line 1\n")
                handle.write("line 2\n")
                handle.write("line 3\n")

            message = _with_turbospectrum_log_context("bsyn failed (rc=1)", log_path)

            self.assertIn("bsyn failed (rc=1)", message)
            self.assertIn(f"log={log_path}", message)
            self.assertIn("excerpt=line 1 | line 2 | line 3", message)


if __name__ == "__main__":
    unittest.main()
