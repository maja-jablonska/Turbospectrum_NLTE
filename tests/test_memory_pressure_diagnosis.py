"""Tests for diagnose_memory_pressure in synthesis_validation.

These guard the operator-facing signal that distinguishes an out-of-memory /
OOM-kill failure (a resource problem, fixed by fewer workers or more mem) from a
genuine bad-parameter failure. The fixtures use signatures taken verbatim from a
real Gadi shard log where 32 workers shared 64 GB and Turbospectrum was killed
mid-write.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from synthesis_validation import diagnose_memory_pressure


def test_explicit_oom_is_flagged():
    d = diagnose_memory_pressure(
        ["error"],
        ["Fortran runtime error: Cannot allocate memory | Error termination."],
    )
    assert d["suspected"] is True
    assert d["explicit_oom"] == 1
    assert "MEMORY PRESSURE LIKELY" in d["log_message"]


def test_killed_mid_write_signatures_are_flagged():
    # Empty + truncated outputs: the writer died before flushing the full grid.
    statuses = ["success", "error", "error", "error"]
    messages = [
        "",
        "Failed to read spectrum p5535...: Spectrum file is empty (0 bytes)",
        "Failed to read spectrum p4635...: Unexpected wavelength count 39405 (expected 313001)",
        "Failed to read spectrum p6332...: could not convert string '********' to float64 at row 23086, column 5.",
    ]
    d = diagnose_memory_pressure(statuses, messages)
    assert d["suspected"] is True
    assert d["truncated_or_empty_output"] == 3
    assert d["explicit_oom"] == 0


def test_real_shard_failure_mix_is_flagged():
    # The exact six-failure mix from shard_2 / shard_20 of the reported run.
    statuses = ["error"] * 6
    messages = [
        "Unexpected wavelength count 39405 (expected 313001)",
        "Spectrum file is empty (0 bytes)",
        "Spectrum file is empty (0 bytes)",
        "could not convert string '********' to float64 at row 23086, column 5.",
        "Fortran runtime error: Cannot allocate memory | Error termination.",
        "Fortran runtime error: Bad real number in item 6 of list input | Error termination.",
    ]
    d = diagnose_memory_pressure(statuses, messages)
    assert d["suspected"] is True
    assert d["n_failed"] == 6
    assert d["explicit_oom"] == 1
    assert d["truncated_or_empty_output"] == 5
    assert d["other"] == 0


def test_genuine_data_error_does_not_trip():
    d = diagnose_memory_pressure(
        ["error"],
        ["Failed to read spectrum: LinelistValidationError: missing species block"],
    )
    assert d["suspected"] is False
    assert d["other"] == 1
    assert d["log_message"] == ""


def test_minority_truncation_does_not_trip():
    # One truncation among several genuine data errors: not a memory verdict.
    statuses = ["error"] * 4
    messages = [
        "Spectrum file is empty (0 bytes)",
        "LinelistValidationError: bad block",
        "ValueError: negative temperature in atmosphere",
        "KeyError: 'feh'",
    ]
    d = diagnose_memory_pressure(statuses, messages)
    assert d["suspected"] is False
    assert d["truncated_or_empty_output"] == 1
    assert d["other"] == 3


def test_no_failures_returns_not_suspected():
    d = diagnose_memory_pressure(["success", "skipped"], ["", ""])
    assert d["suspected"] is False
    assert d["n_failed"] == 0
    assert d["log_message"] == ""
