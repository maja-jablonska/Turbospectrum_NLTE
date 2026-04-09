"""Tests for FailFastTracker in synthesis_validation."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from synthesis_validation import FailFastTracker


def test_disabled_when_threshold_zero():
    tracker = FailFastTracker(threshold=0)
    for _ in range(100):
        tracker.record_failure("error")
    assert not tracker.should_exit


def test_triggers_after_threshold_consecutive_failures():
    tracker = FailFastTracker(threshold=3)
    tracker.record_failure("err1")
    assert not tracker.should_exit
    tracker.record_failure("err2")
    assert not tracker.should_exit
    tracker.record_failure("err3")
    assert tracker.should_exit


def test_success_resets_consecutive_count():
    tracker = FailFastTracker(threshold=3)
    tracker.record_failure("err")
    tracker.record_failure("err")
    tracker.record_success()
    tracker.record_failure("err")
    tracker.record_failure("err")
    assert not tracker.should_exit


def test_stays_triggered_once_set():
    tracker = FailFastTracker(threshold=2)
    tracker.record_failure("err")
    tracker.record_failure("err")
    assert tracker.should_exit
    tracker.record_success()  # shouldn't un-trigger
    assert tracker.should_exit


def test_tracks_repeated_error_message():
    tracker = FailFastTracker(threshold=3)
    tracker.record_failure("numpy.dtype size changed")
    tracker.record_failure("numpy.dtype size changed")
    tracker.record_failure("numpy.dtype size changed")
    assert tracker.should_exit
    assert "numpy.dtype size changed" in tracker.summary()


def test_clears_repeated_msg_on_different_errors():
    tracker = FailFastTracker(threshold=3)
    tracker.record_failure("error A")
    tracker.record_failure("error B")
    tracker.record_failure("error C")
    assert tracker.should_exit
    # repeated_error_msg should be None since errors differ
    assert "repeated error" not in tracker.summary()


def test_total_counts():
    tracker = FailFastTracker(threshold=100)
    tracker.record_success()
    tracker.record_failure("x")
    tracker.record_success()
    tracker.record_failure("y")
    assert tracker.total_done == 4
    assert tracker.total_failures == 2
    assert tracker.consecutive_failures == 1


def test_summary_truncates_long_messages():
    tracker = FailFastTracker(threshold=1)
    long_msg = "x" * 300
    tracker.record_failure(long_msg)
    summary = tracker.summary()
    assert len(summary) < 350
    assert "..." in summary
