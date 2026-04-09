from __future__ import annotations

import json
import logging
from typing import Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SUCCESS_STATUSES = {"success", "skipped"}


class FailFastTracker:
    """Track consecutive failures and trigger early exit when systematic.

    Parameters
    ----------
    threshold:
        Number of consecutive failures that triggers early exit.
        Set to 0 to disable (``should_exit`` always returns ``False``).
    """

    def __init__(self, threshold: int = 10) -> None:
        self.threshold = int(threshold)
        self.consecutive_failures = 0
        self.total_done = 0
        self.total_failures = 0
        self._last_error_msg = ""
        self._repeated_error_msg: str | None = None
        self._triggered = False

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.total_done += 1

    def record_failure(self, error_msg: str = "") -> None:
        self.total_done += 1
        self.total_failures += 1
        if self.consecutive_failures == 0:
            self._repeated_error_msg = error_msg
        elif self._repeated_error_msg is not None and error_msg != self._repeated_error_msg:
            self._repeated_error_msg = None  # errors differ
        self._last_error_msg = error_msg
        self.consecutive_failures += 1

    @property
    def should_exit(self) -> bool:
        if self.threshold <= 0:
            return False
        if self._triggered:
            return True
        if self.consecutive_failures >= self.threshold:
            self._triggered = True
        return self._triggered

    def summary(self) -> str:
        """Human-readable summary for log messages."""
        parts = [
            f"{self.consecutive_failures} consecutive failures",
            f"({self.total_failures}/{self.total_done} total)",
        ]
        if self._repeated_error_msg is not None and self._repeated_error_msg:
            msg = self._repeated_error_msg
            if len(msg) > 200:
                msg = msg[:197] + "..."
            parts.append(f"repeated error: {msg}")
        return "; ".join(parts)


def _status_counts(statuses: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for status in statuses:
        key = str(status)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _format_row_samples(
    indices: np.ndarray,
    statuses: Sequence[str],
    messages: Sequence[str],
    *,
    label: str,
    limit: int = 5,
) -> str:
    if indices.size == 0:
        return ""

    parts: List[str] = []
    for idx in indices[:limit].tolist():
        message = str(messages[int(idx)]).strip()
        if len(message) > 160:
            message = f"{message[:157]}..."
        parts.append(f"row={int(idx)} status={statuses[int(idx)]} msg={message or '<none>'}")

    suffix = "" if indices.size <= limit else f" (+{int(indices.size - limit)} more)"
    return f"{label}: " + "; ".join(parts) + suffix


def validate_synthesis_results(
    *,
    statuses: Sequence[str],
    messages: Sequence[str],
    fluxes: np.ndarray,
    continua: np.ndarray,
    max_failures: int = 0,
) -> Dict[str, int]:
    """Check synthesis results and return status counts.

    Always returns so that callers can write output before deciding
    whether to abort.  When the number of bad rows exceeds
    *max_failures*, the function logs an **error**; otherwise it logs a
    **warning** (or nothing if there are no problems at all).

    Callers should inspect the returned *counts* dict to decide whether
    to exit with a non-zero status after the shard has been saved.

    Parameters
    ----------
    max_failures:
        Maximum number of bad rows to tolerate.  When the total number of
        unique bad rows (union of status/flux/continuum checks) is at most
        this value, the run is considered acceptable.
        The default (0) means any failure is flagged as an error.
    """
    counts = _status_counts(statuses)

    bad_status_rows = np.asarray(
        [idx for idx, status in enumerate(statuses) if str(status).lower() not in SUCCESS_STATUSES],
        dtype=np.int64,
    )
    invalid_flux_rows = np.where(~np.all(np.isfinite(fluxes), axis=1))[0].astype(np.int64, copy=False)
    invalid_cont_rows = np.where(~np.any(np.isfinite(continua), axis=1))[0].astype(np.int64, copy=False)

    if bad_status_rows.size == 0 and invalid_flux_rows.size == 0 and invalid_cont_rows.size == 0:
        return counts

    # Deduplicate across all three checks.
    all_bad = np.union1d(bad_status_rows, np.union1d(invalid_flux_rows, invalid_cont_rows))
    n_bad = int(all_bad.size)

    details = [f"status_counts={json.dumps(counts, sort_keys=True)}"]
    if bad_status_rows.size:
        details.append(
            _format_row_samples(
                bad_status_rows,
                statuses,
                messages,
                label=f"failed rows={int(bad_status_rows.size)}",
            )
        )
    if invalid_flux_rows.size:
        details.append(
            _format_row_samples(
                invalid_flux_rows,
                statuses,
                messages,
                label=f"rows with non-finite flux={int(invalid_flux_rows.size)}",
            )
        )
    if invalid_cont_rows.size:
        details.append(
            _format_row_samples(
                invalid_cont_rows,
                statuses,
                messages,
                label=f"rows with no finite continuum={int(invalid_cont_rows.size)}",
            )
        )

    summary = " | ".join(details)

    if n_bad <= max_failures:
        logger.warning(
            "Synthesis produced %d invalid row(s) (within --max-failures %d). %s",
            n_bad,
            max_failures,
            summary,
        )
    else:
        logger.error(
            "Synthesis produced %d invalid row(s) (exceeds --max-failures %d). %s",
            n_bad,
            max_failures,
            summary,
        )

    return counts
