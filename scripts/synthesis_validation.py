from __future__ import annotations

import json
from typing import Dict, List, Sequence

import numpy as np


SUCCESS_STATUSES = {"success", "skipped"}


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
) -> Dict[str, int]:
    """Reject silently broken runs before they are written to disk."""
    counts = _status_counts(statuses)

    bad_status_rows = np.asarray(
        [idx for idx, status in enumerate(statuses) if str(status).lower() not in SUCCESS_STATUSES],
        dtype=np.int64,
    )
    invalid_flux_rows = np.where(~np.all(np.isfinite(fluxes), axis=1))[0].astype(np.int64, copy=False)
    invalid_cont_rows = np.where(~np.any(np.isfinite(continua), axis=1))[0].astype(np.int64, copy=False)

    if bad_status_rows.size == 0 and invalid_flux_rows.size == 0 and invalid_cont_rows.size == 0:
        return counts

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

    raise RuntimeError("Synthesis produced invalid spectra; aborting write. " + " | ".join(details))
