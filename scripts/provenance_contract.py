from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping, Sequence

import numpy as np

REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "config_hash",
    "grid_definition_hash",
    "git_commit",
    "turbospectrum_version",
    "linelist_identifier",
    "linelist_version",
    "atmosphere_model_identifier",
    "synthesis_timestamp",
    "pipeline_version",
)

MISSING_VALUE_TOKENS: set[str] = {
    "",
    "unknown",
    "not_recorded",
    "none",
    "null",
    "n/a",
    "na",
    "<unset>",
}


def is_meaningful_provenance_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in MISSING_VALUE_TOKENS


def canonical_json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_manifest_sha256(
    root_path: str,
    *,
    suffixes: Sequence[str] | None = None,
    hash_contents: bool = False,
) -> str:
    if not os.path.isdir(root_path):
        return ""

    suffixes_norm = tuple(str(s).lower() for s in (suffixes or ()))
    entries: list[dict[str, Any]] = []

    for dirpath, _dirnames, filenames in os.walk(root_path):
        for fname in sorted(filenames):
            if suffixes_norm and not fname.lower().endswith(suffixes_norm):
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root_path).replace(os.sep, "/")
            entry: dict[str, Any] = {
                "path": rel,
                "size_bytes": int(os.path.getsize(fpath)),
            }
            if hash_contents:
                entry["sha256"] = file_sha256(fpath)
            entries.append(entry)

    if not entries:
        return ""
    entries.sort(key=lambda item: str(item.get("path", "")))
    return canonical_json_sha256(entries)


def stable_numeric_digest(values: np.ndarray, dtype: str) -> str:
    arr = np.asarray(values).astype(dtype, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=9.96921e36, posinf=3.4e38, neginf=-3.4e38)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def stable_string_digest(values: Sequence[Any]) -> str:
    as_text = [str(v) for v in values]
    return canonical_json_sha256(as_text)


def compute_grid_definition_hash(column_data: Mapping[str, np.ndarray]) -> str:
    payload: dict[str, Any] = {"columns": {}}
    for name in sorted(column_data.keys()):
        arr = np.asarray(column_data[name])
        summary: dict[str, Any] = {
            "dtype": str(arr.dtype),
            "shape": [int(x) for x in arr.shape],
        }
        if np.issubdtype(arr.dtype, np.floating):
            summary["digest"] = stable_numeric_digest(arr, dtype="<f8")
        elif np.issubdtype(arr.dtype, np.integer):
            summary["digest"] = stable_numeric_digest(arr, dtype="<i8")
        elif np.issubdtype(arr.dtype, np.bool_):
            summary["digest"] = stable_numeric_digest(arr.astype(np.int8, copy=False), dtype="<i1")
        else:
            summary["digest"] = stable_string_digest(np.asarray(arr).tolist())
        payload["columns"][str(name)] = summary
    payload["column_count"] = len(payload["columns"])
    return canonical_json_sha256(payload)


def compute_binary_manifest_hash(binary_paths: Sequence[str]) -> str:
    entries: list[dict[str, Any]] = []
    for raw_path in sorted({os.path.abspath(str(p)) for p in binary_paths if str(p).strip()}):
        if not os.path.isfile(raw_path):
            continue
        entries.append(
            {
                "path": raw_path,
                "size_bytes": int(os.path.getsize(raw_path)),
                "sha256": file_sha256(raw_path),
            }
        )
    if not entries:
        return ""
    return canonical_json_sha256(entries)


def assert_required_provenance_fields(fields: Mapping[str, Any], *, context: str) -> None:
    missing = [
        key
        for key in REQUIRED_PROVENANCE_FIELDS
        if not is_meaningful_provenance_value(fields.get(key))
    ]
    if missing:
        details = ", ".join(f"{key}={fields.get(key)!r}" for key in missing)
        raise ValueError(f"{context}: missing required provenance fields: {details}")
