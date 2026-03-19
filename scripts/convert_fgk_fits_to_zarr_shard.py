#!/usr/bin/env python3
"""Convert FGK FITS spectra into Zarr shards (HPC/array-job friendly).

This script is a command-line version of the workflow in `read_fgk.ipynb`:

- Discover instruments under an input root directory
- Discover stars under each instrument directory
- Read each FITS file (expects `hdul[1].data` columns: WAVE, FLUX, ERR)
- Write spectra to a Zarr store, organized as:

  /spectra/<instrument>/<star>/<fits_filename>/{wavelength,flux,error}

and store per-file FITS headers under the same file group to avoid the
performance trap of continuously updating one giant `.attrs` mapping.

Parallelization model (recommended for HPC):

- Run as a SLURM array job or any array scheduler.
- Each task writes its OWN output Zarr directory (a "shard"), avoiding
  concurrent writes to the same Zarr store.
- Partitioning is deterministic: file_index % num_tasks == task_id.

Example (manual sharding):

  python scripts/convert_fgk_fits_to_zarr_shard.py \
    --input-root /path/to/fgk_spectra \
    --output fgk_spectra_shards/shard-0000.zarr \
    --task-id 0 --num-tasks 100

Example (SLURM array job):

  #SBATCH --array=0-99
  python scripts/convert_fgk_fits_to_zarr_shard.py \
    --input-root /path/to/fgk_spectra \
    --output fgk_spectra_shards/shard-${SLURM_ARRAY_TASK_ID}.zarr \
    --slurm-array --num-tasks 100

Notes:
- This script uses Zarr v3 when available, but falls back to v2 APIs when needed.
- Large headers are stored as JSON strings (`hdr0_json`, `hdr1_json`) per file group.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# NOTE: We deliberately do NOT import zarr at module import time so that
# `--help` works even when zarr isn't installed in the current environment.
try:  # noqa: TRY300
    import zarr  # type: ignore
except Exception:  # noqa: BLE001
    zarr = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FgkEntry:
    instrument: str
    star: str
    filename: str
    path: Path

    def relkey(self) -> str:
        # Stable human-readable identifier for logging.
        return f"{self.instrument}/{self.star}/{self.filename}"


def _configure_logging(log_level: str, log_file: Optional[str]) -> logging.Logger:
    logger = logging.getLogger("fgk_fits_to_zarr")
    if logger.handlers:
        return logger

    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def _progress(iterable, total: int | None, desc: str, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _ensure_astropy_available() -> None:
    try:
        import astropy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "This script requires astropy on the execution node. "
            "Install with `pip install astropy` in the job environment."
        ) from exc


def _ensure_zarr_available() -> None:
    if zarr is not None:
        return
    raise ImportError(
        "This script requires zarr on the execution node. "
        "Install with `pip install zarr numcodecs` in the job environment."
    )


def _zarr_store(path: str):
    # Match the helper used in other scripts in this repo.
    _ensure_zarr_available()
    if hasattr(zarr, "DirectoryStore"):
        return zarr.DirectoryStore(path)  # type: ignore[attr-defined]
    from zarr import storage as zstorage  # type: ignore

    if hasattr(zstorage, "DirectoryStore"):
        return zstorage.DirectoryStore(path)  # type: ignore[attr-defined]
    if hasattr(zstorage, "LocalStore"):
        return zstorage.LocalStore(path)  # type: ignore[attr-defined]
    raise AttributeError("Unsupported Zarr version: cannot find DirectoryStore/LocalStore")


def _zarr_compression_kwargs(cname: str, clevel: int, shuffle: bool) -> dict:
    # Prefer v3 codecs when available; otherwise fall back to numcodecs (v2).
    _ensure_zarr_available()
    try:
        import zarr.codecs as zc  # type: ignore

        if hasattr(zc, "BloscCodec") and hasattr(zc, "BloscShuffle"):
            shuf = zc.BloscShuffle.bitshuffle if shuffle else None
            return {"compressors": [zc.BloscCodec(cname=cname, clevel=clevel, shuffle=shuf)]}
    except Exception:
        pass

    try:
        from numcodecs import Blosc  # type: ignore

        return {
            "compressor": Blosc(
                cname=cname,
                clevel=int(clevel),
                shuffle=Blosc.BITSHUFFLE if shuffle else Blosc.NOSHUFFLE,
            )
        }
    except Exception:
        return {}


def _string_array_write(group, name: str, value: str, *, compression_kwargs: dict) -> None:
    """
    Store a scalar string in a Zarr group, working for both Zarr v2 and v3.
    """
    # v3
    if hasattr(group, "create_array"):
        try:
            import zarr.codecs as zc  # type: ignore
            from zarr.core.dtype.npy.string import VariableLengthUTF8  # type: ignore

            arr = group.create_array(
                name,
                shape=(),
                dtype=VariableLengthUTF8(),
                serializer=zc.VLenUTF8Codec(),
                **compression_kwargs,
            )
            arr[()] = value
            return
        except Exception:
            # Fall through to v2-style storage if v3 string dtype isn't available.
            pass

    # v2 fallback
    try:
        from numcodecs import VLenUTF8  # type: ignore

        group.array(name, np.array(value, dtype=object), dtype=object, object_codec=VLenUTF8(), **compression_kwargs)
    except Exception:
        # Last resort: store as bytes-ish object
        group.array(name, np.array(value, dtype=object), dtype=object)


def header_to_serializable_dict(header) -> dict:
    # Converts astropy.io.fits.Header to vanilla types; mirrors the notebook approach.
    d: dict = {}
    for k, v in header.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            d[k] = v
        else:
            try:
                d[k] = str(v)
            except Exception:
                d[k] = None
    return d


def discover_entries(input_root: Path, *, instruments: Optional[Sequence[str]] = None) -> List[FgkEntry]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    inst_names: List[str]
    if instruments:
        inst_names = list(instruments)
    else:
        inst_names = sorted(p.name for p in input_root.iterdir() if p.is_dir())

    entries: List[FgkEntry] = []
    for inst in inst_names:
        inst_dir = input_root / inst
        if not inst_dir.exists():
            continue
        for star_dir in sorted(p for p in inst_dir.iterdir() if p.is_dir()):
            star = star_dir.name
            for fits_path in sorted(star_dir.iterdir()):
                if not fits_path.is_file():
                    continue
                if not fits_path.name.lower().endswith(".fits"):
                    continue
                entries.append(FgkEntry(inst, star, fits_path.name, fits_path))

    # Stable ordering for deterministic sharding.
    entries.sort(key=lambda e: (e.instrument, e.star, e.filename))
    return entries


def _resolve_shard(task_id: Optional[int], num_tasks: Optional[int], slurm_array: bool) -> Tuple[int, int]:
    if slurm_array:
        env_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_task is not None:
            task_id = int(env_task)
        # Some clusters use 1-based arrays; use *_MIN when provided.
        env_min = os.environ.get("SLURM_ARRAY_TASK_MIN")
        if env_min is not None and task_id is not None:
            task_id = int(task_id) - int(env_min)

        if num_tasks is None:
            env_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
            if env_count is not None:
                num_tasks = int(env_count)
            else:
                env_max = os.environ.get("SLURM_ARRAY_TASK_MAX")
                if env_max is not None and env_min is not None:
                    num_tasks = int(env_max) - int(env_min) + 1

    if task_id is None or num_tasks is None:
        raise ValueError("Must provide --task-id and --num-tasks (or use --slurm-array with a valid array env).")
    if num_tasks <= 0:
        raise ValueError("--num-tasks must be positive")
    if not (0 <= task_id < num_tasks):
        raise ValueError(f"--task-id must be in [0, {num_tasks - 1}], got {task_id}")
    return task_id, num_tasks


def _iter_shard_indices(n: int, task_id: int, num_tasks: int) -> Iterator[int]:
    for i in range(n):
        if (i % num_tasks) == task_id:
            yield i


def _read_fgk_fits(fits_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    from astropy.io import fits  # local import for faster startup when help is requested

    with fits.open(fits_path, memmap=True) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"Expected at least 2 HDUs in {fits_path}")
        data = hdul[1].data
        if data is None:
            raise ValueError(f"HDU[1].data is None in {fits_path}")

        wave = np.array(data["WAVE"]).reshape(-1)
        flux = np.array(data["FLUX"]).reshape(-1)
        err = np.array(data["ERR"]).reshape(-1)

        hdr0 = json.dumps(header_to_serializable_dict(hdul[0].header), ensure_ascii=False)
        hdr1 = json.dumps(header_to_serializable_dict(hdul[1].header), ensure_ascii=False)

    if wave.shape != flux.shape or wave.shape != err.shape:
        raise ValueError(f"Shape mismatch in {fits_path}: wave={wave.shape} flux={flux.shape} err={err.shape}")

    # Keep wavelength high precision; flux/error typically ok in float32.
    wave = wave.astype(np.float64, copy=False)
    flux = flux.astype(np.float32, copy=False)
    err = err.astype(np.float32, copy=False)

    return wave, flux, err, hdr0, hdr1


def run_shard(
    *,
    input_root: Path,
    output: Path,
    instruments: Optional[Sequence[str]],
    task_id: int,
    num_tasks: int,
    chunk_size: int,
    compressor: str,
    compression_level: int,
    shuffle: bool,
    skip_existing: bool,
    progress: bool,
    logger: logging.Logger,
) -> None:
    _ensure_zarr_available()
    entries = discover_entries(input_root, instruments=instruments)
    n_total = len(entries)
    shard_indices = list(_iter_shard_indices(n_total, task_id, num_tasks))

    logger.info("Discovered %d FITS files under %s", n_total, input_root)
    logger.info("Shard: task_id=%d num_tasks=%d files=%d", task_id, num_tasks, len(shard_indices))

    os.makedirs(os.path.dirname(os.path.abspath(str(output))), exist_ok=True)
    store = _zarr_store(str(output))
    root = zarr.group(store=store, overwrite=True, zarr_format=3) if hasattr(zarr, "group") else zarr.open_group(store=store, mode="w")  # type: ignore

    compression_kwargs = _zarr_compression_kwargs(compressor, compression_level, shuffle)
    spectra_group = root.create_group("spectra") if hasattr(root, "create_group") else root.require_group("spectra")  # type: ignore

    root.attrs["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    root.attrs["input_root"] = str(input_root)
    root.attrs["task_id"] = int(task_id)
    root.attrs["num_tasks"] = int(num_tasks)
    root.attrs["chunk_size"] = int(chunk_size)

    ok = 0
    failed = 0

    for idx in _progress(shard_indices, total=len(shard_indices), desc="FITS→Zarr", enabled=progress):
        entry = entries[idx]
        file_group_path = f"{entry.instrument}/{entry.star}/{entry.filename}"

        try:
            inst_group = spectra_group.require_group(entry.instrument) if hasattr(spectra_group, "require_group") else spectra_group.create_group(entry.instrument)
            star_group = inst_group.require_group(entry.star) if hasattr(inst_group, "require_group") else inst_group.create_group(entry.star)

            if skip_existing and (entry.filename in getattr(star_group, "keys", lambda: [])()):  # type: ignore[misc]
                ok += 1
                continue

            wave, flux, err, hdr0_json, hdr1_json = _read_fgk_fits(entry.path)
            dset_group = star_group.create_group(entry.filename)

            cs = int(chunk_size) if chunk_size > 0 else wave.size
            cs = min(cs, wave.size)
            chunks = (cs,)

            # Store numeric arrays
            if hasattr(dset_group, "create_array"):
                dset_group.create_array("wavelength", data=wave, chunks=chunks, **compression_kwargs)
                dset_group.create_array("flux", data=flux, chunks=chunks, **compression_kwargs)
                dset_group.create_array("error", data=err, chunks=chunks, **compression_kwargs)
            else:
                dset_group.array("wavelength", wave, chunks=chunks, **compression_kwargs)
                dset_group.array("flux", flux, chunks=chunks, **compression_kwargs)
                dset_group.array("error", err, chunks=chunks, **compression_kwargs)

            # Store headers as per-file JSON (fast; avoids giant attrs updates)
            _string_array_write(dset_group, "hdr0_json", hdr0_json, compression_kwargs=compression_kwargs)
            _string_array_write(dset_group, "hdr1_json", hdr1_json, compression_kwargs=compression_kwargs)

            dset_group.attrs["source_path"] = str(entry.path)
            dset_group.attrs["relkey"] = entry.relkey()
            dset_group.attrs["index"] = int(idx)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.error("Failed: %s (%s)", file_group_path, exc)

    root.attrs["ok"] = int(ok)
    root.attrs["failed"] = int(failed)
    logger.info("Finished shard: ok=%d failed=%d output=%s", ok, failed, os.path.abspath(str(output)))


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path, help="Root directory containing instrument subdirectories")
    parser.add_argument("--output", required=True, type=Path, help="Output Zarr shard directory path")
    parser.add_argument(
        "--instrument",
        action="append",
        default=None,
        help="Restrict to a specific instrument (repeatable). Default: discover all instruments.",
    )

    parser.add_argument("--task-id", type=int, default=None, help="Shard id (0-based)")
    parser.add_argument("--num-tasks", type=int, default=None, help="Total number of shards/tasks")
    parser.add_argument("--slurm-array", action="store_true", help="Derive task-id from SLURM_ARRAY_TASK_ID")

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2**14,
        help="Chunk length for 1D arrays (default 16384). Use 0 to chunk as full length.",
    )
    parser.add_argument("--compressor", default="zstd", help="Blosc compressor name (default zstd)")
    parser.add_argument("--compression-level", type=int, default=5, help="Compression level (default 5)")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable bitshuffle (default on)")

    parser.add_argument("--skip-existing", action="store_true", help="Skip writing groups already present in this shard")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    args = parser.parse_args(argv)
    logger = _configure_logging(args.log_level, args.log_file)
    _ensure_astropy_available()
    _ensure_zarr_available()

    task_id, num_tasks = _resolve_shard(args.task_id, args.num_tasks, args.slurm_array)
    shuffle = not bool(args.no_shuffle)

    t0 = time.perf_counter()
    run_shard(
        input_root=args.input_root,
        output=args.output,
        instruments=args.instrument,
        task_id=task_id,
        num_tasks=num_tasks,
        chunk_size=int(args.chunk_size),
        compressor=str(args.compressor),
        compression_level=int(args.compression_level),
        shuffle=shuffle,
        skip_existing=bool(args.skip_existing),
        progress=not bool(args.no_progress),
        logger=logger,
    )
    logger.info("Total wall time: %.2fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()

