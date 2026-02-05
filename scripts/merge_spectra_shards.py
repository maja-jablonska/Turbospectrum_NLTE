#!/usr/bin/env python3
"""Merge sharded synthesized-spectra Zarr stores into one consolidated store.

This is intended to merge outputs produced by:
- `scripts/synthesize_spectra_from_zarr_sharded.py`

Each shard must contain:
- `wavelength` (1D)
- `global_index` (1D int, mapping shard rows -> original grid rows)
- `flux` (2D: shard_rows x wavelength_points)
- `continuum` (2D)
- `status` (1D)
- `message` (1D)

The merger writes a single output Zarr with shape:
- flux: (row_count, wavelength_points)
- continuum: (row_count, wavelength_points)
and fills missing rows with NaN / "missing".
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import zarr


def _zarr_store(path: str):
    if hasattr(zarr, "DirectoryStore"):
        return zarr.DirectoryStore(path)  # type: ignore[attr-defined]
    from zarr import storage as zstorage  # type: ignore

    if hasattr(zstorage, "DirectoryStore"):
        return zstorage.DirectoryStore(path)  # type: ignore[attr-defined]
    if hasattr(zstorage, "LocalStore"):
        return zstorage.LocalStore(path)  # type: ignore[attr-defined]
    raise AttributeError("Unsupported Zarr version: cannot find DirectoryStore/LocalStore")


def _zarr_compression_kwargs(zarr_compressor_cfg: Mapping[str, Any]):
    if not zarr_compressor_cfg:
        return {}

    cname = zarr_compressor_cfg.get("cname", "zstd")
    clevel = int(zarr_compressor_cfg.get("clevel", 5))
    shuffle_enabled = bool(zarr_compressor_cfg.get("shuffle", True))

    try:
        import zarr.codecs as zc  # type: ignore

        if hasattr(zc, "BloscCodec") and hasattr(zc, "BloscShuffle"):
            shuffle = zc.BloscShuffle.bitshuffle if shuffle_enabled else None
            return {"compressors": [zc.BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle)]}
    except Exception:
        pass

    from numcodecs import Blosc  # type: ignore

    return {
        "compressor": Blosc(
            cname=cname,
            clevel=clevel,
            shuffle=Blosc.BITSHUFFLE if shuffle_enabled else Blosc.NOSHUFFLE,
        )
    }


def _list_shards(shard_paths: Sequence[str] | None, shard_dir: str | None) -> List[str]:
    if shard_paths:
        return [os.path.abspath(p) for p in shard_paths]
    if not shard_dir:
        raise ValueError("Provide either --shard or --shard-dir")
    shard_dir = os.path.abspath(shard_dir)
    paths = sorted(str(p) for p in Path(shard_dir).glob("*.zarr"))
    if not paths:
        raise FileNotFoundError(f"No *.zarr shards found in {shard_dir}")
    return paths


def _open_shard(path: str):
    store = _zarr_store(path)
    return zarr.open_group(store=store, mode="r")


def _require_arrays(root, names: Sequence[str]) -> None:
    missing = [n for n in names if n not in root]
    if missing:
        raise KeyError(f"Shard {getattr(root.store, 'path', '?')} missing arrays: {missing}")


def _infer_row_count_from_grid(grid_zarr: str) -> int:
    store = _zarr_store(grid_zarr)
    root = zarr.open_group(store=store, mode="r")
    if "teff" not in root:
        raise KeyError("Grid Zarr missing required column 'teff'")
    return int(root["teff"].shape[0])


def _infer_row_count_from_shards(shards: Sequence[str]) -> int:
    max_idx = -1
    for p in shards:
        root = _open_shard(p)
        _require_arrays(root, ["global_index"])
        idx = np.asarray(root["global_index"][:], dtype=np.int64)
        if idx.size:
            max_idx = max(max_idx, int(idx.max()))
    return max_idx + 1


def _write_vlen_utf8(root, name: str, values: Sequence[str], chunks: int, compression_kwargs: Mapping[str, Any]):
    import zarr.codecs as zc  # type: ignore
    from zarr.core.dtype.npy.string import VariableLengthUTF8  # type: ignore

    arr = root.create_array(
        name,
        shape=(len(values),),
        dtype=VariableLengthUTF8(),
        serializer=zc.VLenUTF8Codec(),
        chunks=min(chunks, len(values)) if len(values) else 1,
        **compression_kwargs,
    )
    arr[:] = list(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-zarr", required=True, help="Consolidated output Zarr path")
    parser.add_argument("--grid-zarr", default=None, help="Optional original grid Zarr to determine total row_count")

    parser.add_argument("--shard", action="append", default=None, help="Shard Zarr path (repeatable)")
    parser.add_argument("--shard-dir", default=None, help="Directory containing shard *.zarr outputs")

    parser.add_argument("--chunk-rows", type=int, default=32, help="Chunking along the sample dimension")
    parser.add_argument("--compressor", default=None, help="JSON string describing compressor options (cname, clevel, shuffle)")
    args = parser.parse_args()

    shards = _list_shards(args.shard, args.shard_dir)
    out_path = os.path.abspath(args.output_zarr)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    compressor_cfg: Dict[str, Any] = {}
    if args.compressor:
        compressor_cfg = json.loads(args.compressor)
    compression_kwargs = _zarr_compression_kwargs(compressor_cfg)

    # Determine row_count.
    if args.grid_zarr:
        row_count = _infer_row_count_from_grid(os.path.abspath(args.grid_zarr))
    else:
        row_count = _infer_row_count_from_shards(shards)
    if row_count <= 0:
        raise ValueError("Could not infer a positive row_count; provide --grid-zarr or non-empty shards")

    # Read wavelength from first shard and validate all shards match.
    first = _open_shard(shards[0])
    _require_arrays(first, ["wavelength", "flux", "continuum", "global_index", "status", "message"])
    wavelengths = np.asarray(first["wavelength"][:], dtype=np.float64)
    wl_count = int(wavelengths.size)
    if wl_count <= 0:
        raise ValueError("Shard wavelength array is empty")

    for p in shards[1:]:
        root = _open_shard(p)
        _require_arrays(root, ["wavelength"])
        other = np.asarray(root["wavelength"][:], dtype=np.float64)
        if other.shape != wavelengths.shape or not np.allclose(other, wavelengths):
            raise ValueError(f"Wavelength mismatch between shards: {shards[0]} vs {p}")

    # Create output store and datasets.
    store = _zarr_store(out_path)
    root_out = zarr.group(store=store, overwrite=True, zarr_format=3)
    chunk_shape = (min(int(args.chunk_rows), row_count), wl_count)
    root_out.create_array("wavelength", data=wavelengths, chunks=wavelengths.shape, **compression_kwargs)
    flux_out = root_out.create_array("flux", shape=(row_count, wl_count), dtype=np.float32, chunks=chunk_shape, **compression_kwargs)
    cont_out = root_out.create_array("continuum", shape=(row_count, wl_count), dtype=np.float32, chunks=chunk_shape, **compression_kwargs)

    # Initialize to NaNs for missing rows.
    flux_out[:] = np.nan
    cont_out[:] = np.nan

    statuses = ["missing"] * row_count
    messages = [""] * row_count

    # Optional: merge any metadata columns present in shards (e.g. teff/logg/feh, etc.).
    # We'll lazily create arrays in output when we see them in the first shard.
    merged_columns: Dict[str, Any] = {}

    for p in shards:
        shard = _open_shard(p)
        _require_arrays(shard, ["global_index", "flux", "continuum", "status", "message"])
        gidx = np.asarray(shard["global_index"][:], dtype=np.int64)
        if gidx.size == 0:
            continue

        if int(gidx.max()) >= row_count:
            raise ValueError(f"Shard {p} contains global_index beyond row_count={row_count}")

        flux = np.asarray(shard["flux"][:], dtype=np.float32)
        cont = np.asarray(shard["continuum"][:], dtype=np.float32)
        if flux.shape != (gidx.size, wl_count) or cont.shape != (gidx.size, wl_count):
            raise ValueError(f"Shard {p} has unexpected flux/continuum shape: {flux.shape} / {cont.shape}")

        # Write shard rows into the consolidated arrays.
        # Use oindex when available for correct fancy indexing.
        try:
            flux_out.oindex[gidx, :] = flux  # type: ignore[attr-defined]
            cont_out.oindex[gidx, :] = cont  # type: ignore[attr-defined]
        except Exception:
            for i, gi in enumerate(gidx.tolist()):
                flux_out[int(gi), :] = flux[i]
                cont_out[int(gi), :] = cont[i]

        shard_status = [str(x) for x in np.asarray(shard["status"][:]).tolist()]
        shard_msg = [str(x) for x in np.asarray(shard["message"][:]).tolist()]
        for i, gi in enumerate(gidx.tolist()):
            statuses[int(gi)] = shard_status[i]
            messages[int(gi)] = shard_msg[i]

        # Merge metadata columns if present (best-effort).
        for name in shard.keys():
            if name in {"wavelength", "global_index", "flux", "continuum", "status", "message"}:
                continue
            arr = shard[name]
            if arr.shape[0] != gidx.size:
                continue
            # Create output array lazily.
            if name not in merged_columns:
                # Determine dtype handling.
                data0 = np.asarray(arr[:])
                if data0.dtype.kind in {"U", "S", "O"}:
                    _write_vlen_utf8(root_out, name, [""] * row_count, chunks=int(args.chunk_rows), compression_kwargs=compression_kwargs)
                    merged_columns[name] = root_out[name]
                else:
                    merged_columns[name] = root_out.create_array(
                        name,
                        shape=(row_count,),
                        dtype=data0.dtype,
                        chunks=min(int(args.chunk_rows), row_count),
                        **compression_kwargs,
                    )
                    merged_columns[name][:] = np.nan if np.issubdtype(data0.dtype, np.floating) else 0

            out_arr = merged_columns[name]
            values = np.asarray(arr[:])
            try:
                out_arr.oindex[gidx] = values  # type: ignore[attr-defined]
            except Exception:
                for i, gi in enumerate(gidx.tolist()):
                    out_arr[int(gi)] = values[i]

    _write_vlen_utf8(root_out, "status", statuses, chunks=int(args.chunk_rows), compression_kwargs=compression_kwargs)
    _write_vlen_utf8(root_out, "message", messages, chunks=int(args.chunk_rows), compression_kwargs=compression_kwargs)

    root_out.attrs.update(
        {
            "row_count": int(row_count),
            "wavelength_count": int(wl_count),
            "shards_merged": len(shards),
            "shards": [os.path.abspath(p) for p in shards],
        }
    )
    print(f"Wrote merged spectra Zarr: {out_path}")


if __name__ == "__main__":
    main()

