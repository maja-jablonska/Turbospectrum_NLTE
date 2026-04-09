"""Shared zarr v2/v3 compatibility helpers.

All scripts should import from this module instead of duplicating
version-detection logic.  The version is detected once via the major
version number, which is reliable across zarr 2.x and 3.x.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import zarr

ZARR_V3: bool = int(zarr.__version__.split(".")[0]) >= 3


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------

def zarr_store(path: str):
    """Return a filesystem-backed Zarr store for the installed version."""
    if ZARR_V3:
        from zarr import storage as zstorage  # type: ignore

        if hasattr(zstorage, "LocalStore"):
            return zstorage.LocalStore(path)  # type: ignore[attr-defined]
        if hasattr(zstorage, "DirectoryStore"):
            return zstorage.DirectoryStore(path)  # type: ignore[attr-defined]
        raise AttributeError("Zarr v3 has no LocalStore or DirectoryStore")

    # zarr v2
    if hasattr(zarr, "DirectoryStore"):
        return zarr.DirectoryStore(path)  # type: ignore[attr-defined]
    from zarr import storage as zstorage  # type: ignore

    return zstorage.DirectoryStore(path)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Group helpers
# ---------------------------------------------------------------------------

def create_root_group(store, *, overwrite: bool = True):
    """Create a new root group (write mode)."""
    if ZARR_V3:
        return zarr.group(store=store, overwrite=overwrite, zarr_format=3)
    return zarr.open_group(store=store, mode="w")


def open_root_group(store_or_path, *, mode: str = "r"):
    """Open an existing group."""
    if isinstance(store_or_path, str):
        store_or_path = zarr_store(store_or_path)
    return zarr.open_group(store=store_or_path, mode=mode)


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def compression_kwargs(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build compression keyword arguments from a config dict.

    Returns kwargs suitable for passing to ``create_array()`` (v3) or
    ``root.array()`` / ``root.create_dataset()`` (v2).
    """
    if not cfg:
        return {}

    cname = str(cfg.get("cname", "zstd"))
    clevel = int(cfg.get("clevel", 5))
    shuffle_enabled = bool(cfg.get("shuffle", True))

    if ZARR_V3:
        try:
            import zarr.codecs as zc  # type: ignore

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


# ---------------------------------------------------------------------------
# Array creation helpers
# ---------------------------------------------------------------------------

def create_array(group, name: str, *, data=None, shape=None, dtype=None,
                 chunks=None, **kwargs):
    """Create a numeric array in a zarr group (v2/v3 compatible)."""
    if ZARR_V3 and hasattr(group, "create_array"):
        kw = dict(kwargs)
        if data is not None:
            kw["data"] = data
        if shape is not None and data is None:
            kw["shape"] = shape
        if dtype is not None:
            kw["dtype"] = dtype
        if chunks is not None:
            kw["chunks"] = chunks
        return group.create_array(name, **kw)

    # zarr v2
    if data is not None:
        return group.array(name, data, chunks=chunks, dtype=dtype, **kwargs)
    return group.empty(name, shape=shape, dtype=dtype, chunks=chunks, **kwargs)


def write_string_array(group, name: str, values: Sequence[str], *,
                       chunks: int | None = None,
                       compression_kw: Mapping[str, Any] | None = None):
    """Write a variable-length UTF-8 string array (v2/v3 compatible)."""
    as_list = list(values)
    comp = dict(compression_kw or {})

    if ZARR_V3 and hasattr(group, "create_array"):
        try:
            import zarr.codecs as zc  # type: ignore
            from zarr.core.dtype.npy.string import VariableLengthUTF8  # type: ignore

            arr = group.create_array(
                name,
                shape=(len(as_list),),
                dtype=VariableLengthUTF8(),
                serializer=zc.VLenUTF8Codec(),
                chunks=int(chunks) if chunks else len(as_list),
                **comp,
            )
            arr[:] = as_list
            return arr
        except Exception:
            pass

    # zarr v2
    from numcodecs import VLenUTF8  # type: ignore

    return group.array(
        name,
        as_list,
        dtype=object,
        object_codec=VLenUTF8(),
        chunks=int(chunks) if chunks else len(as_list),
        **comp,
    )


def write_string_scalar(group, name: str, value: str, *,
                        compression_kw: Mapping[str, Any] | None = None):
    """Write a single string as a 1-element string array."""
    return write_string_array(group, name, [str(value)],
                              chunks=1, compression_kw=compression_kw)


def write_fixed_string_scalar(group, name: str, value: str, min_width: int = 32,
                              compression_kw: Mapping[str, Any] | None = None):
    """Write a fixed-width UTF-8 string scalar for cross-version readability."""
    text = str(value)
    width = max(min_width, len(text) + 8)
    padded = text.ljust(width)
    comp = dict(compression_kw or {})

    if ZARR_V3 and hasattr(group, "create_array"):
        try:
            arr = group.create_array(
                name,
                shape=(1,),
                dtype=f"U{width}",
                chunks=(1,),
                **comp,
            )
            arr[:] = [padded]
            return arr
        except Exception:
            pass

    # zarr v2
    return group.array(
        name,
        np.array([padded], dtype=f"U{width}"),
        chunks=(1,),
        **comp,
    )
