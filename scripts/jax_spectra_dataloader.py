#!/usr/bin/env python3
"""JAX dataloader utilities for Zarr-based spectral ML datasets.

This module targets the synthesis layout written by:
- scripts/synthesize_spectra_from_zarr.py
- scripts/merge_spectra_shards.py

It can also read simpler stores that at least expose `flux` (or another chosen
input array) with row-major sample indexing.

Example
-------
from scripts.jax_spectra_dataloader import create_jax_spectra_dataloaders

loaders = create_jax_spectra_dataloaders(
    zarr_path="runs/local-dev/outputs/zarr/synthesized_spectra.zarr",
    batch_size=64,
    input_key="params",
    target_key="flux",
    normalize_inputs=True,
)

for batch in loaders["train"]:
    x = batch["inputs"]
    y = batch["targets"]
    ...
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Dict, Iterator, Mapping, Sequence

import numpy as np
import zarr

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "jax_spectra_dataloader requires JAX. Install with `pip install \"jax[cpu]\"` "
        "or the accelerator-specific JAX build."
    ) from exc


MIN_STD = 1.0e-6


try:
    from zarr_compat import zarr_store as _zarr_store
except ImportError:
    try:
        from .zarr_compat import zarr_store as _zarr_store
    except ImportError:
        def _zarr_store(path: str):
            if hasattr(zarr, "DirectoryStore"):
                return zarr.DirectoryStore(path)
            from zarr import storage as zstorage
            if hasattr(zstorage, "LocalStore"):
                return zstorage.LocalStore(path)
            return zstorage.DirectoryStore(path)


def _as_1d_int64(values: Sequence[int] | np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.int64)
    if out.ndim != 1:
        raise ValueError(f"Expected 1D indices, got shape {out.shape}")
    return out


def _take_1d(array, row_ids: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(array.oindex[row_ids])  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return np.asarray(array[row_ids])
    except Exception:
        return np.asarray([array[int(i)] for i in row_ids], dtype=array.dtype)


def _take_rows(array, row_ids: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(array.oindex[row_ids, :])  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return np.asarray(array[row_ids, :])
    except Exception:
        rows = [np.asarray(array[int(i), :]) for i in row_ids]
        return np.stack(rows, axis=0) if rows else np.empty((0, int(array.shape[1])), dtype=array.dtype)


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    out: list[str] = []
    for item in np.asarray(values).tolist():
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return tuple(out)


@dataclasses.dataclass(frozen=True)
class StandardizationStats:
    mean: np.ndarray
    std: np.ndarray


class SpectraZarrDataset:
    """Thin random-access wrapper around a Zarr spectral dataset."""

    def __init__(self, zarr_path: str):
        self.zarr_path = zarr_path
        store = _zarr_store(zarr_path)
        self.root = zarr.open_group(store=store, mode="r")
        self.array_names = tuple(sorted(self.root.array_keys()))
        if not self.array_names:
            raise ValueError(f"No arrays found at {zarr_path}")

        self.param_names: tuple[str, ...] = ()
        if "param_names" in self.root:
            self.param_names = _decode_strings(np.asarray(self.root["param_names"][:]))

        self.row_count = self._infer_row_count()
        if self.row_count <= 0:
            raise ValueError(f"Dataset has no rows: {zarr_path}")

    def _infer_row_count(self) -> int:
        preferred = ("flux", "params", "global_index", "model_id")
        for name in preferred:
            if name in self.root:
                arr = self.root[name]
                if arr.ndim >= 1:
                    return int(arr.shape[0])

        for name in self.array_names:
            if name in {"wavelength", "param_names"}:
                continue
            arr = self.root[name]
            if arr.ndim >= 1:
                return int(arr.shape[0])
        return 0

    def has_array(self, name: str) -> bool:
        return name in self.root

    def shape(self, name: str) -> tuple[int, ...]:
        if name not in self.root:
            raise KeyError(f"Array '{name}' not found in {self.zarr_path}")
        return tuple(int(x) for x in self.root[name].shape)

    def resolve_param_columns(self, requested: Sequence[str] | None) -> np.ndarray | None:
        if not requested:
            return None
        if "params" not in self.root:
            raise KeyError("Requested parameter columns but dataset has no 'params' array")
        if not self.param_names:
            raise ValueError("Dataset has 'params' but no 'param_names'; cannot resolve names")

        name_to_index = {name: idx for idx, name in enumerate(self.param_names)}
        missing = [name for name in requested if name not in name_to_index]
        if missing:
            raise KeyError(f"Unknown parameter name(s): {missing}. Available: {list(self.param_names)}")
        return np.asarray([name_to_index[name] for name in requested], dtype=np.int64)

    def get_rows(
        self,
        name: str,
        row_ids: Sequence[int] | np.ndarray,
        column_ids: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        if name not in self.root:
            raise KeyError(f"Array '{name}' not found in {self.zarr_path}")

        arr = self.root[name]
        ridx = _as_1d_int64(row_ids)
        if arr.ndim == 0:
            raise ValueError(f"Array '{name}' is scalar and cannot be indexed by row")
        if arr.ndim == 1:
            values = _take_1d(arr, ridx)
        else:
            values = _take_rows(arr, ridx)
            if column_ids is not None:
                cidx = _as_1d_int64(column_ids)
                values = values[:, cidx]
        return np.asarray(values)


def _infer_io_keys(dataset: SpectraZarrDataset, input_key: str | None, target_key: str | None) -> tuple[str, str | None]:
    if input_key is None:
        input_key = "params" if dataset.has_array("params") else "flux"
    if not dataset.has_array(input_key):
        raise KeyError(f"input_key='{input_key}' is not present. Available arrays: {list(dataset.array_names)}")

    if target_key == "auto":
        if input_key == "params" and dataset.has_array("flux"):
            target_key = "flux"
        elif input_key == "flux" and dataset.has_array("params"):
            target_key = "params"
        else:
            target_key = None
    if target_key is not None and not dataset.has_array(target_key):
        raise KeyError(f"target_key='{target_key}' is not present. Available arrays: {list(dataset.array_names)}")
    return input_key, target_key


def _split_indices(row_count: int, train_fraction: float, val_fraction: float, seed: int) -> Dict[str, np.ndarray]:
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(row_count).astype(np.int64)

    n_train = int(np.floor(row_count * train_fraction))
    n_train = max(1, min(n_train, row_count))
    n_val = int(np.floor(row_count * val_fraction))
    n_val = min(max(0, n_val), row_count - n_train)

    i_train = perm[:n_train]
    i_val = perm[n_train : n_train + n_val]
    i_test = perm[n_train + n_val :]
    return {"train": i_train, "val": i_val, "test": i_test}


def _compute_standardization(
    dataset: SpectraZarrDataset,
    array_name: str,
    row_ids: np.ndarray,
    column_ids: np.ndarray | None,
    block_rows: int,
) -> StandardizationStats:
    if row_ids.size == 0:
        raise ValueError("Cannot compute normalization stats on an empty split")
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")

    sum_vec: np.ndarray | None = None
    sumsq_vec: np.ndarray | None = None
    count_vec: np.ndarray | None = None

    for start in range(0, row_ids.size, block_rows):
        block_ids = row_ids[start : start + block_rows]
        block = dataset.get_rows(array_name, block_ids, column_ids=column_ids).astype(np.float64, copy=False)
        if block.ndim == 1:
            block = block[:, None]
        finite = np.isfinite(block)
        clean = np.where(finite, block, 0.0)
        block_sum = clean.sum(axis=0)
        block_sumsq = np.square(clean).sum(axis=0)
        block_count = finite.sum(axis=0, dtype=np.int64)

        if sum_vec is None:
            sum_vec = block_sum
            sumsq_vec = block_sumsq
            count_vec = block_count
        else:
            sum_vec += block_sum
            sumsq_vec += block_sumsq
            count_vec += block_count

    assert sum_vec is not None and sumsq_vec is not None and count_vec is not None
    mean = np.divide(sum_vec, count_vec, out=np.zeros_like(sum_vec), where=count_vec > 0)
    var = np.divide(sumsq_vec, count_vec, out=np.zeros_like(sumsq_vec), where=count_vec > 0) - np.square(mean)
    var = np.clip(var, 0.0, None)
    std = np.sqrt(var)
    std = np.where((count_vec <= 1) | (std < MIN_STD), 1.0, std)
    return StandardizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def _standardize(values: np.ndarray, stats: StandardizationStats | None) -> np.ndarray:
    if stats is None:
        return values
    return (values - stats.mean) / stats.std


class JAXSpectraDataLoader:
    """Iterable JAX dataloader backed by row-indexed Zarr arrays."""

    def __init__(
        self,
        dataset: SpectraZarrDataset,
        indices: np.ndarray,
        *,
        batch_size: int,
        input_key: str,
        target_key: str | None,
        input_columns: np.ndarray | None = None,
        target_columns: np.ndarray | None = None,
        input_stats: StandardizationStats | None = None,
        target_stats: StandardizationStats | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        device: jax.Device | None = None,
        input_dtype: np.dtype | str = np.float32,
        target_dtype: np.dtype | str = np.float32,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.indices = _as_1d_int64(indices)
        self.batch_size = int(batch_size)
        self.input_key = input_key
        self.target_key = target_key
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.input_stats = input_stats
        self.target_stats = target_stats
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.device = device
        self.input_dtype = np.dtype(input_dtype)
        self.target_dtype = np.dtype(target_dtype)
        self._epoch = 0

    def __len__(self) -> int:
        if self.indices.size == 0:
            return 0
        if self.drop_last:
            return self.indices.size // self.batch_size
        return int(np.ceil(self.indices.size / self.batch_size))

    def __iter__(self) -> Iterator[Mapping[str, jax.Array]]:
        order = self.indices.copy()
        if self.shuffle and order.size > 1:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(order)
        self._epoch += 1

        for start in range(0, order.size, self.batch_size):
            stop = start + self.batch_size
            if self.drop_last and stop > order.size:
                break
            batch_idx = order[start:stop]
            if batch_idx.size == 0:
                continue

            x_np = self.dataset.get_rows(self.input_key, batch_idx, column_ids=self.input_columns)
            x_np = np.asarray(x_np, dtype=self.input_dtype)
            x_np = _standardize(x_np, self.input_stats).astype(self.input_dtype, copy=False)
            x = jnp.asarray(x_np)

            batch: Dict[str, jax.Array] = {"inputs": x, "indices": jnp.asarray(batch_idx)}
            if self.target_key is not None:
                y_np = self.dataset.get_rows(self.target_key, batch_idx, column_ids=self.target_columns)
                y_np = np.asarray(y_np, dtype=self.target_dtype)
                y_np = _standardize(y_np, self.target_stats).astype(self.target_dtype, copy=False)
                batch["targets"] = jnp.asarray(y_np)

            if self.device is not None:
                for key, value in list(batch.items()):
                    batch[key] = jax.device_put(value, self.device)
            yield batch


def create_jax_spectra_dataloaders(
    *,
    zarr_path: str,
    batch_size: int,
    input_key: str | None = None,
    target_key: str | None = "auto",
    input_features: Sequence[str] | None = None,
    target_features: Sequence[str] | None = None,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    shuffle_train: bool = True,
    drop_last: bool = False,
    seed: int = 0,
    normalize_inputs: bool = False,
    normalize_targets: bool = False,
    stats_block_rows: int = 512,
    device: jax.Device | None = None,
    input_dtype: np.dtype | str = np.float32,
    target_dtype: np.dtype | str = np.float32,
) -> Dict[str, JAXSpectraDataLoader]:
    """Build train/val/test JAX loaders from a spectral Zarr store."""

    dataset = SpectraZarrDataset(zarr_path)
    resolved_input_key, resolved_target_key = _infer_io_keys(dataset, input_key, target_key)
    input_columns = dataset.resolve_param_columns(input_features) if resolved_input_key == "params" else None
    target_columns = dataset.resolve_param_columns(target_features) if resolved_target_key == "params" else None

    split_ids = _split_indices(dataset.row_count, train_fraction=train_fraction, val_fraction=val_fraction, seed=seed)
    train_ids = split_ids["train"]

    input_stats = None
    if normalize_inputs:
        input_stats = _compute_standardization(
            dataset=dataset,
            array_name=resolved_input_key,
            row_ids=train_ids,
            column_ids=input_columns,
            block_rows=stats_block_rows,
        )

    target_stats = None
    if normalize_targets and resolved_target_key is not None:
        target_stats = _compute_standardization(
            dataset=dataset,
            array_name=resolved_target_key,
            row_ids=train_ids,
            column_ids=target_columns,
            block_rows=stats_block_rows,
        )

    loaders = {
        "train": JAXSpectraDataLoader(
            dataset,
            split_ids["train"],
            batch_size=batch_size,
            input_key=resolved_input_key,
            target_key=resolved_target_key,
            input_columns=input_columns,
            target_columns=target_columns,
            input_stats=input_stats,
            target_stats=target_stats,
            shuffle=shuffle_train,
            drop_last=drop_last,
            seed=seed,
            device=device,
            input_dtype=input_dtype,
            target_dtype=target_dtype,
        ),
        "val": JAXSpectraDataLoader(
            dataset,
            split_ids["val"],
            batch_size=batch_size,
            input_key=resolved_input_key,
            target_key=resolved_target_key,
            input_columns=input_columns,
            target_columns=target_columns,
            input_stats=input_stats,
            target_stats=target_stats,
            shuffle=False,
            drop_last=False,
            seed=seed,
            device=device,
            input_dtype=input_dtype,
            target_dtype=target_dtype,
        ),
        "test": JAXSpectraDataLoader(
            dataset,
            split_ids["test"],
            batch_size=batch_size,
            input_key=resolved_input_key,
            target_key=resolved_target_key,
            input_columns=input_columns,
            target_columns=target_columns,
            input_stats=input_stats,
            target_stats=target_stats,
            shuffle=False,
            drop_last=False,
            seed=seed,
            device=device,
            input_dtype=input_dtype,
            target_dtype=target_dtype,
        ),
    }
    return loaders


def _parse_feature_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [name.strip() for name in raw.split(",") if name.strip()]
    return names if names else None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Input spectra Zarr path")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Train split fraction")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=0, help="Split/shuffle seed")
    parser.add_argument("--input-key", default=None, help="Input array name (default: params if present, else flux)")
    parser.add_argument(
        "--target-key",
        default="auto",
        help="Target array name. Use 'auto' (default), explicit array name, or 'none'.",
    )
    parser.add_argument("--input-features", default=None, help="Comma-separated subset of param_names for inputs")
    parser.add_argument("--target-features", default=None, help="Comma-separated subset of param_names for targets")
    parser.add_argument("--normalize-inputs", action="store_true", help="Standardize inputs using train split stats")
    parser.add_argument("--normalize-targets", action="store_true", help="Standardize targets using train split stats")
    parser.add_argument("--drop-last", action="store_true", help="Drop final short training batch")
    parser.add_argument("--preview-split", choices=("train", "val", "test"), default="train", help="Split to preview")
    parser.add_argument("--preview-batches", type=int, default=2, help="How many batches to print")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    target_key = None if str(args.target_key).lower() == "none" else args.target_key
    input_features = _parse_feature_list(args.input_features)
    target_features = _parse_feature_list(args.target_features)

    loaders = create_jax_spectra_dataloaders(
        zarr_path=args.zarr_path,
        batch_size=args.batch_size,
        input_key=args.input_key,
        target_key=target_key,
        input_features=input_features,
        target_features=target_features,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        shuffle_train=True,
        drop_last=args.drop_last,
        seed=args.seed,
        normalize_inputs=args.normalize_inputs,
        normalize_targets=args.normalize_targets,
    )

    dataset = loaders["train"].dataset
    print(f"Loaded: {dataset.zarr_path}")
    print(f"Rows: {dataset.row_count}")
    print(f"Arrays: {', '.join(dataset.array_names)}")
    if dataset.param_names:
        print(f"param_names: {dataset.param_names}")
    for split_name in ("train", "val", "test"):
        loader = loaders[split_name]
        print(
            f"{split_name}: rows={loader.indices.size} batches={len(loader)} "
            f"(batch_size={loader.batch_size})"
        )

    preview_loader = loaders[args.preview_split]
    for i, batch in enumerate(preview_loader):
        x = batch["inputs"]
        y = batch.get("targets")
        if y is None:
            print(f"[{args.preview_split} batch {i}] inputs={tuple(x.shape)}")
        else:
            print(f"[{args.preview_split} batch {i}] inputs={tuple(x.shape)} targets={tuple(y.shape)}")
        if i + 1 >= args.preview_batches:
            break


if __name__ == "__main__":
    main()
