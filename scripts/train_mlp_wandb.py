#!/usr/bin/env python3
"""Train FLAX MLP models over hyperparameter sweeps and log to Weights & Biases.

This script is designed for local testing and HPC scheduling (including Gadi):
- Runs one or many hyperparameter configurations from CLI grids or JSON sweep config.
- Supports `--run-index` (or env-driven index) so each array task can run a single config.
- Logs per-epoch metrics to Weights & Biases (online/offline/disabled).
- Trains on spectra targets resampled to a configurable linear/log wavelength axis.

Example
-------
python scripts/train_mlp_wandb.py \
  --zarr-path spectra_tiny.zarr \
  --wandb-project turbospectrum-mlp \
  --wandb-mode offline \
  --hidden-dims-grid 128x256,256x256 \
  --learning-rate-grid 1e-3,3e-4 \
  --epochs-grid 20 \
  --batch-size-grid 32
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zarr


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "mlp_wandb"
DEFAULT_SWEEP_CONFIG = REPO_ROOT / "configs" / "training" / "mlp_wandb_sweep.example.json"


@dataclasses.dataclass(frozen=True)
class RunSpec:
    hidden_dims: tuple[int, ...]
    learning_rate: float
    weight_decay: float
    lambda_hi: float
    lambda_lo: float
    epochs: int
    batch_size: int
    seed: int
    run_name: str | None = None


def _configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("train_mlp_wandb")
    if logger.handlers:
        return logger

    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger.setLevel(log_level)
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def _zarr_store(path: str):
    if hasattr(zarr, "DirectoryStore"):
        return zarr.DirectoryStore(path)  # type: ignore[attr-defined]
    from zarr import storage as zstorage  # type: ignore

    if hasattr(zstorage, "DirectoryStore"):
        return zstorage.DirectoryStore(path)  # type: ignore[attr-defined]
    if hasattr(zstorage, "LocalStore"):
        return zstorage.LocalStore(path)  # type: ignore[attr-defined]
    raise AttributeError("Unsupported Zarr version: cannot find DirectoryStore/LocalStore")


def _installed_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except Exception:  # noqa: BLE001
        return "not installed"


def _open_zarr_root(path: Path):
    return zarr.open_group(store=_zarr_store(str(path)), mode="r")


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    out: list[str] = []
    for item in np.asarray(values).tolist():
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return tuple(out)


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_float_grid(raw: str) -> list[float]:
    values = [float(v) for v in _split_csv(raw)]
    if not values:
        raise ValueError("Expected at least one floating-point value in grid")
    return values


def _parse_int_grid(raw: str) -> list[int]:
    values = [int(v) for v in _split_csv(raw)]
    if not values:
        raise ValueError("Expected at least one integer value in grid")
    return values


def _parse_hidden_dims(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        tokens = [tok for tok in value.strip().lower().split("x") if tok]
    elif isinstance(value, Sequence):
        tokens = [str(tok).strip() for tok in value if str(tok).strip()]
    else:
        raise TypeError(f"Cannot parse hidden_dims from type {type(value)!r}")

    dims = tuple(int(tok) for tok in tokens)
    if not dims:
        raise ValueError("hidden_dims must have at least one layer")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"hidden_dims must be positive, got {dims}")
    return dims


def _parse_hidden_dims_grid(raw: str) -> list[tuple[int, ...]]:
    dims = [_parse_hidden_dims(token) for token in _split_csv(raw)]
    if not dims:
        raise ValueError("hidden_dims grid is empty")
    return dims


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{raw}'")


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-") or "run"


def _spec_to_compact_name(spec: RunSpec, run_index: int) -> str:
    if spec.run_name:
        return _sanitize_name(spec.run_name)
    h = "x".join(str(x) for x in spec.hidden_dims)
    return _sanitize_name(
        f"run{run_index:03d}_h{h}_lr{spec.learning_rate:g}_wd{spec.weight_decay:g}"
        f"_bs{spec.batch_size}_lh{spec.lambda_hi:g}_ll{spec.lambda_lo:g}_ep{spec.epochs}"
    )


def _resolve_default_zarr_path() -> Path | None:
    candidates = (
        REPO_ROOT / "spectra_tiny.zarr",
        REPO_ROOT / "scripts" / "synthesized_spectra.zarr",
        REPO_ROOT / "runs" / "local-dev" / "outputs" / "zarr" / "synthesized_spectra.zarr",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _parse_input_features(raw: str | None, available: Sequence[str]) -> list[str] | None:
    if raw is None:
        return list(available) if available else None
    values = _split_csv(raw)
    if not values:
        return None
    return values


def _run_spec_from_mapping(raw: Mapping[str, Any], defaults: RunSpec, seed_fallback: int) -> RunSpec:
    hidden_dims = _parse_hidden_dims(raw.get("hidden_dims", defaults.hidden_dims))
    learning_rate = float(raw.get("learning_rate", defaults.learning_rate))
    weight_decay = float(raw.get("weight_decay", defaults.weight_decay))
    lambda_hi = float(raw.get("lambda_hi", defaults.lambda_hi))
    lambda_lo = float(raw.get("lambda_lo", defaults.lambda_lo))
    epochs = int(raw.get("epochs", defaults.epochs))
    batch_size = int(raw.get("batch_size", defaults.batch_size))
    seed = int(raw.get("seed", seed_fallback))
    run_name = raw.get("run_name")

    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    return RunSpec(
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lambda_hi=lambda_hi,
        lambda_lo=lambda_lo,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        run_name=str(run_name) if run_name is not None else None,
    )


def _build_specs_from_cli(args: argparse.Namespace) -> list[RunSpec]:
    hidden_dims_grid = _parse_hidden_dims_grid(args.hidden_dims_grid)
    learning_rate_grid = _parse_float_grid(args.learning_rate_grid)
    weight_decay_grid = _parse_float_grid(args.weight_decay_grid)
    lambda_hi_grid = _parse_float_grid(args.lambda_hi_grid)
    lambda_lo_grid = _parse_float_grid(args.lambda_lo_grid)
    epochs_grid = _parse_int_grid(args.epochs_grid)
    batch_size_grid = _parse_int_grid(args.batch_size_grid)

    specs: list[RunSpec] = []
    for combo in itertools.product(
        hidden_dims_grid,
        learning_rate_grid,
        weight_decay_grid,
        lambda_hi_grid,
        lambda_lo_grid,
        epochs_grid,
        batch_size_grid,
    ):
        hidden_dims, lr, wd, lhi, llo, epochs, batch_size = combo
        specs.append(
            RunSpec(
                hidden_dims=tuple(hidden_dims),
                learning_rate=float(lr),
                weight_decay=float(wd),
                lambda_hi=float(lhi),
                lambda_lo=float(llo),
                epochs=int(epochs),
                batch_size=int(batch_size),
                seed=0,  # filled below
            )
        )

    for idx, spec in enumerate(specs):
        specs[idx] = dataclasses.replace(spec, seed=int(args.seed + idx))
    return specs


def _build_specs_from_config(config_path: Path, defaults: RunSpec, base_seed: int) -> list[RunSpec]:
    payload = json.loads(config_path.read_text())

    if isinstance(payload, list):
        runs = payload
        return [_run_spec_from_mapping(run, defaults=defaults, seed_fallback=base_seed + i) for i, run in enumerate(runs)]

    if not isinstance(payload, dict):
        raise ValueError("Sweep config must be a mapping or a list of run mappings")

    if "runs" in payload:
        runs = payload["runs"]
        if not isinstance(runs, list):
            raise ValueError("'runs' in sweep config must be a list")
        return [_run_spec_from_mapping(run, defaults=defaults, seed_fallback=base_seed + i) for i, run in enumerate(runs)]

    if "grid" not in payload:
        raise ValueError("Sweep config must contain either 'runs' or 'grid'")

    grid = payload["grid"]
    if not isinstance(grid, dict):
        raise ValueError("'grid' in sweep config must be a mapping")
    config_defaults = payload.get("defaults", {})
    if config_defaults and not isinstance(config_defaults, dict):
        raise ValueError("'defaults' in sweep config must be a mapping")

    merged_defaults = _run_spec_from_mapping(config_defaults or {}, defaults=defaults, seed_fallback=base_seed)

    hidden_dims_values = grid.get("hidden_dims", [merged_defaults.hidden_dims])
    learning_rate_values = grid.get("learning_rate", [merged_defaults.learning_rate])
    weight_decay_values = grid.get("weight_decay", [merged_defaults.weight_decay])
    lambda_hi_values = grid.get("lambda_hi", [merged_defaults.lambda_hi])
    lambda_lo_values = grid.get("lambda_lo", [merged_defaults.lambda_lo])
    epochs_values = grid.get("epochs", [merged_defaults.epochs])
    batch_size_values = grid.get("batch_size", [merged_defaults.batch_size])

    specs: list[RunSpec] = []
    for i, combo in enumerate(
        itertools.product(
            hidden_dims_values,
            learning_rate_values,
            weight_decay_values,
            lambda_hi_values,
            lambda_lo_values,
            epochs_values,
            batch_size_values,
        )
    ):
        hidden_dims, lr, wd, lhi, llo, epochs, batch_size = combo
        spec = _run_spec_from_mapping(
            {
                "hidden_dims": hidden_dims,
                "learning_rate": lr,
                "weight_decay": wd,
                "lambda_hi": lhi,
                "lambda_lo": llo,
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": base_seed + i,
            },
            defaults=merged_defaults,
            seed_fallback=base_seed + i,
        )
        specs.append(spec)
    return specs


def _resolve_run_index(args: argparse.Namespace) -> int | None:
    if args.run_index is not None:
        return int(args.run_index)
    if not args.run_index_env:
        return None
    raw = os.environ.get(args.run_index_env)
    if raw is None:
        raise ValueError(f"Environment variable '{args.run_index_env}' is not set")
    env_value = int(raw)
    return env_value - int(args.run_index_offset)


class FluxResampler:
    """Vectorized 1D linear interpolation from source axis to target axis."""

    def __init__(self, source_axis: np.ndarray, target_points: int, source_indices: np.ndarray | None = None):
        axis = np.asarray(source_axis, dtype=np.float64)
        if axis.ndim != 1:
            raise ValueError(f"source axis must be 1D, got shape {axis.shape}")
        if axis.size < 2:
            raise ValueError("source axis must have at least 2 values")
        if target_points <= 0:
            raise ValueError("target_points must be positive")
        if np.any(~np.isfinite(axis)):
            raise ValueError("source axis contains non-finite values")

        if axis[0] > axis[-1]:
            axis = axis[::-1]
            self.reverse_flux = True
        else:
            self.reverse_flux = False

        diffs = np.diff(axis)
        if np.any(diffs <= 0.0):
            raise ValueError("source axis must be strictly increasing")

        n_target = min(int(target_points), int(axis.size))
        target_axis = np.linspace(axis[0], axis[-1], num=n_target, dtype=np.float64)
        left = np.searchsorted(axis, target_axis, side="right") - 1
        left = np.clip(left, 0, axis.size - 2)
        right = left + 1
        den = axis[right] - axis[left]
        alpha = np.divide(target_axis - axis[left], den, out=np.zeros_like(target_axis), where=den > 0.0)

        self.source_axis = axis
        self.target_axis = target_axis
        self.source_indices = None if source_indices is None else np.asarray(source_indices, dtype=np.int64)
        self.left = left.astype(np.int64)
        self.right = right.astype(np.int64)
        self.alpha = alpha.astype(np.float32)

    def resample(self, flux_batch: np.ndarray) -> np.ndarray:
        y = np.asarray(flux_batch, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        if self.source_indices is not None:
            y = y[:, self.source_indices]
        if self.reverse_flux:
            y = y[:, ::-1]
        y_left = y[:, self.left]
        y_right = y[:, self.right]
        out = (1.0 - self.alpha[None, :]) * y_left + self.alpha[None, :] * y_right
        return out.astype(np.float32, copy=False)


class BatchAdapter:
    """Convert dataloader batches to model-ready arrays."""

    def __init__(
        self,
        *,
        root,
        train_indices: np.ndarray,
        row_count: int,
        resampler: FluxResampler,
        include_mu: bool,
        mu_key: str,
        logger: logging.Logger,
    ):
        self.resampler = resampler
        self.include_mu = include_mu
        self.mu_source = "disabled"
        self.mu_feature = np.zeros((row_count,), dtype=np.float32)
        if include_mu:
            self.mu_feature, self.mu_source = self._build_mu_feature(
                root=root,
                train_indices=train_indices,
                row_count=row_count,
                mu_key=mu_key,
                logger=logger,
            )

    @staticmethod
    def _build_mu_feature(
        *,
        root,
        train_indices: np.ndarray,
        row_count: int,
        mu_key: str,
        logger: logging.Logger,
    ) -> tuple[np.ndarray, str]:
        chosen_key = mu_key
        if chosen_key == "auto":
            chosen_key = "mu_selected" if "mu_selected" in root else ("mu" if "mu" in root else "")
        if not chosen_key:
            logger.warning("No mu key found; using constant-zero mu feature")
            return np.zeros((row_count,), dtype=np.float32), "constant_zero_fallback"

        if chosen_key not in root:
            logger.warning("mu key '%s' not found; using constant-zero mu feature", chosen_key)
            return np.zeros((row_count,), dtype=np.float32), "constant_zero_fallback"

        mu_raw = np.asarray(root[chosen_key][:], dtype=np.float32)
        mu_source = chosen_key

        if not np.isfinite(mu_raw).any():
            if "mu_selected_index" in root:
                mu_idx = np.asarray(root["mu_selected_index"][:], dtype=np.float32)
                if np.isfinite(mu_idx).any() and np.any(mu_idx >= 0):
                    mu_raw = mu_idx
                    mu_source = "mu_selected_index"
                else:
                    mu_raw = np.zeros((row_count,), dtype=np.float32)
                    mu_source = "constant_zero_fallback"
            else:
                mu_raw = np.zeros((row_count,), dtype=np.float32)
                mu_source = "constant_zero_fallback"

        mu_train = mu_raw[np.asarray(train_indices, dtype=np.int64)]
        finite = np.isfinite(mu_train)
        if finite.any():
            mu_mean = float(np.mean(mu_train[finite]))
            mu_std = float(np.std(mu_train[finite]))
        else:
            mu_mean = 0.0
            mu_std = 1.0
        if mu_std < 1e-6:
            mu_std = 1.0

        mu_feature = (np.nan_to_num(mu_raw, nan=mu_mean, posinf=mu_mean, neginf=mu_mean) - mu_mean) / mu_std
        mu_feature = np.asarray(mu_feature, dtype=np.float32)
        return mu_feature, mu_source

    def batch_to_xy(self, batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        x_params = np.asarray(batch["inputs"], dtype=np.float32)
        if x_params.ndim == 1:
            x_params = x_params[:, None]

        if self.include_mu:
            row_ids = np.asarray(batch["indices"], dtype=np.int64)
            mu = self.mu_feature[row_ids][:, None]
            x = np.concatenate([x_params, mu], axis=1)
        else:
            x = x_params
        x = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

        if "targets" not in batch:
            raise ValueError("Training requires batches with 'targets'")
        y_full = np.asarray(batch["targets"], dtype=np.float32)
        y = self.resampler.resample(y_full)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return x, y


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", "--zarr_path", default=None, help="Path to synthesized spectra Zarr store")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for run artifacts")
    parser.add_argument("--sweep-config", default=None, help="JSON file with either 'runs' or 'grid' definitions")
    parser.add_argument("--example-sweep-config", action="store_true", help="Print example sweep config path and exit")

    parser.add_argument("--seed", type=int, default=7, help="Base seed; each run increments this seed")
    parser.add_argument("--train-fraction", "--train_fraction", type=float, default=0.8, help="Train split fraction")
    parser.add_argument("--val-fraction", "--val_fraction", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--normalize-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize_inputs", dest="normalize_inputs_value", default=None, type=_parse_bool, help=argparse.SUPPRESS)
    parser.add_argument("--input-features", default=None, help="Comma-separated subset of param_names (default: all)")
    parser.add_argument("--include-mu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_mu", dest="include_mu_value", default=None, type=_parse_bool, help=argparse.SUPPRESS)
    parser.add_argument("--mu-key", "--mu_key", default="auto", help="mu array key: auto|mu_selected|mu|<custom>")
    parser.add_argument("--target-points", "--target_points", type=int, default=1024, help="Output points after axis resampling")
    parser.add_argument("--use-log-wavelength", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_log_wavelength", dest="use_log_wavelength_value", default=None, type=_parse_bool, help=argparse.SUPPRESS)
    parser.add_argument("--axis-name", "--axis_name", choices=("wavelength", "log_wavelength"), default=None, help="Optional alias to control output axis")
    parser.add_argument("--target-axis-min", "--target_axis_min", type=float, default=None, help="Optional lower bound for training axis domain")
    parser.add_argument("--target-axis-max", "--target_axis_max", type=float, default=None, help="Optional upper bound for training axis domain")
    parser.add_argument("--jax-platform", "--jax_platform", default="cpu", help="Value for JAX_PLATFORMS before JAX import")

    parser.add_argument("--hidden-dims-grid", default="128x256", help="Comma-separated hidden layer configs, e.g. 128x256,256x256")
    parser.add_argument("--hidden-dims", "--hidden_dims", dest="hidden_dims_single", default=None, help="Single hidden layer config, e.g. 128x256")
    parser.add_argument("--learning-rate-grid", default="1e-3", help="Comma-separated learning rates")
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate_single", type=float, default=None, help="Single learning rate")
    parser.add_argument("--weight-decay-grid", default="0.0", help="Comma-separated weight decay values")
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay_single", type=float, default=None, help="Single weight decay")
    parser.add_argument("--lambda-hi-grid", default="0.1", help="Comma-separated weights for flux>1 penalty")
    parser.add_argument("--lambda-hi", "--lambda_hi", dest="lambda_hi_single", type=float, default=None, help="Single lambda_hi")
    parser.add_argument("--lambda-lo-grid", default="0.0", help="Comma-separated weights for flux<0 penalty")
    parser.add_argument("--lambda-lo", "--lambda_lo", dest="lambda_lo_single", type=float, default=None, help="Single lambda_lo")
    parser.add_argument("--epochs-grid", default="30", help="Comma-separated epoch counts")
    parser.add_argument("--epochs", dest="epochs_single", type=int, default=None, help="Single epoch count")
    parser.add_argument("--batch-size-grid", default="32", help="Comma-separated batch sizes")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size_single", type=int, default=None, help="Single batch size")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on number of runs (before run-index filtering)")
    parser.add_argument("--run-index", type=int, default=None, help="Run only one 0-based index from the generated sweep")
    parser.add_argument("--run-index-env", default="", help="Read run index from env var (e.g. PBS_ARRAY_INDEX)")
    parser.add_argument("--run-index-offset", type=int, default=0, help="Subtract this offset from env run index")

    parser.add_argument("--wandb-project", default="turbospectrum-mlp", help="Weights & Biases project")
    parser.add_argument("--wandb-entity", default=None, help="Weights & Biases entity/user/team")
    parser.add_argument("--wandb-group", default=None, help="Weights & Biases group name")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument(
        "--wandb-sync-after-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run `wandb sync` at the end (useful for offline runs on nodes with internet access).",
    )
    parser.add_argument(
        "--wandb-sync-dir",
        default=None,
        help="Directory passed to `wandb sync` (default: --output-dir).",
    )
    parser.add_argument(
        "--wandb-sync-include-offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include offline runs when syncing (`wandb sync --include-offline`).",
    )
    parser.add_argument(
        "--wandb-sync-best-effort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not fail the job if `wandb sync` fails.",
    )
    parser.add_argument("--save-checkpoints", action="store_true", help="Save best params for each run")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logger = _configure_logging(args.log_level)

    if args.example_sweep_config:
        print(DEFAULT_SWEEP_CONFIG)
        return

    if args.normalize_inputs_value is not None:
        args.normalize_inputs = bool(args.normalize_inputs_value)
    if args.include_mu_value is not None:
        args.include_mu = bool(args.include_mu_value)
    if args.use_log_wavelength_value is not None:
        args.use_log_wavelength = bool(args.use_log_wavelength_value)
    if args.axis_name is not None:
        args.use_log_wavelength = bool(args.axis_name == "log_wavelength")

    if args.hidden_dims_single is not None:
        args.hidden_dims_grid = str(args.hidden_dims_single)
    if args.learning_rate_single is not None:
        args.learning_rate_grid = str(args.learning_rate_single)
    if args.weight_decay_single is not None:
        args.weight_decay_grid = str(args.weight_decay_single)
    if args.lambda_hi_single is not None:
        args.lambda_hi_grid = str(args.lambda_hi_single)
    if args.lambda_lo_single is not None:
        args.lambda_lo_grid = str(args.lambda_lo_single)
    if args.epochs_single is not None:
        args.epochs_grid = str(args.epochs_single)
    if args.batch_size_single is not None:
        args.batch_size_grid = str(args.batch_size_single)

    default_zarr = _resolve_default_zarr_path()
    zarr_path = Path(args.zarr_path) if args.zarr_path else default_zarr
    if zarr_path is None:
        raise FileNotFoundError("No default Zarr dataset found. Pass --zarr-path explicitly.")
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr dataset not found: {zarr_path}")

    os.environ["JAX_PLATFORMS"] = str(args.jax_platform)

    sys.path.insert(0, str(REPO_ROOT))

    try:
        import jax
        import jax.numpy as jnp
        import flax.linen as nn
        from flax import serialization
        from flax.training import train_state
        import optax
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Missing or incompatible JAX/FLAX stack. "
            "Install e.g. `pip install -r requirements-flax-ml.txt`."
        ) from exc

    from scripts.jax_spectra_dataloader import create_jax_spectra_dataloaders

    wandb = None
    if args.wandb_mode != "disabled":
        try:
            import wandb as _wandb
        except Exception as exc:  # noqa: BLE001
            raise ImportError("wandb is required unless --wandb-mode=disabled") from exc
        wandb = _wandb

    versions = {
        "numpy": _installed_version("numpy"),
        "jax": _installed_version("jax"),
        "jaxlib": _installed_version("jaxlib"),
        "flax": _installed_version("flax"),
        "optax": _installed_version("optax"),
    }

    try:
        # Basic runtime math preflight.
        _ = (jnp.asarray([1.0], dtype=jnp.float32) + 1.0).block_until_ready()
        # RNG preflight catches common JAX/NumPy mismatches early (before starting W&B runs).
        key = jax.random.PRNGKey(0)
        _ = jax.random.normal(key, shape=(1,), dtype=jnp.float32).block_until_ready()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"JAX runtime failed for platform '{args.jax_platform}'. "
            f"Installed versions: {versions}. "
            "Set --jax-platform cpu or fix the local JAX/NumPy stack. "
            "Recommended: `pip install --upgrade --force-reinstall -r requirements-flax-ml.txt`."
        ) from exc

    root = _open_zarr_root(zarr_path)
    arrays = set(root.array_keys())
    required = {"params", "flux"}
    missing = [name for name in required if name not in arrays]
    if missing:
        raise KeyError(f"Dataset {zarr_path} missing required arrays: {missing}")
    if "wavelength" not in arrays:
        raise KeyError("Dataset must include 'wavelength' for axis-resampled targets")

    param_names: tuple[str, ...] = ()
    if "param_names" in root:
        param_names = _decode_strings(np.asarray(root["param_names"][:]))
    input_features = _parse_input_features(args.input_features, param_names)
    wavelength = np.asarray(root["wavelength"][:], dtype=np.float64)
    if np.any(~np.isfinite(wavelength)):
        raise ValueError("wavelength contains non-finite values")

    axis_name = "log_wavelength" if bool(args.use_log_wavelength) else "wavelength"
    if args.use_log_wavelength:
        if np.any(wavelength <= 0.0):
            raise ValueError("wavelength must be strictly positive for log-axis targets")
        source_axis = np.log(wavelength)
    else:
        source_axis = wavelength
    source_indices = np.arange(source_axis.size, dtype=np.int64)
    if args.target_axis_min is not None or args.target_axis_max is not None:
        lo = float(source_axis[0]) if args.target_axis_min is None else float(args.target_axis_min)
        hi = float(source_axis[-1]) if args.target_axis_max is None else float(args.target_axis_max)
        if lo > hi:
            lo, hi = hi, lo
        mask = (source_axis >= lo) & (source_axis <= hi)
        if int(mask.sum()) < 2:
            raise ValueError(
                f"Axis clip [{lo}, {hi}] leaves fewer than 2 points for interpolation "
                f"(axis range: {float(source_axis[0])}..{float(source_axis[-1])})"
            )
        source_axis = source_axis[mask]
        source_indices = source_indices[mask]
    resampler = FluxResampler(
        source_axis=source_axis,
        target_points=int(args.target_points),
        source_indices=source_indices,
    )

    base_spec = _run_spec_from_mapping(
        {
            "hidden_dims": _parse_hidden_dims_grid(args.hidden_dims_grid)[0],
            "learning_rate": _parse_float_grid(args.learning_rate_grid)[0],
            "weight_decay": _parse_float_grid(args.weight_decay_grid)[0],
            "lambda_hi": _parse_float_grid(args.lambda_hi_grid)[0],
            "lambda_lo": _parse_float_grid(args.lambda_lo_grid)[0],
            "epochs": _parse_int_grid(args.epochs_grid)[0],
            "batch_size": _parse_int_grid(args.batch_size_grid)[0],
            "seed": int(args.seed),
        },
        defaults=RunSpec(
            hidden_dims=(128, 256),
            learning_rate=1e-3,
            weight_decay=0.0,
            lambda_hi=0.1,
            lambda_lo=0.0,
            epochs=30,
            batch_size=32,
            seed=int(args.seed),
            run_name=None,
        ),
        seed_fallback=int(args.seed),
    )

    if args.sweep_config:
        sweep_config_path = Path(args.sweep_config)
        if not sweep_config_path.exists():
            raise FileNotFoundError(f"Sweep config not found: {sweep_config_path}")
        specs = _build_specs_from_config(sweep_config_path, defaults=base_spec, base_seed=int(args.seed))
    else:
        specs = _build_specs_from_cli(args)

    if not specs:
        raise ValueError("No run specs were generated")

    if args.max_runs is not None:
        specs = specs[: max(0, int(args.max_runs))]
    if not specs:
        raise ValueError("No run specs remain after applying --max-runs")

    run_index = _resolve_run_index(args)
    if run_index is not None:
        if run_index < 0 or run_index >= len(specs):
            raise IndexError(f"run_index={run_index} is out of range [0, {len(specs) - 1}]")
        indexed_specs = [(run_index, specs[run_index])]
    else:
        indexed_specs = list(enumerate(specs))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset: %s", zarr_path)
    logger.info("Rows: %d | Flux dim: %d | Target axis points: %d | Axis: %s", int(root["flux"].shape[0]), int(root["flux"].shape[1]), int(resampler.target_axis.size), axis_name)
    logger.info("Generated %d run specs; executing %d run(s)", len(specs), len(indexed_specs))

    loader_cache: dict[int, Mapping[str, Any]] = {}
    adapter_cache: dict[int, BatchAdapter] = {}

    def get_loaders(batch_size: int):
        if batch_size not in loader_cache:
            loaders = create_jax_spectra_dataloaders(
                zarr_path=str(zarr_path),
                batch_size=int(batch_size),
                input_key="params",
                target_key="flux",
                input_features=input_features,
                train_fraction=float(args.train_fraction),
                val_fraction=float(args.val_fraction),
                shuffle_train=True,
                drop_last=False,
                seed=int(args.seed),
                normalize_inputs=bool(args.normalize_inputs),
                normalize_targets=False,
            )
            loader_cache[batch_size] = loaders
        return loader_cache[batch_size]

    def get_adapter(batch_size: int):
        if batch_size not in adapter_cache:
            loaders = get_loaders(batch_size)
            row_count = int(root["flux"].shape[0])
            adapter_cache[batch_size] = BatchAdapter(
                root=root,
                train_indices=np.asarray(loaders["train"].indices, dtype=np.int64),
                row_count=row_count,
                resampler=resampler,
                include_mu=bool(args.include_mu),
                mu_key=str(args.mu_key),
                logger=logger,
            )
        return adapter_cache[batch_size]

    class FluxMLP(nn.Module):
        hidden_dims: tuple[int, ...]
        output_dim: int

        @nn.compact
        def __call__(self, x):
            h = x
            for width in self.hidden_dims:
                h = nn.Dense(width)(h)
                h = nn.relu(h)
            return nn.Dense(self.output_dim)(h)

    def create_train_state(rng, model, input_dim: int, learning_rate: float, weight_decay: float):
        params = model.init(rng, jnp.ones((1, input_dim), dtype=jnp.float32))["params"]
        tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def loss_components(pred, y, lambda_hi, lambda_lo):
        mse = jnp.mean((pred - y) ** 2)
        hi_pen = jnp.mean(jax.nn.relu(pred - 1.0) ** 2)
        lo_pen = jnp.mean(jax.nn.relu(0.0 - pred) ** 2)
        total = mse + lambda_hi * hi_pen + lambda_lo * lo_pen
        return total, mse, hi_pen, lo_pen

    @jax.jit
    def train_step(state, x, y, lambda_hi, lambda_lo):
        def loss_fn(params):
            pred = state.apply_fn({"params": params}, x)
            total, mse, hi_pen, lo_pen = loss_components(pred, y, lambda_hi, lambda_lo)
            return total, (mse, hi_pen, lo_pen)

        (loss, (mse, hi_pen, lo_pen)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, mse, hi_pen, lo_pen

    @jax.jit
    def eval_step(state, x, y, lambda_hi, lambda_lo):
        pred = state.apply_fn({"params": state.params}, x)
        return loss_components(pred, y, lambda_hi, lambda_lo)

    def evaluate_loader(state, loader, adapter: BatchAdapter, lambda_hi: float, lambda_lo: float) -> dict[str, float]:
        totals, mses, hi_pens, lo_pens = [], [], [], []
        lh = jnp.asarray(lambda_hi, dtype=jnp.float32)
        ll = jnp.asarray(lambda_lo, dtype=jnp.float32)
        for batch in loader:
            x_np, y_np = adapter.batch_to_xy(batch)
            x = jnp.asarray(x_np, dtype=jnp.float32)
            y = jnp.asarray(y_np, dtype=jnp.float32)
            total, mse, hi_pen, lo_pen = eval_step(state, x, y, lh, ll)
            totals.append(float(total))
            mses.append(float(mse))
            hi_pens.append(float(hi_pen))
            lo_pens.append(float(lo_pen))
        if not totals:
            return {"total": float("nan"), "mse": float("nan"), "hi_pen": float("nan"), "lo_pen": float("nan")}
        return {
            "total": float(np.mean(totals)),
            "mse": float(np.mean(mses)),
            "hi_pen": float(np.mean(hi_pens)),
            "lo_pen": float(np.mean(lo_pens)),
        }

    for run_idx, spec in indexed_specs:
        run_name = _spec_to_compact_name(spec, run_idx)
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        run_cfg = {
            "run_index": int(run_idx),
            "zarr_path": str(zarr_path),
            "axis_name": axis_name,
            "target_points": int(resampler.target_axis.size),
            "target_axis_min": float(resampler.target_axis[0]),
            "target_axis_max": float(resampler.target_axis[-1]),
            "include_mu": bool(args.include_mu),
            "mu_key": str(args.mu_key),
            "input_features": input_features,
            "train_fraction": float(args.train_fraction),
            "val_fraction": float(args.val_fraction),
            "normalize_inputs": bool(args.normalize_inputs),
            "seed": int(spec.seed),
            "hidden_dims": list(spec.hidden_dims),
            "learning_rate": float(spec.learning_rate),
            "weight_decay": float(spec.weight_decay),
            "lambda_hi": float(spec.lambda_hi),
            "lambda_lo": float(spec.lambda_lo),
            "epochs": int(spec.epochs),
            "batch_size": int(spec.batch_size),
            "jax_platform": str(args.jax_platform),
        }
        (run_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))

        wb_run = None
        if wandb is not None:
            wb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                tags=[t for t in _split_csv(args.wandb_tags) if t],
                mode=args.wandb_mode,
                name=run_name,
                dir=str(run_dir),
                config=run_cfg,
                reinit=True,
            )

        loaders = get_loaders(spec.batch_size)
        if hasattr(loaders["train"], "_epoch"):
            loaders["train"]._epoch = 0  # type: ignore[attr-defined]
        adapter = get_adapter(spec.batch_size)

        probe = next(iter(loaders["train"]))
        x0, y0 = adapter.batch_to_xy(probe)

        model = FluxMLP(hidden_dims=spec.hidden_dims, output_dim=int(y0.shape[1]))
        state = create_train_state(
            jax.random.PRNGKey(int(spec.seed)),
            model=model,
            input_dim=int(x0.shape[1]),
            learning_rate=float(spec.learning_rate),
            weight_decay=float(spec.weight_decay),
        )

        lambda_hi = jnp.asarray(spec.lambda_hi, dtype=jnp.float32)
        lambda_lo = jnp.asarray(spec.lambda_lo, dtype=jnp.float32)

        logger.info(
            "Run %d/%d: %s | h=%s lr=%g wd=%g bs=%d ep=%d lhi=%g llo=%g mu_source=%s",
            run_idx + 1,
            len(specs),
            run_name,
            spec.hidden_dims,
            spec.learning_rate,
            spec.weight_decay,
            spec.batch_size,
            spec.epochs,
            spec.lambda_hi,
            spec.lambda_lo,
            adapter.mu_source,
        )

        best_val = float("inf")
        best_epoch = -1
        best_params = None
        metrics_path = run_dir / "metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as mf:
            for epoch in range(1, spec.epochs + 1):
                epoch_start = time.time()
                train_totals, train_mses, train_hi_pens, train_lo_pens = [], [], [], []
                for batch in loaders["train"]:
                    x_np, y_np = adapter.batch_to_xy(batch)
                    x = jnp.asarray(x_np, dtype=jnp.float32)
                    y = jnp.asarray(y_np, dtype=jnp.float32)
                    state, total, mse, hi_pen, lo_pen = train_step(state, x, y, lambda_hi, lambda_lo)
                    train_totals.append(float(total))
                    train_mses.append(float(mse))
                    train_hi_pens.append(float(hi_pen))
                    train_lo_pens.append(float(lo_pen))

                train_stats = {
                    "total": float(np.mean(train_totals)) if train_totals else float("nan"),
                    "mse": float(np.mean(train_mses)) if train_mses else float("nan"),
                    "hi_pen": float(np.mean(train_hi_pens)) if train_hi_pens else float("nan"),
                    "lo_pen": float(np.mean(train_lo_pens)) if train_lo_pens else float("nan"),
                }
                val_stats = evaluate_loader(state, loaders["val"], adapter, spec.lambda_hi, spec.lambda_lo)

                if val_stats["mse"] < best_val:
                    best_val = val_stats["mse"]
                    best_epoch = epoch
                    best_params = state.params

                metrics = {
                    "epoch": int(epoch),
                    "seconds": float(time.time() - epoch_start),
                    "Relative Time (Process)": float(time.time() - epoch_start),
                    "train_total": train_stats["total"],
                    "train_mse": train_stats["mse"],
                    "train_hi_pen": train_stats["hi_pen"],
                    "train_lo_pen": train_stats["lo_pen"],
                    "val_total": val_stats["total"],
                    "val_mse": val_stats["mse"],
                    "val_hi_pen": val_stats["hi_pen"],
                    "val_lo_pen": val_stats["lo_pen"],
                }
                mf.write(json.dumps(metrics) + "\n")
                mf.flush()

                if wb_run is not None:
                    wb_run.log(metrics, step=epoch)

                if epoch == 1 or epoch == spec.epochs or epoch % 5 == 0:
                    logger.info(
                        "[%s] epoch=%03d train_mse=%.6f val_mse=%.6f val_hi_pen=%.6f",
                        run_name,
                        epoch,
                        train_stats["mse"],
                        val_stats["mse"],
                        val_stats["hi_pen"],
                    )

        test_stats = evaluate_loader(state, loaders["test"], adapter, spec.lambda_hi, spec.lambda_lo)

        summary = {
            "run_name": run_name,
            "run_index": int(run_idx),
            "best_val_mse": float(best_val),
            "best_epoch": int(best_epoch),
            "final_val_mse": float(val_stats["mse"]) if np.isfinite(val_stats["mse"]) else float("nan"),
            "test_total": float(test_stats["total"]),
            "test_mse": float(test_stats["mse"]),
            "test_hi_pen": float(test_stats["hi_pen"]),
            "test_lo_pen": float(test_stats["lo_pen"]),
            "mu_source": adapter.mu_source,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        if args.save_checkpoints and best_params is not None:
            ckpt_bytes = serialization.to_bytes(best_params)
            (run_dir / "best_params.msgpack").write_bytes(ckpt_bytes)

        if wb_run is not None:
            wb_run.summary.update(summary)
            wb_run.finish()

    logger.info("Completed %d run(s)", len(indexed_specs))

    if args.wandb_sync_after_run and args.wandb_mode != "disabled":
        sync_dir = Path(args.wandb_sync_dir) if args.wandb_sync_dir else output_dir
        cmd = [sys.executable, "-m", "wandb", "sync"]
        if args.wandb_sync_include_offline:
            cmd.append("--include-offline")
        cmd.append(str(sync_dir))
        logger.info("Running W&B sync: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            logger.info("W&B sync completed for %s", sync_dir)
        except subprocess.CalledProcessError as exc:
            msg = f"W&B sync failed (exit code {exc.returncode}) for {sync_dir}"
            if args.wandb_sync_best_effort:
                logger.warning(msg)
            else:
                raise RuntimeError(msg) from exc


if __name__ == "__main__":
    main()
