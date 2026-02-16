import csv
import argparse
import gzip
import io
import json
import math
import os
from itertools import islice, product
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml


DEFAULT_PARAMETER_CONFIG = {
    'teff': {'min': 5000, 'max': 6250, 'step': 250},
    'logg': {'min': 3.5, 'max': 5.0, 'step': 0.5},
    'feh': {'min': -1.0, 'max': 0.5, 'step': 0.5},
    't_value': ["01"],
    'a': ["+0.00"],
    'c': ["+0.00"],
    'n': ["+0.00"],
    'o': ["+0.00"],
    'r': ["+0.00"],
    's': ["+0.00"],
}


def _format_abundance_value(value):
    if isinstance(value, (int, float, np.floating)):
        return f"{value:+.2f}"
    return value


def _resolve_values(name, cfg, rng, is_abundance=False, fallback_count=None):
    """Return a list of values for a parameter from lists or distributions."""
    if cfg is None:
        cfg = DEFAULT_PARAMETER_CONFIG.get(name)

    if isinstance(cfg, (str, float, int, np.floating)):
        values = [cfg]
    elif isinstance(cfg, list):
        values = cfg
    elif isinstance(cfg, dict):
        if 'values' in cfg:
            values = cfg['values'] if isinstance(cfg['values'], list) else [cfg['values']]
        elif {'min', 'max', 'step'}.issubset(cfg):
            values = np.arange(cfg['min'], cfg['max'] + cfg['step'], cfg['step'])
        elif 'distribution' in cfg:
            dist = cfg['distribution'] or {}
            dist_type = dist.get('type', 'gaussian')
            count = dist.get('count', fallback_count)
            if count is None:
                raise ValueError(f"Distribution for '{name}' requires a 'count' or sampling max_rows")
            if rng is None:
                rng = np.random.default_rng()
            if dist_type == 'gaussian':
                mean = dist.get('mean', 0.0)
                sigma = dist.get('sigma', 0.1)
                values = rng.normal(mean, sigma, int(count))
            elif dist_type == 'uniform':
                min_val = dist.get('min')
                max_val = dist.get('max')
                if min_val is None or max_val is None:
                    raise ValueError(f"Uniform distribution for '{name}' requires 'min' and 'max'")
                values = rng.uniform(min_val, max_val, int(count))
            else:
                raise ValueError(f"Unsupported distribution type '{dist_type}' for parameter '{name}'")
        else:
            raise ValueError(f"Could not parse configuration for parameter '{name}'")
    else:
        raise ValueError(f"Unsupported configuration type for parameter '{name}'")

    if is_abundance:
        return [_format_abundance_value(v) for v in values]
    return list(values)


def _sample_parameter_space(parameter_lists, sampling_cfg, rng):
    method = (sampling_cfg or {}).get('method', 'full')
    max_rows = (sampling_cfg or {}).get('max_rows')

    values_only = list(parameter_lists.values())

    if method == 'full':
        combos = product(*values_only)
        return islice(combos, max_rows) if max_rows else combos

    if max_rows is None:
        raise ValueError(f"Sampling method '{method}' requires 'max_rows' to be set")

    if rng is None:
        rng = np.random.default_rng()

    if method == 'random':
        return (tuple(rng.choice(values) for values in values_only) for _ in range(int(max_rows)))

    if method == 'latin_hypercube':
        permutations = []
        for values in values_only:
            repeats = math.ceil(max_rows / len(values))
            base_indices = np.tile(np.arange(len(values)), repeats)[: int(max_rows)]
            permutations.append(np.asarray(base_indices)[rng.permutation(int(max_rows))])

        def lhs_generator():
            for row_idx in range(int(max_rows)):
                yield tuple(values[idx_list[row_idx]] for values, idx_list in zip(values_only, permutations))

        return lhs_generator()

    raise ValueError(f"Unsupported sampling method '{method}'")


def _abs_from(base_dir: str, maybe_relative: str | None) -> str | None:
    if not maybe_relative:
        return None
    return maybe_relative if os.path.isabs(maybe_relative) else os.path.abspath(os.path.join(base_dir, maybe_relative))


def _open_csv_writer(path: str, compression: str | None, level: int | None):
    """
    Open a text handle suitable for csv.writer, honoring optional compression.

    Supported: none, gzip, zstd. For zstd, requires `zstandard`.
    """
    compression_norm = (compression or "none").strip().lower()
    if compression_norm in {"none", ""}:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return open(path, "w", newline="", encoding="utf-8")

    if compression_norm in {"gz", "gzip"}:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return gzip.open(path, "wt", newline="", encoding="utf-8")

    if compression_norm in {"zst", "zstd"}:
        import zstandard as zstd  # type: ignore

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        raw = open(path, "wb")
        compressor = zstd.ZstdCompressor(level=int(level) if level is not None else 3)
        stream = compressor.stream_writer(raw, closefd=True)
        return io.TextIOWrapper(stream, encoding="utf-8", newline="")

    raise ValueError(f"Unsupported csv_compression={compression!r}. Use none/gzip/zstd.")


def _latin_hypercube(bounds: Iterable[Tuple[float, float]], samples: int, rng: np.random.Generator) -> np.ndarray:
    bounds = list(bounds)
    if samples <= 0:
        raise ValueError("num_samples must be positive")
    if not bounds:
        raise ValueError("At least one sampling dimension is required")

    dims = len(bounds)
    lhs = np.empty((samples, dims), dtype=float)
    for dim, (lower, upper) in enumerate(bounds):
        if upper <= lower:
            raise ValueError(f"Upper bound must exceed lower bound for dim {dim}: {lower} >= {upper}")
        stratified = (rng.random(samples) + np.arange(samples)) / samples
        rng.shuffle(stratified)
        lhs[:, dim] = lower + stratified * (upper - lower)
    return lhs


def _abundance_value(raw_value: Any) -> str:
    if raw_value is None:
        return "+0.00"
    return f"{float(raw_value):+0.2f}" if isinstance(raw_value, (int, float, np.floating)) else str(raw_value)


def _choose_series(raw_value: Any, rng: np.random.Generator, length: int, name: str) -> np.ndarray:
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        if not raw_value:
            raise ValueError(f"Configuration for '{name}' lists no options")
        return rng.choice(np.asarray(raw_value, dtype=object), size=length)
    if raw_value is None:
        raise ValueError(f"Configuration for '{name}' is missing")
    return np.full(length, raw_value, dtype=object)


def _resolve_ml_sampling(config: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Build ML-style sampled parameter columns from
    `configs/sampling/config_ml_sampling.json`.

    Returns a dict of column -> ndarray, matching the CSV/Zarr schema expected
    by the synthesis pipeline.
    """
    sample_count = int(config.get("num_samples", 50))
    bounds_cfg = config.get("bounds") or {}

    bounds: List[Tuple[float, float]] = []
    for name in ["teff", "logg", "feh"]:
        spec = bounds_cfg.get(name) or {}
        if "min" not in spec or "max" not in spec:
            raise ValueError(f"Bounds for '{name}' must include 'min' and 'max'")
        bounds.append((float(spec["min"]), float(spec["max"])))

    abund_cfg = config.get("abundances") or {}
    sampled_abundances: List[str] = []
    fixed_abundances: Dict[str, str] = {}
    for element in ["a", "c", "n", "o", "r", "s"]:
        raw_val = abund_cfg.get(element)
        if isinstance(raw_val, Mapping):
            if "min" not in raw_val or "max" not in raw_val:
                raise ValueError(f"Abundance bounds for '{element}' must include 'min' and 'max'")
            bounds.append((float(raw_val["min"]), float(raw_val["max"])))
            sampled_abundances.append(element)
        else:
            fixed_abundances[element] = _abundance_value(raw_val)

    sample_turbvel = bool(config.get("sample_turbvel", False))
    turbvel_options = config.get("turbvel_options") or ["01", "02", "03", "04", "05"]
    if isinstance(turbvel_options, (str, bytes)):
        turbvel_options = [turbvel_options]
    turbvel_options = [f"{int(x):02d}" if isinstance(x, (int, float, np.integer, np.floating)) else str(x) for x in turbvel_options]
    if sample_turbvel:
        if not turbvel_options:
            raise ValueError("sample_turbvel=true requires at least one turbvel option")
        bounds.append((0.0, float(len(turbvel_options))))

    lhs = _latin_hypercube(bounds, sample_count, rng)

    col_idx = 0
    teff = np.rint(lhs[:, col_idx]).astype(int)
    col_idx += 1
    logg = np.round(lhs[:, col_idx], 3)
    col_idx += 1
    feh = np.round(lhs[:, col_idx], 3)
    col_idx += 1

    abundance_samples: Dict[str, np.ndarray] = {}
    for element in sampled_abundances:
        values = np.round(lhs[:, col_idx], 2)
        abundance_samples[element] = np.array([f"{val:+0.2f}" for val in values], dtype=object)
        col_idx += 1

    turbvel_cfg = config.get("turbvel", "01")
    if sample_turbvel:
        indices = np.floor(lhs[:, col_idx]).astype(int)
        indices = np.clip(indices, 0, len(turbvel_options) - 1)
        turbvel = np.asarray([turbvel_options[i] for i in indices], dtype=object)
        col_idx += 1
    else:
        turbvel = _choose_series(turbvel_cfg, rng, sample_count, "turbvel")

    t_value = _choose_series(config.get("t_value_options", ["01"]), rng, sample_count, "t_value_options")

    synthesis_cfg = config.get("synthesis") or {}
    lam_min = float(synthesis_cfg.get("lam_min", 6000))
    lam_max = float(synthesis_cfg.get("lam_max", 6100))
    lam_step = float(synthesis_cfg.get("lam_step", 0.01))
    output_mode = str(synthesis_cfg.get("output_mode", "Flux"))

    grid_version = str(config.get("grid_version", "ml-sample"))
    mode = str(config.get("mode", "1D"))
    calculation_mode = str(config.get("calculation_mode", "LTE"))

    out: Dict[str, np.ndarray] = {
        "grid_version": np.full(sample_count, grid_version, dtype=object),
        "teff": teff,
        "logg": logg,
        "feh": feh,
        "lam_min": np.full(sample_count, lam_min, dtype=float),
        "lam_max": np.full(sample_count, lam_max, dtype=float),
        "lam_step": np.full(sample_count, lam_step, dtype=float),
        "turbvel": turbvel,
        "t_value": t_value,
        "a": abundance_samples.get("a", np.full(sample_count, fixed_abundances.get("a", "+0.00"), dtype=object)),
        "c": abundance_samples.get("c", np.full(sample_count, fixed_abundances.get("c", "+0.00"), dtype=object)),
        "n": abundance_samples.get("n", np.full(sample_count, fixed_abundances.get("n", "+0.00"), dtype=object)),
        "o": abundance_samples.get("o", np.full(sample_count, fixed_abundances.get("o", "+0.00"), dtype=object)),
        "r": abundance_samples.get("r", np.full(sample_count, fixed_abundances.get("r", "+0.00"), dtype=object)),
        "s": abundance_samples.get("s", np.full(sample_count, fixed_abundances.get("s", "+0.00"), dtype=object)),
        "output_mode": np.full(sample_count, output_mode, dtype=object),
        "mode": np.full(sample_count, mode, dtype=object),
        "calculation_mode": np.full(sample_count, calculation_mode, dtype=object),
    }
    return out


def _write_csv_from_columns(columns: Mapping[str, np.ndarray], output_path: str, compression: str | None, level: int | None) -> None:
    with _open_csv_writer(output_path, compression=compression, level=level) as handle:
        writer = csv.writer(handle)
        writer.writerow(list(columns.keys()))
        # Use zip over arrays for row-wise output.
        for row in zip(*[columns[name] for name in columns.keys()]):
            writer.writerow(list(row))


def _compressed_csv_path(output_path: str, compression: str | None) -> str | None:
    compression_norm = (compression or "none").strip().lower()
    if compression_norm in {"none", ""}:
        return None
    if compression_norm in {"gz", "gzip"}:
        return output_path if output_path.endswith(".gz") else f"{output_path}.gz"
    if compression_norm in {"zst", "zstd"}:
        return output_path if output_path.endswith(".zst") else f"{output_path}.zst"
    raise ValueError(f"Unsupported csv_compression={compression!r}. Use none/gzip/zstd.")


def _write_csv_outputs(columns: Mapping[str, np.ndarray], output_path: str, compression: str | None, level: int | None) -> List[str]:
    """
    Write CSV outputs with sane ergonomics.

    - Always writes a plain-text CSV at `output_path` (readable in editors).
    - If compression is requested, also writes a compressed sibling:
      - gzip -> `output_path + ".gz"` (unless already endswith .gz)
      - zstd -> `output_path + ".zst"` (unless already endswith .zst)

    Returns list of written file paths.
    """
    written: List[str] = []
    # Always write a readable CSV to the configured `output_csv` path.
    _write_csv_from_columns(columns, output_path, compression=None, level=None)
    written.append(output_path)

    compressed_path = _compressed_csv_path(output_path, compression=compression)
    if compressed_path and compressed_path != output_path:
        _write_csv_from_columns(columns, compressed_path, compression=compression, level=level)
        written.append(compressed_path)
    return written


def _write_zarr_from_columns(columns: Mapping[str, np.ndarray], zarr_path: str, chunks: int, compressor_cfg: Mapping[str, Any] | None) -> None:
    try:
        import zarr  # type: ignore
        from numcodecs import Blosc, VLenUTF8  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Writing Zarr output requires `zarr` + `numcodecs`. "
            "Install with `pip install zarr numcodecs` (and optionally `zstandard`)."
        ) from exc

    def zarr_store(path: str):
        if hasattr(zarr, "DirectoryStore"):
            return zarr.DirectoryStore(path)  # type: ignore[attr-defined]
        from zarr import storage as zstorage  # type: ignore
        if hasattr(zstorage, "DirectoryStore"):
            return zstorage.DirectoryStore(path)  # type: ignore[attr-defined]
        if hasattr(zstorage, "LocalStore"):
            return zstorage.LocalStore(path)  # type: ignore[attr-defined]
        raise AttributeError("Unsupported Zarr version: cannot find DirectoryStore/LocalStore")

    def compression_kwargs(cfg: Mapping[str, Any] | None) -> Dict[str, Any]:
        if not cfg:
            return {}
        cname = str(cfg.get("cname", "zstd"))
        clevel = int(cfg.get("clevel", 5))
        shuffle_enabled = bool(cfg.get("shuffle", True))
        # Prefer zarr v3 codecs when available.
        try:
            import zarr.codecs as zc  # type: ignore
            if hasattr(zc, "BloscCodec") and hasattr(zc, "BloscShuffle"):
                shuffle = zc.BloscShuffle.bitshuffle if shuffle_enabled else None
                return {"compressors": [zc.BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle)]}
        except Exception:
            pass
        return {
            "compressor": Blosc(
                cname=cname,
                clevel=clevel,
                shuffle=Blosc.BITSHUFFLE if shuffle_enabled else Blosc.NOSHUFFLE,
            )
        }

    os.makedirs(os.path.dirname(os.path.abspath(zarr_path)), exist_ok=True)
    store = zarr_store(zarr_path)
    kwargs = compression_kwargs(compressor_cfg)
    strings_codec = VLenUTF8()

    root = zarr.group(store=store, overwrite=True, zarr_format=3) if hasattr(zarr, "group") else zarr.open_group(store=store, mode="w")  # type: ignore
    for name, values in columns.items():
        if values.dtype == object:
            as_list = values.tolist()
            if hasattr(root, "create_array"):
                import zarr.codecs as zc  # type: ignore
                from zarr.core.dtype.npy.string import VariableLengthUTF8  # type: ignore

                arr = root.create_array(
                    name,
                    shape=(len(as_list),),
                    dtype=VariableLengthUTF8(),
                    serializer=zc.VLenUTF8Codec(),
                    chunks=int(chunks),
                    **kwargs,
                )
                arr[:] = as_list
            else:
                root.array(
                    name,
                    as_list,
                    dtype=object,
                    object_codec=strings_codec,
                    chunks=int(chunks),
                    **kwargs,
                )
        else:
            if hasattr(root, "create_array"):
                root.create_array(name, data=values, chunks=int(chunks), **kwargs)
            else:
                root.array(name, values, chunks=int(chunks), **kwargs)


def generate_grid(
    config_path='configs/sampling/grid_config.yml',
    output_path='runs/local-dev/outputs/grids/parameter_grid.csv',
):
    """
    Generates a CSV parameter grid based on a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
        output_path (str): Path to the output CSV file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return

    sampling_cfg = config.get('sampling', {})
    rng = np.random.default_rng(sampling_cfg.get('seed')) if sampling_cfg.get('seed') is not None else None
    fallback_count = sampling_cfg.get('max_rows')

    atmosphere_cfg = config.get('atmosphere', {})
    abundance_cfg = config.get('abundance', {})

    param_names = ['teff', 'logg', 'feh', 't_value', 'a', 'c', 'n', 'o', 'r', 's']
    parameter_lists = {}
    for name in param_names:
        section = atmosphere_cfg if name in ['teff', 'logg', 'feh', 't_value'] else abundance_cfg
        parameter_lists[name] = _resolve_values(name, section.get(name), rng, is_abundance=name in abundance_cfg or name in ['a', 'c', 'n', 'o', 'r', 's'], fallback_count=fallback_count)

    synthesis_cfg = config.get('synthesis', config)
    lam_min = synthesis_cfg.get('lam_min', 6000)
    lam_max = synthesis_cfg.get('lam_max', 6100)
    lam_step = synthesis_cfg.get('lam_step', 0.01)
    turbvel = config.get('turbvel', "01")
    output_mode = config.get('output_mode', 'Flux')
    mode = config.get('mode', '1D')
    calculation_mode = config.get('calculation_mode', 'LTE')
    grid_version = config.get('grid_version', 'unversioned')

    parameter_combinations = _sample_parameter_space(parameter_lists, sampling_cfg, rng)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'grid_version', 'teff', 'logg', 'feh', 'lam_min', 'lam_max', 'lam_step',
                'turbvel', 't_value', 'a', 'c', 'n', 'o', 'r', 's', 'output_mode', 'mode', 'calculation_mode'
            ])

            for teff, logg, feh, t_val, a_val, c_val, n_val, o_val, r_val, s_val in parameter_combinations:
                writer.writerow([
                    grid_version,
                    int(teff),
                    round(float(logg), 2),
                    round(float(feh), 2),
                    lam_min,
                    lam_max,
                    lam_step,
                    turbvel,
                    t_val,
                    a_val,
                    c_val,
                    n_val,
                    o_val,
                    r_val,
                    s_val,
                    output_mode,
                    mode,
                    calculation_mode
                ])
        print(f"Successfully generated parameter grid at {output_path}")
    except IOError:
        print(f"Error: Could not write to output file {output_path}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    default_json = os.path.join(repo_root, "configs", "sampling", "config_ml_sampling.json")
    default_yaml = os.path.join(repo_root, "configs", "sampling", "grid_config.yml")
    legacy_json = os.path.join(repo_root, "config_ml_sampling.json")
    legacy_yaml = os.path.join(script_dir, "grid_config.yml")
    if not os.path.exists(default_json) and os.path.exists(legacy_json):
        default_json = legacy_json
    if not os.path.exists(default_yaml) and os.path.exists(legacy_yaml):
        default_yaml = legacy_yaml
    default_config = default_json if os.path.exists(default_json) else default_yaml

    parser = argparse.ArgumentParser(
        description=(
            "Generate a parameter grid CSV (and optionally Zarr). "
            "Supports legacy YAML and configs/sampling/config_ml_sampling.json."
        )
    )
    parser.add_argument("--config", default=default_config, help="Path to YAML or JSON config file")
    parser.add_argument("--output", default=None, help="Optional override for CSV output path")
    parser.add_argument("--csv-output", default=None, help="Alias for --output (CSV path)")
    parser.add_argument("--zarr-output", default=None, help="Optional override for Zarr output path")
    parser.add_argument("--no-zarr", action="store_true", help="Do not write Zarr output (even if configured)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)

    if config_path.lower().endswith(".json"):
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)

        csv_out = (
            args.csv_output
            or args.output
            or cfg.get("output_csv")
            or os.path.join(repo_root, "runs", "local-dev", "outputs", "grids", "parameter_grid.csv")
        )
        csv_out_abs = _abs_from(config_dir, csv_out) or os.path.abspath(csv_out)
        zarr_cfg = cfg.get("zarr") or {}
        zarr_out = args.zarr_output or zarr_cfg.get("path")
        zarr_out_abs = _abs_from(config_dir, zarr_out) if zarr_out else None

        seed = cfg.get("seed")
        rng = np.random.default_rng(seed)
        columns = _resolve_ml_sampling(cfg, rng=rng)

        compression = cfg.get("csv_compression")
        compression_level = cfg.get("csv_compression_level")
        written = _write_csv_outputs(columns, csv_out_abs, compression=compression, level=compression_level)
        if len(written) == 1:
            print(f"Successfully generated parameter grid at {written[0]}")
        else:
            print(f"Successfully generated parameter grid at {written[0]}")
            print(f"Successfully generated compressed CSV at {written[1]}")

        if (not args.no_zarr) and zarr_out_abs:
            zarr_chunks = int(zarr_cfg.get("chunks", 2048)) if isinstance(zarr_cfg, Mapping) else 2048
            compressor_cfg = (zarr_cfg.get("compressor") or {}) if isinstance(zarr_cfg, Mapping) else {}
            _write_zarr_from_columns(columns, zarr_out_abs, chunks=zarr_chunks, compressor_cfg=compressor_cfg)
            print(f"Successfully wrote Zarr store at {zarr_out_abs}")
    else:
        # Legacy YAML config path.
        out = args.csv_output or args.output or os.path.join(
            repo_root, "runs", "local-dev", "outputs", "grids", "parameter_grid.csv"
        )
        generate_grid(config_path=config_path, output_path=_abs_from(config_dir, out) or out)
