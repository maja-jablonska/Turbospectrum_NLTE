import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dagster import (
    AssetExecutionContext,
    AssetSelection,
    Config,
    Definitions,
    MaterializeResult,
    ScheduleDefinition,
    asset,
    define_asset_job,
)
from pydantic import Field

# Resolve repository root dynamically from this file location.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _slugify(name: str) -> str:
    text = name.strip()
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or "grid"


class GridDefinitionConfig(Config):
    """Grid definition exposed in Dagster Launchpad."""

    template_config: str = "configs/pipeline/config_pipeline.json"
    grid_name: str = "ui-grid"
    num_samples: int = 1000
    seed: int = 1329397429

    teff_min: float = 4000.0
    teff_max: float = 7000.0
    logg_min: float = 0.0
    logg_max: float = 5.0
    feh_min: float = -2.5
    feh_max: float = 0.5

    sample_turbvel: bool = True
    turbvel_options: list[str] = Field(default_factory=lambda: ["01", "02", "03", "04", "05"])
    t_value_options: list[str] = Field(default_factory=lambda: ["01"])
    abundances: dict[str, Any] = Field(
        default_factory=lambda: {
            "a": "+0.00",
            "c": "+0.00",
            "n": "+0.00",
            "o": "+0.00",
            "r": "+0.00",
            "s": "+0.00",
        }
    )

    lam_min: float = 4000.0
    lam_max: float = 8000.0
    lam_step: float = 0.01
    output_mode: str = "Flux"
    mode: str = "1D"
    calculation_mode: str = "LTE"

    output_root: str = "runs/local-dev/outputs/grids"
    output_stem: str | None = None

    csv_compression: str = "zstd"
    csv_compression_level: int = 5
    zarr_chunks: int = 2048
    zarr_compressor_cname: str = "zstd"
    zarr_compressor_clevel: int = 5
    zarr_compressor_shuffle: bool = True


def _build_pipeline_config_from_ui(
    config: GridDefinitionConfig,
) -> tuple[dict[str, Any], Path, Path]:
    template_path = _resolve_repo_path(config.template_config)
    pipeline_cfg = _load_json(template_path)
    grid_cfg = dict(pipeline_cfg.get("grid") or {})
    outputs_cfg = dict(pipeline_cfg.get("outputs") or {})

    grid_cfg["grid_version"] = config.grid_name
    grid_cfg["num_samples"] = config.num_samples
    grid_cfg["seed"] = config.seed
    grid_cfg["bounds"] = {
        "teff": {"min": config.teff_min, "max": config.teff_max},
        "logg": {"min": config.logg_min, "max": config.logg_max},
        "feh": {"min": config.feh_min, "max": config.feh_max},
    }
    grid_cfg["sample_turbvel"] = config.sample_turbvel
    grid_cfg["turbvel_options"] = config.turbvel_options
    grid_cfg["t_value_options"] = config.t_value_options
    grid_cfg["abundances"] = config.abundances
    grid_cfg["synthesis"] = {
        "lam_min": config.lam_min,
        "lam_max": config.lam_max,
        "lam_step": config.lam_step,
        "output_mode": config.output_mode,
    }
    grid_cfg["mode"] = config.mode
    grid_cfg["calculation_mode"] = config.calculation_mode
    grid_cfg["csv_compression"] = config.csv_compression
    grid_cfg["csv_compression_level"] = config.csv_compression_level
    grid_cfg["zarr"] = {
        "chunks": config.zarr_chunks,
        "compressor": {
            "cname": config.zarr_compressor_cname,
            "clevel": config.zarr_compressor_clevel,
            "shuffle": config.zarr_compressor_shuffle,
        },
    }

    output_root = _resolve_repo_path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = _slugify(config.output_stem or config.grid_name)
    grid_csv = output_root / f"{stem}.csv"
    grid_zarr = output_root / f"{stem}.zarr"
    outputs_cfg["grid_csv"] = str(grid_csv)
    outputs_cfg["grid_zarr"] = str(grid_zarr)

    pipeline_cfg["grid"] = grid_cfg
    pipeline_cfg["outputs"] = outputs_cfg
    return pipeline_cfg, grid_csv, grid_zarr


def _write_temp_pipeline_config(pipeline_cfg: dict[str, Any]) -> Path:
    tmp_dir = REPO_ROOT / "runs" / "local-dev" / "tmp" / "dagster"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"pipeline_ui.{int(time.time())}.{os.getpid()}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(pipeline_cfg, handle, indent=2, sort_keys=True)
    return path


@asset(
    description=(
        "Generate parameter grid outputs from a Dagster UI-provided definition "
        "(grid-only mode, no synthesis)."
    )
)
def build_grid(context: AssetExecutionContext, config: GridDefinitionConfig) -> MaterializeResult:
    python_bin = _env_or_default("TS_PYTHON_BIN", sys.executable)
    pipeline_cfg, grid_csv, grid_zarr = _build_pipeline_config_from_ui(config)
    pipeline_config_path = _write_temp_pipeline_config(pipeline_cfg)

    cmd = [
        python_bin,
        "scripts/pipeline_from_config.py",
        "--config",
        str(pipeline_config_path),
        "--skip-synthesis",
    ]
    context.log.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return MaterializeResult(
        metadata={
            "grid_name": config.grid_name,
            "pipeline_config_path": str(pipeline_config_path),
            "grid_csv": str(grid_csv),
            "grid_zarr": str(grid_zarr),
        }
    )


pipeline_job = define_asset_job(
    name="pipeline_job",
    selection=AssetSelection.assets(build_grid),
)


daily_schedule = ScheduleDefinition(
    job=pipeline_job,
    cron_schedule="0 2 * * *",
)


defs = Definitions(
    assets=[build_grid],
    jobs=[pipeline_job],
    schedules=[daily_schedule],
)
