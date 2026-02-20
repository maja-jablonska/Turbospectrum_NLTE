from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dagster import AssetSelection, Definitions, ScheduleDefinition, asset, define_asset_job

# Resolve repository root dynamically from this file location.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


@asset(
    description=(
        "Generate parameter grid outputs from the pipeline config "
        "(grid-only mode, no synthesis)."
    )
)
def build_grid() -> None:
    python_bin = _env_or_default("TS_PYTHON_BIN", sys.executable)
    pipeline_config = _env_or_default(
        "TS_PIPELINE_CONFIG",
        "configs/pipeline/config_pipeline.json",
    )

    cmd = [
        python_bin,
        "scripts/pipeline_from_config.py",
        "--config",
        pipeline_config,
        "--skip-synthesis",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


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

