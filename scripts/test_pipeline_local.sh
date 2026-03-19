#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)
cd "${REPO_ROOT}"

CONFIG_PATH="configs/pipeline/config_pipeline.json"
RUN_ROOT="${REPO_ROOT}/runs/local-smoke"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WITH_SYNTHESIS=0
KEEP_RUN_ROOT=0
WORKERS=1
NUM_SAMPLES=8
LAM_MIN=5000
LAM_MAX=5005
LAM_STEP=0.5

usage() {
  cat <<'EOF'
Usage: scripts/test_pipeline_local.sh [options]

Smoke-test pipeline validity locally.

Options:
  --config PATH         Pipeline config JSON (default: configs/pipeline/config_pipeline.json)
  --run-root PATH       Run root for smoke outputs (default: runs/local-smoke)
  --python PATH         Python executable (default: python3 or $PYTHON_BIN)
  --workers N           Worker count for synthesis smoke (default: 1)
  --num-samples N       Grid sample count for smoke config (default: 8)
  --lam-min X           Smoke wavelength minimum (default: 5000)
  --lam-max X           Smoke wavelength maximum (default: 5005)
  --lam-step X          Smoke wavelength step (default: 0.5)
  --with-synthesis      Also run one shard synthesis smoke test
  --keep                Keep run-root on success (always kept on failure)
  -h, --help            Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --lam-min) LAM_MIN="$2"; shift 2 ;;
    --lam-max) LAM_MAX="$2"; shift 2 ;;
    --lam-step) LAM_STEP="$2"; shift 2 ;;
    --with-synthesis) WITH_SYNTHESIS=1; shift ;;
    --keep) KEEP_RUN_ROOT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

CONFIG_PATH=$(cd "$(dirname "${CONFIG_PATH}")" >/dev/null 2>&1 && pwd)/$(basename "${CONFIG_PATH}")
RUN_ROOT_ABS="${RUN_ROOT}"
if [[ "${RUN_ROOT_ABS}" != /* ]]; then
  RUN_ROOT_ABS="${REPO_ROOT}/${RUN_ROOT_ABS}"
fi

echo "[local-smoke] repo: ${REPO_ROOT}"
echo "[local-smoke] config: ${CONFIG_PATH}"
echo "[local-smoke] run-root: ${RUN_ROOT_ABS}"
echo "[local-smoke] python: ${PYTHON_BIN}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[local-smoke] ERROR: config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[local-smoke] ERROR: python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cleanup() {
  local code=$?
  if [[ $code -ne 0 ]]; then
    echo "[local-smoke] FAILED (exit=${code}). Keeping run-root for debugging: ${RUN_ROOT_ABS}" >&2
    exit $code
  fi
  if [[ "${KEEP_RUN_ROOT}" -eq 0 ]]; then
    rm -rf "${RUN_ROOT_ABS}"
    echo "[local-smoke] cleaned: ${RUN_ROOT_ABS}"
  else
    echo "[local-smoke] kept: ${RUN_ROOT_ABS}"
  fi
}
trap cleanup EXIT

rm -rf "${RUN_ROOT_ABS}"
mkdir -p \
  "${RUN_ROOT_ABS}/config" \
  "${RUN_ROOT_ABS}/logs/pbs" \
  "${RUN_ROOT_ABS}/logs/shards" \
  "${RUN_ROOT_ABS}/outputs/grids" \
  "${RUN_ROOT_ABS}/outputs/shards" \
  "${RUN_ROOT_ABS}/outputs/zarr" \
  "${RUN_ROOT_ABS}/reports/validation" \
  "${RUN_ROOT_ABS}/tmp"

echo "[local-smoke] validating python deps..."
"${PYTHON_BIN}" - <<'PY'
mods = ["numpy", "zarr"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"Missing required modules: {', '.join(missing)}")
print("python deps OK")
PY

echo "[local-smoke] validating JSON configs..."
"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
for p in sorted(Path("configs").rglob("*.json")):
    with p.open() as f:
        json.load(f)
print("json OK")
PY

echo "[local-smoke] building smoke pipeline config..."
SMOKE_CFG="${RUN_ROOT_ABS}/config/pipeline.smoke.json"
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

src = Path("${CONFIG_PATH}")
dst = Path("${SMOKE_CFG}")
run_root = Path("${RUN_ROOT_ABS}")
num_samples = int("${NUM_SAMPLES}")
lam_min = float("${LAM_MIN}")
lam_max = float("${LAM_MAX}")
lam_step = float("${LAM_STEP}")

with src.open() as f:
    cfg = json.load(f)

grid = cfg.setdefault("grid", {})
grid["num_samples"] = num_samples
grid["seed"] = int(grid.get("seed", 1329397429))
syn = grid.setdefault("synthesis", {})
syn["lam_min"] = lam_min
syn["lam_max"] = lam_max
syn["lam_step"] = lam_step
syn.setdefault("output_mode", "Flux")

outs = cfg.setdefault("outputs", {})
outs["grid_csv"] = str(run_root / "outputs" / "grids" / "parameter_grid.csv")
outs["grid_zarr"] = str(run_root / "outputs" / "grids" / "parameter_grid.zarr")
outs["spectra_zarr"] = str(run_root / "outputs" / "zarr" / "synthesized_spectra.zarr")
outs["spectra_shard_template"] = str(run_root / "outputs" / "shards" / "spectra_shard_{shard_index}.zarr")

runtime = cfg.setdefault("runtime", {})
runtime["workers"] = int("${WORKERS}")
runtime["scratch"] = str(run_root / "tmp")
runtime["chunk_rows"] = 4

ts = cfg.setdefault("turbospectrum", {})
ts.setdefault("project_root", "")
paths = ts.setdefault("paths", {})
paths["output_dir"] = str(run_root / "outputs" / "spectra")
paths["log_dir"] = str(run_root / "logs" / "shards")
paths["tmp_dir"] = str(run_root / "tmp")

with dst.open("w") as f:
    json.dump(cfg, f, indent=2, sort_keys=True)
print(dst)
PY

echo "[local-smoke] running grid-only smoke..."
"${PYTHON_BIN}" scripts/pipeline_from_config.py \
  --config "${SMOKE_CFG}" \
  --skip-synthesis

echo "[local-smoke] validating generated grid..."
"${PYTHON_BIN}" - <<PY
import zarr
from pathlib import Path
grid = Path("${RUN_ROOT_ABS}") / "outputs" / "grids" / "parameter_grid.zarr"
root = zarr.open_group(str(grid), mode="r")
required = ["teff", "logg", "feh", "lam_min", "lam_max", "lam_step", "turbvel"]
missing = [k for k in required if k not in root]
if missing:
    raise SystemExit(f"Missing required grid columns: {missing}")
n = int(root["teff"].shape[0])
if n <= 0:
    raise SystemExit("Grid has zero rows")
print(f"grid OK, rows={n}")
PY

if [[ "${WITH_SYNTHESIS}" -eq 1 ]]; then
  echo "[local-smoke] running synthesis smoke (1 shard)..."
  "${PYTHON_BIN}" scripts/pipeline_from_config.py \
    --config "${SMOKE_CFG}" \
    --skip-grid \
    --synthesis-mode sharded \
    --shard-index 0 \
    --shard-count 1 \
    --workers "${WORKERS}" \
    --scratch "${RUN_ROOT_ABS}/tmp" \
    --output "${RUN_ROOT_ABS}/tmp/spectra_shard_0.tmp.zarr"

  SHARD_PATH="${RUN_ROOT_ABS}/outputs/shards/spectra_shard_0.zarr"
  if [[ ! -d "${SHARD_PATH}" ]]; then
    echo "[local-smoke] ERROR: shard output not found: ${SHARD_PATH}" >&2
    exit 1
  fi

  echo "[local-smoke] validating synthesis shard..."
  "${PYTHON_BIN}" scripts/validate_dataset.py \
    --shard-dir "${RUN_ROOT_ABS}/outputs/shards" \
    --expected-shards 1 | tee "${RUN_ROOT_ABS}/reports/validation/smoke_validate.log"
fi

echo "[local-smoke] SUCCESS"
