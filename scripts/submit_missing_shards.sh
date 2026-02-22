#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_missing_shards.sh --run-root <path> [options]

Required:
  --run-root <path>         Run root (contains outputs/shards and outputs/grids).

Optional:
  --expected-shards <N>     Total shard count. If omitted, inferred from grid Zarr row count.
  --max-array-size <N>      Max array tasks per qsub submission (default: 20).
  --python-bin <path>       Python executable to use on compute nodes.
  --mamba-env <name>        Conda/mamba env name to activate (default: astro).
  --shards-dir <path>       Override shard directory (default: <run-root>/outputs/shards).
  --grid-zarr <path>        Override grid Zarr (default: <run-root>/outputs/grids/parameter_grid.zarr).
  --config <path>           Synthesis config (default: configs/synthesis/config_sample_comprehensive.json).
  --pbs-script <path>       PBS resume script (default: scripts/pbs_resume_missing_shards_array.pbs).
  --missing-file <path>     Output missing-id file (default: <run-root>/missing_shards.txt).
  --max-attempts <N>        Retries per shard task (default: 3).
  --scratch-tmp-root <path> Scratch tmp root for rerun jobs (default: <run-root>/tmp).
  --qsub-extra "<args>"     Extra qsub arguments (quoted string), e.g. "-q normal -P mk27".
  --dry-run                 Print qsub command but do not submit.
EOF
}

RUN_ROOT=""
EXPECTED_SHARDS=""
SHARDS_DIR=""
GRID_ZARR=""
TS_CONFIG="configs/synthesis/config_sample_comprehensive.json"
PBS_SCRIPT="scripts/pbs_resume_missing_shards_array.pbs"
MISSING_FILE=""
MAX_ATTEMPTS="3"
SCRATCH_TMP_ROOT=""
MAX_ARRAY_SIZE="20"
PYTHON_BIN=""
MAMBA_ENV_NAME="astro"
QSUB_EXTRA=""
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --expected-shards) EXPECTED_SHARDS="$2"; shift 2 ;;
    --shards-dir) SHARDS_DIR="$2"; shift 2 ;;
    --grid-zarr) GRID_ZARR="$2"; shift 2 ;;
    --config) TS_CONFIG="$2"; shift 2 ;;
    --pbs-script) PBS_SCRIPT="$2"; shift 2 ;;
    --missing-file) MISSING_FILE="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --scratch-tmp-root) SCRATCH_TMP_ROOT="$2"; shift 2 ;;
    --max-array-size) MAX_ARRAY_SIZE="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --mamba-env) MAMBA_ENV_NAME="$2"; shift 2 ;;
    --qsub-extra) QSUB_EXTRA="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${RUN_ROOT}" ]]; then
  usage
  exit 2
fi

RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
SHARDS_DIR="${SHARDS_DIR:-${RUN_ROOT}/outputs/shards}"
GRID_ZARR="${GRID_ZARR:-${RUN_ROOT}/outputs/grids/parameter_grid.zarr}"
MISSING_FILE="${MISSING_FILE:-${RUN_ROOT}/missing_shards.txt}"
PBS_LOG_DIR="${RUN_ROOT}/logs/pbs"
CHUNK_DIR="${RUN_ROOT}/missing_shards_chunks"
SANITIZED_MISSING_FILE="${CHUNK_DIR}/missing_shards.numeric.txt"

if [[ ! -d "${SHARDS_DIR}" ]]; then
  echo "ERROR: shard directory not found: ${SHARDS_DIR}" >&2
  exit 2
fi
if [[ ! -d "${GRID_ZARR}" ]]; then
  echo "ERROR: grid zarr not found: ${GRID_ZARR}" >&2
  exit 2
fi
if [[ ! -f "${TS_CONFIG}" ]]; then
  echo "ERROR: config not found: ${TS_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${PBS_SCRIPT}" ]]; then
  echo "ERROR: PBS script not found: ${PBS_SCRIPT}" >&2
  exit 2
fi

mkdir -p "${PBS_LOG_DIR}"
mkdir -p "${CHUNK_DIR}"
rm -f "${CHUNK_DIR}"/missing_chunk_*.txt

if [[ -z "${PYTHON_BIN}" ]] && command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: could not determine python. Activate env first or pass --python-bin <path>." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: --python-bin is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import numpy
import zarr
PY
then
  echo "ERROR: selected python is missing required packages (numpy,zarr): ${PYTHON_BIN}" >&2
  echo "Hint: activate your astro env, then rerun with --python-bin \"\$(which python)\"." >&2
  exit 2
fi

if [[ -z "${EXPECTED_SHARDS}" ]]; then
  EXPECTED_SHARDS="$("${PYTHON_BIN}" - <<PY
import zarr
root = zarr.open_group(r"${GRID_ZARR}", mode="r")
print(int(root["teff"].shape[0]))
PY
)"
fi
if ! [[ "${EXPECTED_SHARDS}" =~ ^[0-9]+$ ]] || (( EXPECTED_SHARDS <= 0 )); then
  echo "ERROR: --expected-shards must be a positive integer, got '${EXPECTED_SHARDS}'." >&2
  exit 2
fi
if ! [[ "${MAX_ARRAY_SIZE}" =~ ^[0-9]+$ ]] || (( MAX_ARRAY_SIZE <= 0 )); then
  echo "ERROR: --max-array-size must be a positive integer, got '${MAX_ARRAY_SIZE}'." >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/find_missing_shards.py \
  --shard-dir "${SHARDS_DIR}" \
  --expected-shards "${EXPECTED_SHARDS}" \
  --output "${MISSING_FILE}"

awk '
  /^[[:space:]]*#/ { next }
  /^[[:space:]]*[0-9]+[[:space:]]*$/ {
    gsub(/[[:space:]]/, "", $0)
    print $0
  }
' "${MISSING_FILE}" > "${SANITIZED_MISSING_FILE}"

MISSING_COUNT="$(wc -l < "${SANITIZED_MISSING_FILE}" | tr -d '[:space:]')"
if (( MISSING_COUNT == 0 )); then
  echo "No missing shards detected. Nothing to submit."
  exit 0
fi

echo "Missing shards: ${MISSING_COUNT}"
echo "Max array size per submit: ${MAX_ARRAY_SIZE}"
echo "Missing file: ${MISSING_FILE}"
echo "Sanitized missing file: ${SANITIZED_MISSING_FILE}"
echo "PBS script: ${PBS_SCRIPT}"
echo "Python bin: ${PYTHON_BIN}"
if [[ -n "${SCRATCH_TMP_ROOT}" ]]; then
  echo "Scratch tmp root: ${SCRATCH_TMP_ROOT}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run enabled: printing chunked qsub commands."
fi

START=0
SUBMITTED=0
CHUNK_ID=0
while (( START < MISSING_COUNT )); do
  END=$(( START + MAX_ARRAY_SIZE - 1 ))
  if (( END >= MISSING_COUNT )); then
    END=$(( MISSING_COUNT - 1 ))
  fi

  CHUNK_FILE="${CHUNK_DIR}/missing_chunk_${CHUNK_ID}.txt"
  sed -n "$((START + 1)),$((END + 1))p" "${SANITIZED_MISSING_FILE}" > "${CHUNK_FILE}"

  CHUNK_COUNT=$(( END - START + 1 ))
  ARRAY_RANGE="0-$((CHUNK_COUNT - 1))"
  VARS="RUN_ROOT=${RUN_ROOT},SHARD_COUNT=${EXPECTED_SHARDS},GRID_ZARR=${GRID_ZARR},TS_CONFIG=${TS_CONFIG},OUT_DIR=${SHARDS_DIR},MISSING_IDS_FILE=${CHUNK_FILE},MAX_ATTEMPTS=${MAX_ATTEMPTS},MAMBA_ENV_NAME=${MAMBA_ENV_NAME}"
  if [[ -n "${SCRATCH_TMP_ROOT}" ]]; then
    VARS="${VARS},SCRATCH_TMP_ROOT=${SCRATCH_TMP_ROOT}"
  fi
  if [[ -n "${PYTHON_BIN}" ]]; then
    VARS="${VARS},PYTHON_BIN=${PYTHON_BIN}"
  fi

  if (( CHUNK_COUNT == 1 )); then
    CMD=(
      qsub
      -r y
      -v "${VARS}"
      -o "${PBS_LOG_DIR}/"
      -e "${PBS_LOG_DIR}/"
    )
    DESC="single task from ${CHUNK_FILE}"
  else
    CMD=(
      qsub
      -r y
      -J "${ARRAY_RANGE}"
      -v "${VARS}"
      -o "${PBS_LOG_DIR}/"
      -e "${PBS_LOG_DIR}/"
    )
    DESC="array ${ARRAY_RANGE} from ${CHUNK_FILE}"
  fi
  if [[ -n "${QSUB_EXTRA}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(${QSUB_EXTRA})
    CMD+=("${EXTRA_ARGS[@]}")
  fi
  CMD+=("${PBS_SCRIPT}")

  echo "Submit chunk ${CHUNK_ID}: ${DESC}"
  printf '  %q' "${CMD[@]}"
  echo

  if [[ "${DRY_RUN}" != "1" ]]; then
    JOB_SUBMIT_OUT="$("${CMD[@]}")"
    echo "Submitted: ${JOB_SUBMIT_OUT}"
    SUBMITTED=$((SUBMITTED + 1))
  fi

  START=$(( END + 1 ))
  CHUNK_ID=$(( CHUNK_ID + 1 ))
done

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

echo "Submitted chunks: ${SUBMITTED}"
echo "PBS logs: ${PBS_LOG_DIR}"
echo "Per-shard resume logs: ${RUN_ROOT}/logs/shards_resume"
