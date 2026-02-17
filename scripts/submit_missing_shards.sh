#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_missing_shards.sh --run-root <path> --expected-shards <N> [options]

Required:
  --run-root <path>         Run root (contains outputs/shards and outputs/grids).
  --expected-shards <N>     Total shard count (e.g. 1000).

Optional:
  --shards-dir <path>       Override shard directory (default: <run-root>/outputs/shards).
  --grid-zarr <path>        Override grid Zarr (default: <run-root>/outputs/grids/parameter_grid.zarr).
  --config <path>           Synthesis config (default: configs/synthesis/config_sample_comprehensive.json).
  --pbs-script <path>       PBS resume script (default: scripts/pbs_resume_missing_shards_array.pbs).
  --missing-file <path>     Output missing-id file (default: <run-root>/missing_shards.txt).
  --max-attempts <N>        Retries per shard task (default: 3).
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
    --qsub-extra) QSUB_EXTRA="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${RUN_ROOT}" || -z "${EXPECTED_SHARDS}" ]]; then
  usage
  exit 2
fi
if ! [[ "${EXPECTED_SHARDS}" =~ ^[0-9]+$ ]] || (( EXPECTED_SHARDS <= 0 )); then
  echo "ERROR: --expected-shards must be a positive integer." >&2
  exit 2
fi

RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
SHARDS_DIR="${SHARDS_DIR:-${RUN_ROOT}/outputs/shards}"
GRID_ZARR="${GRID_ZARR:-${RUN_ROOT}/outputs/grids/parameter_grid.zarr}"
MISSING_FILE="${MISSING_FILE:-${RUN_ROOT}/missing_shards.txt}"

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

python scripts/find_missing_shards.py \
  --shard-dir "${SHARDS_DIR}" \
  --expected-shards "${EXPECTED_SHARDS}" \
  --output "${MISSING_FILE}"

MISSING_COUNT="$(wc -l < "${MISSING_FILE}" | tr -d '[:space:]')"
if (( MISSING_COUNT == 0 )); then
  echo "No missing shards detected. Nothing to submit."
  exit 0
fi

ARRAY_RANGE="0-$((MISSING_COUNT - 1))"
VARS="RUN_ROOT=${RUN_ROOT},SHARD_COUNT=${EXPECTED_SHARDS},GRID_ZARR=${GRID_ZARR},TS_CONFIG=${TS_CONFIG},OUT_DIR=${SHARDS_DIR},MISSING_IDS_FILE=${MISSING_FILE},MAX_ATTEMPTS=${MAX_ATTEMPTS}"

CMD=(qsub -J "${ARRAY_RANGE}" -v "${VARS}")
if [[ -n "${QSUB_EXTRA}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${QSUB_EXTRA})
  CMD+=("${EXTRA_ARGS[@]}")
fi
CMD+=("${PBS_SCRIPT}")

echo "Missing shards: ${MISSING_COUNT}"
echo "Array range: ${ARRAY_RANGE}"
echo "Missing file: ${MISSING_FILE}"
echo "Submit command:"
printf '  %q' "${CMD[@]}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${CMD[@]}"
