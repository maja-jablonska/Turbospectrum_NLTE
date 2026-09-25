#!/bin/bash
# Submit a full array-mode production run as three chained PBS jobs:
#
#   1. grid      SKIP_SYNTHESIS=1            builds the grid zarr once
#   2. shards    -J 0-(N-1), afterok:grid    one shard per array task
#   3. finalize  FINALIZE_ONLY=1, afterany   validate + merge whatever finished
#
# finalize uses afterany so it also runs when some shard tasks failed: it then
# writes a partial merge and exits 3 (see turbospectrum_pipeline_example.pbs),
# and a plain resubmission of this script picks up only the missing shards.
#
# Usage (login node):
#   scripts/submit_pipeline.sh CONFIG_PATH SHARD_COUNT [extra -v vars, e.g. RUN_ROOT=/scratch/...,WORKERS=48]
#
# Extra qsub options for all three jobs: QSUB_EXTRA="-l walltime=24:00:00".
# Set DRY_RUN=1 to print the qsub commands without submitting.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

CONFIG_PATH="$1"
SHARD_COUNT="$2"
EXTRA_VARS="${3:-}"
QSUB_BIN="${QSUB_BIN:-qsub}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
PBS_SCRIPT="${REPO}/turbospectrum_pipeline_example.pbs"

if ! [[ "${SHARD_COUNT}" =~ ^[0-9]+$ ]] || (( SHARD_COUNT < 1 )); then
    echo "SHARD_COUNT must be a positive integer, got '${SHARD_COUNT}'" >&2
    exit 2
fi
[[ "${CONFIG_PATH}" == /* ]] || CONFIG_PATH="${REPO}/${CONFIG_PATH}"
[[ -f "${CONFIG_PATH}" ]] || { echo "config not found: ${CONFIG_PATH}" >&2; exit 2; }

VARS="CONFIG_PATH=${CONFIG_PATH},SHARD_COUNT=${SHARD_COUNT}"
[[ -n "${EXTRA_VARS}" ]] && VARS+=",${EXTRA_VARS}"
# shellcheck disable=SC2206
EXTRA=(${QSUB_EXTRA:-})

submit() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "${QSUB_BIN} $*" >&2
        echo "DRYRUN.$RANDOM"
    else
        "${QSUB_BIN}" "$@"
    fi
}

cd "${REPO}"  # PBS_O_WORKDIR for the jobs
grid_id="$(submit ${EXTRA[@]+"${EXTRA[@]}"} -N ts_grid -v "${VARS},SKIP_SYNTHESIS=1" "${PBS_SCRIPT}")"
array_id="$(submit ${EXTRA[@]+"${EXTRA[@]}"} -N ts_shards -W depend=afterok:"${grid_id}" \
    -J "0-$((SHARD_COUNT - 1))" -v "${VARS}" "${PBS_SCRIPT}")"
final_id="$(submit ${EXTRA[@]+"${EXTRA[@]}"} -N ts_final -W depend=afterany:"${array_id}" \
    -v "${VARS},FINALIZE_ONLY=1" "${PBS_SCRIPT}")"

echo "grid=${grid_id}"
echo "shards=${array_id}"
echo "finalize=${final_id}"
