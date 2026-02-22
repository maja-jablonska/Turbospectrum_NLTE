#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)

# Default runtime location is scratch/system tmp to avoid filling project storage.
SCRATCH_BASE="${SCRATCH:-${TMPDIR:-/tmp}}"
SCRATCH_BASE="${SCRATCH_BASE%/}"
DEFAULT_RUN_ROOT="${RUN_ROOT:-${SCRATCH_BASE}/turbospectrum_nlte/${USER:-user}}"

# Directory for model atmospheres
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/input_files/model_atmospheres/1D/marcs_standard_comp}"

# Directory for continuous opacity files
export OPAC_PATH="${OPAC_PATH:-${PROJECT_ROOT}/COM/contopac}"

# Directory for synthetic spectra
export SPECTRA_PATH="${SPECTRA_PATH:-${DEFAULT_RUN_ROOT}/outputs/spectra}"

# Directory for line lists
export LINELIST_PATH="${LINELIST_PATH:-${PROJECT_ROOT}/input_files/linelists}"

# Path to babsma executable
export BABSMA_EXEC="${BABSMA_EXEC:-${PROJECT_ROOT}/exec-gf/babsma_lu}"

# Path to bsyn executable
export BSYN_EXEC="${BSYN_EXEC:-${PROJECT_ROOT}/exec-gf/bsyn_lu}"

# Path to interpolator executables
export INTERPOL_EXEC="${INTERPOL_EXEC:-${PROJECT_ROOT}/interpolator/interpol_modeles}"
export INTERPOL_NLTE_EXEC="${INTERPOL_NLTE_EXEC:-${PROJECT_ROOT}/interpolator/interpol_modeles_nlte}"
export INTERPOL_3D_EXEC="${INTERPOL_3D_EXEC:-${PROJECT_ROOT}/interpolator/interpol_multi}"
export INTERPOL_3D_NLTE_EXEC="${INTERPOL_3D_NLTE_EXEC:-${PROJECT_ROOT}/interpolator/interpol_multi_nlte}"

# Directory for logs
export LOG_PATH="${LOG_PATH:-${DEFAULT_RUN_ROOT}/logs/shards}"

# Temporary directory for pipeline files
export TMP_PATH="${TMP_PATH:-${DEFAULT_RUN_ROOT}/tmp}"

# Ensure mkstemp() and other tmp-file users write outside the project tree by default.
mkdir -p "${TMP_PATH}"
export TMPDIR="${TMP_PATH}"
export TMP="${TMP_PATH}"
export TEMP="${TMP_PATH}"
