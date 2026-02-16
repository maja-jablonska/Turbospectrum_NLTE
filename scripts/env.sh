#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)
DEFAULT_RUN_ROOT="${PROJECT_ROOT}/runs/local-dev"

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
