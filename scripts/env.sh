#!/bin/bash

# Directory for model atmospheres
export MODEL_PATH="/Users/mjablons/Documents/Turbospectrum_NLTE/input_files/model_atmospheres/1D/marcs_standard_comp"

# Directory for continuous opacity files
export OPAC_PATH="$PWD/COM/contopac"

# Directory for synthetic spectra
export SPECTRA_PATH="/Users/mjablons/Documents/Turbospectrum_NLTE/spectra"

# Directory for line lists
export LINELIST_PATH="/Users/mjablons/Documents/Turbospectrum_NLTE/input_files/linelists"

# Path to babsma executable
export BABSMA_EXEC="$PWD/exec-gf/babsma_lu"

# Path to bsyn executable
export BSYN_EXEC="$PWD/exec-gf/bsyn_lu"

# Path to interpolator executables
export INTERPOL_EXEC="$PWD/interpolator/interpol_modeles"
export INTERPOL_NLTE_EXEC="$PWD/interpolator/interpol_modeles_nlte"
export INTERPOL_3D_EXEC="$PWD/interpolator/interpol_multi"
export INTERPOL_3D_NLTE_EXEC="$PWD/interpolator/interpol_multi_nlte"

# Directory for logs
export LOG_PATH="$PWD/logs"

# Temporary directory for pipeline files
export TMP_PATH="$PWD/tmp"
