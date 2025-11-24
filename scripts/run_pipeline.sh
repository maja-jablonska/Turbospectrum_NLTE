#!/bin/bash

# Get the directory of this script to make paths relative to it
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Change to the project root directory to ensure all paths are correct
cd "$PROJECT_ROOT" || exit 1

# Load environment variables
if [ -f "scripts/env.sh" ]; then
    source "scripts/env.sh"
else
    echo "ERROR: Environment file not found at scripts/env.sh"
    exit 1
fi

# Create necessary directories if they don't exist
mkdir -p "$LOG_PATH"
mkdir -p "$TMP_PATH"
mkdir -p "$MODEL_PATH"
mkdir -p "$OPAC_PATH"
mkdir -p "$SPECTRA_PATH"

# Function to get CPU count for Linux and macOS
get_cpu_count() {
    if [[ "$(uname)" == "Darwin" ]]; then
        sysctl -n hw.ncpu
    else
        nproc
    fi
}

# Function to run a single pipeline instance
run_instance() {
    teff=$1
    logg=$2
    feh=$3
    lam_min=$4
    lam_max=$5
    lam_step=$6
    turbvel=$7

    # Format parameters to match the expected model filename format (e.g., g+4.40, z+0.00)
    logg_fmt=$(printf "%+1.2f" "$logg")
    feh_fmt=$(printf "%+1.2f" "$feh")

    model_name="p${teff}_g${logg_fmt}_m0.0_t01_st_z${feh_fmt}_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod"
    log_file="$LOG_PATH/${model_name}.log"

    # Check if the model atmosphere file exists before starting
    if [ ! -f "$MODEL_PATH/${model_name}" ]; then
        echo "ERROR: Model file not found: $MODEL_PATH/${model_name}" > "$log_file"
        return 1
    fi

    echo "INFO: Starting pipeline for $model_name" > "$log_file"

    # Run babsma
    echo "Running babsma for $model_name" >> "$log_file"
    "$BABSMA_EXEC" <<EOF >> "$log_file" 2>&1
'LAMBDA_MIN:'  '$lam_min'
'LAMBDA_MAX:'  '$lam_max'
'LAMBDA_STEP:' '$lam_step'
'MODELINPUT:' '$MODEL_PATH/${model_name}'
'MARCS-FILE:' '.true.'
'MODELOPAC:' '$OPAC_PATH/${model_name}opac'
'ABUND_SOURCE:' 'magg'
'METALLICITY:'    '$feh'
'ALPHA/Fe   :'    '0.00'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
'INDIVIDUAL ABUNDANCES:'   '0'
'XIFIX:' 'T'
$turbvel
EOF

    # Check if babsma was successful before proceeding
    if [ $? -ne 0 ]; then
        echo "ERROR: babsma failed for $model_name. See log for details." >> "$log_file"
        return 1
    fi

    # Run bsyn
    echo "Running bsyn for $model_name" >> "$log_file"
    "$BSYN_EXEC" <<EOF >> "$log_file" 2>&1
'NLTE :'          '.false.'
'NLTEINFOFILE:'  'DATA/SPECIES_LTE_NLTE.dat'
'LAMBDA_MIN:'     '$lam_min'
'LAMBDA_MAX:'     '$lam_max'
'LAMBDA_STEP:'    '$lam_step'
'INTENSITY/FLUX:' 'Flux'
'MODELOPAC:' '$OPAC_PATH/${model_name}opac'
'RESULTFILE :' '$SPECTRA_PATH/${model_name}.spec'
'ABUND_SOURCE:'   'magg'
'METALLICITY:'    '$feh'
'ALPHA/Fe   :'    '0.00'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
'INDIVIDUAL ABUNDANCES:'   '0'
'ISOTOPES : ' '0'
'LIST_OF_LINELISTS:' '$LINELIST_PATH/list.list'
'SPHERICAL:'  'F'
  30
  300.00
  15
  1.30
EOF
    if [ $? -eq 0 ]; then
        echo "INFO: Successfully finished pipeline for $model_name" >> "$log_file"
    else
        echo "ERROR: bsyn failed for $model_name. See log for details." >> "$log_file"
        return 1
    fi
}

export -f run_instance

CPU_COUNT=$(get_cpu_count)
echo "INFO: Starting pipeline with up to $CPU_COUNT parallel processes."

# Read the parameter grid (skip header), then process each line in parallel
tail -n +2 "scripts/parameter_grid.csv" | while IFS=, read -r teff logg feh lam_min lam_max lam_step turbvel
do
    run_instance "$teff" "$logg" "$feh" "$lam_min" "$lam_max" "$lam_step" "$turbvel" &

    # Limit the number of concurrent jobs
    if [[ $(jobs -r -p | wc -l) -ge $CPU_COUNT ]]; then
        wait -n
    fi
done

# Wait for all remaining background jobs to complete
wait

echo "INFO: Pipeline finished. Logs are in $LOG_PATH"