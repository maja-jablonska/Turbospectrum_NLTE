#!/bin/bash

# Get the directory of this script to make paths relative to it
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
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
run_synthesis() {
    teff=$1
    logg=$2
    feh=$3
    lam_min=$4
    lam_max=$5
    lam_step=$6
    turbvel=$7
    t_value=$8
    a_val=$9
    c_val=${10}
    n_val=${11}
    o_val=${12}
    r_val=${13}
    s_val=${14}
    output_mode=${15}
    mode=${16}
    calculation_mode=${17}

    logg_fmt=$(printf "%+.1f" "$logg")
    feh_fmt=$(printf "%+.2f" "$feh")

        model_name="p${teff}_g${logg_fmt}_m0.0_t${t_value}_st_z${feh_fmt}_a${a_val}_c${c_val}_n${n_val}_o${o_val}_r${r_val}_s${s_val}.mod"
    log_file="$LOG_PATH/${model_name}_${calculation_type}_${mode}_${calculation_mode}.log"

    # Set NLTE and SPHERICAL flags based on the mode
    nlte_flag=".false."
    if [ "$calculation_mode" == "NLTE" ]; then
        nlte_flag=".true."
    fi

    spherical_flag=".false."
    if [ "$mode" == "3D" ]; then
        spherical_flag=".true."
    fi

    # Check if the model atmosphere file exists before starting
    if [ ! -f "$MODEL_PATH/${model_name}" ]; then
        echo "ERROR: Model file not found: $MODEL_PATH/${model_name}" > "$log_file"
        return 1
    fi

    echo "INFO: Starting $calculation_type synthesis for $model_name ($mode, $calculation_mode)" > "$log_file"

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
'ALPHA/Fe   :'    '$a_val'
'HELIUM     :'    '0.00' # Note: 'c', 'n', 'o' abundances are not directly mapped here.
'R-PROCESS  :'    '$r_val'
'S-PROCESS  :'    '$s_val'
'INDIVIDUAL ABUNDANCES:'   '0'
'XIFIX:' 'T'
$turbvel
EOF

    if [ $? -ne 0 ]; then
        echo "ERROR: babsma failed for $model_name. See log for details." >> "$log_file"
        return 1
    fi

    # Run bsyn
    echo "Running bsyn for $model_name" >> "$log_file"
    "$BSYN_EXEC" <<EOF >> "$log_file" 2>&1
'NLTE :'          '$nlte_flag'
'NLTEINFOFILE:'  'DATA/SPECIES_LTE_NLTE.dat'
'LAMBDA_MIN:'     '$lam_min'
'LAMBDA_MAX:'     '$lam_max'
'LAMBDA_STEP:'    '$lam_step'
'INTENSITY/FLUX:' '$calculation_type'
'MODELOPAC:' '$OPAC_PATH/${model_name}opac'
'RESULTFILE :' '$SPECTRA_PATH/${model_name}.${calculation_type}.${mode}.${calculation_mode}.spec'
'ABUND_SOURCE:'   'magg'
'METALLICITY:'    '$feh'
'ALPHA/Fe   :'    '$a_val'
'HELIUM     :'    '0.00' # Note: 'c', 'n', 'o' abundances are not directly mapped here.
'R-PROCESS  :'    '$r_val'
'S-PROCESS  :'    '$s_val'
'INDIVIDUAL ABUNDANCES:'   '0'
'ISOTOPES : ' '0'
'LIST_OF_LINELISTS:' '$LINELIST_PATH/list.list'
'SPHERICAL:'  '$spherical_flag'
  30
  300.00
  15
  1.30
EOF

    if [ $? -eq 0 ]; then
        echo "INFO: Successfully finished $calculation_type synthesis for $model_name" >> "$log_file"
    else
        echo "ERROR: bsyn failed for $model_name. See log for details." >> "$log_file"
        return 1
    fi
}

export -f run_synthesis

CPU_COUNT=$(get_cpu_count)
echo "INFO: Starting spectra synthesis with up to $CPU_COUNT parallel processes."

# Read the parameter grid (skip header), then process each line in parallel
tail -n +2 "scripts/parameter_grid.csv" | while IFS=, read -r teff logg feh lam_min lam_max lam_step turbvel t_value a_val c_val n_val o_val r_val s_val output_mode mode calculation_mode
do
    # Remove carriage return characters from variables
    calculation_mode=$(echo "$calculation_mode" | tr -d '\r')
    mode=$(echo "$mode" | tr -d '\r')

    # Limit the number of concurrent jobs
    if [[ $(jobs -r -p | wc -l) -ge $CPU_COUNT ]]; then
        wait # Wait for any background job to finish
    fi
    run_synthesis "$teff" "$logg" "$feh" "$lam_min" "$lam_max" "$lam_step" "$turbvel" "$t_value" "$output_mode" "$mode" "$calculation_mode" &

done

# Wait for all remaining background jobs to complete
wait

echo "INFO: Spectra synthesis finished. Logs are in $LOG_PATH"
