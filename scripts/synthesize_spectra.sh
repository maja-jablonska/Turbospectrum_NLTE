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

DEFAULT_GRID_CSV="${DEFAULT_RUN_ROOT}/outputs/grids/parameter_grid.csv"
LEGACY_GRID_CSV="scripts/parameter_grid.csv"
PARAMETER_GRID_CSV="${PARAMETER_GRID_CSV:-$DEFAULT_GRID_CSV}"
if [ ! -f "$PARAMETER_GRID_CSV" ] && [ -f "$LEGACY_GRID_CSV" ]; then
    PARAMETER_GRID_CSV="$LEGACY_GRID_CSV"
fi
if [ ! -f "$PARAMETER_GRID_CSV" ]; then
    echo "ERROR: Parameter grid CSV not found at '$PARAMETER_GRID_CSV' (also checked '$LEGACY_GRID_CSV')."
    exit 1
fi

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
    grid_version=$1
    teff=$2
    logg=$3
    feh=$4
    lam_min=$5
    lam_max=$6
    lam_step=$7
    turbvel=$8
    t_value=$9
    a_val=${10}
    c_val=${11}
    n_val=${12}
    o_val=${13}
    r_val=${14}
    s_val=${15}
    output_mode=${16}
    mode=${17}
    calculation_mode=${18}

    calculation_type="$output_mode"

    logg_fmt=$(printf "%+.2f" "$logg")
    feh_fmt=$(printf "%+.2f" "$feh")

    model_name="p${teff}_g${logg_fmt}_m0.0_t${t_value}_st_z${feh_fmt}_a${a_val}_c${c_val}_n${n_val}_o${o_val}_r${r_val}_s${s_val}.mod"
    log_file="$LOG_PATH/${grid_version}_${model_name}_${calculation_type}_${mode}_${calculation_mode}.log"

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
tail -n +2 "$PARAMETER_GRID_CSV" | while IFS=, read -r c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18
do
    if [[ "$c1" =~ ^[0-9]+$ ]]; then
        grid_version="legacy"
        teff="$c1"
        logg="$c2"
        feh="$c3"
        lam_min="$c4"
        lam_max="$c5"
        lam_step="$c6"
        turbvel="$c7"
        t_value="$c8"
        a_val="$c9"
        c_val="$c10"
        n_val="$c11"
        o_val="$c12"
        r_val="$c13"
        s_val="$c14"
        output_mode="$c15"
        mode="$c16"
        calculation_mode="$c17"
    else
        grid_version="$c1"
        teff="$c2"
        logg="$c3"
        feh="$c4"
        lam_min="$c5"
        lam_max="$c6"
        lam_step="$c7"
        turbvel="$c8"
        t_value="$c9"
        a_val="$c10"
        c_val="$c11"
        n_val="$c12"
        o_val="$c13"
        r_val="$c14"
        s_val="$c15"
        output_mode="$c16"
        mode="$c17"
        calculation_mode="$c18"
    fi

    # Remove carriage return characters from variables
    calculation_mode=$(echo "$calculation_mode" | tr -d '\r')
    mode=$(echo "$mode" | tr -d '\r')
    output_mode=$(echo "$output_mode" | tr -d '\r')
    grid_version=$(echo "$grid_version" | tr -d '\r')

    # Limit the number of concurrent jobs
    if [[ $(jobs -r -p | wc -l) -ge $CPU_COUNT ]]; then
        wait # Wait for any background job to finish
    fi
    run_synthesis "$grid_version" "$teff" "$logg" "$feh" "$lam_min" "$lam_max" "$lam_step" "$turbvel" "$t_value" "$a_val" "$c_val" "$n_val" "$o_val" "$r_val" "$s_val" "$output_mode" "$mode" "$calculation_mode" &

done

# Wait for all remaining background jobs to complete
wait

echo "INFO: Spectra synthesis finished. Logs are in $LOG_PATH"
