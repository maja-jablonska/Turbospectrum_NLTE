#!/bin/bash
# set -x # Enable debugging

# Get the directory of this script to make paths relative to it
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# --- USER CONFIGURATION: DEFINE YOUR MODEL GRID HERE ---
TEFF_GRID=(5000 5250 5500 5750 6000 6250)
LOGG_GRID=(3.5 4.0 4.5 5.0)
FEH_GRID=(-1.0 -0.5 0.0 0.5)
# --------------------------------------------------------

cd "$PROJECT_ROOT" || exit 1

if [ -f "scripts/env.sh" ]; then
    source "scripts/env.sh"
else
    echo "ERROR: Environment file not found at scripts/env.sh"
    exit 1
fi

mkdir -p "$LOG_PATH"
mkdir -p "$MODEL_PATH"

# Function to find bracketing values for a target in a grid array.
find_bracketing_values() {
    local target=$1
    shift
    local grid=("$@")
    local i

    if (( $(echo "$target <= ${grid[0]}" | bc -l) )); then
        echo "${grid[0]} ${grid[1]}"
        return
    fi

    local last_idx=$((${#grid[@]} - 1))
    if (( $(echo "$target >= ${grid[$last_idx]}" | bc -l) )); then
        local second_last_idx=$((${#grid[@]} - 2))
        echo "${grid[$second_last_idx]} ${grid[$last_idx]}"
        return
    fi

    for ((i=0; i<${#grid[@]}-1; i++)); do
        if (( $(echo "$target >= ${grid[$i]}" | bc -l) )) && (( $(echo "$target < ${grid[$i+1]}" | bc -l) )); then
            echo "${grid[$i]} ${grid[$i+1]}"
            return
        fi
    done
}

# Function to run a single interpolation instance
run_interpolation() {
    teff=$1
    logg=$2
    feh=$3
    turbvel=$4
    t_value=$5
    a_val=$6
    c_val=$7
    n_val=$8
    o_val=$9
    r_val=${10}
    s_val=${11}
    mode=${12}
    calculation_mode=${13}
    output_model_name=${14}

    read teff_low teff_high < <(find_bracketing_values $teff "${TEFF_GRID[@]}")
    read logg_low logg_high < <(find_bracketing_values $logg "${LOGG_GRID[@]}")
    read feh_low feh_high < <(find_bracketing_values $feh "${FEH_GRID[@]}")

    echo "INFO: Bracketing Teff: $teff_low, $teff_high"
    echo "INFO: Bracketing logg: $logg_low, $logg_high"
    echo "INFO: Bracketing [Fe/H]: $feh_low, $feh_high"

    logg_low_fmt=$(printf "%+.1f" "$logg_low")
    logg_high_fmt=$(printf "%+.1f" "$logg_high")
    feh_low_fmt=$(printf "%+.2f" "$feh_low")
    feh_high_fmt=$(printf "%+.2f" "$feh_high")

    # Define the 8 bracketing model files
    local bracketing_models=(
        "$MODEL_PATH/p${teff_low}_g${logg_low_fmt}_m0.0_t${turbvel}_st_z${feh_low_fmt}_a${a_val}_c${c_val}_n${n_val}_o${o_val}_r${r_val}_s${s_val}.mod"
        "$MODEL_PATH/p${teff_low}_g${logg_low_fmt}_m0.0_t${turbvel}_st_z${feh_high_fmt}_a${a_val}_c${c_val}_n${n_val}_o${o_val}_r${r_val}_s${s_val}.mod"
        "$MODEL_PATH/p${teff_low}_g${logg_high_fmt}_m0.0_t${turbvel}_st_z${feh_low_fmt}_a${a_val}_c${c_val}_n${n_val}_o${o_val}_r${r_val}_s${s_val}.mod"
        "$MODEL_PATH/p${teff_low}_g${logg_high_fmt}_m0.0_t${turbvel}_st_z${feh_high_fmt}_a${a_val}_c${c_val}_n${n_val}_o${o_val}_r${r_val}_s${s_val}.mod"
        "$MODEL_PATH/p${teff_high}_g${logg_low_fmt}_m0.0_t${turbvel}_st_z${feh_low_fmt}_a${a_val}_c${c_val}_n${n_val}_o${o_val}_r${r_val}_s${s_val}.mod"
        "$MODEL_PATH/p${teff_high}_g${logg_low_fmt}_m0.0_t${turbvel}_st_z${feh_high_fmt}_a${a_val}_c${c_val}_n${n_val}_o${o_val}_r${r_val}_s${s_val}.mod"
        "$MODEL_PATH/p${teff_high}_g${logg_high_fmt}_m0.0_t${turbvel}_st_z${feh_low_fmt}_a${a_val}_c${c_val}_n${n_val}_o${o_val}_r${r_val}_s${s_val}.mod"
        "$MODEL_PATH/p${teff_high}_g${logg_high_fmt}_m0.0_t${turbvel}_st_z${feh_high_fmt}_a${a_val}_c${c_val}_n${n_val}_o${o_val}_r${r_val}_s${s_val}.mod"
    )

    # *** CRITICAL: Verify that all input models exist before proceeding ***
    for model_file in "${bracketing_models[@]}"; do
        if [ ! -f "$model_file" ]; then
            echo "FATAL ERROR: Prerequisite model file for interpolation not found:"
            echo "  Expected: $model_file"
            echo "  Please check the exact filename in your '$MODEL_PATH' directory."
            echo "  If the file exists but has a slightly different name (e.g., formatting of logg or feh), please provide the exact name of the existing file."
            echo "  You may need to run './scripts/download_data.sh --atmospheres MARCS' again if the file is genuinely missing."
            return 1
        fi
    done

    log_file="$LOG_PATH/${output_model_name}_${mode}_${calculation_mode}.log"
    local input_list=""
    for model_file in "${bracketing_models[@]}"; do
        input_list+="'$model_file'\n"
    done

    case "${mode}_${calculation_mode}" in
        1D_LTE | 3D_LTE)
            [ "$mode" == "1D_LTE" ] && interpol_exec="$INTERPOL_EXEC" || interpol_exec="$INTERPOL_3D_EXEC"
            input="${input_list}'$MODEL_PATH/$output_model_name'\n'$TMP_PATH/${output_model_name}.alt'\n$teff\n$logg\n$feh\n.false.\n.false.\n''"
            ;;
        1D_NLTE | 3D_NLTE)
            [ "$mode" == "1D_NLTE" ] && interpol_exec="$INTERPOL_NLTE_EXEC" || interpol_exec="$INTERPOL_3D_NLTE_EXEC"
            # NOTE: This part is still hardcoded and may need adjustment
            nlte_bin_file="DATA/H/output_NLTEgrid4TS_May-21-2021.bin"
            nlte_aux_file="DATA/H/auxData_NLTEgrid4TS_May-21-2021_solar_1D.dat"
            input="${input_list}'$MODEL_PATH/$output_model_name'\n'$TMP_PATH/${output_model_name}.alt'\n'$TMP_PATH/${output_model_name}_coef.dat'\n'$nlte_bin_file'\n'$nlte_aux_file'\n9\n$teff\n$logg\n$feh\n12.00\n.false.\n.false.\n''"
            ;;
        *)
            echo "ERROR: Invalid mode or calculation_mode: ${mode}_${calculation_mode}" > "$log_file"
            return 1
            ;;
    esac

    echo "INFO: Starting interpolation for $output_model_name ($mode, $calculation_mode)" > "$log_file"
    echo -e "$input" | "$interpol_exec" >> "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "INFO: Successfully interpolated model $output_model_name" >> "$log_file"
    else
        echo "ERROR: Interpolation failed for $output_model_name. See log for details." >> "$log_file"
        return 1
    fi
}

export -f run_interpolation find_bracketing_values

tail -n +2 "scripts/parameter_grid.csv" | while IFS=, read -r teff logg feh lam_min lam_max lam_step turbvel t_value a_val c_val n_val o_val r_val s_val output_mode mode calculation_mode
do
    calculation_mode=$(echo "$calculation_mode" | tr -d '\r')
    mode=$(echo "$mode" | tr -d '\r')

    logg_fmt=$(printf "%+1.2f" "$logg")
    feh_fmt=$(printf "%+1.2f" "$feh")
    output_model_name="p${teff}_g${logg_fmt}_m0.0_t${turbvel}_st_z${feh_fmt}_a${a_val}_c${c_val}_n${n_val}_o${o_val}_r${r_val}_s${s_val}.mod"

    if [ -f "$MODEL_PATH/$output_model_name" ]; then
        echo "INFO: Model $output_model_name already exists. Skipping interpolation."
        continue
    fi
    
    echo "INFO: Model $output_model_name not found. Starting interpolation."
    run_interpolation "$teff" "$logg" "$feh" "$turbvel" "$t_value" "$a_val" "$c_val" "$n_val" "$o_val" "$r_val" "$s_val" "$mode" "$calculation_mode" "$output_model_name"
    # If interpolation fails, stop the script.
    if [ $? -ne 0 ]; then
        echo "Stopping script due to interpolation failure."
        exit 1
    fi
done

echo "INFO: Model interpolation finished. Logs are in $LOG_PATH"
