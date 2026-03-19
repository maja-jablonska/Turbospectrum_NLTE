#!/bin/bash

# Script to download default files for Turbospectrum
# Based on the information in DOC/Turbospectrum_v20_Documentation_v6.pdf

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)

# Function to display usage instructions
usage() {
    echo "Usage: $0 [options]"
    echo "Downloads default files for Turbospectrum."
    echo ""
    echo "Options:"
    echo "  --atmospheres <type>   Download model atmospheres. <type> can be 'MARCS', 'STAGGER', or 'all'."
    echo "  --nlte-atoms <atoms>   Download NLTE data. <atoms> is a space-separated list (e.g., 'O Mg Si')."
    echo "                         Use 'all' to download data for all available atoms."
    echo "  --linelists            Download the recommended line lists."
    echo "  --gold-sample          Download the gold sample dataset (configurable via GOLD_SAMPLE_URL/GOLD_SAMPLE_PATH)."
    echo "  --force                Re-download files even if a previous run completed successfully."
    echo "  --all                  Download all available default files."
    echo "  -h, --help             Display this help message."
}

# --- Configuration ---

# Source environment variables to get destination paths
if [ -f "${SCRIPT_DIR}/env.sh" ]; then
    source "${SCRIPT_DIR}/env.sh"
else
    echo "Error: env.sh not found in the scripts directory."
    exit 1
fi

# The base URL for the data repository mentioned in the documentation
BASE_URL="https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/files/?p="

# Define paths for NLTE data, as they are not in env.sh
NLTE_BASE_PATH="${NLTE_BASE_PATH:-${PROJECT_ROOT}/input_files/nlte_data}"
NLTE_ATOM_PATH="${NLTE_ATOM_PATH:-$NLTE_BASE_PATH/model_atoms}"
NLTE_GRID_PATH="${NLTE_GRID_PATH:-$NLTE_BASE_PATH/departure_grids}"

# Allow overriding STAGGER path
STAGGER_PATH="${STAGGER_PATH:-${PROJECT_ROOT}/input_files/model_atmospheres/STAGGER_grid}"

# Optional gold sample download location (override via env)
GOLD_SAMPLE_URL="${GOLD_SAMPLE_URL:-${BASE_URL}/gold_sample/}"
GOLD_SAMPLE_PATH="${GOLD_SAMPLE_PATH:-${PROJECT_ROOT}/gold_sample}"

# Whether to force re-download even if a previous run completed
FORCE_DOWNLOAD=false

# --- Helpers ---

should_skip_download() {
    local target_dir="$1"
    local label="$2"
    local marker="$target_dir/.download_complete"

    if [ -f "$marker" ] && [ "$FORCE_DOWNLOAD" = false ]; then
        echo "$label already downloaded at $target_dir. Skipping (use --force to re-download)."
        return 0
    fi
    return 1
}

mark_download_complete() {
    local target_dir="$1"
    mkdir -p "$target_dir"
    touch "$target_dir/.download_complete"
}

download_with_resume() {
    local url="$1"
    local target_dir="$2"
    local cut_dirs="$3"
    local label="$4"
    local accept="$5"

    mkdir -p "$target_dir"
    mkdir -p "$TMP_PATH"

    if should_skip_download "$target_dir" "$label"; then
        return 0
    fi

    echo "Syncing $label from $url to $target_dir..."
    local wget_args=(
        -q --show-progress -r -np -nH
        --cut-dirs="$cut_dirs"
        --no-check-certificate
        --continue
        --timestamping
        -P "$target_dir"
        -R "index.html*"
    )

    if [ -n "$accept" ]; then
        wget_args+=(--accept="$accept")
    fi

    if wget "${wget_args[@]}" "$url"; then
        mark_download_complete "$target_dir"
        echo "$label download complete."
        return 0
    else
        echo "Warning: $label download encountered errors. You can re-run the script to resume."
        return 1
    fi
}


# --- Download Functions ---

# Download MARCS model atmospheres
download_marcs() {
    local target_dir="$MODEL_PATH"
    local zip_url="https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/files/?p=/atmospheres/marcs_standard_comp.zip"
    local zip_file="$TMP_PATH/marcs_standard_comp.zip"

    if should_skip_download "$target_dir" "MARCS atmospheres"; then
        return 0
    fi

    if [ ! -f "$zip_file" ]; then
        echo ""
        echo "===================================================================================="
        echo "  MARCS model atmosphere zip file not found: $zip_file"
        echo "  Please manually download the zip file from the following URL:"
        echo "  $zip_url"
        echo "  And save it to: $zip_file"
        echo "  After downloading, run this script again."
        echo "===================================================================================="
        echo ""
        return 1
    fi

    echo "Unzipping models to $target_dir..."
    mkdir -p "$target_dir"
    mkdir -p "$TMP_PATH"

    unzip -o "$zip_file" -d "$target_dir"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to unzip MARCS models. The zip file might be corrupted."
        echo "Please delete '$zip_file' and try downloading it manually again."
        return 1
    fi

    # Clean up the downloaded zip file
    rm "$zip_file"

    # Remove the model_list file if it exists, as it's no longer needed
    local model_list_file="$target_dir/model_list"
    if [ -f "$model_list_file" ]; then
        rm "$model_list_file"
        echo "Removed old model_list file."
    fi

    mark_download_complete "$target_dir"
    echo "MARCS atmospheres extraction complete."
}

# Download STAGGER model atmospheres
download_stagger() {
    download_with_resume "${BASE_URL}/STAGGER_grid/" "$STAGGER_PATH" 4 "STAGGER atmospheres"
}

# Download NLTE data (model atoms and departure coefficient grids)
download_nlte() {
    echo "Downloading NLTE data..."
    echo "Model atoms will be saved to: $NLTE_ATOM_PATH"
    echo "Departure coefficient grids will be saved to: $NLTE_GRID_PATH"

    local status=0
    download_with_resume "${BASE_URL}/NLTE_data/" "$NLTE_ATOM_PATH" 5 "NLTE model atoms" "atom.*" || status=1
    download_with_resume "${BASE_URL}/NLTE_data/" "$NLTE_GRID_PATH" 5 "NLTE departure coefficient grids" "NLTEgrid*,auxData*" || status=1

    if [ $status -eq 0 ]; then
        echo "NLTE data download complete (resume-safe)."
    else
        echo "NLTE data download completed with warnings; re-run to fetch any missing files."
    fi

    return $status
}

# Download recommended line lists
download_linelists() {
    download_with_resume "${BASE_URL}/Linelists/" "$LINELIST_PATH" 4 "Line lists"
}

# Download gold sample dataset (path configurable via GOLD_SAMPLE_URL)
download_gold_sample() {
    download_with_resume "$GOLD_SAMPLE_URL" "$GOLD_SAMPLE_PATH" 4 "Gold sample"
}


# --- Main Script Logic ---

# If no arguments are provided, show usage
if [ "$#" -eq 0 ]; then
    usage
    exit 1
fi

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_DOWNLOAD=true
            ;;
        --atmospheres)
            if [[ -z "$2" ]]; then
                echo "Error: --atmospheres requires an argument."
                exit 1
            fi

            if [[ "$2" == "MARCS" ]]; then
                download_marcs
            elif [[ "$2" == "STAGGER" ]]; then
                download_stagger
            elif [[ "$2" == "all" ]]; then
                download_marcs
                download_stagger
            else
                echo "Error: Invalid atmosphere type '$2'. Use 'MARCS', 'STAGGER', or 'all'."
                exit 1
            fi
            shift
            ;;
        --nlte-atoms)
            # The script will download all NLTE data regardless of the specific atoms listed
            # to ensure all necessary files are present.
            download_nlte
            # Shift past the list of atoms
            shift
            while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
                shift
            done
            continue
            ;;
        --linelists)
            download_linelists
            ;;
        --gold-sample)
            download_gold_sample
            ;;
        --all)
            download_marcs
            download_stagger
            download_nlte
            download_linelists
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

echo "All requested downloads are complete."
