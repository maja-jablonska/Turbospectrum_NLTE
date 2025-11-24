#!/bin/bash

# Script to download default files for Turbospectrum
# Based on the information in DOC/Turbospectrum_v20_Documentation_v6.pdf

# Function to display usage instructions
usage() {
    echo "Usage: $0 [options]"
    echo "Downloads default files for Turbospectrum."
    echo ""
    echo "Options:"
    echo "  --atmospheres <type>   Download model atmospheres. <type> can be 'MARCS', 'STAGGER', or 'all'."
    echo "  --nlte-atoms <atoms>   Download NLTE data. <atoms> is a space-separated list (e.g., "O Mg Si")."
    echo "                         Use 'all' to download data for all available atoms."
    echo "  --linelists            Download the recommended line lists."
    echo "  --all                  Download all available default files."
    echo "  -h, --help             Display this help message."
}

# --- Configuration ---

# Source environment variables to get destination paths
if [ -f "$(dirname "$0")/env.sh" ]; then
    source "$(dirname "$0")/env.sh"
else
    echo "Error: env.sh not found in the scripts directory."
    exit 1
fi

# The base URL for the data repository mentioned in the documentation
BASE_URL="https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/files/?p="

# Define paths for NLTE data, as they are not in env.sh
NLTE_BASE_PATH="/Users/mjablons/Documents/Turbospectrum_NLTE/input_files/nlte_data"
NLTE_ATOM_PATH="$NLTE_BASE_PATH/model_atoms"
NLTE_GRID_PATH="$NLTE_BASE_PATH/departure_grids"


# --- Download Functions ---

# Download MARCS model atmospheres
download_marcs() {
    local target_dir="$MODEL_PATH"
    local zip_url="https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/files/?p=/atmospheres/marcs_standard_comp.zip"
    local zip_file="$TMP_PATH/marcs_standard_comp.zip"

    echo "Downloading MARCS model atmospheres to $target_dir..."
    mkdir -p "$target_dir"
    mkdir -p "$TMP_PATH"

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

    echo "MARCS atmospheres extraction complete."
}

# Download STAGGER model atmospheres
download_stagger() {
    local target_dir="/Users/mjablons/Documents/Turbospectrum_NLTE/input_files/model_atmospheres/STAGGER_grid"
    echo "Downloading STAGGER model atmospheres to $target_dir..."
    mkdir -p "$target_dir"
    wget -q --show-progress -r -np -nH --cut-dirs=4 --no-check-certificate -R "index.html*" -P "$target_dir" "${BASE_URL}/STAGGER_grid/"
    echo "STAGGER atmospheres download complete."
}

# Download NLTE data (model atoms and departure coefficient grids)
download_nlte() {
    echo "Downloading NLTE data..."
    echo "Model atoms will be saved to: $NLTE_ATOM_PATH"
    echo "Departure coefficient grids will be saved to: $NLTE_GRID_PATH"
    mkdir -p "$NLTE_ATOM_PATH"
    mkdir -p "$NLTE_GRID_PATH"

    # Note: Due to the complex mapping of atoms to filenames, this script downloads all
    # available NLTE data to ensure consistency.
    echo "Downloading all model atoms..."
    wget -q --show-progress -r -np -nH --cut-dirs=5 --no-check-certificate -R "index.html*" --accept="atom.*" -P "$NLTE_ATOM_PATH" "${BASE_URL}/NLTE_data/"

    echo "Downloading all departure coefficient grids..."
    wget -q --show-progress -r -np -nH --cut-dirs=5 --no-check-certificate -R "index.html*" --accept="NLTEgrid*,auxData*" -P "$NLTE_GRID_PATH" "${BASE_URL}/NLTE_data/"

    echo "NLTE data download complete."
}

# Download recommended line lists
download_linelists() {
    echo "Downloading recommended line lists to $LINELIST_PATH..."
    mkdir -p "$LINELIST_PATH"
    wget -q --show-progress -r -np -nH --cut-dirs=4 --no-check-certificate -R "index.html*" -P "$LINELIST_PATH" "${BASE_URL}/Linelists/"
    echo "Line lists download complete."
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
        --atmospheres)
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