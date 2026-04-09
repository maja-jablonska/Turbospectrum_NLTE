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
    echo "  --output-dir <path>    Base directory for all downloaded files (default: <project>/input_files)."
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

# Keeper / Seafile shared-link identifiers.
# The "files/?p=" URL serves HTML; append "&dl=1" for direct download.
# The API endpoint lists directory contents as JSON.
SHARE_TOKEN="6eaecbf95b88448f98a4"
KEEPER_DL="https://keeper.mpdl.mpg.de/d/${SHARE_TOKEN}/files/"
KEEPER_API="https://keeper.mpdl.mpg.de/api/v2.1/share-links/${SHARE_TOKEN}/dirents/"
# Legacy alias kept for any external references.
BASE_URL="${KEEPER_DL}?p="

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

# --- Pre-scan for --output-dir (must apply before any download runs) ---

_prev_arg=""
for _arg in "$@"; do
    if [[ "$_prev_arg" == "--output-dir" ]]; then
        OUTPUT_DIR="$_arg"
        break
    fi
    _prev_arg="$_arg"
done
unset _prev_arg _arg

if [[ -n "${OUTPUT_DIR:-}" ]]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
    MODEL_PATH="${OUTPUT_DIR}/model_atmospheres/1D/marcs_standard_comp"
    STAGGER_PATH="${OUTPUT_DIR}/model_atmospheres/STAGGER_grid"
    NLTE_BASE_PATH="${OUTPUT_DIR}/nlte_data"
    NLTE_ATOM_PATH="${NLTE_BASE_PATH}/model_atoms"
    NLTE_GRID_PATH="${NLTE_BASE_PATH}/departure_grids"
    LINELIST_PATH="${OUTPUT_DIR}/linelists"
    GOLD_SAMPLE_PATH="${OUTPUT_DIR}/gold_sample"
    TMP_PATH="${OUTPUT_DIR}/.tmp"
    mkdir -p "$TMP_PATH"
    echo "Output directory: $OUTPUT_DIR"
fi

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

# Download a directory tree from the Keeper Seafile share via its REST API.
# Handles recursive subdirectories. Skips files that already exist locally.
#
# Usage: download_keeper_dir <remote_path> <target_dir> <label> [accept_glob] [_depth]
#   remote_path  — path on the share, e.g. /linelist/
#   target_dir   — local destination
#   label        — human-readable label for progress messages
#   accept_glob  — optional comma-separated fnmatch patterns (e.g. "atom.*")
#   _depth       — internal recursion counter (do not set manually)
download_keeper_dir() {
    local remote_path="$1"
    local target_dir="$2"
    local label="$3"
    local accept="${4:-}"
    local _depth="${5:-0}"

    if [ "$_depth" -eq 0 ]; then
        if should_skip_download "$target_dir" "$label"; then
            return 0
        fi
        echo "Downloading $label from Keeper share..."
    fi

    mkdir -p "$target_dir"

    local listing
    listing=$(curl -sL "${KEEPER_API}?path=${remote_path}" 2>/dev/null)

    if [ -z "$listing" ] || ! python3 -c "import json,sys; json.load(sys.stdin)" <<< "$listing" >/dev/null 2>&1; then
        echo "Error: could not list ${remote_path} on Keeper share."
        return 1
    fi

    # Emit tab-separated entries: "F\tpath\tname" for files, "D\tpath\tname" for dirs.
    local entries
    entries=$(python3 -c "
import json, sys, fnmatch
data = json.loads(sys.stdin.read())
accept = '''${accept}'''
for item in data.get('dirent_list', []):
    if item.get('is_dir'):
        print('D\t' + item['folder_path'] + '\t' + item['folder_name'])
    else:
        name = item['file_name']
        if accept and not any(fnmatch.fnmatch(name, p.strip()) for p in accept.split(',')):
            continue
        print('F\t' + item['file_path'] + '\t' + name)
" <<< "$listing" 2>/dev/null)

    local failed=0
    while IFS=$'\t' read -r kind path name; do
        [ -z "$kind" ] && continue
        if [ "$kind" = "D" ]; then
            download_keeper_dir "$path" "$target_dir/$name" "$label" "$accept" $((_depth + 1)) || failed=$((failed + 1))
        elif [ "$kind" = "F" ]; then
            local dest="$target_dir/$name"
            if [ -f "$dest" ] && [ "$FORCE_DOWNLOAD" = false ]; then
                continue
            fi
            echo "  $name"
            if ! wget -q --show-progress --continue -O "$dest" "${KEEPER_DL}?p=${path}&dl=1"; then
                echo "  Warning: failed to download $name"
                rm -f "$dest"
                failed=$((failed + 1))
            fi
        fi
    done <<< "$entries"

    if [ "$_depth" -eq 0 ]; then
        if [ "$failed" -eq 0 ]; then
            mark_download_complete "$target_dir"
            echo "$label download complete."
        else
            echo "Warning: $label had $failed failure(s). Re-run to retry."
        fi
    fi
    return "$failed"
}


# --- Download Functions ---

# Download MARCS model atmospheres
download_marcs() {
    local target_dir="$MODEL_PATH"
    # The &dl=1 parameter triggers a direct download (302 → seafhttp blob)
    # instead of the HTML file-browser page.
    local zip_url="https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/files/?p=/atmospheres/marcs_standard_comp.zip&dl=1"
    local zip_file="$TMP_PATH/marcs_standard_comp.zip"
    local max_retries=3

    if should_skip_download "$target_dir" "MARCS atmospheres"; then
        return 0
    fi

    mkdir -p "$target_dir"
    mkdir -p "$TMP_PATH"

    if [ ! -f "$zip_file" ]; then
        echo "Downloading MARCS atmospheres (~135 MB)..."
        local attempt
        for attempt in $(seq 1 "$max_retries"); do
            if wget -q --show-progress \
                    --continue \
                    --content-disposition \
                    -O "$zip_file" \
                    "$zip_url"; then
                break
            fi
            echo "Warning: MARCS download attempt $attempt/$max_retries failed."
            if [ "$attempt" -eq "$max_retries" ]; then
                echo "Error: could not download MARCS ZIP after $max_retries attempts."
                echo "You can download manually from:"
                echo "  $zip_url"
                echo "Save to: $zip_file"
                echo "Then re-run this script."
                rm -f "$zip_file"
                return 1
            fi
            sleep $(( attempt * 5 ))
        done
    fi

    # Verify the ZIP is not truncated (content-length is ~141 MB).
    local min_size=100000000  # 100 MB sanity floor
    local actual_size
    actual_size="$(wc -c < "$zip_file" 2>/dev/null || echo 0)"
    if [ "$actual_size" -lt "$min_size" ]; then
        echo "Error: MARCS ZIP looks truncated (${actual_size} bytes < ${min_size})."
        echo "Deleting and re-run to retry."
        rm -f "$zip_file"
        return 1
    fi

    echo "Unzipping models to $target_dir..."
    if ! unzip -o "$zip_file" -d "$target_dir"; then
        echo "Error: Failed to unzip MARCS models. The zip file might be corrupted."
        echo "Please delete '$zip_file' and try downloading it manually again."
        return 1
    fi

    # Clean up the downloaded zip file
    rm -f "$zip_file"

    # Remove the model_list file if it exists, as it's no longer needed
    local model_list_file="$target_dir/model_list"
    if [ -f "$model_list_file" ]; then
        rm "$model_list_file"
        echo "Removed old model_list file."
    fi

    mark_download_complete "$target_dir"
    echo "MARCS atmospheres extraction complete."
}

# Download STAGGER model atmospheres (ZIP on Keeper)
download_stagger() {
    local target_dir="$STAGGER_PATH"
    local zip_url="${KEEPER_DL}?p=/atmospheres/average_stagger_grid_forTSv20.zip&dl=1"
    local zip_file="$TMP_PATH/average_stagger_grid_forTSv20.zip"

    if should_skip_download "$target_dir" "STAGGER atmospheres"; then
        return 0
    fi

    mkdir -p "$target_dir" "$TMP_PATH"

    if [ ! -f "$zip_file" ]; then
        echo "Downloading STAGGER atmospheres..."
        if ! wget -q --show-progress --continue -O "$zip_file" "$zip_url"; then
            echo "Error: failed to download STAGGER ZIP."
            rm -f "$zip_file"
            return 1
        fi
    fi

    echo "Unzipping STAGGER models to $target_dir..."
    if ! unzip -o "$zip_file" -d "$target_dir"; then
        echo "Error: failed to unzip STAGGER models."
        return 1
    fi

    rm -f "$zip_file"
    mark_download_complete "$target_dir"
    echo "STAGGER atmospheres extraction complete."
}

# Download NLTE data (model atoms + departure grids from /dep-grids/)
download_nlte() {
    echo "Downloading NLTE data..."
    echo "Model atoms will be saved to: $NLTE_ATOM_PATH"
    echo "Departure coefficient grids will be saved to: $NLTE_GRID_PATH"

    local status=0
    # Model atoms (atom.*) and departure grids (NLTEgrid*, auxData*) are
    # colocated under per-element subdirectories on the share.
    download_keeper_dir "/dep-grids/" "$NLTE_ATOM_PATH" "NLTE model atoms" "atom.*" || status=1
    download_keeper_dir "/dep-grids/" "$NLTE_GRID_PATH" "NLTE departure grids" "NLTEgrid*,auxData*" || status=1

    if [ $status -eq 0 ]; then
        echo "NLTE data download complete."
    else
        echo "NLTE data download completed with warnings; re-run to fetch any missing files."
    fi

    return $status
}

# Download recommended line lists (from /linelist/ on Keeper)
download_linelists() {
    download_keeper_dir "/linelist/" "$LINELIST_PATH" "Line lists"
}

# Download gold sample dataset.
# Note: use download_gold_sample.py for ESO archive spectra;
# this option is for pre-packaged gold sample bundles if hosted on Keeper.
download_gold_sample() {
    if [ -d "$GOLD_SAMPLE_PATH" ] && [ "$(ls -A "$GOLD_SAMPLE_PATH" 2>/dev/null)" ]; then
        if should_skip_download "$GOLD_SAMPLE_PATH" "Gold sample"; then
            return 0
        fi
    fi
    echo "Gold sample: use download_gold_sample.py for ESO archive spectra."
    echo "  python download_gold_sample.py --output $GOLD_SAMPLE_PATH"
    return 0
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
        --output-dir)
            # Already handled in pre-scan; just consume the argument.
            shift
            ;;
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
