# TurbospectrumNLTE (aka Turbospectrum2020)
Public

Synthetic stellar spectra calculator LTE / NLTE

Bertrand Plez
LUPM, Montpellier University, France

in collaboration with Jeff Gerber, Ekaterina Magg, and Maria Bergemann

The next version of TS (Turbospectrum), with NLTE capabilities.
In order to compute NLTE stellar spectra, additional data is needed.
See documentation in DOC folder

A wrapper can be found there: 
https://github.com/EkaterinaSe/TurboSpectrum-Wrapper/
and another one there:
https://github.com/JGerbs13/TSFitPy

## Downloading Data

A script is provided to download the necessary data to run Turbospectrum. This includes model atmospheres, NLTE data, and line lists. Downloads are resume-safe: existing files are skipped, and partial downloads continue where they left off. Use `--force` to re-download everything if needed.

### Usage

To use the script, run it from the command line with the desired options. You can download specific parts of the data, or all of it at once.

First, make the script executable:
```bash
chmod +x scripts/download_data.sh
```

Then, run the script with one of the following options:

*   **Download all data:**
    ```bash
    ./scripts/download_data.sh --all
    ```

*   **Download specific model atmospheres:**
    ```bash
    # Download MARCS atmospheres
    ./scripts/download_data.sh --atmospheres MARCS

    # Download STAGGER atmospheres
    ./scripts/download_data.sh --atmospheres STAGGER
    ```

*   **Download NLTE data:**
    ```bash
    # Download all NLTE data
    ./scripts/download_data.sh --nlte-atoms all
    ```

*   **Download line lists:**
    ```bash
    ./scripts/download_data.sh --linelists
    ```

*   **Download the gold sample dataset (path configurable via `GOLD_SAMPLE_URL`/`GOLD_SAMPLE_PATH`):**
    ```bash
    ./scripts/download_data.sh --gold-sample
    ```

For more information, you can view the help message:
```bash
./scripts/download_data.sh --help
```

## Workflow

This section outlines the typical workflow for generating a grid of synthetic spectra.

### Rules-Aligned Repository Layout

The repository now separates static configuration from generated artifacts:

```text
configs/
  pipeline/      # one-config pipeline definitions
  sampling/      # grid sampling definitions
  synthesis/     # Turbospectrum runtime config
runs/
  local-dev/
    config/
    task_queue/
    logs/
      pbs/
      shards/
    outputs/
      grids/
      shards/
      zarr/
    tmp/
    reports/
```

This mirrors the operational guidance in `OPERATIONS.md` and keeps outputs immutable and easy to resume.

### 1. Configure the Parameter Grid

The first step is to define the grid of stellar parameters for which you want to compute spectra. This is done by editing `configs/sampling/grid_config.yml`.

This YAML file allows you to specify the minimum, maximum, and step size for `teff`, `logg`, and `feh`. You can also set the desired wavelength range and other synthesis parameters.

### 2. Generate the Parameter CSV

Once you have configured `grid_config.yml`, you can generate the `parameter_grid.csv` file by running the `generate_grid.py` script. This script requires the `PyYAML` and `numpy` packages to be installed so it can resolve lists, ranges, and sampled distributions.

```bash
# Install dependencies if you haven't already
pip install PyYAML numpy

# Run the script to generate the grid
python3 scripts/generate_grid.py
```

This will create `runs/local-dev/outputs/grids/parameter_grid.csv`, which is used by subsequent scripts in the pipeline.

#### Quick sampling for machine learning

If you just need a compact, Latin-Hypercube-sampled grid for ML experiments, edit `configs/sampling/config_ml_sampling.json` to set parameter bounds, abundance defaults, and synthesis settings, then run:

```bash
python3 scripts/sample_machine_learning_grid.py  # add --resume to append up to num_samples
```

The helper writes a compressed Zarr store at `runs/local-dev/outputs/grids/ml_parameter_grid.zarr` (override with `--zarr-output`). It uses Polars for high-throughput table construction and Zarr with configurable chunking/compression for HPC-friendly downstream consumption. Install dependencies with `pip install polars zarr numcodecs`. The layout matches the existing synthesis scripts, so you can plug it directly into `scripts/synthesize_spectra.sh` after copying or renaming it as needed. You can optionally include turbvel and element abundances in the Latin Hypercube by toggling `sample_turbvel` and providing bounded abundance entries in the config; turbvel sampling is constrained to the standard `01`–`05` codes for compatibility with the HPC batch runners.

### One-config pipeline (recommended)

To avoid keeping a separate grid config and Turbospectrum config in sync, you can use a single pipeline config file that includes both sections. Start from `configs/pipeline/config_pipeline.example.json` and run:

```bash
python3 scripts/pipeline_from_config.py --config configs/pipeline/config_pipeline.json
```

This will:
- generate the grid outputs (CSV + Zarr)
- synthesize spectra into a single output Zarr using multiprocessing

If you need to run multiple independent “shards” (e.g. separate PBS jobs without arrays), use:

```bash
python3 scripts/pipeline_from_config.py --config configs/pipeline/config_pipeline.json --synthesis-mode sharded --shard-index 0 --shard-count 10
```

After all shards finish, merge them into one consolidated store:

```bash
python3 scripts/merge_spectra_shards.py \
  --shard-dir runs/local-dev/outputs/shards \
  --output-zarr runs/local-dev/outputs/zarr/synthesized_spectra.zarr
```

### 3. Interpolate Model Atmospheres

With the parameter grid generated, the next step is to ensure that a model atmosphere exists for each point in the grid. The `interpolate_models.sh` script handles this by interpolating new models from the existing grid as needed.

The script reads `runs/local-dev/outputs/grids/parameter_grid.csv` (or `PARAMETER_GRID_CSV` if set) and, for each entry, checks if the required model atmosphere exists. If not, it interpolates one.

To run the script:
```bash
./scripts/interpolate_models.sh
```

This will populate the `input_files/model_atmospheres/` directory with any newly interpolated models.

### 4. Synthesize a Grid of Spectra

With atmospheres and a parameter grid in place, you can generate spectra for every sampled point using `scripts/synthesize_spectra.sh`. Make sure `scripts/env.sh` points to your local paths for model atmospheres, line lists, and Turbospectrum executables.

```bash
# Generate a reproducible grid with sampling controls
python3 scripts/generate_grid.py

# Ensure atmospheres exist for each grid point
./scripts/interpolate_models.sh

# Synthesize Flux/Intensity spectra in parallel across the grid
./scripts/synthesize_spectra.sh
```

Each run reads `runs/local-dev/outputs/grids/parameter_grid.csv` by default (including `grid_version`, abundances, and sampling metadata), uses the corresponding model file, and writes logs under `runs/local-dev/logs/shards/` unless overridden in config. Synthetic spectra are written to the directory specified by `SPECTRA_PATH` in `scripts/env.sh`.

## Pipeline Smoke Tests

Use these small scripts to test pipeline validity in two stages:

### 1. Local smoke test

```bash
./scripts/test_pipeline_local.sh \
  --config configs/pipeline/config_pipeline.json
```

This validates config JSON parsing, generates a small grid via `pipeline_from_config.py`, and verifies required grid columns.  
To include one-shard synthesis + dataset validation:

```bash
./scripts/test_pipeline_local.sh \
  --config configs/pipeline/config_pipeline.json \
  --with-synthesis \
  --keep
```

### 2. Gadi smoke test (PBS)

Submit:

```bash
qsub scripts/test_pipeline_gadi.pbs
```

Optional overrides at submit time:

```bash
qsub -v PROJECT=mk27,PYTHON_BIN=/path/to/python,CONFIG_PATH=configs/pipeline/config_pipeline.json scripts/test_pipeline_gadi.pbs
```

The Gadi job wraps the local smoke script and keeps outputs under:

`/scratch/<PROJECT>/<USER>/turbospec_smoke/<PBS_JOBID>/`
