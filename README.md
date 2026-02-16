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

The default workflow is now a single pipeline config and a single pipeline command.  
The older split flow (manual grid/interpolation/synthesis scripts) is still available, but it is now optional legacy tooling.

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

### 1. Edit One Pipeline Config

Start from the example and keep all run settings in one place:

```bash
cp configs/pipeline/config_pipeline.example.json configs/pipeline/config_pipeline.json
```

Then edit:
- `grid` (sampling bounds and wavelength settings)
- `turbospectrum` (paths/executables and synthesis runtime settings)
- `outputs` (grid/spectra output paths)

#### Quick sampling for machine learning

If you just need a compact, Latin-Hypercube-sampled grid for ML experiments, edit `configs/sampling/config_ml_sampling.json` to set parameter bounds, abundance defaults, and synthesis settings, then run:

```bash
python3 scripts/sample_machine_learning_grid.py  # add --resume to append up to num_samples
```

The helper writes a compressed Zarr store at `runs/local-dev/outputs/grids/ml_parameter_grid.zarr` (override with `--zarr-output`). It uses Polars for high-throughput table construction and Zarr with configurable chunking/compression for HPC-friendly downstream consumption. Install dependencies with `pip install polars zarr numcodecs`. You can optionally include turbvel and element abundances in the Latin Hypercube by toggling `sample_turbvel` and providing bounded abundance entries in the config; turbvel sampling is constrained to the standard `01`–`05` codes for compatibility with the batch runners.

### 2. Run The Pipeline (Single Command)

Run from one config:

```bash
python3 scripts/pipeline_from_config.py --config configs/pipeline/config_pipeline.json
```

This command will:
- generate the grid outputs (CSV + Zarr)
- synthesize spectra into a single output Zarr using multiprocessing

You can still split phases only when needed:

```bash
python3 scripts/pipeline_from_config.py --config configs/pipeline/config_pipeline.json --skip-synthesis
python3 scripts/pipeline_from_config.py --config configs/pipeline/config_pipeline.json --skip-grid
```

### 3. Optional Sharded Runs + Merge

If you need independent shard jobs (e.g. separate PBS submissions), use:

```bash
python3 scripts/pipeline_from_config.py --config configs/pipeline/config_pipeline.json --synthesis-mode sharded --shard-index 0 --shard-count 10
```

After all shards finish, merge to one consolidated store:

```bash
python3 scripts/merge_spectra_shards.py \
  --shard-dir runs/local-dev/outputs/shards \
  --grid-zarr runs/local-dev/outputs/grids/parameter_grid.zarr \
  --output-zarr runs/local-dev/outputs/zarr/synthesized_spectra.zarr
```

### Legacy split scripts (optional)

Manual split scripts are still present for backward compatibility:
- `scripts/generate_grid.py`
- `scripts/interpolate_models.sh`
- `scripts/synthesize_spectra.sh`

Use them only if you explicitly need manual phase-by-phase execution.

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
qsub -v PROJECT=mk27,MAMBA_ENV_NAME=astro,CONFIG_PATH=configs/pipeline/config_pipeline.json scripts/test_pipeline_gadi.pbs
```

The Gadi job wraps the local smoke script and keeps outputs under:

`/scratch/<PROJECT>/<USER>/turbospec_smoke/<PBS_JOBID>/`

## Main Gadi Synthesis (PBS)

For long sharded production runs on Gadi, use:

```bash
qsub turbospectrum.pbs
```

Common overrides:

```bash
qsub -v MAMBA_ENV_NAME=astro,CONFIG_PATH=configs/pipeline/config_pipeline.json,RUN_ROOT=/scratch/mk27/$USER/turbospec,WORKERS=16 turbospectrum.pbs
```

Behavior summary:
- uses the one-config pipeline (`pipeline_from_config.py --synthesis-mode sharded`) for shard generation
- reads default grid/shard/merged output paths from `outputs` in the pipeline config
- validates shard completeness with `scripts/validate_dataset.py` before merge
