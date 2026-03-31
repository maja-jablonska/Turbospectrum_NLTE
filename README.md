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

Ida Hellqvist (https://github.com/littlepadawan) has developped a wrapper (TASS) that allows the automatic generation of large amounts of synthetic spectra from a grid of model atmospheres. It is available there:
https://github.com/littlepadawan/TASS

Her Bachelor thesis is available here: 
https://uu.diva-portal.org/smash/record.jsf?pid=diva2:1880829

Molecular line lists can be found at: 
https://box.in2p3.fr/s/Sn72KPCmC8rYQqa

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

The helper writes a compressed Zarr store plus a parameter lookup parquet:
- grid Zarr: `runs/local-dev/outputs/grids/ml_parameter_grid.zarr` (override with `--zarr-output`)
- parameter index: `runs/local-dev/outputs/grids/index.parquet` (override with `--index-parquet-output` or config `index_parquet`)

`index.parquet` contains `row_index` and all grid parameter columns, so you can quickly filter by parameters and map directly to spectrum rows in downstream Zarr outputs. The helper uses Polars for high-throughput table construction and Zarr with configurable chunking/compression for HPC-friendly downstream ingestion. Install dependencies with `pip install polars zarr numcodecs`. You can optionally include turbvel and element abundances in the Latin Hypercube by toggling `sample_turbvel` and providing bounded abundance entries in the config; turbvel sampling is constrained to the standard `01`–`05` codes for compatibility with the batch runners.

#### JAX dataloader for spectra training

For ML training in JAX, use the built-in dataloader helper on a synthesized spectra store:

```bash
python3 scripts/jax_spectra_dataloader.py runs/local-dev/outputs/zarr/synthesized_spectra.zarr \
  --batch-size 64 \
  --input-key params \
  --target-key flux \
  --normalize-inputs
```

Programmatic usage:

```python
from scripts.jax_spectra_dataloader import create_jax_spectra_dataloaders

loaders = create_jax_spectra_dataloaders(
    zarr_path="runs/local-dev/outputs/zarr/synthesized_spectra.zarr",
    batch_size=64,
    input_key="params",
    target_key="flux",
    normalize_inputs=True,
)

for batch in loaders["train"]:
    x = batch["inputs"]   # jax.Array
    y = batch["targets"]  # jax.Array
```

Install dependencies with `pip install zarr numpy "jax[cpu]"` (or your accelerator-specific JAX build).

#### Lazy spectrum lookup by parameter combination

To fetch one spectrum lazily (without loading the full `flux` array), use:

```bash
python3 scripts/lazy_spectrum_loader.py spectra_tiny.zarr \
  --param teff=5000 \
  --param logg=4.0 \
  --param feh=-1.0 \
  --param vmicro=2.0 \
  --param a=0.0 \
  --param c=0.0 \
  --param n=0.0 \
  --param o=0.0 \
  --param r=0.0 \
  --param s=0.0
```

Programmatic usage:

```python
from scripts.lazy_spectrum_loader import LazySpectrumLoader

loader = LazySpectrumLoader("spectra_tiny.zarr")
match = loader.fetch_spectrum(
    {"teff": 5000.0, "logg": 4.0, "feh": -1.0, "vmicro": 2.0, "a": 0.0, "c": 0.0, "n": 0.0, "o": 0.0, "r": 0.0, "s": 0.0}
)
print(match.row_index, match.spectrum.shape)
```

### 2. Run The Pipeline (Single Command)

Run from one config:

```bash
python3 scripts/pipeline_from_config.py --config configs/pipeline/config_pipeline.json
```

This command will:
- generate the grid outputs (CSV + Zarr + `index.parquet`)
- synthesize spectra into a single output Zarr using multiprocessing

You can still split phases only when needed:

```bash
python3 scripts/pipeline_from_config.py --config configs/pipeline/config_pipeline.json --skip-synthesis
python3 scripts/pipeline_from_config.py --config configs/pipeline/config_pipeline.json --skip-grid
```

If you need to reset generated outputs/state before a fresh run:

```bash
# dry-run (shows what would be removed)
python3 scripts/clean_pipeline_outputs.py --config configs/pipeline/config_pipeline.json

# apply deletion
python3 scripts/clean_pipeline_outputs.py --config configs/pipeline/config_pipeline.json --apply
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
qsub turbospectrum_example.pbs
```

Common overrides:

```bash
qsub -v MAMBA_ENV_NAME=astro,CONFIG_PATH=configs/pipeline/config_pipeline.json,RUN_ROOT=/scratch/mk27/$USER/turbospec,WORKERS=16,ROWS_PER_SHARD=32 turbospectrum_example.pbs
```

Post-processing tuning overrides (optional):

```bash
qsub -v WORKERS=32,VALIDATE_WORKERS=32,MERGE_CHUNK_ROWS=128 turbospectrum_example.pbs
```

Behavior summary:
- uses the one-config pipeline (`pipeline_from_config.py --synthesis-mode sharded`) for shard generation
- reads default grid/shard/merged output paths from `outputs` in the pipeline config
- uses `RUN_ROOT` for queue state/log/tmp directories (and as a fallback when output paths are missing)
- validates shard completeness with `scripts/validate_dataset.py` before merge
- preserves valid shard outputs across reruns and resumes only the missing shard IDs on later submissions
- defaults to legacy one-row-per-shard scheduling unless `SHARD_COUNT` or `ROWS_PER_SHARD` is set in the environment or `runtime`
- for large grids, set `ROWS_PER_SHARD` or `runtime.rows_per_shard` to reduce per-shard startup overhead
- use `FORCE_RESTART=1` only when you want to discard prior shard outputs and start again from scratch

## Regular-grid Gadi Synthesis (PBS)

For the regular Cartesian-grid workflow, use:

```bash
qsub turbospectrum_regular_grid_example.pbs
```

The regular-grid wrapper now supports the same chunked shard layout idea as `turbospectrum_example_array.pbs`, so you can avoid the inefficient one-row-per-shard pattern.

Single-job chunked run:

```bash
qsub -v CONFIG_PATH=configs/pipeline/config_regular_grid.example.json,ROWS_PER_SHARD=128,WORKERS=32 turbospectrum_regular_grid_example.pbs
```

PBS array synthesis:

```bash
qsub -J 0-511 -v CONFIG_PATH=configs/pipeline/config_regular_grid.example.json,ROWS_PER_SHARD=128,WORKERS=32 turbospectrum_regular_grid_example.pbs
```

Finalize/merge after the array completes:

```bash
qsub -v CONFIG_PATH=configs/pipeline/config_regular_grid.example.json,FINALIZE_ONLY=1 turbospectrum_regular_grid_example.pbs
```

Behavior summary:
- reads `runtime.shard_count` / `runtime.rows_per_shard` from the regular-grid config and falls back to one-row shards only when neither is set
- in array mode, task `0` prepares the grid once and later tasks wait for it before starting synthesis
- array tasks beyond the true shard count exit cleanly, so it is safe to submit a slightly oversized `-J` range
- non-array runs still support resume/merge in one submission, but chunked shards reduce startup overhead and usually improve CPU utilization substantially
- `FINALIZE_ONLY=1` skips synthesis and just validates/merges the shard directory, which is the intended post-array follow-up step

## W&B on Gadi/HPC (MLP)

W&B defaults for `wandb_agent.pbs` and `wandb_sync.pbs` are now centralized in:

`configs/training/wandb_hpc.env`

Edit that file once to set:
- `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_GROUP` / `WANDB_TAGS`
- `RUN_DIR` (MLP run outputs)
- `WANDB_LOCAL_DIR` (where local W&B files are written)
- `WANDB_SYNC_DIR` (directory used by sync job)

Default local W&B files are easy to find at:

`runs/mlp_wandb/wandb/`

You can still override any setting per job with `qsub -v`.
