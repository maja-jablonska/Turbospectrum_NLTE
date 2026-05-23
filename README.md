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

## Setup Guide

End-to-end, from a fresh clone to a synthesized spectra store. Each step links to a deeper section below.

### 0. Prerequisites

- A Fortran compiler: `gfortran` (default), or Intel `ifort` / `ifx`.
- Python 3.10+ with the pipeline dependencies:
  ```bash
  pip install numpy zarr numcodecs polars pytest
  # optional extras: "jax[cpu]" (training dataloader), pandas/pyarrow
  ```

### 1. Build the Fortran binaries

```bash
cd exec-gf && make all        # builds babsma_lu and bsyn_lu in exec-gf/
# Intel alternatives: exec-intel (ifort) or exec-ifx (ifx)
```

The model-atmosphere interpolator builds separately — see `interpolator/readme`.

### 2. Download input data

Model atmospheres, line lists, and (optionally) NLTE data:

```bash
./scripts/download_data.sh --all         # or pick parts: --atmospheres MARCS / --linelists / --nlte-atoms all
```

See [Downloading Data](#downloading-data) for all options. Files land under `input_files/` and `DATA/`.

### 3. Configure (one file)

Copy the canonical template and edit the four sections:

```bash
cp configs/pipeline/config_pipeline.example.json configs/pipeline/config_pipeline.json
```

| Section | What to set |
|---|---|
| `grid` | `sampling` mode (see below), the wavelength window (`synthesis.lam_min/lam_max/lam_step`), `output_mode` (`Flux`/`Intensity`), and the `io` compression block. |
| `turbospectrum` | `paths.model_atmosphere_path` / `paths.linelist_path` / `paths.linelist_files`, compiler (`gf`/`intel`). For NLTE just set `grid.synthesis.calculation_mode: "NLTE"` (the single switch — see [NLTE Quickstart](#nlte-quickstart)); `nlte.enabled` + the info file are derived. |
| `outputs` | Just `run_root` — all grid/spectra paths default to `outputs/...` under it. |
| `runtime` | `workers` (null = auto), `chunk_rows`, optional `scratch`. |

Conventions that keep configs consistent:
- **Sampling mode**: set `grid.sampling` to choose how the parameter grid is built — `"lhs"` (Latin-Hypercube: `num_samples` + `bounds`) or `"grid"` (regular Cartesian product: `grid.axes`, e.g. `"teff": "3800:7500:250"`). The **same** `pipeline_from_config.py` command runs either mode and produces the same column schema. Omit `sampling` and it auto-detects (an `axes` block → `grid`, otherwise `lhs`), so existing configs keep working unchanged.
- **Paths**: set `run_root` once; leave output paths to their defaults (`_DEFAULT_OUTPUT_PATHS` in `scripts/pipeline_from_config.py`). Override a single key only when you need to deviate.
- **Compression**: use the grouped `grid.io` block (`csv_compression`, `csv_compression_level`, `zarr_chunks`, `zarr_compressor`). It is read identically by every grid path; omit it to accept the `zstd` defaults.
- **Comments**: any key starting with `_` is documentation and is stripped before use.

For specialized starting points: regular Cartesian grids → `config_regular_grid.example.json`; minimal NLTE (single switch) → `config_nlte_minimal.example.json`; NLTE Fe with per-abundance ASCII departures → `config_regular_grid_nlte_fe.example.json`; intensity (mu) runs → `config_regular_grid_nlte_fe_intensity.example.json`. The full annotated schema lives in `configs/synthesis/config_sample_comprehensive.json`.

### 4. Run

```bash
python3 scripts/pipeline_from_config.py --config configs/pipeline/config_pipeline.json
```

This generates the grid (CSV + Zarr + `index.parquet`) and synthesizes spectra into one Zarr store. See [Run The Pipeline](#2-run-the-pipeline-single-command) for phase splits, resets, and sharded HPC runs.

### 5. Inspect outputs

Under `run_root`:
- `outputs/grids/parameter_grid.zarr` + `index.parquet` — the parameter grid and lookup table.
- `outputs/zarr/synthesized_spectra.zarr` — the spectra (`params`, `flux`, wavelength).

Load one spectrum by parameters with `scripts/lazy_spectrum_loader.py`, or stream batches for training with `scripts/jax_spectra_dataloader.py` (both shown under [Workflow](#workflow)).

### Next steps

- **HPC (Gadi/PBS)**: [Main Gadi Synthesis](#main-gadi-synthesis-pbs) and [Regular-grid Gadi Synthesis](#regular-grid-gadi-synthesis-pbs).
- **Large/sharded runs**: [Optional Sharded Runs + Merge](#3-optional-sharded-runs--merge) and [Shard Layout And Chunking](#shard-layout-and-chunking).
- **Tests**: `python3 -m pytest tests/ -v` (see [Testing](#testing)).

## NLTE Quickstart

NLTE is a **single switch**: set `grid.synthesis.calculation_mode: "NLTE"` and the pipeline turns on `nlte.enabled` and defaults the species map (`DATA/SPECIES_LTE_NLTE.dat`) for you — you no longer keep two flags in sync.

You don't have to download data or hand-write any of this. The fast path:

```bash
# 1. Download the NLTE data AND generate a matching DATA/SPECIES_LTE_NLTE.dat
#    (the shipped one points at an HPC path; this writes one for YOUR machine,
#     backing up any existing file). Restrict NLTE species with --nlte Fe Mg ...
python3 scripts/setup_nlte.py

# 2. Generate a ready-to-run pipeline config (runs the preflight for you)
python3 scripts/init_nlte_config.py --output configs/pipeline/config_nlte.json
#    --workflow ascii --ascii-dir <dir>  for ASCII per-abundance departures instead

# 3. Run (the same preflight runs automatically before synthesis)
python3 scripts/pipeline_from_config.py --config configs/pipeline/config_nlte.json
```

Prefer to do it by hand? Copy the annotated template `configs/pipeline/config_nlte_minimal.example.json`, point the two path headers in `DATA/SPECIES_LTE_NLTE.dat` at your downloaded `input_files/nlte_data/{model_atoms,departure_grids}`, then check it before a long run:

```bash
./scripts/download_data.sh --nlte-atoms all
python3 scripts/validate_nlte_config.py --config configs/pipeline/config_nlte_minimal.json
```

Two NLTE workflows share this one mental model:

- **Binary departure grids** (the default) — selected via the `nlte_info_file` species map; what `--nlte-atoms` downloads and what `setup_nlte.py` configures.
- **ASCII per-abundance departures** — built by *exporting* the binary grids to per-model ASCII files (the `_abu±X.XXX` files keyed by abundance), then pointed at by an `nlte_ascii_departures` block (top level or under `grid`). Generate them in the same step:

  ```bash
  python3 scripts/setup_nlte.py --ascii-export   # also writes per-species dirs under DATA/DEP/nlte_departures_ascii/<El>/
  ```

  Point the config at the per-species directory (e.g. `…/Fe`); see `config_regular_grid_nlte_fe.example.json`. The export tool itself is `scripts/export_nlte_grid_ascii.py` if you need to run it standalone.

The pre-synthesis preflight runs in both `pipeline_from_config.py` and `synthesize_regular_grid.py`; bypass it with `--skip-nlte-preflight`.

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

There is also a ready-made example config for a larger ML dataset:

```bash
python3 scripts/sample_machine_learning_grid.py \
  --config configs/sampling/config_ml_dataset.example.json
```

If you want one config that handles both ML grid generation and later synthesis, use the pipeline example instead:

```bash
python3 scripts/pipeline_from_config.py \
  --config configs/pipeline/config_pipeline.example.json
```

The helper writes a compressed Zarr store plus a parameter lookup parquet:
- grid Zarr: `runs/local-dev/outputs/grids/ml_parameter_grid.zarr` (override with `--zarr-output`)
- parameter index: `runs/local-dev/outputs/grids/index.parquet` (override with `--index-parquet-output` or config `index_parquet`)
- in sampling configs, `run_root` is the base directory and relative output paths like `outputs/grids/...` resolve underneath it
- after synthesis, both `turbvel` and `t_value` are preserved as numeric parameter columns in addition to `vmicro` when present in the source grid

`index.parquet` contains `row_index` and all grid parameter columns, so you can quickly filter by parameters and map directly to spectrum rows in downstream Zarr outputs. The helper uses Polars for high-throughput table construction and Zarr with configurable chunking/compression for HPC-friendly downstream ingestion. Install dependencies with `pip install polars zarr numcodecs`. You can optionally include element abundances in the Latin Hypercube by providing bounded abundance entries (`{"min": ..., "max": ...}`) in the config. Microturbulence can be sampled either continuously or discretely — see below.

#### Microturbulence (vmicro) sampling

In the **Latin-Hypercube path**, `vmicro` can be sampled two mutually exclusive ways (setting both raises an error):

- **Continuous (recommended)** — treat it like `teff`/`logg`/`feh`: add `vmicro` to `bounds` and it is drawn continuously (km/s) by the Latin Hypercube. The sampled float is stored as the synthesis microturbulence (`turbvel`, the value passed to `bsyn`), while the atmosphere label (`t_value`) is snapped to the nearest entry of `t_value_options`. This snaps the atmosphere selection to a real grid point while letting synthesis use an arbitrary value.

  ```json
  "bounds": {
    "teff":   {"min": 4000, "max": 7000},
    "logg":   {"min": 0.0,  "max": 5.0},
    "feh":    {"min": -2.5, "max": 0.5},
    "vmicro": {"min": 0.5,  "max": 3.0}
  },
  "t_value_options": ["00", "01", "02", "05"]
  ```

- **Discrete (legacy)** — set `sample_turbvel: true` with `turbvel_options` to draw from the standard `01`–`05` codes, for compatibility with the batch runners.

  ```json
  "sample_turbvel": true,
  "turbvel_options": ["01", "02", "03", "04", "05"],
  "t_value_options": ["00", "01", "02", "05"]
  ```

For a **regular (Cartesian) grid** (`sampling: "grid"`), use a numeric `vmicro` axis — a `start:end:step` range or a uniform comma list in km/s — parsed exactly like the `teff`/`logg`/`feh` axes and stored as floats. It is mutually exclusive with the legacy discrete `turbvel` axis (e.g. `"01,02,05"`), which still works.

  ```json
  "sampling": "grid",
  "axes": {
    "teff":   "4000:7000:250",
    "logg":   "0.0:5.0:0.5",
    "feh":    "-2.5:0.5:0.25",
    "vmicro": "0.5:3.0:0.5",
    "t_value_options": ["00", "01", "02", "05"]
  }
  ```

After synthesis, `turbvel`, `t_value`, and `vmicro` are all preserved as numeric parameter columns.

#### MARCS grid-bounds warnings

When the grid is generated (both LHS and regular-grid paths), the four atmosphere-selection axes are checked against the MARCS standard-composition grid envelope:

| axis     | envelope        |
|----------|-----------------|
| `teff`   | 2500–8000 K     |
| `logg`   | −1.0 to 5.5     |
| `feh`    | −5.0 to 1.0     |
| `vmicro` | 0–5 km/s        |

Sampled rows outside the envelope emit a warning naming the axis, the out-of-range row count, and the sampled range. These rows are **not** dropped — synthesis still runs by snapping to the nearest available atmosphere — but that snap clamps to the grid edge, so the stored atmosphere no longer reflects the requested parameters. Abundances are applied on top of a fixed-composition atmosphere and are deliberately not range-checked.

Override or silence the check from the config:

```json
"warn_outside_marcs_bounds": false,
"marcs_bounds": { "teff": {"min": 3000, "max": 7500} }
```

`marcs_bounds` overrides only the axes you list (e.g. to match a sub-grid you actually have on disk); unlisted axes keep their defaults.

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

### Shard Layout And Chunking

The repository now uses one shared shard-layout rule across the PBS wrappers via `scripts/resolve_shard_layout.py`.

- `SHARD_COUNT` means “split the dataset into this many shards”.
- `ROWS_PER_SHARD` means “target this many rows per shard”.
- `CHUNK_ROWS` in `rechunk_array_example.pbs` is the rechunking equivalent of `ROWS_PER_SHARD`.
- `chunk_rows` in pipeline config or synthesis scripts controls Zarr storage chunking, not how many PBS shards/jobs run.

The wrappers resolve the missing value with ceiling division:

```text
rows_per_shard = ceil(row_count / shard_count)
shard_count    = ceil(row_count / rows_per_shard)
```

If both are supplied, they must cover the full dataset. If neither is supplied:

- `rechunk_array_example.pbs` defaults to `CHUNK_ROWS=1500`
- `turbospectrum_regular_grid_example.pbs` defaults to one-row shards unless config or env overrides say otherwise
- `turbospectrum_example_array.pbs` requires one of `SHARD_COUNT` or `ROWS_PER_SHARD`

You can preview the resolved layout before submitting:

```bash
python3 scripts/resolve_shard_layout.py \
  --zarr-path runs/local-dev/outputs/grids/parameter_grid.zarr \
  --shard-count 10
```

Example output:

```text
ROW_COUNT=15001
ROWS_PER_SHARD=1501
SHARD_COUNT=10
ARRAY_RANGE=0-9
```

For rechunking on PBS, you can now submit from the login node with `bash` and let the script compute and submit the correct array itself:

```bash
INPUT_ZARR=/g/data/y89/mj8805/your_input.zarr \
OUTPUT_DIR=/scratch/mk27/$USER/harps2_rechunked \
SHARD_COUNT=10 \
PYTHON=$(which python) \
bash rechunk_array_example.pbs
```

For synthesis array jobs, compute the layout first or set the matching `-J` range yourself:

```bash
python3 scripts/resolve_shard_layout.py \
  --zarr-path runs/local-dev/outputs/grids/parameter_grid.zarr \
  --rows-per-shard 128

qsub -J 0-9 -v CONFIG_PATH=configs/pipeline/config_pipeline.json,SHARD_COUNT=10 turbospectrum_example_array.pbs

qsub -J 0-9 -v CONFIG_PATH=configs/pipeline/config_regular_grid.example.json,SHARD_COUNT=10,WORKERS=32 turbospectrum_regular_grid_example.pbs
```

The array wrappers now log `ROW_COUNT`, `ROWS_PER_SHARD`, `SHARD_COUNT`, and `ARRAY_RANGE` so it is obvious how the grid is being split before any synthesis work starts.

### Legacy split scripts (optional)

Manual split scripts are still present for backward compatibility:
- `scripts/generate_grid.py`
- `scripts/interpolate_models.sh`
- `scripts/synthesize_spectra.sh`

Use them only if you explicitly need manual phase-by-phase execution.

## Testing

Run the full local test suite before submitting large jobs:

```bash
python3 -m pytest tests/ -v
```

All tests are self-contained, use temporary directories, and require no external data downloads. Dependencies: `pytest`, `numpy`, `zarr`.

### Test coverage by capability

| Capability                       | Test file(s)                                                 |
|----------------------------------|--------------------------------------------------------------|
| Grid generation (regular + LHS)  | `test_grid_generation.py`                                    |
| ML sampling & abundance bounds   | `test_grid_generation.py`, `test_nlte_ascii_departures.py`   |
| Sampling dispatch, vmicro & MARCS bounds | `test_sampling_dispatch.py`                          |
| Spectrum output parsing          | `test_spectrum_output.py`, `test_spectrum_reconstruction.py`  |
| Continuum reconstruction         | `test_spectrum_reconstruction.py`                            |
| Flux/Intensity mode extraction   | `test_spectrum_reconstruction.py`                            |
| Flux metadata inference          | `test_spectrum_reconstruction.py`                            |
| NLTE departure handling (ASCII)  | `test_nlte_ascii_departures.py`                              |
| Shard merging                    | `test_merge_spectra_shards.py`                               |
| Dataset validation               | `test_validate_dataset.py`                                   |
| Shard completeness checking      | `test_validate_dataset.py`                                   |
| Provenance hashing & contract    | `test_provenance_contract.py`                                |
| Lazy spectrum lookup (Zarr)      | `test_lazy_spectrum_loader.py`                               |
| JAX dataloader (splits, stats)   | `test_jax_spectra_dataloader.py`                             |
| Shard layout resolution          | `test_shard_layout.py`                                       |
| Linelist expansion & validation  | `test_linelist_expansion.py`                                 |
| Mu sampling (Intensity mode)     | `test_mu_sampling.py`                                        |
| Fail-fast early abort            | `test_fail_fast_tracker.py`                                  |
| Task identity & parameter hashes | `test_synthesis_task_identity.py`                            |
| Config path normalization        | `test_turbospectrum_config_paths.py`                         |
| Log excerpt extraction           | `test_turbospectrum_logging.py`                              |
| Turbulence parameter persistence | `test_turbulence_parameter_persistence.py`                   |
| Worker count & memory limits     | `test_worker_count.py`                                       |
| Pipeline config & retry logic    | `test_pipeline_from_config.py`                               |
| Regular grid array preparation   | `test_regular_grid_array_prepare.py`                         |
| Regular grid validation          | `test_regular_grid_validation.py`                            |

### Running specific test groups

```bash
# Grid and sampling only
python3 -m pytest tests/test_grid_generation.py tests/test_nlte_ascii_departures.py -v

# Output parsing and validation
python3 -m pytest tests/test_spectrum_reconstruction.py tests/test_validate_dataset.py tests/test_provenance_contract.py -v

# Data loading (lazy + JAX)
python3 -m pytest tests/test_lazy_spectrum_loader.py tests/test_jax_spectra_dataloader.py -v

# HPC layout and sharding
python3 -m pytest tests/test_shard_layout.py tests/test_merge_spectra_shards.py -v
```

### Pipeline smoke tests

For integration-level validation (config parsing, grid generation, optional single-shard synthesis):

```bash
./scripts/test_pipeline_local.sh \
  --config configs/pipeline/config_pipeline.json

# Include one-shard synthesis + dataset validation:
./scripts/test_pipeline_local.sh \
  --config configs/pipeline/config_pipeline.json \
  --with-synthesis --keep
```

On Gadi:

```bash
qsub scripts/test_pipeline_gadi.pbs
# or with overrides:
qsub -v PROJECT=mk27,MAMBA_ENV_NAME=astro,CONFIG_PATH=configs/pipeline/config_pipeline.json scripts/test_pipeline_gadi.pbs
```

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

Optional early-abort safety:

```bash
qsub -v INITIAL_FAILURE_ABORT_THRESHOLD=3 turbospectrum_example.pbs
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

Optional early-abort safety for cold starts:

```bash
qsub -v CONFIG_PATH=configs/pipeline/config_regular_grid.example.json,INITIAL_FAILURE_ABORT_THRESHOLD=3 turbospectrum_regular_grid_example.pbs
```

Behavior summary:
- reads `runtime.shard_count` / `runtime.rows_per_shard` from the regular-grid config and falls back to one-row shards only when neither is set
- in array mode, task `0` prepares the grid once and later tasks wait for it before starting synthesis
- array tasks beyond the true shard count exit cleanly, so it is safe to submit a slightly oversized `-J` range
- non-array runs still support resume/merge in one submission, but chunked shards reduce startup overhead and usually improve CPU utilization substantially
- `FINALIZE_ONLY=1` skips synthesis and just validates/merges the shard directory, which is the intended post-array follow-up step
- `INITIAL_FAILURE_ABORT_THRESHOLD=N` aborts the job once the first `N` shard calculations all fail before any success, and writes a stop marker plus summary under `RUN_ROOT`

### Tiny NLTE Regular-grid Gadi Smoke Jobs

For a very small Fe NLTE smoke test on Gadi, using the regular-grid workflow and ASCII departures, use:

```bash
qsub turbospectrum_regular_grid_nlte_tiny_gadi.pbs
```

Tiny PBS array version:

```bash
qsub turbospectrum_regular_grid_nlte_tiny_array_gadi.pbs
```

Finalize-only follow-up after the tiny array completes:

```bash
qsub turbospectrum_regular_grid_nlte_tiny_finalize_gadi.pbs
```

Notes:
- these wrappers default to `configs/pipeline/config_regular_grid_nlte_fe_tiny_gadi.json`
- the tiny config uses 8 grid rows and defaults to `ROWS_PER_SHARD=2`, so the array job intentionally overshoots and lets extra tasks exit cleanly
- outputs default under `/scratch/<PROJECT>/<USER>/Turbospectrum_NLTE_regular_nlte_tiny` unless you override `RUN_ROOT`
- the goal is a quick scheduler and NLTE wiring check, not a science-scale production run

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
