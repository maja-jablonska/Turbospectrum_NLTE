# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Required Reading (Architectural Contracts)

Before proposing structural changes, read these — they define non-negotiable constraints. If a change would violate an invariant, flag the conflict instead of generating the code.

- `SYSTEM_RULES.md` — the five load-bearing rules (idempotency, immutability, atomic writes, config-as-truth, no nested multiprocessing)
- `ARCHITECTURE.md` — shard-first design, chunk constants, parallelism model
- `INVARIANTS.md` — properties that must always hold
- `FAILURE_MODES.md` — known HPC failure modes and required defenses
- `OPERATIONS.md` — how to launch/monitor/resume production runs
- `DATA_CONTRACT.md` + `DATA_SCHEMA.md` — output schema, provenance, chunking

The `.cursor/rules.md` file instructs all agents to obey these docs.

## System Purpose

A scalable Python wrapper around the **Turbospectrum** Fortran radiative-transfer code (`source/`, built via `exec-gf/Makefile` etc.) for generating large synthetic stellar spectral grids (LTE + NLTE) for ML / emulator training. The Fortran code itself is legacy science code (Bertrand Plez et al.); almost all active development happens in `scripts/` and `configs/`.

## Build & Run Commands

### Fortran binaries (Turbospectrum itself)

```bash
cd exec-gf && make all       # gfortran — primary dev build
cd exec-intel && make all    # Intel ifort
cd exec-ifx && make all      # Intel ifx (newer)
```

Builds `babsma_lu` and `bsyn_lu` in the exec directory. Interpolator binaries live in `interpolator/` (see `interpolator/readme`).

### Python tests

```bash
python3 -m pytest tests/ -v                          # full suite
python3 -m pytest tests/test_shard_layout.py -v      # single file
python3 -m pytest tests/ -k "nlte" -v                # by keyword
./scripts/test_pipeline_local.sh --config configs/pipeline/config_pipeline.json  # integration smoke
```

All tests are self-contained (temp dirs, no external downloads). Deps: `pytest numpy zarr`.

### Main pipeline entry points

```bash
# One-config pipeline (preferred): grid gen + synthesis in one go
python3 scripts/pipeline_from_config.py --config configs/pipeline/config_pipeline.json

# Phase splits
python3 scripts/pipeline_from_config.py --config ... --skip-synthesis
python3 scripts/pipeline_from_config.py --config ... --skip-grid
python3 scripts/pipeline_from_config.py --config ... --synthesis-mode sharded --shard-index N --shard-count M

# Merge sharded output
python3 scripts/merge_spectra_shards.py --shard-dir ... --grid-zarr ... --output-zarr ...

# Reset generated outputs (dry-run by default; --apply to delete)
python3 scripts/clean_pipeline_outputs.py --config ... --apply

# Compute shard layout before submitting
python3 scripts/resolve_shard_layout.py --zarr-path ... --shard-count N
```

### Data download

```bash
./scripts/download_data.sh --all           # atmospheres + NLTE + linelists
./scripts/download_data.sh --atmospheres MARCS
./scripts/download_data.sh --nlte-atoms all
./scripts/download_data.sh --linelists
```

### PBS / Gadi

Production synthesis uses a single config-driven script, `turbospectrum_pipeline_example.pbs`, for **both** LHS/ML and regular/Cartesian grids — the grid type is selected by the config (`grid.sampling: "lhs"|"grid"`), not by the script name. Common overrides via `qsub -v CONFIG_PATH=...,RUN_ROOT=...,WORKERS=32,ROWS_PER_SHARD=128,...`. `FORCE_RESTART=1` wipes the grid zarr and regenerates it, then also discards prior shard outputs and the merged zarr (grid is kept intact under `FINALIZE_ONLY=1`); default behavior resumes missing shards. `FINALIZE_ONLY=1` skips synthesis and just validates/merges. `SKIP_SYNTHESIS=1` (or `runtime.skip_synthesis`) builds the grid only. `AUTO_PRUNE_STALE_SHARDS=1` (opt-in) drops layout-mismatched shards before resuming. (`scripts/synthesize_regular_grid.py` remains as the regular-grid sampler library imported by `generate_grid.py`, not a standalone production entry point.)

## Repository Layout (Big Picture)

```
source/, exec-*/, interpolator/, Utilities/    # Fortran: Turbospectrum + model atmosphere interp
scripts/                                       # Python wrapper (active dev)
configs/
  pipeline/    sampling/    synthesis/    training/
runs/<run_name>/
  config/  task_queue/  logs/{pbs,shards}/  outputs/{grids,shards,zarr}/  tmp/  reports/
tests/          # pytest, all self-contained
dagster_orch/   # optional Dagster orchestration (dg tool, see pyproject.toml)
DATA/, DOC/, COM/   # input data & Turbospectrum docs (vendor)
```

Generated artifacts belong under `runs/`, never under `scripts/`. Legacy root-level `.pbs` and grid CSVs are ignored by `.gitignore` except explicitly whitelisted examples.

## Architectural Cheat Sheet

**Pipeline data flow:** `config → grid CSV + parameter_grid.zarr (with index.parquet) → sharded synthesis → merged spectra.zarr → (optional) rechunked ML-ready store`.

**Shard-first:** every shard must be independent, idempotent, restartable. No cross-shard communication, no shared mutable state, no module-level caches that leak across shards.

**Parallelism:** primary scaling is shard-level (one PBS slot per shard worker). Do NOT add Python multiprocessing *inside* a shard without profiling evidence — nested parallelism collapses HPC throughput. Thread limits (`OMP_NUM_THREADS=1`, etc.) are set by the PBS wrappers.

**Atomic writes:** synthesis scripts accept `--output-tmp TMP_PATH`; they write there then `os.rename()` to the final `--output-zarr`. Temp and final must be on the same filesystem. `pipeline_from_config.py` forwards `--output` as `--output-tmp`.

**Chunk constants (keep aligned — grep all if you change one):**

| Stage           | File                          | Constant                   | Row chunk | λ chunk         |
|-----------------|-------------------------------|----------------------------|-----------|-----------------|
| Shard synthesis | `run_turbospectrum_shard.py`  | `_SHARD_ROW_CHUNK = 64`    | 64        | min(65536, L)   |
| Merge           | `merge_spectra_shards.py`     | `--chunk-rows` default 128 | 128       | full L          |
| Rechunk (ML)    | `rechunk_worker.py`           | `ROW_CHUNK=128, LAMBDA_CHUNK=8192` | 128 | 8192       |

PBS wrappers default `MERGE_CHUNK_ROWS=128` to match. Pipeline config `runtime.chunk_rows` can override.

**Shard layout rules** (`scripts/resolve_shard_layout.py` is the shared resolver):
- `SHARD_COUNT` = split dataset into N shards
- `ROWS_PER_SHARD` = target rows per shard
- Resolver does ceiling-division to fill in the missing one
- Default behavior when neither is supplied depends on the wrapper (see README.md "Shard Layout And Chunking")

**Provenance:** outputs must embed config hash, git SHA, Turbospectrum version, linelist/atmosphere identifiers — see `scripts/provenance_contract.py` and `tests/test_provenance_contract.py`. Grid names should be content-addressable (`grid_<physics_hash>.zarr`).

**Validation before merge:** `scripts/validate_dataset.py` runs automatically in the PBS flow; merge must fail loudly on shard-completeness / wavelength / NaN violations.

**PBS variable on Gadi:** use `PBS_NCPUS` (not `PBS_NP`). `run_turbospectrum.py` prefers `PBS_NCPUS` with `PBS_NP` as legacy fallback. Use flat `#PBS -l ncpus=N,mem=NGB` — avoid `select=…` which allocates whole nodes.

## Non-Obvious Conventions

- Config is the single source of truth. Any physics-affecting parameter that can't be traced back to a config file is considered an architectural leak.
- `.gitignore` blacklists `/configs/**` and `*.pbs` by default; new template files must include `_example` in the name (or be explicitly whitelisted) to be committed.
- The "split" legacy scripts (`scripts/generate_grid.py`, `scripts/interpolate_models.sh`, `scripts/synthesize_spectra.sh`) are kept for back-compat only — prefer `pipeline_from_config.py`.
- Param names in stored arrays are lowercase snake_case (`feh`, not `[Fe/H]`); symbol→name mapping lives in attrs.
- When deleting "unused" files under `runs/`, `COM/`, `scratch`, or `$TMPDIR`, treat them as potentially in-progress work from another shard/job — investigate before `rm`.
