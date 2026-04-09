## Purpose

This document describes **how to run the Turbospectrum grid system safely and repeatably** (local → small HPC test → full-scale production), including:

* preflight checks
* launching PBS shard jobs
* monitoring progress
* handling failures
* resuming runs without corruption
* post-run validation

The goal is:  **unattended execution at scale.**

## Key Concepts

### Grid Identity

Every production grid run must be uniquely identified by:

* config hash
* git commit SHA
* Turbospectrum binary version/hash
* linelist version/hash
* atmosphere source/version/hash

This identity must be logged and (ideally) embedded in outputs.

### Directory Layout (Recommended)

Example run root:

```
/scratch/<proj>/<user>/turbogrids/<grid_name>/
  config/                  (copied config + resolved configs)
  inputs/                  (linelists, atmospheres if staged)
  task_queue/              (task file, lock file)
  logs/
    pbs/
    shards/
  outputs/
    zarr/ or hdf5/         (final outputs)
  tmp/                     (optional run-local temp, preferably node-local)
  reports/
    validation/
    summary/
```

**Rule:** Never run large grids from `$HOME`. Use scratch/project.


## Preflight Checklist (Do This Every Time)

### 1) Pin the Code

* Ensure you’re on the intended git commit.
* Record: `git rev-parse HEAD`
* Dirty working tree is forbidden for production runs.

### 2) Resolve Config

* Copy config into `run_root/config/`
* Store a config hash (e.g., SHA256 of the final resolved YAML).
* Ensure shard count and parameter space are frozen.

### 3) Confirm Storage Budget

Check:

* available space in run root filesystem
* expected output volume
* expected temp usage (esp. here-doc / Turbospectrum intermediates)

If space is tight, reduce concurrency or change output strategy before launch.

### 4) Validate Inputs

Confirm:

* linelist files exist and are readable
* atmosphere models exist and are readable
* Turbospectrum binaries are callable (`--help` or a small test)

### 5) Choose Concurrency Intentionally

Decide:

* how many shards in flight simultaneously
* resources per shard (cores, memory, walltime)
* queue/project settings

Avoid “just throw 96 cores at it.”

## Modes of Operation

### Mode A — Local Smoke Test (Required)

Goal: verify end-to-end pipeline on 1–5 grid points.

Run:

* 1 shard
* minimal wavelength range if possible
* full write + validate

Pass criteria:

* produces valid output
* validation checks pass
* logs include grid identity + shard parameters

### Mode B — Small HPC Pilot (Strongly Recommended)

Goal: catch PBS/locking/temp/IO issues early.

Run:

* 10–100 shards
* realistic runtime
* realistic output format

Pass criteria:

* no lock hangs
* no temp explosions
* no widespread partial writes
* stable runtime distribution (within reason)

### Mode C — Production Scale

Goal: full grid at 10k–100k+ shards, unattended.

Requirements:

* preflight complete
* pilot run successful
* validation pipeline ready

## Task Queue Setup

### Task File Creation

Task indices must be deterministic and frozen.

Example (bash):

```
seq 0 $((N_SHARDS-1)) > task_queue/tasks.txt
```

**Important:** Don’t rely on implicit shard count. `N_SHARDS` must be explicit.

### Lock File

Use a single lock file:

```
task_queue/tasks.lock
```

Lock should guard only task acquisition, not shard execution.

## PBS Launch Procedure (Typical)

### 1) Create Run Root

* Create the run directory.
* Copy configs and record hashes/versions.

### 2) Generate Task Queue

* Generate `tasks.txt`.
* Ensure `tasks.txt` length equals `N_SHARDS`.

### 3) Submit Worker Job(s)

Submit a PBS job that launches workers which:

* acquire shard IDs
* run shard
* validate output
* log results
* repeat until tasks empty

**Recommended pattern:** one PBS job runs a loop that processes many shards serially (or a moderate worker pool), instead of one PBS job per shard. This reduces scheduler overhead.

## Environment & Modules

### Always Log Runtime Environment

Each job should log:

* hostname
* module list
* python version
* `pip freeze` / `conda env export` (or a pinned env file)
* Turbospectrum binary path + hash if possible
* relevant env vars (`TMPDIR`, threading vars, etc.)

### Threading Controls

To avoid accidental over-subscription:

Set defaults like:

* `OMP_NUM_THREADS=1`
* `MKL_NUM_THREADS=1`
* `OPENBLAS_NUM_THREADS=1`

Unless intentionally using threaded libs.

### PBS Variable Reference (NCI Gadi)

Use `PBS_NCPUS` (not `PBS_NP`) for the CPU count on Gadi PBS Pro. The auto-detection in `run_turbospectrum.py` checks `PBS_NCPUS` first, with `PBS_NP` as a legacy fallback.

Resource directives should use the flat format:

```
#PBS -l ncpus=32
#PBS -l mem=64GB
```

Avoid the `select=1:ncpus=32:mem=64gb` form — it allocates a full node even when packing would suffice.

For multi-socket nodes (e.g. 96-CPU benchmark), `numactl --localalloc` is applied automatically when available (see `pbs_benchmark_workers_96.pbs`).

## Temp & Scratch Policy

### Required

All heavy temp must go to node-local scratch when available:

* `$TMPDIR` on PBS systems
* `/scratch` local if provided
* `/dev/shm` only if you know it’s safe and large enough

### On Shard Start

Check free space. Abort early if below threshold.

### On Shard Exit (Success or Fail)

Clean temp aggressively.

Temp survival across runs is considered a bug.

### Post-Merge Cleanup (Inode Reclamation)

After a successful merge, the shard directory should be removed to reclaim inodes. The PBS merge scripts (`merge_shards_example.pbs`, `turbospectrum_regular_grid_example.pbs`) do this automatically unless disabled:

```bash
CLEANUP_SHARDS=0 qsub merge_shards_example.pbs   # keep shards after merge
```

Default behavior: shards are deleted after merge; scratch tmp is cleaned after synthesis.

### Chunk Layout Reference

See ARCHITECTURE.md § "Chunk Constants" for the authoritative table of chunk sizes at each pipeline stage (`_SHARD_ROW_CHUNK`, `--chunk-rows`, `ROW_CHUNK`/`LAMBDA_CHUNK`).

All PBS scripts default `MERGE_CHUNK_ROWS=128`, aligned with the Python argparse default in `merge_spectra_shards.py`. JSON configs may override this (e.g. `runtime.chunk_rows`). Tiny test configs intentionally use smaller values.

## Monitoring

### What to Monitor During Runs

**Throughput**

* shards completed per hour
* spectra per second (if meaningful)

**Failure rate**

* failures per 1000 shards
* retry distribution

**Runtime variance**

* if variance explodes: likely filesystem contention

**CPU utilization**

* consistently low CPU% often indicates IO-bound execution

### Where to Look

* `logs/pbs/` for scheduler-level errors
* `logs/shards/` for per-shard logs
* summary counters (recommended: append-only CSV/Parquet)

## Failure Handling & Recovery

### Golden Rule

**Never manually edit outputs to “patch things up.”**

Instead, delete invalid outputs and rerun shards.

### Typical Failures

#### 1) “No space left on device”

Action:

* stop submitting more workers
* identify whether `$TMPDIR` or run root is full
* reduce concurrency and rerun failed shards

Prevent:

* temp checks + cleanup
* throttle concurrent shards

#### 2) Lock Hang / Deadlock

Action:

* verify no process still holds lock
* ensure lock is only used for task acquisition
* if corrupted lock state: regenerate task queue from missing shards list

Prevent:

* keep lock hold times extremely short
* log lock acquisition / release timing (optional)

#### 3) Partial Output Corruption

Action:

* validation should detect
* delete corrupted artifact
* rerun shard

Prevent:

* atomic writes (write temp → rename)
* robust validation beyond file existence

### Atomic Output (Required for Production)

**Your pipeline must support `--output TMP_PATH`.** If it currently writes directly to the final shard path, change that. Atomic rename is one of the biggest stability upgrades possible.

* **`pipeline_from_config.py`**: Pass `--output TMP_PATH` to write to a temp path. The pipeline forwards this as `--output-tmp` to the synthesis scripts.
* **`run_turbospectrum_shard.py`** and **`synthesize_spectra_from_zarr.py`**: Accept `--output-tmp TMP_PATH`. Write to `TMP_PATH`, then `os.rename(TMP_PATH, final_path)`. The final path is always `--output-zarr`.
* **Same filesystem**: Temp and final paths must be on the same filesystem for true atomic rename. Cross-FS rename does copy+delete (not atomic).

#### 4) Systemic Bug (Many Shards Failing)

Action:

* stop the run
* reproduce on 1 shard locally
* fix
* resume from remaining shards

Prevent:

* pilot runs
* smoke tests

## Resuming a Run

Resuming must be safe and deterministic.

### Recommended Resume Flow

1. Scan outputs and validate.
2. Build a list of missing/invalid shard IDs.
3. Regenerate `tasks.txt` from that list.
4. Relaunch workers.

Never “continue” by hoping the old task queue is clean unless you are certain.

## Post-Run Validation

### Minimal Required Outputs

* summary of shard status (success/fail/retried)
* validation report counts
* histogram of runtimes
* list of failed shards (should be empty or explicitly explained)

### Dataset Integrity Checks

* no missing shard IDs
* no duplicate shard IDs
* schema consistent across all outputs
* config hash and git SHA recorded

## Operational Defaults (Recommended Starting Point)

For large runs on shared HPC:

* keep concurrency conservative at first
* scale up only after stability confirmed
* prefer fewer workers doing more shards each (reduces scheduler overhead)
* avoid nested multiprocessing inside shard code

### Inode Budget

At production scale, inode exhaustion is a real risk on parallel filesystems. Current per-shard footprint:

```
files_per_shard ≈ 2 * ceil(rows / 64) + ~50
```

For a 100-shard run with 1000 rows/shard: ~3,200 chunk files total. After merge + cleanup: ~2,500 files in the final Zarr.

If file count grows faster than shard count, investigate chunk sizes. The authoritative chunk constants are in ARCHITECTURE.md.

## Safety Switches (Strongly Recommended)

The worker should support:

* `--max-shards` (limit per job)
* `--max-failures` (stop if too many)
* `--retry N`
* `--retry-backoff-seconds`
* `--dry-run` (print planned actions)
* `--output TMP_PATH` (atomic write: write to temp, then rename to final)

These prevent runaway disasters.

## Operational Philosophy

> Scale is not a bigger run.
>
> Scale is a different kind of run.

The system is considered operationally mature when:

* runs can execute unattended
* failures are bounded and recoverable
* output integrity is provable

---
