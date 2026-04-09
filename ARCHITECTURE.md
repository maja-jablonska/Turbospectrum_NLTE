AI AGENTS: This file defines non-negotiable system constraints. You must read this before proposing structural code changes.

## System Purpose

This repository provides a scalable wrapper around **Turbospectrum** for generating large synthetic spectral grids suitable for machine learning, emulator training, and astrophysical inference.

The system is designed for:

* HPC execution
* massive parameter sweeps
* fault tolerance
* deterministic outputs
* zero manual intervention

## Core Architectural Principles

### 1. Determinism Over Convenience

Every run must be reproducible.

**Rules:**

* No hidden randomness
* All parameters must originate from config files
* Environment must be loggable
* Versions must be traceable

If a spectrum cannot be reproduced → it is a bug.

### 2. Shard-First Design

The system assumes the grid is too large for monolithic execution.

Each shard must be:

* independent
* restartable
* idempotent
* safe to re-run

**Invariant:**

Running the same shard twice must never corrupt outputs.

### 3. Failure Is Expected

At 10k–100k shards:

* nodes die
* disks fill
* temp files explode
* MPI hiccups
* PBS kills jobs

Failure handling is not optional — it is architecture.

**Required behavior:**

If a shard fails:

✅ log reason

✅ release lock

✅ retry safely

✅ never poison shared state

### 4. IO Is the Primary Bottleneck

Not CPU.

Design assumes filesystem pressure dominates runtime.

**Therefore:**

* minimize file creation
* batch writes
* prefer sequential access
* avoid metadata storms

Small-file explosions are forbidden.

## System Components

### Config Layer

Defines the parameter space.

**Must be:**

* declarative
* versionable
* hashable

Recommended pattern:

```
configs/
    grid_v1.yaml
```

Hash configs → embed hash into outputs.

This prevents silent grid drift.

### Task Generator

Produces shard indices deterministically.

Example:

```
seq 0 N_SHARDS-1
```

Never infer shard counts dynamically.

Shard cardinality must remain stable across retries.

### Execution Layer

Each shard executes:

```
config → atmosphere → Turbospectrum → spectrum → postprocess → store
```

No cross-shard communication allowed.

Ever.

If shards need to talk → architecture is wrong.

### Locking Strategy

Atomic task acquisition only.

Example pattern:

```
flock → pop task → release
```

Never rely on:

* directory existence
* naive file checks
* timestamps

Race conditions scale superlinearly.

### Storage Strategy

Outputs must favor **few large files** over many small ones.

Preferred formats:

* Zarr (large grids)
* Parquet (metadata)
* HDF5 (acceptable)

Avoid raw FITS explosions unless bundled.

### Chunk Constants (Authoritative Reference)

The pipeline uses different chunk layouts at each stage, tuned for that stage's access pattern:

| Stage | File | Constant / Default | Row chunk | Wavelength chunk |
|-------|------|--------------------|-----------|------------------|
| Shard synthesis | `run_turbospectrum_shard.py` | `_SHARD_ROW_CHUNK = 64` | 64 | min(65536, L) |
| Merge | `merge_spectra_shards.py` | `--chunk-rows` (default 128) | 128 | full L |
| Rechunk (ML) | `rechunk_worker.py` | `ROW_CHUNK = 128`, `LAMBDA_CHUNK = 8192` | 128 | 8192 |

**Why they differ:**

* **Shard synthesis** chunks at 64 rows to balance write granularity against inode count. Each worker may flush partial results; small-ish chunks keep memory low while avoiding one-file-per-spectrum.
* **Merge** uses 128-row chunks with full-wavelength extent because downstream reads are row-major (load N spectra, all wavelengths). This matches the PBS `MERGE_CHUNK_ROWS` default.
* **Rechunk** splits wavelengths into 8192-point tiles for ML streaming where mini-batch reads benefit from smaller, cache-friendly chunks.

**Inode budget rule of thumb:**

```
files_per_shard ≈ 2 * ceil(rows / _SHARD_ROW_CHUNK) + ~50 overhead
```

For 100 shards x 1000 rows: ~3,200 chunk files (at row-chunk 64) vs ~200,000 with the old row-chunk of 1.

All chunk defaults must stay aligned. If you change one, grep for `_SHARD_ROW_CHUNK`, `ROW_CHUNK`, `MERGE_CHUNK_ROWS`, and `chunk_rows` across the codebase.

### Temporary Storage

All temp files must use node-local scratch when available.

Example:

```
$TMPDIR
/scratch
/dev/shm   (if safe)
```

Never write heavy temps to shared home.

Most HPC failures originate here.

### Memory Philosophy

Memory must be predictable.

Avoid:

* Python object explosions
* giant intermediate arrays
* implicit copies

Prefer:

* streaming
* generators
* chunking

If memory usage is surprising → treat as a bug.

### Parallelism Model

**Primary parallelism: shard-level.**

NOT Python multiprocessing inside shards unless proven safe.

Why:

Nested parallelism destroys HPC efficiency.

Bad:

```
96-core node
→ shard
→ multiprocessing pool of 96
```

You just created scheduler warfare.

### CPU vs Memory Bound Detection

If CPU usage < ~30%:

You are probably IO or memory bound.

Do NOT blindly add cores.

Fix data movement.

## Idempotency Rules

Before computing a shard:

Check if output exists AND is valid.

Validation must be stronger than file existence.

Example:

* size threshold
* checksum
* readable header

Corrupted outputs must be deletable safely.

## Logging Requirements

Each shard must emit:

* start time
* parameters
* node
* memory snapshot
* retry count
* failure reason

Logs must be grep-friendly.

Avoid pretty logs.

You are debugging at scale.

## Retry Philosophy

Retries must assume partial execution occurred.

Therefore:

* clean temp files
* avoid append-only formats unless transactional
* never duplicate spectra

Max retries should be finite.

Infinite retry loops are cluster abuse.

## Anti-Patterns (Forbidden)

### ❌ Hidden Global State

No module-level caches that leak across shards.

### ❌ Dynamic Parameter Mutation

Grid definitions must never change mid-run.

### ❌ Silent Exception Handling

If you catch — you log.

Always.

### ❌ Writing Thousands of Tiny Files

Metadata servers will punish you.

## Testing Strategy

Three tiers:

### Smoke Tests

Run locally.

Generate:

* 3–5 spectra
* full pipeline

Detect integration breakage early.

### Shard Simulation

Launch ~10 shards.

Validate:

* locking
* retry
* IO behavior

Most architecture bugs appear here.

### Scale Rehearsal

Before 100k shards → run 1k.

Always.

Clusters are unforgiving.

## Observability

At scale, intuition fails.

Track:

* spectra / second
* failure rate
* retry distribution
* median runtime
* IO throughput

If runtime variance is huge → investigate immediately.

Usually filesystem contention.

## Evolution Rules

When modifying architecture:

Ask:

> Does this still work at 10× scale?

If unsure — assume no.

Optimize for future grid sizes, not current ones.

Spectral ML pipelines only grow.

## Design Goal

This system should eventually run unattended for weeks and produce a training-grade spectral database with zero manual repair.

If humans must babysit jobs → architecture is insufficient.
