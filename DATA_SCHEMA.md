# SPICE Synthetic Spectral Grid — Zarr Specification

## Design Goals

This schema is optimized for:

* deterministic dataset regeneration
* efficient ML streaming
* minimal metadata ambiguity
* HPC-friendly writes
* object-store compatibility
* forward schema evolution

The dataset must be  **append-safe and immutable once published.**

# Layout 1 — Synthesis

### Top-Level Layout

```
<grid_root>.zarr
│
├── flux                (float32)   [N_models, N_lambda]
├── wavelength         (float32)   [N_lambda]
├── params             (float32)   [N_models, N_params]
├── param_names        (U32)       [N_params]
│
├── model_id           (uint64)    [N_models]
├── physics_hash       (fixed str) scalar
├── schema_version     (str)       scalar
│
└── attrs
```


# Core Arrays

## flux

**Shape**

```
[N_models, N_lambda]
```

**Meaning**

Continuum-normalized flux unless explicitly overridden in attrs.

**Chunking (VERY IMPORTANT)**

Recommended for final/published datasets:

```
(256–1024 models, full wavelength)
```

Example:

```
chunks = (512, N_lambda)
```

### Pipeline chunk flow

The pipeline uses stage-specific chunking that progressively coarsens toward ML access patterns:

```
Shard output  →  Merged output  →  Rechunked (ML-ready)
(64, ≤65536)     (128, L)          (128, 8192)
```

| Stage | Source | Row chunk | Wavelength chunk | Rationale |
|-------|--------|-----------|------------------|-----------|
| Shard synthesis | `_SHARD_ROW_CHUNK` in `run_turbospectrum_shard.py` | 64 | min(65536, L) | Balance write granularity vs inode count |
| Merge | `--chunk-rows` in `merge_spectra_shards.py` (default 128) | 128 | full L | Row-major reads; matches PBS `MERGE_CHUNK_ROWS` |
| Rechunk | `ROW_CHUNK`, `LAMBDA_CHUNK` in `rechunk_worker.py` | 128 | 8192 | Cache-friendly mini-batch I/O for GPU streaming |

Metadata is consolidated at merge time via `zarr.consolidate_metadata()`.

### Why this chunking?

Most future access patterns will be:

👉 load many spectra

not

👉 slice wavelengths

This optimizes for:

* ML batching
* GPU streaming
* fewer object reads
* minimal inode pressure on parallel filesystems

Avoid tiny wavelength chunks — they destroy performance.

Avoid row-chunk = 1 — it creates one file per spectrum and exhausts inodes at scale.

## wavelength

```
[N_lambda]
```

Constraints:

* strictly monotonic
* identical for all models

If grids diverge in wavelength:

👉 create a new dataset

NOT a new axis.

Mixed wavelength grids cause silent ML bugs.

### params

```
[N_models, N_params]
```

Ordered exactly as:

```
param_names
```

Example parameters:

```
Teff
logg
[Fe/H]
vmicro
alpha
vsini
```

### Critical Rule

Parameter ordering is  **immutable** .

Never reorder.

If changed → bump schema_version.

## param_names

Unicode string array.

Example:

```
["teff", "logg", "feh", "vmicro"]
```

Prefer lowercase snake_case.

Avoid symbols in storage (`[Fe/H]`) — map them in metadata instead.

AI agents parse this more reliably.

### MARCS atmosphere composition columns

When the synthesis resolves a MARCS model atmosphere (exact match or
nearest-neighbour), the composition of the atmosphere *actually used* is recorded
as five additional, immutable trailing parameter columns:

```
marcs_fe_h   # [Fe/H] of the MARCS model
marcs_a_fe   # [alpha/Fe] (standard enhancement vs metallicity)
marcs_c_fe   # [C/Fe]   (solar-scaled in standard composition)
marcs_n_fe   # [N/Fe]   (solar-scaled in standard composition)
marcs_o_fe   # [O/Fe]   (follows alpha in standard composition)
```

These are properties of the model-atmosphere *structure* the lines were
synthesised against — distinct from the requested input abundances
(`feh`, `a`, `c`, `n`, `o`), which may differ (e.g. nearest-neighbour atmosphere
selection, or standard-composition alpha enhancement). They are parsed from the
MARCS `.mod` filename and carried per-row through the shard → merge pipeline.
Rows whose model could not be resolved store `NaN`. Units: `dex`.

### MARCS atmosphere parameter columns

The stellar parameters of the MARCS model atmosphere *actually used* (exact match
or nearest-neighbour) are recorded as three additional trailing parameter columns,
parsed from the resolved model filename:

```
marcs_teff   # Teff  of the MARCS model used (K)
marcs_logg   # log g of the MARCS model used (dex)
marcs_turb   # microturbulence of the MARCS model used (km/s, from the t-label)
```

These are **always reported** for every resolved row, like the composition
columns above. For an exact model match they equal the requested params
(`teff`, `logg`, `turbvel`); when the requested point had no exact model and
synthesis fell back to the nearest atmosphere, they are the grid point the
selection was *clamped to*, so any difference from the requested params is the
clipping amount. Rows whose model could not be resolved (e.g. skipped/existing
outputs) store `NaN`.

The atmosphere's [Fe/H] is **not** duplicated here — it is already reported by the
composition column `marcs_fe_h` above (both decode the model's `z` field). So the
full snapped-atmosphere descriptor is `marcs_teff`, `marcs_logg`, `marcs_turb`
plus `marcs_fe_h`.

# Identity Layer (Extremely High Value)

## model_id

```
uint64
```

Must be deterministic from parameters.

Recommended:

👉 hash(params row)

Example:

```
xxhash64(params)
```

Why this matters:

* enables deduplication
* safe merging
* lineage tracking
* fast diffing

Random IDs are a long-term mistake.

## physics_hash

Scalar string.

Represents the  **entire physics configuration** :

* linelist
* atmosphere grid
* NLTE/LTE
* opacity tables
* synthesis code version

Should be generated from a canonical config file.

Example:

```
sha256(config.yaml)
```

This is your scientific fingerprint.

## schema_version

Example:

```
"1.0.0"
```

Follow semantic versioning:

* MAJOR → breaking layout change
* MINOR → additive metadata
* PATCH → documentation

Never silently mutate schema.

# Attributes (Global)

Stored at root:

```
{
  "title": "SPICE Synthetic Spectral Grid",
  "generator": "spice v0.9.2",
  "created_utc": "2026-02-16T02:11:00Z",

  "flux_definition": "continuum_normalized",

  "wavelength_unit": "angstrom",
  "flux_unit": "relative",

  "parameter_units": {
    "teff": "K",
    "logg": "dex",
    "feh": "dex"
  },

  "physics_hash": "...",
  "git_commit": "...",
  "contact": "maintainer@email"
}
```

Duplicate critical hashes intentionally.

Redundancy > archaeology.

# Naming Convention (MANDATORY)

All published grids must use  **content-addressable naming**:

```
grid_<physics_hash>.zarr
```

Example:

```
grid_7f3a9c2e1b6d.zarr
```

### Definition of `physics_hash`

`physics_hash` is a deterministic cryptographic hash derived from the full physics configuration used to generate the grid.

It MUST include:

* linelist + version
* atmosphere models
* NLTE/LTE choice
* opacity tables
* abundance scale
* turbulence assumptions
* synthesis code version
* wavelength definition
* any physics-altering flags

Recommended:

```
physics_hash = sha256(canonical_config.yaml)
```

# Provenance Layer (REQUIRED)

## Purpose

The provenance group preserves the **full scientific lineage** of the dataset.

This protects against:

* lost configs
* undocumented physics changes
* paper reproducibility failures
* collaborator confusion
* future archaeology

Storage is cheap. Scientific ambiguity is not.

## Layout

```
/provenance
    canonical_config.yaml
    synthesis_config.yaml
    input_config.json
    linelist_manifest.json
    atmosphere_manifest.json
    software_manifest.json
    environment.txt
```

### input_config.json (CRITICAL)

The verbatim total input config exactly as supplied via `--config`, recorded
before normalization or filtering to `TurbospectrumConfig` fields. Unlike
`synthesis_config.yaml` (which holds the effective dataclass config after
normalization and any scratch overrides), this preserves every key from the
original file — including ones the dataclass drops — so the dataset can always be
traced back to the exact input that produced it.

### canonical_config.yaml (CRITICAL)

The single source of truth used to generate `physics_hash`.

Must contain *everything* needed to regenerate the grid.

Example sections:

```
linelist:
  name: gaiaeso
  version: 6.1

atmospheres:
  grid: phoenix
  geometry: spherical

physics:
  nlte: true
  microturbulence: 1.0

synthesis:
  code: turbospectrum
  version: 20.1

wavelength:
  start: 3500
  end: 9000
  resolution: 300000
```

### software_manifest.json

Example:

```
{
  "spice_version": "0.9.2",
  "turbospectrum_version": "20.1",
  "compiler": "gcc 13.2",
  "git_commit": "a81c4d2"
}
```


Silent software drift is a major source of irreproducibility.

Track it.

### linelist_manifest.json

Include:

* source
* version
* checksum
* preprocessing steps

Example:

```
{
  "source": "Gaia-ESO",
  "version": "6.1",
  "sha256": "...."
}
```

Never rely on filenames alone.

### environment.txt

Output of:

```
pip freeze
```

or

```
conda env export
```

# Layout 2 — Training Dataset

## Naming Convention

```
train_<dataset_hash>.zarr
```

Where:

```
dataset_hash = hash(
    physics_hash +
    preprocessing_config +
    wavelength_grid +
    label_definition +
    normalization +
    masking_rules
)
```

Training datasets are also  **immutable artifacts**.

Treat them like compiled ML binaries.

# Top-Level Layout

```
train_<dataset_hash>.zarr
│
├── spectra            float32   [N_samples, N_lambda]
├── labels             float32   [N_samples, N_labels]
├── wavelength         float32   [N_lambda]
│
├── sample_id          uint64
├── source_model_id    uint64
├── physics_hash       str
├── dataset_hash       str
│
├── splits/
│     train_indices
│     val_indices
│     test_indices
│
├── provenance/
│
└── attrs
```

## Critical Design Choice — Rechunk Aggressively

### Training chunking SHOULD differ from synthesis.

Recommended:

```
chunks = (1024–4096 samples, full_lambda)
```

The current rechunk step (`rechunk_worker.py` via `rechunk_array_example.pbs`) uses:

```
ROW_CHUNK    = 128    # rows per chunk
LAMBDA_CHUNK = 8192   # wavelength points per chunk
PARAM_CHUNK  = 512    # param matrix row chunk
ONE_D_CHUNK  = 1024   # 1D auxiliary array chunk
```

Why?

Because GPUs want large contiguous batches.

Not scattered shards.

Rechunk once → benefit forever.

Yes, the rechunk job may be heavy.

It is worth it.

# Should You Duplicate Spectra?

Here is the subtle but important guidance:

## Duplicate intentionally.

Storage is cheaper than GPU underutilization.

Attempting lazy cross-reads from synthesis grids usually results in:

* fragmented IO
* unpredictable latency
* miserable scaling

Controlled duplication is professional infrastructure — not waste.

# Label Matrix

Avoid embedding labels inside metadata.

Use a dense matrix:

```
labels [N_samples, N_labels]
```

Example:

```
Teff
logg
FeH
vmicro
vsini
alpha
```

Maintain:

```
label_names
label_units
```

# Deterministic Splits (Extremely Important)

Never rely on random splits during training.

Store them.

```
/splits/train_indices
```

Now every paper is reproducible.

Every ablation is traceable.

Every collaborator gets identical partitions.

This is quietly becoming expected in serious ML research.

# Provenance for Training Sets

Training datasets must record both physics AND preprocessing.

## Layout

```
/provenance
    source_grid.txt
    preprocessing.yaml
    normalization.yaml
    masking.yaml
    augmentations.yaml
```
