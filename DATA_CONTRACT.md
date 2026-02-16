## Purpose

This document defines the required structure, metadata, and scientific validity criteria for all synthetic spectra produced by this system.

Any dataset that violates this contract must be considered **invalid for scientific use** until corrected or regenerated.

The contract exists to guarantee:

* reproducibility
* schema stability
* ML compatibility
* astrophysical correctness
* long-term usability

This is not documentation — it is an enforceable specification.

# Guiding Principle

> A spectrum is only useful if its provenance and physics are unambiguous.

Speed is irrelevant if the data cannot be trusted.

# Dataset Identity (Mandatory)

Every generated dataset must embed enough metadata to uniquely answer:

> “Exactly what produced this spectrum?”

At minimum:

### Required Provenance Fields

* config hash
* grid definition hash
* git commit SHA
* Turbospectrum version or binary hash
* linelist identifier + version
* atmosphere model identifier
* synthesis timestamp
* pipeline version

These must be stored either:

✅ inside the Zarr root attrs

or

✅ in a colocated `metadata.json`

Prefer embedding inside the dataset.

External metadata is easier to lose.

# Structural Contract

## Required Arrays

Each shard or merged dataset must contain:

### `wavelength`

**Shape:** `(N_lambda,)`

Requirements:

* strictly increasing
* evenly spaced OR spacing explicitly recorded
* units specified (e.g., Angstrom)
* identical across all shards within a grid

If wavelength grids differ → it is not one dataset.

### `flux`

**Shape:** `(N_spectra, N_lambda)`

Requirements:

* finite values only
* no NaNs
* no infinities
* no negative flux unless physically justified and documented

Flux units must be explicit:

Example:

```
continuum normalized
or
erg / s / cm^2 / Å
```

Implicit units are forbidden.

---

### `parameters`

Table or arrays describing stellar labels.

Typical fields:

* `teff`
* `logg`
* `[Fe/H]`
* microturbulence
* alpha enhancement
* rotational velocity (if applied)

Rules:

* parameter ordering must match flux axis
* no silent column renaming
* units must be defined

Schema drift is not allowed.

### `status`

Per-spectrum synthesis status.

Example values:

```
success
convergence_warning
fallback_model
```

Never silently drop failed models.

Failed physics is still information.

# Scientific Validity Checks

These should ideally run automatically post-shard.

## Flux Sanity

Reject spectra with:

* NaNs
* infinities
* completely flat regions (suggesting failure)
* zero vectors
* extreme spikes inconsistent with line physics

Heuristic checks are encouraged.

Perfect physics validation is not required — obvious corruption detection is.

## Wavelength Consistency

All spectra within a dataset must share an identical wavelength grid.

Tolerance-based matching is discouraged.

Exact equality is strongly preferred.

Why?

Interpolation differences poison ML models.

## Parameter Space Integrity

The realized grid must match the declared grid.

No:

* missing regions
* duplicated parameter points
* mutated ranges

If the grid changes → version the dataset.

Never quietly extend it.

# Immutability Rule

Once published internally:

> A dataset must never be modified in place.

Allowed actions:

✅ create v2

✅ patch via new dataset

✅ deprecate old versions

Forbidden:

❌ overwriting spectra

❌ changing metadata silently

ML experiments must always be traceable to a fixed dataset.

# Versioning Strategy

Dataset versions should increment when:

* physics inputs change
* wavelength grid changes
* parameter space changes
* normalization changes
* synthesis code materially changes

Minor metadata additions may remain within the same version.

When uncertain — bump the version.

Storage is cheaper than scientific ambiguity.

# Merge Contract

A merged dataset is valid only if:

* all shards pass validation
* schemas are identical
* provenance matches
* wavelength grids match exactly

Merge scripts must fail loudly if this is violated.

Silent coercion is forbidden.

# ML Compatibility Layer (Strongly Recommended)

To future-proof the dataset:

## Record Normalization State

Explicitly state whether spectra are:

* continuum normalized
* pseudo-normalized
* raw flux

ML models are extremely sensitive to this.

## Record Masking

If pixels are masked:

* define mask value
* explain criteria

Never leave models guessing.

## Record Resolution

Provide:

* resolving power

  or
* instrumental FWHM

This becomes critical for emulator generalization.

# Provenance Example (Zarr attrs)

Example structure:

```
{
  "dataset_version": "1.0",
  "config_hash": "...",
  "git_sha": "...",
  "turbospectrum_version": "...",
  "linelist": "VALD_2024",
  "atmosphere_models": "MARCS",
  "flux_units": "continuum_normalized"
}
```


Keep it boring.

Boring metadata survives decades.

# Validation Levels

## Level 0 — Structural

Arrays exist and shapes match.

## Level 1 — Numerical

No NaNs, infinities, obvious corruption.

## Level 2 — Scientific

Basic astrophysical sanity.

Example:

* hotter stars generally higher continuum
* metallicity affects line density

(Not strict — just gross-error detection.)

# Deprecation Policy

If a dataset is discovered to have issues:

Do not delete silently.

Mark as:

```
DEPRECATED
reason: `<text>`
replacement: `<dataset>`
```

Someone may have trained on it.

Protect them from unknowingly publishing on flawed data.

# Design Philosophy

> Future researchers should be able to use this dataset without contacting its creators.

If interpretation requires tribal knowledge…

The contract has failed.
