## Purpose of This Document

This file defines properties of the system that must  **always remain true,** regardless of refactors, optimizations, or feature additions.

If a proposed change violates an invariant, the change is presumed incorrect unless the invariant itself is deliberately redesigned.

These are not suggestions.

They are architectural constraints.

# Core System Invariants

## 1. Reproducibility Is Absolute

Given:

* identical config
* identical Turbospectrum version
* identical line lists
* identical atmosphere models

The system must produce **bitwise-stable or scientifically equivalent spectra.**

### Implications

* No hidden randomness
* No dependence on execution order
* No environment-sensitive defaults
* No floating config values

If reproducibility fails → treat as a critical defect.

## 2. Shards Are Fully Independent

A shard must never depend on:

* another shard’s output
* execution timing
* shared mutable state
* global caches

### Hard Rule

> Any shard must be runnable alone on a fresh machine and succeed.

If this is not true — the architecture is broken.

## 3. Outputs Are Immutable

Once a spectrum is written and validated:

It must **never be modified in place.**

Allowed actions:

✅ write

✅ verify

✅ read

Forbidden:

❌ partial overwrite

❌ append without transactional safety

❌ silent regeneration

If regeneration is required → write a new version.

This protects downstream ML training from silent data drift.

## 4. Idempotent Execution

Running the same shard multiple times must result in exactly one valid output.

Never duplicates.

Never corruption.

### Required Behavior

Before computation:

Validate whether output already exists.

Validation must exceed file existence.

Examples:

* schema check
* readable header
* wavelength count
* checksum

If validation fails → delete and recompute safely.

## 5. Failures Must Not Poison the System

A failed shard must leave the system in a clean state.

No:

* locked task files
* half-written outputs
* unreleased scratch space
* dangling temp files

After failure, the system must behave as if the shard never ran.

## 6. The Filesystem Is a Shared, Scarce Resource

Design assumes metadata pressure is a primary scaling constraint.

Therefore:

### Forbidden

❌ millions of tiny files

❌ excessive directory fan-out

❌ repeated open/close cycles

### Preferred

✅ batched writes

✅ chunked storage

✅ sequential access

If a change increases file count dramatically — it is likely wrong.

## 7. Memory Usage Must Be Predictable

The memory footprint of a shard must not scale unexpectedly with:

* wavelength resolution
* parameter count
* retry behavior

Avoid hidden allocations.

Implicit array copies are considered defects at scale.

If peak memory cannot be estimated → redesign.

## 8. Parallelism Must Be Explicit

The system uses **shard-level parallelism as the primary scaling mechanism.**

Nested parallelism is prohibited unless explicitly justified.

Danger pattern:

```
PBS allocates 96 cores
→ shard launches multiprocessing
→ scheduler contention
```

This destroys cluster efficiency.

Assume one shard = one compute slot unless proven otherwise.

## 9. The Parameter Space Is Frozen Per Run

Once a grid begins execution:

The parameter definition must never change.

No:

* inserting new grid points
* mutating ranges
* altering sampling

If the grid changes → it is a new grid.

Version it.

Hash it.

Record it.

## 10. Config Is the Single Source of Truth

No parameter influencing physics may originate from:

* CLI defaults
* environment variables
* hidden constants

Everything must trace back to config.

If you cannot answer:

> “Where is this value defined?”

The architecture is leaking.

## 11. Logs Must Enable Post-Mortem Debugging

Every shard must emit enough information to diagnose failure **without rerunning it.**

Minimum expectation:

* shard id
* parameters
* node
* runtime
* retry count
* failure reason

Pretty logs are irrelevant.

Searchability is mandatory.

## 12. Temporary Storage Must Evaporate

Temp files must never accumulate across retries or job restarts.

Node-local scratch is strongly preferred.

After shard completion:

There should be nothing left to clean manually.

Manual cleanup is architectural failure.

## 13. Scientific Integrity Overrides Throughput

Performance optimizations must never alter:

* spectral fidelity
* wavelength grids
* radiative transfer correctness

Faster wrong data is worse than slower correct data.

Always.

## 14. Schema Stability Matters

Output formats must not change silently.

If schema evolves:

→ bump version

→ document change

→ maintain backward readability when feasible

Your future ML pipelines depend on this stability.

## 15. Automation Is the End State

The system is considered healthy only when:

* grids run unattended
* retries self-heal
* outputs validate automatically

If humans must monitor runs constantly, an invariant is likely being violated.

# Operational Invariants

## Never Trust “File Exists”

Always validate.

Cluster failures frequently produce ghost files.

## Never Trust Defaults

Explicit is scalable.

Implicit is fragile.

## Never Assume the Cluster Is Healthy

Nodes die.

Disks fill.

Networks stall.

Design accordingly.

# When an Invariant May Be Broken

Rare — but possible.

Required process:

1. Document why the invariant no longer serves the system.
2. Update architecture docs.
3. Version the change.
4. Announce it in the repo.

Undocumented invariant drift is how scientific pipelines rot.

# Guiding Philosophy

> Predictability scales.
>
> Cleverness does not.

Favor systems that are:

* boring
* explicit
* observable
* restartable

The goal is not elegance.

The goal is running **100k+ spectra without drama.**
