## Purpose

This document enumerates known and expected failure modes for the Turbospectrum grid system and defines the architectural responses required to prevent systemic collapse.

Failure is assumed at scale.

The system must degrade gracefully rather than catastrophically.

# Severity Levels

### 🔴 Critical

Can corrupt the dataset or halt the entire grid.

### 🟠 High

Breaks many shards or wastes large compute allocations.

### 🟡 Medium

Localized disruption; recoverable via retry.

### 🟢 Low

Annoying but not dangerous.

# Filesystem Failures (Most Common HPC Killer)

## 🔴 Metadata Server Saturation

### Cause

Large numbers of tiny files created simultaneously.

Typical triggers:

* writing individual spectra
* verbose logging per wavelength chunk
* temp file explosions

### Symptoms

* IO calls stall
* shards appear frozen
* runtime variance explodes
* cluster-wide slowdown

### Architectural Defense

Prefer:

✅ Zarr

✅ HDF5

✅ batched writes

Never emit one-file-per-spectrum at scale.

**Heuristic:**

If file count grows faster than shard count → danger.

## 🔴 Scratch / Temp Exhaustion

### Cause

* Turbospectrum intermediates
* atmosphere staging
* retry leaks
* here-doc temp files
* unclean crashes

(This is extremely common — you’ve already brushed against it.)

### Symptoms

* `No space left on device`
* shard failures cluster on specific nodes
* retries cascade

### Defense

Require temp root:

<pre class="overflow-visible! px-0!" data-start="2036" data-end="2051"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>$TMPDIR</span><span>
</span></span></code></div></div></pre>

On shard start:

* check free space
* abort early if below threshold

On shard exit:

* delete aggressively

Temp survival across runs is a design bug.

## 🟠 Ghost Files (Partial Writes)

### Cause

Job killed mid-write.

Filesystem leaves behind:

* zero-byte files
* truncated HDF5
* broken Zarr chunks

### Dangerous Because

Your pipeline may interpret them as valid outputs.

Silent scientific corruption is worse than crashes.

### Defense

Never trust existence.

Validate:

* readable format
* expected wavelength count
* minimum size

If validation fails:

→ delete

→ recompute

Automatically.

# Scheduler & Node Failures

## 🔴 Node Death

### Cause

* hardware faults
* kernel panic
* scheduler eviction
* network partition

Not rare at large scale.

### Symptoms

* shard disappears
* no logs
* no output

### Defense

Shards must be:

✅ idempotent

✅ restartable

✅ stateless

Your task queue must assume some shards simply vanish.

Because they will.

## 🟠 Retry Storms

### Cause

Systemic bug triggers mass failure.

Retry logic multiplies load.

Example cascade:

```
bug → 5k shards fail
retry → filesystem overload
more failures → retry again
```

Now you have a self-amplifying outage.

### Defense

Retries must include:

* jitter
* backoff
* retry cap

Never retry instantly.

## 🟠 Scheduler Starvation

### Cause

Oversized jobs block queue throughput.

Example:

Requesting 96 cores for IO-bound work.

Cluster gives you fewer starts → grid crawls.

### Defense

Right-size resources.

Measure:

* CPU utilization
* memory pressure

Do not assume “bigger node = faster.”

Often false.

# Parallelism Failures

## 🔴 Nested Parallelism Collapse

### Cause

Shard launches multiprocessing.

Scheduler already parallelizes.

Result:

CPU thrashing

cache contention

terrible throughput

### Defense

Default invariant:

> One shard = one compute slot.

Break this only with strong profiling evidence.

## 🟠 Memory Fragmentation

Long-running Python + heavy arrays can fragment memory even when totals look safe.

### Symptoms

* sudden OOM after hours
* inconsistent shard runtimes

### Defense

Prefer shorter shard lifetimes over mega-shards.

Fresh processes reset memory state.

# Turbospectrum-Specific Risks

## 🟠 Atmosphere / Linelist Drift

If upstream inputs change mid-grid:

You create a scientifically heterogeneous dataset.

Often unnoticed until ML behaves strangely.

### Defense

Hash and log:

* linelists
* atmospheres
* Turbospectrum binary

Embed hashes into outputs.

Grid identity must be provable.

## 🔴 Silent Numerical Pathologies

Rare stellar parameters may trigger:

* convergence failures
* NaNs
* zero flux regions

### Dangerous Because

The pipeline may not crash.

It just emits garbage.

### Defense

Validate spectra:

Check for:

* NaNs
* negative flux
* flatlines
* wavelength mismatches

Auto-fail shards that violate physics sanity checks.

# Locking Failures

## 🟠 Dead Task Locks

Crash while holding a lock.

Remaining shards stall forever.

### Defense

Prefer atomic task pop patterns.

Avoid long-lived locks.

Design so that **no lock survives process death.**

# Observability Failures

## 🟡 Log Black Holes

If logs are missing or too sparse:

You cannot diagnose cluster-scale behavior.

### Defense

Every shard logs:

* start
* finish
* runtime
* retry count

Minimum viable observability.

# Human-Induced Failures

(Yes — these are common.)

## 🔴 Mid-Run Code Changes

Altering code during a grid run creates:

* heterogeneous outputs
* unreproducible science

### Defense

Freeze execution commit.

Record git SHA in outputs.

No exceptions.

## 🟠 Config Mutation

Someone tweaks parameter ranges mid-flight.

Now your grid has undefined boundaries.

### Defense

Hash configs.

Treat hash mismatch as a different grid.

# Early Warning Signals (Watch These!)

If you see:

### Runtime variance exploding

→ filesystem contention likely.

### CPU low across nodes

→ IO bound.

### Failures clustering

→ systemic bug.

### Retry counts rising

→ investigate immediately.

Do not “wait and see.”

Clusters punish hesitation.

# Design Philosophy

Assume:

* disks fill
* nodes die
* files corrupt
* retries collide
* inputs drift

Your architecture is successful when these events are **non-catastrophic.**

# The Meta-Invariant

> The system must fail in ways that are visible, bounded, and recoverable.

Invisible failure is the only unacceptable kind.
