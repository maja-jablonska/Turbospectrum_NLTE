# Repository Structure Revision

This document summarizes structure changes made to align the repository with:

- `ARCHITECTURE.md`
- `INVARIANTS.md`
- `FAILURE_MODES.md`
- `OPERATIONS.md`
- `DATA_CONTRACT.md`
- `SYSTEM_RULES.md`

## Applied Changes

1. Canonical configuration hierarchy:

```text
configs/
  pipeline/
  sampling/
  synthesis/
```

2. Default artifact outputs moved out of `scripts/` into run roots:

```text
runs/local-dev/outputs/...
```

3. Script defaults updated to use `configs/...` paths and run-root outputs.

4. Legacy `scripts/parameter_grid.csv` compatibility preserved in shell wrappers via fallback logic.

5. `.gitignore` tightened to avoid accidental commits of runtime stores/caches.
6. Redundant JSON templates/configs removed:
   - removed duplicate `configs/pipeline/config_pipeline_template.json`
   - removed unused near-duplicate `configs/synthesis/config.json`

## Optimization and Usability Recommendations

1. Add a `scripts/init_run_root.py` helper:
   - creates `runs/<run_name>/...` tree
   - copies resolved configs into `runs/<run_name>/config/`
   - writes hash/provenance manifest

2. Add a validation gate before merge:
   - run `scripts/validate_dataset.py` automatically
   - fail merge if shard completeness/wavelength/NaN checks fail

3. Standardize logs to split:
   - `logs/pbs/` scheduler-level
   - `logs/shards/` shard-level grep-friendly logs

4. Make scratch usage explicit in all entrypoints:
   - require/validate `TMPDIR` or `--scratch`
   - fail fast on low free space

5. Add a small `Makefile` with stable commands:
   - `make grid`
   - `make synth`
   - `make validate`
   - `make merge`

6. For production runs, enforce immutable output versions:
   - `runs/<grid_name>-v1`, `runs/<grid_name>-v2`
   - never overwrite existing merged datasets in place
