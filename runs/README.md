# Run Root Layout

Use a dedicated run root per experiment, for example:

```text
runs/<run_name>/
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
    validation/
    summary/
```

This layout aligns with `OPERATIONS.md` and supports:

- idempotent retries (shards can be re-run safely),
- immutable outputs (new run/version instead of in-place edits),
- low-friction resume/recovery (task queue + shard outputs in one place),
- cleaner repository root (source code separate from generated artifacts).
