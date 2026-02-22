# Configuration Layout

This repository keeps all runtime configuration under `configs/`:

- `configs/pipeline/`: one-config pipeline files (`grid` + `turbospectrum` + `outputs`).
- `configs/sampling/`: parameter-grid sampling configs (YAML/JSON).
- `configs/synthesis/`: Turbospectrum synthesis runtime configs.

Current canonical files:

- `configs/pipeline/config_pipeline.example.json`: template.
- `configs/pipeline/config_pipeline.json`: active pipeline config.
- `configs/pipeline/config_regular_grid.example.json`: regular Cartesian grid template (linear interpolation workflows).
- `configs/sampling/config_ml_sampling.json`: sampling-only config.
- `configs/synthesis/config_sample_comprehensive.json`: full nested synthesis schema.
- `configs/synthesis/config_sample.json`: compact flat synthesis schema (legacy-compatible).

Why this exists:

- Keeps config as a single source of truth (`SYSTEM_RULES.md`).
- Prevents drift between scripts and ad-hoc root-level files.
- Makes run provenance easier to hash and archive.

Recommended usage:

- Start from `configs/pipeline/config_pipeline.example.json`.
- Write outputs to `runs/<run_name>/...` (already the default in provided configs).
