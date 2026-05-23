# Configuration Layout

This repository keeps all runtime configuration under `configs/`:

- `configs/pipeline/`: one-config pipeline files (`grid` + `turbospectrum` + `outputs`).
- `configs/sampling/`: parameter-grid sampling configs (YAML/JSON).
- `configs/synthesis/`: Turbospectrum synthesis runtime configs.

Current canonical files:

- `configs/pipeline/config_pipeline.example.json`: canonical one-config pipeline template (Latin-Hypercube sampling + synthesis). Copy to `config_pipeline.json` (the working name the scripts and PBS wrappers default to) and edit.
- `configs/pipeline/config_regular_grid.example.json`: regular Cartesian grid template (linear interpolation workflows).
- `configs/pipeline/config_regular_grid_nlte_fe.example.json`: Fe NLTE regular-grid template for flux synthesis with per-abundance ASCII departures.
- `configs/pipeline/config_regular_grid_nlte_fe_intensity.example.json`: Fe NLTE regular-grid template for intensity synthesis with per-abundance ASCII departures and `mu_range` sampling.
- `configs/sampling/config_ml_sampling.json`: sampling-only config.
- `configs/sampling/config_ml_dataset.example.json`: example ML-dataset sampling config with broad stellar-parameter and abundance bounds.
- `configs/synthesis/config_sample_comprehensive.json`: full nested synthesis schema.
- `configs/synthesis/config_sample.json`: compact flat synthesis schema (legacy-compatible).
- `configs/training/wandb_hpc.env`: shared Gadi/HPC defaults for W&B project/entity and local sync paths.

Why this exists:

- Keeps config as a single source of truth (`SYSTEM_RULES.md`).
- Prevents drift between scripts and ad-hoc root-level files.
- Makes run provenance easier to hash and archive.

Recommended usage:

- Start from `configs/pipeline/config_pipeline.example.json` and copy it to `config_pipeline.json` (the default the scripts and PBS wrappers expect).
- Write outputs to `runs/<run_name>/...` (already the default in provided configs).
- In regular-grid and sampling configs, set `run_root` once and keep output paths as `outputs/...` relative to that run root.
