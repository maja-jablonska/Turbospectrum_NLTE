# CAVEATS.md

Known data/physics caveats that are *expected behavior*, not bugs. Check here
before re-investigating a suspicious-looking diagnostic. Each entry says what
you'll observe, why it happens, and what (if anything) to actually worry about.

## MARCS microturbulence: no metal-poor plane-parallel t05 models

**Observed:** In requested-vs-recorded plots, the `vmicro` vs `marcs_turb`
panel shows r ≈ 0 with two horizontal bands (turb = 2 and turb = 5) that both
span the full requested vmicro range. First seen in the HARPS rejected-space
box `E_dwarfs_t05` (feh ∈ [−5, −0.75], logg ∈ [2.75, 5.25], vmicro ∈ [3.5, 5]).

**Why:** The standard-composition MARCS store
(`input_files/model_atmospheres/1D/marcs_standard_comp`) is asymmetric in
turbulence coverage:

| Geometry            | t00/t01/t02        | t05                       |
|---------------------|--------------------|---------------------------|
| plane-parallel (p*) | all metallicities  | **z = +0.00 only** (171)  |
| spherical (s*)      | all metallicities (t02) | all metallicities, logg ≤ 3.5 |

`find_nearest_model` (`scripts/run_turbospectrum.py`) treats vmicro as a weak
axis (`SNAP_SCALES["vmicro"] = 2.0`) precisely so a t-label mismatch never
absorbs multi-dex Teff/logg/[Fe/H] errors. For a metal-poor dwarf (snap
logg ≥ 4) requesting vmicro 3.5–5, a solar t05 atmosphere would cost ~64 in the
feh term versus ~2.25 for a correct-metallicity t02 — so t02 always wins. The
two bands are therefore availability-driven, not request-driven: turb = 5 rows
snapped to spherical models (logg ≤ 3.5), turb = 2 rows to plane-parallel.

**Why it's (mostly) harmless:** babsma applies the *requested* vmicro via
XIFIX regardless of the atmosphere's t-label, so line opacity always sees the
right microturbulence. Only the atmosphere *structure* is second-order off.
`marcs_turb` records the atmosphere file's t-label, not the synthesis vmicro.

**What to actually watch:** boxes that straddle logg ≈ 3.5–4 mix spherical-t05
structures (below) with plane-parallel-t02 structures (above) — a geometry +
t-label discontinuity mid-box that an emulator can pick up as a systematic at
that boundary. To see it, split the vmicro panel by `marcs_logg` or geometry.

## MARCS metallicity grid: 1-dex gaps below [Fe/H] = −3

**Observed:** `feh` vs `marcs_fe_h` bands at −5 and −4 are twice as wide as the
bands at −2.5 and above; snap error reaches ±0.5 dex in the metal-poor tail.

**Why:** The MARCS z grid steps −5, −4, −3, −2.5, −2, … — nothing exists
between −5/−4 and −4/−3. The pipeline `snap_prefilter` defaults used in the
rejected-space configs cap `max_dteff` (125 K) and `max_dlogg` (0.25) but set
`max_dfeh: null`, so these rows are *not* dropped. Set `max_dfeh` in the config
if a run needs a bounded metallicity mismatch instead.
