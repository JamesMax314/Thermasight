# Validation — known thermal locations

Phase 1 cannot complete until the convergence map agrees with a meaningful
set of these locations on at least three independent test tiles. See
`CLAUDE.md §6 rule 1`.

## Format

Each entry is one row in the table below:

* **Name** — short name for the feature.
* **OS grid** — British National Grid ref (six-figure, ≈ 100 m precision).
* **BNG E / N** — easting / northing in metres (EPSG:27700).
* **Conditions** — wind direction(s) under which this trigger is reliable.
* **Confidence** — `high` (personal flights or multiple XContest tracks),
  `med` (anecdote / single track), `low` (inferred from terrain only).
* **Notes** — anything the model should respect (lee-side, time-of-day,
  early-morning-only, etc.).

## Locations (initial seed)

> Populate this table from pilot logs and XContest tracks before
> attempting the Phase 1 validation gate. The rows below are placeholders
> and **must not** be used as ground truth without verification.

| Name                  | OS grid | BNG E   | BNG N   | Conditions | Confidence | Notes |
|-----------------------|---------|---------|---------|------------|------------|-------|
| Pen-y-ghent east face | SD8373  | 483700  | 473700  | W–SW       | tbd        | Limestone scarp, classic afternoon trigger |
| Ingleborough W face   | SD7474  | 474700  | 474700  | E–SE       | tbd        | Big convex break above Chapel-le-Dale |
| Malham Cove lip       | SD8964  | 489700  | 464700  | S–SW       | tbd        | Strong thermal release off the cliff lip |
| Gordale Scar rim      | SD9163  | 491700  | 463700  | S–SW       | tbd        |  |

When a row's confidence is upgraded to `high`, record the source (flight
date, XContest track URL, etc.) in a comment alongside the row.

## Validation log

### 2026-05-07 — Wild Boar Fell + Mallerstang (informal gate clearance)

The 15 km × 20 km mosaic
(`data/processed/mallerstang_wildboar_1m.tif`, twelve 5 km EA LIDAR
blocks tiling BNG `[370000, 385000] × [490000, 510000]`) was rendered
at 2 m via `python -m thermal_model preview --what convergence` (with
the default Gaussian-smoothed pipeline). The operator (an experienced
paraglider familiar with the area) confirmed by visual inspection
that **most** known thermal triggers across Wild Boar Fell, Hugh
Seat, the Mallerstang scarp, and the surrounding edges appear as
bright lineaments or hot spots in the convergence overlay. A
minority of known triggers do not appear; the likely explanation is
wind drift between heat source and release point, which is a Phase 3
concern (the Phase 1 model has no wind, no sun position, no aspect
weighting).

**Caveat**: this is a single-mosaic, qualitative clearance — *not*
the formal "three independent test tiles" gate originally specified
in `CLAUDE.md §10`. The Phase 1 core ("rising air ≡ flow on inverted
DEM") is taken to be sound enough to begin Phase 2. The formal
three-tile gate (Pen-y-ghent / Ingleborough / Malham) should be
revisited if Phase 2 or Phase 3 results suggest the convergence base
layer is mismodelled.

### 2026-05-09 — Mallerstang re-render under the Phase 3.1 leaky pipeline

The same 15 km × 20 km mosaic re-rendered at 5 m through the
Phase 3.1 leaky-bucket pipeline (`run_model` driving
`leaky_weighted_accumulation`) under the canonical conditions
matched to the Phase 3 baseline:

* **Date/time**: 2026-07-15 13:00 BST (typical SW summer afternoon).
* **Wind**: from 225° at 6 m/s.
* **Smoothing**: 10 m Gaussian.
* **Resolve flats**: on (Garbrecht-Martz between pit-fill and routing).
* **Leak shape**: production defaults — `kappa_ref = 0.005 m⁻¹`,
  `q_ref = 1×10⁶ J/m²`, `slope_min = 2.5°`, `slope_scale = 15°`,
  `f_min = 0.15`, `f_max = 1.0`.

Driven via `outputs/mallerstang_leaky_render.py`; the render
completes in ~37 s on a Macbook (numba JIT backend; reading and
resampling the 1 m source DEM accounts for ~6 s of that).
Side-by-side visual comparison against the Phase 3 baseline lives
at `outputs/mallerstang_phase3_vs_phase31.png`.

**Energy conservation**: `Σ leak + residual_at_sinks ≡ Σ heating`
to float precision. **98.7%** of injected heat consumed as triggers,
**1.3%** escapes at sinks/edges. This is the core Phase 3.1 fix
in numbers — the predecessor pipeline routed all upstream heat to
the summit (the post-hoc `slope_mask` then suppressed its display
but the underlying field was still inflated); the leaky kernel
now consumes heat near where it sources, with only 1.3% reaching
real-terrain summits.

**Visual gate, item by item:**

| Gate criterion | Phase 3 baseline | Phase 3.1 |
|---|---|---|
| SW flanks of Wild Boar Fell bright | ✓ | ✓ |
| Mallerstang Edge cliff line lit | ✓ | ✓ (most prominent feature in `leak`) |
| NE lee-side enhancement vs zero-wind | ✓ (per Phase 3 log) | trusted via kernel mirror-spur test |
| Summit plateau dim | ✗ (heated by upstream catchment) | ✓ (Wild Boar Fell summit interior dark in `leak`) |
| Cycle-period contrast cliff vs ridge | not modelled | ✓ — `cycle_period_s` short (~10²s) on scarp lips, longer (~10³s) on broader ridges, transparent on flats |

The Phase 3.1 trigger raster (rank-normalised `leak`) reproduces the
Phase 3 baseline character — same Mallerstang Edge dominance, same
Wild Boar Fell summit-ring pattern, same SW-bright/NE-enhanced
asymmetry. **Plus** the new pipeline removes the summit-plateau
inheritance that motivated the reformulation: leak is bright on the
**rim** of the Wild Boar Fell summit (where convex breaks live) but
dark in the **interior** (the flat plateau). The cycle-period raster
adds a new dimension of pilot-relevant information that the previous
formulation could not express.

**Output artefacts** (gitignored under `outputs/`):

* `mallerstang_leaky_trigger_5m.png` — rank-normalised T panel.
* `mallerstang_leaky_leak_5m.png` — absolute leak (W/m²).
* `mallerstang_leaky_cycle_5m.png` — cycle period (s, log).
* `mallerstang_leaky_compare_5m.png` — 4-panel overview
  (DEM, T, leak, τ).
* `mallerstang_phase3_vs_phase31.png` — Phase 3 baseline vs
  Phase 3.1 side-by-side.

**Caveat**: this is a single-area qualitative clearance, matched
1:1 against the Phase 3 baseline. Production parameter defaults
were not swept here — the synthetic-fixture defaults from the
Stage 1 spike survived contact with real Mallerstang LIDAR
without obvious failure, but a sensitivity sweep
(`f_min`, `kappa_ref`, `q_ref`, `slope_scale`) and operator-led
tuning against `docs/VALIDATION.md` ground-truth locations is the
natural follow-up under Phase 4. **Stage 2 of Phase 3.1 is closed
on this single-tile gate.** Revisit when Phase 4 (land cover +
time-of-day) starts and the parameter sweep can fold in the
land-cover absorptivity field.
