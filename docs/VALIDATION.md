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

#### Addendum, 2026-05-09 — Stage 3 curvature pre-smooth fold-in

The first real-LIDAR render under the Stage 2 pipeline (no-wind
midday today: 2026-05-09 12:00 BST, 5 m, wind 0 m/s) produced a
visible per-cell speckle on the leak raster — a spray of isolated
bright cells uncorrelated with terrain. Diagnosis lives in
`docs/model_correction.md` Stage 3 follow-up note: when Stage 2
folded $\kappa^+$ into $f_{\text{drain}}$ and $q_{\text{storage}}$,
the predecessor formulation's `MODEL.md` §6 ¶282–284
LIDAR-speckle pre-smooth was unintentionally dropped, so single-cell
$\kappa^+$ outliers were saturating
$\mathrm{sat}(\kappa^+/\kappa_{\text{ref}})$ on isolated cells.

Restored as a first-class `run_model` parameter
`curvature_smoothing_sigma_m` (default 10 m). Sweep at
σ ∈ {0, 5, 10, 20} m saved at
`outputs/mallerstang_curvature_sigma_sweep_2026-05-09_1200.png`
(no wind, midday today, 5 m, otherwise production defaults).
Per-panel statistics:

| σ_curv (m) | leak / heating | residual_at_sinks (W/m²·m²) | closure |
|------------|----------------|------------------------------|---------|
| 0          | 99.0 %         | 7.63 × 10⁷                   | 1.000000 |
| 5          | 98.4 %         | 1.15 × 10⁸                   | 1.000000 |
| 10 (default) | 97.7 %       | 1.66 × 10⁸                   | 1.000000 |
| 20         | 96.4 %         | 2.68 × 10⁸                   | 1.000000 |

Energy conservation is exact at every σ (the kernel is
conservation-exact regardless of its inputs). The mild drift in
leak/heating (99 % → 96 %) reflects more energy reaching genuine
sinks (real-terrain summits and domain-boundary outlets) under
smoother routing — physically expected, not a regression.

Visual gate (no-wind midday, this addendum):

* σ = 0: dense per-cell speckle covers most of the tile, obscuring
  the underlying ridge/scarp pattern.
* σ = 5: visible thinning; speckle still present.
* σ = 10 (default): speckle gone; Mallerstang Edge cliff line
  dominates, Wild Boar Fell summit-rim picked out as expected,
  bowl SW of Wild Boar Fell visible. The leak field now correlates
  with the underlying terrain morphology rather than per-cell LIDAR
  noise.
* σ = 20: over-smoothed; some real ridge detail starts to flatten
  alongside the speckle.

Refreshed canonical no-wind midday panels at the new default σ:

* `outputs/mallerstang_leak_5m_nowind_2026-05-09_1200.png`
* `outputs/mallerstang_cycle_5m_nowind_2026-05-09_1200.png`

The canonical SW-summer-afternoon (225° @ 6 m/s, 2026-07-15 13:00
BST) Phase 3.1 visual gate above is **not** re-rendered in this
addendum — that gate is unrelated to the speckle issue (the wind
tilt and lee enhancement are not affected by curvature noise) and
the Stage 2 gate criteria (Mallerstang Edge dominant, Wild Boar
Fell summit interior dim, NE lee enhancement) are independent of
the new pre-smooth knob. A follow-up render under the new default
is on the Phase 4 calibration sweep agenda.

### 2026-05-11 — Phase 3.2 drafting / leak aggregation

Same 15 km × 20 km Wild Boar Fell + Mallerstang mosaic
(`data/processed/mallerstang_wildboar_1m.tif`), 5 m resampled,
re-rendered with the new `draft_aggregation_sigma_m` knob via
`outputs/mallerstang_draft_render.py`. Two passes: σ_draft = 0
(reproduces the Phase 3.1 baseline trigger raster) and σ_draft = 75 m
(the new production default). All other parameters at the Stage 3
defaults (smoothing 10 m, curvature smoothing 10 m, resolve_flats on,
leak shape defaults). Conditions matched to the Phase 3.1 visual gate
exactly:

* **Date/time**: 2026-07-15 13:00 BST.
* **Wind**: from 225° at 6 m/s.

Each pass takes ~37 s on a Macbook (numba JIT leaky kernel + the new
post-kernel Gaussian smooth).

#### Headline numbers

| Metric | σ_draft = 0 (baseline) | σ_draft = 75 m (production) |
|---|---|---|
| `Σ heating` | 7.942 × 10⁹ | 7.942 × 10⁹ |
| `Σ leak / Σ heating` | 97.7 % | 97.7 % |
| `residual_at_sinks_total` | 1.85 × 10⁸ | 1.85 × 10⁸ |
| `Σ draft_potential / Σ heating` | 97.7 % | 92.0 % |
| `draft_mask_loss_total / Σ leak` | 0.00 % | 5.81 % |
| Clusters at q95, min_cells=3 | 9560 | **371** |
| Median cluster τ | 54 s | 102 s |
| `\|leak_σ=0 − leak_σ=75\|.sum()` | 0.000 | 0.000 |

The σ=0 / σ=75 passes produce **bit-exact** `leak` fields, confirming
the aggregation is purely post-kernel and the conservation invariant on
the underlying physical field is preserved. The 5.81 % mask loss is the
energy thrown away by the post-smooth slope mask zeroing bleed onto
flat plateaus — well below the 10 %-ish threshold I'd worry at, so the
aggregation is physically defensible at σ = 75 m on this terrain.

#### Cluster collapse — the headline result

9560 → 371 clusters. The predecessor was rank-normalising cell-level
`leak`, so the q95 threshold caught a few thousand cell-level peaks
(many of them isolated LIDAR-curvature artefacts and single-cell scarp
lips). The aggregated `draft_potential` field merges these into
coherent thermal-scale features — and the diffuse-spur clusters that
the predecessor lost to single-cell ranking now reach the trigger
raster.

Median cluster τ doubles (54 s → 102 s), reflecting the
cyclic-dump-regime spur clusters surfacing alongside the
consistent-trigger-regime scarp clusters that the predecessor saw
exclusively. This is the bimodal physics the Phase 3.1 reformulation
introduced, finally reaching the deliverable layer.

#### Visual gate, item by item

| Phase 3.1 gate criterion | σ_draft = 0 | σ_draft = 75 m |
|---|---|---|
| SW flanks of Wild Boar Fell bright | ✓ (per Phase 3.1 log) | ✓ broader, more coherent |
| Mallerstang Edge cliff line lit | ✓ | ✓ still the dominant linear feature |
| NE lee-side enhancement vs zero-wind | ✓ | ✓ (wind tilt unchanged) |
| Summit plateau interior dim | ✓ | ✓ (slope mask reapplied post-smooth) |
| Cycle-period contrast cliff vs ridge | ✓ | ✓ (cycle period not aggregated) |
| **Diffuse spur shoulders survive q95** | ✗ | **✓ (the Phase 3.2 fix)** |

The Δ-trigger panel (σ=75 − σ=0, RdBu_r) shows the rescue-the-spur
effect spatially: red zones (rescued) over broad spur shoulders and
slope faces, blue zones (demoted) over the isolated cell-level peaks
that the predecessor over-rewarded. The two effects cancel cleanly —
this is a redistribution of trigger emphasis from cells to thermal-
scale features, not a uniform rescale.

#### Output artefacts (gitignored under `outputs/`)

* `mallerstang_draft_compare_5m.png` — 6-panel side-by-side:
  * Row 0: σ=0 trigger raster | σ=75 trigger raster | Δ trigger.
  * Row 1: σ=0 leak (W/m²) | σ=75 `draft_potential` (W/m²) |
    σ=75 cycle period τ (s, log).

#### Caveat

This is a single-area qualitative clearance, matched 1:1 against the
Phase 3.1 baseline. The σ = 75 m default was chosen from physical
reasoning (≈ one thermal column radius at low trigger altitude) and
survives this contact with real LIDAR; a formal sensitivity sweep
across σ ∈ {0, 25, 50, 75, 100, 150} m is the natural follow-up to
confirm the default empirically. Operator-led calibration against
`docs/VALIDATION.md` known-trigger locations is also outstanding —
the cluster count change (9560 → 371) suggests the q95 + min_cells=3
defaults are now closer to "actual thermal columns" than they were
before drafting, which may justify revisiting the threshold.

**Phase 3.2 visual gate cleared on this single-tile render.**
Sensitivity sweep + ground-truth comparison deferred to the Phase 4
calibration agenda.
