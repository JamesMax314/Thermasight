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
