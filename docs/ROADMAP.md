# Roadmap

**Current phase: Phase 3 — wind tilt + ground-level triggers.**

Phase 2 (solar + heating) closed 2026-05-07. Phase 3 was reformulated
on the same date — the original "wind drift" framing has been
superseded by the ground-level trigger-prediction model in
`docs/model_correction.md`. Update this header when a phase completes.
Do not skip phases. Do not start the next phase until its
predecessor's gate passes.

---

## Phase 0 — Skeleton

- [x] Repository layout per `CLAUDE.md §3`.
- [x] `environment.yml` with the conda-managed geospatial stack.
- [x] `pyproject.toml` declaring the package + ruff/mypy config.
- [x] `.gitignore` excluding raw and processed LIDAR data.
- [x] DEM read/write with NaN nodata convention.
- [x] CLI entrypoint plumbing (`python -m thermal_model`).
- [x] Test harness with synthetic-DEM fixture and round-trip test.
- [x] CI workflow (lint + tests on a conda-forge environment).
- [x] First real LIDAR fixture under `data/fixtures/`
  (`wild_boar_fell_east_256_1m.tif`, EA Composite 2022 1 m).

**Gate**: `pytest`, `ruff check`, and `mypy thermal_model` all pass in CI.

## Phase 1 — Terrain morphometrics + inverted-DEM convergence

- [x] Slope, aspect, and profile curvature (Horn / Zevenbergen & Thorne).
- [x] Pit-fill on inverted DEM (priority-flood, Barnes et al. 2014).
  Now backed by a richdem C++ implementation in addition to the
  numpy reference; auto-selected via `use_richdem` (mirroring the
  `flow_accumulation` pattern). richdem is ~100x faster on large
  rasters; without it the pure-Python heap stays as the offline
  fallback and as the test reference.
- [x] D∞ flow accumulation via `richdem` with a numpy fallback
  (`thermal_model/physics/flow.py`: `dinf_flow_directions` and
  `flow_accumulation`; Tarboton 1997, eight-facet selection,
  topological pass in descending elevation; weights, NaN nodata,
  auto-selecting backend; reference implementation is the numpy path).
- [x] Diagnostic plots overlaid on hillshade
  (`thermal_model/viz/`: Lambertian `hillshade`, generic `plot_overlay`,
  and `plot_convergence` / `plot_slope` / `plot_aspect` /
  `plot_profile_curvature` for sanity checks. Phase 2 will introduce a
  separate, physical hillshade in `solar/` for cast-shadow analysis).
  Drivable from the CLI via `python -m thermal_model preview` (pulled
  forward from Phase 3 since the diagnostic plots are not useful
  without a way to run them on a tile).
- [x] Property tests: rotation/scaling invariants. Hypothesis-based
  tests in `test_terrain_morphometry.py` and `test_physics_flow.py`
  pin: slope/aspect/curvature equivariance under `np.rot90`,
  `tan(slope) * k = tan(slope_at_cell*k_scale)` cell-size scaling,
  D∞ direction angle gains `k*pi/2` per CCW turn (math convention)
  with slopes invariant, D∞ slope magnitudes scale `1/k`, and the
  D∞ flow-accumulation field is exactly equivariant under 90° rotation
  and exactly invariant under cell-size scaling (the eight facets
  are rotationally symmetric and accumulation in default cell-count
  mode is dimensionless).
- [x] DEM mosaic pipeline (`thermal_model/io/mosaic.py`: `mosaic_dems()`
  wrapping `rasterio.merge.merge` with project on-disk conventions:
  `nodata=-9999`, deflate-compressed tiled float32. Validates CRS and
  cell-size consistency across inputs before merging. Drivable from
  `python -m thermal_model mosaic`. Pulled in from CLAUDE.md §9 to
  assemble whole-hill tiles from EA LIDAR 5 km blocks for the Phase 1
  validation gate; the 256 m fixture used for I/O testing is too small
  to validate convergence against any single thermal location).
- [x] Streak-artefact mitigation on the convergence map. Pure
  priority-flood pit-fill leaves filled flat regions with a BFS
  chamfer-distance gradient; D∞ accumulation on that gradient produces
  long parallel streaks perpendicular to ridges that abut flat
  plateaus. Two mitigations available:
  * `thermal_model.physics.resolve_flats` — Garbrecht & Martz (1997)
    flat-direction resolution via richdem (`rd.ResolveFlats`); has a
    stochastic numpy fallback. Principled, but slow on large rasters
    (~7 min on 75M cells).
  * `plot_convergence(smooth_sigma_m=...)` — Gaussian blur of the
    inverted DEM before pit-fill, kernel sigma in metres
    (default 10 m). Softens the ridge/flat boundary so the BFS
    frontier doesn't enter the flat along a sharp line. Cheap
    (~1 s on 75 M cells), good enough for diagnostic plots; replaced
    ResolveFlats as the `plot_convergence` default after benchmarking.

**Gate**: convergence raster agrees with `docs/VALIDATION.md` on three
independent test tiles. Document the comparison in `docs/VALIDATION.md`.

**Gate status (2026-05-07): cleared informally.** Operator visually
confirmed agreement with most known thermal triggers across the
Wild Boar Fell + Mallerstang 15 km × 20 km mosaic; see
`docs/VALIDATION.md` § Validation log. This is a single-tile
qualitative pass rather than the originally-specified three-tile
formal gate. Revisit the gate if Phase 2 or Phase 3 results suggest
the Phase 1 convergence layer is wrong.

## Phase 2 — Solar + heating

- [x] Sun position + clear-sky irradiance via `pvlib`
  (`thermal_model/solar/`: `solar_position` returns a frozen
  `SolarPosition(azimuth_rad, altitude_rad)` in compass-bearing
  convention matching `terrain.aspect`; `clear_sky_irradiance` uses
  Ineichen-Perez via `pvlib.location.Location.get_clearsky` with a
  pinned default Linke turbidity of 3.0 to keep the call offline;
  `slope_irradiance` projects DNI onto each cell via the
  cos(angle-of-incidence) formula and adds an isotropic Liu-Jordan
  diffuse term, returning beam and diffuse separately so the
  cast-shadow mask can attenuate beam alone in the next step.
  Anisotropic diffuse and ground-reflected components are deferred).
- [x] Hillshade and cast-shadow mask via horizon scan
  (`thermal_model/solar/shadow.py`: `cast_shadow_mask` returns a
  float64 mask of {0, 1, NaN} from a vectorised horizon scan along
  the solar azimuth. Steps one cell along the dominant grid axis
  with a fractional offset on the other, sampling each step's
  terrain via `scipy.ndimage.map_coordinates` (bilinear) and
  comparing against the sun ray's height. Stops once the cumulative
  rise exceeds the DEM's relief — typically a few hundred steps for
  UK terrain. Snaps near-zero sun-direction components to exact
  zero to keep cardinal-direction sun on-grid. Below-horizon sun
  yields all-zero, near-zenith sun yields all-one. Multiplies the
  beam component of `slope_irradiance` only; diffuse is independent
  of cast shadows. Distinct from the cosmetic Lambertian
  `viz.hillshade` used for diagnostic plots).
- [x] Heating field $H = I \cdot \alpha \cdot s$
  (`thermal_model/physics/heating.py`: `heating_field` assembles
  the W/m² ground heating from the slope-projected irradiance, the
  cast-shadow mask, and a shortwave absorptivity $\alpha$. Cast
  shadow attenuates the *beam* component only — diffuse comes from
  the whole sky and is not blocked by a single upwind ridge — so
  the practical formula is
  $H = \alpha \cdot (s \cdot I_{\mathrm{beam}} + I_{\mathrm{diffuse}})$.
  $\alpha$ accepts either a scalar (Phase 2 default) or a per-cell
  array (Phase 4 land cover). The default `DEFAULT_ABSORPTIVITY =
  0.80` is the dry grass / heather upland Dales surface from
  `docs/DATA.md`. Soft (fractional) shadow masks in $[0, 1]$ are
  accepted to keep the door open for future smooth-occluder
  models. NaN propagates from any input. A diagnostic plotter
  `viz.plot_heating` and a `preview --what heating` CLI subcommand
  drive the full Phase 2 pipeline from a single DEM and ISO
  `--datetime`; lat/lon default to the DEM centre via reprojection
  from its CRS, and elevation defaults to the median of finite
  cells).
- [x] Coupling $P = \sqrt{H \cdot C}$ with $(p, q)$ exposed
  (`thermal_model/physics/coupling.py`: `thermal_potential`
  computes $P = H^p \cdot C^q$. Default $(p, q) = (0.5, 0.5)$).
  **Superseded 2026-05-07** — heating now enters the pipeline as
  the `weights` parameter of D∞ flow accumulation (see Phase 3),
  not as a separate post-hoc multiplier. `thermal_potential` is
  retained in the codebase for backward compatibility but is
  **no longer wired into the production trigger pipeline**. Phase 4
  time-of-day weighting will scale the heating weights themselves
  rather than sweep $(p, q)$ on a separate coupling step. See
  `docs/MODEL.md` §10.1 for the historical record.

## Phase 3 — Wind tilt + ground-level triggers

**Reformulated 2026-05-07** in two passes:

1. The original "wind drift" framing was solving the wrong problem
   (it predicted where airborne thermals end up, not where they
   source from the ground). Wind now enters as a terrain tilt
   before inversion.
2. The interim "geometric mean of normalised heating × convergence"
   step was also dropped. Heating is now the **per-cell weight on
   the D∞ flow accumulation**, so the routing intrinsically
   integrates "where the air is warm" with "where it converges" in
   one pass. There is no separate coupling step.

See `docs/model_correction.md` for the full justification. The model
predicts ground-level trigger locations. There is no in-air drift
step in the main pipeline.

- [ ] `physics/wind_tilt.py` — `wind_tilt_ramp(dem, cell_size_m,
  wind_from_deg, wind_speed_ms, k)` returning the tilted DEM. Pure
  numpy; cell-size aware; documents the sign convention against
  cardinal-wind cases (N→S, S→N, W→E, E→W, SW→NE).
- [ ] Confirm `physics.flow_accumulation` accepts a `weights`
  raster on both the `richdem` and numpy fallback paths (Phase 1
  task list claims it does — verify on the installed `richdem`
  version before wiring the pipeline). If `richdem`'s
  `FlowAccumulation` exposes `weights=`, use it directly; if not,
  obtain D∞ proportions via `rd.FlowProportions` and run the
  weighted topological pass through the existing numpy reference
  path.
- [ ] `physics/pipeline.py` — `run_model(...)` orchestrating the
  §6-of-`model_correction.md` block: smooth → wind tilt → heating
  field from raw DEM → invert + pit-fill + D∞ accumulation
  **weighted by heating** → multiply by normalised
  max(profile_curv, 0) → multiply by min-slope mask (~2.5°).
  Returns the trigger-potential raster and intermediate
  diagnostics. **No separate "energy" raster** — that step has
  been merged into the routing.
- [ ] Trigger-point clustering on the trigger-potential raster.
  Connected components (`scipy.ndimage.label`) on a high-percentile
  mask, ranked by mean strength, with a min-cluster-cells filter.
  (`scikit-learn` DBSCAN is *not* added — connected components is
  the equivalent operation on a regular raster and avoids the dep
  per `CLAUDE.md` §4.)
- [ ] GeoTIFF + KMZ export of trigger points (`simplekml`,
  reprojection to WGS84 from the DEM CRS).
- [x] CLI subcommand: `preview` (pulled forward to Phase 1 alongside
  the diagnostic plots).
- [ ] CLI subcommand: `run` — wires the full pipeline. Args:
  `--dem`, `--datetime`, `--wind-from`, `--wind-speed`,
  `--wind-tilt-k` (default 0.03), `--out` (trigger GeoTIFF),
  `--kmz` (trigger points). The deprecated `--release-height` and
  `--climb-rate` are *not* added.
- [ ] Diagnostic plotter `viz.plot_trigger_potential` and a
  `preview --what trigger` CLI hook. The diagnostic should also
  expose `--what weighted-convergence` so the heating-weighted
  flow accumulation can be inspected on its own.

### Quarantined / removed

The previous `physics/drift.py` and the `drift_field()` /
`drift_distance_m` API are removed from the main pipeline. If
post-detachment in-air drift is ever needed (e.g. for XC track
correlation) it lives in a separately-named utility module under
`thermal_model/utils/` with a docstring stating it is *not* part of
the trigger-prediction pipeline. Operator approval required before
reintroducing.

### Validation

**Gate: heating-weighted convergence must distinguish geometrically
equivalent spurs by aspect.** Construct a synthetic test case with
two mirror-image spurs of identical geometry, one S-facing and one
N-facing, under a noon midsummer sun. Compute the trigger raster
for both. The S-facing spur must score higher than the N-facing
spur — this is exactly the case the previous post-hoc multiplier
got right by accident (local $H = 0$ on the shadowed face zeros
the cell) but the new formulation gets right by physics: shadowed
upstream cells inject zero W/m² into the routing, so the shadowed
spur receives no upstream thermal energy. A second-order check:
if the shadowed spur's catchment is artificially relit (e.g. by
removing the cast-shadow mask), its score should rise toward the
sunlit spur — verifying that the routing actually transports the
upstream warmth to the convergent point rather than just multiplying
by it locally.

After implementation, the trigger raster on the Wild Boar Fell +
Mallerstang mosaic for a typical SW summer afternoon (5–8 m/s from
210–240°, 1200–1400 BST) should show:

* SW-facing lower flanks of Wild Boar Fell bright (sun + convergence
  + convex spurs).
* Lee-side (NE) enhancement of the main E-facing scarp relative to a
  zero-wind baseline (the tilt's principal observable).
* Mallerstang Edge cliff line lit by curvature × moderate
  weighted-convergence even where plan convergence is laminar.
* Flat summit plateau dark (slope mask + low weighted-convergence
  after smoothing).
* Valley floors suppressed (slope mask).
* **Shadowed convergent points downstream of sunny faces should
  appear**, not be zeroed out. If they vanish entirely (cf. the old
  post-hoc multiplier), the heating raster is being applied as a
  local mask rather than as a flow weight — re-check the wiring of
  `weights=heating` into `flow_accumulation`.

If the NE side of Wild Boar Fell does not enhance under SW wind
relative to the zero-wind baseline, the tilt has the wrong sign;
re-check the ramp formula (`docs/model_correction.md` §4).

## Phase 4 — Land cover + time-of-day

- [ ] UKCEH land cover ingestion + absorptivity table.
- [ ] Time-of-day weighting between heating and convergence.
- [ ] CLI flags for land cover and time-window sweeps.

## Phase 5 — Real physics

- [ ] WindNinja-driven terrain-aware wind field, replacing the
  empirical wind-tilt coefficient `k`. The wind field itself encodes
  the boundary-layer flow distortion that the linear ramp
  approximates.
- [ ] Lagrangian plume model running alongside the (tilted) inverted-
  DEM hydrological analogy, for cross-validation rather than
  replacement.
- [ ] Comparison of the two convergence maps and the trigger raster
  against `docs/VALIDATION.md` as the joint validation step.
