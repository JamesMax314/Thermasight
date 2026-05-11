# Roadmap

**Current phase: Phase 4 — land cover + time-of-day.** UKCEH land
cover wiring landed 2026-05-10 (see Phase 4 § Land cover, below).
Mallerstang re-render against the WMS-fetched LCM is the validation
follow-up; production 21-class α LUT is operator-authored TBD.

Phase 3.2 (drafting / aggregation, 2026-05-11) landed in parallel —
a post-kernel Gaussian smooth of the `leak` field with the slope
mask reapplied, exposed on `RunResult.draft_potential` and driving
`trigger_potential` and clustering. Default σ = 75 m. Rescues
diffuse spur clusters that the cell-level rank-of-leak threshold
would lose; see Phase 3.2 below.

Phase 2 (solar + heating) closed 2026-05-07. Phase 3 was reformulated
on the same date — the original "wind drift" framing was superseded
by the ground-level trigger-prediction model in
`docs/model_correction.md` — and closed 2026-05-08 with the
mirror-spur pytest gate plus operator visual confirmation on the
Wild Boar Fell + Mallerstang mosaic (see Phase 3 § Validation log).

Phase 3.1 opened 2026-05-09 to address two physical defects of the
Phase 3 pipeline (energy double-counting along the flow path; no
mechanism for the cyclic-dump regime on gentle terrain). Stage 1 of
3.1 (a standalone leaky-bucket kernel + synthetic-fixture tests)
landed 2026-05-09. Stage 2 (production fold-in: `run_model`,
`RunResult`, `triggers/cluster.py`, `viz/`, CLI, docs, plus the
Mallerstang re-render) closed 2026-05-09. Production `run_model`
now drives the leaky kernel; energy conservation is pinned at the
pipeline level (98.7 % of injected heat consumed as triggers on
the Mallerstang mosaic, 1.3 % residual at sinks). See
`docs/VALIDATION.md` § 2026-05-09 for the Mallerstang clearance.
Stage 3 (curvature pre-smooth fold-in, 2026-05-09 follow-up)
restores the predecessor's `MODEL.md` §6 ¶282–284 LIDAR-speckle
suppressor as the new `curvature_smoothing_sigma_m` parameter on
`run_model` (default 10 m), fixing per-cell speckle visible on the
no-wind midday Mallerstang leak panel.

Update this header when a phase or stage completes. Do not skip
phases. Do not start the next phase until its predecessor's gate
passes.

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

- [x] `physics/wind_tilt.py` — `wind_tilt_ramp(dem, cell_size_m,
  wind_from_deg, wind_speed_ms, k)` returning the tilted DEM. Pure
  numpy; cell-size aware; documents the sign convention against
  cardinal-wind cases (N→S, S→N, W→E, E→W, SW→NE). Adds
  `delta = k·|u|·(col_m·sinθ − row_m·cosθ)` for `θ = wind-to`
  bearing, preserves dem dtype, propagates NaN. Tests in
  `test_physics_wind_tilt.py` pin the cardinal sign convention,
  linearity in $k$ and $|u|$, reversibility under +180°, NaN/dtype
  preservation, and a hypothesis property that the per-metre slope
  along the wind-to direction is exactly $k|u|$ regardless of
  cell size or grid shape.
- [x] Confirm `physics.flow_accumulation` accepts a `weights`
  raster on both the `richdem` and numpy fallback paths.
  Verified on the installed `richdem`: `FlowAccumulation` exposes
  `weights=`, and a unit-vs-3× weights probe shows the kwarg is
  honoured exactly (`3·w → 3·acc`), so the existing direct-call
  wiring in `_flow_accumulation_richdem` is correct and the
  `FlowProportions` hybrid is not needed. Pinned a public weights
  contract in `physics.flow._validate_weights`: `weights.shape ==
  dem.shape`, finite (no NaN, no Inf) at every finite-`dem` cell,
  NaN allowed only at NaN-`dem` cells. The contract is checked in
  the public `flow_accumulation` entrypoint before backend
  dispatch, replacing the previous silent NaN-masking on the
  richdem branch. Tests in `test_physics_flow.py` pin the
  contract on both backends and add cross-backend agreement under
  random weights (within the same 20 % broad-strokes tolerance as
  the existing unweighted smoke test).
- [x] `physics/pipeline.py` — `run_model(...)` orchestrating the
  §6-of-`model_correction.md` block: smooth → wind tilt → heating
  field from raw DEM → invert + pit-fill (`epsilon=1e-3`) →
  Garbrecht-Martz flat resolution (`resolve_flats=True` by default,
  toggleable via `--no-resolve-flats`) → D∞ accumulation **weighted
  by heating** → rank-normalise convergence and positive curvature
  → multiply by min-slope mask (~2.5°). Returns a frozen
  `RunResult` carrying the trigger-potential raster plus all
  intermediates (smoothed DEM, tilted DEM, heating, weighted
  convergence, profile curvature, slope, slope mask). No separate
  "energy" raster — heating enters as the per-cell weight on the
  flow accumulation, so the integration is intrinsic. Edge cells
  where the 3×3 stencil cannot resolve heating are substituted
  with `0.0` weight (a finite-DEM cell with no informed estimate
  contributes nothing to the routing) so the `flow_accumulation`
  weights contract is satisfied. **Normalisation note**: an early
  draft used q99 clipping for both factors, but on the Mallerstang
  mosaic that collapsed the trigger raster to near-zero everywhere
  (each factor reaches 1 only on its top 1 %, and the product
  vanishes off that intersection). Rank normalisation
  (`scipy.stats.rankdata / N`) replaces it: each factor spreads
  uniformly over `[0, 1]`, the product spans the unit interval,
  and the result is robust to LIDAR-speckle outliers in curvature.
  **Streak-artefact note**: pit-fill on the inverted, tilted DEM
  leaves formerly-flat regions (raised plateaus / summit tops)
  with a BFS chamfer-distance gradient that D∞ rasters into
  parallel streaks; `resolve_flats` between fill and accumulate
  replaces that with a Garbrecht-Martz two-component gradient and
  is on by default.
- [x] Trigger-point clustering on the trigger-potential raster.
  `thermal_model.triggers.cluster_triggers` runs `scipy.ndimage.label`
  (8-connectivity by default) on a high-percentile mask of the
  strictly-positive trigger field, drops components below
  `min_cluster_cells` (default 3), and returns a list of
  `TriggerPoint(row, col, mean_strength, n_cells)` ranked by mean
  strength. (`scikit-learn` DBSCAN is *not* added — connected
  components is the equivalent operation on a regular raster and
  avoids the dep per `CLAUDE.md` §4.)
- [x] GeoTIFF + KMZ export of trigger points. GeoTIFF reuses
  `io.write_raster_like`. KMZ via `thermal_model.triggers.write_kmz`:
  raster centroid → projected (x, y) via the DEM's affine transform
  → WGS84 (lon, lat) via `pyproj.Transformer` → `simplekml`. Each
  cluster is a placemark named by its rank with the mean strength
  and cell count in the description.
- [x] CLI subcommand: `preview` (pulled forward to Phase 1 alongside
  the diagnostic plots).
- [x] CLI subcommand: `run` — wires the full pipeline. Args:
  `--dem`, `--datetime`, `--wind-from`, `--wind-speed`,
  `--wind-tilt-k` (default 0.03), `--out` (trigger GeoTIFF),
  `--kmz` (optional trigger-point KMZ), plus
  `--smoothing-sigma`, `--min-slope`, `--absorptivity`,
  `--linke-turbidity`, `--lat`, `--lon`, `--elevation`,
  `--cluster-quantile`, `--min-cluster-cells`. The deprecated
  `--release-height` and `--climb-rate` are *not* added.
- [x] Diagnostic plotters `viz.plot_trigger_potential` and
  `viz.plot_weighted_convergence`, plus `preview --what trigger`
  / `--what weighted-convergence` CLI hooks. The wind-requiring
  previews share the lat/lon/elevation/datetime resolution helper
  with the heating preview and add `--wind-from`, `--wind-speed`,
  `--wind-tilt-k`, `--smoothing-sigma`, `--min-slope` flags.

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

### Validation log

#### 2026-05-08 — Phase 3 informal gate clearance

* **Mirror-spur pytest gate cleared.** Two synthetic spurs (S- and
  N-facing, geometrically identical) at noon midsummer: S-facing
  trigger > N-facing, and removing the cast shadow lifts the
  N-facing toward the S-facing — confirming the routing transports
  upstream warmth (`tests/test_physics_pipeline.py`,
  `test_mirror_spur_south_outscores_north_at_noon_midsummer` and
  `test_mirror_spur_relit_north_rises_toward_south`).
* **Visual gate (Wild Boar Fell + Mallerstang) cleared informally.**
  Trigger preview at 5 m on the 15 km × 20 km mosaic for a typical
  SW summer afternoon (225° @ 6 m/s, 13:00 BST mid-July) shows
  ridges, scarps, and spur shoulders lit coherently across the
  tile, with Mallerstang Edge picked out by curvature, the bowl SW
  of Wild Boar Fell summit highlighted, and a visible NE-ward
  shift relative to the zero-wind baseline (the lee-side bias).
  Hash artefacts visible in the first cut (q99×q99 normalisation
  with no flat resolution) were resolved by switching to rank
  normalisation and adding `physics.resolve_flats` between
  `fill_pits` and `flow_accumulation` in `run_model`. Outputs
  archived under `outputs/mallerstang_trigger_5m_v2.png` and
  `outputs/mallerstang_model_surface_5m.png`.

This is a single-area qualitative pass, not a multi-tile formal
gate; revisit if Phase 4 results suggest the trigger raster is
mismodelled.

## Phase 3.1 — Leaky-bucket reformulation

**Opened 2026-05-09.** The Phase 3 pipeline routes heating as a
weight on D∞ flow accumulation, then multiplies the result by
positive curvature and a slope mask. Two physical defects motivate
this reformulation:

1. **Energy double-counting along the flow path.** Weighted D∞
   accumulation is monotonic toward the global sink (real-terrain
   summit on the inverted DEM). A convex break midway up the hill
   registers high trigger potential, *and* every cell upstream of
   it sees the same energy in its weighted-convergence value, *and*
   the summit ultimately receives the catchment total. The post-hoc
   `κ̂⁺ × slope_mask` multiply suppresses the *display* of the
   summit but does nothing about the inflated convergence values at
   intermediate breaks; the same parcel of energy is counted at
   every cell along its path.

2. **No mechanism for cyclic mass release on gentle terrain.**
   Pilots observe that gentle slopes "fill up then dump" — the
   boundary layer accumulates buoyancy past a capacity threshold
   and releases as one large thermal, then quiet. The Phase 3
   model has no notion of capacity or cycle time; gentle-terrain
   triggers are entirely suppressed by the slope mask rather than
   being modelled as long-period dumps. This loses pilot-relevant
   information: a hill that cycles every 30 minutes with big
   releases is a real but different kind of thermal source from a
   scarp that cycles every minute with small consistent ones.

The reformulation replaces the post-hoc multiply with a
**leaky-bucket weighted accumulation**. Each cell consumes a
curvature/slope-dependent fraction `(1 − f_drain)` of its
through-flow as trigger output and forwards only `f_drain` onward;
a per-cell storage capacity `Q` produces a cycle period
`τ = Q / leak`. Energy is conserved along the path (no
double-counting). The full physics derivation lives in
`docs/MODEL.md` §11; the design conversation is preserved in
`~/.claude/plans/please-read-docs-for-whimsical-scone.md`.

The work is **staged**:

* **Stage 1 (this section)** — standalone kernel + synthetic-fixture
  validation, no production-pipeline changes. Lets the algorithm be
  tested end-to-end before any Phase 3 code is touched.
* **Stage 2 (gated on Stage 1 + Mallerstang visual review)** —
  fold the kernel into `run_model`, restructure `RunResult`, add
  `cycle_period` as a first-class output, JIT the topological pass
  with `numba`, update CLI / viz / `docs/MODEL.md` § 5–§ 7 / docs/
  `model_correction.md`. After Stage 2, `Phase 3` is closed and
  Phase 4 (land cover + time-of-day) opens.

### Stage 1 — spike (closed 2026-05-09)

- [x] `physics/leaky_accum.py` — the kernel `leaky_weighted_accumulation`
  plus `f_drain_field` and `q_storage_field` shape helpers and the
  `LeakyResult` frozen dataclass. Pure numpy, mirrors
  `physics.flow._flow_accumulation_numpy`'s topological
  descending-elevation pass; reuses `_FACETS`, `_facet_slopes`,
  `_validate_dem`, `_validate_weights` from `flow.py` so the
  unit-`f_drain` limit reduces exactly to `flow_accumulation`. The
  saturating shape function uses `1 − exp(−x)` so the leak
  asymptotes near `f_min` for realistic Dales-scale curvature
  values, not only at impossible extremes.
- [x] `tests/test_physics_leaky_accum.py` — 21 tests covering:
  * **Energy conservation** under uniform and random weights:
    `nansum(leak) + residual ≡ nansum(weights)` to machine precision.
    The strongest invariant; catches almost any wiring error in the
    topological pass.
  * **Two limit cases** that bridge the new kernel to the existing
    `flow_accumulation`: `f_drain ≡ 1` ⇒ `forward` matches
    `flow_accumulation` cell-for-cell; `f_drain ≡ 0` ⇒ `leak`
    equals the input weights with nothing forwarded.
  * **Cycle-period dimensionality**: `τ = Q / leak` exactly where
    `leak > 0`, `+inf` where `leak == 0`.
  * **Mirror-spur Phase 3 gate ported**: S-spur outscores N-spur
    on `leak` at noon midsummer; relighting the cast shadow
    narrows the gap, confirming the leaky kernel transports
    upstream warmth via the routing rather than just multiplying
    locally.
  * **Synthetic gentle-ridge** ⇒ leak peaks in the transition band
    where the ramp meets the flat top, with a long cycle period
    (~10⁴ s) — the cyclic-dump regime.
  * **Synthetic sharp-break** ⇒ leak peaks at the cliff lip with
    a short cycle period (~10² s) — the consistent-trigger regime.
  * **Weights / `f_drain` / `q_storage` contracts** mirroring the
    `flow_accumulation` shape and finiteness checks.
  * **NaN propagation** through the output rasters with a finite
    `residual_at_sinks_total` scalar.
- [x] `physics/__init__.py` — re-exports `leaky_weighted_accumulation`,
  `LeakyResult`, `f_drain_field`, `q_storage_field`,
  `F_MIN_DEFAULT`, `F_MAX_DEFAULT`. The kernel is **not** imported
  by `pipeline.run_model`; production behaviour is unchanged.
- [x] `environment.yml` — adds `numba` for the Stage 2 JIT pass.
  The Stage 1 kernel is pure numpy; numba is a pre-emptive
  dependency so the Stage 2 fold-in does not also need to update
  the conda env.
- [x] **Visual sanity check** on the Wild Boar Fell east 256×256
  fixture: rendered as four panels (current trigger; leaky leak
  rank-normalised; log-cycle-period clipped to [60 s, 1 hr]; diff)
  in `outputs/leaky_spike_compare.png`. Closure error
  `nansum(leak) + residual − nansum(weights) = 0` to float
  precision. Leak field tracks the same major scarp / ridge
  features as the current trigger raster; cycle-period raster
  shows short cycles concentrated on sharp features (the
  pilot-relevant signature).
- [x] **Synthetic-fixture visualisations** in
  `outputs/leaky_spike_{mirror_spur,gentle_ridge,sharp_break}.png`
  illustrating the three test fixtures. The contrast between the
  gentle ridge (4 % heating consumed as triggers, ~10⁴ s cycle —
  rare big dump) and the sharp break (48 % heating consumed,
  ~10² s cycle — reliable consistent thermals) reproduces the
  bimodal physics that motivated the reformulation.

**Stage 1 gate**: 21 new tests pass; `ruff check`, `ruff format
--check`, `mypy thermal_model`, `pytest` all green; visual sanity
check matches expectations; production code untouched.
**Cleared 2026-05-09** (commit `a8ad771` on
`feat/leaky-accum-spike`).

### Stage 2 — production fold-in (closed 2026-05-09)

Branch: `feat/phase3.1-leaky-pipeline-fold-in`. All code-side and
validation items complete; Mallerstang re-render reproduced the
Phase 3 visual gate plus the new summit-plateau dimming and
cycle-period contrast — see `docs/VALIDATION.md` § 2026-05-09.

- [x] `physics/pipeline.py:run_model` — replaced the
  `flow_accumulation(weights=heating)` + post-hoc `rank_norm(wc) ×
  rank_norm(κ⁺) × slope_mask` step with
  `leaky_weighted_accumulation(...)`. Curvature and slope feed
  `f_drain_field` and `q_storage_field` (computed from the **raw**
  DEM, matching the existing convention). Production trigger raster
  is `rank_normalise(leak)` for backward-compatible display.
- [x] `physics/pipeline.py:RunResult` — gained `leak` (W/m²),
  `forward` (W/m², diagnostic), `cycle_period_s` (s),
  `residual_at_sinks_total` (scalar). Dropped `slope_mask` as a
  public field. `weighted_convergence` is now `leak + forward`
  (the pre-leak through-flow), preserved for backward compatibility
  with the existing viz / KMZ consumers.
- [x] `physics/leaky_accum.py` — `_leaky_pass_numba` JIT-compiled
  topological sweep auto-selected when `numba` is importable; the
  pure-numpy reference path stays as the test oracle. Cross-backend
  agreement pinned by `test_leaky_accum_numba_and_numpy_agree` on
  random fixtures (1e-12 rtol). Bench: 4× speedup on a 1024×1024
  raster (numpy 2.07 s → numba 0.50 s); for Mallerstang's 75 M
  cells this brings a single run from minutes to ~30 s.
- [x] `cli.py` — `run` gained `--f-min`, `--f-max`, `--kappa-ref`,
  `--q-ref`, `--slope-scale`, `--leak-out` (optional GeoTIFF for
  absolute leak), `--cycle-period-out` (optional GeoTIFF for
  cycle period; +inf cells written as nodata). `preview --what`
  gained `leak` and `cycle-period`.
- [x] `viz/` — new `plot_leak` and `plot_cycle_period` plotters
  exported from `viz/__init__.py`. `plot_cycle_period` uses
  `plasma_r` so light = short cycle (consistent thermals) and dark
  = long cycle (cyclic dumps); non-leaking cells render
  transparent against the hillshade backdrop.
- [x] `triggers/cluster.py` — `cluster_triggers` gained an optional
  `cycle_period_s` parameter. `TriggerPoint` gained
  `mean_cycle_period_s` (None when no cycle raster supplied).
  CLI defaults to clustering on `RunResult.leak` (absolute units)
  with `cycle_period_s=result.cycle_period_s`. KMZ description
  includes the cycle period in the most pilot-readable unit
  (s / min / hr).
- [x] `docs/MODEL.md` — added a "superseded by §11" banner on §5;
  §11 now flagged as the canonical production formulation.
  §5–§7 retained as the historical predecessor record.
- [x] `docs/model_correction.md` — top-of-doc banner updated to
  reflect Stage 2 landing; §3 + §5 of that document are now the
  predecessor formulation.
- [x] `tests/test_physics_pipeline.py` — mirror-spur tests assert
  against `RunResult.leak` (mirror-spur S>N still holds; relighting
  narrows the gap). New `test_run_model_energy_conservation` pins
  `nansum(leak) + residual ≡ nansum(heating)` at the pipeline
  level. New `test_run_model_cycle_period_finite_at_triggers` pins
  the τ dimensional contract.
- [x] **Mallerstang re-render** at 5 m under the canonical
  validation conditions (225° @ 6 m/s, 13:00 BST mid-July). Visual
  gate cleared: SW flanks bright, Mallerstang Edge dominant, Wild
  Boar Fell summit-plateau interior dim (the Phase 3 artefact is
  gone), cycle-period contrast visible (short on cliff lines, long
  on rounded ridges). Energy-conservation closure: 98.7 % leak,
  1.3 % residual, exact to float precision. Full write-up in
  `docs/VALIDATION.md` § 2026-05-09. A formal parameter sweep
  (`f_min`, `kappa_ref`, `q_ref`, `slope_scale`) was not
  performed — the synthetic-fixture defaults survived contact
  with real LIDAR; sensitivity-tuning is deferred to Phase 4
  alongside the land-cover absorptivity work.

**Stage 2 gate (cleared 2026-05-09)**: full test suite green
(289 passed at landing, including 21 + 1 leaky-kernel tests, 7
pipeline tests, 10 trigger tests, 23 CLI tests, 29 viz tests);
`run_model` outputs the new fields; Mallerstang trigger raster on
a typical SW summer afternoon (225° @ 6 m/s, 13:00 BST mid-July)
reproduces the Phase 3 visual gate **plus** dimming of the
summit-plateau artefacts that motivated the reformulation, with
plausible cycle-period contrast between the cliff lines (short)
and the rounded ridges (long).

### Stage 3 — curvature pre-smooth fold-in (2026-05-09 follow-up)

Operator inspection of the no-wind midday Mallerstang leak / cycle
panels (`outputs/mallerstang_leak_5m_nowind_2026-05-09_1200.png`,
`mallerstang_cycle_5m_nowind_2026-05-09_1200.png`) showed a
per-cell speckle uncorrelated with terrain features — a spray of
isolated bright cells. Diagnosis: when Stage 2 folded $\kappa^+$
into $f_{\text{drain}}$ / $q_{\text{storage}}$, the predecessor
formulation's `MODEL.md` §6 ¶282–284 prescription
("a Gaussian pre-smooth at one DEM cell suppresses LIDAR speckle
in $\kappa_{\text{prof}}$") was unintentionally dropped. Production
`run_model` was deriving curvature/slope for the leaky shape
functions from the raw DEM, so single-cell LIDAR κ⁺ outliers
saturated $\mathrm{sat}(\kappa^+/\kappa_{\text{ref}})$ and pulled
$f_{\text{drain}}$ to its $f_{\min}$ floor on isolated cells.

- [x] `physics/pipeline.py:run_model` — added
  `curvature_smoothing_sigma_m` (default 10 m). When σ > 0, a
  Gaussian-smoothed copy of the raw DEM (NaN-aware, reusing the
  existing `_gaussian_smooth_nan` helper) feeds slope and profile
  curvature into the leaky shape functions. The raw-DEM slope,
  aspect, and curvature feeding irradiance and the `RunResult`
  diagnostic fields are unchanged. Cast shadow and heating still
  use the raw DEM. σ = 0 reproduces pre-2026-05-09 behaviour
  exactly.
- [x] `cli.py` — `--curvature-smoothing-sigma` flag on both `run`
  and `preview` subcommands. Default 10 m. Plumbed through
  `_cmd_run`, `_cmd_preview`, and the four `viz` plotters that
  call `run_model` (`plot_trigger_potential`,
  `plot_weighted_convergence`, `plot_leak`, `plot_cycle_period`).
- [x] `tests/test_physics_pipeline.py` — three new tests covering
  spatial-roughness drop on a noisy-ramp fixture (≥ 2× reduction),
  σ=0 reproducing pre-fix behaviour (regression guard against
  unintended branch interactions), and energy conservation
  preserved at σ=20 m on the mirror-spur fixture.
- [x] `MODEL.md` §11.8, §6, §11.6 — documented the pre-smooth as a
  carry-forward of the §6 prescription with the
  predecessor-vs-production wording.
- [x] Mallerstang re-render at no-wind midday today + canonical
  SW-afternoon: see `docs/VALIDATION.md` § 2026-05-09 addendum.

**Stage 3 gate**: speckle gone from the no-wind midday Mallerstang
leak panel; canonical SW-afternoon visual gate from Stage 2 still
holds; energy closure preserved (98.7 % leak / 1.3 % residual on the
canonical SW render); test suite green (294 passed; 1 pre-existing
hypothesis-property failure in `test_aspect_rotates_under_rot90`
unrelated to this change).

## Phase 3.2 — Drafting / leak aggregation (2026-05-11)

The Phase 3.1 leaky kernel routes heating energy correctly but the
downstream display layer was rank-normalising the cell-level `leak`
field and clustering at $q_{95}$, which penalised diffuse spurs
relative to concentrated scarp lips. The `cycle_period_s` field
showed the spurs working; the trigger raster and KMZ did not. See
`docs/TODO.md` "Drafting" for the motivating diagnosis.

Physically, rising buoyant plumes coalesce as they ascend; a pilot
at trigger height samples a ground footprint several thermal-column
radii across. Treating cell-level leak as the trigger granularity
is the wrong scale.

- [x] `physics/pipeline.py:run_model` — added
  `draft_aggregation_sigma_m` kwarg (default 75 m, ≈ one thermal
  column radius at low trigger altitude). After the leaky kernel
  call, smooth `leak` with the existing NaN-aware
  `_gaussian_smooth_nan` helper at the requested σ, then reapply
  the slope mask (using the same `min_slope_rad` the kernel uses
  for `f_drain`/`q_storage`, so the mask is consistent with kernel
  behaviour). σ = 0 reproduces the predecessor `trigger_potential
  = rank_normalise(leak)` cell-for-cell.
- [x] `RunResult` — new fields `draft_potential` (W/m², the
  aggregated field) and `draft_mask_loss_total` (scalar diagnostic
  for energy thrown away by the post-smooth slope mask).
  `trigger_potential` is redefined as
  `rank_normalise(draft_potential)`. `leak` and
  `residual_at_sinks_total` are unchanged — the conservation
  invariant on the underlying physical field is preserved.
- [x] `triggers/cluster.py:cluster_triggers` — new optional
  `leak_weights` kwarg. When supplied alongside `cycle_period_s`,
  per-cluster cycle period switches from arithmetic mean of τ to
  the leak-weighted mean
  `Σᵢ leakᵢ·τᵢ / Σᵢ leakᵢ` over cluster cells with finite τ and
  positive leak — "the dominant cycle of the cells actually
  producing this thermal", which is the right per-thermal summary
  now that smoothed clusters can span many low-leak cells. Without
  `leak_weights` the legacy arithmetic-mean path stays in place.
- [x] CLI: `--draft-aggregation-sigma` (m, default 75) on `run` and
  `preview`. New `preview --what draft` choice. `_cmd_run` now
  clusters on `RunResult.draft_potential` with
  `leak_weights=RunResult.leak` so the KMZ inherits both the
  rescue-the-spur benefit and the physically-meaningful per-cluster
  τ.
- [x] `viz/`: new `plot_draft_potential` plotter modelled on
  `plot_leak`. The four wind-requiring plotters
  (`plot_trigger_potential`, `plot_weighted_convergence`,
  `plot_leak`, `plot_cycle_period`) all gained the new σ kwarg and
  forward it to `run_model`. Re-exported from
  `thermal_model.viz.__init__`.
- [x] Tests: σ=0 collapses `draft_potential` to `leak` exactly;
  diffuse-vs-concentrated 64×64 toy gives near-equal centre rank
  (the TODO.md 3×3 toy generalised); plateau-with-rim fixture
  asserts the post-smooth slope mask zeros plateau-interior bleed
  while preserving rim leak; energy conservation on `leak`
  unchanged at σ=75. Cluster: leak-weighted mean asserted on a
  hand-built 3-cell cluster (leak ∈ {10, 90, 50}, τ ∈ {60, 600,
  300} s) — leak-weighted mean = 464 s, arithmetic mean = 320 s,
  the two paths verifiably disagree. Plus an explicit
  zero-weight-cell exclusion guard. CLI smoke for
  `--draft-aggregation-sigma` and `preview --what draft`. Viz
  smoke for `plot_draft_potential`. **All twelve new tests pass.**
- [x] `docs/MODEL.md` — added §11.9 documenting the aggregation
  step.

**Phase 3.2 gate (code-side, cleared 2026-05-11)**: 333 passed,
1 pre-existing hypothesis failure in
`test_aspect_rotates_under_rot90` unrelated to this branch; ruff /
format / mypy clean. **Visual gate**: Mallerstang re-render at 5 m
under canonical SW summer afternoon conditions (225° @ 6 m/s,
2026-07-15 13:00 BST) is the operator follow-up — expectations:
SW spur shoulders appear as broad bright zones on `draft_potential`
where `leak` shows them faint; Mallerstang Edge cliff line still
dominates (we're not handicapping scarps, only rescuing spurs);
Wild Boar Fell summit-plateau interior remains dim
(slope mask working); new spur clusters survive q95 thresholding
and reach the KMZ with leak-weighted cycle periods longer than the
existing scarp-lip clusters. Sensitivity sweep at σ ∈ {0, 25, 50,
75, 100, 150} m to confirm 75 m default. Write up in
`docs/VALIDATION.md`.

## Phase 4 — Land cover + time-of-day

- [x] UKCEH land cover ingestion + absorptivity table
  **(wired 2026-05-10 on branch `feat/phase4-land-cover-heating`).**
  * `thermal_model/io/land_cover.py` — `LandCover` dataclass mirroring
    `DEM`; `read_land_cover` + `absorptivity_from_land_cover`. Reproject
    via `rasterio.warp.reproject` with `Resampling.nearest` (categorical
    data — bilinear would invent class codes), vectorised 256-entry
    LUT lookup, no per-cell loops. LCM nodata cells fall back to
    `unknown_fill` (default `DEFAULT_ABSORPTIVITY = 0.80`), **not** to
    NaN — a sliver of unclassified land must not zero the heating
    weight that feeds the leaky-bucket routing. DEM-NaN cells propagate
    to NaN α (so the routing's NaN-handling stays load-bearing).
  * `thermal_model/io/land_cover_wms.py` — `fetch_lcm_for_dem(dem)`
    issues a single `GetMap` against
    `LC.10m.GB` on the public UKCEH WMS spanning the DEM footprint at
    10 m in EPSG:27700, decodes the PNG, and reverse-maps RGB → class
    via the hardcoded `UKCEH_LCM_PALETTE` (sampled from the live
    `GetLegendGraphic` 2026-05-10). On-disk cache under
    `data/cache/lcm/<layer>/<sha1>.png` keyed by canonical URL.
    Stdlib `urllib.request` only — no new conda deps. Oversize bbox
    (> 2048 px at 10 m) raises `NotImplementedError`; chunked WMS
    fetch is follow-up scope. Two coastal palette collisions in the
    WMS rendering (15↔16, 17↔18) are documented and resolved to the
    rock-side class; they don't appear in inland Yorkshire work.
  * **Production `UKCEH_LCM_ABSORPTIVITY` is deliberately empty** —
    operator authors the full 21-class α table. Until then, every
    class falls through to `DEFAULT_ABSORPTIVITY` (= scalar-α run
    behaviour, the safe default). `DALES_LCM_ABSORPTIVITY` ships a
    minimal 8-class Dales-focused LUT for tests and the validation
    render (heather, bog, rock, grass, freshwater, urban, suburban).
  * CLI: `--land-cover PATH` and `--land-cover-wms` on both `run` and
    `preview`, in an `add_mutually_exclusive_group` with
    `--absorptivity`. Sibling flags `--lcm-layer` (default
    `LC.10m.GB`) and `--no-lcm-cache`. `_resolve_heating_args` in
    `cli.py` widens its `absorptivity` kwarg contract from `float`
    to `float | np.ndarray`; `run_model` and `heating_field` already
    accept either (Phase 2 ROADMAP), so no further plumbing changes
    were needed downstream.
  * Viz: `plot_absorptivity` (continuous α overlay on hillshade) and
    `plot_land_cover` (categorical view with `ListedColormap` +
    `BoundaryNorm`, optional class-name legend). Re-exported from
    `thermal_model.viz`. The existing `plot_heating` /
    `plot_trigger_potential` / `plot_weighted_convergence` /
    `plot_leak` / `plot_cycle_period` plotters widened their
    `absorptivity` type hint to `float | np.ndarray` for consistency
    with the CLI plumbing.
  * Fixtures: `data/fixtures/wild_boar_fell_east_256_lcm.tif` (256×256
    uint8 categorical, EPSG:27700, transform-aligned with the existing
    DEM fixture; six classes including a class-99 unknown sliver) plus
    a `synthetic_lcm_path` 4-quadrant fixture in `tests/conftest.py`.
    Build script `tools/build_lcm_fixture.py` regenerates the
    Wild Boar Fell tile from the DEM fixture's transform.
  * Tests:
    * `tests/test_io_land_cover.py` — 14 tests covering round-trip,
      nearest-neighbour resampling correctness across cell-size
      changes, CRS mismatch warning, NaN propagation from the DEM,
      LCM-nodata fallback (the load-bearing routing-preservation
      contract), unknown-class fallback + warning, lookup / fill
      overrides, the empty production LUT regression guard, and the
      `LandCover.shape` property.
    * `tests/test_io_land_cover_wms.py` — 7 tests with mocked
      `urllib.request.urlopen`: palette-encoded PNG round-trip,
      off-palette pixels → `-1` + warning, GetMap URL construction,
      cache hit / no-cache plumbing, oversize-bbox
      `NotImplementedError`, missing-CRS guard.
    * `tests/test_physics_pipeline.py::test_uniform_array_alpha_matches_scalar_alpha_cell_for_cell`
      — **the strongest Phase 4 gate**: a uniform α-array produces
      cell-for-cell identical `RunResult.leak`,
      `RunResult.trigger_potential`, and `RunResult.heating_wm2` to a
      scalar α of the same value. Catches any silent broadcasting or
      NaN-substitution drift in the array plumbing.
    * `tests/test_cli.py` — 5 new tests: local-file
      `preview --what heating --land-cover`, full
      `run --land-cover` writing a trigger GeoTIFF, two
      mutual-exclusion smoke tests (exit code 2), and a
      monkeypatched-WMS `preview --land-cover-wms` end-to-end test.
    * `tests/test_viz.py` — 4 new smoke tests for `plot_absorptivity`
      (incl. NaN propagation) and `plot_land_cover` (shape-mismatch
      guard included).
  * Validation render: see `outputs/mallerstang_phase4_render.py` (a
    standalone script mirroring the Phase 3.1
    `outputs/mallerstang_leaky_render.py` pattern). Drives
    `fetch_lcm_for_dem` against the Mallerstang DEM and renders the
    α / categorical / leak / side-by-side panels under the canonical
    SW-summer-afternoon conditions. The Mallerstang re-render against
    real UKCEH data is the validation follow-up to this branch; once
    a Mallerstang DEM is on disk the operator runs the script and the
    closure stats / visual gate land in `docs/VALIDATION.md`.
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
