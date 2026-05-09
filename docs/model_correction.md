# Model correction — ground-level trigger prediction

**Status**: supersedes the wind-drift description in the original prototype
(`thermal_model.py`) and any references to `drift_field()` in `physics/`.

**Read this before touching anything in `thermal_model/physics/`.**

> **Phase 3.1 reformulation in flight (opened 2026-05-09).** The
> "weighted convergence × κ̂⁺ × slope_mask" formulation in §3 + §5
> below is being replaced by a *leaky-bucket weighted accumulation*
> — each cell consumes a curvature/slope-dependent fraction of its
> through-flow as trigger output and forwards the rest, with a
> per-cell storage capacity giving a cycle period τ as a new
> output. This addresses two physical defects of the formulation
> below: energy double-counting along the flow path, and the
> absence of a cyclic-dump regime on gentle terrain. **Stage 1**
> of 3.1 (the kernel as a standalone module) has landed. **Stage 2**
> (production fold-in) is pending. Until Stage 2 closes, the
> production pipeline still follows §3 + §5 of this document. The
> full derivation of the replacement is in `docs/MODEL.md` § 11;
> the staging is in `docs/ROADMAP.md` § Phase 3.1.

---

## 1. What the model is actually predicting

The model predicts **where on the ground a thermal will source and trigger**,
not where it will be once it is airborne. This distinction drives every design
decision below.

A thermal trigger location has three necessary conditions, all of which must
be met simultaneously at a single ground cell:

1. **Warm air has pooled there.** Surface boundary-layer air, heated by
   the terrain below it, has been guided by topography to converge at this
   cell. Without a supply of warm air there is nothing to trigger.

2. **The cell has received enough solar energy.** Convergence alone is not
   sufficient. Two geometrically equivalent spurs facing different directions
   will have very different thermal strength depending on sun angle and time
   of day.

3. **The terrain is convex enough to cause detachment.** Rising warm air
   clings to a slope until the slope curves away beneath it. At a convex
   break — a cliff lip, a spur shoulder, a sudden steepening — the air can
   no longer follow the surface and detaches as a thermal bubble.

The model therefore has three layers: **convergence**, **heating**, and
**curvature**, combined as:

```
trigger_potential = f(convergence, heating, profile_curvature)
```

---

## 2. The inverted-treacle convergence layer (corrected understanding)

### What the inverted DEM flow accumulation represents

Warm air near the terrain surface behaves like a thin fluid. It rises up
slopes under buoyancy and, just like water flowing downhill, it follows the
path of steepest gradient — but *uphill* rather than downhill.

Inverting the DEM (negating heights) converts "steepest uphill" into
"steepest downhill", so standard hydrological flow-accumulation algorithms
applied to the inverted DEM give the correct routing for surface warm-air
flow. The accumulation value at each cell is the total terrain area whose
warm air drains *through* that cell on its way to the summit.

High accumulation = many cells of warm air funnelling through this point
= strong thermal source if the other conditions (heating, curvature) are met.

### What it does NOT represent

The convergence map does not predict where a thermal will be at altitude,
nor how far it has drifted once airborne. It predicts the ground-level
concentration of surface warm air *before* detachment.

### Processing order

```
DEM
  ↓  gaussian_smooth(sigma=10–25 m)     # remove sub-thermal noise
  ↓  + wind_tilt_ramp                   # see §4 — do this BEFORE inversion
  ↓  negate                             # invert
  ↓  fill_pits(epsilon=True)            # richdem: route across flats
  ↓  D∞_flow_accumulation               # richdem: smooth, no D8 striping
  =  convergence[row, col]
```

The Gaussian smooth and the wind tilt both happen on the **real DEM before
inversion**. Never tilt the already-inverted DEM.

---

## 3. The solar heating layer

### Role

The heating layer is a per-cell weight that answers: how much energy is this
patch of ground injecting into the air above it right now?

It multiplies the convergence map. A spur with high convergence but no sun
(north-facing, deep shadow, wet peat) is suppressed. A moderately convergent
south-facing rocky slope in strong afternoon sun is boosted.

### Computation

```python
slope_rad, aspect_rad = slope_aspect(dem, cell_size)
sun_elev, sun_az = solar_position(lat, lon, datetime_utc)   # pvlib preferred
irradiance = solar_irradiance(slope_rad, aspect_rad, sun_elev, sun_az,
                              dem=dem, cell_size=cell_size,
                              cast_shadows=True)
heating = irradiance * absorption_coefficient(land_cover)
```

For land cover absorption, dry rock and bare soil (~0.80–0.85) heat strongly.
Wet peat and bog (~0.20–0.35 effective, because evaporation absorbs most
incoming energy) should be heavily suppressed. Heather and dry grass sit
around 0.70–0.78.

### Integration with convergence (heating becomes the flow weight)

Heating is **not** combined with convergence as a separate layer
multiplied in afterwards. Instead, the heating field is the
per-cell weight on the D∞ flow accumulation of §2. Each cell
contributes its own W/m² to the routing rather than a unit count of
1, so the accumulation result at every cell is the total upstream
thermal energy that has flowed through it on its way to the summit:

```python
weighted_convergence = dinf_flow_accum(
    invert(tilted_dem),
    weights=heating,
)
```

Why this is better than the post-hoc multiplier
$\sqrt{\hat H \cdot \hat C}$ that an earlier draft of this document
proposed:

- A **local** multiplier zeros every cell where $H = 0$. A shadowed
  convergent point fed by a sunny upwind face is then wrongly
  marked dead, even though warm air from that upwind face is
  physically flowing through and pooling at that point. Weighted
  accumulation routes the upstream warmth to the convergent point
  and gets the right answer.
- $C$ and $H$ are not independent layers — convergence is the
  *transport* of the warm air heating produced. Folding the
  coupling into the routing itself reflects that physics.
- A shadowed spur whose entire catchment is also shadowed
  correctly stays low under weighted accumulation, because no
  upstream cell has injected energy. Both formulations agree on
  this case; they differ on the "shadowed-downstream-of-sunny" case
  above, which the model gets right only when the integration is
  intrinsic to the routing.

`richdem`'s `FlowAccumulation(method='Dinf', weights=...)` accepts
a per-cell weight raster as an `rdarray`; the existing project
helper `physics.flow_accumulation` already plumbs this through (see
`docs/ROADMAP.md` Phase 1). The numpy fallback path must accept
weights too so CI keeps working without `richdem`. Verify the
installed `richdem` version exposes the `weights` keyword before
relying on it; if not, the fallback is to obtain D∞ fractions via
`rd.FlowProportions` and apply a manual weighted accumulation pass
in topological (descending-elevation) order — equivalent, and
already the structure of the numpy reference path.

---

## 4. Wind as terrain tilt — the principal correction

### What was wrong in the original prototype

The original code contained a `drift_field()` function that shifted the
thermal potential map by a fixed vector representing how far a thermal would
drift *after detachment* while rising to a specified height. This is the
wrong problem. The model is trying to find *where on the ground* thermals
source, not where a pilot will encounter them at altitude.

**Remove or clearly quarantine `drift_field()`.** It should not be part of
the main pipeline. If thermal-in-air drift is ever needed (e.g. for XC track
correlation) it belongs in a separate, clearly labelled utility.

### The correct physical role of wind

Wind in the boundary layer sweeps warm surface air in the downwind direction.
The effect on thermal sourcing is:

- Windward faces (facing into the wind) have increased turbulent mixing; warm
  air is continually displaced before it can pool.
- Lee faces (sheltered from wind, downwind side of ridges and spurs) allow
  warm air to accumulate. Thermals preferentially trigger on the lee side of
  terrain features.
- The stronger the wind, the further the effective convergence zones are
  displaced toward the lee.

This is equivalent to tilting the terrain slightly in the downwind direction
before computing the inverted-DEM flow accumulation. On a tilted-DEM, the
inverted flow accumulates preferentially on the downwind (lee) side of
features, which matches observations.

### Implementation: the directional ramp

```python
def wind_tilt_ramp(dem: np.ndarray, cell_size: float,
                   wind_from_deg: float, wind_speed_ms: float,
                   k: float = 0.03) -> np.ndarray:
    """Return a tilted DEM biased toward the downwind (lee) side.

    Adds a linear ramp to the DEM that is highest in the downwind direction.
    When subsequently inverted and flow-accumulated, the convergence zones
    shift toward the lee side of terrain features.

    Parameters
    ----------
    dem : ndarray
        Real terrain heights, metres, north-up raster.
    cell_size : float
        Cell size in metres (must be in projected CRS, e.g. EPSG:27700).
    wind_from_deg : float
        Meteorological wind direction (degrees FROM which wind blows, 0=N).
    wind_speed_ms : float
        Wind speed in m/s.
    k : float
        Tilt coefficient in s/m. Controls how strongly wind shifts the
        convergence zones. See tuning guidance below.
    """
    rows, cols = dem.shape
    col_idx, row_idx = np.meshgrid(np.arange(cols), np.arange(rows))
    col_m = col_idx * cell_size   # metres east
    row_m = row_idx * cell_size   # metres south (row increases southward)

    # Wind-to direction: where the wind is flowing toward.
    wind_to_rad = np.radians((wind_from_deg + 180) % 360)

    # Ramp: positive in the downwind direction.
    # East component: sin(wind_to_rad) × col_m
    # North component: cos(wind_to_rad) × (-row_m)  [north = decreasing row]
    ramp = col_m * np.sin(wind_to_rad) - row_m * np.cos(wind_to_rad)

    return dem + k * wind_speed_ms * ramp
```

### Sign convention verification

| Wind from | Flows toward | Ramp highest toward |
|-----------|-------------|---------------------|
| N (0°)   | S (180°)    | south (large row)   |
| S (180°) | N (0°)      | north (small row)   |
| W (270°) | E (90°)     | east (large col)    |
| SW (225°)| NE (45°)    | northeast           |

Verify by checking: does the convergence map show brighter features on the
expected lee side of Wild Boar Fell for a SW wind?

### Tuning the k parameter

`k` [s/m] determines how much effective height per metre of horizontal
distance per m/s of wind speed is added.

`k × wind_speed_ms` is dimensionless and represents the fractional slope
added. For k=0.03 and wind_speed=5 m/s: 0.15 metres of effective height
per metre of horizontal distance, or a 15% additional slope.

**Practical guidance:**

- k=0.01: very subtle, minimal lee-side bias. Good for light wind (<3 m/s).
- k=0.03: moderate. Reasonable starting point.
- k=0.05: strong bias. Significant shift toward lee side. Use for strong
  wind (>8 m/s) or as an upper-bound check.
- k>0.10: probably too strong; will override terrain geometry entirely.

**Tuning procedure**: run the model for a day with known thermal conditions
at 2–3 candidate k values. Compare predicted trigger zones against pilot
reports and GPS tracks from `docs/VALIDATION.md`. Choose the k that
minimises miss-distance to known sources.

k should eventually become wind-speed-dependent (lower k for light wind,
higher for strong). For now, a single value with manual selection is
sufficient.

### What wind does NOT do in this model

- It does not drift thermals after detachment. That is a separate, out-of-
  scope calculation.
- It does not model rotors, wave, or mechanical turbulence.
- It does not account for wind shear (single surface-layer vector only).
- It does not model lee-wave uplift or convergence lines between valley winds
  and synoptic winds.

These are known limitations, not bugs.

---

## 5. Profile curvature — the trigger detector

### Why convergence alone is insufficient for cliff tops

The convergence map detects **plan convergence**: where flows from multiple
directions concentrate. A spur tip lights up because air from the north and
south flanks both drain through the tip cell.

A long straight cliff face does not light up strongly on the convergence map
because flow across it is *laminar* — all cells drain in parallel toward the
summit. The cliff top is a through-flow zone, not a concentration zone, in
the inverted-DEM sense.

Yet cliff tops are strong trigger points in practice because of **profile
convexity**: the terrain curves away beneath the rising air, causing
detachment. This is a second physical mechanism, orthogonal to plan
convergence.

### Computation

```python
def profile_curvature(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Curvature along the steepest-descent direction (Horn-weighted).

    Positive = convex (slope steepens going downhill = trigger terrain).
    Negative = concave (slope flattens going downhill = bowl terrain).

    Compute from the ORIGINAL DEM, not the wind-tilted DEM.
    The tilt is a flow-routing device; real trigger geometry is
    a property of actual terrain shape.
    """
    dz_dy, dz_dx = np.gradient(dem, cell_size)
    # Second derivatives (can use Horn-weighted version for robustness)
    _, dzz_xx = np.gradient(dz_dx, cell_size)
    dzz_yy, _ = np.gradient(dz_dy, cell_size)
    dzz_xy, _ = np.gradient(dz_dx, cell_size)
    # Actually: re-derive from Horn or use scipy for cleaner implementation.
    p = dz_dx**2 + dz_dy**2
    q = p + 1.0
    eps = 1e-9
    prof = np.where(
        p > eps,
        (dzz_xx*dz_dx**2 + 2*dzz_xy*dz_dx*dz_dy + dzz_yy*dz_dy**2)
        / (p * np.power(q, 1.5)),
        0.0,
    )
    return prof
```

### Using curvature in the trigger map

```python
# Only positive curvature (convex terrain) contributes to triggers.
curv_positive = np.maximum(profile_curv, 0)

# Normalise to [0, 1].
curv_norm = curv_positive / max(np.percentile(curv_positive, 99), 1e-9)

# Normalise the heating-weighted convergence the same way.
wc_norm = np.clip(
    weighted_convergence
    / max(np.percentile(weighted_convergence[weighted_convergence > 0], 99), 1e-9),
    0, 1,
)

# Combine: both upstream-thermal-energy delivery AND convexity must be present.
trigger_potential = wc_norm * curv_norm
```

This allows cliff tops (high curvature, moderate weighted-convergence) and
heated spur tips (moderate curvature, high weighted-convergence) to both
appear as triggers, which matches experience: both terrain types produce
thermals, but through different dominant mechanisms.

### Minimum slope gate

Cells on flat summits or valley floors should be suppressed regardless of
their curvature or heating values. Apply a minimum slope mask:

```python
slope_rad, _ = slope_aspect(dem, cell_size)
slope_mask = slope_rad > np.radians(2.5)   # 2.5° threshold
trigger_potential = trigger_potential * slope_mask
```

This also suppresses residual flat-top artefacts from the pit-filling step.

---

## 6. The corrected full pipeline

```
Inputs
  dem          : float32 ndarray, metres, north-up, EPSG:27700
  cell_size    : float, metres
  lat, lon     : float, decimal degrees (centre of tile)
  datetime_utc : datetime (aware)
  wind_from    : float, degrees meteorological (FROM direction)
  wind_speed   : float, m/s
  land_cover   : str ndarray, optional (same shape as dem)
  k            : float, wind tilt coefficient (default 0.03)

Pipeline
  1. smooth_dem    = gaussian_smooth(dem, sigma_cells=10)
                     # sigma_cells ≈ sigma_m / cell_size, typically 10–25 m
                     # suppress sub-thermal-scale noise before flow routing

  2. tilted_dem    = wind_tilt_ramp(smooth_dem, cell_size,
                                    wind_from, wind_speed, k)
                     # tilt BEFORE inversion

  3. slope, aspect = slope_aspect(dem, cell_size)
                     # from ORIGINAL dem, not smoothed or tilted

  4. heating       = solar_irradiance(slope, aspect, sun_elev, sun_az,
                                      dem=dem, cast_shadows=True)
                     * absorption(land_cover)
                     # W/m², the per-cell flow-accumulation weight in step 5

  5. weighted_conv = dinf_flow_accum(invert(tilted_dem), weights=heating)
                     # richdem: fill pits, then D∞ accumulation with
                     # heating as per-cell weight. There is NO separate
                     # "combine convergence and heating" step — the
                     # integration is intrinsic to the weighted routing.

  6. profile_curv  = profile_curvature(dem, cell_size)
                     # from ORIGINAL dem

  7. slope_mask    = slope > radians(2.5)

  8. trigger       = normalise(weighted_conv)
                     * normalise(max(profile_curv, 0))
                     * slope_mask

Outputs
  trigger_potential  : float32, [0, 1], primary product
  weighted_conv      : float32, raw upstream W/m² (× cell-area), diagnostic
  heating            : float32, W/m², diagnostic
  (No separate `thermal_energy` raster — that intermediate is gone.)
```

### Which DEM each step uses

| Step | DEM input | Other inputs | Reason |
|------|-----------|--------------|--------|
| Gaussian smooth | raw dem | — | noise removal before all processing |
| Wind tilt | smoothed dem | wind vector, k | tilt applied to clean terrain |
| Slope/aspect | raw dem | — | real terrain gradient |
| Solar irradiance | raw dem | sun position | cast shadows from real terrain |
| Heating | — | irradiance, land_cover | W/m² weight raster for step 5 |
| Inversion + weighted flow accum | tilted smoothed dem | weights = heating | routing on effective terrain, weighted by W/m² so each cell contributes its own thermal energy injection |
| Profile curvature | raw dem | — | real terrain shape drives detachment |
| Slope mask | raw dem | — | real terrain slope |

Heating is no longer a "downstream layer multiplied in"; it is an
**input** to the flow-accumulation step, alongside the DEM. The
output of the routing is *already* the heating-weighted convergence;
no further coupling step exists.

---

## 7. Parameter reference

| Parameter | Symbol | Recommended range | Notes |
|-----------|--------|------------------|-------|
| Gaussian sigma | σ | 10–25 m | Start at 10 m. Increase if too noisy. |
| Wind tilt coefficient | k | 0.02–0.05 s/m | Tune against VALIDATION.md. |
| Min slope | θ_min | 2–3° | Kills flat-summit artefacts. |
| Profile curv threshold | — | 90th–95th percentile | Or use continuous weighting. |
| Weighted-convergence clip | — | 99th percentile | Before normalising. |

(The previous separate "convergence clip" and "heating clip" rows are
no longer needed: heating enters the routing as a weight, and the
single normalised quantity downstream is the heating-weighted
convergence.)

---

## 8. What to change in the codebase

### Remove or quarantine

`physics/drift_field.py` (or `drift_field()` wherever it lives): this
function solved the wrong problem. Move it to `utils/airborne_drift.py` with
a clear docstring stating it is NOT part of the main trigger-prediction
pipeline. Do not import it from `physics/`.

### Add

- `physics/wind_tilt.py`: the `wind_tilt_ramp()` function from §4.
- `physics/pipeline.py`: the corrected `run_model()` orchestrating §6.
  Replace the old `run_model()` that called `drift_field`. **No
  intermediate `thermal_energy` raster** — heating is passed as the
  `weights` argument to `flow_accumulation`, and the output of that
  call is already the heating-weighted convergence.

### Modify

`physics/scene.py` (or equivalent): update `ThermalScene` dataclass
to store `tilted_dem` and `weighted_convergence` as fields, and
rename any `drifted_*` or `thermal_energy` fields to remove the
superseded framings. Add `trigger_potential` as the primary output
field.

`physics/coupling.py` (`thermal_potential` $= H^p \cdot C^q$): no
longer wired into the production pipeline. Keep the function for
backward compatibility and one-off sensitivity studies, but
`pipeline.py` must not import it. The geometric-mean coupling that
this function implements has been superseded by passing heating as
a flow-accumulation weight (see "Integration with convergence" in
§3 above).

`cli.py`: remove `--release-height` and `--climb-rate` arguments
(they were inputs to the drift model). Add `--wind-tilt-k` as an
optional float argument with default 0.03.

`docs/MODEL.md`: the pipeline diagram there (if it exists) needs
updating to match §6 above.

---

## 9. Validation expectations

After implementing the corrected pipeline, the trigger map for Wild Boar Fell
+ Mallerstang on a typical SW summer afternoon (wind 5–8 m/s from 210–240°,
moderate convection by 1200–1400 BST) should show:

- Bright spots on the SW-facing lower flanks of Wild Boar Fell (direct sun,
  good convergence, convex spur shoulders).
- Enhancement on the lee (NE) side of the main E-facing scarp relative to
  zero-wind baseline (wind tilt effect).
- Mallerstang Edge cliff-line showing trigger potential from profile
  curvature × moderate energy (even with lower plan convergence than spur
  tips).
- Flat summit plateau of Wild Boar Fell remaining dark (slope mask + low
  convergence after smoothing).
- Valley floors suppressed (slope mask + lack of plan convergence on inverted
  DEM).

If the NE side of Wild Boar Fell is not enhanced relative to the zero-wind
baseline, the wind tilt has the wrong sign — check the ramp formula's sign
convention against §4.

See `docs/VALIDATION.md` for specific grid references of known reliable
thermal sources to use as ground truth.

---

## 10. Known limitations of the corrected model

**Single wind vector.** Wind varies spatially with terrain (channelling,
acceleration over ridges, valley flows). A future upgrade is a terrain-aware
wind field (e.g. WindNinja) which would make k irrelevant — the wind field
itself encodes the boundary-layer flow distortion.

**k is empirical.** There is no clean analytical derivation of k from
boundary-layer physics at this level of abstraction. It must be calibrated
against observations. It will likely vary with atmospheric stability class
(unstable days may need a different k than stable days).

**No time-of-day convergence migration.** As the sun moves, the heated zones
move, and the dominant trigger locations migrate. Currently the pipeline
treats each time step independently. A composite over e.g. 1100–1600 BST
at 30-minute intervals would give a better daily-average trigger map.

**Profile curvature is noisy at 2 m resolution.** Even after smoothing,
2 m LIDAR has enough ground-point scatter to produce spiky curvature values.
The curvature should be computed on the Gaussian-smoothed DEM (or a
moderately smoothed version), not the raw DEM, for the threshold step.
However the `slope_aspect` for irradiance should still use the raw DEM so
that shadowing is computed correctly.

**No rotor or convergence-line modelling.** Lee-side rotors and valley-wind
convergence lines are genuine thermal triggers not captured by the
gradient-based convergence model. These would require a flow solver.
