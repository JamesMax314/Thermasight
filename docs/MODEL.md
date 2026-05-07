# Model — math and physics

This document is the technical companion to the conceptual sketch in
`CLAUDE.md §2`. Read that first.

> **Model reformulated 2026-05-07** (drift → wind-tilt) **and refined
> the same day** (post-hoc heating × convergence multiplier →
> heating-weighted flow accumulation). Wind enters as a terrain tilt
> before flow accumulation (§3); heating enters as the per-cell
> weight on the same flow accumulation (§5), so the routing intrinsically
> integrates "where the air is warm" with "where it converges". The
> trigger raster is then `normalise(weighted_convergence) × κ̂⁺ ×
> slope_mask` (§7). The superseded drift formulation is preserved
> verbatim in §10 as a record. Production code follows §1–§8; do not
> reintroduce §10 without explicit operator approval. The full
> rationale for both reformulations lives in
> `docs/model_correction.md`.

---

## 1. What the model predicts

The model predicts **ground-level thermal source and trigger
locations** — where on a raster cell a thermal forms and detaches
from the terrain. It does *not* predict where a thermal ends up at
altitude, nor the climb profile a pilot will see.

Three conditions must hold simultaneously at a single ground cell:

1. Heated boundary-layer air has accumulated there (convergence).
2. Enough solar energy is being injected into the air column there
   (heating).
3. The terrain is convex enough at that point to detach the rising
   air column (profile curvature).

A minimum slope gate (~2.5°) additionally suppresses flat-summit and
valley-floor artefacts that would otherwise inherit high heating or
curvature.

---

## 2. The hydrological analogy (Phase 1 — still the core)

The working hypothesis: rising warm air on real terrain follows the
same gradient logic as falling water on inverted terrain. Concretely,
given a DEM `z(x, y)`, define the inverted surface

$$
\tilde z(x, y) = \max(z) - z(x, y).
$$

Hydrological flow accumulation on $\tilde z$ — the upslope
contributing area of every cell — gives the **thermal convergence**
field $C(x, y)$. Convex features on real terrain (spurs, cliff lips,
shoulders) are concave on $\tilde z$, so flow pools there and $C$ is
large.

In production, the inversion is applied to the *Gaussian-smoothed,
wind-tilted* DEM (see §3), not the raw DEM. The smoothing kills
sub-thermal noise; the tilt biases the resulting convergence toward
the lee side of features, which is where surface warm air actually
pools.

### Implementation notes

* Use D∞ (Tarboton, 1997) flow accumulation in production via
  `richdem`. D8 is too quantised for ridge-and-spur geometry.
* A pure-numpy fallback lives in `thermal_model/physics/_fallback.py`
  for CI and for environments where `richdem` is unavailable.
* Pre-fill pits on $\tilde z$ before accumulating, to suppress sinks
  introduced by quantisation noise and by the wind-tilt ramp.
* Convergence is logged — $\log_{10}(1 + C)$ is the displayable
  quantity.

### Validation gate

The convergence raster must light up on the known thermal locations
in `docs/VALIDATION.md` for at least three independent test tiles
before any Phase 2 work begins. See `CLAUDE.md §10`. The Phase 1 gate
was cleared informally on 2026-05-07; see
`docs/VALIDATION.md` § Validation log.

---

## 3. Wind as terrain tilt (Phase 3 — replaces the old drift step)

Wind in the surface boundary layer sweeps warm air downwind. Warm air
accumulates on the lee side of ridges and spurs and is continually
displaced from windward faces. The model captures this by adding a
linear ramp to the smoothed DEM *before inversion*:

$$
z_{\text{tilted}}(x, y) = z_{\text{smoothed}}(x, y)
  + k \, |\mathbf{u}| \bigl( x \sin\theta - y \cos\theta \bigr)
$$

where:

* $\theta$ is the **wind-to** bearing in radians, clockwise from
  north (so wind *from* 225° → wind *to* 045° → $\theta = \pi/4$).
* $|\mathbf{u}|$ is wind speed in m/s.
* $k$ is an empirical coefficient with units s/m. Default
  $k = 0.03$.
* $x$ is metres east, $y$ is metres north (so a row index $i$
  corresponds to $y = -i \cdot \text{cell\_size}$ in a north-up
  raster — the implementation handles the row-flip).

The ramp is *highest* in the wind-to direction; after inversion that
becomes the *lowest* part of the inverted surface, so flow
accumulates preferentially on the lee side of every feature.

### Sign convention check

| Wind from | Wind to       | Ramp peaks toward | Convergence biased toward |
|-----------|---------------|-------------------|---------------------------|
| N (0°)    | S (180°)      | south             | south (lee of N→S wind)   |
| S (180°)  | N (0°)        | north             | north                     |
| W (270°)  | E (90°)       | east              | east                      |
| SW (225°) | NE (45°)      | northeast         | northeast                 |

If the bias goes the wrong way, the ramp's sign is wrong; re-derive
from the table. The Wild Boar Fell + Mallerstang validation case
(SW summer afternoon) is the canonical sign-check; the NE flank
should brighten under SW wind relative to the zero-wind baseline.

### Tuning $k$

$k \cdot |\mathbf{u}|$ is dimensionless and represents the
fractional slope added to the terrain. For $k = 0.03$ at
$|\mathbf{u}| = 5$ m/s that's $0.15$ m of effective height per metre
of horizontal distance — a $\approx 8.5°$ extra slope. Practical
ranges:

| $k$ (s/m) | Bias       | When to use                                |
|-----------|------------|---------------------------------------------|
| 0.01      | very subtle | light wind ($< 3$ m/s)                     |
| 0.03      | moderate   | default, sensible starting point           |
| 0.05      | strong     | strong wind ($> 8$ m/s); upper-bound check |
| $> 0.10$  | excessive  | overrides terrain geometry; debug only     |

$k$ is calibrated against `docs/VALIDATION.md`. It will likely need
to be wind-speed-dependent eventually (lower $k$ for light wind,
higher for strong); for now a single value, manually selected, is
sufficient. Phase 5 replaces $k$ with a terrain-aware wind field
(`WindNinja`), at which point the empirical coefficient becomes
unnecessary.

### What the wind tilt does *not* do

* It does not advect thermals after detachment — that is a separate,
  out-of-scope problem (see §10).
* It does not model rotors, lee-wave uplift, or wind shear.
* It does not capture valley-wind / synoptic-wind convergence lines.

These are documented limitations, not bugs.

---

## 4. Heating field (Phase 2)

$$
H(x, y, t) = \alpha(x, y) \, \bigl( s(x, y, t) \cdot I_{\text{beam}}(x, y, t)
                                   + I_{\text{diffuse}}(x, y, t) \bigr)
$$

* $I_{\text{beam}}$, $I_{\text{diffuse}}$ — slope-projected direct
  and isotropic-diffuse irradiance from `pvlib`'s clear-sky model
  (Ineichen-Perez, default Linke turbidity 3.0).
* $\alpha$ — surface absorptivity (1 − albedo), looked up from land
  cover if available (`docs/DATA.md`); else a single Phase-2 default
  (0.80 — dry grass / heather upland Dales surface).
* $s \in \{0, 1\}$ — cast-shadow mask from a horizon scan along the
  solar azimuth. Multiplies the beam component only; diffuse comes
  from the whole sky and is not blocked by a single upwind ridge.

Slope and aspect are computed from the **raw** DEM by Horn's method
(1981); profile curvature uses Zevenbergen & Thorne (1987) on the
same 3×3 stencil. Real geometry drives shadows and gradients — the
heating field never sees the smoothed or wind-tilted DEM.

---

## 5. Heating-weighted convergence (Phase 3)

Heating and convergence are not combined as separate fields with a
post-hoc multiplier. Instead, the heating field $H(\mathbf{x})$ in
W/m² is the per-cell **weight** on the D∞ flow accumulation of §2,
so each cell contributes its own thermal-energy injection rate to
the routing rather than a unit count:

$$
C_w(\mathbf{x})
  = \mathrm{D}\!\infty\text{-accum}\bigl(\,
      \tilde z_{\text{tilted}};\; \text{weights}=H \,\bigr)(\mathbf{x}).
$$

$C_w(\mathbf{x})$ is the total upstream W/m² that has flowed through
$\mathbf{x}$ along the inverted-and-tilted gradient. Units are
$\mathrm{W/m^2 \cdot \text{cells}}$ (or, in metric mode,
$\mathrm{W/m^2 \cdot m^2} = \mathrm{W}$ — total upstream radiative
power, which is conceptually right but the magnitude is calibrated
against percentile rather than absolute W).

### Why weighted accumulation, not a separate multiplier

The previous formulation used $E = \sqrt{\hat H \cdot \hat C}$ on
normalised inputs (§10.1, briefly held). Two problems:

1. A *local* multiplier zeros the cell whenever $H = 0$ at that cell,
   even if a strong upstream sunlit catchment is feeding warm air
   into it. The physical answer for a shadowed convergent point fed
   by a sunny upwind slope is "warm air pools here"; the local
   multiplier wrongly says "cold cell, no thermal".
2. The post-hoc combination treats $C$ and $H$ as independent
   layers when they are physically coupled — convergence is the
   *transport* of the warm air the heating produced.

Weighted accumulation handles both correctly: routing carries the
heating signal along the gradient and accumulates it where the air
pools. A shadowed convergent point downstream of a sunny face gets
the right answer. A shadowed spur whose entire catchment is also
shadowed correctly stays low because no upstream cell injected
energy.

### Implementation note

Use `richdem`'s `FlowAccumulation` with `method='Dinf'` and the
`weights` keyword carrying the heating raster as an `rdarray` with
the same nodata convention as the DEM. The Phase 1 `flow_accumulation`
helper already exposes a weights argument (see `docs/ROADMAP.md`
Phase 1); the Phase 3 wiring is purely a question of what gets
passed in. The numpy fallback path must also accept and apply
weights to keep CI working without `richdem`.

### Relationship to `physics.coupling.thermal_potential`

The existing $P = H^p \cdot C^q$ helper is **no longer used in the
production pipeline.** It is retained for backward compatibility and
sensitivity-analysis exploration only; new code paths should not
import it. Phase 4's planned time-of-day weighting (heating-weighted
morning, convergence-weighted afternoon per `CLAUDE.md` §5) is now
expressed by scaling the heating weights themselves rather than by
sweeping $(p, q)$ on a separate coupling step.

---

## 6. Profile curvature filter (Phase 3)

Profile curvature $\kappa_{\text{prof}}$ is the second derivative of
elevation in the steepest-descent direction (Zevenbergen & Thorne
sign convention: positive = convex, negative = concave). It is
computed from the **raw** DEM, not the smoothed or wind-tilted one
— the tilt is a flow-routing device; real terrain shape drives
detachment.

Only positive curvature contributes to triggers; concave terrain
never triggers regardless of how much energy or convergence is
present.

$$
\hat\kappa^+(\mathbf{x})
  = \mathrm{clip}\!\left(
      \frac{ \max\bigl(\kappa_{\text{prof}}(\mathbf{x}), 0\bigr) }
           { q_{99}\bigl(\max(\kappa_{\text{prof}}, 0) \bigr) },\
      0,\ 1
    \right).
$$

A Gaussian pre-smooth at one DEM cell suppresses LIDAR speckle in
$\kappa_{\text{prof}}$; this is independent of the σ ≈ 10–25 m
smooth applied to the DEM before flow routing.

---

## 7. Trigger composition (Phase 3)

The trigger-potential raster is the product of normalised
weighted-convergence, normalised positive curvature, and a
minimum-slope gate:

$$
T(\mathbf{x})
  = \hat C_w(\mathbf{x}) \cdot \hat\kappa^+(\mathbf{x})
    \cdot \mathbb{1}\!\bigl[\, \mathrm{slope}(\mathbf{x}) > \theta_{\min} \,\bigr]
$$

where

$$
\hat C_w
  = \mathrm{clip}\!\left(
      \frac{C_w}{q_{99}(C_w \mid C_w > 0)},\ 0,\ 1
    \right)
$$

(percentile-normalisation matching $\hat\kappa^+$ in §6) and
$\theta_{\min} \approx 2.5°$ kills flat-summit and valley-floor
artefacts. $T \in [0, 1]$ by construction (each factor is in
$[0, 1]$).

This composition allows two distinct trigger styles to coexist in
the output:

* **Heated spur tips** — moderate curvature, high $\hat C_w$
  (sunny upstream, geometrically convergent here). The convergence
  factor dominates.
* **Cliff edges and scarps** — high curvature, moderate $\hat C_w$
  (laminar through-flow rather than plan convergence). The
  curvature factor dominates.

Both are real trigger types in practice; an additive composition
would over-weight one style; the multiplicative form requires both
factors to be non-negligible, matching the §1 "all three conditions"
rule (heating + convergence are now jointly encoded in
$\hat C_w$).

---

## 8. Trigger clustering and export (Phase 3)

Cells where $T$ exceeds a high percentile (default 95th of
strictly-positive $T$) form a binary mask. Connected components on
this mask (8-neighbour by default) yields discrete trigger clusters
via `scipy.ndimage.label`. Components below `min_cluster_cells`
(default 3) are dropped as noise. Surviving clusters are ranked by
mean $T$.

DBSCAN with $\varepsilon = \text{cell\_size}$ on a regular raster
reduces to connected components plus a min-component-size filter, so
we use the natural raster primitive (`scipy.ndimage.label`) and
avoid the `scikit-learn` dependency (`CLAUDE.md` §4).

Each cluster becomes a point: centroid in raster coordinates →
projected (x, y) via the DEM transform → reprojected to WGS84 via
`pyproj` → KMZ via `simplekml`. The KMZ is the deliverable for
XCTrack / SeeYou / Google Earth.

---

## 9. Planned upgrade — Lagrangian plume (Phase 5)

The hydrological analogy is a static, gradient-only proxy. Phase 5
adds a Lagrangian plume model alongside it: parcels are seeded over
hot cells with a buoyancy budget and advected through a terrain-aware
wind field (`WindNinja`), entraining ambient air and detraining when
neutral. The output is a release-density raster directly comparable
with the (wind-tilted) convergence map of Phase 3, which is how the
upgrade will be validated. The two approaches are complementary
rather than competing: agreement between them on validation tiles is
the gate.

`WindNinja` also supersedes the empirical wind-tilt coefficient $k$
in §3; the terrain-aware wind field directly encodes the
boundary-layer flow distortion that $k$ approximates.

---

## 10. Old model — wind drift (superseded 2026-05-07)

> **Kept as a record. Do not implement this in
> `thermal_model/physics/` without explicit operator approval.**
>
> The drift formulation predicted the wrong thing: where an airborne
> thermal *ends up*, not where it sources from the ground. The model's
> actual job is to localise the trigger itself, which is a function
> of where boundary-layer air pools (the wind-tilt mechanism in §3),
> not where the parcel is once it's airborne. See
> `docs/model_correction.md` §4 for the full justification.

The original Phase 3 plan introduced two steps that have since been
removed:

### 10.1 Coupling (old, two superseded variants)

**Variant A (original, raw inputs).**

$$
P = \sqrt{H \cdot C} \qquad \text{(or, more generally, } P = H^p \cdot C^q\text{)}
$$

applied to *un-normalised* $H$ and $C$. The dynamic-range mismatch
between $H$ (0 to ~$10^3$ W/m²) and $C$ (1 to ~$10^5$ cell counts)
let single high-$C$ cells dominate the ranking even on a sqrt scale.
Implemented in `physics/coupling.thermal_potential`; retained in the
codebase for backward compatibility but **not** wired into the new
pipeline.

**Variant B (briefly-held, normalised inputs).**

$$
E = \sqrt{ \hat H \cdot \hat C }
$$

with $\hat{\cdot}$ percentile-normalised to $[0, 1]$. Held for
~hours during the 2026-05-07 reformulation before being superseded
by heating-weighted accumulation (§5). Why discarded: a *local*
multiplier zeros the cell wherever $H = 0$ at that cell, even when a
sunny upwind catchment is feeding warm air through the gradient to
this convergent point. Weighted accumulation gets this case right by
construction; the post-hoc geometric mean does not.

Both variants treated $C$ and $H$ as independent fields combined
after the fact. The current model recognises that they are
physically coupled — convergence is the *transport* of the warm air
that heating produced — and folds the coupling into the routing
itself.

### 10.2 Wind drift (removed)

Given a single wind vector $\mathbf{u}$, a release height $h$, and a
climb rate $w$, the source was mapped to its release location by
translating the potential field:

$$
P_{\text{drift}}(\mathbf{x}) = P\bigl(\mathbf{x} - \mathbf{u}\, h / w\bigr).
$$

Implemented as a sub-pixel `scipy.ndimage.shift`. The trigger mask
was then the intersection of $P_{\text{drift}}$ above a high
percentile with strongly-positive profile curvature.

This step has been **deleted** from the main pipeline. The old
`thermal_model/physics/drift.py` and the `drift_field()` /
`drift_distance_m()` API surface are gone. The CLI flags
`--release-height` and `--climb-rate` are also gone; the new wind
parameter is `--wind-tilt-k` (§3).

If post-detachment in-air drift is ever needed for a separate
purpose (e.g. XC track correlation), it must live in a separately-
named utility module under `thermal_model/utils/`, with a docstring
stating it is *not* part of the trigger-prediction pipeline. See
`docs/ROADMAP.md` Phase 3 quarantine note.
