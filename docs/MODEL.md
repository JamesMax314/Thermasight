# Model — math and physics

This document is the technical companion to the conceptual sketch in
`CLAUDE.md §2`. Read that first.

## 1. The hydrological analogy (Phase 1)

The working hypothesis: rising warm air on real terrain follows the same
gradient logic as falling water on inverted terrain. Concretely, given a DEM
`z(x, y)`, define the inverted surface

$$
\tilde z(x, y) = \max(z) - z(x, y).
$$

Hydrological flow accumulation on $\tilde z$ — the upslope contributing area
of every cell — gives the **thermal convergence** field $C(x, y)$. Convex
features on the real terrain (spurs, cliff lips, shoulders) are concave on
$\tilde z$, so flow pools there and $C$ is large.

### Implementation notes

* Use D∞ (Tarboton, 1997) flow accumulation in production via `richdem`. D8
  is too quantised for ridge-and-spur geometry.
* A pure-numpy fallback lives in `thermal_model/physics/_fallback.py` for CI
  and for environments where `richdem` is unavailable.
* Pre-fill pits on $\tilde z$ before accumulating, to suppress sinks
  introduced by quantisation noise.
* Convergence is logged — $\log_{10}(1 + C)$ is the displayable quantity.

### Validation gate

The convergence raster must light up on the known thermal locations in
`docs/VALIDATION.md` for at least three independent test tiles before any
Phase 2 work begins. See `CLAUDE.md §10`.

## 2. Heating field (Phase 2)

$$
H(x, y, t) = I(x, y, t) \cdot \alpha(x, y) \cdot s(x, y, t)
$$

* $I$ — clear-sky direct + diffuse irradiance on the slope, from `pvlib`.
* $\alpha$ — surface absorptivity (1 − albedo), looked up from land cover
  if available (`docs/DATA.md`), else a slope-and-aspect-only default.
* $s \in \{0, 1\}$ — cast-shadow mask from horizon scan along the solar
  azimuth.

Slope and aspect are computed from the DEM by Horn's method.

## 3. Coupling (Phase 2 → 3)

The thermal potential

$$
P = \sqrt{H \cdot C}
$$

is preferred over a plain product because it dampens the dynamic range of
$C$. A configurable exponent pair $(p, q)$ is exposed for sensitivity
analysis: $P = H^p \cdot C^q$.

## 4. Wind drift (Phase 3)

Given a single wind vector $\mathbf{u}$, a release height $h$, and a climb
rate $w$, the source is mapped to its release location by translating the
potential field:

$$
P_{\text{drift}}(\mathbf{x}) = P\bigl(\mathbf{x} - \mathbf{u}\, h / w\bigr).
$$

Implemented as a sub-pixel `scipy.ndimage.shift`. Single-vector drift is a
known weakness for lee-side triggering (`CLAUDE.md §5`).

## 5. Trigger detection (Phase 3)

A trigger is a pixel where $P_{\text{drift}}$ is locally high **and**
profile curvature is strongly positive (a convex break in slope). Profile
curvature is computed in the steepest-descent direction; a Gaussian
pre-smooth at one DEM cell suppresses LIDAR speckle.

Triggers are clustered (DBSCAN on the high-percentile mask) and exported as
points in GeoTIFF + KMZ.

## 6. Planned upgrade — Lagrangian plume (Phase 5)

The hydrological analogy is a static, gradient-only proxy. Phase 5 replaces
it with a Lagrangian plume model: parcels are seeded over hot cells with a
buoyancy budget and advected through a terrain-aware wind field
(`WindNinja`), entraining ambient air and detraining when neutral. The
output is a release-density raster directly comparable with the convergence
map of Phase 1, which is how we'll validate the upgrade.
