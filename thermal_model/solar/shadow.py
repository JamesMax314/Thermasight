"""Cast-shadow mask from a horizon scan along the solar azimuth.

The heating field of the model is

    H = I * alpha * s

where ``I`` is slope-projected irradiance from
:mod:`thermal_model.solar.irradiance`, ``alpha`` is surface
absorptivity, and ``s in {0, 1}`` is the *cast*-shadow mask: 1 on
cells whose line of sight to the sun is unobstructed, 0 on cells
shadowed by upwind terrain.

Self-shading of a slope whose normal points away from the sun
(``cos(theta_i) < 0``) is already handled inside
:func:`slope_irradiance` and is independent of nearby terrain. The
mask in this module is only the *cast* shadow from upwind features.

Algorithm
---------
For sun compass azimuth ``A`` and altitude ``alpha``:

1. Choose a 1-cell step along the grid's dominant axis (whichever of
   row or column has the larger absolute component in the sun's
   horizontal direction) and a fractional step on the orthogonal
   axis, so the path follows the sun bearing exactly.
2. At each step ``k``, advance every cell simultaneously, bilinearly
   sample the DEM at the offset position, and compare it to the
   sun ray's height at that distance,
   ``z + k * step_horiz_m * tan(alpha)``. Cells whose sample exceeds
   the ray height are shadowed at this step (and remain so).
3. Stop once ``min(z) + k * rise > max(z)`` — beyond that point no
   terrain ahead can rise far enough to occlude any cell.

The march is vectorised: each step is one
:func:`scipy.ndimage.map_coordinates` call on the whole grid. Total
cost is ``O(N * K)`` cells where ``K`` is the relief-bounded number
of steps. For UK terrain (max relief ~600 m) at midday in summer
(altitude ~50°), ``K`` is a few hundred.

Limitations
-----------
* Bilinear sampling on integer-step coordinates means a sub-cell
  feature exactly between two grid steps can be missed. For natural
  terrain (ridges, scarps, summits) — which is what shadows useful
  thermals — this is not a real source of error.
* This is a *direct-beam* shadow: it does not attenuate the diffuse
  component, since diffuse comes from the whole sky and is not
  blocked by a single upwind ridge.
"""

from __future__ import annotations

import numpy as np

from thermal_model.solar.position import SolarPosition


def cast_shadow_mask(
    dem: np.ndarray,
    cell_size_m: float,
    sun: SolarPosition,
) -> np.ndarray:
    """Cast-shadow mask under sun position ``sun`` via a horizon scan.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array in metres with NaN nodata.
    cell_size_m : float
        Square cell size in metres.
    sun : SolarPosition
        Sun position with a compass-bearing azimuth and altitude
        above the horizon.

    Returns
    -------
    np.ndarray
        Float64 mask, same shape as ``dem``:

        * ``1.0`` on sunlit cells,
        * ``0.0`` on cells shadowed by upwind terrain,
        * ``NaN`` on NaN input cells.

        With the sun below the horizon the entire finite domain is
        ``0.0`` (no direct beam anywhere). With the sun essentially
        at zenith the entire finite domain is ``1.0``.

    Notes
    -----
    Numerical ties at the ridge crest are broken in favour of
    "sunlit" — the comparison is strictly greater-than.

    The mask multiplies the *beam* component of the heating field
    only; the diffuse component is independent of cast shadows.
    """
    if dem.ndim != 2:
        raise ValueError(f"DEM must be 2-D, got shape {dem.shape}")
    if cell_size_m <= 0:
        raise ValueError(f"cell_size_m must be positive, got {cell_size_m}")

    rows, cols = dem.shape
    nan_mask = np.isnan(dem)

    if not sun.is_above_horizon:
        out = np.zeros((rows, cols), dtype=np.float64)
        out[nan_mask] = np.nan
        return out

    # Sun horizontal direction unit vector in (row, col). Compass
    # azimuth is clockwise from north; raster rows increase southward,
    # so the row component is -cos(azimuth) (north-positive flipped).
    sun_drow_unit = -float(np.cos(sun.azimuth_rad))
    sun_dcol_unit = float(np.sin(sun.azimuth_rad))
    # Snap near-zero components of cardinal-direction sun to exact
    # zero. ``cos(pi/2)`` is ``6.12e-17`` rather than 0, so a sun due
    # east would otherwise drift the sample row below 0 by a few
    # ULPs and ``map_coordinates`` with ``cval=-inf`` would treat the
    # whole top edge as off-grid.
    if abs(sun_drow_unit) < 1e-12:
        sun_drow_unit = 0.0
    if abs(sun_dcol_unit) < 1e-12:
        sun_dcol_unit = 0.0

    primary = max(abs(sun_drow_unit), abs(sun_dcol_unit))
    if primary < 1e-12:
        # Sun essentially at zenith — no horizontal projection, so no
        # cast shadows are possible from any finite-relief terrain.
        out = np.ones((rows, cols), dtype=np.float64)
        out[nan_mask] = np.nan
        return out

    drow_step = sun_drow_unit / primary
    dcol_step = sun_dcol_unit / primary
    horiz_per_step_m = cell_size_m * float(np.hypot(drow_step, dcol_step))
    rise_per_step_m = horiz_per_step_m * float(np.tan(sun.altitude_rad))

    # Bound the march by the DEM's relief: once the ray has risen by
    # more than (max_z - min_z), no upwind terrain can occlude any cell.
    finite_z = dem[~nan_mask]
    if finite_z.size == 0:
        return np.full((rows, cols), np.nan, dtype=np.float64)
    relief_m = float(finite_z.max() - finite_z.min())
    if rise_per_step_m > 0.0:
        max_steps_relief = int(np.ceil(relief_m / rise_per_step_m)) + 1
    else:
        # Sun on the horizon: cap by grid size instead.
        max_steps_relief = int(np.hypot(rows, cols)) + 1
    max_steps_grid = int(np.hypot(rows, cols)) + 1
    max_steps = max(1, min(max_steps_relief, max_steps_grid))

    # Bilinear sampling treats off-grid as -inf so escaping rays never
    # raise the horizon. NaN cells are also -inf so they don't shadow
    # downwind cells; their own mask value is restored to NaN at the end.
    import scipy.ndimage as ndi

    z = dem.astype(np.float64, copy=False)
    z_for_sampling = np.where(nan_mask, -np.inf, z)

    rows_idx, cols_idx = np.indices((rows, cols), dtype=np.float64)
    in_shadow = np.zeros((rows, cols), dtype=bool)

    for k in range(1, max_steps + 1):
        sample_row = rows_idx + k * drow_step
        sample_col = cols_idx + k * dcol_step
        coords = np.stack([sample_row, sample_col], axis=0)
        sampled = ndi.map_coordinates(
            z_for_sampling,
            coords,
            order=1,
            mode="constant",
            cval=-np.inf,
        )
        ray_height = z + k * rise_per_step_m
        # NaN cells: ray_height is NaN, comparison is False — no spurious
        # shadow. NaN value is restored after the loop.
        with np.errstate(invalid="ignore"):
            new_shadow = sampled > ray_height
        in_shadow |= new_shadow

    out = np.where(in_shadow, 0.0, 1.0)
    out[nan_mask] = np.nan
    return out
