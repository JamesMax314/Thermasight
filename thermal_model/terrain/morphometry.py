"""Terrain morphometrics from a gridded DEM.

Slope and aspect use Horn's (1981) third-order finite-difference
operator on a 3x3 window. Profile curvature uses the Zevenbergen &
Thorne (1987) quadratic-surface fit on the same window.

All operators take a 2-D ``np.ndarray`` of elevations in metres with
``np.nan`` as the nodata sentinel and a square cell size in metres,
and return an array of the same shape. Any 3x3 window that touches the
array edge or contains a NaN produces NaN at its centre.

Conventions
-----------
* ``slope`` is returned in radians, in [0, pi/2].
* ``aspect`` is the compass bearing of the **downslope** direction in
  radians: 0 = north, pi/2 = east, pi = south, 3*pi/2 = west.
  Cells with zero gradient (true flats) return NaN.
* ``profile_curvature`` follows Zevenbergen & Thorne's sign convention:
  **positive = convex** (ridges, slope breaks where downhill flow
  decelerates); negative = concave (hollows). Units are 1/metre.

References
----------
Horn, B.K.P. (1981). Hill shading and the reflectance map. Proc. IEEE,
69(1), 14-47.

Zevenbergen, L.W. & Thorne, C.R. (1987). Quantitative analysis of land
surface topography. Earth Surf. Process. Landforms, 12(1), 47-56.
"""

from __future__ import annotations

import numpy as np


def _validate_window(dem: np.ndarray, cell_size_m: float) -> None:
    if dem.ndim != 2:
        raise ValueError(f"DEM must be 2-D, got shape {dem.shape}")
    if cell_size_m <= 0:
        raise ValueError(f"cell_size_m must be positive, got {cell_size_m}")
    if dem.shape[0] < 3 or dem.shape[1] < 3:
        raise ValueError(f"DEM must be at least 3x3 for a 3x3 stencil, got {dem.shape}")


def _horn_gradients(
    dem: np.ndarray, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """East-positive and north-positive gradients via Horn (1981).

    The centre cell does not appear in Horn's formula but is included in
    the validity mask: any NaN anywhere in the 3x3 window invalidates
    the gradient at the centre, so slope/aspect/curvature stay
    consistent on holed inputs.
    """
    _validate_window(dem, cell_size_m)
    z = dem
    z_nw, z_n, z_ne = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    z_w, z_c, z_e = z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:]
    z_sw, z_s, z_se = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]

    dz_dx = ((z_ne + 2 * z_e + z_se) - (z_nw + 2 * z_w + z_sw)) / (8 * cell_size_m)
    # Raster row index increases southward, so flip sign for north-positive.
    dz_drow = ((z_sw + 2 * z_s + z_se) - (z_nw + 2 * z_n + z_ne)) / (8 * cell_size_m)
    dz_dy = -dz_drow

    invalid = (
        np.isnan(z_nw)
        | np.isnan(z_n)
        | np.isnan(z_ne)
        | np.isnan(z_w)
        | np.isnan(z_c)
        | np.isnan(z_e)
        | np.isnan(z_sw)
        | np.isnan(z_s)
        | np.isnan(z_se)
    )
    dz_dx = np.where(invalid, np.nan, dz_dx)
    dz_dy = np.where(invalid, np.nan, dz_dy)

    out_x = np.full(z.shape, np.nan, dtype=np.float64)
    out_y = np.full(z.shape, np.nan, dtype=np.float64)
    out_x[1:-1, 1:-1] = dz_dx
    out_y[1:-1, 1:-1] = dz_dy
    return out_x, out_y


def slope(dem: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Slope in radians from Horn's operator.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array in metres with NaN nodata.
    cell_size_m : float
        Square cell size in metres.

    Returns
    -------
    np.ndarray
        Slope in radians, same shape as ``dem``. NaN on the array edge
        and on any cell whose 3x3 window contains a NaN.
    """
    dz_dx, dz_dy = _horn_gradients(dem, cell_size_m)
    out: np.ndarray = np.arctan(np.hypot(dz_dx, dz_dy))
    return out


def aspect(dem: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Aspect of the downslope direction, in radians, from Horn's operator.

    The bearing is measured clockwise from north: 0 = N, pi/2 = E,
    pi = S, 3*pi/2 = W. Cells with zero horizontal gradient (true
    flats) return NaN, since the downslope direction is undefined.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array in metres with NaN nodata.
    cell_size_m : float
        Square cell size in metres.

    Returns
    -------
    np.ndarray
        Aspect in radians on [0, 2*pi), NaN on edges, NaN windows, and
        flats.
    """
    dz_dx, dz_dy = _horn_gradients(dem, cell_size_m)
    flat = (dz_dx == 0) & (dz_dy == 0)
    bearing = np.arctan2(-dz_dx, -dz_dy)
    bearing = np.where(bearing < 0, bearing + 2 * np.pi, bearing)
    return np.where(flat, np.nan, bearing)


def profile_curvature(dem: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Profile curvature via Zevenbergen & Thorne (1987).

    Profile curvature is the second derivative of the surface in the
    direction of steepest descent. Positive values are convex (ridges,
    slope breaks); negative values are concave (hollows). Units are
    1/metre.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array in metres with NaN nodata.
    cell_size_m : float
        Square cell size in metres.

    Returns
    -------
    np.ndarray
        Profile curvature in 1/metre, same shape as ``dem``. NaN on the
        array edge, on cells whose 3x3 window contains a NaN, and on
        flat cells where the curvature direction is undefined.
    """
    _validate_window(dem, cell_size_m)
    z = dem
    cell = float(cell_size_m)
    cell_sq = cell * cell

    z1, z2, z3 = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    z4, z5, z6 = z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:]
    z7, z8, z9 = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]

    fxx = ((z4 + z6) / 2.0 - z5) / cell_sq
    fyy = ((z2 + z8) / 2.0 - z5) / cell_sq
    fxy = (-z1 + z3 + z7 - z9) / (4.0 * cell_sq)
    fx = (-z4 + z6) / (2.0 * cell)
    fy = (z2 - z8) / (2.0 * cell)

    p_sq = fx * fx + fy * fy
    denom = p_sq * np.power(1.0 + p_sq, 1.5)

    with np.errstate(divide="ignore", invalid="ignore"):
        kprof = -(fxx * fx * fx + fyy * fy * fy + fxy * fx * fy) / denom

    kprof = np.where(p_sq == 0, np.nan, kprof)

    out = np.full(z.shape, np.nan, dtype=np.float64)
    out[1:-1, 1:-1] = kprof
    return out
