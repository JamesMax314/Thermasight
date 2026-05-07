"""Lambertian hillshade for diagnostic visualization.

This is a cosmetic shader: dot the unit surface normal with a fixed
sun direction and clip to ``[0, 1]``. It does not model cast shadows
or atmospheric extinction; that physics belongs in
``thermal_model.solar`` and is a Phase 2 concern.

The default sun position (azimuth 315°, altitude 45°) follows the
cartographic convention: a NW sun makes north- and east-facing slopes
brighter, which our eyes read as relief without flipping our innate
shape-from-shading instincts (the "crater illusion" in DEMs lit from
below).
"""

from __future__ import annotations

import numpy as np


def hillshade(
    dem: np.ndarray,
    cell_size_m: float,
    *,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Lambertian hillshade in ``[0, 1]``.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array in metres with NaN nodata.
    cell_size_m : float
        Square cell size in metres.
    azimuth_deg : float, default 315.0
        Sun azimuth in degrees clockwise from north (compass bearing).
        315 (NW) is the cartographic default.
    altitude_deg : float, default 45.0
        Sun altitude above the horizon in degrees, in ``[0, 90]``.
    z_factor : float, default 1.0
        Vertical exaggeration. Useful for low-relief terrain on a tall
        figure; physical analyses should leave this at ``1.0``.

    Returns
    -------
    np.ndarray
        Lambertian shading on ``[0, 1]``, same shape as ``dem``. NaN
        propagates from the input through the gradient stencil, so
        cells immediately neighbouring a NaN will themselves be NaN.

    Notes
    -----
    Negative dot products (surfaces facing away from the sun) are
    clipped to zero — this is self-shadowing in the Lambertian sense,
    not the cast shadow of an upstream ridge. For cast shadows use the
    Phase 2 solar module.
    """
    if dem.ndim != 2:
        raise ValueError(f"DEM must be 2-D, got shape {dem.shape}")
    if cell_size_m <= 0:
        raise ValueError(f"cell_size_m must be positive, got {cell_size_m}")
    if not 0.0 <= altitude_deg <= 90.0:
        raise ValueError(f"altitude_deg must be in [0, 90], got {altitude_deg}")

    # Sun direction in (x=east, y=north, z=up). Compass azimuth runs
    # clockwise from north; convert to math angle (CCW from east).
    math_az_rad = np.deg2rad(90.0 - azimuth_deg)
    alt_rad = np.deg2rad(altitude_deg)
    sun_x = float(np.cos(alt_rad) * np.cos(math_az_rad))
    sun_y = float(np.cos(alt_rad) * np.sin(math_az_rad))
    sun_z = float(np.sin(alt_rad))

    # Central differences. np.gradient returns gradients in axis order:
    # axis 0 is rows (south-positive), axis 1 is cols (east-positive).
    dz_drow, dz_dcol = np.gradient(dem.astype(np.float64), cell_size_m)
    dz_dx = z_factor * dz_dcol
    dz_dy = -z_factor * dz_drow  # rows increase southward; flip for north-up

    # Surface normal (-dz_dx, -dz_dy, 1), normalised.
    norm_mag = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy + 1.0)
    nx = -dz_dx / norm_mag
    ny = -dz_dy / norm_mag
    nz = 1.0 / norm_mag

    shade = nx * sun_x + ny * sun_y + nz * sun_z
    clipped = np.clip(shade, 0.0, 1.0)
    # np.gradient computes a NaN cell's derivative from its neighbours,
    # so without this the NaN cell itself would end up finite.
    out: np.ndarray = np.where(np.isnan(dem), np.nan, clipped)
    return out
