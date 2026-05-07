"""Wind as a terrain tilt — Phase 3's principal correction.

Wind in the surface boundary layer sweeps warm air downwind. Warm air
accumulates on the lee side of ridges and spurs and is continually
displaced from windward faces. The model captures this by adding a
linear ramp to the (smoothed) DEM **before inversion**: the ramp is
highest in the wind-to direction, so once the surface is inverted
that direction becomes the *lowest* part of the inverted surface and
flow accumulates preferentially on the lee side of every feature.

See ``docs/MODEL.md`` §3 and ``docs/model_correction.md`` §4 for the
derivation and tuning guidance. The ramp added is

    delta(row, col) = k * |u| * (col_m * sin(theta) - row_m * cos(theta))

where ``theta`` is the wind-*to* bearing in radians (clockwise from
north), ``col_m = col * cell_size_m`` and ``row_m = row * cell_size_m``
in a north-up raster (row index increases southward).

Cardinal-direction sign-convention table (used by the unit tests):

    wind_from_deg  wind_to    ramp peak    convergence biases toward
    -------------  ---------  -----------  -------------------------
    0   (N)        S (180)    south        south (lee of N->S wind)
    180 (S)        N (0)      north        north
    270 (W)        E (90)     east         east
    90  (E)        W (270)    west         west
    225 (SW)       NE (45)    northeast    northeast
"""

from __future__ import annotations

import math

import numpy as np


def wind_tilt_ramp(
    dem: np.ndarray,
    cell_size_m: float,
    wind_from_deg: float,
    wind_speed_ms: float,
    k: float = 0.03,
) -> np.ndarray:
    """Return ``dem`` with a linear wind-direction ramp added.

    The ramp biases the inverted-DEM flow accumulation toward the lee
    side of features, which is where boundary-layer warm air actually
    pools. The tilt must be applied **before** inverting the DEM.

    Parameters
    ----------
    dem : np.ndarray
        2-D north-up DEM, metres, projected CRS (EPSG:27700 in this
        project). Must have a floating dtype; NaN is the nodata
        sentinel and propagates to the output.
    cell_size_m : float
        Cell size in metres. Must be positive. The ramp's per-metre
        slope is ``k * wind_speed_ms`` regardless of cell size, so the
        per-cell delta scales with ``cell_size_m`` (a coarser raster
        sees a larger height jump between neighbouring cells but the
        same overall surface tilt).
    wind_from_deg : float
        Meteorological wind direction in degrees (the direction the
        wind is blowing *from*). 0 = north, 90 = east, 180 = south,
        270 = west. Values outside ``[0, 360)`` are wrapped.
    wind_speed_ms : float
        Wind speed in metres per second. Zero gives a no-op (output
        equals input).
    k : float, default 0.03
        Tilt coefficient in s/m. ``k * wind_speed_ms`` is the
        dimensionless fractional slope added to the terrain. See
        ``docs/MODEL.md`` §3 for tuning ranges (0.01 light, 0.03
        moderate / default, 0.05 strong).

    Returns
    -------
    np.ndarray
        Tilted DEM, same shape and dtype as ``dem``. NaN cells in the
        input remain NaN in the output.

    Raises
    ------
    ValueError
        If ``dem`` is not 2-D, does not have a floating dtype, or if
        ``cell_size_m`` is not positive.
    """
    if dem.ndim != 2:
        raise ValueError(f"dem must be 2-D, got shape {dem.shape}")
    if not np.issubdtype(dem.dtype, np.floating):
        raise ValueError(f"dem must have a floating dtype, got {dem.dtype}")
    if cell_size_m <= 0:
        raise ValueError(f"cell_size_m must be positive, got {cell_size_m}")

    rows, cols = dem.shape
    wind_to_rad = math.radians((wind_from_deg + 180.0) % 360.0)
    sin_to = math.sin(wind_to_rad)
    cos_to = math.cos(wind_to_rad)

    col_m = (np.arange(cols, dtype=np.float64) * cell_size_m)[None, :]
    row_m = (np.arange(rows, dtype=np.float64) * cell_size_m)[:, None]
    delta = k * wind_speed_ms * (col_m * sin_to - row_m * cos_to)

    out: np.ndarray = (dem + delta).astype(dem.dtype, copy=False)
    return out
