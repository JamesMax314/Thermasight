"""Sun position from a single instant and a single (lat, lon).

A DEM tile is small enough that the sun position varies negligibly
across it (≪ 0.01° over a 20 km tile), so the whole pipeline uses one
``SolarPosition`` per run, evaluated at the tile centre. Per-cell
azimuth/altitude variation is not worth the cost.

The actual ephemeris is delegated to :func:`pvlib.solarposition.get_solarposition`,
which uses the SPA (Solar Position Algorithm) of Reda & Andreas (2004),
accurate to ~0.0003° from -2000 to 6000 CE. We do not reinvent it.

Conventions
-----------
* ``azimuth_rad`` is a **compass bearing** from north in radians,
  on ``[0, 2*pi)``: 0 = N, pi/2 = E, pi = S, 3*pi/2 = W. This matches
  :func:`thermal_model.terrain.aspect` and the hillshade input.
* ``altitude_rad`` is the angle above the horizon in radians, on
  ``[-pi/2, pi/2]``. Negative when the sun is below the horizon.
* ``zenith_rad`` is exposed as a derived property for convenience.

References
----------
Reda, I. & Andreas, A. (2004). Solar position algorithm for solar
radiation applications. Solar Energy, 76(5), 577-589.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SolarPosition:
    """Sun position at a single instant and location.

    Attributes
    ----------
    azimuth_rad : float
        Compass bearing of the sun from north, in radians, on
        ``[0, 2*pi)``.
    altitude_rad : float
        Angle of the sun above the horizon, in radians, on
        ``[-pi/2, pi/2]``. Negative when below the horizon.
    """

    azimuth_rad: float
    altitude_rad: float

    @property
    def zenith_rad(self) -> float:
        """Angle from zenith in radians, on ``[0, pi]``."""
        return math.pi / 2.0 - self.altitude_rad

    @property
    def is_above_horizon(self) -> bool:
        """``True`` if the sun is strictly above the horizon."""
        return self.altitude_rad > 0.0


def solar_position(
    when: datetime,
    latitude_deg: float,
    longitude_deg: float,
    elevation_m: float = 0.0,
) -> SolarPosition:
    """Apparent sun position at ``(latitude_deg, longitude_deg)`` at ``when``.

    Parameters
    ----------
    when : datetime.datetime
        The instant of interest. **Must be timezone-aware**: a naive
        datetime is rejected because solar position is meaningless
        without a defined time reference.
    latitude_deg : float
        Latitude in degrees, north-positive. Yorkshire Dales ≈ 54.2°.
    longitude_deg : float
        Longitude in degrees, east-positive. Yorkshire Dales ≈ -2.3°.
    elevation_m : float, default 0.0
        Site elevation in metres. Affects atmospheric refraction and so
        the *apparent* (refracted) altitude by a few hundredths of a
        degree near the horizon. Use the tile-centre elevation if
        available; ``0.0`` is fine for most uses.

    Returns
    -------
    SolarPosition
        Apparent (refraction-corrected) sun position with compass-bearing
        azimuth.

    Raises
    ------
    ValueError
        If ``when`` is timezone-naive.
    """
    if when.tzinfo is None:
        raise ValueError(
            "solar_position requires a timezone-aware datetime; got naive "
            f"{when!r}. Attach a tzinfo (e.g. ZoneInfo('Europe/London'))."
        )

    # pvlib needs a pandas DatetimeIndex; we use it only here at the
    # boundary and immediately convert back to plain floats. The project
    # otherwise avoids pandas (CLAUDE.md §4).
    import pandas as pd
    import pvlib

    times = pd.DatetimeIndex([when])
    result = pvlib.solarposition.get_solarposition(
        times,
        latitude=latitude_deg,
        longitude=longitude_deg,
        altitude=elevation_m,
    )
    azimuth_deg = float(result["azimuth"].iloc[0])
    elevation_above_horizon_deg = float(result["apparent_elevation"].iloc[0])
    return SolarPosition(
        azimuth_rad=math.radians(azimuth_deg),
        altitude_rad=math.radians(elevation_above_horizon_deg),
    )
