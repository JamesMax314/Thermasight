"""Clear-sky irradiance and its projection onto sloping terrain.

Two stages:

1. :func:`clear_sky_irradiance` — given a single instant and location,
   return the three standard clear-sky components for a horizontal
   surface: GHI (global horizontal), DNI (direct normal), and DHI
   (diffuse horizontal), all in W/m². Uses the Ineichen-Perez clear-sky
   model (Ineichen & Perez 2002) via ``pvlib``.

2. :func:`slope_irradiance` — project the clear-sky DNI/DHI onto every
   cell of the DEM given its slope and aspect, returning beam and
   diffuse components separately so the cast-shadow mask (added in a
   later step) can attenuate beam without touching diffuse.

The diffuse model here is **isotropic** (Liu & Jordan 1960): a tilted
surface sees ``DHI * (1 + cos(slope)) / 2``. Anisotropic models
(Hay-Davies, Perez) capture circumsolar and horizon-brightening and
will matter for low-sun, clear-sky conditions; they are an upgrade for
later. Ground-reflected irradiance is omitted for now (Phase 4 will add
it alongside the land-cover albedo lookup).

References
----------
Ineichen, P. & Perez, R. (2002). A new airmass independent formulation
for the Linke turbidity coefficient. Solar Energy, 73(3), 151-157.

Liu, B.Y.H. & Jordan, R.C. (1960). The interrelationship and
characteristic distribution of direct, diffuse and total solar
radiation. Solar Energy, 4(3), 1-19.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from thermal_model.solar.position import SolarPosition


@dataclass(frozen=True)
class ClearSkyIrradiance:
    """Clear-sky irradiance components for a horizontal surface.

    All values are in W/m². Below-horizon sun returns zeros for all
    three components.

    Attributes
    ----------
    ghi_wm2 : float
        Global horizontal irradiance.
    dni_wm2 : float
        Direct normal irradiance (perpendicular to the sun's rays).
    dhi_wm2 : float
        Diffuse horizontal irradiance.
    """

    ghi_wm2: float
    dni_wm2: float
    dhi_wm2: float


@dataclass(frozen=True)
class SlopeIrradiance:
    """Per-cell beam and diffuse irradiance on the DEM surface.

    Both arrays have the same shape as the input slope/aspect arrays,
    in W/m². NaN propagates from invalid slope/aspect cells. The cast
    shadow of upstream terrain is **not** applied — multiply ``beam_wm2``
    by the binary cast-shadow mask before summing for the heating field.

    Attributes
    ----------
    beam_wm2 : np.ndarray
        Direct-beam irradiance on the tilted cell, clipped at zero on
        cells whose surface normal points away from the sun
        (self-shading).
    diffuse_wm2 : np.ndarray
        Isotropic-sky diffuse irradiance on the tilted cell.
    """

    beam_wm2: np.ndarray
    diffuse_wm2: np.ndarray

    @property
    def total_wm2(self) -> np.ndarray:
        """Sum of beam and diffuse, ignoring cast shadows."""
        out: np.ndarray = self.beam_wm2 + self.diffuse_wm2
        return out


def clear_sky_irradiance(
    when: datetime,
    latitude_deg: float,
    longitude_deg: float,
    elevation_m: float = 0.0,
    *,
    linke_turbidity: float = 3.0,
) -> ClearSkyIrradiance:
    """Clear-sky GHI/DNI/DHI on a horizontal surface, in W/m².

    Parameters
    ----------
    when : datetime.datetime
        Instant of interest. Must be timezone-aware.
    latitude_deg, longitude_deg : float
        Site coordinates in degrees (N-positive, E-positive).
    elevation_m : float, default 0.0
        Site elevation in metres. Affects airmass and so the Ineichen
        clear-sky output. Use the tile-mean elevation when known.
    linke_turbidity : float, default 3.0
        Linke turbidity coefficient: a dimensionless climate parameter
        capturing aerosol load and water vapour above the site.
        Typical values: 2 (very clean cold air), 3 (temperate clear
        day, the UK summer default used here), 4–5 (hazy/humid),
        6+ (urban/dusty). Pinning a default avoids the pvlib
        Linke-turbidity lookup table and keeps the function offline.

    Returns
    -------
    ClearSkyIrradiance
        GHI / DNI / DHI on a horizontal surface. All zero when the sun
        is below the horizon.

    Raises
    ------
    ValueError
        If ``when`` is timezone-naive.
    """
    if when.tzinfo is None:
        raise ValueError(
            "clear_sky_irradiance requires a timezone-aware datetime; "
            f"got naive {when!r}."
        )

    import pandas as pd
    import pvlib

    times = pd.DatetimeIndex([when])
    location = pvlib.location.Location(
        latitude=latitude_deg,
        longitude=longitude_deg,
        altitude=elevation_m,
    )
    cs = location.get_clearsky(
        times,
        model="ineichen",
        linke_turbidity=linke_turbidity,
    )
    return ClearSkyIrradiance(
        ghi_wm2=float(cs["ghi"].iloc[0]),
        dni_wm2=float(cs["dni"].iloc[0]),
        dhi_wm2=float(cs["dhi"].iloc[0]),
    )


def slope_irradiance(
    slope_rad: np.ndarray,
    aspect_rad: np.ndarray,
    sun: SolarPosition,
    irradiance: ClearSkyIrradiance,
) -> SlopeIrradiance:
    """Project clear-sky DNI/DHI onto a sloping surface, per cell.

    The beam component on a cell with slope ``β`` and aspect (downslope
    compass bearing) ``A`` is

        beam = max(DNI * cos(theta_i), 0)
        cos(theta_i) = cos(zenith) cos(β) + sin(zenith) sin(β) cos(A_sun − A)

    where ``A_sun`` is the sun's compass-bearing azimuth. The clip at
    zero is self-shading: a slope whose outward normal points away from
    the sun receives no direct beam.

    The diffuse component uses the Liu–Jordan isotropic-sky model:

        diffuse = DHI * (1 + cos(β)) / 2

    Cast shadows from upstream terrain are not modelled here; that mask
    is applied separately to ``beam_wm2`` in the heating-field step.

    Parameters
    ----------
    slope_rad : np.ndarray
        Surface slope in radians, ``[0, pi/2]``. NaN propagates.
    aspect_rad : np.ndarray
        Downslope compass bearing in radians, ``[0, 2*pi)``. NaN on
        true flat cells; tolerated as long as the corresponding
        ``slope_rad`` is exactly zero (the aspect contribution vanishes
        when the surface is horizontal).
    sun : SolarPosition
        Sun position for this run, from :func:`solar_position`.
    irradiance : ClearSkyIrradiance
        Clear-sky components for a horizontal surface, from
        :func:`clear_sky_irradiance`.

    Returns
    -------
    SlopeIrradiance
        Beam and diffuse W/m² arrays, same shape as ``slope_rad``.

    Notes
    -----
    Matches :func:`thermal_model.terrain.aspect`'s convention for
    ``aspect_rad``: a compass bearing in radians, north = 0, clockwise.
    The sun azimuth in :class:`SolarPosition` uses the same convention,
    so ``A_sun − A`` is the relative angle between the slope's facing
    direction and the sun's bearing.

    For a horizontal surface (``slope_rad == 0``) the formula reduces
    to ``DNI * cos(zenith) + DHI``, which equals GHI by construction.
    """
    if slope_rad.shape != aspect_rad.shape:
        raise ValueError(
            f"slope_rad shape {slope_rad.shape} does not match "
            f"aspect_rad shape {aspect_rad.shape}"
        )

    # Aspect is undefined on true flats (slope == 0). The aspect term in
    # cos(theta_i) is multiplied by sin(slope)=0 there, so the value
    # cancels — but NaN * 0 is still NaN in IEEE arithmetic, so swap in
    # a finite stand-in for those cells before the multiply.
    flat = (slope_rad == 0.0) & np.isnan(aspect_rad)
    aspect_safe = np.where(flat, 0.0, aspect_rad)

    cos_zenith = math.cos(sun.zenith_rad)
    sin_zenith = math.sin(sun.zenith_rad)
    cos_slope = np.cos(slope_rad)
    sin_slope = np.sin(slope_rad)
    cos_relative = np.cos(sun.azimuth_rad - aspect_safe)

    cos_aoi = cos_zenith * cos_slope + sin_zenith * sin_slope * cos_relative

    beam = np.maximum(irradiance.dni_wm2 * cos_aoi, 0.0)
    # Below-horizon sun: pvlib returns DNI = 0, so beam is already zero.
    # Belt-and-braces clip in case a caller hands in a hand-built
    # ClearSkyIrradiance with non-zero DNI and a below-horizon sun.
    if not sun.is_above_horizon:
        beam = np.zeros_like(beam)

    diffuse = irradiance.dhi_wm2 * 0.5 * (1.0 + cos_slope)

    # Propagate NaN from slope (edges, holes) into both outputs.
    nan_mask = np.isnan(slope_rad)
    beam = np.where(nan_mask, np.nan, beam)
    diffuse = np.where(nan_mask, np.nan, diffuse)

    return SlopeIrradiance(beam_wm2=beam, diffuse_wm2=diffuse)
