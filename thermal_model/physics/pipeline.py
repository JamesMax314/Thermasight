"""End-to-end Phase 3 trigger-prediction pipeline.

Orchestrates the §6 block of ``docs/model_correction.md``: smooth →
wind tilt → heating from the *raw* DEM → invert + pit-fill + D∞
accumulation **weighted by heating** → multiply by normalised
positive profile curvature → multiply by a minimum-slope mask. The
output is the trigger-potential raster on ``[0, 1]``.

There is **no** separate "combine heating and convergence" step.
Heating enters the routing as the per-cell weight on the D∞ flow
accumulation, so the integration is intrinsic. See
``docs/MODEL.md`` §5 and ``docs/model_correction.md`` §3 for the
justification (a *local* multiplier wrongly zeros shadowed convergent
points fed by sunny upwind faces; weighted accumulation routes the
upstream warmth to them and gets the right answer).

Which DEM each step uses (matches the table in
``docs/model_correction.md`` §6):

* Gaussian smooth: raw DEM.
* Wind tilt: smoothed DEM.
* Slope, aspect, profile curvature, slope mask: raw DEM
  (real geometry drives shadows, gradients, and detachment).
* Cast-shadow mask: raw DEM.
* Inversion + pit-fill + weighted D∞ accumulation: smoothed +
  tilted DEM, with the heating field as ``weights=``.

This module has no I/O. The CLI ``run`` subcommand is the wrapper
that reads a DEM and writes GeoTIFF / KMZ outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy import ndimage, stats

from thermal_model.physics.flow import flow_accumulation
from thermal_model.physics.heating import DEFAULT_ABSORPTIVITY, heating_field
from thermal_model.physics.hydrology import fill_pits
from thermal_model.physics.hydrology import resolve_flats as _resolve_flats_fn
from thermal_model.physics.wind_tilt import wind_tilt_ramp
from thermal_model.solar.irradiance import clear_sky_irradiance, slope_irradiance
from thermal_model.solar.position import solar_position
from thermal_model.solar.shadow import cast_shadow_mask
from thermal_model.terrain.morphometry import aspect, profile_curvature, slope


@dataclass(frozen=True)
class RunResult:
    """Output of :func:`run_model`.

    The primary deliverable is :attr:`trigger_potential`. The other
    fields are diagnostics — exposed so callers can plot, export, or
    sensitivity-test individual stages without re-running the whole
    pipeline.

    Attributes
    ----------
    trigger_potential : np.ndarray
        Float64, ``[0, 1]``, NaN-passthrough. The product of normalised
        weighted-convergence, normalised positive profile curvature,
        and the minimum-slope gate. The headline raster.
    weighted_convergence : np.ndarray
        Float64, raw upstream W/m² × cell-count from the heating-
        weighted D∞ flow accumulation. Diagnostic; not normalised.
    heating_wm2 : np.ndarray
        Float64, W/m². The per-cell weight passed to the flow
        accumulation. NaN at finite-DEM edge cells where the 3×3
        stencil could not resolve slope/aspect; those entries are
        substituted with ``0.0`` before being passed as ``weights=``
        (a finite-DEM cell with no informed heating estimate
        contributes nothing to the routing, which is the conservative
        choice).
    smoothed_dem_m : np.ndarray
        Float64, metres. The Gaussian-smoothed input DEM.
    tilted_dem_m : np.ndarray
        Float64, metres. The smoothed DEM with the wind-tilt ramp
        added (input to the inversion).
    profile_curvature : np.ndarray
        Float64, 1/m. From the *raw* DEM. Positive = convex.
    slope_rad : np.ndarray
        Float64, radians. From the *raw* DEM.
    slope_mask : np.ndarray
        Bool, ``True`` where ``slope_rad > min_slope_rad``.
    """

    trigger_potential: np.ndarray
    weighted_convergence: np.ndarray
    heating_wm2: np.ndarray
    smoothed_dem_m: np.ndarray
    tilted_dem_m: np.ndarray
    profile_curvature: np.ndarray
    slope_rad: np.ndarray
    slope_mask: np.ndarray


def _gaussian_smooth_nan(dem: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Gaussian-smooth a DEM with NaN passthrough.

    ``scipy.ndimage.gaussian_filter`` does not honour NaN, so we stamp
    NaN cells with the finite mean before convolving and restore NaN
    afterwards. The bias near NaN edges is bounded by the kernel scale
    — at the default σ ≈ 10 m that is far below any real terrain
    feature. Same recipe as ``viz.diagnostics.plot_convergence``.
    """
    if sigma_cells <= 0:
        return dem.astype(np.float64, copy=True)
    nan_mask = np.isnan(dem)
    if not nan_mask.any():
        out = ndimage.gaussian_filter(
            dem.astype(np.float64, copy=False), sigma=sigma_cells
        )
        return np.asarray(out, dtype=np.float64)
    stamped = np.where(nan_mask, float(np.nanmean(dem)), dem).astype(np.float64)
    smoothed = ndimage.gaussian_filter(stamped, sigma=sigma_cells)
    smoothed[nan_mask] = np.nan
    return np.asarray(smoothed, dtype=np.float64)


def _rank_normalise(field: np.ndarray) -> np.ndarray:
    """Percentile-rank normalisation of strictly-positive finite values.

    Each strictly-positive finite cell is replaced by its rank divided
    by the count of strictly-positive finite cells, so the output is
    spread uniformly over ``(0, 1]``. Non-positive finite cells map to
    ``0``; NaN cells stay NaN.

    Why rank rather than the previous q99-clip:

    * The trigger raster is the product of two normalised factors
      (weighted convergence × positive profile curvature). With q99
      clipping each factor only reaches ~1 on its top 1 % of cells, so
      the product reaches usable values only where both factors are
      simultaneously near 1 — vanishingly rare. The Mallerstang
      preview at 5 m showed this collapse: convergence had 7 decades
      of dynamic range but the trigger raster was almost entirely
      zero.
    * Rank normalisation is order-preserving (so the relative
      brightness of cells is unchanged) but spreads values uniformly
      over ``[0, 1]``, so the product spans the unit interval too.
    * Rank is robust to outliers (a single LIDAR-speckle curvature
      spike no longer dominates the normalisation scale).
    """
    out = np.zeros(field.shape, dtype=np.float64)
    out[np.isnan(field)] = np.nan
    positive = np.isfinite(field) & (field > 0)
    if not positive.any():
        return out
    values = field[positive]
    # ``method='average'`` returns ranks in [1, N]; divide by N so the
    # top cell maps to exactly 1.0 and the bottom-positive cell to
    # ~1/N.
    ranks = stats.rankdata(values, method="average") / float(values.size)
    out[positive] = ranks
    return out


def run_model(
    dem: np.ndarray,
    cell_size_m: float,
    when: datetime,
    latitude_deg: float,
    longitude_deg: float,
    *,
    wind_from_deg: float,
    wind_speed_ms: float,
    wind_tilt_k: float = 0.03,
    smoothing_sigma_m: float = 10.0,
    min_slope_deg: float = 2.5,
    absorptivity: float | np.ndarray = DEFAULT_ABSORPTIVITY,
    elevation_m: float | None = None,
    linke_turbidity: float = 3.0,
    pit_fill_epsilon: float = 1.0e-3,
    resolve_flats: bool = True,
) -> RunResult:
    """Run the Phase 3 trigger-prediction pipeline on ``dem``.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array in metres with NaN nodata. Real terrain;
        do not pass a pre-smoothed or pre-tilted array.
    cell_size_m : float
        Square cell size in metres. Must be positive. The CRS is
        assumed projected with metric units (EPSG:27700 in this
        project — the caller is responsible for CRS handling, which
        lives in :mod:`thermal_model.io`).
    when : datetime.datetime
        Timezone-aware instant for the sun position and irradiance.
    latitude_deg, longitude_deg : float
        Site coordinates in degrees (N-positive, E-positive). For a
        georeferenced DEM, derive these from the centre via
        :class:`pyproj.Transformer`.
    wind_from_deg : float
        Meteorological wind direction in degrees (the direction the
        wind is blowing *from*).
    wind_speed_ms : float
        Wind speed in m/s. Zero gives a no-tilt baseline.
    wind_tilt_k : float, default 0.03
        Tilt coefficient in s/m. ``k * wind_speed_ms`` is the
        dimensionless fractional slope added to the terrain. See
        ``docs/MODEL.md`` §3 for tuning ranges.
    smoothing_sigma_m : float, default 10.0
        Gaussian smoothing scale in metres applied to the raw DEM
        before wind tilt and flow routing. ``CLAUDE.md`` §2 gives the
        10–25 m envelope; 10 m is the lower bound and the project
        default. ``0`` disables smoothing.
    min_slope_deg : float, default 2.5
        Minimum slope (degrees) for a cell to count as a candidate
        trigger. Kills flat-summit and valley-floor artefacts.
    absorptivity : float or np.ndarray, default 0.80
        Shortwave absorptivity ``alpha = 1 - albedo``. Phase 2 ships a
        scalar default; Phase 4 will switch in a per-cell array driven
        by land cover.
    elevation_m : float, optional
        Site elevation in metres for the clear-sky model. Defaults to
        the median of finite DEM cells.
    linke_turbidity : float, default 3.0
        Linke turbidity for the Ineichen-Perez clear-sky model.
    pit_fill_epsilon : float, default 1e-3
        Per-step elevation bump for the priority-flood pit-fill on
        the inverted, tilted DEM. Any positive value triggers the
        monotonic-fill behaviour on the richdem backend; the numpy
        fallback uses the literal value as the bump. A monotonic
        fill is preferred here because the wind-tilt ramp can
        introduce tiny artificial flats at the windward edge that
        a plain (epsilon=0) fill would leave unrouted.
    resolve_flats : bool, default True
        Whether to run :func:`thermal_model.physics.resolve_flats`
        between pit-fill and flow accumulation. The pit-filled
        inverted-and-tilted DEM contains formerly-flat regions
        (raised plateaus, e.g. summit tops) whose flow direction
        otherwise inherits the priority-flood BFS chamfer-distance
        gradient, producing characteristic parallel-streak
        artefacts in the convergence map. Garbrecht-Martz flat
        resolution (richdem backend) replaces those with a
        physically defensible gradient. Slow on large mosaics
        (~7 min on 75 M cells) but a one-off cost per run; turn
        off via ``resolve_flats=False`` for fast iteration.

    Notes
    -----
    The trigger raster is composed by **rank-normalising** the
    weighted convergence and the strictly-positive profile curvature
    separately, then multiplying by the slope mask. Rank
    normalisation spreads each factor uniformly over ``[0, 1]`` so
    the multiplicative product retains usable dynamic range.
    Previously the pipeline used q99 clipping, which collapsed the
    product to near-zero everywhere because each factor only
    approached ``1`` on its top 1 %.

    Returns
    -------
    RunResult
        The trigger-potential raster plus all diagnostic
        intermediates. See :class:`RunResult`.

    Raises
    ------
    ValueError
        For invalid inputs (non-2-D DEM, non-positive cell size,
        non-positive smoothing/quantile, etc.).
    """
    if dem.ndim != 2:
        raise ValueError(f"dem must be 2-D, got shape {dem.shape}")
    if cell_size_m <= 0:
        raise ValueError(f"cell_size_m must be positive, got {cell_size_m}")
    if smoothing_sigma_m < 0:
        raise ValueError(
            f"smoothing_sigma_m must be non-negative, got {smoothing_sigma_m}"
        )
    if pit_fill_epsilon < 0:
        raise ValueError(
            f"pit_fill_epsilon must be non-negative, got {pit_fill_epsilon}"
        )

    dem64 = dem.astype(np.float64, copy=False)
    nan_mask = np.isnan(dem64)

    if elevation_m is None:
        finite = dem64[~nan_mask]
        if finite.size == 0:
            raise ValueError("dem contains no finite cells; cannot derive elevation_m")
        elevation_m = float(np.median(finite))

    sigma_cells = smoothing_sigma_m / float(cell_size_m)
    smoothed = _gaussian_smooth_nan(dem64, sigma_cells)
    tilted = wind_tilt_ramp(
        smoothed, cell_size_m, wind_from_deg, wind_speed_ms, k=wind_tilt_k
    )

    slope_rad = slope(dem64, cell_size_m)
    aspect_rad = aspect(dem64, cell_size_m)
    kprof = profile_curvature(dem64, cell_size_m)

    sun = solar_position(when, latitude_deg, longitude_deg, elevation_m=elevation_m)
    cs = clear_sky_irradiance(
        when,
        latitude_deg,
        longitude_deg,
        elevation_m=elevation_m,
        linke_turbidity=linke_turbidity,
    )
    irr = slope_irradiance(slope_rad, aspect_rad, sun, cs)
    shadow = cast_shadow_mask(dem64, cell_size_m, sun)
    heating = heating_field(irr, shadow, absorptivity=absorptivity)

    # The flow-accumulation weights contract requires finite weights at
    # every finite-DEM cell. Slope/aspect/curvature return NaN on the
    # 1-cell boundary and on cells whose 3x3 window touches a NaN, so
    # heating inherits NaN there. Substitute 0.0 at those finite-DEM
    # cells: an edge cell with no informed estimate injects zero
    # thermal energy into the routing, which is the conservative
    # answer for cells we cannot evaluate. NaN-DEM cells stay NaN.
    weights = np.where(np.isnan(heating) & ~nan_mask, 0.0, heating)

    # Inversion uses the *finite* nanmax of the tilted surface so the
    # inverted surface is non-negative on its finite domain. A naive
    # ``np.max`` on an array with NaN would return NaN.
    finite_tilted = tilted[~nan_mask]
    tilted_max = float(np.nanmax(finite_tilted)) if finite_tilted.size else 0.0
    inverted_tilted = tilted_max - tilted

    inverted_filled = fill_pits(inverted_tilted, epsilon=pit_fill_epsilon)
    if resolve_flats:
        inverted_filled = _resolve_flats_fn(inverted_filled)
    weighted_conv = flow_accumulation(inverted_filled, cell_size_m, weights=weights)

    wc_norm = _rank_normalise(weighted_conv)
    curv_norm = _rank_normalise(np.where(kprof > 0, kprof, 0.0))

    min_slope_rad = math.radians(min_slope_deg)
    slope_mask = np.where(np.isnan(slope_rad), False, slope_rad > min_slope_rad)
    slope_mask = slope_mask.astype(bool, copy=False)

    trigger = wc_norm * curv_norm * slope_mask.astype(np.float64, copy=False)
    # Restore NaN where the raw DEM was NaN — the multiplications
    # above coerce NaN factors but the slope-mask cast can flip
    # NaN-derived False back to a valid 0.0. Make the nodata
    # convention explicit.
    trigger = np.where(nan_mask, np.nan, trigger)

    return RunResult(
        trigger_potential=trigger.astype(np.float64, copy=False),
        weighted_convergence=weighted_conv.astype(np.float64, copy=False),
        heating_wm2=heating.astype(np.float64, copy=False),
        smoothed_dem_m=smoothed,
        tilted_dem_m=tilted.astype(np.float64, copy=False),
        profile_curvature=kprof.astype(np.float64, copy=False),
        slope_rad=slope_rad.astype(np.float64, copy=False),
        slope_mask=slope_mask,
    )
