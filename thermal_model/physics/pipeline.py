"""End-to-end Phase 3.1 trigger-prediction pipeline.

Orchestrates the §11 block of ``docs/MODEL.md``: smooth → wind tilt
→ heating from the *raw* DEM → invert + pit-fill + leaky-bucket
weighted D∞ accumulation. The primary output is the per-cell ``leak``
raster (W/m² of time-averaged trigger-release rate); a secondary
``cycle_period_s`` raster gives the period between successive
releases at each cell.

The leaky kernel replaces the Phase 3 post-hoc multiply
(``rank_norm(weighted_convergence) × rank_norm(κ⁺) × slope_mask``).
Each cell consumes a curvature/slope-dependent fraction
``(1 − f_drain)`` of its through-flow as trigger output and forwards
only ``f_drain`` of it onward. Energy is conserved along the path:
``Σ leak + residual_at_sinks_total ≡ Σ heating``. There is no
separate "combine heating and convergence" step — heating enters
the routing as the per-cell weight, and the integration is intrinsic.
See ``docs/MODEL.md`` §11 for the derivation.

Backward-compatible display: ``RunResult.trigger_potential`` is
``rank_normalise(leak)``, so existing CLI / viz / KMZ consumers
that expected a ``[0, 1]`` raster keep working.

Which DEM each step uses:

* Gaussian smooth: raw DEM.
* Wind tilt: smoothed DEM.
* Slope, aspect, profile curvature (for irradiance and the
  RunResult diagnostics): raw DEM (real geometry drives shadows
  and the per-cell DNI projection).
* Slope and profile curvature (for the leaky shape functions
  ``f_drain`` and ``q_storage``): a separately smoothed copy of
  the raw DEM with σ = ``curvature_smoothing_sigma_m`` (default
  10 m). Carries forward the ``MODEL.md`` §6 ¶282–284
  prescription — single-cell LIDAR κ⁺ outliers would otherwise
  saturate ``sat(κ⁺/κ_ref)`` and pull ``f_drain`` to its
  ``f_min`` floor, producing a per-cell speckle on the leak
  raster.
* Cast-shadow mask: raw DEM.
* Inversion + pit-fill + leaky weighted accumulation: smoothed +
  tilted DEM, with the heating field as the per-cell weight.

This module has no I/O. The CLI ``run`` subcommand is the wrapper
that reads a DEM and writes GeoTIFF / KMZ outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy import ndimage, stats

from thermal_model.physics.heating import DEFAULT_ABSORPTIVITY, heating_field
from thermal_model.physics.hydrology import fill_pits
from thermal_model.physics.hydrology import resolve_flats as _resolve_flats_fn
from thermal_model.physics.leaky_accum import (
    F_MAX_DEFAULT,
    F_MIN_DEFAULT,
    f_drain_field,
    leaky_weighted_accumulation,
    q_storage_field,
)
from thermal_model.physics.wind_tilt import wind_tilt_ramp
from thermal_model.solar.irradiance import clear_sky_irradiance, slope_irradiance
from thermal_model.solar.position import solar_position
from thermal_model.solar.shadow import cast_shadow_mask
from thermal_model.terrain.morphometry import aspect, profile_curvature, slope


@dataclass(frozen=True)
class RunResult:
    """Output of :func:`run_model`.

    The primary deliverable is :attr:`leak` (the absolute trigger
    output) with :attr:`cycle_period_s` as its pilot-facing companion.
    :attr:`trigger_potential` is preserved as ``rank_normalise(leak)``
    so existing CLI / viz / KMZ consumers keep working without a
    rewrite.

    Attributes
    ----------
    trigger_potential : np.ndarray
        Float64, ``[0, 1]``, NaN-passthrough. The primary display
        raster. Defined as
        ``max(rank_normalise(leak), rank_normalise(draft_potential))``
        — the per-cell maximum of the two physically meaningful
        fields, each independently ranked within its own positive-cell
        population. A scarp lip in the top 1 % of ``leak`` (high
        absolute W/m², concentrated) and a spur shoulder in the top
        1 % of ``draft_potential`` (lower W/m² after Gaussian
        smoothing, but spread across many cells) both land near 1.0
        despite their order-of-magnitude difference in absolute units.
        The blend rescues diffuse spur clusters that pure
        cell-level ``rank_normalise(leak)`` lost to the q95 threshold,
        without the scarp-suppression that pure
        ``rank_normalise(draft_potential)`` introduced
        (Gaussian aggregation is sum-preserving, so concentrated peaks
        get diluted across the kernel footprint). See ``docs/MODEL.md``
        §11.9 for the derivation and
        ``outputs/mallerstang_p34_rank_blend.png`` for the
        validation render.
    leak : np.ndarray
        Float64, W/m² (when heating drives the weights). The per-cell
        time-averaged trigger-release rate from the leaky-bucket
        accumulation. Absolute units, useful for cross-tile comparison.
        NaN at NaN-DEM cells. The conservation-exact physical field:
        ``Σ leak + residual_at_sinks_total ≡ Σ heating``.
    draft_potential : np.ndarray
        Float64, W/m². A Gaussian-aggregated view of :attr:`leak` at a
        thermal-merging scale (default σ = 75 m), with a post-smooth
        slope mask reapplied so flat plateaus and valley floors stay
        at ``0``. Models the fact that rising buoyant plumes coalesce
        as they rise — a diffuse spur leaking 1 W/m² over many cells
        produces a thermal as flyable as a scarp leaking 100 W/m² in
        one cell, provided the total power is comparable. This is the
        field clustering and the rank-normalised
        :attr:`trigger_potential` are derived from when
        ``draft_aggregation_sigma_m > 0``; with ``= 0`` it collapses
        to :attr:`leak` (subject to the same slope mask). NaN at
        NaN-DEM cells.
    forward : np.ndarray
        Float64, W/m². The post-leak through-flow that was passed to
        the D∞ neighbours. ``leak + forward`` is the pre-leak
        through-flow (analogous to the old ``weighted_convergence``).
        Diagnostic.
    cycle_period_s : np.ndarray
        Float64, seconds. ``q_storage / leak``; ``+inf`` where a cell
        does not leak. The buoyancy-cycle period at each cell — short
        on sharp scarps (consistent triggers), long on gentle ridges
        (cyclic mass-release dumps). NaN at NaN-DEM cells.
    residual_at_sinks_total : float
        Total ``forward`` that reached cells with no D∞ outflow (true
        sinks on the inverted DEM — real-terrain summits and
        domain-boundary outlets) without being consumed. A diagnostic
        scalar for parameter tuning: a large fraction of the total
        injected weight ending up here means ``f_drain`` is too high
        (or ``q_storage`` too high) and triggers are being
        under-counted.
    draft_mask_loss_total : float
        Total leak (W/m² × cells) discarded by the post-smooth slope
        mask on :attr:`draft_potential`. Gaussian aggregation spreads
        rim / scarp leak onto adjacent flat plateaus; the slope mask
        zeros those cells so the Phase 3.1 "summit interior dim" gate
        holds. The energy thrown away is logged here as a diagnostic
        — it should be a small fraction of ``Σ leak`` for the
        aggregation to be physically meaningful. ``0.0`` when
        ``draft_aggregation_sigma_m == 0`` (no aggregation, no
        bleed).
    weighted_convergence : np.ndarray
        Float64, ``leak + forward``. The pre-leak through-flow at
        each cell — the heating-weighted D∞ accumulation that the
        old pipeline reported. Kept as a diagnostic and for backward
        compatibility with ``viz.plot_weighted_convergence``.
    heating_wm2 : np.ndarray
        Float64, W/m². The per-cell weight passed to the leaky
        accumulation. NaN at finite-DEM edge cells where the 3×3
        stencil could not resolve slope/aspect; those entries are
        substituted with ``0.0`` before being passed as ``weights=``.
    smoothed_dem_m : np.ndarray
        Float64, metres. The Gaussian-smoothed input DEM.
    tilted_dem_m : np.ndarray
        Float64, metres. The smoothed DEM with the wind-tilt ramp
        added (input to the inversion).
    profile_curvature : np.ndarray
        Float64, 1/m. From the *raw* DEM. Positive = convex. Drives
        ``f_drain`` and ``q_storage`` along with :attr:`slope_rad`.
    slope_rad : np.ndarray
        Float64, radians. From the *raw* DEM.
    """

    trigger_potential: np.ndarray
    leak: np.ndarray
    draft_potential: np.ndarray
    forward: np.ndarray
    cycle_period_s: np.ndarray
    residual_at_sinks_total: float
    draft_mask_loss_total: float
    weighted_convergence: np.ndarray
    heating_wm2: np.ndarray
    smoothed_dem_m: np.ndarray
    tilted_dem_m: np.ndarray
    profile_curvature: np.ndarray
    slope_rad: np.ndarray


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
    curvature_smoothing_sigma_m: float = 10.0,
    draft_aggregation_sigma_m: float = 75.0,
    min_slope_deg: float = 2.5,
    slope_scale_deg: float = 15.0,
    kappa_ref: float = 0.005,
    q_ref: float = 1.0e6,
    f_min: float = F_MIN_DEFAULT,
    f_max: float = F_MAX_DEFAULT,
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
    curvature_smoothing_sigma_m : float, default 10.0
        Gaussian smoothing scale in metres applied to the raw DEM
        before slope and profile curvature are derived **for the
        leaky shape functions** ``f_drain`` and ``q_storage``.
        Independent of ``smoothing_sigma_m`` (which only affects the
        routing path) and of the raw slope/aspect/curvature that
        feed irradiance and the ``RunResult`` diagnostics. Suppresses
        single-cell LIDAR speckle that would otherwise saturate
        ``sat(κ⁺/κ_ref)`` on isolated cells and produce a per-cell
        spray on the leak raster. Carries forward the ``MODEL.md``
        §6 ¶282–284 prescription. ``0`` disables (raw curvature
        feeds the shape functions, reproducing pre-2026-05-09
        behaviour).
    draft_aggregation_sigma_m : float, default 75.0
        Gaussian smoothing scale in metres applied to ``leak`` to
        produce :attr:`RunResult.draft_potential`, the field that
        drives :attr:`RunResult.trigger_potential` and downstream
        clustering. Models the coalescence of buoyant plumes as they
        rise: a diffuse spur and a concentrated scarp lip with
        comparable total power produce thermals of comparable
        flyability, but the predecessor's cell-level rank
        normalisation gave the concentrated scarp an unfair advantage
        in the percentile threshold. 75 m is roughly one thermal
        column radius at low trigger altitude on UK terrain. The
        Gaussian is sum-preserving so the conservation invariant on
        the underlying ``leak`` field is unaffected; a post-smooth
        slope mask (using ``min_slope_deg``) preserves the
        ``summit-plateau dim`` gate by zeroing bleed onto flat
        cells, with the energy thrown away reported on
        :attr:`RunResult.draft_mask_loss_total`. ``0`` disables
        aggregation: ``draft_potential`` becomes ``leak`` (with the
        slope mask still applied) and the rank-normalised
        ``trigger_potential`` collapses to the pre-aggregation
        formulation.
    min_slope_deg : float, default 2.5
        Slope (degrees) below which a cell contributes nothing to
        ``sharpness`` in the leaky shape function — i.e. ``f_drain``
        stays at its ``f_max`` floor and the cell forwards everything.
        Encodes "flat surfaces don't trigger" and replaces the hard
        post-hoc slope-mask cutoff of the previous pipeline.
    slope_scale_deg : float, default 15.0
        Reference slope scale (degrees). At
        ``slope − min_slope_deg = slope_scale_deg`` the slope
        contribution to ``sharpness`` is ``1 − exp(−1) ≈ 0.63``;
        the saturation reaches ``≈ 0.95`` by 3× the scale. Larger
        values make the slope dependence gentler.
    kappa_ref : float, default 0.005 (1/m)
        Reference profile-curvature scale. At
        ``profile_curv = kappa_ref`` the curvature contribution to
        ``sharpness`` is ``≈ 0.63``. ``0.005 m⁻¹`` corresponds to a
        slope changing by roughly 1° over 10 m, which marks the
        threshold between "rounded ridge" and "convex break" on
        Dales-scale terrain.
    q_ref : float, default 1.0e6 (J/m² when heating is W/m²)
        Reference buoyancy storage capacity. Sets the cycle-period
        magnitude — at ``leak = 1 W/m²`` and the default ``q_ref``,
        ``cycle_period`` is ``10⁶ s ≈ 11.6 days``; realistic Dales
        ``leak`` rates are 10²–10³ W/m² so realistic cycle periods
        are tens to thousands of seconds.
    f_min : float, default 0.15
        Skimming floor on the drain fraction — at the sharpest
        terrain, this fraction of through-flow always passes by as
        boundary-layer skim. See
        :data:`thermal_model.physics.F_MIN_DEFAULT`.
    f_max : float, default 1.0
        Maximum drain fraction. On flats and concave terrain
        ``f_drain = f_max`` (forward everything; no leak).
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
    The trigger raster is the **leak** field from a leaky-bucket
    weighted D∞ accumulation: each cell consumes a curvature- and
    slope-dependent fraction ``(1 − f_drain)`` of its through-flow
    as trigger output and forwards only ``f_drain`` of it onward,
    with a per-cell buoyancy storage capacity ``Q`` giving the
    cycle period ``τ = Q / leak``. ``RunResult.trigger_potential``
    is ``rank_normalise(leak)`` for backward-compatible display;
    ``RunResult.leak`` carries the absolute units.

    Returns
    -------
    RunResult
        The trigger / leak / cycle-period rasters plus diagnostic
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
    if curvature_smoothing_sigma_m < 0:
        raise ValueError(
            "curvature_smoothing_sigma_m must be non-negative, "
            f"got {curvature_smoothing_sigma_m}"
        )
    if draft_aggregation_sigma_m < 0:
        raise ValueError(
            "draft_aggregation_sigma_m must be non-negative, "
            f"got {draft_aggregation_sigma_m}"
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

    # Curvature/slope feeding the leaky shape functions are derived
    # from a Gaussian-smoothed copy of the raw DEM (σ =
    # curvature_smoothing_sigma_m). Suppresses single-cell LIDAR
    # speckle that would otherwise saturate sat(κ⁺/κ_ref) on isolated
    # cells. σ = 0 reproduces pre-2026-05-09 behaviour exactly.
    if curvature_smoothing_sigma_m > 0:
        dem_for_shape = _gaussian_smooth_nan(
            dem64, curvature_smoothing_sigma_m / float(cell_size_m)
        )
        slope_for_shape_raw = slope(dem_for_shape, cell_size_m)
        kprof_for_shape_raw = profile_curvature(dem_for_shape, cell_size_m)
    else:
        slope_for_shape_raw = slope_rad
        kprof_for_shape_raw = kprof

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

    # Build f_drain and q_storage from the *raw* DEM curvature and
    # slope. Detachment geometry is a property of real terrain — the
    # tilted / inverted surface is purely a routing device. Edge
    # cells where the 3×3 stencil cannot resolve curvature/slope are
    # substituted with zeros (no curvature / no slope ⇒ ``f_drain ≡
    # f_max``, ``q_storage ≡ q_ref``), matching the same edge-NaN
    # handling we apply to ``heating`` before passing it as weights.
    min_slope_rad = math.radians(min_slope_deg)
    slope_scale_rad = math.radians(slope_scale_deg)
    kprof_for_shape = np.where(
        np.isnan(kprof_for_shape_raw) & ~nan_mask, 0.0, kprof_for_shape_raw
    )
    slope_for_shape = np.where(
        np.isnan(slope_for_shape_raw) & ~nan_mask, 0.0, slope_for_shape_raw
    )
    f_drain = f_drain_field(
        kprof_for_shape,
        slope_for_shape,
        kappa_ref=kappa_ref,
        slope_min_rad=min_slope_rad,
        slope_scale_rad=slope_scale_rad,
        f_min=f_min,
        f_max=f_max,
    )
    q_storage = q_storage_field(
        kprof_for_shape,
        slope_for_shape,
        q_ref=q_ref,
        kappa_ref=kappa_ref,
        slope_min_rad=min_slope_rad,
        slope_scale_rad=slope_scale_rad,
    )
    # Re-mask NaN-DEM cells on the leak-shape fields — the input
    # replacement above produced finite values everywhere; restore
    # NaN at NaN-DEM cells to satisfy the kernel's contract.
    f_drain = np.where(nan_mask, np.nan, f_drain)
    q_storage = np.where(nan_mask, np.nan, q_storage)

    leaky = leaky_weighted_accumulation(
        inverted_filled,
        cell_size_m,
        f_drain=f_drain,
        q_storage=q_storage,
        weights=weights,
    )

    weighted_conv = leaky.leak + leaky.forward  # pre-leak through-flow

    # Drafting / aggregation step. Rising buoyant plumes coalesce, so
    # a pilot at trigger height samples a footprint several thermal-
    # column radii across. A diffuse spur leak and a concentrated
    # scarp leak with comparable total power produce thermals of
    # comparable flyability — but cell-level rank normalisation gave
    # the scarp an unfair advantage in the percentile threshold. We
    # Gaussian-aggregate the leak field at a thermal-merging scale,
    # then reapply the slope mask so bleed onto flat plateaus does
    # not reintroduce the Phase 3.1 "summit interior bright" artefact.
    # ``_gaussian_smooth_nan`` is sum-preserving on interior cells so
    # the conservation invariant on the underlying ``leak`` field is
    # unaffected; the energy thrown away by the mask is reported
    # separately as ``draft_mask_loss_total``.
    draft_sigma_cells = draft_aggregation_sigma_m / float(cell_size_m)
    draft_smoothed = _gaussian_smooth_nan(leaky.leak, draft_sigma_cells)
    slope_mask = slope_for_shape > min_slope_rad
    draft = np.where(slope_mask & ~nan_mask, draft_smoothed, 0.0)
    draft = np.where(nan_mask, np.nan, draft)
    # Energy lost to the post-smooth slope mask. Both arrays have the
    # same finite-cell support so the difference is well-defined.
    draft_mask_loss = float(
        np.nansum(np.where(slope_mask, 0.0, np.where(nan_mask, 0.0, draft_smoothed)))
    )

    # ``trigger_potential`` is the per-cell maximum of two
    # independently rank-normalised fields: the cell-level ``leak``
    # (the scarp regime — concentrated peaks, high absolute magnitudes)
    # and the aggregated ``draft_potential`` (the spur regime — broad
    # coalesced lift, lower magnitudes after Gaussian smoothing). Each
    # field is ranked within its own positive-cell population, so a
    # scarp lip in the top 1 % of leak and a spur shoulder in the top
    # 1 % of draft both land near 1.0 regardless of their absolute
    # W/m² difference. The per-cell ``fmax`` then picks "the regime in
    # which this cell is most extreme". This blend reproduces the
    # spur-rescue effect of drafting while preserving scarp visibility
    # that pure ``rank_norm(draft)`` over-suppresses — see
    # ``outputs/mallerstang_p34_rank_blend.png`` for the 2026-05-11
    # validation render that motivated the choice. Both
    # rank-normalised inputs are in [0, 1] with NaN passthrough so
    # ``fmax`` preserves the NaN-DEM support without explicit masking.
    leak_rank = _rank_normalise(leaky.leak)
    draft_rank = _rank_normalise(draft)
    trigger = np.fmax(leak_rank, draft_rank)
    trigger = np.where(nan_mask, np.nan, trigger)

    return RunResult(
        trigger_potential=trigger.astype(np.float64, copy=False),
        leak=leaky.leak,
        draft_potential=draft.astype(np.float64, copy=False),
        forward=leaky.forward,
        cycle_period_s=leaky.cycle_period,
        residual_at_sinks_total=leaky.residual_at_sinks_total,
        draft_mask_loss_total=draft_mask_loss,
        weighted_convergence=weighted_conv.astype(np.float64, copy=False),
        heating_wm2=heating.astype(np.float64, copy=False),
        smoothed_dem_m=smoothed,
        tilted_dem_m=tilted.astype(np.float64, copy=False),
        profile_curvature=kprof.astype(np.float64, copy=False),
        slope_rad=slope_rad.astype(np.float64, copy=False),
    )
