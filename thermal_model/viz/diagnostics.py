"""Diagnostic matplotlib plotters for DEM analysis.

Each plotter takes a DEM (and any pre-computed overlay) and renders a
hillshade backdrop with a semi-transparent overlay. These figures are
*diagnostics*, not deliverables — the project's outputs are GeoTIFF
and KMZ. Use them to sanity-check intermediate fields during
development.

Pass ``ax`` to compose into multi-panel figures; without it, each
function creates its own ``Figure`` and ``Axes``. Every plotter
returns the ``Axes`` so the caller can layer further annotations
(contours, scatter overlays, scale bars).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from scipy import ndimage

from thermal_model.physics import (
    DEFAULT_ABSORPTIVITY,
    fill_pits,
    flow_accumulation,
    heating_field,
    run_model,
)
from thermal_model.solar import (
    cast_shadow_mask,
    clear_sky_irradiance,
    slope_irradiance,
    solar_position,
)
from thermal_model.terrain import aspect, profile_curvature, slope
from thermal_model.viz.hillshade import hillshade

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap


def _ensure_axes(ax: Axes | None, figsize: tuple[float, float] = (6.0, 6.0)) -> Axes:
    if ax is not None:
        return ax
    _fig, new_ax = plt.subplots(figsize=figsize)
    return new_ax


def plot_overlay(
    dem: np.ndarray,
    overlay: np.ndarray,
    cell_size_m: float,
    *,
    ax: Axes | None = None,
    cmap: str | Colormap = "viridis",
    log: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.6,
    label: str = "",
    title: str = "",
    colorbar: bool = True,
    contours: bool = False,
    contour_levels: int | Sequence[float] = 10,
    contour_color: str = "white",
    contour_linewidth: float = 0.5,
    contour_alpha: float = 0.6,
) -> Axes:
    """Hillshade backdrop with a semi-transparent scalar overlay.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array used for the hillshade backdrop.
    overlay : np.ndarray
        Scalar field to overlay; must match ``dem.shape``. NaN cells
        render transparent.
    cell_size_m : float
        Square cell size in metres.
    ax : matplotlib.axes.Axes, optional
        Target axes. If omitted, a new figure and axes are created.
    cmap : str or Colormap, default "viridis"
        Colormap for the overlay.
    log : bool, default False
        If True, use a logarithmic norm. The overlay must contain at
        least one strictly positive value.
    vmin, vmax : float, optional
        Overlay value bounds. ``None`` uses the data extremes (with
        positives only when ``log=True``).
    alpha : float, default 0.6
        Overlay opacity in ``[0, 1]``.
    label : str, default ""
        Colorbar label.
    title : str, default ""
        Axes title.
    colorbar : bool, default True
        Attach a colorbar to ``ax``.
    contours : bool, default False
        Draw elevation contour lines on top of the overlay, sourced
        from ``dem``.
    contour_levels : int or sequence of float, default 10
        Either the number of evenly-spaced contour levels, or an
        explicit sequence of elevation values in metres.
    contour_color : str, default "white"
        Contour line colour.
    contour_linewidth : float, default 0.5
        Contour line width in points.
    contour_alpha : float, default 0.6
        Contour line opacity in ``[0, 1]``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if dem.shape != overlay.shape:
        raise ValueError(
            f"dem and overlay must have the same shape, "
            f"got {dem.shape} vs {overlay.shape}"
        )
    ax = _ensure_axes(ax)
    rows, cols = dem.shape
    width_m = cols * cell_size_m
    height_m = rows * cell_size_m
    # imshow extent: (left, right, bottom, top). With origin='upper'
    # (matplotlib's default) row 0 lives at the top, so bottom=H, top=0.
    extent = (0.0, width_m, height_m, 0.0)

    shade = hillshade(dem, cell_size_m)
    ax.imshow(
        shade,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=extent,
    )

    norm: Normalize
    if log:
        positive = overlay[np.isfinite(overlay) & (overlay > 0)]
        if positive.size == 0:
            raise ValueError("log=True but overlay has no positive values")
        lo = float(positive.min()) if vmin is None else float(vmin)
        hi = float(positive.max()) if vmax is None else float(vmax)
        norm = LogNorm(vmin=lo, vmax=hi)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        overlay,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        interpolation="nearest",
        extent=extent,
    )
    if contours:
        # Skip drawing on flats; matplotlib emits a warning when min == max.
        finite = dem[np.isfinite(dem)]
        if finite.size and float(finite.min()) < float(finite.max()):
            # Cell-centre coords so the contour grid lines up with the
            # imshow extent.
            xs = (np.arange(cols) + 0.5) * cell_size_m
            ys = (np.arange(rows) + 0.5) * cell_size_m
            ax.contour(
                xs,
                ys,
                dem,
                levels=contour_levels,
                colors=contour_color,
                linewidths=contour_linewidth,
                alpha=contour_alpha,
            )
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if label:
            cbar.set_label(label)
    if title:
        ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return ax


def plot_convergence(
    dem: np.ndarray,
    cell_size_m: float,
    *,
    ax: Axes | None = None,
    smooth_sigma_m: float = 10.0,
    cmap: str | Colormap = "magma",
    alpha: float = 0.6,
    title: str = "Inverted-DEM flow accumulation (convergence)",
    contours: bool = True,
    contour_levels: int | Sequence[float] = 10,
) -> Axes:
    """Inverted-treacle convergence map on a hillshade backdrop.

    Pipeline:

    1. Invert the DEM (``max(z) - z``).
    2. Gaussian-smooth the inverted DEM with a kernel of standard
       deviation ``smooth_sigma_m`` metres (converted to cells via
       ``cell_size_m``). Softens abrupt ridge-into-flat-plateau
       transitions so the priority-flood frontier doesn't enter the
       flat as a single line, which would otherwise produce
       parallel-streak artefacts perpendicular to the ridge after
       flow accumulation. ``0`` disables smoothing.
    3. Fill closed depressions on the smoothed inverted DEM
       (:func:`thermal_model.physics.fill_pits`).
    4. D-infinity flow accumulation
       (:func:`thermal_model.physics.flow_accumulation`).

    The result is the project's headline convergence diagnostic:
    bright cells mark predicted thermal-source convergence under the
    hydrological analogy of CLAUDE.md §2.

    For a more principled (and substantially slower) flat-resolution
    alternative, run :func:`thermal_model.physics.resolve_flats`
    between fill and accumulation in your own pipeline; the smoothing
    here is a fast diagnostic shortcut.

    By default, white elevation contours are drawn on top of the
    convergence overlay so the bright cells can be read against
    terrain shape.
    """
    if smooth_sigma_m < 0:
        raise ValueError(f"smooth_sigma_m must be non-negative, got {smooth_sigma_m}")
    inverted = float(np.nanmax(dem)) - dem

    sigma_cells = smooth_sigma_m / float(cell_size_m)
    if sigma_cells > 0:
        nan_mask = np.isnan(inverted)
        if nan_mask.any():
            # gaussian_filter doesn't honour NaN, so stamp NaN cells
            # with the finite mean before convolving and restore NaN
            # afterwards. The bias near NaN-adjacent cells is bounded
            # by the kernel scale, which at the default sigma is well
            # below any real terrain feature.
            stamped = np.where(nan_mask, float(np.nanmean(inverted)), inverted)
            smoothed = ndimage.gaussian_filter(stamped, sigma=sigma_cells)
            smoothed[nan_mask] = np.nan
        else:
            smoothed = ndimage.gaussian_filter(inverted, sigma=sigma_cells)
        prepared = smoothed
    else:
        prepared = inverted

    filled = fill_pits(prepared, epsilon=0.0)
    acc = flow_accumulation(filled, cell_size_m)
    return plot_overlay(
        dem,
        acc,
        cell_size_m,
        ax=ax,
        cmap=cmap,
        log=True,
        alpha=alpha,
        label="upstream cell count (log)",
        title=title,
        contours=contours,
        contour_levels=contour_levels,
    )


def plot_slope(
    dem: np.ndarray,
    cell_size_m: float,
    *,
    ax: Axes | None = None,
    cmap: str | Colormap = "viridis",
    alpha: float = 0.6,
    title: str = "Slope (rad)",
) -> Axes:
    """Slope overlay on hillshade."""
    sl = slope(dem, cell_size_m)
    return plot_overlay(
        dem,
        sl,
        cell_size_m,
        ax=ax,
        cmap=cmap,
        vmin=0.0,
        vmax=float(np.pi / 2),
        alpha=alpha,
        label="slope (rad)",
        title=title,
    )


def plot_aspect(
    dem: np.ndarray,
    cell_size_m: float,
    *,
    ax: Axes | None = None,
    cmap: str | Colormap = "twilight_shifted",
    alpha: float = 0.6,
    title: str = "Aspect (compass bearing, rad)",
) -> Axes:
    """Aspect overlay on hillshade.

    Uses a cyclic colormap so that 0 (north) and 2*pi (north) map to
    the same colour.
    """
    asp = aspect(dem, cell_size_m)
    return plot_overlay(
        dem,
        asp,
        cell_size_m,
        ax=ax,
        cmap=cmap,
        vmin=0.0,
        vmax=float(2 * np.pi),
        alpha=alpha,
        label="aspect (rad)",
        title=title,
    )


def plot_heating(
    dem: np.ndarray,
    cell_size_m: float,
    when: datetime,
    latitude_deg: float,
    longitude_deg: float,
    *,
    ax: Axes | None = None,
    elevation_m: float | None = None,
    linke_turbidity: float = 3.0,
    absorptivity: float = DEFAULT_ABSORPTIVITY,
    cmap: str | Colormap = "inferno",
    alpha: float = 0.7,
    title: str | None = None,
    contours: bool = True,
    contour_levels: int | Sequence[float] = 10,
    vmin: float | None = 0.0,
    vmax: float | None = None,
) -> Axes:
    """Heating field ``H`` (W/m²) overlaid on a hillshade backdrop.

    Runs the full Phase 2 pipeline:

    1. Slope and aspect from the DEM (Horn 1981).
    2. Sun position and clear-sky irradiance at ``when`` from
       :func:`thermal_model.solar.solar_position` and
       :func:`thermal_model.solar.clear_sky_irradiance`.
    3. Slope-projected beam + diffuse irradiance via
       :func:`thermal_model.solar.slope_irradiance`.
    4. Cast-shadow mask via
       :func:`thermal_model.solar.cast_shadow_mask` (horizon scan).
    5. Heating field
       ``H = absorptivity * (s * I_beam + I_diffuse)`` from
       :func:`thermal_model.physics.heating_field`.

    The result is overlaid on the Lambertian hillshade with
    elevation contours, axis units in metres, and a colorbar in
    W/m².

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array in metres with NaN nodata.
    cell_size_m : float
        Square cell size in metres.
    when : datetime.datetime
        Timezone-aware instant for the sun position and irradiance.
    latitude_deg, longitude_deg : float
        Site coordinates in degrees (N-positive, E-positive). For a
        georeferenced DEM, these are typically derived from the
        DEM's centre via reprojection to EPSG:4326 — see the CLI
        ``preview`` subcommand.
    ax : matplotlib.axes.Axes, optional
        Target axes. If omitted, a new figure and axes are created.
    elevation_m : float, optional
        Site elevation in metres. Affects atmospheric airmass in
        the clear-sky model. Defaults to the median of finite DEM
        cells.
    linke_turbidity : float, default 3.0
        Linke turbidity for the Ineichen-Perez clear-sky model.
    absorptivity : float, default ``DEFAULT_ABSORPTIVITY`` (0.80)
        Shortwave absorptivity ``alpha = 1 - albedo``. The Phase 2
        single-value default; Phase 4 will switch in a per-cell
        array driven by land cover.
    cmap : str or Colormap, default "inferno"
        Colormap for the heating overlay. Dark on cold/shadowed,
        bright on hot.
    alpha : float, default 0.7
        Overlay opacity in ``[0, 1]``.
    title : str, optional
        Axes title. Defaults to a one-line summary of the
        timestamp and the sun's azimuth/altitude.
    contours : bool, default True
        Draw elevation contours on top of the heating overlay.
    contour_levels : int or sequence of float, default 10
        Number of evenly-spaced contour levels, or an explicit
        sequence of elevation values in metres.
    vmin, vmax : float, optional
        Colorbar limits in W/m². ``vmin`` defaults to ``0`` (a
        natural floor for solar heating); ``vmax`` defaults to the
        finite maximum of the heating field.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if elevation_m is None:
        finite = dem[np.isfinite(dem)]
        if finite.size == 0:
            raise ValueError("DEM contains no finite cells; cannot derive elevation.")
        elevation_m = float(np.median(finite))

    sun = solar_position(when, latitude_deg, longitude_deg, elevation_m=elevation_m)
    cs = clear_sky_irradiance(
        when,
        latitude_deg,
        longitude_deg,
        elevation_m=elevation_m,
        linke_turbidity=linke_turbidity,
    )
    slope_rad = slope(dem, cell_size_m)
    aspect_rad = aspect(dem, cell_size_m)
    irr = slope_irradiance(slope_rad, aspect_rad, sun, cs)
    shadow = cast_shadow_mask(dem, cell_size_m, sun)
    h = heating_field(irr, shadow, absorptivity=absorptivity)

    if title is None:
        sun_az_deg = math.degrees(sun.azimuth_rad) % 360.0
        sun_alt_deg = math.degrees(sun.altitude_rad)
        title = (
            f"Heating H (W/m²) — {when.isoformat(timespec='minutes')}\n"
            f"sun: az={sun_az_deg:.1f}°, alt={sun_alt_deg:.1f}° "
            f"at ({latitude_deg:.3f}°, {longitude_deg:.3f}°)"
        )

    return plot_overlay(
        dem,
        h,
        cell_size_m,
        ax=ax,
        cmap=cmap,
        log=False,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        label="heating (W/m²)",
        title=title,
        contours=contours,
        contour_levels=contour_levels,
    )


def plot_trigger_potential(
    dem: np.ndarray,
    cell_size_m: float,
    when: datetime,
    latitude_deg: float,
    longitude_deg: float,
    *,
    wind_from_deg: float,
    wind_speed_ms: float,
    ax: Axes | None = None,
    wind_tilt_k: float = 0.03,
    smoothing_sigma_m: float = 10.0,
    min_slope_deg: float = 2.5,
    absorptivity: float = DEFAULT_ABSORPTIVITY,
    elevation_m: float | None = None,
    linke_turbidity: float = 3.0,
    resolve_flats: bool = True,
    cmap: str | Colormap = "magma",
    alpha: float = 0.7,
    title: str | None = None,
    contours: bool = True,
    contour_levels: int | Sequence[float] = 10,
) -> Axes:
    """Trigger-potential raster overlaid on a hillshade backdrop.

    Runs the full Phase 3 pipeline via
    :func:`thermal_model.physics.run_model` and overlays the result
    on the Lambertian hillshade. Bright cells mark predicted
    ground-level thermal source/trigger locations.
    """
    result = run_model(
        dem,
        cell_size_m,
        when,
        latitude_deg,
        longitude_deg,
        wind_from_deg=wind_from_deg,
        wind_speed_ms=wind_speed_ms,
        wind_tilt_k=wind_tilt_k,
        smoothing_sigma_m=smoothing_sigma_m,
        min_slope_deg=min_slope_deg,
        absorptivity=absorptivity,
        elevation_m=elevation_m,
        linke_turbidity=linke_turbidity,
        resolve_flats=resolve_flats,
    )

    if title is None:
        title = (
            f"Trigger potential T — {when.isoformat(timespec='minutes')}\n"
            f"wind from {wind_from_deg:.0f}° @ {wind_speed_ms:.1f} m/s, "
            f"k={wind_tilt_k}"
        )

    return plot_overlay(
        dem,
        result.trigger_potential,
        cell_size_m,
        ax=ax,
        cmap=cmap,
        log=False,
        vmin=0.0,
        vmax=1.0,
        alpha=alpha,
        label="trigger potential T",
        title=title,
        contours=contours,
        contour_levels=contour_levels,
    )


def plot_weighted_convergence(
    dem: np.ndarray,
    cell_size_m: float,
    when: datetime,
    latitude_deg: float,
    longitude_deg: float,
    *,
    wind_from_deg: float,
    wind_speed_ms: float,
    ax: Axes | None = None,
    wind_tilt_k: float = 0.03,
    smoothing_sigma_m: float = 10.0,
    min_slope_deg: float = 2.5,
    absorptivity: float = DEFAULT_ABSORPTIVITY,
    elevation_m: float | None = None,
    linke_turbidity: float = 3.0,
    resolve_flats: bool = True,
    cmap: str | Colormap = "magma",
    alpha: float = 0.6,
    title: str | None = None,
    contours: bool = True,
    contour_levels: int | Sequence[float] = 10,
) -> Axes:
    """Heating-weighted D∞ flow accumulation overlay on a hillshade.

    The intermediate quantity from the Phase 3 pipeline: total upstream
    W/m² × cell-count flowing through each cell on the inverted, wind-
    tilted DEM. Bright lineaments are convergence corridors that have
    inherited warmth from sunlit upstream catchments. Logged via
    :class:`matplotlib.colors.LogNorm` since the dynamic range
    spans several decades.
    """
    result = run_model(
        dem,
        cell_size_m,
        when,
        latitude_deg,
        longitude_deg,
        wind_from_deg=wind_from_deg,
        wind_speed_ms=wind_speed_ms,
        wind_tilt_k=wind_tilt_k,
        smoothing_sigma_m=smoothing_sigma_m,
        min_slope_deg=min_slope_deg,
        absorptivity=absorptivity,
        elevation_m=elevation_m,
        linke_turbidity=linke_turbidity,
        resolve_flats=resolve_flats,
    )

    if title is None:
        title = (
            f"Heating-weighted convergence — {when.isoformat(timespec='minutes')}\n"
            f"wind from {wind_from_deg:.0f}° @ {wind_speed_ms:.1f} m/s, "
            f"k={wind_tilt_k}"
        )

    return plot_overlay(
        dem,
        result.weighted_convergence,
        cell_size_m,
        ax=ax,
        cmap=cmap,
        log=True,
        alpha=alpha,
        label="upstream W/m² × cells (log)",
        title=title,
        contours=contours,
        contour_levels=contour_levels,
    )


def plot_leak(
    dem: np.ndarray,
    cell_size_m: float,
    when: datetime,
    latitude_deg: float,
    longitude_deg: float,
    *,
    wind_from_deg: float,
    wind_speed_ms: float,
    ax: Axes | None = None,
    wind_tilt_k: float = 0.03,
    smoothing_sigma_m: float = 10.0,
    min_slope_deg: float = 2.5,
    slope_scale_deg: float = 15.0,
    kappa_ref: float = 0.005,
    q_ref: float = 1.0e6,
    f_min: float = 0.15,
    f_max: float = 1.0,
    absorptivity: float = DEFAULT_ABSORPTIVITY,
    elevation_m: float | None = None,
    linke_turbidity: float = 3.0,
    resolve_flats: bool = True,
    cmap: str | Colormap = "magma",
    alpha: float = 0.7,
    log: bool = True,
    title: str | None = None,
    contours: bool = True,
    contour_levels: int | Sequence[float] = 10,
) -> Axes:
    """Absolute trigger leak (W/m²) overlaid on a hillshade backdrop.

    Runs the full Phase 3.1 pipeline via
    :func:`thermal_model.physics.run_model` and overlays
    ``RunResult.leak`` on the Lambertian hillshade. Bright cells mark
    predicted ground-level trigger locations in *absolute* W/m² units
    (the time-averaged release rate at each cell), in contrast to
    :func:`plot_trigger_potential` which plots the rank-normalised
    ``[0, 1]`` companion. Use this when cross-tile magnitude
    comparison matters; rank-normalised display is fine when only
    relative ranking within a single tile matters.

    Defaults to a logarithmic colour scale (``log=True``) — real leak
    fields span several decades and a linear scale would compress the
    bulk into the colormap floor. Pass ``log=False`` for a linear
    rendering on synthetic / uniform fixtures.
    """
    result = run_model(
        dem,
        cell_size_m,
        when,
        latitude_deg,
        longitude_deg,
        wind_from_deg=wind_from_deg,
        wind_speed_ms=wind_speed_ms,
        wind_tilt_k=wind_tilt_k,
        smoothing_sigma_m=smoothing_sigma_m,
        min_slope_deg=min_slope_deg,
        slope_scale_deg=slope_scale_deg,
        kappa_ref=kappa_ref,
        q_ref=q_ref,
        f_min=f_min,
        f_max=f_max,
        absorptivity=absorptivity,
        elevation_m=elevation_m,
        linke_turbidity=linke_turbidity,
        resolve_flats=resolve_flats,
    )

    if title is None:
        title = (
            f"Trigger leak (W/m²) — {when.isoformat(timespec='minutes')}\n"
            f"wind from {wind_from_deg:.0f}° @ {wind_speed_ms:.1f} m/s, "
            f"k={wind_tilt_k}"
        )

    return plot_overlay(
        dem,
        result.leak,
        cell_size_m,
        ax=ax,
        cmap=cmap,
        log=log,
        alpha=alpha,
        label="leak (W/m²)" if not log else "leak (W/m², log)",
        title=title,
        contours=contours,
        contour_levels=contour_levels,
    )


def plot_cycle_period(
    dem: np.ndarray,
    cell_size_m: float,
    when: datetime,
    latitude_deg: float,
    longitude_deg: float,
    *,
    wind_from_deg: float,
    wind_speed_ms: float,
    ax: Axes | None = None,
    wind_tilt_k: float = 0.03,
    smoothing_sigma_m: float = 10.0,
    min_slope_deg: float = 2.5,
    slope_scale_deg: float = 15.0,
    kappa_ref: float = 0.005,
    q_ref: float = 1.0e6,
    f_min: float = 0.15,
    f_max: float = 1.0,
    absorptivity: float = DEFAULT_ABSORPTIVITY,
    elevation_m: float | None = None,
    linke_turbidity: float = 3.0,
    resolve_flats: bool = True,
    cmap: str | Colormap = "plasma_r",
    alpha: float = 0.75,
    vmin_s: float = 60.0,
    vmax_s: float = 3600.0,
    title: str | None = None,
    contours: bool = True,
    contour_levels: int | Sequence[float] = 10,
) -> Axes:
    """Buoyancy-cycle period τ (seconds, log-scale) overlaid on a hillshade.

    Runs the full Phase 3.1 pipeline via
    :func:`thermal_model.physics.run_model` and overlays
    ``RunResult.cycle_period_s`` on the Lambertian hillshade. Cells
    where the leak is zero (no triggering) carry ``+inf`` in the
    cycle raster; they are converted to ``NaN`` here so they render
    transparent over the hillshade backdrop — distinct from cells
    that *do* trigger but on a long cycle.

    The colour scale is logarithmic and clipped to ``[vmin_s,
    vmax_s]`` (default 60 s to 1 hr) so the pilot-relevant band
    dominates the display. Defaults to ``plasma_r`` so light = short
    cycle (reliable consistent thermals) and dark = long cycle
    (sporadic mass-release dumps).
    """
    if vmin_s <= 0 or vmax_s <= vmin_s:
        raise ValueError(
            f"need 0 < vmin_s ({vmin_s}) < vmax_s ({vmax_s}) for log scale"
        )

    result = run_model(
        dem,
        cell_size_m,
        when,
        latitude_deg,
        longitude_deg,
        wind_from_deg=wind_from_deg,
        wind_speed_ms=wind_speed_ms,
        wind_tilt_k=wind_tilt_k,
        smoothing_sigma_m=smoothing_sigma_m,
        min_slope_deg=min_slope_deg,
        slope_scale_deg=slope_scale_deg,
        kappa_ref=kappa_ref,
        q_ref=q_ref,
        f_min=f_min,
        f_max=f_max,
        absorptivity=absorptivity,
        elevation_m=elevation_m,
        linke_turbidity=linke_turbidity,
        resolve_flats=resolve_flats,
    )

    # Mask non-leaking cells (cycle_period_s == +inf) so they render
    # transparent. plot_overlay treats NaN as transparent.
    cycle_for_plot = np.where(
        np.isfinite(result.cycle_period_s), result.cycle_period_s, np.nan
    )

    if title is None:
        title = (
            f"Cycle period τ (s, log) — {when.isoformat(timespec='minutes')}\n"
            f"wind from {wind_from_deg:.0f}° @ {wind_speed_ms:.1f} m/s, "
            f"k={wind_tilt_k}"
        )

    return plot_overlay(
        dem,
        cycle_for_plot,
        cell_size_m,
        ax=ax,
        cmap=cmap,
        log=True,
        vmin=vmin_s,
        vmax=vmax_s,
        alpha=alpha,
        label="cycle period τ (s, log)",
        title=title,
        contours=contours,
        contour_levels=contour_levels,
    )


def plot_profile_curvature(
    dem: np.ndarray,
    cell_size_m: float,
    *,
    ax: Axes | None = None,
    cmap: str | Colormap = "RdBu_r",
    alpha: float = 0.6,
    clip_quantile: float = 0.98,
    title: str = "Profile curvature (1/m): red = convex",
) -> Axes:
    """Profile curvature overlay on hillshade.

    Uses a diverging colormap symmetric about zero. The colour limits
    are set from a robust quantile of ``|kprof|`` so that a few
    extreme outlier cells don't wash the rest of the field out. Per
    CLAUDE.md §5, positive (convex, red) curvature is the trigger
    proxy we ultimately care about.
    """
    kprof = profile_curvature(dem, cell_size_m)
    finite = kprof[np.isfinite(kprof)]
    if finite.size == 0:
        clip = 1.0
    else:
        clip = float(np.quantile(np.abs(finite), clip_quantile))
        if clip == 0.0:
            clip = 1.0
    return plot_overlay(
        dem,
        kprof,
        cell_size_m,
        ax=ax,
        cmap=cmap,
        vmin=-clip,
        vmax=clip,
        alpha=alpha,
        label="profile curvature (1/m)",
        title=title,
    )
