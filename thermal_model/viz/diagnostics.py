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

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from thermal_model.physics import fill_pits, flow_accumulation
from thermal_model.terrain import aspect, profile_curvature, slope
from thermal_model.viz.hillshade import hillshade

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    epsilon: float = 1e-3,
    cmap: str | Colormap = "magma",
    alpha: float = 0.6,
    title: str = "Inverted-DEM flow accumulation (convergence)",
    contours: bool = True,
    contour_levels: int | Sequence[float] = 10,
) -> Axes:
    """Inverted-treacle convergence map on a hillshade backdrop.

    Pipelines :func:`thermal_model.physics.fill_pits` (with a small
    positive ``epsilon`` so internal flats route through), inversion,
    and :func:`thermal_model.physics.flow_accumulation`. The result is
    the project's headline convergence diagnostic: bright cells mark
    predicted thermal-source convergence under the hydrological
    analogy of CLAUDE.md §2.

    By default, white elevation contours are drawn on top of the
    convergence overlay so the bright cells can be read against
    terrain shape.
    """
    inverted = float(np.nanmax(dem)) - dem
    filled = fill_pits(inverted, epsilon=epsilon)
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
