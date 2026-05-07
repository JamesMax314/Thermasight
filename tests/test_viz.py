"""Tests for thermal_model.viz: hillshade and diagnostic plotters.

The Agg backend is selected in ``conftest.py`` so this module can use
plain ``import matplotlib.pyplot``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from thermal_model.viz import (
    hillshade,
    plot_aspect,
    plot_convergence,
    plot_heating,
    plot_overlay,
    plot_profile_curvature,
    plot_slope,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_hill(n: int, height: float = 80.0) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    cx = cy = (n - 1) / 2.0
    sigma = n / 6.0
    return 400.0 + height * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


# ---------------------------------------------------------------------------
# Hillshade numerics
# ---------------------------------------------------------------------------


def test_hillshade_flat_surface_equals_sin_altitude() -> None:
    # On a perfectly flat surface the normal is straight up. Lambertian
    # shade reduces to sin(altitude).
    flat = np.full((6, 6), 100.0)
    for alt in (0.0, 30.0, 45.0, 90.0):
        shade = hillshade(flat, cell_size_m=1.0, altitude_deg=alt)
        np.testing.assert_allclose(shade, np.sin(np.deg2rad(alt)), atol=1e-12)


def test_hillshade_tilted_plane_is_constant() -> None:
    # np.gradient is constant on a constant-gradient plane (forward/
    # backward at edges, central in the interior all give the same
    # value). Hillshade is therefore constant across the array.
    cols = np.arange(8.0)
    plane = np.broadcast_to(-cols, (8, 8)).astype(np.float64).copy()
    shade = hillshade(plane, cell_size_m=1.0)
    np.testing.assert_allclose(shade, shade[0, 0], atol=1e-12)


def test_hillshade_east_facing_slope_lit_by_east_sun() -> None:
    # z = -c is an east-facing 45-deg slope. A sun due east at 45-deg
    # altitude lies along the surface normal: shade = 1.
    cols = np.arange(8.0)
    plane = np.broadcast_to(-cols, (8, 8)).astype(np.float64).copy()
    shade = hillshade(plane, cell_size_m=1.0, azimuth_deg=90.0, altitude_deg=45.0)
    np.testing.assert_allclose(shade, 1.0, atol=1e-12)


def test_hillshade_east_facing_slope_self_shadowed_by_west_sun() -> None:
    cols = np.arange(8.0)
    plane = np.broadcast_to(-cols, (8, 8)).astype(np.float64).copy()
    shade = hillshade(plane, cell_size_m=1.0, azimuth_deg=270.0, altitude_deg=45.0)
    # Negative dot product clipped to zero: self-shadowed.
    np.testing.assert_allclose(shade, 0.0, atol=1e-12)


def test_hillshade_nan_propagates() -> None:
    dem = np.full((6, 6), 100.0)
    dem[2, 2] = np.nan
    shade = hillshade(dem, cell_size_m=1.0)
    assert np.isnan(shade[2, 2])
    assert np.isfinite(shade[0, 0])


def test_hillshade_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        hillshade(np.zeros((3, 4, 5)), cell_size_m=1.0)


def test_hillshade_rejects_non_positive_cell_size() -> None:
    with pytest.raises(ValueError):
        hillshade(np.zeros((4, 4)), cell_size_m=-1.0)


def test_hillshade_rejects_out_of_range_altitude() -> None:
    with pytest.raises(ValueError):
        hillshade(np.zeros((4, 4)), cell_size_m=1.0, altitude_deg=120.0)
    with pytest.raises(ValueError):
        hillshade(np.zeros((4, 4)), cell_size_m=1.0, altitude_deg=-1.0)


# ---------------------------------------------------------------------------
# Plot smoke tests
# ---------------------------------------------------------------------------


def test_plot_overlay_smoke() -> None:
    dem = _gaussian_hill(15)
    overlay = np.linspace(0.0, 1.0, dem.size).reshape(dem.shape)
    ax = plot_overlay(dem, overlay, cell_size_m=1.0, label="test", title="Overlay")
    # Two AxesImage artists: hillshade backdrop + overlay.
    assert len(ax.images) == 2
    assert ax.get_title() == "Overlay"
    assert "m" in ax.get_xlabel()
    assert "m" in ax.get_ylabel()
    plt.close(ax.figure)


def test_plot_overlay_axes_span_extent_in_metres() -> None:
    # 20 cells x 5 m/cell = 100 m on each side.
    dem = _gaussian_hill(20)
    overlay = np.linspace(0.1, 1.0, dem.size).reshape(dem.shape)
    ax = plot_overlay(dem, overlay, cell_size_m=5.0)
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    assert xlo == pytest.approx(0.0)
    assert xhi == pytest.approx(100.0)
    # imshow with extent=(0, W, H, 0) leaves the axis with y inverted:
    # ylo (drawn at the bottom of the figure) is at y=H, yhi at y=0.
    assert ylo == pytest.approx(100.0)
    assert yhi == pytest.approx(0.0)
    plt.close(ax.figure)


def test_plot_overlay_log_requires_positive_overlay() -> None:
    dem = _gaussian_hill(8)
    overlay = np.zeros_like(dem)
    with pytest.raises(ValueError):
        plot_overlay(dem, overlay, cell_size_m=1.0, log=True)


def test_plot_overlay_rejects_shape_mismatch() -> None:
    dem = np.zeros((5, 5))
    bad = np.zeros((4, 5))
    with pytest.raises(ValueError):
        plot_overlay(dem, bad, cell_size_m=1.0)


def test_plot_overlay_accepts_existing_axes() -> None:
    dem = _gaussian_hill(10)
    overlay = np.linspace(0.0, 1.0, dem.size).reshape(dem.shape)
    fig, ax = plt.subplots()
    returned = plot_overlay(dem, overlay, cell_size_m=1.0, ax=ax, colorbar=False)
    assert returned is ax
    plt.close(fig)


def test_plot_convergence_smoke() -> None:
    dem = _gaussian_hill(21, height=60.0)
    ax = plot_convergence(dem, cell_size_m=1.0)
    assert len(ax.images) == 2
    # Default plot_convergence draws white elevation contours on top.
    assert len(ax.collections) > 0
    plt.close(ax.figure)


def test_plot_convergence_contours_can_be_disabled() -> None:
    dem = _gaussian_hill(15, height=60.0)
    ax = plot_convergence(dem, cell_size_m=1.0, contours=False)
    assert len(ax.collections) == 0
    plt.close(ax.figure)


def test_plot_convergence_smoothing_can_be_disabled() -> None:
    dem = _gaussian_hill(15, height=60.0)
    ax = plot_convergence(dem, cell_size_m=1.0, smooth_sigma_m=0.0)
    assert len(ax.images) == 2
    plt.close(ax.figure)


def test_plot_convergence_rejects_negative_sigma() -> None:
    dem = _gaussian_hill(8, height=20.0)
    with pytest.raises(ValueError):
        plot_convergence(dem, cell_size_m=1.0, smooth_sigma_m=-1.0)


def test_plot_convergence_smoothing_handles_nan_cells() -> None:
    dem = _gaussian_hill(12, height=30.0)
    dem[4, 4] = np.nan
    ax = plot_convergence(dem, cell_size_m=1.0, smooth_sigma_m=2.0)
    assert len(ax.images) == 2
    plt.close(ax.figure)


def test_plot_overlay_contours_skipped_on_flat_dem() -> None:
    # A flat DEM has no contours to draw; matplotlib would warn if asked.
    flat = np.full((10, 10), 5.0)
    overlay = np.linspace(0.1, 1.0, flat.size).reshape(flat.shape)
    ax = plot_overlay(flat, overlay, cell_size_m=1.0, contours=True)
    assert len(ax.collections) == 0
    plt.close(ax.figure)


def test_plot_slope_smoke() -> None:
    dem = _gaussian_hill(15)
    ax = plot_slope(dem, cell_size_m=1.0)
    assert len(ax.images) == 2
    plt.close(ax.figure)


def test_plot_aspect_smoke() -> None:
    dem = _gaussian_hill(15)
    ax = plot_aspect(dem, cell_size_m=1.0)
    assert len(ax.images) == 2
    plt.close(ax.figure)


def test_plot_profile_curvature_smoke() -> None:
    dem = _gaussian_hill(15)
    ax = plot_profile_curvature(dem, cell_size_m=1.0)
    assert len(ax.images) == 2
    plt.close(ax.figure)


def test_plot_profile_curvature_handles_all_nan_field() -> None:
    # On a flat plate, profile curvature is NaN everywhere. The
    # plotter's robust-clip path must not blow up on an empty finite
    # subset.
    flat = np.full((10, 10), 5.0)
    ax = plot_profile_curvature(flat, cell_size_m=1.0)
    assert len(ax.images) == 2
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Heating
# ---------------------------------------------------------------------------


def test_plot_heating_smoke() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    dem = _gaussian_hill(32, height=60.0)
    when = datetime(2026, 6, 21, 12, 0, tzinfo=ZoneInfo("Europe/London"))
    ax = plot_heating(
        dem, cell_size_m=5.0, when=when, latitude_deg=54.2, longitude_deg=-2.3
    )
    # Hillshade backdrop + heating overlay = 2 images.
    assert len(ax.images) == 2
    # Default contours on.
    assert len(ax.collections) > 0
    # Axis labels are in metres (project convention).
    assert ax.get_xlabel() == "x (m)"
    assert ax.get_ylabel() == "y (m)"
    plt.close(ax.figure)


def test_plot_heating_contours_can_be_disabled() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    dem = _gaussian_hill(16, height=40.0)
    when = datetime(2026, 6, 21, 12, 0, tzinfo=ZoneInfo("Europe/London"))
    ax = plot_heating(
        dem,
        cell_size_m=5.0,
        when=when,
        latitude_deg=54.2,
        longitude_deg=-2.3,
        contours=False,
    )
    assert len(ax.collections) == 0
    plt.close(ax.figure)


def test_plot_heating_rejects_naive_datetime() -> None:
    from datetime import datetime

    dem = _gaussian_hill(16, height=40.0)
    when_naive = datetime(2026, 6, 21, 12, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        plot_heating(
            dem,
            cell_size_m=5.0,
            when=when_naive,
            latitude_deg=54.2,
            longitude_deg=-2.3,
        )
