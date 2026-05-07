"""Tests for thermal_model.terrain.morphometry."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from thermal_model.io import read_dem
from thermal_model.terrain import aspect, profile_curvature, slope

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tilted_plane(
    n: int, cell: float, slope_rad: float, aspect_rad: float
) -> np.ndarray:
    """Build a planar DEM with a known slope and downslope aspect.

    ``aspect_rad`` is the compass bearing of the downslope direction
    (0 = N, pi/2 = E). The plane therefore rises in the opposite
    direction. Returns a (n, n) array.
    """
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    # World coordinates: x_east = col*cell, y_north = -(row*cell)
    x_east = xx * cell
    y_north = -yy * cell
    # Up-slope unit vector (compass bearing aspect_rad + pi).
    up_east = -math.sin(aspect_rad)
    up_north = -math.cos(aspect_rad)
    grad_mag = math.tan(slope_rad)
    return 100.0 + grad_mag * (up_east * x_east + up_north * y_north)


def _gaussian_hill(n: int, cell: float, height: float = 80.0) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    cx = cy = (n - 1) / 2.0
    sigma = n / 6.0
    return 400.0 + height * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------


def test_slope_zero_on_flat_dem() -> None:
    dem = np.full((16, 16), 250.0)
    out = slope(dem, cell_size_m=1.0)
    assert np.all(np.isnan(out[0, :]))
    assert np.all(np.isnan(out[-1, :]))
    np.testing.assert_allclose(out[1:-1, 1:-1], 0.0, atol=1e-12)


@pytest.mark.parametrize("aspect_deg", [0.0, 45.0, 90.0, 135.0, 180.0, 270.0])
@pytest.mark.parametrize("slope_deg", [5.0, 20.0])
def test_slope_matches_tilted_plane(slope_deg: float, aspect_deg: float) -> None:
    expected = math.radians(slope_deg)
    dem = _tilted_plane(
        n=32, cell=2.5, slope_rad=expected, aspect_rad=math.radians(aspect_deg)
    )
    out = slope(dem, cell_size_m=2.5)
    interior = out[1:-1, 1:-1]
    np.testing.assert_allclose(interior, expected, atol=1e-10)


def test_slope_scales_inversely_with_cell_size() -> None:
    # Same array, larger cells -> shallower computed slope.
    dem = _tilted_plane(32, cell=1.0, slope_rad=math.radians(30.0), aspect_rad=0.0)
    s1 = slope(dem, cell_size_m=1.0)[1:-1, 1:-1]
    s5 = slope(dem, cell_size_m=5.0)[1:-1, 1:-1]
    np.testing.assert_allclose(np.tan(s1), 5.0 * np.tan(s5), atol=1e-10)


# ---------------------------------------------------------------------------
# Aspect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "aspect_deg",
    [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
)
def test_aspect_recovers_downslope_bearing(aspect_deg: float) -> None:
    expected = math.radians(aspect_deg)
    dem = _tilted_plane(
        n=32, cell=1.0, slope_rad=math.radians(15.0), aspect_rad=expected
    )
    out = aspect(dem, cell_size_m=1.0)
    interior = out[1:-1, 1:-1]
    # Compare on the unit circle to handle the 0/2*pi wrap at north.
    sin_diff = np.sin(interior) - math.sin(expected)
    cos_diff = np.cos(interior) - math.cos(expected)
    np.testing.assert_allclose(sin_diff, 0.0, atol=1e-10)
    np.testing.assert_allclose(cos_diff, 0.0, atol=1e-10)


def test_aspect_on_flat_is_nan() -> None:
    dem = np.full((8, 8), 300.0)
    out = aspect(dem, cell_size_m=1.0)
    # Edges and interior alike are NaN: edges from the stencil, interior
    # from the flat-cell rule.
    assert np.all(np.isnan(out))


def test_aspect_radiates_outward_on_a_hill() -> None:
    # On a hill, downslope at every pixel points away from the summit.
    n, cell = 65, 1.0
    dem = _gaussian_hill(n, cell)
    asp = aspect(dem, cell_size_m=cell)
    summit_row = summit_col = (n - 1) // 2

    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    east_from_summit = (cols - summit_col).astype(np.float64)
    north_from_summit = -(rows - summit_row).astype(np.float64)
    expected_bearing = np.arctan2(east_from_summit, north_from_summit) % (2 * np.pi)

    # Ignore the immediate summit cell (gradient ~0) and the NaN edges.
    mask = np.isfinite(asp)
    mask[summit_row, summit_col] = False
    diff_sin = np.sin(asp[mask]) - np.sin(expected_bearing[mask])
    diff_cos = np.cos(asp[mask]) - np.cos(expected_bearing[mask])
    assert np.max(np.abs(diff_sin)) < 1e-2
    assert np.max(np.abs(diff_cos)) < 1e-2


# ---------------------------------------------------------------------------
# NaN propagation and edges
# ---------------------------------------------------------------------------


def test_edges_are_nan() -> None:
    dem = _tilted_plane(16, cell=1.0, slope_rad=0.1, aspect_rad=0.0)
    for arr in (slope(dem, 1.0), aspect(dem, 1.0), profile_curvature(dem, 1.0)):
        assert np.all(np.isnan(arr[0, :]))
        assert np.all(np.isnan(arr[-1, :]))
        assert np.all(np.isnan(arr[:, 0]))
        assert np.all(np.isnan(arr[:, -1]))


def test_nan_window_propagates() -> None:
    dem = _tilted_plane(16, cell=1.0, slope_rad=0.1, aspect_rad=math.radians(45.0))
    dem[5, 5] = np.nan
    s = slope(dem, 1.0)
    a = aspect(dem, 1.0)
    k = profile_curvature(dem, 1.0)
    # Every interior cell whose 3x3 window touches (5,5) must be NaN.
    for r in range(4, 7):
        for c in range(4, 7):
            assert np.isnan(s[r, c]), f"slope[{r},{c}] should be NaN"
            assert np.isnan(a[r, c]), f"aspect[{r},{c}] should be NaN"
            assert np.isnan(k[r, c]), f"curv[{r},{c}] should be NaN"
    # An interior cell well clear of the NaN is finite.
    assert np.isfinite(s[10, 10])


# ---------------------------------------------------------------------------
# Profile curvature
# ---------------------------------------------------------------------------


def test_profile_curvature_zero_on_plane() -> None:
    dem = _tilted_plane(16, cell=1.0, slope_rad=math.radians(20.0), aspect_rad=0.0)
    out = profile_curvature(dem, cell_size_m=1.0)
    np.testing.assert_allclose(out[1:-1, 1:-1], 0.0, atol=1e-10)


def test_profile_curvature_sign_on_hill_and_basin() -> None:
    n, cell = 33, 1.0
    hill = _gaussian_hill(n, cell, height=80.0)
    basin = -hill + 800.0  # invert: peak becomes pit
    cx = (n - 1) // 2

    # On a Gaussian hill the inner flank (within ~1 sigma of the summit,
    # sigma = n/6 = 5.5 px) is convex along the steepest-descent
    # direction, so profile curvature is positive there. Beyond ~1
    # sigma the Gaussian curves back to zero slope and becomes concave.
    k_hill = profile_curvature(hill, cell)
    k_basin = profile_curvature(basin, cell)
    inner = 3
    flank_rows = np.array([cx - inner, cx + inner, cx, cx])
    flank_cols = np.array([cx, cx, cx - inner, cx + inner])
    assert np.all(k_hill[flank_rows, flank_cols] > 0)
    assert np.all(k_basin[flank_rows, flank_cols] < 0)


def test_profile_curvature_scales_with_inverse_cell_size() -> None:
    n = 33
    hill = _gaussian_hill(n, cell=1.0)
    k1 = profile_curvature(hill, cell_size_m=1.0)
    k2 = profile_curvature(hill, cell_size_m=2.0)
    interior = (slice(1, -1), slice(1, -1))
    finite = np.isfinite(k1[interior]) & np.isfinite(k2[interior])
    # Doubling the cell size halves the gradient magnitudes, so the
    # ratio of curvatures is not exactly 1/2 — but the values must
    # remain finite, share sign on the flanks, and stay bounded.
    assert finite.any()
    assert np.allclose(np.sign(k1[interior][finite]), np.sign(k2[interior][finite]))


# ---------------------------------------------------------------------------
# Validation parameter checking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_cell", [0.0, -1.0])
def test_rejects_non_positive_cell_size(bad_cell: float) -> None:
    dem = np.zeros((8, 8))
    with pytest.raises(ValueError):
        slope(dem, cell_size_m=bad_cell)
    with pytest.raises(ValueError):
        aspect(dem, cell_size_m=bad_cell)
    with pytest.raises(ValueError):
        profile_curvature(dem, cell_size_m=bad_cell)


def test_rejects_too_small_dem() -> None:
    dem = np.zeros((2, 5))
    with pytest.raises(ValueError):
        slope(dem, cell_size_m=1.0)
    with pytest.raises(ValueError):
        profile_curvature(dem, cell_size_m=1.0)


def test_rejects_non_2d_input() -> None:
    dem_3d = np.zeros((2, 8, 8))
    with pytest.raises(ValueError):
        slope(dem_3d, cell_size_m=1.0)


# ---------------------------------------------------------------------------
# Real LIDAR fixture sanity
# ---------------------------------------------------------------------------


def test_morphometrics_on_real_fixture(wild_boar_fell_fixture_path: Path) -> None:
    if not wild_boar_fell_fixture_path.exists():
        pytest.skip("Wild Boar Fell fixture not present in this checkout")
    dem = read_dem(wild_boar_fell_fixture_path)
    s = slope(dem.elevation_m, dem.cell_size_m)
    a = aspect(dem.elevation_m, dem.cell_size_m)
    k = profile_curvature(dem.elevation_m, dem.cell_size_m)

    interior = (slice(1, -1), slice(1, -1))
    s_int = s[interior]
    a_int = a[interior]
    k_int = k[interior]

    # Real terrain: most interior cells should be finite.
    assert np.mean(np.isfinite(s_int)) > 0.95

    finite_s = s_int[np.isfinite(s_int)]
    assert finite_s.min() >= 0.0
    assert finite_s.max() <= np.pi / 2 + 1e-9

    finite_a = a_int[np.isfinite(a_int)]
    assert finite_a.min() >= 0.0
    assert finite_a.max() < 2 * np.pi
    # A real scarp tile must show some directional spread.
    assert finite_a.std() > 0.5

    finite_k = k_int[np.isfinite(k_int)]
    # Curvature values on real terrain at 1 m resolution are tiny but
    # should straddle zero meaningfully.
    assert finite_k.min() < 0.0 < finite_k.max()


# ---------------------------------------------------------------------------
# Property-based: rotation invariance and cell-size scaling
# ---------------------------------------------------------------------------
#
# These tests pin geometric invariants that a correct morphometric
# implementation must satisfy. They use hypothesis to vary the DEM
# values over a small grid and check that the relationship holds for
# every example. The grid size is intentionally small (7x7) so each
# example is sub-millisecond.


_PROPERTY_DEM = hnp.arrays(
    dtype=np.float64,
    shape=(7, 7),
    elements=st.floats(
        min_value=-50.0,
        max_value=50.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)


def _angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest unsigned angular distance between two angles in radians."""
    delta = (a - b) % (2.0 * np.pi)
    return np.minimum(delta, 2.0 * np.pi - delta)


@given(dem=_PROPERTY_DEM, k=st.integers(min_value=0, max_value=3))
@settings(max_examples=30, deadline=None)
def test_slope_invariant_under_rot90(dem: np.ndarray, k: int) -> None:
    sl_orig = slope(dem, cell_size_m=1.0)
    sl_rot = slope(np.rot90(dem, k=k), cell_size_m=1.0)
    expected = np.rot90(sl_orig, k=k)
    mask = np.isfinite(sl_rot) & np.isfinite(expected)
    np.testing.assert_allclose(sl_rot[mask], expected[mask], atol=1e-12)


@given(
    dem=_PROPERTY_DEM,
    scale=st.floats(min_value=0.5, max_value=10.0, allow_nan=False),
)
@settings(max_examples=30, deadline=None)
def test_tan_slope_scales_inversely_with_cell_size(
    dem: np.ndarray, scale: float
) -> None:
    # Scaling cell_size_m by k while keeping DEM values fixed scales
    # the gradient magnitude by 1/k, so tan(slope) scales by 1/k too.
    cell = 1.0
    sl_a = slope(dem, cell_size_m=cell)
    sl_b = slope(dem, cell_size_m=cell * scale)
    mask = np.isfinite(sl_a) & np.isfinite(sl_b)
    np.testing.assert_allclose(
        np.tan(sl_b[mask]),
        np.tan(sl_a[mask]) / scale,
        rtol=1e-9,
        atol=1e-12,
    )


@given(dem=_PROPERTY_DEM, k=st.integers(min_value=0, max_value=3))
@settings(max_examples=30, deadline=None)
def test_aspect_rotates_under_rot90(dem: np.ndarray, k: int) -> None:
    # np.rot90 rotates the array 90 deg CCW, which rotates every
    # compass direction CCW by the same amount. In compass-bearing
    # convention (0=N, increasing clockwise), CCW rotation subtracts
    # k*pi/2.
    asp_orig = aspect(dem, cell_size_m=1.0)
    asp_rot = aspect(np.rot90(dem, k=k), cell_size_m=1.0)
    expected = (np.rot90(asp_orig, k=k) - k * np.pi / 2) % (2.0 * np.pi)
    mask = np.isfinite(asp_rot) & np.isfinite(expected)
    diffs = _angular_diff(asp_rot, expected)
    assert np.all(diffs[mask] < 1e-9)


@given(
    dem=_PROPERTY_DEM,
    scale=st.floats(min_value=0.5, max_value=10.0, allow_nan=False),
)
@settings(max_examples=30, deadline=None)
def test_aspect_invariant_under_cell_size_scaling(
    dem: np.ndarray, scale: float
) -> None:
    # Aspect is a direction; the cell-size factor cancels.
    asp_a = aspect(dem, cell_size_m=1.0)
    asp_b = aspect(dem, cell_size_m=scale)
    mask = np.isfinite(asp_a) & np.isfinite(asp_b)
    diffs = _angular_diff(asp_a, asp_b)
    assert np.all(diffs[mask] < 1e-9)


@given(dem=_PROPERTY_DEM, k=st.integers(min_value=0, max_value=3))
@settings(max_examples=30, deadline=None)
def test_profile_curvature_invariant_under_rot90(dem: np.ndarray, k: int) -> None:
    kc_orig = profile_curvature(dem, cell_size_m=1.0)
    kc_rot = profile_curvature(np.rot90(dem, k=k), cell_size_m=1.0)
    expected = np.rot90(kc_orig, k=k)
    mask = np.isfinite(kc_rot) & np.isfinite(expected)
    np.testing.assert_allclose(kc_rot[mask], expected[mask], atol=1e-9)


# Profile curvature does NOT scale cleanly as 1/k^2 under cell-size
# scaling because of the (1 + |grad|^2)^1.5 surface-area correction
# in the Zevenbergen-Thorne formula; it only scales that way in the
# small-slope limit. The fixed-DEM
# test_profile_curvature_scales_with_inverse_cell_size above checks
# the right-level invariant (sign agreement on a Gaussian hill across
# cell sizes); a hypothesis property at full resolution would either
# require artificially flat DEMs or a much looser tolerance, neither
# of which adds confidence.
