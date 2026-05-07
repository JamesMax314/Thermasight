"""Tests for thermal_model.physics.flow (D-infinity directions + accumulation)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from thermal_model.physics import dinf_flow_directions, flow_accumulation
from thermal_model.physics.flow import _flow_accumulation_numpy

_HAS_RICHDEM = importlib.util.find_spec("richdem") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_hill(n: int, height: float = 80.0) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    cx = cy = (n - 1) / 2.0
    sigma = n / 6.0
    return 400.0 + height * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


# ---------------------------------------------------------------------------
# D-infinity directions
# ---------------------------------------------------------------------------


def test_dinf_directions_tilted_east_plane() -> None:
    # z = -c: every cell with an east neighbour drains pure east
    # (theta = 0, slope = 1/cell). The rightmost column is the natural
    # outlet and has no downhill facet, so it returns NaN.
    cols = np.arange(8.0)
    plane = np.broadcast_to(-cols, (8, 8)).astype(np.float64).copy()
    angle, slope = dinf_flow_directions(plane, cell_size_m=1.0)
    np.testing.assert_allclose(angle[:, :-1], 0.0, atol=1e-12)
    np.testing.assert_allclose(slope[:, :-1], 1.0, atol=1e-12)
    assert np.all(np.isnan(angle[:, -1]))
    assert np.all(np.isnan(slope[:, -1]))


def test_dinf_directions_tilted_south_plane() -> None:
    # z = -r: every cell with a south neighbour drains pure south
    # (theta = 3*pi/2, slope = 1/cell). The bottom row is the outlet.
    rows = np.arange(8.0).reshape(-1, 1)
    plane = np.broadcast_to(-rows, (8, 8)).astype(np.float64).copy()
    angle, slope = dinf_flow_directions(plane, cell_size_m=1.0)
    np.testing.assert_allclose(angle[:-1, :], 3 * np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(slope[:-1, :], 1.0, atol=1e-12)
    assert np.all(np.isnan(angle[-1, :]))
    assert np.all(np.isnan(slope[-1, :]))


def test_dinf_directions_diagonal_southwest_plane() -> None:
    # z = c - r: gradient points NE on the surface, so flow goes SW.
    # SW corresponds to theta = 5*pi/4 in math convention.
    n = 8
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    plane = xx - yy
    angle, slope = dinf_flow_directions(plane, cell_size_m=1.0)
    # Interior cells should be exactly 5*pi/4. Edges may snap to a
    # cardinal facet because one of the diagonal neighbours is off-grid;
    # check the strict interior.
    interior = angle[1:-1, 1:-1]
    np.testing.assert_allclose(interior, 5 * np.pi / 4, atol=1e-12)
    np.testing.assert_allclose(slope[1:-1, 1:-1], np.sqrt(2.0), atol=1e-12)


def test_dinf_directions_flat_returns_nan() -> None:
    # A flat surface has no downhill anywhere.
    flat = np.full((6, 6), 5.0)
    angle, slope = dinf_flow_directions(flat, cell_size_m=1.0)
    assert np.all(np.isnan(angle))
    assert np.all(np.isnan(slope))


def test_dinf_directions_pit_returns_nan() -> None:
    # A flat plateau with one strict pit at the centre. The pit cell
    # has no downhill direction: every facet points uphill.
    dem = np.full((7, 7), 10.0)
    dem[3, 3] = 0.0
    angle, slope = dinf_flow_directions(dem, cell_size_m=1.0)
    assert np.isnan(angle[3, 3])
    assert np.isnan(slope[3, 3])


# ---------------------------------------------------------------------------
# Flow accumulation - numpy fallback
# ---------------------------------------------------------------------------


def test_accumulation_tilted_east_is_column_count() -> None:
    # On z = -c, every cell flows pure east. Each cell at column c
    # collects all c upstream cells in its row, plus its own
    # contribution: acc = c + 1.
    n = 6
    cols = np.arange(n, dtype=np.float64)
    plane = np.broadcast_to(-cols, (n, n)).copy()
    acc = flow_accumulation(plane, cell_size_m=1.0, use_richdem=False)
    expected = np.broadcast_to(cols + 1, (n, n)).astype(np.float64)
    np.testing.assert_allclose(acc, expected, atol=1e-12)


def test_accumulation_self_contribution_only_at_global_peak() -> None:
    # On a Gaussian hill the summit has no upstream, so acc == 1.
    # Every other finite cell has acc >= 1.
    hill = _gaussian_hill(15)
    acc = flow_accumulation(hill, cell_size_m=1.0, use_richdem=False)
    centre = (hill.shape[0] - 1) // 2
    assert acc[centre, centre] == pytest.approx(1.0, abs=1e-12)
    assert np.all(acc >= 1.0 - 1e-12)


def test_accumulation_is_nonnegative_and_self_lower_bound() -> None:
    rng = np.random.default_rng(11)
    dem = rng.uniform(0.0, 100.0, size=(20, 20))
    acc = flow_accumulation(dem, cell_size_m=2.0, use_richdem=False)
    assert np.all(acc >= 1.0 - 1e-9)


def test_accumulation_default_matches_unit_weights() -> None:
    rng = np.random.default_rng(3)
    dem = rng.uniform(0.0, 50.0, size=(12, 12))
    acc_default = flow_accumulation(dem, cell_size_m=1.0, use_richdem=False)
    acc_unit = flow_accumulation(
        dem, cell_size_m=1.0, weights=np.ones_like(dem), use_richdem=False
    )
    np.testing.assert_allclose(acc_default, acc_unit, atol=1e-12)


def test_accumulation_scales_linearly_with_weights() -> None:
    rng = np.random.default_rng(4)
    dem = rng.uniform(0.0, 50.0, size=(10, 10))
    w1 = np.full_like(dem, 1.0)
    w3 = np.full_like(dem, 3.0)
    a1 = flow_accumulation(dem, cell_size_m=1.0, weights=w1, use_richdem=False)
    a3 = flow_accumulation(dem, cell_size_m=1.0, weights=w3, use_richdem=False)
    np.testing.assert_allclose(a3, 3.0 * a1, atol=1e-9)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_accumulation_nan_passes_through() -> None:
    n = 8
    cols = np.arange(n, dtype=np.float64)
    plane = np.broadcast_to(-cols, (n, n)).astype(np.float64).copy()
    plane[2, 2] = np.nan
    acc = flow_accumulation(plane, cell_size_m=1.0, use_richdem=False)
    assert np.isnan(acc[2, 2])
    assert np.all(np.isfinite(acc[~np.isnan(plane)]))


def test_accumulation_all_nan_returns_all_nan() -> None:
    dem = np.full((5, 5), np.nan)
    acc = flow_accumulation(dem, cell_size_m=1.0, use_richdem=False)
    assert np.all(np.isnan(acc))


# ---------------------------------------------------------------------------
# Inverted-DEM use case (the project's reason for existing)
# ---------------------------------------------------------------------------


def test_accumulation_runs_on_filled_inverted_hill() -> None:
    # End-to-end shape check: pit-fill the inverted DEM, then
    # accumulate. This is the exact pipeline used by the convergence
    # map. We don't pin specific values here (that's Phase 1's
    # validation gate); just confirm finite, non-negative output.
    from thermal_model.physics import fill_pits

    hill = _gaussian_hill(21, height=60.0)
    inverted = float(np.nanmax(hill)) - hill
    filled = fill_pits(inverted, epsilon=1e-3)
    acc = flow_accumulation(filled, cell_size_m=1.0, use_richdem=False)
    assert np.all(np.isfinite(acc))
    assert np.all(acc >= 1.0 - 1e-9)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        flow_accumulation(np.zeros((3, 4, 5)), cell_size_m=1.0)


def test_rejects_non_positive_cell_size() -> None:
    with pytest.raises(ValueError):
        flow_accumulation(np.zeros((4, 4)), cell_size_m=0.0)
    with pytest.raises(ValueError):
        flow_accumulation(np.zeros((4, 4)), cell_size_m=-1.0)


def test_rejects_too_small_dem() -> None:
    with pytest.raises(ValueError):
        flow_accumulation(np.zeros((1, 5)), cell_size_m=1.0)


def test_rejects_mismatched_weights_shape() -> None:
    dem = np.zeros((5, 5))
    bad = np.zeros((4, 5))
    with pytest.raises(ValueError):
        _flow_accumulation_numpy(dem, cell_size_m=1.0, weights=bad)


def test_rejects_use_richdem_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the availability check to report False even if richdem
    # happens to be installed in the test env.
    monkeypatch.setattr("thermal_model.physics.flow._have_richdem", lambda: False)
    with pytest.raises(ImportError):
        flow_accumulation(np.zeros((4, 4)), cell_size_m=1.0, use_richdem=True)


# ---------------------------------------------------------------------------
# richdem path (smoke test only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_RICHDEM, reason="richdem not installed")
def test_richdem_path_runs_and_agrees_in_broad_strokes() -> None:
    from thermal_model.physics import fill_pits

    rng = np.random.default_rng(5)
    dem = fill_pits(rng.uniform(0.0, 30.0, size=(16, 16)), epsilon=1e-3)
    acc_np = flow_accumulation(dem, cell_size_m=1.0, use_richdem=False)
    acc_rd = flow_accumulation(dem, cell_size_m=1.0, use_richdem=True)
    assert acc_np.shape == acc_rd.shape == dem.shape
    # Both should sum upstream cells; totals should be in the same
    # ballpark (within 20%) on a small random fixture. Pixel-scale
    # disagreement is allowed because of flat-resolution differences.
    sum_np = float(np.nansum(acc_np))
    sum_rd = float(np.nansum(acc_rd))
    assert sum_rd == pytest.approx(sum_np, rel=0.2)


# ---------------------------------------------------------------------------
# Property-based: rotation equivariance and cell-size scaling
# ---------------------------------------------------------------------------


def _angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest unsigned angular distance between two angles in radians."""
    delta = (a - b) % (2.0 * np.pi)
    return np.minimum(delta, 2.0 * np.pi - delta)


# Use a tilted random DEM so flow direction selection has no ties:
# overlay a small unique-per-cell tilt on a random base. This avoids
# argmax tie-breaking ambiguity that would otherwise spoil exact
# rotation equivariance.
def _tilted_random_dem(seed: int, shape: tuple[int, int] = (7, 7)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(-10.0, 10.0, size=shape)
    yy, xx = np.indices(shape).astype(np.float64)
    return base + 100.0 * yy + 100.0 * xx


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


@given(seed=st.integers(min_value=0, max_value=9999), k=st.integers(0, 3))
@settings(max_examples=30, deadline=None)
def test_dinf_directions_rotate_under_rot90(seed: int, k: int) -> None:
    # In math convention (CCW from east), np.rot90 rotates the
    # array CCW so every flow direction also rotates CCW: angle
    # gains k*pi/2.
    dem = _tilted_random_dem(seed)
    angle_orig, slope_orig = dinf_flow_directions(dem, cell_size_m=1.0)
    angle_rot, slope_rot = dinf_flow_directions(np.rot90(dem, k=k), cell_size_m=1.0)

    expected_angle = (np.rot90(angle_orig, k=k) + k * (np.pi / 2)) % (2.0 * np.pi)
    expected_slope = np.rot90(slope_orig, k=k)
    mask = np.isfinite(angle_rot) & np.isfinite(expected_angle)
    diffs = _angular_diff(angle_rot, expected_angle)
    assert np.all(diffs[mask] < 1e-9)
    np.testing.assert_allclose(slope_rot[mask], expected_slope[mask], atol=1e-12)


@given(
    seed=st.integers(min_value=0, max_value=9999),
    scale=st.floats(min_value=0.5, max_value=10.0, allow_nan=False),
)
@settings(max_examples=30, deadline=None)
def test_dinf_slopes_scale_inversely_with_cell_size(seed: int, scale: float) -> None:
    # Steepest-facet slope is a tangent (m of rise per m of run);
    # scaling cell_size_m by k while keeping the array fixed scales
    # the run by k and so the tangent by 1/k. Direction is invariant.
    dem = _tilted_random_dem(seed)
    angle_a, slope_a = dinf_flow_directions(dem, cell_size_m=1.0)
    angle_b, slope_b = dinf_flow_directions(dem, cell_size_m=scale)
    mask = np.isfinite(slope_a) & np.isfinite(slope_b)
    np.testing.assert_allclose(
        slope_b[mask], slope_a[mask] / scale, rtol=1e-9, atol=1e-12
    )
    diffs = _angular_diff(angle_b, angle_a)
    assert np.all(diffs[mask] < 1e-9)


@given(seed=st.integers(min_value=0, max_value=9999), k=st.integers(0, 3))
@settings(max_examples=20, deadline=None)
def test_flow_accumulation_equivariant_under_rot90(seed: int, k: int) -> None:
    # The eight D-infinity facets are rotationally symmetric: a 90 deg
    # rotation of the input maps each facet to another facet without
    # changing slopes. Combined with our deterministic argmax
    # tie-breaking, that should make accumulation exactly equivariant.
    dem = _tilted_random_dem(seed)
    acc_orig = flow_accumulation(dem, cell_size_m=1.0, use_richdem=False)
    acc_rot = flow_accumulation(np.rot90(dem, k=k), cell_size_m=1.0, use_richdem=False)
    expected = np.rot90(acc_orig, k=k)
    np.testing.assert_allclose(acc_rot, expected, atol=1e-9)


@given(
    seed=st.integers(min_value=0, max_value=9999),
    scale=st.floats(min_value=0.5, max_value=10.0, allow_nan=False),
)
@settings(max_examples=20, deadline=None)
def test_flow_accumulation_invariant_under_cell_size_scaling(
    seed: int, scale: float
) -> None:
    # Default flow_accumulation returns upstream cell count, which is
    # dimensionless; cell-size scaling affects only the slope
    # magnitudes used for facet selection (proportionally), not the
    # selected facet itself. So accumulation must be unchanged.
    dem = _tilted_random_dem(seed)
    acc_a = flow_accumulation(dem, cell_size_m=1.0, use_richdem=False)
    acc_b = flow_accumulation(dem, cell_size_m=scale, use_richdem=False)
    np.testing.assert_allclose(acc_a, acc_b, atol=1e-9)
