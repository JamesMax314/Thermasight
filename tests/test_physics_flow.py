"""Tests for thermal_model.physics.flow (D-infinity directions + accumulation)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

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
