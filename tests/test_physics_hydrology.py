"""Tests for thermal_model.physics.hydrology."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermal_model.physics import fill_pits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_hill(n: int, cell: float, height: float = 80.0) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    cx = cy = (n - 1) / 2.0
    sigma = n / 6.0
    return 400.0 + height * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


def _has_strict_pit(dem: np.ndarray) -> bool:
    """True if any finite, non-boundary, non-NaN-adjacent cell is strictly
    below all eight finite neighbours.
    """
    rows, cols = dem.shape
    nan_mask = np.isnan(dem)
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if nan_mask[r, c]:
                continue
            window = dem[r - 1 : r + 2, c - 1 : c + 2]
            if np.any(np.isnan(window)):
                continue
            others = np.delete(window.ravel(), 4)
            if dem[r, c] < others.min():
                return True
    return False


# ---------------------------------------------------------------------------
# Trivial properties
# ---------------------------------------------------------------------------


def test_no_op_on_descending_plane() -> None:
    # Tilted plane: every cell already drains monotonically to the right
    # boundary, so fill_pits is a no-op.
    cols = np.arange(8.0)
    plane = np.broadcast_to(5.0 - cols, (8, 8)).copy()
    out = fill_pits(plane)
    np.testing.assert_array_equal(out, plane)


def test_no_op_on_gaussian_hill() -> None:
    hill = _gaussian_hill(17, cell=1.0)
    out = fill_pits(hill)
    np.testing.assert_allclose(out, hill, atol=1e-12)


def test_filled_is_never_below_input() -> None:
    rng = np.random.default_rng(42)
    dem = rng.uniform(0, 10, size=(20, 20))
    # Punch a few pits.
    dem[5, 5] = -50
    dem[12, 8] = -30
    dem[3, 17] = -10
    out = fill_pits(dem)
    assert np.all(out >= dem - 1e-12)


def test_no_strict_pits_remain() -> None:
    rng = np.random.default_rng(7)
    dem = rng.uniform(0, 100, size=(25, 25))
    dem[10, 10] = -200
    dem[15, 15] = -50
    dem[20, 5] = -10
    assert _has_strict_pit(dem)
    out = fill_pits(dem)
    assert not _has_strict_pit(out)


# ---------------------------------------------------------------------------
# Specific fill behaviour
# ---------------------------------------------------------------------------


def test_isolated_pit_in_descending_plane_fills_to_spill() -> None:
    # 5 columns, descending east-to-west to give monotonic drainage.
    plane = np.broadcast_to(np.arange(5.0, 0.0, -1.0), (5, 5)).copy()
    plane[2, 1] = -5.0  # gouge a pit one column in from the high side
    out = fill_pits(plane)

    # The pit's lowest neighbour is the plane value at (2,2) = 3.
    assert math.isclose(out[2, 1], 3.0, abs_tol=1e-12)
    # All other cells unchanged.
    expected = plane.copy()
    expected[2, 1] = 3.0
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_inverted_gaussian_summit_is_filled_to_rim() -> None:
    n = 33
    hill = _gaussian_hill(n, cell=1.0, height=80.0)
    inverted = float(np.nanmax(hill)) - hill
    filled = fill_pits(inverted)
    centre = (n - 1) // 2

    boundary_min = min(
        inverted[0, :].min(),
        inverted[-1, :].min(),
        inverted[:, 0].min(),
        inverted[:, -1].min(),
    )
    # The deep pit at the (inverted) summit is raised at least to the
    # lowest spill point on the array boundary.
    assert filled[centre, centre] >= boundary_min - 1e-9
    assert filled[centre, centre] > inverted[centre, centre] + 1.0
    # Boundary cells themselves are never raised.
    np.testing.assert_array_equal(filled[0, :], inverted[0, :])
    np.testing.assert_array_equal(filled[-1, :], inverted[-1, :])
    np.testing.assert_array_equal(filled[:, 0], inverted[:, 0])
    np.testing.assert_array_equal(filled[:, -1], inverted[:, -1])


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_cells_pass_through() -> None:
    dem = np.full((10, 10), 5.0)
    dem[3, 3] = np.nan
    dem[7, 5] = np.nan
    out = fill_pits(dem)
    assert np.isnan(out[3, 3])
    assert np.isnan(out[7, 5])
    finite = ~np.isnan(out)
    np.testing.assert_allclose(out[finite], 5.0, atol=1e-12)


def test_nan_acts_as_drainage_outlet() -> None:
    # 7x7 plateau at z=10, except one interior pit at z=0 with one
    # NaN neighbour. The pit drains via the NaN, not via the rim, so
    # it fills only to its lowest finite neighbour.
    dem = np.full((7, 7), 10.0)
    dem[3, 3] = 0.0
    dem[3, 4] = np.nan
    out = fill_pits(dem)
    # The pit cell has a NaN neighbour, so it is itself a seed cell.
    # It is never raised because its draining path runs into the NaN.
    assert math.isclose(out[3, 3], 0.0, abs_tol=1e-12)


def test_all_nan_input_passes_through() -> None:
    dem = np.full((6, 6), np.nan)
    out = fill_pits(dem)
    assert np.all(np.isnan(out))


# ---------------------------------------------------------------------------
# Epsilon mode
# ---------------------------------------------------------------------------


def test_epsilon_zero_leaves_flats_flat() -> None:
    dem = np.full((7, 7), 5.0)
    dem[1:-1, 1:-1] = 0.0
    out = fill_pits(dem, epsilon=0.0)
    np.testing.assert_allclose(out[1:-1, 1:-1], 5.0, atol=1e-12)


def test_epsilon_creates_strictly_monotone_fill_inside_a_pit() -> None:
    dem = np.full((7, 7), 5.0)
    dem[1:-1, 1:-1] = 0.0
    out = fill_pits(dem, epsilon=0.01)
    # Outer ring of the filled basin (distance 1 from rim) is at the
    # spill elevation; the centre cell sits one or more BFS steps deeper
    # and is therefore strictly higher.
    assert out[3, 3] > out[1, 1] + 1e-9
    # Boundary itself is untouched.
    np.testing.assert_allclose(out[0, :], 5.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_negative_epsilon() -> None:
    dem = np.zeros((4, 4))
    with pytest.raises(ValueError):
        fill_pits(dem, epsilon=-1e-6)


def test_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        fill_pits(np.zeros((3, 4, 5)))


def test_rejects_too_small_dem() -> None:
    with pytest.raises(ValueError):
        fill_pits(np.zeros((1, 5)))
