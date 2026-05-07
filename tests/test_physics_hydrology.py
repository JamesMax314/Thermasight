"""Tests for thermal_model.physics.hydrology."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from thermal_model.physics import fill_pits, resolve_flats

_HAS_RICHDEM = importlib.util.find_spec("richdem") is not None

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
    out = fill_pits(plane, use_richdem=False)
    np.testing.assert_array_equal(out, plane)


def test_no_op_on_gaussian_hill() -> None:
    hill = _gaussian_hill(17, cell=1.0)
    out = fill_pits(hill, use_richdem=False)
    np.testing.assert_allclose(out, hill, atol=1e-12)


def test_filled_is_never_below_input() -> None:
    rng = np.random.default_rng(42)
    dem = rng.uniform(0, 10, size=(20, 20))
    # Punch a few pits.
    dem[5, 5] = -50
    dem[12, 8] = -30
    dem[3, 17] = -10
    out = fill_pits(dem, use_richdem=False)
    assert np.all(out >= dem - 1e-12)


def test_no_strict_pits_remain() -> None:
    rng = np.random.default_rng(7)
    dem = rng.uniform(0, 100, size=(25, 25))
    dem[10, 10] = -200
    dem[15, 15] = -50
    dem[20, 5] = -10
    assert _has_strict_pit(dem)
    out = fill_pits(dem, use_richdem=False)
    assert not _has_strict_pit(out)


# ---------------------------------------------------------------------------
# Specific fill behaviour
# ---------------------------------------------------------------------------


def test_isolated_pit_in_descending_plane_fills_to_spill() -> None:
    # 5 columns, descending east-to-west to give monotonic drainage.
    plane = np.broadcast_to(np.arange(5.0, 0.0, -1.0), (5, 5)).copy()
    plane[2, 1] = -5.0  # gouge a pit one column in from the high side
    out = fill_pits(plane, use_richdem=False)

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
    filled = fill_pits(inverted, use_richdem=False)
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
    out = fill_pits(dem, use_richdem=False)
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
    out = fill_pits(dem, use_richdem=False)
    # The pit cell has a NaN neighbour, so it is itself a seed cell.
    # It is never raised because its draining path runs into the NaN.
    assert math.isclose(out[3, 3], 0.0, abs_tol=1e-12)


def test_all_nan_input_passes_through() -> None:
    dem = np.full((6, 6), np.nan)
    out = fill_pits(dem, use_richdem=False)
    assert np.all(np.isnan(out))


# ---------------------------------------------------------------------------
# Epsilon mode
# ---------------------------------------------------------------------------


def test_epsilon_zero_leaves_flats_flat() -> None:
    dem = np.full((7, 7), 5.0)
    dem[1:-1, 1:-1] = 0.0
    out = fill_pits(dem, epsilon=0.0, use_richdem=False)
    np.testing.assert_allclose(out[1:-1, 1:-1], 5.0, atol=1e-12)


def test_epsilon_creates_strictly_monotone_fill_inside_a_pit() -> None:
    dem = np.full((7, 7), 5.0)
    dem[1:-1, 1:-1] = 0.0
    out = fill_pits(dem, epsilon=0.01, use_richdem=False)
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


# ---------------------------------------------------------------------------
# richdem path (skipped when richdem is not installed)
# ---------------------------------------------------------------------------


def test_rejects_use_richdem_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("thermal_model.physics.hydrology._have_richdem", lambda: False)
    with pytest.raises(ImportError):
        fill_pits(np.zeros((4, 4)), use_richdem=True)


@pytest.mark.skipif(not _HAS_RICHDEM, reason="richdem not installed")
def test_richdem_fills_strict_pit_to_spill() -> None:
    # On a strict pit inside a sloping plane, both backends should
    # produce the same fill height (no flats, so the epsilon-bump
    # discrepancy is irrelevant).
    plane = np.broadcast_to(np.arange(5.0, 0.0, -1.0), (5, 5)).copy()
    plane[2, 1] = -5.0
    out_np = fill_pits(plane, use_richdem=False)
    out_rd = fill_pits(plane, use_richdem=True)
    np.testing.assert_allclose(out_rd, out_np, atol=1e-9)


@pytest.mark.skipif(not _HAS_RICHDEM, reason="richdem not installed")
def test_richdem_agrees_with_numpy_on_random_dem() -> None:
    rng = np.random.default_rng(123)
    dem = rng.uniform(0.0, 100.0, size=(30, 30))
    dem[10, 10] = -100.0
    dem[15, 22] = -50.0
    out_np = fill_pits(dem, use_richdem=False)
    out_rd = fill_pits(dem, use_richdem=True)
    # Output >= input on both, no strict pits remain on either, and
    # the per-cell results agree to within FP slop.
    assert np.all(out_rd >= dem - 1e-9)
    assert not _has_strict_pit(out_rd)
    np.testing.assert_allclose(out_rd, out_np, atol=1e-6)


# ---------------------------------------------------------------------------
# resolve_flats
# ---------------------------------------------------------------------------


def test_resolve_flats_numpy_perturbs_only_flat_cells() -> None:
    # Plateau-on-a-slope: the central plateau is a flat region, the
    # surrounding tilted plane drains. Only plateau cells should be
    # perturbed; the tilted cells must be unchanged.
    rows, cols = 11, 11
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    plane = -xx  # drains east
    dem = plane.copy()
    # Stamp a 5x5 flat plateau in the middle, raised above the plane.
    dem[3:8, 3:8] = 100.0
    # Pit-fill the inverted DEM (where the plateau is a pit) with
    # epsilon=0 to leave the flat exactly flat.
    inverted = float(dem.max()) - dem
    filled = fill_pits(inverted, epsilon=0.0, use_richdem=False)

    out = resolve_flats(filled, use_richdem=False, fallback_amplitude=1e-6)

    # Cells outside the plateau should be untouched.
    untouched = (xx < 3) | (xx > 7) | (yy < 3) | (yy > 7)
    np.testing.assert_array_equal(out[untouched], filled[untouched])
    # At least one plateau-interior cell should have moved.
    interior = (xx >= 4) & (xx <= 6) & (yy >= 4) & (yy <= 6)
    assert np.any(out[interior] != filled[interior])
    # The perturbation magnitude is bounded by the noise scale.
    assert np.max(np.abs(out - filled)) < 1e-4


def test_resolve_flats_amplitude_zero_is_noop() -> None:
    rows, cols = 9, 9
    dem = np.full((rows, cols), 5.0)  # entirely flat
    out = resolve_flats(dem, use_richdem=False, fallback_amplitude=0.0)
    np.testing.assert_array_equal(out, dem.astype(np.float64))


def test_resolve_flats_preserves_nan_numpy() -> None:
    dem = np.full((8, 8), 5.0)
    dem[2, 2] = np.nan
    out = resolve_flats(dem, use_richdem=False)
    assert np.isnan(out[2, 2])


def test_resolve_flats_validates_shape() -> None:
    with pytest.raises(ValueError):
        resolve_flats(np.zeros((3, 4, 5)))
    with pytest.raises(ValueError):
        resolve_flats(np.zeros((1, 4)))


def test_resolve_flats_validates_amplitude() -> None:
    with pytest.raises(ValueError):
        resolve_flats(np.zeros((4, 4)), fallback_amplitude=-1.0)


def test_resolve_flats_rejects_use_richdem_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("thermal_model.physics.hydrology._have_richdem", lambda: False)
    with pytest.raises(ImportError):
        resolve_flats(np.zeros((4, 4)), use_richdem=True)


def test_resolve_flats_numpy_breaks_streak_artefact() -> None:
    # Construct a small DEM with a sharp ridge butting onto a flat
    # plateau on the lee side. After fill+resolve+accumulate, the
    # accumulation field on the plateau should not have parallel
    # ridge-perpendicular streaks (i.e. column-to-column variance
    # should be far from a near-constant ramp).
    rows, cols = 25, 25
    dem = np.zeros((rows, cols), dtype=np.float64)
    # West half: tilted plane rising to the ridge.
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    dem += np.where(xx < 12, xx * 5.0, 0.0)
    # Ridge column at x = 12.
    dem[:, 12] = 80.0
    # East half: flat plateau (fill_pits will leave it flat after
    # raising it to spill).
    dem[:, 13:] = 60.0

    inverted = float(dem.max()) - dem
    filled = fill_pits(inverted, epsilon=0.0, use_richdem=False)
    resolved = resolve_flats(filled, use_richdem=False)

    # The plateau on the inverted DEM is a pit-bottom flat. After
    # resolution, the per-cell perturbation should differ row-to-row
    # within the plateau, breaking the would-be ridge-aligned ramp.
    plateau = resolved[:, 13:]
    row_means = plateau.mean(axis=1)
    # If the artefact were present, row_means would be (near) equal
    # across rows; resolve_flats injects per-cell noise, so we expect
    # finite per-row variance.
    assert row_means.std() > 0


@pytest.mark.skipif(not _HAS_RICHDEM, reason="richdem not installed")
def test_resolve_flats_richdem_runs_on_filled_dem() -> None:
    # Smoke + invariants on the richdem path.
    rows, cols = 20, 20
    rng = np.random.default_rng(2)
    dem = rng.uniform(0.0, 50.0, size=(rows, cols))
    dem[5:15, 5:15] = 30.0  # flat plateau
    inverted = float(dem.max()) - dem
    filled = fill_pits(inverted, epsilon=0.0, use_richdem=True)
    resolved = resolve_flats(filled, use_richdem=True)
    # Same shape, NaN preserved (none here), no NaN introduced.
    assert resolved.shape == dem.shape
    assert np.all(np.isfinite(resolved))


@pytest.mark.skipif(not _HAS_RICHDEM, reason="richdem not installed")
def test_resolve_flats_richdem_preserves_nan() -> None:
    dem = np.full((8, 8), 5.0)
    dem[2, 2] = np.nan
    out = resolve_flats(dem, use_richdem=True)
    assert np.isnan(out[2, 2])


@pytest.mark.skipif(not _HAS_RICHDEM, reason="richdem not installed")
def test_richdem_preserves_nan() -> None:
    dem = np.full((8, 8), 5.0)
    dem[2, 2] = np.nan
    dem[5, 6] = np.nan
    out = fill_pits(dem, use_richdem=True)
    assert np.isnan(out[2, 2])
    assert np.isnan(out[5, 6])
    finite_mask = ~np.isnan(out)
    np.testing.assert_allclose(out[finite_mask], 5.0, atol=1e-12)
