"""Tests for thermal_model.physics.wind_tilt."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from thermal_model.physics import wind_tilt_ramp

# ---------------------------------------------------------------------------
# Cardinal-direction sign convention
#
# These pin the table in ``docs/model_correction.md`` §4 and the module
# docstring. The ramp must peak in the wind-*to* direction so that, once
# the surface is inverted, lee-side cells become the lowest points and
# warm air pools there.
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_dem() -> np.ndarray:
    return np.zeros((5, 5), dtype=np.float64)


def _argmax_2d(arr: np.ndarray) -> tuple[int, int]:
    flat = int(np.argmax(arr))
    return divmod(flat, arr.shape[1])


def test_wind_from_north_ramp_peaks_south(flat_dem: np.ndarray) -> None:
    out = wind_tilt_ramp(
        flat_dem, cell_size_m=1.0, wind_from_deg=0.0, wind_speed_ms=10.0
    )
    # Wind blows north -> south, so the lee (bottom row, large row index).
    assert _argmax_2d(out)[0] == flat_dem.shape[0] - 1
    # The ramp is uniform across columns under N->S wind.
    np.testing.assert_allclose(out[-1, :], out[-1, 0])


def test_wind_from_south_ramp_peaks_north(flat_dem: np.ndarray) -> None:
    out = wind_tilt_ramp(
        flat_dem, cell_size_m=1.0, wind_from_deg=180.0, wind_speed_ms=10.0
    )
    assert _argmax_2d(out)[0] == 0
    np.testing.assert_allclose(out[0, :], out[0, 0])


def test_wind_from_west_ramp_peaks_east(flat_dem: np.ndarray) -> None:
    out = wind_tilt_ramp(
        flat_dem, cell_size_m=1.0, wind_from_deg=270.0, wind_speed_ms=10.0
    )
    assert _argmax_2d(out)[1] == flat_dem.shape[1] - 1
    np.testing.assert_allclose(out[:, -1], out[0, -1])


def test_wind_from_east_ramp_peaks_west(flat_dem: np.ndarray) -> None:
    out = wind_tilt_ramp(
        flat_dem, cell_size_m=1.0, wind_from_deg=90.0, wind_speed_ms=10.0
    )
    assert _argmax_2d(out)[1] == 0
    np.testing.assert_allclose(out[:, 0], out[0, 0], atol=1e-12)


def test_wind_from_southwest_ramp_peaks_northeast(flat_dem: np.ndarray) -> None:
    out = wind_tilt_ramp(
        flat_dem, cell_size_m=1.0, wind_from_deg=225.0, wind_speed_ms=10.0
    )
    # SW -> NE: top-right corner (row=0, col=ncols-1).
    assert _argmax_2d(out) == (0, flat_dem.shape[1] - 1)


# ---------------------------------------------------------------------------
# Algebraic invariants
# ---------------------------------------------------------------------------


def test_ramp_is_added_to_dem_not_replaced() -> None:
    # An arbitrary DEM should be preserved in the difference: the ramp
    # is purely additive, not multiplicative or replacing.
    rng = np.random.default_rng(0)
    dem = rng.uniform(100.0, 900.0, size=(8, 8))
    out = wind_tilt_ramp(dem, cell_size_m=2.0, wind_from_deg=225.0, wind_speed_ms=5.0)
    delta = out - dem
    flat_out = wind_tilt_ramp(
        np.zeros_like(dem), cell_size_m=2.0, wind_from_deg=225.0, wind_speed_ms=5.0
    )
    np.testing.assert_allclose(delta, flat_out, atol=1e-12)


def test_ramp_is_linear_in_wind_speed() -> None:
    dem = np.zeros((6, 6), dtype=np.float64)
    a = wind_tilt_ramp(dem, cell_size_m=1.0, wind_from_deg=225.0, wind_speed_ms=3.0)
    b = wind_tilt_ramp(dem, cell_size_m=1.0, wind_from_deg=225.0, wind_speed_ms=6.0)
    np.testing.assert_allclose(b, 2.0 * a)


def test_ramp_is_linear_in_k() -> None:
    dem = np.zeros((6, 6), dtype=np.float64)
    a = wind_tilt_ramp(
        dem, cell_size_m=1.0, wind_from_deg=225.0, wind_speed_ms=5.0, k=0.02
    )
    b = wind_tilt_ramp(
        dem, cell_size_m=1.0, wind_from_deg=225.0, wind_speed_ms=5.0, k=0.04
    )
    np.testing.assert_allclose(b, 2.0 * a)


def test_reversing_wind_reverses_ramp() -> None:
    # Wind from theta and wind from theta+180 push warm air in opposite
    # directions, so the added ramps must be exact negatives.
    dem = np.zeros((6, 6), dtype=np.float64)
    a = wind_tilt_ramp(dem, cell_size_m=1.0, wind_from_deg=210.0, wind_speed_ms=5.0)
    b = wind_tilt_ramp(dem, cell_size_m=1.0, wind_from_deg=30.0, wind_speed_ms=5.0)
    np.testing.assert_allclose(b, -a, atol=1e-12)


def test_zero_wind_speed_is_a_noop() -> None:
    rng = np.random.default_rng(1)
    dem = rng.uniform(100.0, 900.0, size=(4, 7))
    out = wind_tilt_ramp(dem, cell_size_m=1.5, wind_from_deg=120.0, wind_speed_ms=0.0)
    np.testing.assert_array_equal(out, dem)


def test_zero_k_is_a_noop() -> None:
    rng = np.random.default_rng(2)
    dem = rng.uniform(100.0, 900.0, size=(4, 7))
    out = wind_tilt_ramp(
        dem, cell_size_m=1.5, wind_from_deg=120.0, wind_speed_ms=8.0, k=0.0
    )
    np.testing.assert_array_equal(out, dem)


def test_ramp_per_metre_slope_equals_k_times_wind_speed() -> None:
    # The tilt's per-metre slope along the wind-to direction must be
    # exactly k * |u|, regardless of cell size or grid size.
    dem = np.zeros((4, 5), dtype=np.float64)
    cell = 2.0
    wind_from = 270.0  # wind to east, so the ramp tilts purely along +east.
    u = 7.0
    k = 0.03
    out = wind_tilt_ramp(
        dem, cell_size_m=cell, wind_from_deg=wind_from, wind_speed_ms=u, k=k
    )
    east_slope_per_metre = (out[0, 1] - out[0, 0]) / cell
    np.testing.assert_allclose(east_slope_per_metre, k * u)


# ---------------------------------------------------------------------------
# NaN propagation and dtype preservation
# ---------------------------------------------------------------------------


def test_nan_cells_remain_nan() -> None:
    dem = np.array(
        [
            [100.0, 110.0, np.nan],
            [120.0, np.nan, 140.0],
            [np.nan, 160.0, 170.0],
        ],
        dtype=np.float64,
    )
    out = wind_tilt_ramp(dem, cell_size_m=1.0, wind_from_deg=225.0, wind_speed_ms=5.0)
    assert np.array_equal(np.isnan(out), np.isnan(dem))
    finite = ~np.isnan(dem)
    assert np.all(np.isfinite(out[finite]))


def test_dtype_preserved_float32() -> None:
    dem = np.zeros((4, 4), dtype=np.float32)
    out = wind_tilt_ramp(dem, cell_size_m=1.0, wind_from_deg=225.0, wind_speed_ms=5.0)
    assert out.dtype == np.float32


def test_dtype_preserved_float64() -> None:
    dem = np.zeros((4, 4), dtype=np.float64)
    out = wind_tilt_ramp(dem, cell_size_m=1.0, wind_from_deg=225.0, wind_speed_ms=5.0)
    assert out.dtype == np.float64


def test_shape_preserved() -> None:
    dem = np.zeros((3, 11), dtype=np.float64)
    out = wind_tilt_ramp(dem, cell_size_m=1.0, wind_from_deg=45.0, wind_speed_ms=4.0)
    assert out.shape == dem.shape


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_non_2d_dem() -> None:
    with pytest.raises(ValueError, match="2-D"):
        wind_tilt_ramp(
            np.zeros((4, 4, 4), dtype=np.float64),
            cell_size_m=1.0,
            wind_from_deg=0.0,
            wind_speed_ms=1.0,
        )


def test_rejects_integer_dem() -> None:
    with pytest.raises(ValueError, match="floating dtype"):
        wind_tilt_ramp(
            np.zeros((4, 4), dtype=np.int32),
            cell_size_m=1.0,
            wind_from_deg=0.0,
            wind_speed_ms=1.0,
        )


def test_rejects_non_positive_cell_size() -> None:
    dem = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="cell_size_m"):
        wind_tilt_ramp(dem, cell_size_m=0.0, wind_from_deg=0.0, wind_speed_ms=1.0)
    with pytest.raises(ValueError, match="cell_size_m"):
        wind_tilt_ramp(dem, cell_size_m=-1.0, wind_from_deg=0.0, wind_speed_ms=1.0)


def test_wind_direction_wraps_at_360() -> None:
    # wind_from_deg=370 must give the same ramp as wind_from_deg=10.
    dem = np.zeros((5, 5), dtype=np.float64)
    a = wind_tilt_ramp(dem, cell_size_m=1.0, wind_from_deg=10.0, wind_speed_ms=5.0)
    b = wind_tilt_ramp(dem, cell_size_m=1.0, wind_from_deg=370.0, wind_speed_ms=5.0)
    np.testing.assert_allclose(a, b)


# ---------------------------------------------------------------------------
# Property: world-space tilt is invariant under cell-size choice
#
# CLAUDE.md §7 calls out: "scaling cell size by k scales the wind-tilt
# ramp's pixel-space displacement by 1/k". The cleanest underlying
# invariant is that the per-metre slope along the wind-to direction is
# exactly k * |u|, no matter what cell_size_m or grid you pick. The
# per-pixel slope is then k * |u| * cell_size_m, so doubling cell_size
# halves the number of pixels needed to reach a given height jump.
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    wind_from_deg=st.floats(min_value=0.0, max_value=360.0),
    wind_speed_ms=st.floats(min_value=0.1, max_value=30.0),
    k=st.floats(min_value=0.001, max_value=0.1),
    cell_size_m=st.floats(min_value=0.5, max_value=25.0),
    rows=st.integers(min_value=4, max_value=12),
    cols=st.integers(min_value=4, max_value=12),
)
def test_per_metre_slope_along_wind_equals_k_times_speed(
    wind_from_deg: float,
    wind_speed_ms: float,
    k: float,
    cell_size_m: float,
    rows: int,
    cols: int,
) -> None:
    dem = np.zeros((rows, cols), dtype=np.float64)
    out = wind_tilt_ramp(
        dem,
        cell_size_m=cell_size_m,
        wind_from_deg=wind_from_deg,
        wind_speed_ms=wind_speed_ms,
        k=k,
    )

    # Pick the displacement vector along wind-to in (east, north) metres.
    wind_to_rad = math.radians((wind_from_deg + 180.0) % 360.0)
    sin_to = math.sin(wind_to_rad)
    cos_to = math.cos(wind_to_rad)

    # Slope across one east-step (col +1) at row 0: delta = k*u*cell*sin_to.
    east_step = (out[0, 1] - out[0, 0]) / cell_size_m
    np.testing.assert_allclose(east_step, k * wind_speed_ms * sin_to, atol=1e-12)

    # Slope across one south-step (row +1) at col 0: delta = -k*u*cell*cos_to,
    # because south is the +row direction and the ramp formula uses
    # -row_m * cos_to. Equivalently, the per-metre north slope is +k*u*cos_to.
    south_step = (out[1, 0] - out[0, 0]) / cell_size_m
    np.testing.assert_allclose(south_step, -k * wind_speed_ms * cos_to, atol=1e-12)
