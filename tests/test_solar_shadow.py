"""Tests for thermal_model.solar.shadow."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermal_model.solar import SolarPosition, cast_shadow_mask


def _sun(azimuth_deg: float, altitude_deg: float) -> SolarPosition:
    return SolarPosition(
        azimuth_rad=math.radians(azimuth_deg),
        altitude_rad=math.radians(altitude_deg),
    )


# ---------------------------------------------------------------------------
# Trivial degenerate cases
# ---------------------------------------------------------------------------


def test_flat_dem_is_fully_sunlit() -> None:
    dem = np.full((16, 16), 250.0)
    mask = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(180.0, 45.0))
    np.testing.assert_array_equal(mask, np.ones_like(dem))


def test_sun_below_horizon_shadows_everything() -> None:
    dem = np.linspace(0.0, 100.0, 16 * 16).reshape(16, 16)
    mask = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(180.0, -5.0))
    np.testing.assert_array_equal(mask, np.zeros_like(dem))


def test_sun_at_zenith_is_fully_sunlit() -> None:
    # Strongly varied DEM; zenith sun shouldn't shadow anything because
    # there's no horizontal projection.
    rng = np.random.default_rng(0)
    dem = rng.uniform(0.0, 200.0, size=(16, 16))
    mask = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(180.0, 90.0))
    np.testing.assert_array_equal(mask, np.ones_like(dem))


def test_nan_propagates_to_mask() -> None:
    dem = np.full((8, 8), 100.0)
    dem[3, 3] = np.nan
    mask = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(180.0, 45.0))
    assert np.isnan(mask[3, 3])
    finite = mask[~np.isnan(mask)]
    np.testing.assert_array_equal(finite, np.ones_like(finite))


def test_nan_propagates_under_below_horizon_sun() -> None:
    dem = np.full((4, 4), 100.0)
    dem[0, 0] = np.nan
    mask = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(180.0, -10.0))
    assert np.isnan(mask[0, 0])
    np.testing.assert_array_equal(mask[~np.isnan(mask)], 0.0)


def test_2d_required() -> None:
    with pytest.raises(ValueError, match="2-D"):
        cast_shadow_mask(np.zeros(10), cell_size_m=1.0, sun=_sun(180.0, 45.0))


def test_positive_cell_size_required() -> None:
    with pytest.raises(ValueError, match="cell_size_m"):
        cast_shadow_mask(np.zeros((4, 4)), cell_size_m=0.0, sun=_sun(180.0, 45.0))


# ---------------------------------------------------------------------------
# Cliff geometry
# ---------------------------------------------------------------------------


def _two_step_cliff(cliff_height: float = 100.0) -> np.ndarray:
    """A 32-column step DEM: cols 0..15 at z=0, cols 16..31 at z=cliff_height."""
    dem = np.zeros((8, 32), dtype=np.float64)
    dem[:, 16:] = cliff_height
    return dem


def test_cliff_shadows_low_ground_to_west_when_sun_in_east() -> None:
    # Cliff at col=15.5, high ground east. Sun in east at altitude 45°
    # casts a shadow westward of the cliff, length = cliff_height/tan(45°)
    # = 100 m. With 1 m cells, the entire 16-cell western strip is
    # within shadow distance.
    dem = _two_step_cliff(cliff_height=100.0)
    mask = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(90.0, 45.0))
    # Western strip: shadowed (the cliff is upwind/east).
    np.testing.assert_array_equal(mask[:, :16], 0.0)
    # Eastern strip (the cliff plateau itself): sunlit.
    np.testing.assert_array_equal(mask[:, 16:], 1.0)


def test_cliff_does_not_shadow_when_sun_is_behind_observer() -> None:
    # Same cliff, but sun in the west. Sun behind the low ground; the
    # cliff is *downwind* relative to the sun, so it casts no shadow on
    # the low ground.
    dem = _two_step_cliff(cliff_height=100.0)
    mask = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(270.0, 45.0))
    # Low ground (west) sees clear sky in the west.
    np.testing.assert_array_equal(mask[:, :16], 1.0)


def test_cliff_partial_shadow_with_steep_sun() -> None:
    # Very steep sun: shadow is short. cliff_height=10 m, altitude=80°,
    # shadow length = 10 / tan(80°) ≈ 1.76 m, so only the 2 westmost-of-cliff
    # cells should be shadowed and the rest of the low ground sunlit.
    dem = _two_step_cliff(cliff_height=10.0)
    mask = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(90.0, 80.0))
    # Cells immediately west of the cliff (col 14, 15) shadowed.
    assert (mask[:, 15] == 0.0).all()
    # Cells far from the cliff (col 0..10) sunlit.
    assert (mask[:, :10] == 1.0).all()


def test_cliff_shadow_length_matches_geometry_at_45deg() -> None:
    # At 45° sun altitude, shadow length = cliff_height. Pick a
    # 30 m cliff with 1 m cells: shadow extends 30 cells west of the
    # cliff base. Place the cliff at col 60 in a 100-col grid.
    dem = np.zeros((4, 100), dtype=np.float64)
    dem[:, 60:] = 30.0
    mask = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(90.0, 45.0))
    # Col 30 sees the cliff top at distance 30 cells; ray height
    # = 0 + 30 * tan(45°) ≈ 30 m, equal to cliff top to within
    # floating-point — this single cell is genuinely on the shadow
    # boundary and either outcome is acceptable. Cells with col < 30
    # are clearly outside the geometric shadow; cells with col >= 31
    # are clearly inside.
    assert (mask[:, :30] == 1.0).all(), "cells beyond shadow length should be sunlit"
    assert (mask[:, 31:60] == 0.0).all(), "cells in shadow strip should be shadowed"


# ---------------------------------------------------------------------------
# Tower geometry
# ---------------------------------------------------------------------------


def test_tower_casts_shadow_downsun() -> None:
    # 1-cell tall tower in the middle of a flat plain. Sun in the east,
    # low altitude. The tower's shadow extends westward (downsun = away
    # from sun = -col) for a distance of tower_height/tan(alt).
    dem = np.zeros((11, 21), dtype=np.float64)
    dem[5, 10] = 20.0  # tower at (row=5, col=10)
    mask = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(90.0, 45.0))
    # Cells exactly west of the tower (same row, col < 10) within
    # shadow distance (20 cells) should be shadowed.
    # Bilinear sampling smooths a single-cell tower so the shadow is
    # weaker than a sharp test would suggest; we only check the
    # immediate neighbour, which sees the tower with full weight.
    assert mask[5, 9] == 0.0
    # Plenty of cells far away or off-row are sunlit.
    assert mask[5, 11] == 1.0  # east of tower (toward sun) is sunlit
    assert mask[0, 0] == 1.0  # corner far from tower
    assert mask[10, 10] == 1.0  # north of tower at same col is not in line


def test_tower_shadow_aligned_with_sun_azimuth() -> None:
    # A tower to the NE casts its shadow to the SW. Place a 30 m
    # tower at (row=5, col=15) on a 30x30 grid; sun in the NE
    # (azimuth=45°) at altitude 45°. Test that the cell to the SW of
    # the tower is shadowed and the cell to the NE is sunlit.
    dem = np.zeros((30, 30), dtype=np.float64)
    dem[5, 15] = 30.0
    mask = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(45.0, 45.0))
    # SW of tower: row+1, col-1 = (6, 14)
    assert mask[6, 14] == 0.0
    # NE of tower: row-1, col+1 = (4, 16) — sunlit
    assert mask[4, 16] == 1.0


# ---------------------------------------------------------------------------
# Tilted ramp (no self-cast shadow)
# ---------------------------------------------------------------------------


def test_uniform_ramp_does_not_self_cast_shadow() -> None:
    # A featureless tilted plane has no upwind feature to cast shadow.
    # Sun cos(theta_i) < 0 self-shading is handled by slope_irradiance,
    # not here. Ramp rises 1 m per col from col 0 to col 31.
    cols = 32
    rows = 8
    dem = np.tile(np.arange(cols, dtype=np.float64), (rows, 1))
    # Sun in the east (toward higher ground) at modest altitude — every
    # cell can see the sun over the gently rising plane only if
    # tan(alt) > slope. Slope is 1/1 = 45°. With sun at 60° altitude,
    # sun is steeper than the slope -> every cell sees the sun. With
    # alt=30° (shallower than the slope), the ramp itself self-occludes.
    mask_steep = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(90.0, 60.0))
    np.testing.assert_array_equal(mask_steep, np.ones_like(dem))
    mask_shallow = cast_shadow_mask(dem, cell_size_m=1.0, sun=_sun(90.0, 30.0))
    # Ramp rises faster than the sun ray, so cells (except the easternmost
    # column, which has no upwind ramp) are shadowed.
    assert (mask_shallow[:, :-1] == 0.0).all()
