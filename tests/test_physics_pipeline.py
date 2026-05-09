"""Tests for thermal_model.physics.pipeline.

The headline test is the **mirror-spur validation gate** for Phase 3
(``docs/ROADMAP.md`` Phase 3 § Validation): a synthetic DEM with two
geometrically identical spurs facing south and north respectively
must yield a higher trigger score on the south-facing spur under a
noon midsummer sun. This is the case the previous post-hoc multiplier
got right by accident (local ``H = 0`` zeros the cell) but the new
heating-weighted-accumulation formulation gets right by physics:
shadowed upstream cells inject zero W/m² into the routing, so the
shadowed spur receives no upstream thermal energy.

A second-order check artificially relits the shadowed catchment by
substituting an all-sunlit shadow mask in a manual pipeline run; the
N-spur trigger should rise toward the S-spur. This verifies that the
routing actually transports upstream warmth to the convergent point
rather than just multiplying by it locally.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from thermal_model.physics import run_model
from thermal_model.physics.flow import flow_accumulation
from thermal_model.physics.heating import heating_field
from thermal_model.physics.hydrology import fill_pits
from thermal_model.physics.pipeline import _gaussian_smooth_nan, _rank_normalise
from thermal_model.physics.wind_tilt import wind_tilt_ramp
from thermal_model.solar.irradiance import clear_sky_irradiance, slope_irradiance
from thermal_model.solar.position import solar_position
from thermal_model.terrain.morphometry import aspect, profile_curvature, slope

# Noon midsummer at a midland-UK latitude. The sun is high (~60° alt)
# and slightly south of zenith, so a south-facing slope receives much
# more direct beam than a north-facing one.
NOON_MIDSUMMER = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
LAT_DEG = 54.4
LON_DEG = -2.3
CELL_SIZE_M = 5.0


def _mirror_spur_dem(
    rows: int = 121, cols: int = 121, *, cell_size_m: float = CELL_SIZE_M
) -> np.ndarray:
    """Two mirror-image triangular spurs, one south-facing, one north-facing.

    The DEM is a horizontal "wall" running east-west across the middle
    of the array, with two triangular spurs sticking out either side
    of the wall. The geometries are exact mirror images:

      * south spur: extends downward from the wall toward large row
        index (south), with a sharp ridgeline along col=cols//2 and a
        triangular taper toward the tip. Faces south.
      * north spur: same shape, mirrored to small-row-index (north).

    Each spur is a 30 m-high pyramidal feature. The wall itself is
    raised so the spurs project from a high plateau, ensuring their
    tips are convex relative to the surrounding terrain.
    """
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    centre_row = rows // 2
    centre_col = cols // 2

    # Distance from the central east-west wall.
    dist_from_wall = np.abs(yy - centre_row)
    # Distance from the central north-south ridgeline.
    dist_from_ridge = np.abs(xx - centre_col)

    spur_length = 40.0  # rows from the wall
    spur_half_width = 20.0  # at the wall base

    # Triangular taper: the spur is half_width at the wall and tapers
    # linearly to zero at spur_length. A cell is on the spur if:
    # dist_from_ridge < half_width * (1 - dist_from_wall / spur_length)
    fraction = np.clip(1.0 - dist_from_wall / spur_length, 0.0, 1.0)
    on_spur = dist_from_ridge < spur_half_width * fraction
    spur_height = (
        30.0
        * fraction
        * np.where(
            on_spur,
            1.0 - dist_from_ridge / np.maximum(spur_half_width * fraction, 1e-9),
            0.0,
        )
    )
    spur_height = np.where(on_spur, spur_height, 0.0)

    # Wall plateau at 100 m elevation; spurs project upward and outward.
    wall_height = np.where(dist_from_wall < 5.0, 30.0, 0.0)
    elevation = 100.0 + np.maximum(spur_height, wall_height)
    return elevation.astype(np.float64)


def _split_spur_means(field: np.ndarray) -> tuple[float, float]:
    """Mean of ``field`` over the south and north halves of the array.

    Returns ``(south_mean, north_mean)``. NaN cells are ignored. The
    split is at the central row.
    """
    rows = field.shape[0]
    centre = rows // 2
    north_half = field[:centre, :]
    south_half = field[centre + 1 :, :]
    return float(np.nanmean(south_half)), float(np.nanmean(north_half))


def test_mirror_spur_south_outscores_north_at_noon_midsummer() -> None:
    """Phase 3 validation gate: S-facing spur > N-facing spur."""
    dem = _mirror_spur_dem()
    result = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=0.0,  # zero wind so the asymmetry is purely solar
        wind_speed_ms=0.0,
        smoothing_sigma_m=5.0,
    )
    south_mean_t, north_mean_t = _split_spur_means(result.trigger_potential)
    assert south_mean_t > north_mean_t, (
        f"south-spur mean trigger {south_mean_t} should exceed "
        f"north-spur mean {north_mean_t}"
    )

    # Also: heating itself should be higher on the south half.
    south_h, north_h = _split_spur_means(result.heating_wm2)
    assert south_h > north_h


def test_mirror_spur_relit_north_rises_toward_south() -> None:
    """Removing the cast shadow (all-sunlit) lifts the N-spur trigger.

    A purely *local* multiplier would not produce this effect at all if
    the original difference came from the local shadow mask zeroing
    the cell — relighting that local cell would lift it directly. The
    interesting check is that the *upstream* warm-air transport via
    weighted accumulation accounts for a meaningful share of the gap:
    relighting the shadowed catchment should narrow (or close) the
    south-vs-north gap, not just shuffle it.
    """
    dem = _mirror_spur_dem()
    sun = solar_position(NOON_MIDSUMMER, LAT_DEG, LON_DEG, elevation_m=130.0)
    cs = clear_sky_irradiance(NOON_MIDSUMMER, LAT_DEG, LON_DEG, elevation_m=130.0)
    slope_rad = slope(dem, CELL_SIZE_M)
    aspect_rad = aspect(dem, CELL_SIZE_M)
    irr = slope_irradiance(slope_rad, aspect_rad, sun, cs)

    # All-sunlit shadow mask: every cell receives full beam.
    sunlit = np.ones_like(dem)

    heating = heating_field(irr, sunlit)
    weights = np.where(np.isnan(heating), 0.0, heating)

    # Same routing as run_model, with zero wind so the only asymmetry
    # left is the slope-projected irradiance (S-faces hit more
    # squarely than N-faces). The cast-shadow asymmetry is gone.
    smoothed = _gaussian_smooth_nan(dem, sigma_cells=5.0 / CELL_SIZE_M)
    tilted = wind_tilt_ramp(
        smoothed, CELL_SIZE_M, wind_from_deg=0.0, wind_speed_ms=0.0, k=0.03
    )
    inverted = float(np.nanmax(tilted)) - tilted
    filled = fill_pits(inverted, epsilon=1.0e-3)
    weighted_conv = flow_accumulation(filled, CELL_SIZE_M, weights=weights)

    kprof = profile_curvature(dem, CELL_SIZE_M)
    wc_norm = _rank_normalise(weighted_conv)
    curv_norm = _rank_normalise(np.where(kprof > 0, kprof, 0.0))
    slope_mask = (slope_rad > np.radians(2.5)).astype(np.float64)
    relit_trigger = wc_norm * curv_norm * slope_mask

    relit_south, relit_north = _split_spur_means(relit_trigger)
    relit_gap = relit_south - relit_north

    # Reference run with cast shadows in play.
    reference = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=0.0,
        wind_speed_ms=0.0,
        smoothing_sigma_m=5.0,
    )
    ref_south, ref_north = _split_spur_means(reference.trigger_potential)
    ref_gap = ref_south - ref_north

    # Relighting must close the south-vs-north gap (or at least narrow
    # it). A *local* multiplier would not necessarily produce this
    # effect from the *upstream* relighting alone, since the local
    # shadow mask at the convergent cell is the same in both runs at
    # noon midsummer (the spur tips themselves are sunlit). The fact
    # that the gap shrinks here means the routing is transporting
    # upstream warmth, which is what we want.
    assert relit_gap < ref_gap, (
        f"relit gap {relit_gap} should be smaller than reference "
        f"gap {ref_gap}; weighted accumulation may not be transporting "
        f"upstream warmth"
    )


# ---------------------------------------------------------------------------
# Smaller invariants on RunResult.
# ---------------------------------------------------------------------------


def test_run_model_trigger_potential_in_unit_interval() -> None:
    dem = _mirror_spur_dem(rows=51, cols=51)
    result = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=225.0,
        wind_speed_ms=4.0,
    )
    finite = result.trigger_potential[np.isfinite(result.trigger_potential)]
    assert finite.size > 0
    assert float(finite.min()) >= 0.0
    assert float(finite.max()) <= 1.0


def test_run_model_propagates_dem_nan() -> None:
    dem = _mirror_spur_dem(rows=51, cols=51)
    dem[0, 0] = np.nan
    result = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=180.0,
        wind_speed_ms=2.0,
    )
    assert np.isnan(result.trigger_potential[0, 0])
    assert np.isnan(result.weighted_convergence[0, 0])
    assert np.isnan(result.smoothed_dem_m[0, 0])
    assert np.isnan(result.tilted_dem_m[0, 0])


def test_run_model_invalid_inputs() -> None:
    dem = _mirror_spur_dem(rows=21, cols=21)
    with pytest.raises(ValueError):
        run_model(
            dem[0],
            CELL_SIZE_M,
            NOON_MIDSUMMER,
            LAT_DEG,
            LON_DEG,
            wind_from_deg=0.0,
            wind_speed_ms=0.0,
        )
    with pytest.raises(ValueError):
        run_model(
            dem,
            -1.0,
            NOON_MIDSUMMER,
            LAT_DEG,
            LON_DEG,
            wind_from_deg=0.0,
            wind_speed_ms=0.0,
        )
    with pytest.raises(ValueError):
        run_model(
            dem,
            CELL_SIZE_M,
            NOON_MIDSUMMER,
            LAT_DEG,
            LON_DEG,
            wind_from_deg=0.0,
            wind_speed_ms=0.0,
            smoothing_sigma_m=-1.0,
        )
