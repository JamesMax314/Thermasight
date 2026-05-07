"""Tests for thermal_model.solar.irradiance."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from thermal_model.solar import (
    ClearSkyIrradiance,
    SolarPosition,
    clear_sky_irradiance,
    slope_irradiance,
    solar_position,
)

DALES_LAT = 54.2
DALES_LON = -2.3


# ---------------------------------------------------------------------------
# Clear-sky irradiance
# ---------------------------------------------------------------------------


def test_clear_sky_midday_summer_is_strong() -> None:
    when = datetime(2026, 6, 21, 12, 0, tzinfo=ZoneInfo("Europe/London"))
    cs = clear_sky_irradiance(when, DALES_LAT, DALES_LON)
    assert cs.ghi_wm2 > 600.0
    assert cs.dni_wm2 > 500.0
    assert cs.dhi_wm2 > 50.0
    # Sanity: GHI = DNI*cos(zenith) + DHI for a horizontal surface.
    sun = solar_position(when, DALES_LAT, DALES_LON)
    expected_ghi = cs.dni_wm2 * math.cos(sun.zenith_rad) + cs.dhi_wm2
    assert cs.ghi_wm2 == pytest.approx(expected_ghi, rel=0.02)


def test_clear_sky_at_night_is_zero() -> None:
    when = datetime(2026, 1, 15, 0, 0, tzinfo=UTC)
    cs = clear_sky_irradiance(when, DALES_LAT, DALES_LON)
    assert cs.ghi_wm2 == pytest.approx(0.0, abs=1e-6)
    assert cs.dni_wm2 == pytest.approx(0.0, abs=1e-6)
    assert cs.dhi_wm2 == pytest.approx(0.0, abs=1e-6)


def test_clear_sky_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        clear_sky_irradiance(datetime(2026, 5, 6, 13, 0), DALES_LAT, DALES_LON)


def test_clear_sky_higher_turbidity_reduces_dni() -> None:
    when = datetime(2026, 6, 21, 12, 0, tzinfo=ZoneInfo("Europe/London"))
    clean = clear_sky_irradiance(when, DALES_LAT, DALES_LON, linke_turbidity=2.0)
    hazy = clear_sky_irradiance(when, DALES_LAT, DALES_LON, linke_turbidity=5.0)
    assert hazy.dni_wm2 < clean.dni_wm2


# ---------------------------------------------------------------------------
# Slope irradiance — geometric checks
# ---------------------------------------------------------------------------


def _flat_slope_aspect(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    slope = np.zeros(shape, dtype=np.float64)
    aspect = np.full(shape, np.nan, dtype=np.float64)  # undefined on flats
    return slope, aspect


def test_slope_irradiance_flat_surface_recovers_ghi() -> None:
    # On a horizontal surface, beam + diffuse should equal GHI exactly.
    sun = SolarPosition(
        azimuth_rad=math.radians(180.0), altitude_rad=math.radians(60.0)
    )
    cs = ClearSkyIrradiance(ghi_wm2=850.0, dni_wm2=900.0, dhi_wm2=100.0)
    # Reconstruct DHI so GHI = DNI cos(zenith) + DHI exactly:
    expected_ghi = cs.dni_wm2 * math.cos(sun.zenith_rad) + cs.dhi_wm2

    slope, aspect = _flat_slope_aspect((4, 4))
    si = slope_irradiance(slope, aspect, sun, cs)
    np.testing.assert_allclose(si.total_wm2, expected_ghi, rtol=1e-12)
    np.testing.assert_allclose(
        si.beam_wm2, cs.dni_wm2 * math.cos(sun.zenith_rad), rtol=1e-12
    )
    np.testing.assert_allclose(si.diffuse_wm2, cs.dhi_wm2, rtol=1e-12)


def test_slope_irradiance_normal_to_sun_gets_full_dni() -> None:
    # Slope tilted exactly into the sun: cos(theta_i) = 1, so beam = DNI.
    sun_az_deg = 180.0  # sun due south
    sun_alt_deg = 30.0
    sun = SolarPosition(
        azimuth_rad=math.radians(sun_az_deg),
        altitude_rad=math.radians(sun_alt_deg),
    )
    cs = ClearSkyIrradiance(ghi_wm2=500.0, dni_wm2=900.0, dhi_wm2=80.0)

    # A plane normal to the sun has slope = (90 - altitude) and faces
    # the sun's azimuth.
    slope = np.full((3, 3), math.radians(90.0 - sun_alt_deg))
    aspect = np.full((3, 3), math.radians(sun_az_deg))
    si = slope_irradiance(slope, aspect, sun, cs)
    np.testing.assert_allclose(si.beam_wm2, cs.dni_wm2, rtol=1e-12)


def test_slope_irradiance_back_of_hill_gets_no_beam() -> None:
    # Slope facing directly away from the sun: cos(theta_i) <= 0, so
    # beam is clipped to zero. Diffuse is unaffected.
    sun = SolarPosition(
        azimuth_rad=math.radians(180.0), altitude_rad=math.radians(45.0)
    )
    cs = ClearSkyIrradiance(ghi_wm2=600.0, dni_wm2=800.0, dhi_wm2=120.0)
    # 60° slope facing due north — opposite of due-south sun.
    slope = np.full((3, 3), math.radians(60.0))
    aspect = np.full((3, 3), math.radians(0.0))
    si = slope_irradiance(slope, aspect, sun, cs)
    np.testing.assert_allclose(si.beam_wm2, 0.0, atol=1e-12)
    # Diffuse still positive on a tilted surface.
    assert np.all(si.diffuse_wm2 > 0.0)


def test_slope_irradiance_below_horizon_sun_gives_no_beam() -> None:
    sun = SolarPosition(azimuth_rad=math.radians(0.0), altitude_rad=math.radians(-10.0))
    # Pretend a non-zero DNI somehow; the function should still zero
    # the beam because the sun is below the horizon.
    cs = ClearSkyIrradiance(ghi_wm2=0.0, dni_wm2=100.0, dhi_wm2=0.0)
    slope = np.full((4, 4), math.radians(20.0))
    aspect = np.full((4, 4), math.radians(180.0))
    si = slope_irradiance(slope, aspect, sun, cs)
    np.testing.assert_allclose(si.beam_wm2, 0.0, atol=1e-12)


def test_slope_irradiance_diffuse_decreases_with_slope() -> None:
    # Isotropic diffuse on a tilted plane is DHI * (1 + cos(slope))/2:
    # 1.0 at horizontal, 0.5 at vertical.
    sun = SolarPosition(
        azimuth_rad=math.radians(180.0), altitude_rad=math.radians(45.0)
    )
    cs = ClearSkyIrradiance(ghi_wm2=600.0, dni_wm2=800.0, dhi_wm2=200.0)
    slope = np.array([[0.0, math.radians(45.0), math.radians(90.0)]], dtype=np.float64)
    aspect = np.array(
        [[np.nan, math.radians(180.0), math.radians(180.0)]], dtype=np.float64
    )
    si = slope_irradiance(slope, aspect, sun, cs)
    np.testing.assert_allclose(si.diffuse_wm2[0, 0], cs.dhi_wm2, rtol=1e-12)
    np.testing.assert_allclose(
        si.diffuse_wm2[0, 1],
        cs.dhi_wm2 * 0.5 * (1 + math.cos(math.radians(45))),
        rtol=1e-12,
    )
    np.testing.assert_allclose(si.diffuse_wm2[0, 2], cs.dhi_wm2 * 0.5, rtol=1e-12)


def test_slope_irradiance_propagates_nan_from_slope() -> None:
    sun = SolarPosition(
        azimuth_rad=math.radians(180.0), altitude_rad=math.radians(45.0)
    )
    cs = ClearSkyIrradiance(ghi_wm2=600.0, dni_wm2=800.0, dhi_wm2=200.0)
    slope = np.array([[math.radians(20.0), np.nan]], dtype=np.float64)
    aspect = np.array([[math.radians(180.0), math.radians(180.0)]], dtype=np.float64)
    si = slope_irradiance(slope, aspect, sun, cs)
    assert np.isfinite(si.beam_wm2[0, 0])
    assert np.isfinite(si.diffuse_wm2[0, 0])
    assert np.isnan(si.beam_wm2[0, 1])
    assert np.isnan(si.diffuse_wm2[0, 1])


def test_slope_irradiance_handles_nan_aspect_on_true_flats() -> None:
    # aspect is NaN on flat cells (slope == 0); the contribution from
    # the aspect term must vanish there rather than poisoning the cell.
    sun = SolarPosition(
        azimuth_rad=math.radians(180.0), altitude_rad=math.radians(45.0)
    )
    cs = ClearSkyIrradiance(ghi_wm2=600.0, dni_wm2=800.0, dhi_wm2=200.0)
    slope = np.zeros((2, 2), dtype=np.float64)
    aspect = np.full((2, 2), np.nan, dtype=np.float64)
    si = slope_irradiance(slope, aspect, sun, cs)
    assert np.all(np.isfinite(si.beam_wm2))
    assert np.all(np.isfinite(si.diffuse_wm2))


def test_slope_irradiance_shape_mismatch_raises() -> None:
    sun = SolarPosition(
        azimuth_rad=math.radians(180.0), altitude_rad=math.radians(45.0)
    )
    cs = ClearSkyIrradiance(ghi_wm2=600.0, dni_wm2=800.0, dhi_wm2=200.0)
    with pytest.raises(ValueError, match="shape"):
        slope_irradiance(np.zeros((3, 3)), np.zeros((3, 4)), sun, cs)


# ---------------------------------------------------------------------------
# Integration: real sun position into slope irradiance
# ---------------------------------------------------------------------------


def test_slope_irradiance_pipeline_morning_se_face_brighter_than_nw_face() -> None:
    # Morning sun in the SE: a SE-facing slope should receive more
    # beam than a NW-facing slope of the same angle.
    when = datetime(2026, 6, 21, 8, 0, tzinfo=ZoneInfo("Europe/London"))
    sun = solar_position(when, DALES_LAT, DALES_LON)
    cs = clear_sky_irradiance(when, DALES_LAT, DALES_LON)

    slope = np.array([[math.radians(30.0), math.radians(30.0)]], dtype=np.float64)
    aspect = np.array(
        [[math.radians(135.0), math.radians(315.0)]], dtype=np.float64
    )  # SE, NW
    si = slope_irradiance(slope, aspect, sun, cs)
    assert si.beam_wm2[0, 0] > si.beam_wm2[0, 1]
