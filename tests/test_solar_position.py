"""Tests for thermal_model.solar.position."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from thermal_model.solar import SolarPosition, solar_position

# Yorkshire Dales centre — used as the canonical site for these tests.
DALES_LAT = 54.2
DALES_LON = -2.3


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def test_solar_position_summer_noon_uk_is_high_and_southerly() -> None:
    when = datetime(2026, 6, 21, 12, 0, tzinfo=ZoneInfo("Europe/London"))
    sun = solar_position(when, DALES_LAT, DALES_LON)
    assert sun.is_above_horizon
    # On the summer solstice in the UK, peak solar altitude is
    # ~90 - latitude + 23.4 ≈ 59°. Clock noon in BST is offset from
    # solar noon by roughly an hour, so the sun is a few degrees past
    # peak — generous tolerance covers it.
    assert math.degrees(sun.altitude_rad) == pytest.approx(58.0, abs=4.0)
    # Solar azimuth at solar noon is due south (180°). UK clock noon in
    # BST is offset from solar noon by ~1h plus a longitude correction;
    # azimuth should still be within a few tens of degrees of south.
    az_deg = math.degrees(sun.azimuth_rad)
    assert 140.0 < az_deg < 200.0


def test_solar_position_midnight_is_below_horizon() -> None:
    when = datetime(2026, 1, 15, 0, 0, tzinfo=UTC)
    sun = solar_position(when, DALES_LAT, DALES_LON)
    assert not sun.is_above_horizon
    assert sun.altitude_rad < 0.0


def test_solar_position_zenith_property_is_consistent_with_altitude() -> None:
    when = datetime(2026, 5, 6, 13, 0, tzinfo=ZoneInfo("Europe/London"))
    sun = solar_position(when, DALES_LAT, DALES_LON)
    assert sun.zenith_rad == pytest.approx(math.pi / 2.0 - sun.altitude_rad)


def test_solar_position_azimuth_in_range() -> None:
    when = datetime(2026, 5, 6, 9, 0, tzinfo=ZoneInfo("Europe/London"))
    sun = solar_position(when, DALES_LAT, DALES_LON)
    assert 0.0 <= sun.azimuth_rad < 2.0 * math.pi


def test_solar_position_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        solar_position(datetime(2026, 5, 6, 13, 0), DALES_LAT, DALES_LON)


def test_solar_position_morning_east_afternoon_west() -> None:
    morning = solar_position(
        datetime(2026, 6, 21, 7, 0, tzinfo=ZoneInfo("Europe/London")),
        DALES_LAT,
        DALES_LON,
    )
    afternoon = solar_position(
        datetime(2026, 6, 21, 17, 0, tzinfo=ZoneInfo("Europe/London")),
        DALES_LAT,
        DALES_LON,
    )
    morning_az = math.degrees(morning.azimuth_rad)
    afternoon_az = math.degrees(afternoon.azimuth_rad)
    # Morning sun is in the eastern half (0–180°), afternoon in the west.
    assert 30.0 < morning_az < 150.0
    assert 210.0 < afternoon_az < 330.0


def test_solar_position_returns_dataclass() -> None:
    when = datetime(2026, 5, 6, 13, 0, tzinfo=ZoneInfo("Europe/London"))
    sun = solar_position(when, DALES_LAT, DALES_LON)
    assert isinstance(sun, SolarPosition)
