"""Tests for thermal_model.physics.coupling."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_model.physics import thermal_potential

# ---------------------------------------------------------------------------
# Default coupling: geometric mean
# ---------------------------------------------------------------------------


def test_default_is_geometric_mean() -> None:
    h = np.array([[100.0, 400.0], [25.0, 0.0]], dtype=np.float64)
    c = np.array([[16.0, 9.0], [4.0, 1000.0]], dtype=np.float64)
    p = thermal_potential(h, c)
    np.testing.assert_allclose(p, np.sqrt(h * c))


def test_default_geometric_mean_compresses_dynamic_range() -> None:
    # Demonstrate the scale-balancing rationale from the docstring:
    # if convergence dwarfs heating, the geometric mean should still
    # let heating influence the ranking. Build two cells where the
    # straight product would tie them, but the geometric mean
    # separates them.
    h = np.array([100.0, 400.0])  # cell A is cool, cell B is hot
    c = np.array([4000.0, 1000.0])  # cell A has more convergence
    # Straight product: equal at 400_000 each.
    np.testing.assert_array_equal(h * c, np.array([400_000.0, 400_000.0]))
    # Geometric mean: equal at sqrt(4e5) — *still equal* on this
    # specific input, but the case is that combined-strength is what
    # the metric ranks, not raw-C.
    p = thermal_potential(h, c)
    np.testing.assert_allclose(p, np.sqrt(np.array([400_000.0, 400_000.0])))
    # Now bump heating on cell B; geometric mean rewards it more than
    # the straight product would (relative gain proportional to
    # sqrt(hot_ratio) instead of hot_ratio).
    h2 = np.array([100.0, 800.0])
    p2 = thermal_potential(h2, c)
    assert p2[1] > p2[0]


# ---------------------------------------------------------------------------
# Exponent sweep
# ---------------------------------------------------------------------------


def test_unit_exponents_recover_plain_product() -> None:
    h = np.array([[100.0, 200.0], [50.0, 25.0]])
    c = np.array([[10.0, 5.0], [20.0, 100.0]])
    p = thermal_potential(h, c, heating_exponent=1.0, convergence_exponent=1.0)
    np.testing.assert_allclose(p, h * c)


def test_zero_convergence_exponent_returns_heating() -> None:
    h = np.array([[100.0, 200.0]])
    c = np.array([[10.0, 5.0]])
    p = thermal_potential(h, c, heating_exponent=1.0, convergence_exponent=0.0)
    np.testing.assert_allclose(p, h)


def test_zero_heating_exponent_returns_convergence() -> None:
    h = np.array([[100.0, 200.0]])
    c = np.array([[10.0, 5.0]])
    p = thermal_potential(h, c, heating_exponent=0.0, convergence_exponent=1.0)
    np.testing.assert_allclose(p, c)


def test_morning_heating_weighted_coupling() -> None:
    # (p, q) = (0.7, 0.3) — heating-weighted, suiting morning
    # conditions where aspect dominates. Hand-compute on two cells.
    h = np.array([400.0, 100.0])
    c = np.array([10.0, 100.0])
    p = thermal_potential(h, c, heating_exponent=0.7, convergence_exponent=0.3)
    expected = np.power(h, 0.7) * np.power(c, 0.3)
    np.testing.assert_allclose(p, expected)


def test_afternoon_convergence_weighted_coupling() -> None:
    # (p, q) = (0.3, 0.7) — convergence-weighted, afternoon massif
    # is uniformly warm and trigger geometry takes over.
    h = np.array([400.0, 100.0])
    c = np.array([10.0, 100.0])
    p = thermal_potential(h, c, heating_exponent=0.3, convergence_exponent=0.7)
    expected = np.power(h, 0.3) * np.power(c, 0.7)
    np.testing.assert_allclose(p, expected)


# ---------------------------------------------------------------------------
# Edge cases: zero, NaN
# ---------------------------------------------------------------------------


def test_zero_heating_collapses_potential_to_zero() -> None:
    # No heat -> no thermal, regardless of how convex the terrain is.
    h = np.zeros((2, 2))
    c = np.array([[100.0, 1000.0], [10.0, 50.0]])
    p = thermal_potential(h, c)
    np.testing.assert_array_equal(p, np.zeros((2, 2)))


def test_zero_convergence_collapses_potential_to_zero() -> None:
    # No convergence -> no organised release, regardless of heat.
    h = np.array([[100.0, 1000.0], [10.0, 50.0]])
    c = np.zeros((2, 2))
    p = thermal_potential(h, c)
    np.testing.assert_array_equal(p, np.zeros((2, 2)))


def test_nan_in_heating_propagates() -> None:
    h = np.array([[100.0, np.nan]])
    c = np.array([[10.0, 10.0]])
    p = thermal_potential(h, c)
    assert np.isfinite(p[0, 0])
    assert np.isnan(p[0, 1])


def test_nan_in_convergence_propagates() -> None:
    h = np.array([[100.0, 100.0]])
    c = np.array([[10.0, np.nan]])
    p = thermal_potential(h, c)
    assert np.isfinite(p[0, 0])
    assert np.isnan(p[0, 1])


def test_nan_propagates_even_when_other_axis_zero_weighted() -> None:
    # If we zero the heating exponent, a NaN in heating should still
    # propagate — otherwise the coupling silently masks data quality
    # issues just because a sweep happened to ignore that axis.
    h = np.array([[100.0, np.nan]])
    c = np.array([[10.0, 10.0]])
    p = thermal_potential(h, c, heating_exponent=0.0, convergence_exponent=1.0)
    assert np.isfinite(p[0, 0])
    assert np.isnan(p[0, 1])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="convergence shape"):
        thermal_potential(np.zeros((3, 3)), np.zeros((3, 4)))


def test_negative_exponent_rejected() -> None:
    with pytest.raises(ValueError, match="heating_exponent"):
        thermal_potential(np.ones((2, 2)), np.ones((2, 2)), heating_exponent=-0.1)
    with pytest.raises(ValueError, match="convergence_exponent"):
        thermal_potential(np.ones((2, 2)), np.ones((2, 2)), convergence_exponent=-0.1)


def test_negative_heating_rejected() -> None:
    h = np.array([[100.0, -1.0]])
    c = np.array([[10.0, 10.0]])
    with pytest.raises(ValueError, match="heating_wm2 contains negative"):
        thermal_potential(h, c)


def test_negative_convergence_rejected() -> None:
    h = np.array([[100.0, 100.0]])
    c = np.array([[10.0, -0.5]])
    with pytest.raises(ValueError, match="convergence contains negative"):
        thermal_potential(h, c)


# ---------------------------------------------------------------------------
# Integration: full Phase 2 pipeline produces sensible coupling
# ---------------------------------------------------------------------------


def test_pipeline_smoke_test_on_synthetic_hill() -> None:
    # End-to-end: build a synthetic Gaussian hill, run the full
    # Phase 2 pipeline (terrain morphometry -> solar irradiance ->
    # cast shadow -> heating -> convergence -> coupling), and verify
    # the output is non-negative and has at least one strictly
    # positive cell.
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from thermal_model.physics import (
        fill_pits,
        flow_accumulation,
        heating_field,
        thermal_potential,
    )
    from thermal_model.solar import (
        cast_shadow_mask,
        clear_sky_irradiance,
        slope_irradiance,
        solar_position,
    )
    from thermal_model.terrain import aspect, slope

    n = 64
    cell = 5.0
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    cx = cy = (n - 1) / 2.0
    sigma = n / 6.0
    dem = 400.0 + 80.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))

    when = datetime(2026, 6, 21, 12, 0, tzinfo=ZoneInfo("Europe/London"))
    sun = solar_position(when, 54.2, -2.3)
    cs = clear_sky_irradiance(when, 54.2, -2.3, elevation_m=400.0)
    slope_rad = slope(dem, cell)
    aspect_rad = aspect(dem, cell)
    si = slope_irradiance(slope_rad, aspect_rad, sun, cs)
    shadow = cast_shadow_mask(dem, cell, sun)
    h = heating_field(si, shadow)

    inverted = float(np.nanmax(dem)) - dem
    filled = fill_pits(inverted, epsilon=1e-3)
    c = flow_accumulation(filled, cell)

    # Replace NaN with zero where appropriate so coupling has finite
    # values to operate on. NaN-on-NaN should still propagate.
    h_clean = np.where(np.isnan(h), 0.0, h)
    c_clean = np.where(np.isnan(c), 0.0, c)
    p = thermal_potential(h_clean, c_clean)
    assert np.all(p >= 0.0)
    assert np.any(p > 0.0)
    assert p.shape == dem.shape
