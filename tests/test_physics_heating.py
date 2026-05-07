"""Tests for thermal_model.physics.heating."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_model.physics import DEFAULT_ABSORPTIVITY, heating_field
from thermal_model.solar import SlopeIrradiance


def _irradiance(beam: float, diffuse: float, shape: tuple[int, int]) -> SlopeIrradiance:
    return SlopeIrradiance(
        beam_wm2=np.full(shape, beam, dtype=np.float64),
        diffuse_wm2=np.full(shape, diffuse, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Formula correctness
# ---------------------------------------------------------------------------


def test_heating_unit_alpha_full_sun_equals_total_irradiance() -> None:
    irr = _irradiance(beam=600.0, diffuse=150.0, shape=(4, 4))
    sun_lit = np.ones((4, 4), dtype=np.float64)
    h = heating_field(irr, sun_lit, absorptivity=1.0)
    np.testing.assert_array_equal(h, np.full((4, 4), 750.0))


def test_heating_default_alpha_matches_data_md_upland_default() -> None:
    # CLAUDE.md / docs/DATA.md pin the upland default to dry grass /
    # heather, alpha = 0.80. Surface this constant in the public API
    # so callers don't have to hard-code it.
    assert DEFAULT_ABSORPTIVITY == pytest.approx(0.80)
    irr = _irradiance(beam=500.0, diffuse=100.0, shape=(2, 2))
    sun_lit = np.ones((2, 2), dtype=np.float64)
    h = heating_field(irr, sun_lit)
    np.testing.assert_allclose(h, np.full((2, 2), 0.80 * 600.0))


def test_heating_shadow_kills_beam_but_keeps_diffuse() -> None:
    # In cast shadow (s = 0), beam is zero; diffuse still warms the
    # ground (the sky is still bright above).
    irr = _irradiance(beam=600.0, diffuse=150.0, shape=(2, 2))
    shadowed = np.zeros((2, 2), dtype=np.float64)
    h = heating_field(irr, shadowed, absorptivity=0.8)
    np.testing.assert_allclose(h, np.full((2, 2), 0.8 * 150.0))


def test_heating_zero_absorptivity_yields_zero_anywhere() -> None:
    irr = _irradiance(beam=600.0, diffuse=150.0, shape=(3, 3))
    sun_lit = np.ones((3, 3), dtype=np.float64)
    h = heating_field(irr, sun_lit, absorptivity=0.0)
    np.testing.assert_array_equal(h, np.zeros((3, 3)))


def test_heating_mixed_shadow_mask() -> None:
    # Half the cells in shadow, half sunlit. Verify the formula picks
    # up beam only on sunlit cells.
    beam = 500.0
    diffuse = 100.0
    alpha = 0.85
    irr = _irradiance(beam, diffuse, shape=(2, 4))
    s = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float64)
    h = heating_field(irr, s, absorptivity=alpha)
    expected_sunlit = alpha * (beam + diffuse)
    expected_shadow = alpha * diffuse
    expected = np.where(s == 1.0, expected_sunlit, expected_shadow)
    np.testing.assert_allclose(h, expected)


def test_heating_soft_shadow_mask_attenuates_beam_linearly() -> None:
    # A fractional shadow value (e.g. partial occlusion in a future
    # soft-shadow model) should attenuate beam linearly.
    irr = _irradiance(beam=400.0, diffuse=100.0, shape=(1, 3))
    s = np.array([[1.0, 0.5, 0.0]], dtype=np.float64)
    h = heating_field(irr, s, absorptivity=1.0)
    np.testing.assert_allclose(h, np.array([[500.0, 300.0, 100.0]]))


# ---------------------------------------------------------------------------
# Per-cell absorptivity (Phase 4 dry-run)
# ---------------------------------------------------------------------------


def test_heating_array_absorptivity() -> None:
    # Mimic a per-cell alpha array (what Phase 4's land cover will
    # produce) and verify broadcasting.
    irr = _irradiance(beam=500.0, diffuse=100.0, shape=(2, 2))
    sun_lit = np.ones((2, 2), dtype=np.float64)
    alpha = np.array([[0.80, 0.40], [0.85, 0.05]], dtype=np.float64)
    h = heating_field(irr, sun_lit, absorptivity=alpha)
    expected = alpha * (500.0 + 100.0)
    np.testing.assert_allclose(h, expected)


# ---------------------------------------------------------------------------
# NaN propagation
# ---------------------------------------------------------------------------


def test_heating_nan_in_irradiance_propagates() -> None:
    beam = np.array([[500.0, np.nan]], dtype=np.float64)
    diffuse = np.array([[100.0, 100.0]], dtype=np.float64)
    irr = SlopeIrradiance(beam_wm2=beam, diffuse_wm2=diffuse)
    s = np.ones((1, 2), dtype=np.float64)
    h = heating_field(irr, s)
    assert np.isfinite(h[0, 0])
    assert np.isnan(h[0, 1])


def test_heating_nan_in_shadow_propagates() -> None:
    irr = _irradiance(beam=500.0, diffuse=100.0, shape=(1, 2))
    s = np.array([[1.0, np.nan]], dtype=np.float64)
    h = heating_field(irr, s)
    assert np.isfinite(h[0, 0])
    assert np.isnan(h[0, 1])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_heating_rejects_shape_mismatch_in_irradiance() -> None:
    irr = SlopeIrradiance(beam_wm2=np.zeros((3, 3)), diffuse_wm2=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="diffuse_wm2 shape"):
        heating_field(irr, np.ones((3, 3)))


def test_heating_rejects_shadow_shape_mismatch() -> None:
    irr = _irradiance(beam=500.0, diffuse=100.0, shape=(3, 3))
    with pytest.raises(ValueError, match="shadow_mask shape"):
        heating_field(irr, np.ones((3, 4)))


def test_heating_rejects_absorptivity_shape_mismatch() -> None:
    irr = _irradiance(beam=500.0, diffuse=100.0, shape=(3, 3))
    with pytest.raises(ValueError, match="absorptivity shape"):
        heating_field(irr, np.ones((3, 3)), absorptivity=np.ones((3, 4)))


def test_heating_rejects_out_of_range_scalar_absorptivity() -> None:
    irr = _irradiance(beam=500.0, diffuse=100.0, shape=(2, 2))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        heating_field(irr, np.ones((2, 2)), absorptivity=1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        heating_field(irr, np.ones((2, 2)), absorptivity=-0.1)
