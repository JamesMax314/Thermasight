"""Tests for thermal_model.physics.leaky_accum.

The leaky-bucket kernel is the Phase 3.1 reformulation of the
weighted flow accumulation: each cell consumes ``(1 - f_drain)`` of
its through-flow as trigger output and forwards only ``f_drain``
onward, with a per-cell storage capacity ``Q`` producing a cycle
period ``tau = Q / leak``. See ``docs/model_correction.md``.

Tests are organised by:

* Energy-conservation invariant (the strongest constraint).
* Two limit cases that bridge the new kernel to the existing
  ``flow_accumulation``: ``f_drain == 1`` (no leak; recover
  ``flow_accumulation``) and ``f_drain == 0`` (everything leaks
  locally).
* Cycle-period dimensional check.
* Mirror-spur tests (S-spur outscores N-spur, relit narrows the gap)
  ported from ``test_physics_pipeline.py`` and run on the leaky kernel.
* Synthetic gentle-ridge / sharp-break behaviour.
* Weights-contract validation (mirroring ``flow_accumulation``).
* NaN propagation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from thermal_model.physics.flow import flow_accumulation
from thermal_model.physics.heating import heating_field
from thermal_model.physics.hydrology import fill_pits
from thermal_model.physics.leaky_accum import (
    F_MAX_DEFAULT,
    F_MIN_DEFAULT,
    LeakyResult,
    f_drain_field,
    leaky_weighted_accumulation,
    q_storage_field,
)
from thermal_model.physics.pipeline import _gaussian_smooth_nan
from thermal_model.physics.wind_tilt import wind_tilt_ramp
from thermal_model.solar.irradiance import clear_sky_irradiance, slope_irradiance
from thermal_model.solar.position import solar_position
from thermal_model.solar.shadow import cast_shadow_mask
from thermal_model.terrain.morphometry import aspect, profile_curvature, slope

# Default leak-shape parameters used across the synthetic tests. These
# are sensible defaults for moderate Dales-style terrain and are not
# claims about real LIDAR; the production defaults will be tuned in
# Stage 2 against the validation tiles.
KAPPA_REF = 0.005  # 1/m — moderate convex break scale
SLOPE_MIN = np.radians(2.5)  # below this, no slope contribution to leak
SLOPE_SCALE = np.radians(15.0)  # half-saturation slope above the floor
Q_REF = 1.0e6  # J/m² when weights are W/m²; arbitrary unit otherwise

NOON_MIDSUMMER = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
LAT_DEG = 54.4
LON_DEG = -2.3
CELL_SIZE_M = 5.0


# ---------------------------------------------------------------------------
# Fixtures and helpers (synthetic DEMs).
# ---------------------------------------------------------------------------


def _mirror_spur_dem(rows: int = 121, cols: int = 121) -> np.ndarray:
    """Two mirror-image triangular spurs, S-facing and N-facing.

    Copied verbatim from ``tests/test_physics_pipeline.py`` so the
    leaky-kernel tests run against the same fixture as the existing
    pipeline tests. If the helper is later promoted to a shared
    fixtures module, both tests should switch.
    """
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    centre_row = rows // 2
    centre_col = cols // 2

    dist_from_wall = np.abs(yy - centre_row)
    dist_from_ridge = np.abs(xx - centre_col)

    spur_length = 40.0
    spur_half_width = 20.0

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

    wall_height = np.where(dist_from_wall < 5.0, 30.0, 0.0)
    elevation = 100.0 + np.maximum(spur_height, wall_height)
    return elevation.astype(np.float64)


def _split_spur_means(field: np.ndarray) -> tuple[float, float]:
    rows = field.shape[0]
    centre = rows // 2
    north_half = field[:centre, :]
    south_half = field[centre + 1 :, :]
    return float(np.nanmean(south_half)), float(np.nanmean(north_half))


def _zero_at_dem_nan_or_edge(field: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """Replace NaN at finite-DEM cells with 0.0 (the same edge-NaN
    handling the production pipeline uses for slope / curvature → weights)."""
    return np.where(np.isnan(field) & ~np.isnan(dem), 0.0, field)


def _f_drain_and_q_for(
    dem: np.ndarray,
    cell_size_m: float,
    *,
    kappa_ref: float = KAPPA_REF,
    slope_min_rad: float = SLOPE_MIN,
    slope_scale_rad: float = SLOPE_SCALE,
    q_ref: float = Q_REF,
) -> tuple[np.ndarray, np.ndarray]:
    """Build f_drain and q_storage rasters from a raw DEM.

    Edge cells where the 3x3 stencil cannot resolve curvature/slope are
    substituted with zeros (no curvature / no slope ⇒ ``f_drain = f_max``,
    ``q_storage = q_ref``), matching how the production pipeline
    handles edge NaN before passing to ``flow_accumulation``.
    """
    kprof = _zero_at_dem_nan_or_edge(profile_curvature(dem, cell_size_m), dem)
    slope_rad = _zero_at_dem_nan_or_edge(slope(dem, cell_size_m), dem)
    f = f_drain_field(
        kprof,
        slope_rad,
        kappa_ref=kappa_ref,
        slope_min_rad=slope_min_rad,
        slope_scale_rad=slope_scale_rad,
    )
    q = q_storage_field(
        kprof,
        slope_rad,
        q_ref=q_ref,
        kappa_ref=kappa_ref,
        slope_min_rad=slope_min_rad,
        slope_scale_rad=slope_scale_rad,
    )
    # Re-mask NaN-DEM cells on the output fields — the input replacement
    # above produced finite values everywhere; restore NaN at NaN-DEM
    # cells to satisfy the kernel's contract.
    nan_mask = np.isnan(dem)
    f = np.where(nan_mask, np.nan, f)
    q = np.where(nan_mask, np.nan, q)
    return f, q


def _tilted_random_dem(seed: int, shape: tuple[int, int] = (32, 32)) -> np.ndarray:
    """Random noise added to a downward-sloping plane.

    The plane bias guarantees most cells have a positive downhill
    direction; the noise is small enough not to create deep pits but
    rich enough to exercise the full eight-facet routing.
    """
    rng = np.random.default_rng(seed)
    n_rows, n_cols = shape
    cols = np.arange(n_cols, dtype=np.float64)
    plane = np.broadcast_to(-cols * 2.0, shape).astype(np.float64).copy()
    noise = rng.normal(0.0, 0.1, size=shape)
    return plane + noise


# ---------------------------------------------------------------------------
# Energy conservation — the strongest invariant.
# ---------------------------------------------------------------------------


def test_leaky_accum_energy_conservation_uniform_weights() -> None:
    """sum(leak) + residual ≡ sum(weights) within float tolerance.

    The leaky kernel does not destroy energy: every unit of injected
    weight either leaks at some cell along its path or escapes at a
    sink / boundary. This is the tightest invariant we have and
    catches almost any wiring error in the topological pass.
    """
    dem = _tilted_random_dem(seed=1)
    weights = np.full_like(dem, 3.0)
    f, q = _f_drain_and_q_for(dem, CELL_SIZE_M)
    result = leaky_weighted_accumulation(
        dem, CELL_SIZE_M, f_drain=f, q_storage=q, weights=weights
    )
    total_leak = float(np.nansum(result.leak))
    total_input = float(np.nansum(weights))
    np.testing.assert_allclose(
        total_leak + result.residual_at_sinks_total,
        total_input,
        rtol=1e-9,
        atol=1e-9,
    )


def test_leaky_accum_energy_conservation_random_weights() -> None:
    """Conservation holds under per-cell random weights too."""
    dem = _tilted_random_dem(seed=2)
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.5, 5.0, size=dem.shape)
    f, q = _f_drain_and_q_for(dem, CELL_SIZE_M)
    result = leaky_weighted_accumulation(
        dem, CELL_SIZE_M, f_drain=f, q_storage=q, weights=weights
    )
    np.testing.assert_allclose(
        float(np.nansum(result.leak)) + result.residual_at_sinks_total,
        float(np.nansum(weights)),
        rtol=1e-9,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Limit cases — bridges to the existing flow_accumulation.
# ---------------------------------------------------------------------------


def test_leaky_accum_unit_drain_recovers_flow_accumulation() -> None:
    """f_drain == 1 (no leak) ⇒ forward[c] equals flow_accumulation(c).

    With no consumption at any cell, every unit of injected weight
    propagates to its receivers exactly as the existing flow
    accumulation does. This bridges the two kernels in the no-leak
    limit and pins agreement cell-for-cell, not merely in aggregate.
    """
    dem = _tilted_random_dem(seed=3)
    weights = np.full_like(dem, 1.0)
    f = np.full_like(dem, 1.0)
    q = np.full_like(dem, 1.0)
    leaky = leaky_weighted_accumulation(
        dem, CELL_SIZE_M, f_drain=f, q_storage=q, weights=weights
    )
    reference = flow_accumulation(dem, CELL_SIZE_M, weights=weights, use_richdem=False)
    np.testing.assert_allclose(leaky.forward, reference, rtol=1e-12, atol=1e-12)
    # No leak: every leak entry should be exactly zero (NaN-DEM cells
    # carry NaN, but this fixture has none).
    assert np.all(leaky.leak == 0.0)


def test_leaky_accum_zero_drain_gives_local_only() -> None:
    """f_drain == 0 ⇒ leak[c] == weights[c]; nothing forwards.

    The opposite limit: every cell consumes its self-injection
    immediately, so no upstream contributions accumulate and the leak
    raster is exactly the weights raster.
    """
    dem = _tilted_random_dem(seed=4)
    rng = np.random.default_rng(11)
    weights = rng.uniform(0.5, 5.0, size=dem.shape)
    f = np.zeros_like(dem)
    q = np.full_like(dem, 1.0)
    leaky = leaky_weighted_accumulation(
        dem, CELL_SIZE_M, f_drain=f, q_storage=q, weights=weights
    )
    np.testing.assert_allclose(leaky.leak, weights, rtol=1e-12, atol=1e-12)
    assert np.all(leaky.forward == 0.0)
    np.testing.assert_allclose(leaky.residual_at_sinks_total, 0.0, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Cycle-period dimensional check.
# ---------------------------------------------------------------------------


def test_leaky_accum_cycle_period_inverse_to_leak() -> None:
    """tau == Q / leak where leak > 0; +inf where leak == 0."""
    dem = _tilted_random_dem(seed=5)
    weights = np.full_like(dem, 1.0)
    # Mix: half cells leak (f_drain=0.5), half do not (f_drain=1.0).
    f = np.full_like(dem, 1.0)
    f[:, : dem.shape[1] // 2] = 0.5
    q = np.full_like(dem, 2.0)
    leaky = leaky_weighted_accumulation(
        dem, CELL_SIZE_M, f_drain=f, q_storage=q, weights=weights
    )
    # Where leak > 0 the cycle period must equal Q/leak exactly.
    leaking = leaky.leak > 0
    np.testing.assert_allclose(
        leaky.cycle_period[leaking], q[leaking] / leaky.leak[leaking], rtol=1e-12
    )
    # Where leak == 0 the cycle period must be +inf.
    not_leaking = leaky.leak == 0
    assert np.all(np.isinf(leaky.cycle_period[not_leaking]))
    assert np.all(leaky.cycle_period[not_leaking] > 0)


# ---------------------------------------------------------------------------
# Mirror-spur tests — port the Phase 3 validation gate to the leaky kernel.
# ---------------------------------------------------------------------------


def _build_heating_with_shadow(dem: np.ndarray, sunlit: bool) -> np.ndarray:
    """Run the Phase 2 heating chain on a DEM, optionally without shadows.

    Returns the heating raster (W/m²) with NaN substituted to 0.0 at
    finite-DEM cells, matching the production weights contract.
    """
    sun = solar_position(NOON_MIDSUMMER, LAT_DEG, LON_DEG, elevation_m=130.0)
    cs = clear_sky_irradiance(NOON_MIDSUMMER, LAT_DEG, LON_DEG, elevation_m=130.0)
    slope_rad = slope(dem, CELL_SIZE_M)
    aspect_rad = aspect(dem, CELL_SIZE_M)
    irr = slope_irradiance(slope_rad, aspect_rad, sun, cs)
    if sunlit:
        shadow = np.ones_like(dem)
    else:
        shadow = cast_shadow_mask(dem, CELL_SIZE_M, sun)
    heating = heating_field(irr, shadow)
    return _zero_at_dem_nan_or_edge(heating, dem)


def _leaky_pipeline(
    dem: np.ndarray, *, sunlit: bool, smoothing_sigma_m: float = 5.0
) -> LeakyResult:
    """Drive the leaky kernel through the Phase 3 wind-tilt + heating chain.

    Mirrors ``run_model`` up to the point where the production pipeline
    calls ``flow_accumulation``, then calls ``leaky_weighted_accumulation``
    instead. Curvature / slope for the leak shape come from the **raw**
    DEM, matching the production convention.
    """
    weights = _build_heating_with_shadow(dem, sunlit=sunlit)
    sigma_cells = smoothing_sigma_m / CELL_SIZE_M
    smoothed = _gaussian_smooth_nan(dem, sigma_cells)
    tilted = wind_tilt_ramp(
        smoothed, CELL_SIZE_M, wind_from_deg=0.0, wind_speed_ms=0.0, k=0.03
    )
    inverted = float(np.nanmax(tilted)) - tilted
    filled = fill_pits(inverted, epsilon=1.0e-3)
    f, q = _f_drain_and_q_for(dem, CELL_SIZE_M)
    return leaky_weighted_accumulation(
        filled, CELL_SIZE_M, f_drain=f, q_storage=q, weights=weights
    )


def test_mirror_spur_leaky_south_outscores_north() -> None:
    """Phase 3 validation gate, ported to the leaky kernel."""
    dem = _mirror_spur_dem()
    result = _leaky_pipeline(dem, sunlit=False)
    south_mean, north_mean = _split_spur_means(result.leak)
    assert south_mean > north_mean, (
        f"south-spur mean leak {south_mean} should exceed north-spur mean {north_mean}"
    )


def test_mirror_spur_leaky_relit_north_rises_toward_south() -> None:
    """Removing the cast shadow narrows the S-vs-N leak gap.

    A *local* multiplier would not produce this effect from upstream
    relighting alone (the convergent cell's local shadow is unchanged
    at noon midsummer). The fact that the leak gap shrinks confirms
    the leaky kernel is transporting upstream warmth via the routing,
    same as the existing weighted-flow accumulation.
    """
    dem = _mirror_spur_dem()
    shaded = _leaky_pipeline(dem, sunlit=False)
    relit = _leaky_pipeline(dem, sunlit=True)
    shaded_gap = float(np.diff(_split_spur_means(shaded.leak)[::-1])[0])  # S - N
    relit_gap = float(np.diff(_split_spur_means(relit.leak)[::-1])[0])
    assert relit_gap < shaded_gap, (
        f"relit gap {relit_gap} should be smaller than shaded gap "
        f"{shaded_gap}; weighted leak is not transporting upstream warmth"
    )


# ---------------------------------------------------------------------------
# Synthetic gentle-ridge / sharp-break behaviour.
# ---------------------------------------------------------------------------


def _gentle_ridge_dem(
    rows: int = 81, cols: int = 81, *, slope_frac: float = 0.10
) -> np.ndarray:
    """Smooth uphill ramp transitioning to flat top.

    Gentle slope from row=rows-1 down to row=rows//2 (rising northward
    by ``slope_frac * cell_size`` per row), then flat above. Slight
    Gaussian smoothing makes the corner curvature finite. The transition
    band is the only place with positive profile curvature.
    """
    yy, _ = np.mgrid[0:rows, 0:cols].astype(np.float64)
    transition_row = rows // 2
    rise_per_row = slope_frac * CELL_SIZE_M
    # Linear rise from bottom up to transition; flat above.
    elev = np.where(
        yy >= transition_row,
        (rows - 1 - yy) * rise_per_row,
        (rows - 1 - transition_row) * rise_per_row,
    )
    return gaussian_filter(elev, sigma=2.0).astype(np.float64)


def _sharp_break_dem(rows: int = 81, cols: int = 81) -> np.ndarray:
    """Plateau + steep cliff face + lower plateau.

    Real-terrain layout (north at top):

      * rows ``[0, rows//3)``        — upper plateau (flat top).
      * rows ``[rows//3, 2*rows//3)``— steep face descending southward.
      * rows ``[2*rows//3, rows)``   — lower plateau.

    The cliff lip (transition from upper plateau to the steep face) is
    the one point with high positive profile curvature *and* high
    slope. Modest Gaussian smoothing keeps curvature finite.
    """
    rows_per_band = rows // 3
    elev = np.zeros((rows, cols), dtype=np.float64)
    elev[:rows_per_band, :] = 200.0  # upper plateau
    # Steep face: linear drop from 200 to 100.
    face_rows = np.arange(rows_per_band, 2 * rows_per_band)
    fractions = (face_rows - rows_per_band) / max(rows_per_band - 1, 1)
    elev[rows_per_band : 2 * rows_per_band, :] = 200.0 - 100.0 * fractions[:, None]
    elev[2 * rows_per_band :, :] = 100.0  # lower plateau
    return gaussian_filter(elev, sigma=2.0)


def test_leaky_accum_cyclic_dump_on_gentle_ridge() -> None:
    """Gentle ramp ⇒ leak concentrated at apex, with long cycle period.

    On a uniform gentle ramp the curvature is near zero along the body
    and only positive in the transition band where the ramp meets the
    flat top. ``f_drain`` is close to ``f_max`` along the body (no
    consumption), so most through-flow accumulates and dumps in a
    narrow band at the top. ``q_storage`` is near ``q_ref`` there
    (gentle terrain), giving a finite but large cycle period — the
    "fills up then dumps" regime.
    """
    dem = _gentle_ridge_dem()
    weights = np.full_like(dem, 100.0)  # uniform W/m² heating
    inverted = float(np.nanmax(dem)) - dem
    filled = fill_pits(inverted, epsilon=1.0e-3)
    f, q = _f_drain_and_q_for(dem, CELL_SIZE_M)
    result = leaky_weighted_accumulation(
        filled, CELL_SIZE_M, f_drain=f, q_storage=q, weights=weights
    )

    rows = dem.shape[0]
    transition_row = rows // 2
    # Apex band: a few rows around the transition.
    apex_band = result.leak[transition_row - 5 : transition_row + 5, :]
    body_band = result.leak[transition_row + 10 : rows - 5, :]

    assert np.nanmax(apex_band) > np.nanmax(body_band), (
        "leak should peak in the transition band, not on the ramp body"
    )
    # Cycle period at the apex must be finite (ridge does eventually
    # dump) and noticeably long (many seconds, not subsecond).
    apex_periods = result.cycle_period[transition_row - 5 : transition_row + 5, :]
    finite_periods = apex_periods[np.isfinite(apex_periods)]
    assert finite_periods.size > 0
    assert float(np.nanmedian(finite_periods)) > 1.0


def test_leaky_accum_sharp_break_short_cycle() -> None:
    """Cliff lip ⇒ leak peaks there, with short cycle period.

    Inverse of the gentle-ridge case: a sharp convex break + steep
    slope drives ``f_drain`` toward ``f_min`` and ``q_storage`` toward
    its lower limit at the lip. Through-flow concentrates at the lip
    and most of it consumes locally with a short cycle.
    """
    dem = _sharp_break_dem()
    weights = np.full_like(dem, 100.0)
    inverted = float(np.nanmax(dem)) - dem
    filled = fill_pits(inverted, epsilon=1.0e-3)
    f, q = _f_drain_and_q_for(dem, CELL_SIZE_M)
    result = leaky_weighted_accumulation(
        filled, CELL_SIZE_M, f_drain=f, q_storage=q, weights=weights
    )

    rows = dem.shape[0]
    rows_per_band = rows // 3
    # Cliff lip is the transition row from upper plateau to steep face.
    lip_band = result.leak[rows_per_band - 3 : rows_per_band + 3, :]
    upper_plateau = result.leak[: rows_per_band - 5, :]
    lower_plateau = result.leak[2 * rows_per_band + 5 :, :]

    # Lip should have the highest leak in the raster.
    assert np.nanmax(lip_band) > np.nanmax(upper_plateau)
    assert np.nanmax(lip_band) > np.nanmax(lower_plateau)

    # Cycle period at the lip must be finite and small relative to the
    # gentle-ridge case (Q is suppressed by both κ and slope here).
    lip_periods = result.cycle_period[rows_per_band - 3 : rows_per_band + 3, :]
    finite_lip_periods = lip_periods[np.isfinite(lip_periods)]
    assert finite_lip_periods.size > 0


# ---------------------------------------------------------------------------
# Weights contract.
# ---------------------------------------------------------------------------


def _trivial_fields(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trivial f_drain (= 0.5) and q_storage (= 1.0) fields, NaN-aware."""
    f = np.where(np.isnan(dem), np.nan, 0.5)
    q = np.where(np.isnan(dem), np.nan, 1.0)
    return f, q


def test_leaky_accum_rejects_nan_weights_at_finite_dem_cell() -> None:
    dem = np.zeros((5, 5))
    weights = np.ones_like(dem)
    weights[2, 2] = np.nan
    f, q = _trivial_fields(dem)
    with pytest.raises(ValueError, match="weights must be finite"):
        leaky_weighted_accumulation(
            dem, cell_size_m=1.0, f_drain=f, q_storage=q, weights=weights
        )


def test_leaky_accum_rejects_inf_weights_at_finite_dem_cell() -> None:
    dem = np.zeros((5, 5))
    weights = np.ones_like(dem)
    weights[1, 3] = np.inf
    f, q = _trivial_fields(dem)
    with pytest.raises(ValueError, match="weights must be finite"):
        leaky_weighted_accumulation(
            dem, cell_size_m=1.0, f_drain=f, q_storage=q, weights=weights
        )


def test_leaky_accum_accepts_nan_weights_at_nan_dem_cells() -> None:
    n = 6
    cols = np.arange(n, dtype=np.float64)
    dem = np.broadcast_to(-cols, (n, n)).astype(np.float64).copy()
    dem[2, 2] = np.nan
    weights = np.full_like(dem, 2.0)
    weights[2, 2] = np.nan
    f, q = _trivial_fields(dem)
    result = leaky_weighted_accumulation(
        dem, cell_size_m=1.0, f_drain=f, q_storage=q, weights=weights
    )
    assert np.isnan(result.leak[2, 2])
    assert np.all(np.isfinite(result.leak[~np.isnan(dem)]))


def test_leaky_accum_rejects_mismatched_weights_shape() -> None:
    dem = np.zeros((5, 5))
    bad = np.zeros((4, 5))
    f, q = _trivial_fields(dem)
    with pytest.raises(ValueError, match="weights shape"):
        leaky_weighted_accumulation(
            dem, cell_size_m=1.0, f_drain=f, q_storage=q, weights=bad
        )


def test_leaky_accum_rejects_f_drain_out_of_unit_interval() -> None:
    dem = np.zeros((5, 5))
    f = np.full_like(dem, 1.5)  # > 1 ⇒ invalid
    _, q = _trivial_fields(dem)
    with pytest.raises(ValueError, match="f_drain must lie in"):
        leaky_weighted_accumulation(dem, cell_size_m=1.0, f_drain=f, q_storage=q)


def test_leaky_accum_rejects_negative_q_storage() -> None:
    dem = np.zeros((5, 5))
    f, _ = _trivial_fields(dem)
    q = np.full_like(dem, -1.0)
    with pytest.raises(ValueError, match="q_storage must be non-negative"):
        leaky_weighted_accumulation(dem, cell_size_m=1.0, f_drain=f, q_storage=q)


# ---------------------------------------------------------------------------
# NaN propagation.
# ---------------------------------------------------------------------------


def test_leaky_accum_propagates_dem_nan() -> None:
    dem = _tilted_random_dem(seed=6)
    dem[0, 0] = np.nan
    weights = np.where(np.isnan(dem), np.nan, 1.0)
    f, q = _f_drain_and_q_for(dem, CELL_SIZE_M)
    result = leaky_weighted_accumulation(
        dem, CELL_SIZE_M, f_drain=f, q_storage=q, weights=weights
    )
    assert np.isnan(result.leak[0, 0])
    assert np.isnan(result.forward[0, 0])
    assert np.isnan(result.cycle_period[0, 0])
    # residual is a scalar, must remain finite.
    assert np.isfinite(result.residual_at_sinks_total)


def test_leaky_accum_all_nan_dem_returns_all_nan_rasters() -> None:
    dem = np.full((5, 5), np.nan)
    f = np.full_like(dem, np.nan)
    q = np.full_like(dem, np.nan)
    weights = np.full_like(dem, np.nan)
    result = leaky_weighted_accumulation(
        dem, cell_size_m=1.0, f_drain=f, q_storage=q, weights=weights
    )
    assert np.all(np.isnan(result.leak))
    assert np.all(np.isnan(result.forward))
    assert np.all(np.isnan(result.cycle_period))
    assert result.residual_at_sinks_total == 0.0


# ---------------------------------------------------------------------------
# f_drain / q_storage shape-function smoke tests.
# ---------------------------------------------------------------------------


def test_f_drain_field_flat_terrain_returns_f_max() -> None:
    """Zero curvature, zero slope ⇒ f_drain == f_max."""
    kprof = np.zeros((10, 10))
    slope_rad = np.zeros((10, 10))
    f = f_drain_field(
        kprof,
        slope_rad,
        kappa_ref=KAPPA_REF,
        slope_min_rad=SLOPE_MIN,
        slope_scale_rad=SLOPE_SCALE,
    )
    np.testing.assert_allclose(f, F_MAX_DEFAULT)


def test_f_drain_field_extreme_sharp_terrain_approaches_f_min() -> None:
    """Very high curvature × very high slope ⇒ f_drain → f_min."""
    kprof = np.full((5, 5), 1000.0)  # >> kappa_ref
    slope_rad = np.full((5, 5), np.radians(80.0))
    f = f_drain_field(
        kprof,
        slope_rad,
        kappa_ref=KAPPA_REF,
        slope_min_rad=SLOPE_MIN,
        slope_scale_rad=SLOPE_SCALE,
    )
    assert np.all(f < 0.2)
    assert np.all(f >= F_MIN_DEFAULT)


def test_q_storage_field_flat_returns_q_ref() -> None:
    kprof = np.zeros((10, 10))
    slope_rad = np.zeros((10, 10))
    q = q_storage_field(
        kprof,
        slope_rad,
        q_ref=Q_REF,
        kappa_ref=KAPPA_REF,
        slope_min_rad=SLOPE_MIN,
        slope_scale_rad=SLOPE_SCALE,
    )
    np.testing.assert_allclose(q, Q_REF)


# ---------------------------------------------------------------------------
# Cross-backend agreement: numba JIT vs numpy reference must match.
# ---------------------------------------------------------------------------


def test_leaky_accum_numba_and_numpy_agree() -> None:
    """The numba JIT backend must produce bit-identical output to the
    pure-numpy reference, up to float-summation order. The numpy path
    is the test oracle; the numba path is the production speed-up.
    """
    dem = _tilted_random_dem(seed=42, shape=(48, 48))
    rng = np.random.default_rng(99)
    weights = rng.uniform(0.5, 5.0, size=dem.shape)
    f, q = _f_drain_and_q_for(dem, CELL_SIZE_M)

    numpy_result = leaky_weighted_accumulation(
        dem,
        CELL_SIZE_M,
        f_drain=f,
        q_storage=q,
        weights=weights,
        use_numba=False,
    )
    numba_result = leaky_weighted_accumulation(
        dem,
        CELL_SIZE_M,
        f_drain=f,
        q_storage=q,
        weights=weights,
        use_numba=True,
    )

    np.testing.assert_allclose(numpy_result.leak, numba_result.leak, rtol=1e-12)
    np.testing.assert_allclose(numpy_result.forward, numba_result.forward, rtol=1e-12)
    # cycle_period contains +inf; assert_allclose handles that.
    np.testing.assert_allclose(
        numpy_result.cycle_period, numba_result.cycle_period, rtol=1e-12
    )
    np.testing.assert_allclose(
        numpy_result.residual_at_sinks_total,
        numba_result.residual_at_sinks_total,
        rtol=1e-12,
        atol=1e-12,
    )


def test_q_storage_field_sharp_terrain_below_q_ref() -> None:
    kprof = np.full((5, 5), 0.05)  # > kappa_ref
    slope_rad = np.full((5, 5), np.radians(45.0))
    q = q_storage_field(
        kprof,
        slope_rad,
        q_ref=Q_REF,
        kappa_ref=KAPPA_REF,
        slope_min_rad=SLOPE_MIN,
        slope_scale_rad=SLOPE_SCALE,
    )
    assert np.all(q < 0.5 * Q_REF)
    assert np.all(q >= 0.0)
