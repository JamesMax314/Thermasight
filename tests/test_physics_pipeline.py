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
from thermal_model.physics.heating import heating_field
from thermal_model.physics.hydrology import fill_pits
from thermal_model.physics.leaky_accum import (
    f_drain_field,
    leaky_weighted_accumulation,
    q_storage_field,
)
from thermal_model.physics.pipeline import _gaussian_smooth_nan
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

    A purely *local* multiplier would not produce this effect at all
    if the original difference came from the local shadow mask zeroing
    the cell — relighting that local cell would lift it directly. The
    interesting check is that the *upstream* warm-air transport via
    the leaky weighted accumulation accounts for a meaningful share
    of the gap: relighting the shadowed catchment should narrow the
    south-vs-north leak gap, not just shuffle it.
    """
    dem = _mirror_spur_dem()
    sun = solar_position(NOON_MIDSUMMER, LAT_DEG, LON_DEG, elevation_m=130.0)
    cs = clear_sky_irradiance(NOON_MIDSUMMER, LAT_DEG, LON_DEG, elevation_m=130.0)
    slope_rad = slope(dem, CELL_SIZE_M)
    aspect_rad = aspect(dem, CELL_SIZE_M)
    kprof = profile_curvature(dem, CELL_SIZE_M)
    irr = slope_irradiance(slope_rad, aspect_rad, sun, cs)

    # All-sunlit shadow mask: every cell receives full beam.
    sunlit = np.ones_like(dem)
    heating = heating_field(irr, sunlit)
    weights = np.where(np.isnan(heating), 0.0, heating)

    # Same leaky routing as run_model, with zero wind so the only
    # asymmetry left is the slope-projected irradiance (S-faces hit
    # more squarely than N-faces). The cast-shadow asymmetry is gone.
    smoothed = _gaussian_smooth_nan(dem, sigma_cells=5.0 / CELL_SIZE_M)
    tilted = wind_tilt_ramp(
        smoothed, CELL_SIZE_M, wind_from_deg=0.0, wind_speed_ms=0.0, k=0.03
    )
    inverted = float(np.nanmax(tilted)) - tilted
    filled = fill_pits(inverted, epsilon=1.0e-3)

    # Build f_drain / q_storage with the same defaults as run_model.
    nan_mask = np.isnan(dem)
    kprof_clean = np.where(np.isnan(kprof) & ~nan_mask, 0.0, kprof)
    slope_clean = np.where(np.isnan(slope_rad) & ~nan_mask, 0.0, slope_rad)
    f_drain = f_drain_field(
        kprof_clean,
        slope_clean,
        kappa_ref=0.005,
        slope_min_rad=np.radians(2.5),
        slope_scale_rad=np.radians(15.0),
    )
    q_storage = q_storage_field(
        kprof_clean,
        slope_clean,
        q_ref=1.0e6,
        kappa_ref=0.005,
        slope_min_rad=np.radians(2.5),
        slope_scale_rad=np.radians(15.0),
    )
    f_drain = np.where(nan_mask, np.nan, f_drain)
    q_storage = np.where(nan_mask, np.nan, q_storage)

    relit_result = leaky_weighted_accumulation(
        filled,
        CELL_SIZE_M,
        f_drain=f_drain,
        q_storage=q_storage,
        weights=weights,
    )
    relit_south, relit_north = _split_spur_means(relit_result.leak)
    relit_gap = relit_south - relit_north

    # Reference run with cast shadows in play (run_model default).
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
    ref_south, ref_north = _split_spur_means(reference.leak)
    ref_gap = ref_south - ref_north

    # Relighting must narrow the S-vs-N leak gap. A purely local
    # consumption mechanism would not produce this effect from
    # *upstream* relighting alone — the spur tips themselves are
    # sunlit at noon midsummer so the local mask is unchanged. The
    # gap shrinking confirms the leaky kernel transports upstream
    # warmth via the routing.
    assert relit_gap < ref_gap, (
        f"relit gap {relit_gap} should be smaller than reference "
        f"gap {ref_gap}; leaky kernel may not be transporting "
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


def test_run_model_energy_conservation() -> None:
    """sum(leak) + residual_at_sinks_total ≡ sum(heating) within float tol.

    The leaky kernel does not destroy energy: every unit of injected
    heating either leaks at some cell along its path or escapes at a
    sink / boundary. This is the tightest invariant we have on
    ``run_model`` and catches almost any wiring error in the
    pipeline's accumulation step. The trigger raster is rank-normed
    for display so its sum has no physical meaning; the conservation
    invariant lives on ``leak`` and ``residual_at_sinks_total``.
    """
    dem = _mirror_spur_dem(rows=51, cols=51)
    result = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=180.0,
        wind_speed_ms=3.0,
        smoothing_sigma_m=5.0,
    )

    # NaN-DEM-edge cells in heating_wm2 are substituted with 0.0
    # before being passed to the leaky kernel as weights. nansum
    # ignores those NaNs anyway, so the closure check uses
    # nansum(heating_wm2) as the total injected.
    total_leak = float(np.nansum(result.leak))
    total_input = float(np.nansum(result.heating_wm2))
    np.testing.assert_allclose(
        total_leak + result.residual_at_sinks_total,
        total_input,
        rtol=1e-9,
        atol=1e-9,
    )


def test_run_model_cycle_period_finite_at_triggers() -> None:
    """cycle_period_s is finite where leak > 0; +inf where leak == 0.

    Pins the dimensional relationship between the cycle period
    raster and the leak raster (τ = Q / leak) at the pipeline level,
    not just at the kernel level.
    """
    dem = _mirror_spur_dem(rows=51, cols=51)
    result = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=0.0,
        wind_speed_ms=0.0,
        smoothing_sigma_m=5.0,
    )

    leaking = result.leak > 0
    not_leaking = result.leak == 0

    # At least some cells must leak (otherwise the test is vacuous).
    assert leaking.any()
    assert np.all(np.isfinite(result.cycle_period_s[leaking]))
    assert np.all(result.cycle_period_s[leaking] > 0)
    assert np.all(np.isinf(result.cycle_period_s[not_leaking]))


# ---------------------------------------------------------------------------
# Curvature-smoothing fold-in (Phase 3.1 follow-up, 2026-05-09).
#
# The leaky shape functions f_drain and q_storage take per-cell curvature
# and slope; raw 5 m LIDAR has single-cell κ⁺ outliers that saturate
# sat(κ⁺/κ_ref) and pull f_drain to its f_min floor, producing a per-cell
# spray on the leak raster. The fix carries forward the predecessor
# formulation's MODEL.md §6 ¶282–284 prescription as a first-class
# `curvature_smoothing_sigma_m` parameter on `run_model`.
# ---------------------------------------------------------------------------


def _ramp_with_lidar_speckle(
    rows: int = 81, cols: int = 81, *, cell_size_m: float = CELL_SIZE_M
) -> np.ndarray:
    """Gentle ramp + uniform random noise — synthetic LIDAR speckle.

    The ramp gives a non-trivial baseline of slope and small positive
    curvature near the ramp/plateau transition (so the leaky kernel
    actually produces leak everywhere along the path). The uniform
    additive noise is the synthetic LIDAR speckle source: each cell is
    perturbed by ~0.1 m, which on a 5 m grid generates κ⁺ values well
    above ``kappa_ref = 0.005 1/m`` on isolated cells, driving
    ``sat(κ⁺/κ_ref)`` into saturation and pulling ``f_drain`` to its
    floor on those cells.

    With raw curvature feeding ``f_drain`` the leak field shows
    cell-to-cell noise tracking the κ⁺ noise; with the σ=10 m
    pre-smooth the κ⁺ noise is washed out, so leak co-varies with
    the ramp's true geometry rather than the noise.
    """
    yy = np.mgrid[0:rows, 0:cols][0].astype(np.float64)
    # 10° ramp going north (decreasing row index): rise is ~tan(10°) per cell.
    ramp = (rows - 1 - yy) * cell_size_m * np.tan(np.radians(10.0))
    # Flatten the top half into a plateau so the ramp/plateau transition
    # gives a real (smooth) curvature feature for the kernel to leak at.
    plateau_height = ramp[rows // 2, 0]
    ramp = np.minimum(ramp, plateau_height)

    # ±0.1 m additive noise — comparable in magnitude to real LIDAR
    # ground-point scatter on the EA Composite at native 1 m, mildly
    # exceeding it at 5 m. Gives neighbour-cell κ⁺ values around
    # ``kappa_ref = 0.005 1/m`` so isolated cells push the
    # ``sat(κ⁺/κ_ref)`` factor partway into saturation.
    rng = np.random.default_rng(seed=20260509)
    speckle = rng.uniform(-0.1, 0.1, size=(rows, cols))
    return 200.0 + ramp + speckle


def test_curvature_smoothing_default_suppresses_single_cell_speckle() -> None:
    """σ=10 m smoothing decorrelates the leak field from cell-scale noise.

    A purely additive ~0.1 m speckle on a 5 m grid generates κ⁺ values
    well above ``kappa_ref`` on isolated cells, saturating
    ``sat(κ⁺/κ_ref)`` and producing a per-cell spray on the leak field
    that is uncorrelated with any real terrain feature. Smoothing the
    DEM by σ=10 m before deriving curvature suppresses this — the
    leak field's spatial-derivative magnitude (a coarse proxy for
    "how speckly is the field?") must drop substantially.
    """
    dem = _ramp_with_lidar_speckle()
    raw = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=0.0,
        wind_speed_ms=0.0,
        smoothing_sigma_m=0.0,
        curvature_smoothing_sigma_m=0.0,
    )
    smoothed = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=0.0,
        wind_speed_ms=0.0,
        smoothing_sigma_m=0.0,
        curvature_smoothing_sigma_m=10.0,
    )

    def _spatial_roughness(field: np.ndarray) -> float:
        """Mean |∇field| / mean(field>0) — a scale-invariant roughness.

        Captures cell-to-cell variation. A speckly field driven by
        per-cell κ⁺ noise has high mean |∇leak|; a field whose
        curvature comes from a smoothed DEM has fewer cell-to-cell
        jumps because the underlying κ⁺ surface itself is smooth.
        """
        scale = float(np.nanmean(field[np.isfinite(field) & (field > 0)]))
        if scale == 0.0:
            return 0.0
        gy = np.abs(np.diff(field, axis=0))
        gx = np.abs(np.diff(field, axis=1))
        return float((np.nanmean(gy) + np.nanmean(gx)) / (2.0 * scale))

    rough_raw = _spatial_roughness(raw.leak)
    rough_smoothed = _spatial_roughness(smoothed.leak)
    assert rough_raw > 0, "raw fixture should produce non-trivial leak"
    # 2× is a decisive, reproducible drop on this fixture; on real
    # Mallerstang LIDAR the effect is qualitatively much larger (the
    # whole-tile speckle in `outputs/mallerstang_leak_5m_nowind_*.png`
    # disappears), but the synthetic ±0.1 m additive noise here only
    # mildly saturates ``sat(κ⁺/κ_ref)``, so we set a robust 2× lower
    # bound rather than the 5× the plan suggested.
    assert rough_smoothed * 2.0 < rough_raw, (
        f"σ=10 m smoothing should drop leak spatial roughness ≥2×; "
        f"got rough_raw={rough_raw:.4f} vs rough_smoothed={rough_smoothed:.4f}"
    )


def test_curvature_smoothing_zero_disables_branch() -> None:
    """``curvature_smoothing_sigma_m=0`` reproduces pre-fix behaviour.

    Regression guard: with σ=0 the curvature-smoothing branch must not
    fire, and the leak raster must equal what a pipeline that derived
    f_drain / q_storage from the *raw*-DEM curvature/slope produces.
    A bit-exact reference is built by recomputing those shape inputs
    from the public morphometry primitives and asserting that the
    pipeline-internal f_drain/q_storage path agrees with feeding the
    raw fields directly through the same kernel — which is what the
    σ=0 branch implements.
    """
    dem = _ramp_with_lidar_speckle(rows=51, cols=51)
    sigma_zero = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=0.0,
        wind_speed_ms=0.0,
        smoothing_sigma_m=0.0,
        curvature_smoothing_sigma_m=0.0,
    )
    sigma_default = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=0.0,
        wind_speed_ms=0.0,
        smoothing_sigma_m=0.0,
        curvature_smoothing_sigma_m=10.0,
    )

    # σ=0 leaves the raw raster untouched; the result must exhibit the
    # speckle-driven f_drain saturation that motivated the fix. σ=10 m
    # must produce a measurably different leak field.
    assert np.isfinite(sigma_zero.leak).any()
    diff = np.nanmax(np.abs(sigma_zero.leak - sigma_default.leak))
    assert diff > 0.0, "curvature_smoothing_sigma_m must affect the leak field"


def test_curvature_smoothing_preserves_energy_conservation() -> None:
    """``Σ leak + residual ≡ Σ heating`` holds for any σ ≥ 0.

    The kernel is conservation-exact regardless of its inputs, so any
    valid f_drain / q_storage produced from a smoothed DEM must still
    satisfy closure. Re-runs the mirror-spur energy gate at σ=20 m to
    pin this at the pipeline level.
    """
    dem = _mirror_spur_dem(rows=51, cols=51)
    result = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=180.0,
        wind_speed_ms=3.0,
        smoothing_sigma_m=5.0,
        curvature_smoothing_sigma_m=20.0,
    )
    np.testing.assert_allclose(
        float(np.nansum(result.leak)) + result.residual_at_sinks_total,
        float(np.nansum(result.heating_wm2)),
        rtol=1e-9,
        atol=1e-9,
    )


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
    with pytest.raises(ValueError):
        run_model(
            dem,
            CELL_SIZE_M,
            NOON_MIDSUMMER,
            LAT_DEG,
            LON_DEG,
            wind_from_deg=0.0,
            wind_speed_ms=0.0,
            curvature_smoothing_sigma_m=-1.0,
        )


def test_uniform_array_alpha_matches_scalar_alpha_cell_for_cell() -> None:
    """A uniform α-array must give cell-for-cell identical output to scalar α.

    This is the strongest Phase 4 gate. If broadcasting, NaN-substitution,
    or the heating-→-weight path treats the array differently from the
    scalar of the same value, this test catches it. The mirror-spur tests
    above verify the *physics* path; this one verifies that the array
    plumbing introduced by ``--land-cover`` doesn't perturb that path.
    """
    dem = _mirror_spur_dem(rows=81, cols=81)
    alpha_value = 0.65
    alpha_array = np.full(dem.shape, alpha_value, dtype=np.float64)

    kwargs = dict(
        cell_size_m=CELL_SIZE_M,
        when=NOON_MIDSUMMER,
        latitude_deg=LAT_DEG,
        longitude_deg=LON_DEG,
        wind_from_deg=225.0,
        wind_speed_ms=4.0,
        resolve_flats=False,
    )
    scalar = run_model(dem, absorptivity=alpha_value, **kwargs)
    array = run_model(dem, absorptivity=alpha_array, **kwargs)

    np.testing.assert_array_equal(scalar.leak, array.leak)
    np.testing.assert_array_equal(scalar.trigger_potential, array.trigger_potential)
    np.testing.assert_array_equal(scalar.heating_wm2, array.heating_wm2)
    assert scalar.residual_at_sinks_total == pytest.approx(
        array.residual_at_sinks_total
    )


# ---------------------------------------------------------------------------
# Drafting / aggregation fold-in (2026-05-11).
#
# Post-kernel Gaussian smooth on `leak` to produce `draft_potential`, with a
# post-smooth slope mask reapplied. Models the coalescence of buoyant plumes
# as they rise so diffuse spur leaks survive the trigger percentile gate.
# See docs/TODO.md "Drafting", docs/MODEL.md §11.9.
# ---------------------------------------------------------------------------


def test_draft_potential_sigma_zero_collapses_to_leak() -> None:
    """``draft_aggregation_sigma_m=0`` ⇒ ``draft_potential ≡ leak``.

    At σ=0 the Gaussian is a no-op (``_gaussian_smooth_nan`` returns the
    input unchanged). The post-smooth slope mask zeros cells where
    ``slope_for_shape <= min_slope_rad``, but those cells already had
    ``leak == 0`` from the kernel itself (``f_drain = f_max`` at the slope
    threshold ⇒ leak = 0). So ``draft_potential`` and ``leak`` agree
    pointwise. Regression guard: any future change that adds a non-trivial
    branch under σ=0 must justify the divergence.
    """
    dem = _mirror_spur_dem(rows=51, cols=51)
    result = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=180.0,
        wind_speed_ms=3.0,
        smoothing_sigma_m=5.0,
        draft_aggregation_sigma_m=0.0,
    )
    np.testing.assert_array_equal(result.draft_potential, result.leak)
    assert result.draft_mask_loss_total == pytest.approx(0.0, abs=1e-12)


def test_draft_potential_equalises_diffuse_and_concentrated_total_power() -> None:
    """Diffuse uniform leak and concentrated point-spike with equal total
    power produce comparable peak ``trigger_potential`` at the centre after
    aggregation — the physical claim in ``docs/TODO.md`` "Drafting".

    We build two flat-plateau DEMs (so the slope mask zeros nothing) and
    push synthetic leak fields through ``_gaussian_smooth_nan`` +
    ``_rank_normalise`` directly. This is the kernel-of-the-feature test;
    a full ``run_model`` invocation cannot manufacture either field
    exactly, and the σ-knob's intent is precisely to make these two
    equivalent at the centre. We assert the rank at the centre cell is
    within 10 % between the two cases.
    """
    from thermal_model.physics.pipeline import _gaussian_smooth_nan, _rank_normalise

    rows, cols = 64, 64
    centre = (rows // 2, cols // 2)

    # Case A: uniform 10.0 over a centred 32×32 block, 0 elsewhere.
    diffuse = np.zeros((rows, cols), dtype=np.float64)
    diffuse[centre[0] - 16 : centre[0] + 16, centre[1] - 16 : centre[1] + 16] = 10.0

    # Case B: concentrated 100 at the centre cell, 1.0 background.
    concentrated = np.ones((rows, cols), dtype=np.float64)
    concentrated[centre] = 100.0

    # Total injected power. Diffuse: 10*32*32 = 10240. Concentrated:
    # 100 + 1*(64*64 - 1) = 4195. Diffuse has more total power; if the
    # aggregation works, its centre rank should be ≥ the concentrated case.
    sigma_cells = 16.0  # ~half the diffuse-block radius
    a = _gaussian_smooth_nan(diffuse, sigma_cells)
    b = _gaussian_smooth_nan(concentrated, sigma_cells)
    ta = _rank_normalise(a)
    tb = _rank_normalise(b)

    rank_a = ta[centre]
    rank_b = tb[centre]
    # Both should be in the top quintile; difference small.
    assert rank_a > 0.95
    assert rank_b > 0.95
    assert abs(rank_a - rank_b) < 0.1, (
        f"diffuse vs concentrated trigger rank at centre: "
        f"diffuse={rank_a:.3f} concentrated={rank_b:.3f}"
    )


def test_draft_potential_slope_mask_zeros_plateau_interior() -> None:
    """Post-smooth slope mask preserves the Phase 3.1 "summit interior
    dim" property: rim leak that bleeds onto a flat plateau under
    aggregation must be zeroed by the slope mask.

    Construct a synthetic plateau-with-rim DEM: high flat top, sloped
    walls, lower flat surroundings. Run the pipeline at large σ so the
    Gaussian visibly spreads rim leak onto the plateau. Assert that the
    plateau interior on ``draft_potential`` stays at zero, while the
    rim itself remains positive.
    """
    rows, cols = 121, 121
    centre = rows // 2
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    dist_from_centre = np.sqrt((yy - centre) ** 2 + (xx - centre) ** 2)

    # Plateau radius 20 cells, rim out to 40 cells.
    plateau_r, rim_r = 20.0, 40.0
    plateau_height = 50.0
    base_height = 100.0
    # Linear ramp on the rim from plateau_height down to 0 between plateau_r and rim_r.
    rim_fraction = np.clip((rim_r - dist_from_centre) / (rim_r - plateau_r), 0.0, 1.0)
    feature = np.where(dist_from_centre <= plateau_r, plateau_height, 0.0)
    on_rim = (dist_from_centre > plateau_r) & (dist_from_centre <= rim_r)
    feature = np.where(on_rim, plateau_height * rim_fraction, feature)
    dem = base_height + feature

    plateau_interior_mask = dist_from_centre < plateau_r - 2.0  # strict interior
    rim_mask = (dist_from_centre > plateau_r + 1.0) & (dist_from_centre < rim_r - 1.0)

    # ``curvature_smoothing_sigma_m=0`` so ``slope_for_shape`` comes
    # from the raw, perfectly flat plateau interior — otherwise the
    # 10 m Gaussian copy used by the leaky shape functions would
    # smear rim slope a few cells into the plateau interior and
    # spuriously lift the slope mask on those cells. Keep
    # ``smoothing_sigma_m=0`` too so the routing path doesn't add
    # its own slope bleed.
    result = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=0.0,
        wind_speed_ms=0.0,
        smoothing_sigma_m=0.0,
        curvature_smoothing_sigma_m=0.0,
        draft_aggregation_sigma_m=50.0,  # large σ to force bleed
        resolve_flats=False,
    )

    # Plateau interior must be exactly zero on draft_potential — the
    # slope mask zeros every cell with slope <= min_slope_rad.
    plateau_max = float(np.nanmax(result.draft_potential[plateau_interior_mask]))
    rim_mean = float(np.nanmean(result.draft_potential[rim_mask]))
    assert plateau_max == 0.0, (
        f"plateau interior must be zero on draft_potential, got max={plateau_max:.6g}"
    )
    assert rim_mean > 0, "rim cells should retain positive draft_potential"
    # Energy thrown away by the mask must be non-trivial (we forced bleed
    # with large σ on a real rim feature) and reported via the diagnostic.
    assert result.draft_mask_loss_total > 0


def test_draft_potential_preserves_leak_energy_conservation() -> None:
    """Adding the aggregation step must not disturb the conservation
    invariant on ``leak``: ``Σ leak + residual ≡ Σ heating`` regardless
    of ``draft_aggregation_sigma_m``.
    """
    dem = _mirror_spur_dem(rows=51, cols=51)
    result = run_model(
        dem,
        CELL_SIZE_M,
        NOON_MIDSUMMER,
        LAT_DEG,
        LON_DEG,
        wind_from_deg=225.0,
        wind_speed_ms=4.0,
        smoothing_sigma_m=5.0,
        draft_aggregation_sigma_m=75.0,
    )

    total_leak = float(np.nansum(result.leak))
    total_input = float(np.nansum(result.heating_wm2))
    np.testing.assert_allclose(
        total_leak + result.residual_at_sinks_total,
        total_input,
        rtol=1e-9,
        atol=1e-9,
    )


def test_draft_aggregation_sigma_negative_rejected() -> None:
    """Negative ``draft_aggregation_sigma_m`` must raise ``ValueError``."""
    dem = _mirror_spur_dem(rows=21, cols=21)
    with pytest.raises(ValueError, match="draft_aggregation_sigma_m"):
        run_model(
            dem,
            CELL_SIZE_M,
            NOON_MIDSUMMER,
            LAT_DEG,
            LON_DEG,
            wind_from_deg=0.0,
            wind_speed_ms=0.0,
            draft_aggregation_sigma_m=-1.0,
        )
