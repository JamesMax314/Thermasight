"""Leaky-bucket weighted flow accumulation (Phase 3.1).

The Phase 3 trigger pipeline routes heating as a per-cell weight on
D-infinity flow accumulation of the inverted, wind-tilted DEM and
multiplies the result by a positive-curvature factor and a slope
mask. That formulation has two physical defects: it double-counts
energy along the flow path (every cell on a flow path inherits the
full upstream catchment), and it has no notion of cycle time on
gentle terrain (where the boundary layer fills up and dumps in one
release, rather than steady triggering at a sharp break).

This module replaces the post-hoc multiply with a *leaky-bucket*
weighted accumulation. Each cell consumes a curvature/slope-dependent
fraction ``(1 - f_drain)`` of its through-flow as trigger output and
forwards only ``f_drain`` of it onward. A per-cell storage capacity
``Q`` gives the buoyancy reservoir; the cycle period
``tau = Q / leak`` is the time between successive releases at that
cell in steady state.

Energy is conserved along the path, in the units of ``weights``::

    sum(leak) + residual_at_sinks ≡ sum(weights)

This is exact in the pure-numpy implementation up to float rounding.
A pinned property test enforces it.

See ``docs/model_correction.md`` and ``docs/MODEL.md`` §5 for the full
derivation. This module is the Stage 1 spike of the reformulation;
the production pipeline (``thermal_model.physics.pipeline.run_model``)
is not yet wired to use it.

References
----------
Tarboton, D.G. (1997). A new method for the determination of flow
directions and upslope areas in grid digital elevation models. Water
Resources Research, 33(2), 309-319.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from thermal_model.physics.flow import (
    _FACETS,
    _facet_slopes,
    _validate_dem,
    _validate_weights,
)

# Per-facet receiver offsets pre-baked into int64 arrays for the numba
# topological pass. The numpy pass uses the same arrays for parity.
_FACET_OFFSETS_CARD: np.ndarray = np.array(
    [[f[0][0], f[0][1]] for f in _FACETS], dtype=np.int64
)
_FACET_OFFSETS_DIAG: np.ndarray = np.array(
    [[f[1][0], f[1][1]] for f in _FACETS], dtype=np.int64
)


def _have_numba() -> bool:
    return importlib.util.find_spec("numba") is not None


_HAS_NUMBA: bool = _have_numba()


F_MIN_DEFAULT = 0.15
"""Skimming floor on the drain fraction.

Even at the sharpest convex break, some warm air slips past the
trigger as boundary-layer skim — there is no geometry that can
detach 100% of through-flow. Defaults to ``0.15`` (15% always
forwards).
"""

F_MAX_DEFAULT = 1.0
"""Maximum drain fraction. On flats and concave terrain, all through-flow
forwards (zero leak)."""


@dataclass(frozen=True)
class LeakyResult:
    """Outputs of :func:`leaky_weighted_accumulation`.

    Attributes
    ----------
    leak : np.ndarray
        Per-cell time-averaged release rate, same units as ``weights``
        (W/m² when heating is the weight). NaN at NaN-DEM cells.
    forward : np.ndarray
        Per-cell post-leak through-flow that was passed to D-infinity
        neighbours. ``forward[c] = f_drain[c] * r[c]`` where ``r[c]``
        is the through-flow rate at ``c`` (self-injection plus all
        upstream contributions). NaN at NaN-DEM cells.
    cycle_period : np.ndarray
        Per-cell buoyancy-cycle period, ``q_storage / leak``. ``+inf``
        where ``leak == 0`` (no release: cycle never completes). NaN
        at NaN-DEM cells.
    residual_at_sinks_total : float
        Total ``forward`` that reached cells with no D-infinity outflow
        (true sinks on the inverted DEM — real-terrain summits and
        domain-boundary outlets) without being consumed. A diagnostic
        for parameter tuning: a large fraction of total injected weight
        ending up here means ``f_drain`` is too high (or ``q_storage``
        too high) and triggers are being under-counted.
    """

    leak: np.ndarray
    forward: np.ndarray
    cycle_period: np.ndarray
    residual_at_sinks_total: float


def _saturating(x: np.ndarray) -> np.ndarray:
    """Soft non-negative saturation: 0 at ``x <= 0``, → 1 for large ``x``.

    Equivalent to ``1 - exp(-max(x, 0))``. Smoothly bounded on
    ``[0, 1)``, exactly zero for non-positive inputs (so flats with
    ``profile_curv <= 0`` get zero contribution to ``sharpness``).
    Exponential saturation reaches ``≈ 0.95`` by ``x = 3`` and
    ``≈ 0.99`` by ``x = 5``, which is fast enough that realistic
    terrain values (a few reference scales) drive ``f_drain`` close
    to ``f_min`` rather than asymptoting only at impossible extremes.
    """
    x_pos = np.maximum(x, 0.0)
    out: np.ndarray = 1.0 - np.exp(-x_pos)
    return out


def _validate_field_shape(name: str, dem: np.ndarray, field: np.ndarray) -> None:
    if field.shape != dem.shape:
        raise ValueError(f"{name} shape {field.shape} != dem shape {dem.shape}")


def _validate_f_drain(dem: np.ndarray, f_drain: np.ndarray) -> None:
    _validate_field_shape("f_drain", dem, f_drain)
    finite_dem = ~np.isnan(dem)
    finite_at_dem = f_drain[finite_dem]
    if not np.all(np.isfinite(finite_at_dem)):
        raise ValueError(
            "f_drain must be finite (no NaN, no Inf) at every cell where "
            "dem is finite. NaN is permitted only at NaN-dem cells."
        )
    if (finite_at_dem < 0.0).any() or (finite_at_dem > 1.0).any():
        raise ValueError("f_drain must lie in [0, 1] at every finite-dem cell")


def _validate_q_storage(dem: np.ndarray, q_storage: np.ndarray) -> None:
    _validate_field_shape("q_storage", dem, q_storage)
    finite_dem = ~np.isnan(dem)
    finite_at_dem = q_storage[finite_dem]
    if not np.all(np.isfinite(finite_at_dem)):
        raise ValueError(
            "q_storage must be finite at every finite-dem cell. "
            "NaN allowed only at NaN-dem cells."
        )
    if (finite_at_dem < 0.0).any():
        raise ValueError("q_storage must be non-negative at every finite-dem cell")


def f_drain_field(
    profile_curv: np.ndarray,
    slope_rad: np.ndarray,
    *,
    kappa_ref: float,
    slope_min_rad: float,
    slope_scale_rad: float,
    f_min: float = F_MIN_DEFAULT,
    f_max: float = F_MAX_DEFAULT,
) -> np.ndarray:
    """Per-cell drain fraction as a function of curvature and slope.

    Parameters
    ----------
    profile_curv : np.ndarray
        Profile curvature (Zevenbergen-Thorne convention: positive =
        convex, negative = concave), units of inverse metres. Computed
        from the **raw** DEM — detachment is a property of real terrain
        shape, not the wind-tilted routing surface.
    slope_rad : np.ndarray
        Slope magnitude in radians. From the raw DEM.
    kappa_ref : float
        Reference curvature scale (1/m). At ``profile_curv = kappa_ref``
        the curvature contribution to ``sharpness`` is ``0.5``.
    slope_min_rad : float
        Slope below which the slope contribution to ``sharpness`` is
        zero. Encodes "flat surfaces don't trigger".
    slope_scale_rad : float
        Reference slope scale (radians). At
        ``slope - slope_min_rad = slope_scale_rad`` the slope
        contribution to ``sharpness`` is ``0.5``.
    f_min : float
        Skimming floor on the drain fraction. See :data:`F_MIN_DEFAULT`.
    f_max : float
        Maximum drain fraction. See :data:`F_MAX_DEFAULT`.

    Returns
    -------
    np.ndarray
        Per-cell drain fraction, clipped to ``[f_min, f_max]``.

    Notes
    -----
    The shape function is::

        sharpness = saturating(kappa_pos / kappa_ref)
                    * saturating((slope - slope_min) / slope_scale)
        f_drain   = f_max - (f_max - f_min) * sharpness

    where ``saturating(x) = max(x, 0) / (max(x, 0) + 1)``. The
    saturation is exactly zero for non-positive inputs so:

      * concave or zero curvature → ``sharpness_curv = 0``
      * slope at or below ``slope_min`` → ``sharpness_slope = 0``

    In either case ``f_drain = f_max`` (forward everything). Both
    factors must be non-negligible for ``f_drain`` to drop toward
    ``f_min`` — a sharp curvature on a flat surface still forwards;
    a steep slope with no convexity also forwards.
    """
    if not (0.0 <= f_min <= f_max <= 1.0):
        raise ValueError(f"require 0 <= f_min ({f_min}) <= f_max ({f_max}) <= 1")
    if kappa_ref <= 0.0:
        raise ValueError(f"kappa_ref must be positive, got {kappa_ref}")
    if slope_scale_rad <= 0.0:
        raise ValueError(f"slope_scale_rad must be positive, got {slope_scale_rad}")
    if profile_curv.shape != slope_rad.shape:
        raise ValueError(
            f"profile_curv shape {profile_curv.shape} "
            f"!= slope_rad shape {slope_rad.shape}"
        )

    sharpness_curv = _saturating(profile_curv / kappa_ref)
    sharpness_slope = _saturating((slope_rad - slope_min_rad) / slope_scale_rad)
    sharpness = sharpness_curv * sharpness_slope
    f_drain = f_max - (f_max - f_min) * sharpness
    out: np.ndarray = np.clip(f_drain, f_min, f_max)
    return out


def q_storage_field(
    profile_curv: np.ndarray,
    slope_rad: np.ndarray,
    *,
    q_ref: float,
    kappa_ref: float,
    slope_min_rad: float,
    slope_scale_rad: float,
) -> np.ndarray:
    """Per-cell buoyancy storage capacity, units of ``weights × time``.

    Storage is large on gentle / flat terrain (the boundary layer can
    grow tall before the buoyancy cap is overcome) and small on
    sharp / steep terrain (the geometry forces release at low buildup).

    Parameters
    ----------
    profile_curv, slope_rad
        See :func:`f_drain_field`.
    q_ref : float
        Reference storage capacity (J/m² when ``weights`` is in W/m²).
        The flat / no-curvature limit of ``Q``.
    kappa_ref : float
        Curvature scale (1/m), shared with :func:`f_drain_field` so the
        two fields are dimensionally coherent.
    slope_min_rad : float
        Slope below which the slope term is unattenuated.
    slope_scale_rad : float
        Reference slope scale (radians).

    Returns
    -------
    np.ndarray
        Per-cell storage capacity. Always non-negative and ``<= q_ref``.

    Notes
    -----
    The shape function is::

        Q = q_ref * exp(-max(kappa, 0) / kappa_ref)
                  * exp(-max(slope - slope_min, 0) / slope_scale)

    Both exponentials are clipped at ``1.0`` for non-positive arguments
    (so flats / concave terrain get the full ``q_ref``).
    """
    if q_ref <= 0.0:
        raise ValueError(f"q_ref must be positive, got {q_ref}")
    if kappa_ref <= 0.0:
        raise ValueError(f"kappa_ref must be positive, got {kappa_ref}")
    if slope_scale_rad <= 0.0:
        raise ValueError(f"slope_scale_rad must be positive, got {slope_scale_rad}")
    if profile_curv.shape != slope_rad.shape:
        raise ValueError(
            f"profile_curv shape {profile_curv.shape} "
            f"!= slope_rad shape {slope_rad.shape}"
        )

    curv_term = np.exp(-np.maximum(profile_curv, 0.0) / kappa_ref)
    slope_term = np.exp(-np.maximum(slope_rad - slope_min_rad, 0.0) / slope_scale_rad)
    out: np.ndarray = q_ref * curv_term * slope_term
    return out


def _leaky_pass_python(
    finite_rows: np.ndarray,
    finite_cols: np.ndarray,
    order: np.ndarray,
    r: np.ndarray,
    f_drain: np.ndarray,
    has_flow: np.ndarray,
    nan_mask: np.ndarray,
    best_facet: np.ndarray,
    best_r: np.ndarray,
    leak: np.ndarray,
    forward: np.ndarray,
) -> float:
    """Topological-order leaky pass — pure numpy / Python loop.

    Reference implementation. Mutates ``r``, ``leak``, ``forward`` in
    place and returns ``residual_at_sinks_total``. The numba JIT
    companion (:func:`_leaky_pass_numba`) takes the same arguments
    and produces bit-identical output up to float-summation order.
    """
    rows, cols = r.shape
    quarter_pi = np.pi / 4.0
    residual_total = 0.0

    for k in range(order.size):
        idx = int(order[k])
        i = int(finite_rows[idx])
        j = int(finite_cols[idx])
        r_ij = float(r[i, j])
        f_ij = float(f_drain[i, j])
        leak_ij = (1.0 - f_ij) * r_ij
        fwd_ij = f_ij * r_ij
        leak[i, j] = leak_ij
        forward[i, j] = fwd_ij

        if not has_flow[i, j]:
            residual_total += fwd_ij
            continue

        f = int(best_facet[i, j])
        rval = float(best_r[i, j])
        prop_diag = rval / quarter_pi
        prop_card = 1.0 - prop_diag

        dr1 = int(_FACET_OFFSETS_CARD[f, 0])
        dc1 = int(_FACET_OFFSETS_CARD[f, 1])
        dr2 = int(_FACET_OFFSETS_DIAG[f, 0])
        dc2 = int(_FACET_OFFSETS_DIAG[f, 1])

        if prop_card > 0.0:
            nr = i + dr1
            nc = j + dc1
            if 0 <= nr < rows and 0 <= nc < cols and not nan_mask[nr, nc]:
                r[nr, nc] += prop_card * fwd_ij
            else:
                residual_total += prop_card * fwd_ij
        if prop_diag > 0.0:
            nr = i + dr2
            nc = j + dc2
            if 0 <= nr < rows and 0 <= nc < cols and not nan_mask[nr, nc]:
                r[nr, nc] += prop_diag * fwd_ij
            else:
                residual_total += prop_diag * fwd_ij

    return residual_total


# numba JIT companion. Decorated only if numba is importable so the
# module loads cleanly without it; the numpy reference path serves as
# the fallback. The function body is identical to ``_leaky_pass_python``
# in shape so the two are easy to diff.
if _HAS_NUMBA:
    import numba  # type: ignore[import-untyped]

    @numba.njit(cache=True)  # type: ignore[untyped-decorator]
    def _leaky_pass_numba(
        finite_rows: np.ndarray,
        finite_cols: np.ndarray,
        order: np.ndarray,
        r: np.ndarray,
        f_drain: np.ndarray,
        has_flow: np.ndarray,
        nan_mask: np.ndarray,
        best_facet: np.ndarray,
        best_r: np.ndarray,
        facet_card: np.ndarray,
        facet_diag: np.ndarray,
        leak: np.ndarray,
        forward: np.ndarray,
    ) -> float:
        rows = r.shape[0]
        cols = r.shape[1]
        quarter_pi = np.pi / 4.0
        residual_total = 0.0
        n = order.size
        for k in range(n):
            idx = order[k]
            i = finite_rows[idx]
            j = finite_cols[idx]
            r_ij = r[i, j]
            f_ij = f_drain[i, j]
            leak_ij = (1.0 - f_ij) * r_ij
            fwd_ij = f_ij * r_ij
            leak[i, j] = leak_ij
            forward[i, j] = fwd_ij

            if not has_flow[i, j]:
                residual_total += fwd_ij
                continue

            f = best_facet[i, j]
            rval = best_r[i, j]
            prop_diag = rval / quarter_pi
            prop_card = 1.0 - prop_diag

            dr1 = facet_card[f, 0]
            dc1 = facet_card[f, 1]
            dr2 = facet_diag[f, 0]
            dc2 = facet_diag[f, 1]

            if prop_card > 0.0:
                nr = i + dr1
                nc = j + dc1
                if 0 <= nr < rows and 0 <= nc < cols and not nan_mask[nr, nc]:
                    r[nr, nc] += prop_card * fwd_ij
                else:
                    residual_total += prop_card * fwd_ij
            if prop_diag > 0.0:
                nr = i + dr2
                nc = j + dc2
                if 0 <= nr < rows and 0 <= nc < cols and not nan_mask[nr, nc]:
                    r[nr, nc] += prop_diag * fwd_ij
                else:
                    residual_total += prop_diag * fwd_ij

        return residual_total

else:
    _leaky_pass_numba = None


def leaky_weighted_accumulation(
    dem: np.ndarray,
    cell_size_m: float,
    *,
    f_drain: np.ndarray,
    q_storage: np.ndarray,
    weights: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> LeakyResult:
    """Leaky-bucket D-infinity weighted flow accumulation.

    Each finite cell starts with its own contribution (``1.0`` by
    default, or the corresponding entry of ``weights``) and accumulates
    contributions from upstream cells under D-infinity routing — the
    same as :func:`thermal_model.physics.flow_accumulation` so far.
    The departure: at each cell ``c`` the through-flow ``r[c]`` is
    split into ``leak[c] = (1 - f_drain[c]) * r[c]`` (consumed locally
    as trigger output) and ``forward[c] = f_drain[c] * r[c]`` (passed
    to the two D-infinity neighbours). At sinks (no positive downhill
    direction) ``forward[c]`` is added to ``residual_at_sinks_total``
    rather than routed.

    The cycle period at each cell is
    ``cycle_period[c] = q_storage[c] / leak[c]``, diverging where the
    cell does not leak.

    Parameters
    ----------
    dem : np.ndarray
        Pit-filled inverted (and ideally flat-resolved) DEM in metres.
        NaN nodata.
    cell_size_m : float
        Square cell size in metres.
    f_drain : np.ndarray
        Per-cell drain fraction, ``[0, 1]`` at finite-DEM cells, NaN
        permitted at NaN-DEM cells. See :func:`f_drain_field` for a
        typical construction.
    q_storage : np.ndarray
        Per-cell storage capacity (J/m² when ``weights`` is W/m²),
        non-negative at finite-DEM cells. See :func:`q_storage_field`.
    weights : np.ndarray, optional
        Per-cell injected through-flow rate. Same shape as ``dem``,
        same finiteness contract as
        :func:`thermal_model.physics.flow_accumulation`. If omitted,
        every finite cell contributes ``1.0`` (so the limit
        ``f_drain ≡ 1`` reduces exactly to the unweighted upstream
        cell count).
    use_numba : bool, optional
        ``True`` to require the numba JIT backend (raises
        ``ImportError`` if unavailable), ``False`` to force the
        pure-numpy reference path, ``None`` (default) to use numba
        when importable and the reference path otherwise. Mirrors the
        ``use_richdem`` pattern in
        :func:`thermal_model.physics.flow_accumulation`.

    Returns
    -------
    LeakyResult
        Frozen dataclass with ``leak``, ``forward``, ``cycle_period``,
        ``residual_at_sinks_total``.

    Notes
    -----
    Energy conservation invariant (within float tolerance, in the
    units of ``weights``)::

        nansum(leak) + residual_at_sinks_total ≡ nansum(weights_or_unit)

    No multiplication by ``cell_area`` is performed: the kernel routes
    weights as-is. Callers that want absolute power (W) should pass
    ``weights = heating_wm2 * cell_size_m**2``.

    NaN-DEM cells produce NaN in every output raster. The
    ``residual_at_sinks_total`` scalar is always finite. The
    ``forward`` raster reports the *post-leak* through-flow at each
    cell — what was passed to D-infinity neighbours — *not* the
    pre-leak accumulation.

    The numba backend is the production path (~50–200x faster on
    Mallerstang-scale rasters); the numpy reference path is the
    test oracle and is used by CI environments that lack numba.
    """
    _validate_dem(dem, cell_size_m)
    if weights is not None:
        _validate_weights(dem, weights)
    _validate_f_drain(dem, f_drain)
    _validate_q_storage(dem, q_storage)

    rows, cols = dem.shape

    # Direction selection: same code path as flow.flow_accumulation so
    # the f_drain ≡ 1 limit reduces to it cell-for-cell.
    slopes, angles = _facet_slopes(dem, cell_size_m)
    best_facet = np.argmax(slopes, axis=0)
    rr, cc = np.indices((rows, cols))
    best_slope = slopes[best_facet, rr, cc]
    best_r = angles[best_facet, rr, cc]

    nan_mask = np.isnan(dem)
    if weights is None:
        r = np.where(nan_mask, np.nan, 1.0)
    else:
        r = weights.astype(np.float64, copy=True)
        r[nan_mask] = np.nan

    leak = np.zeros_like(r)
    forward = np.zeros_like(r)
    leak[nan_mask] = np.nan
    forward[nan_mask] = np.nan
    residual_total = 0.0

    finite = ~nan_mask
    if not finite.any():
        cycle_period = np.full_like(r, np.nan)
        return LeakyResult(
            leak=leak,
            forward=forward,
            cycle_period=cycle_period,
            residual_at_sinks_total=0.0,
        )

    has_flow = (finite & (best_slope > 0)).astype(np.bool_)

    # Topological order: descending DEM elevation. On a pit-filled
    # inverted DEM every D-infinity receiver lies strictly below the
    # sender (uphill on real terrain), so this orders the DAG correctly
    # in a single pass. Iterate over *all* finite cells (not just
    # has_flow ones) because sinks still need leak / forward computed
    # — they just route to residual instead of neighbours.
    finite_rows_idx, finite_cols_idx = np.where(finite)
    finite_rows_arr = finite_rows_idx.astype(np.int64)
    finite_cols_arr = finite_cols_idx.astype(np.int64)
    elevs = dem[finite_rows_idx, finite_cols_idx]
    order = np.argsort(-elevs, kind="stable").astype(np.int64)

    if use_numba is True and not _HAS_NUMBA:
        raise ImportError(
            "use_numba=True but the 'numba' package is not importable; "
            "install it (it ships in environment.yml) or set use_numba=False."
        )
    if use_numba is None:
        use_numba = _HAS_NUMBA

    if use_numba:
        assert _leaky_pass_numba is not None  # for mypy
        residual_total = float(
            _leaky_pass_numba(
                finite_rows_arr,
                finite_cols_arr,
                order,
                r,
                f_drain.astype(np.float64, copy=False),
                has_flow,
                nan_mask,
                best_facet.astype(np.int64, copy=False),
                best_r,
                _FACET_OFFSETS_CARD,
                _FACET_OFFSETS_DIAG,
                leak,
                forward,
            )
        )
    else:
        residual_total = _leaky_pass_python(
            finite_rows_arr,
            finite_cols_arr,
            order,
            r,
            f_drain,
            has_flow,
            nan_mask,
            best_facet,
            best_r,
            leak,
            forward,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        cycle_period = np.where(leak > 0.0, q_storage / leak, np.inf)
    cycle_period[nan_mask] = np.nan

    return LeakyResult(
        leak=leak,
        forward=forward,
        cycle_period=cycle_period,
        residual_at_sinks_total=residual_total,
    )
