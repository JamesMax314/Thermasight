"""D-infinity flow directions and accumulation (Tarboton 1997).

The convergence engine of this project is flow accumulation on the
*inverted* DEM (``max(z) - z``); see ``CLAUDE.md`` §2 and
``docs/MODEL.md``. This module is domain-agnostic: it accumulates
whatever surface it is given. Pit-filling and inversion are the
caller's responsibility — see :func:`thermal_model.physics.fill_pits`.

Tarboton's D-infinity assigns each cell a continuous downslope
direction by selecting the steepest of eight triangular *facets*. Each
facet is bounded by one cardinal and one diagonal neighbour. The
chosen facet's downhill direction is an angle, and flow is split
between the two bounding neighbours in proportion to how the angle
sits within the facet. This avoids the cardinal-axis striping artefacts
of D8.

Accumulation is computed by sorting finite, draining cells in
descending elevation and propagating each cell's running total to its
one or two downstream neighbours in a single pass. On a pit-filled DEM
every cell's chosen receivers strictly lie below it, so descending
elevation is a valid topological order.

Two backends are available behind :func:`flow_accumulation`:
``richdem`` (faster on large rasters) and a pure-numpy fallback. The
fallback is the reference implementation; the richdem path is used
automatically when ``richdem`` is importable.

References
----------
Tarboton, D.G. (1997). A new method for the determination of flow
directions and upslope areas in grid digital elevation models. Water
Resources Research, 33(2), 309-319.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

# Eight triangular facets per Tarboton (1997), Table 1, expressed in
# (drow, dcol) raster offsets. Row index increases southward.
#
# Each facet is (cardinal_offset, diagonal_offset, base_angle, sign).
# The continuous global angle of flow within facet f is::
#
#     theta = base_angle + sign * r,   r in [0, pi/4]
#
# where r is the local angle from the cardinal toward the diagonal.
# Angles are math-convention: theta=0 points east, increases
# counter-clockwise. The eight facets together tile [0, 2*pi).
_FACETS: tuple[tuple[tuple[int, int], tuple[int, int], float, int], ...] = (
    ((0, 1), (-1, 1), 0.0, +1),  # facet 1: E,  NE
    ((-1, 0), (-1, 1), np.pi / 2, -1),  # facet 2: N,  NE  (mirrored)
    ((-1, 0), (-1, -1), np.pi / 2, +1),  # facet 3: N,  NW
    ((0, -1), (-1, -1), np.pi, -1),  # facet 4: W,  NW  (mirrored)
    ((0, -1), (1, -1), np.pi, +1),  # facet 5: W,  SW
    ((1, 0), (1, -1), 3 * np.pi / 2, -1),  # facet 6: S,  SW  (mirrored)
    ((1, 0), (1, 1), 3 * np.pi / 2, +1),  # facet 7: S,  SE
    ((0, 1), (1, 1), 2 * np.pi, -1),  # facet 8: E,  SE  (mirrored)
)


def _validate_dem(dem: np.ndarray, cell_size_m: float) -> None:
    if dem.ndim != 2:
        raise ValueError(f"DEM must be 2-D, got shape {dem.shape}")
    if cell_size_m <= 0:
        raise ValueError(f"cell_size_m must be positive, got {cell_size_m}")
    if dem.shape[0] < 2 or dem.shape[1] < 2:
        raise ValueError(f"DEM must be at least 2x2, got {dem.shape}")


def _facet_slopes(dem: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-facet slope and within-facet angle for every cell.

    Returns
    -------
    slopes : np.ndarray
        Shape ``(8, rows, cols)``. Steepest descent within each facet,
        in metres-per-metre (a tangent). Facets that point uphill or
        whose neighbours are NaN return ``-np.inf``.
    angles : np.ndarray
        Shape ``(8, rows, cols)``. Within-facet angle ``r`` clipped to
        ``[0, pi/4]``.
    """
    rows, cols = dem.shape
    z_padded = np.full((rows + 2, cols + 2), np.nan, dtype=np.float64)
    z_padded[1:-1, 1:-1] = dem
    z0 = z_padded[1:-1, 1:-1]

    inv_d = 1.0 / float(cell_size_m)
    inv_diag = 1.0 / (float(cell_size_m) * np.sqrt(2.0))
    quarter_pi = np.pi / 4.0

    slopes = np.full((8, rows, cols), -np.inf, dtype=np.float64)
    angles = np.zeros((8, rows, cols), dtype=np.float64)

    for f, ((dr1, dc1), (dr2, dc2), _base, _sign) in enumerate(_FACETS):
        z1 = z_padded[1 + dr1 : 1 + dr1 + rows, 1 + dc1 : 1 + dc1 + cols]
        z2 = z_padded[1 + dr2 : 1 + dr2 + rows, 1 + dc2 : 1 + dc2 + cols]

        s1 = (z0 - z1) * inv_d  # slope along the cardinal edge
        s2 = (z1 - z2) * inv_d  # slope orthogonal to it, within the facet
        with np.errstate(invalid="ignore"):
            r_raw = np.arctan2(s2, s1)
            s_in = np.hypot(s1, s2)

        # Within-facet rule (Tarboton 1997, eq. 1-3):
        #   r < 0      -> snap to cardinal edge:  r = 0,    s = s1
        #   r > pi/4   -> snap to diagonal edge:  r = pi/4, s = (z0 - z2) / (d*sqrt(2))
        #   otherwise  -> r unchanged,            s = hypot(s1, s2)
        r_clipped = np.clip(r_raw, 0.0, quarter_pi)
        s_diag = (z0 - z2) * inv_diag
        s = np.where(r_raw < 0, s1, np.where(r_raw > quarter_pi, s_diag, s_in))

        # NaN slopes (edge cells, NaN neighbours) -> -inf so argmax skips.
        s = np.where(np.isnan(s), -np.inf, s)
        slopes[f] = s
        angles[f] = r_clipped

    return slopes, angles


def dinf_flow_directions(
    dem: np.ndarray, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """D-infinity downslope direction and slope magnitude per cell.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array in metres with NaN nodata. The DEM should
        already be pit-filled if the caller wants every interior cell
        to drain; see :func:`thermal_model.physics.fill_pits`.
    cell_size_m : float
        Square cell size in metres.

    Returns
    -------
    angle : np.ndarray
        Downslope direction in radians on ``[0, 2*pi)``, math
        convention (0 = east, pi/2 = north, pi = west, 3*pi/2 = south).
        ``NaN`` on cells with no positive downhill slope (true flats,
        local minima, NaN cells, and cells whose 3x3 window contains
        only NaN neighbours).
    slope : np.ndarray
        Magnitude of the steepest facet slope, in metres-per-metre.
        ``NaN`` on the same cells as ``angle``.

    Notes
    -----
    The angle convention is math-style (CCW from east), matching
    Tarboton (1997). It differs from the compass-bearing convention
    used by :func:`thermal_model.terrain.aspect`, which is intended for
    human-readable aspect maps. Flow code consumes this angle
    internally and the choice is mostly a documentation convenience.
    """
    _validate_dem(dem, cell_size_m)

    slopes, angles = _facet_slopes(dem, cell_size_m)
    best_facet = np.argmax(slopes, axis=0)
    rows, cols = dem.shape
    rr, cc = np.indices((rows, cols))
    best_slope = slopes[best_facet, rr, cc]
    best_r = angles[best_facet, rr, cc]

    bases = np.array([f[2] for f in _FACETS], dtype=np.float64)
    signs = np.array([f[3] for f in _FACETS], dtype=np.float64)
    theta = bases[best_facet] + signs[best_facet] * best_r
    theta = np.mod(theta, 2.0 * np.pi)

    has_flow = best_slope > 0
    angle_out = np.where(has_flow, theta, np.nan)
    slope_out = np.where(has_flow, best_slope, np.nan)
    return angle_out, slope_out


def _flow_accumulation_numpy(
    dem: np.ndarray,
    cell_size_m: float,
    weights: np.ndarray | None,
) -> np.ndarray:
    """Reference numpy implementation of D-infinity flow accumulation."""
    rows, cols = dem.shape
    slopes, angles = _facet_slopes(dem, cell_size_m)
    best_facet = np.argmax(slopes, axis=0)
    rr, cc = np.indices((rows, cols))
    best_slope = slopes[best_facet, rr, cc]
    best_r = angles[best_facet, rr, cc]

    nan_mask = np.isnan(dem)
    if weights is None:
        acc = np.where(nan_mask, np.nan, 1.0)
    else:
        if weights.shape != dem.shape:
            raise ValueError(f"weights shape {weights.shape} != dem shape {dem.shape}")
        acc = weights.astype(np.float64, copy=True)
        acc[nan_mask] = np.nan

    # Cells that distribute: finite, with at least one downhill facet.
    has_flow = (~nan_mask) & (best_slope > 0)
    if not has_flow.any():
        return acc

    quarter_pi = np.pi / 4.0
    facet_offsets: Sequence[tuple[tuple[int, int], tuple[int, int]]] = tuple(
        (f[0], f[1]) for f in _FACETS
    )

    # Topological order: descending elevation. On a pit-filled DEM
    # every receiver lies strictly below the sender, so this orders
    # the DAG correctly in a single pass.
    flow_rows, flow_cols = np.where(has_flow)
    elevs = dem[flow_rows, flow_cols]
    order = np.argsort(-elevs, kind="stable")

    for k in order:
        r = int(flow_rows[k])
        c = int(flow_cols[k])
        f = int(best_facet[r, c])
        rval = float(best_r[r, c])

        prop_diag = rval / quarter_pi
        prop_card = 1.0 - prop_diag
        val = float(acc[r, c])

        (dr1, dc1), (dr2, dc2) = facet_offsets[f]
        if prop_card > 0.0:
            nr, nc = r + dr1, c + dc1
            if 0 <= nr < rows and 0 <= nc < cols and not nan_mask[nr, nc]:
                acc[nr, nc] += prop_card * val
        if prop_diag > 0.0:
            nr, nc = r + dr2, c + dc2
            if 0 <= nr < rows and 0 <= nc < cols and not nan_mask[nr, nc]:
                acc[nr, nc] += prop_diag * val

    return acc


def _flow_accumulation_richdem(
    dem: np.ndarray,
    cell_size_m: float,
    weights: np.ndarray | None,
) -> np.ndarray:
    """richdem-backed D-infinity flow accumulation."""
    import richdem as rd

    nan_mask = np.isnan(dem)
    sentinel = -1.0e30
    dem_clean = np.where(nan_mask, sentinel, dem).astype(np.float64)
    rdem = rd.rdarray(dem_clean, no_data=sentinel)
    rdem.geotransform = (0.0, float(cell_size_m), 0.0, 0.0, 0.0, -float(cell_size_m))

    if weights is None:
        acc = rd.FlowAccumulation(rdem, method="Dinf")
    else:
        if weights.shape != dem.shape:
            raise ValueError(f"weights shape {weights.shape} != dem shape {dem.shape}")
        rweights = rd.rdarray(
            np.where(nan_mask, 0.0, weights).astype(np.float64),
            no_data=sentinel,
        )
        rweights.geotransform = rdem.geotransform
        acc = rd.FlowAccumulation(rdem, method="Dinf", weights=rweights)

    out = np.asarray(acc, dtype=np.float64)
    out[nan_mask] = np.nan
    return out


def _have_richdem() -> bool:
    return importlib.util.find_spec("richdem") is not None


def flow_accumulation(
    dem: np.ndarray,
    cell_size_m: float,
    *,
    weights: np.ndarray | None = None,
    use_richdem: bool | None = None,
) -> np.ndarray:
    """D-infinity flow accumulation per cell.

    Each finite cell starts with its own contribution (``1.0`` by
    default, or the corresponding entry of ``weights``) and accumulates
    contributions from upstream cells. Flow is routed by D-infinity:
    each cell sends its accumulated total to one or two downhill
    neighbours in proportion to its downslope angle within the chosen
    facet.

    On a pit-filled DEM, the result is a non-negative array whose value
    at each cell equals the sum of weights upstream of it (including
    itself) under the D-infinity routing.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array in metres with NaN nodata. The caller is
        responsible for any preconditioning (pit-fill, inversion).
    cell_size_m : float
        Square cell size in metres. Affects only the slope magnitudes
        used for direction selection; the accumulation itself is
        cell-count-based unless ``weights`` are supplied.
    weights : np.ndarray, optional
        Per-cell starting contribution. Same shape as ``dem``. If
        omitted, every finite cell contributes ``1.0`` (i.e. the result
        is upstream cell count, including self).
    use_richdem : bool, optional
        ``True`` to require the richdem backend (raises ``ImportError``
        if unavailable), ``False`` to force the numpy fallback,
        ``None`` (default) to use richdem when importable and the
        fallback otherwise.

    Returns
    -------
    np.ndarray
        Float64 accumulation array, same shape as ``dem``. NaN at NaN
        input cells.

    Notes
    -----
    The numpy fallback is the reference implementation. The richdem
    backend may differ on pixel-scale details (flat resolution, edge
    handling); for property-level checks the two agree to within a few
    percent on small fixtures. Property-based tests pin the fallback;
    the richdem path has a smoke test.

    Cells with no positive downhill slope (true flats, local minima,
    boundary outlets) keep their starting contribution but route none
    of it onward. To avoid trapping flow at internal flats, pre-fill
    with :func:`thermal_model.physics.fill_pits` using a small positive
    epsilon.
    """
    _validate_dem(dem, cell_size_m)

    if use_richdem is True and not _have_richdem():
        raise ImportError(
            "use_richdem=True but the 'richdem' package is not importable; "
            "install it (it ships in environment.yml) or set use_richdem=False."
        )
    if use_richdem is None:
        use_richdem = _have_richdem()

    if use_richdem:
        return _flow_accumulation_richdem(dem, cell_size_m, weights)
    return _flow_accumulation_numpy(dem, cell_size_m, weights)
