"""Hydrological preconditioning: pit-fill and flat resolution.

Two operations live here, in the order a flow-routing pipeline
applies them:

1. :func:`fill_pits` — raises every closed depression to its spill
   elevation so that every finite cell has a non-increasing path to
   the array boundary or a NaN cell. Algorithm is the priority-flood
   of Barnes, Lehman & Mulla (2014).

2. :func:`resolve_flats` — perturbs cells with no strictly-lower
   neighbour by a tiny gradient toward the flat region's outflow
   point. Removes the parallel-streak artefact that pure pit-fill
   leaves behind on flat regions (where the BFS frontier produces a
   chamfer-distance gradient, not a physically meaningful one).

Both are fed the *inverted* DEM (``max(z) - z``) before flow
accumulation in this project's convergence pipeline.

Both have a richdem C++ backend (auto-selected when importable, ~100x
faster for large rasters) and a pure-numpy fallback (the reference
implementation, used for testability and offline runs). The
``use_richdem`` kwarg overrides backend selection.

NaN cells are treated as drainage outlets: they pass through
unchanged and any finite cell adjacent to one is treated as if it sat
on the array boundary.

References
----------
Barnes, R., Lehman, C., & Mulla, D. (2014). Priority-flood: An optimal
depression-filling and watershed-labeling algorithm for digital
elevation models. Computers & Geosciences, 62, 117-127.

Garbrecht, J. & Martz, L.W. (1997). The assignment of drainage
direction over flat surfaces in raster digital elevation models.
Journal of Hydrology, 193(1-4), 204-213.
"""

from __future__ import annotations

import heapq
import importlib.util
import itertools

import numpy as np
from scipy import ndimage

_NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _have_richdem() -> bool:
    return importlib.util.find_spec("richdem") is not None


def _fill_pits_numpy(dem: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Reference numpy implementation of priority-flood pit-fill."""
    rows, cols = dem.shape
    filled = dem.astype(np.float64, copy=True)
    nan_mask = np.isnan(filled)
    if nan_mask.all():
        return filled

    visited = nan_mask.copy()

    seed = np.zeros(dem.shape, dtype=bool)
    seed[0, :] = True
    seed[-1, :] = True
    seed[:, 0] = True
    seed[:, -1] = True
    if nan_mask.any():
        nan_dilated = np.asarray(
            ndimage.binary_dilation(nan_mask, structure=np.ones((3, 3), dtype=bool)),
            dtype=bool,
        )
        seed |= nan_dilated
    seed &= ~nan_mask

    counter = itertools.count()
    heap: list[tuple[float, int, int, int]] = []
    seed_rows, seed_cols = np.where(seed)
    for r, c in zip(seed_rows.tolist(), seed_cols.tolist(), strict=True):
        heapq.heappush(heap, (float(filled[r, c]), next(counter), r, c))
        visited[r, c] = True

    while heap:
        z, _, r, c = heapq.heappop(heap)
        for dr, dc in _NEIGHBOUR_OFFSETS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if visited[nr, nc]:
                continue
            new_z = max(float(filled[nr, nc]), z + epsilon)
            filled[nr, nc] = new_z
            heapq.heappush(heap, (new_z, next(counter), nr, nc))
            visited[nr, nc] = True

    return filled


def _fill_pits_richdem(dem: np.ndarray, *, epsilon: float) -> np.ndarray:
    """richdem-backed priority-flood pit-fill.

    richdem expects a sentinel-marked nodata array, so we swap NaN for
    a sentinel safely below the data range, run the fill, then
    restore NaN. richdem's ``epsilon`` flag is boolean — when on, it
    bumps each step by the smallest representable float of the DEM
    dtype. We treat any positive ``epsilon`` here as "ask for the
    monotonic fill"; the exact bump magnitude is not user-configurable
    on this path.
    """
    import richdem as rd

    nan_mask = np.isnan(dem)
    arr = dem.astype(np.float64, copy=True)
    if nan_mask.all():
        return arr

    finite = arr[~nan_mask]
    sentinel = float(finite.min()) - 1.0e6
    arr_with_sentinel = np.where(nan_mask, sentinel, arr)

    rdem = rd.rdarray(arr_with_sentinel, no_data=sentinel)
    # FillDepressions ignores the geotransform but rdarray expects one.
    rdem.geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

    filled_rd = rd.FillDepressions(
        rdem, epsilon=epsilon > 0, in_place=False, topology="D8"
    )
    filled = np.asarray(filled_rd, dtype=np.float64)
    filled[nan_mask] = np.nan
    return filled


def fill_pits(
    dem: np.ndarray,
    *,
    epsilon: float = 0.0,
    use_richdem: bool | None = None,
) -> np.ndarray:
    """Fill closed depressions in ``dem`` via priority-flood.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array in metres with ``np.nan`` as nodata.
    epsilon : float, default 0.0
        Per-step elevation bump applied along the flood front. Must be
        non-negative. ``epsilon = 0`` produces a plain fill in which
        flat regions stay flat. A small positive value (e.g. ``1e-3``)
        forces strict monotonicity, which is useful as preconditioning
        for D-infinity flow routing on flat fills.

        On the richdem backend, ``epsilon`` is interpreted as a
        boolean: any positive value triggers a monotonic fill that
        bumps by the smallest representable float of the DEM dtype
        (a few orders of magnitude smaller than the typical numpy
        default). For convergence accumulation either bump suffices.
    use_richdem : bool, optional
        ``True`` to require the richdem backend (raises
        :class:`ImportError` if unavailable), ``False`` to force the
        numpy fallback, ``None`` (default) to use richdem when
        importable and the fallback otherwise.

    Returns
    -------
    np.ndarray
        Float64 copy of ``dem`` with all closed depressions raised.
        NaN cells pass through unchanged. Output is always ``>=``
        input cell-wise on finite cells.

    Notes
    -----
    The numpy fallback is the reference implementation. The richdem
    path is roughly 100x faster on large rasters; pixel-scale
    differences may exist on flat regions because of the smaller
    epsilon bump richdem applies, but the high-flow-accumulation
    skeleton is unchanged.
    """
    if dem.ndim != 2:
        raise ValueError(f"DEM must be 2-D, got shape {dem.shape}")
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if dem.shape[0] < 2 or dem.shape[1] < 2:
        raise ValueError(f"DEM must be at least 2x2, got {dem.shape}")

    if use_richdem is True and not _have_richdem():
        raise ImportError(
            "use_richdem=True but the 'richdem' package is not importable; "
            "install it (it ships in environment.yml) or set use_richdem=False."
        )
    if use_richdem is None:
        use_richdem = _have_richdem()

    if use_richdem:
        return _fill_pits_richdem(dem, epsilon=epsilon)
    return _fill_pits_numpy(dem, epsilon=epsilon)


def _resolve_flats_richdem(dem: np.ndarray) -> np.ndarray:
    """richdem-backed Garbrecht-Martz flat resolution."""
    import richdem as rd

    nan_mask = np.isnan(dem)
    arr = dem.astype(np.float64, copy=True)
    if nan_mask.all():
        return arr

    finite = arr[~nan_mask]
    sentinel = float(finite.min()) - 1.0e6
    arr_with_sentinel = np.where(nan_mask, sentinel, arr)

    rdem = rd.rdarray(arr_with_sentinel, no_data=sentinel)
    rdem.geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

    resolved_rd = rd.ResolveFlats(rdem, in_place=False)
    resolved = np.asarray(resolved_rd, dtype=np.float64)
    resolved[nan_mask] = np.nan
    return resolved


def _resolve_flats_numpy(dem: np.ndarray, *, amplitude: float, seed: int) -> np.ndarray:
    """Stochastic flat-resolution fallback.

    Identifies cells with no strictly-lower 8-neighbour (true flats
    on a pit-filled DEM, plus cells with all-equal neighbours) and
    adds a tiny Gaussian perturbation. This breaks the BFS-derived
    chamfer-distance symmetry without imposing any geometric pattern
    of its own. Strictly less principled than Garbrecht-Martz; use
    the richdem backend when available.
    """
    rows, cols = dem.shape
    arr = dem.astype(np.float64, copy=True)
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return arr

    padded = np.full((rows + 2, cols + 2), np.nan, dtype=np.float64)
    padded[1:-1, 1:-1] = arr
    centre = padded[1:-1, 1:-1]

    has_lower = np.zeros(arr.shape, dtype=bool)
    for dr, dc in _NEIGHBOUR_OFFSETS:
        nbr = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        with np.errstate(invalid="ignore"):
            this_drains = (nbr < centre) | np.isnan(nbr)
        has_lower |= this_drains

    flat_mask = (~has_lower) & (~nan_mask)
    if not flat_mask.any():
        return arr

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, amplitude, size=arr.shape)
    arr = np.where(flat_mask, arr + noise, arr)
    return arr


def resolve_flats(
    dem: np.ndarray,
    *,
    use_richdem: bool | None = None,
    fallback_amplitude: float = 1.0e-6,
    fallback_seed: int = 0,
) -> np.ndarray:
    """Assign smooth gradients on flat regions of a pit-filled DEM.

    Pit-fill leaves cells inside formerly-closed depressions exactly
    flat (after the spill elevation has been raised). D-infinity flow
    routing on those flats inherits whatever fine-scale tie-breaking
    the pit-fill happened to use — typically a chamfer-distance
    field aligned with the BFS frontier — and the resulting
    accumulation map shows characteristic parallel streaks running
    perpendicular to ridges that abut flat regions. This function
    replaces that field with a physically defensible gradient.

    On the richdem backend this is the Garbrecht-Martz (1997)
    two-component gradient — *away from* higher terrain bordering the
    flat plus *toward* the flat's outflow — combined and normalised
    so each flat cell drains toward the actual outlet.

    The numpy fallback is a stochastic substitute: tiny Gaussian
    perturbation on cells with no strictly-lower 8-neighbour. It
    breaks BFS symmetry and is good enough for testability, but it
    does not produce the physically meaningful direction field that
    Garbrecht-Martz does.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array, ideally already pit-filled. NaN nodata.
    use_richdem : bool, optional
        ``True`` to require richdem (raises :class:`ImportError` if
        unavailable), ``False`` to force the numpy fallback, ``None``
        (default) to use richdem when importable.
    fallback_amplitude : float, default ``1e-6``
        Standard deviation of the Gaussian perturbation applied to
        flat cells in the numpy fallback. Should be far below the
        smallest real terrain feature you care about (1 µm at default
        is ~10 orders of magnitude below DEM noise).
    fallback_seed : int, default ``0``
        RNG seed for the numpy fallback so callers get deterministic
        output.

    Returns
    -------
    np.ndarray
        Float64 copy of ``dem`` with flat regions perturbed.
    """
    if dem.ndim != 2:
        raise ValueError(f"DEM must be 2-D, got shape {dem.shape}")
    if dem.shape[0] < 2 or dem.shape[1] < 2:
        raise ValueError(f"DEM must be at least 2x2, got {dem.shape}")
    if fallback_amplitude < 0:
        raise ValueError(
            f"fallback_amplitude must be non-negative, got {fallback_amplitude}"
        )

    if use_richdem is True and not _have_richdem():
        raise ImportError(
            "use_richdem=True but the 'richdem' package is not importable; "
            "install it (it ships in environment.yml) or set use_richdem=False."
        )
    if use_richdem is None:
        use_richdem = _have_richdem()

    if use_richdem:
        return _resolve_flats_richdem(dem)
    return _resolve_flats_numpy(dem, amplitude=fallback_amplitude, seed=fallback_seed)
