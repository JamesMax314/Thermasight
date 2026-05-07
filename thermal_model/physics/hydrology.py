"""Priority-flood depression filling for hydrological flow routing.

:func:`fill_pits` returns a copy of the input DEM with every closed
depression raised to its spill elevation, so that every finite cell
has a non-increasing path to either the array boundary or a NaN cell.
This is the standard preprocessing step before flow accumulation; on
the inverted DEM (``max(z) - z``) it ensures the inverted pits at
real-terrain summits do not trap synthetic flow.

The algorithm is the priority-flood with epsilon-fill of Barnes,
Lehman & Mulla (2014). Boundary cells and finite cells adjacent to
NaN are seeded onto a min-heap; the heap is drained in elevation
order, raising each unvisited neighbour to ``max(its own z,
parent z + epsilon)``. Time complexity is O(N log N).

Two backends are available behind :func:`fill_pits`:

* a pure-numpy reference implementation, used for testability and as
  the offline path; and
* a ``richdem`` C++ implementation that is roughly 100x faster for
  large rasters and is auto-selected when importable.

The numpy fallback is the reference; both should agree on
well-conditioned inputs to within a few floating-point ulps.

NaN cells are treated as drainage outlets: they pass through
unchanged and any finite cell adjacent to one is treated as if it sat
on the array boundary.

References
----------
Barnes, R., Lehman, C., & Mulla, D. (2014). Priority-flood: An optimal
depression-filling and watershed-labeling algorithm for digital
elevation models. Computers & Geosciences, 62, 117-127.
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
