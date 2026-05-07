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

NaN cells are treated as drainage outlets: they pass through unchanged
and any finite cell adjacent to one is treated as if it sat on the
array boundary.

References
----------
Barnes, R., Lehman, C., & Mulla, D. (2014). Priority-flood: An optimal
depression-filling and watershed-labeling algorithm for digital
elevation models. Computers & Geosciences, 62, 117-127.
"""

from __future__ import annotations

import heapq
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


def fill_pits(dem: np.ndarray, *, epsilon: float = 0.0) -> np.ndarray:
    """Fill closed depressions in ``dem`` via priority-flood.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array in metres with ``np.nan`` as nodata.
    epsilon : float, default 0.0
        Per-step elevation bump applied along the flood front. Must be
        non-negative. ``epsilon = 0`` produces a plain fill in which
        flat regions stay flat. A small positive value (e.g. ``1e-3``)
        forces strict monotonicity, which is occasionally useful as
        preconditioning for D8/D-infinity flow routing on flat fills.

    Returns
    -------
    np.ndarray
        Float64 copy of ``dem`` with all closed depressions raised.
        NaN cells pass through unchanged. Output is always >= input
        cell-wise on finite cells.
    """
    if dem.ndim != 2:
        raise ValueError(f"DEM must be 2-D, got shape {dem.shape}")
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    rows, cols = dem.shape
    if rows < 2 or cols < 2:
        raise ValueError(f"DEM must be at least 2x2, got {dem.shape}")

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
