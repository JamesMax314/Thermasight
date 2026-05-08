"""Connected-component clustering of the trigger-potential raster.

The trigger-potential raster from :func:`thermal_model.physics.run_model`
is a continuous field on ``[0, 1]``. To drop it into XCTrack / SeeYou /
Google Earth we need discrete points. The recipe (``docs/MODEL.md`` §8):

1. Threshold at a high percentile of strictly-positive ``T`` (default
   95th).
2. Label connected components on the binary mask
   (:func:`scipy.ndimage.label`, 8-connectivity by default).
3. Drop components below ``min_cluster_cells`` (default 3) as noise.
4. For each surviving component, take the centroid (in raster
   coordinates) and the mean ``T`` over its cells. Rank by mean ``T``
   descending.

The output is a list of :class:`TriggerPoint`. Centroids are reported
in raster (row, col) — the conversion to projected (x, y) and then to
WGS84 lives in :mod:`thermal_model.triggers.export`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class TriggerPoint:
    """One clustered trigger location in raster coordinates.

    Attributes
    ----------
    row, col : float
        Centroid position in raster index units (the centre of the
        ``(0, 0)`` cell is at ``(row=0, col=0)``). Fractional because
        the centroid is mass-weighted by the cells in the cluster.
    mean_strength : float
        Mean of the trigger-potential raster over the cluster's cells,
        on ``[0, 1]``.
    n_cells : int
        Number of cells in the cluster.
    """

    row: float
    col: float
    mean_strength: float
    n_cells: int


def _eight_connected_structure() -> np.ndarray:
    """3x3 all-ones structuring element for 8-connectivity."""
    return np.ones((3, 3), dtype=bool)


def _four_connected_structure() -> np.ndarray:
    """3x3 plus-shaped structuring element for 4-connectivity."""
    s = np.zeros((3, 3), dtype=bool)
    s[1, :] = True
    s[:, 1] = True
    return s


def cluster_triggers(
    trigger_potential: np.ndarray,
    *,
    threshold_quantile: float = 0.95,
    min_cluster_cells: int = 3,
    connectivity: int = 8,
) -> list[TriggerPoint]:
    """Cluster a trigger-potential raster into discrete trigger points.

    Parameters
    ----------
    trigger_potential : np.ndarray
        2-D float raster from :func:`thermal_model.physics.run_model`.
        Values on ``[0, 1]`` with NaN nodata.
    threshold_quantile : float, default 0.95
        Percentile of strictly-positive finite cells used as the binary
        threshold. Cells with ``T > threshold`` are candidates. Must be
        in ``(0, 1)``.
    min_cluster_cells : int, default 3
        Minimum cluster size; smaller components are discarded as
        noise. Must be ``>= 1``.
    connectivity : int, default 8
        Connectivity for component labelling: ``4`` (rook-move
        neighbours only) or ``8`` (rook + bishop). 8 is the default
        for a regular raster — DBSCAN with ``eps = cell_size`` reduces
        to 8-connected components on a regular grid.

    Returns
    -------
    list of TriggerPoint
        Surviving clusters, sorted by ``mean_strength`` descending.
        Empty list if no cluster meets the size threshold.

    Raises
    ------
    ValueError
        If ``trigger_potential`` is not 2-D, the quantile is out of
        range, ``min_cluster_cells < 1``, or ``connectivity`` is not
        4 or 8.
    """
    if trigger_potential.ndim != 2:
        raise ValueError(
            f"trigger_potential must be 2-D, got shape {trigger_potential.shape}"
        )
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError(
            f"threshold_quantile must be in (0, 1), got {threshold_quantile}"
        )
    if min_cluster_cells < 1:
        raise ValueError(f"min_cluster_cells must be >= 1, got {min_cluster_cells}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    arr = np.asarray(trigger_potential, dtype=np.float64)
    finite = np.isfinite(arr)
    positive = finite & (arr > 0.0)
    if not positive.any():
        return []

    threshold = float(np.quantile(arr[positive], threshold_quantile))
    if not np.isfinite(threshold):
        return []

    mask = finite & (arr > threshold)
    if not mask.any():
        return []

    structure = (
        _eight_connected_structure()
        if connectivity == 8
        else _four_connected_structure()
    )
    labels_arr, n_components = ndimage.label(mask, structure=structure)
    if n_components == 0:
        return []

    component_ids = np.arange(1, n_components + 1)
    sizes = ndimage.sum(mask.astype(np.int64), labels_arr, index=component_ids)
    means = ndimage.mean(arr, labels_arr, index=component_ids)
    centroids = ndimage.center_of_mass(arr, labels_arr, index=component_ids.tolist())

    sizes_arr = np.atleast_1d(np.asarray(sizes, dtype=np.int64))
    means_arr = np.atleast_1d(np.asarray(means, dtype=np.float64))
    centroids_seq = centroids if isinstance(centroids, list) else [centroids]

    points: list[TriggerPoint] = []
    for size, mean_t, centre in zip(
        sizes_arr.tolist(), means_arr.tolist(), centroids_seq, strict=True
    ):
        if size < min_cluster_cells:
            continue
        row, col = centre
        points.append(
            TriggerPoint(
                row=float(row),
                col=float(col),
                mean_strength=float(mean_t),
                n_cells=int(size),
            )
        )

    points.sort(key=lambda p: p.mean_strength, reverse=True)
    return points
