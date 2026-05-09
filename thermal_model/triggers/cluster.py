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
        Mean of the strength raster over the cluster's cells. The
        unit is whatever the raster passed to :func:`cluster_triggers`
        carried — ``[0, 1]`` rank-norm for ``trigger_potential``,
        absolute W/m² for ``leak``.
    n_cells : int
        Number of cells in the cluster.
    mean_cycle_period_s : float or None
        Mean buoyancy-cycle period over the cluster's cells, in
        seconds, when a ``cycle_period_s`` raster is supplied to
        :func:`cluster_triggers`. ``None`` when not supplied. Cells
        with infinite cycle period are excluded from the mean (only
        cells that actually leak contribute), so this value is finite
        for any cluster that survives the strength threshold.
    """

    row: float
    col: float
    mean_strength: float
    n_cells: int
    mean_cycle_period_s: float | None = None


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
    cycle_period_s: np.ndarray | None = None,
) -> list[TriggerPoint]:
    """Cluster a trigger raster into discrete trigger points.

    Parameters
    ----------
    trigger_potential : np.ndarray
        2-D float raster representing per-cell trigger strength. From
        :func:`thermal_model.physics.run_model` this is either
        ``RunResult.trigger_potential`` (rank-normalised, ``[0, 1]``)
        or ``RunResult.leak`` (absolute W/m²) — both are
        order-preserving so the same cells survive the percentile
        threshold either way; only the ``mean_strength`` units differ.
        NaN nodata.
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
    cycle_period_s : np.ndarray, optional
        Per-cell buoyancy-cycle period (s) from
        ``RunResult.cycle_period_s``. When supplied, each
        :class:`TriggerPoint` carries the mean cycle period over its
        cluster (excluding cells with infinite period — i.e. only
        cells that actually leak contribute). When omitted, the
        ``mean_cycle_period_s`` field is left as ``None``.

    Returns
    -------
    list of TriggerPoint
        Surviving clusters, sorted by ``mean_strength`` descending.
        Empty list if no cluster meets the size threshold.

    Raises
    ------
    ValueError
        If ``trigger_potential`` is not 2-D, the quantile is out of
        range, ``min_cluster_cells < 1``, ``connectivity`` is not
        4 or 8, or ``cycle_period_s`` is given with a mismatched
        shape.
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
    if cycle_period_s is not None and cycle_period_s.shape != trigger_potential.shape:
        raise ValueError(
            f"cycle_period_s shape {cycle_period_s.shape} != "
            f"trigger_potential shape {trigger_potential.shape}"
        )

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

    # Optional cycle-period averaging. Only include finite cells
    # within each labelled cluster; +inf cells (leak == 0) would
    # contaminate the mean even though they cannot be in the mask
    # (mask requires arr > threshold > 0, and a cell with leak == 0
    # also has its trigger raster value at 0). In practice the mask
    # already excludes those cells; the np.where here is defensive
    # and handles the case where someone passes in a strength raster
    # decoupled from cycle_period_s (e.g. a normalised version).
    cycle_means_arr: np.ndarray | None = None
    if cycle_period_s is not None:
        cps = np.asarray(cycle_period_s, dtype=np.float64)
        finite_cycle = np.isfinite(cps) & (labels_arr > 0)
        cycle_for_mean = np.where(finite_cycle, cps, 0.0)
        cycle_count = ndimage.sum(
            finite_cycle.astype(np.int64), labels_arr, index=component_ids
        )
        cycle_sum = ndimage.sum(cycle_for_mean, labels_arr, index=component_ids)
        cycle_count_arr = np.atleast_1d(np.asarray(cycle_count, dtype=np.int64))
        cycle_sum_arr = np.atleast_1d(np.asarray(cycle_sum, dtype=np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            cycle_means_arr = np.where(
                cycle_count_arr > 0,
                cycle_sum_arr / cycle_count_arr.astype(np.float64),
                np.nan,
            )

    points: list[TriggerPoint] = []
    for k, (size, mean_t, centre) in enumerate(
        zip(sizes_arr.tolist(), means_arr.tolist(), centroids_seq, strict=True)
    ):
        if size < min_cluster_cells:
            continue
        row, col = centre
        mean_cycle: float | None = None
        if cycle_means_arr is not None:
            value = float(cycle_means_arr[k])
            mean_cycle = value if np.isfinite(value) else None
        points.append(
            TriggerPoint(
                row=float(row),
                col=float(col),
                mean_strength=float(mean_t),
                n_cells=int(size),
                mean_cycle_period_s=mean_cycle,
            )
        )

    points.sort(key=lambda p: p.mean_strength, reverse=True)
    return points
