"""Tests for thermal_model.triggers."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin

from thermal_model.triggers import TriggerPoint, cluster_triggers, write_kmz


def _three_blob_raster() -> np.ndarray:
    """40×40 trigger raster with three deliberate blobs of varying strength.

    Layout (separated so each blob is a distinct component):

    * weak background (0.4) over a 10×10 region — used to set a
      meaningful threshold for the percentile filter.
    * medium (0.7) 4×4 blob.
    * bright (0.95) 4×4 blob.
    * single-cell (0.99) noise — should be dropped by min_cluster_cells.
    """
    arr = np.zeros((40, 40), dtype=np.float64)
    arr[0:10, 0:10] = 0.4  # weak background, will be filtered out by threshold
    arr[15:19, 15:19] = 0.7  # medium, 16 cells
    arr[25:29, 25:29] = 0.95  # bright, 16 cells
    arr[35, 35] = 0.99  # single cell — dropped by min_cluster_cells
    return arr


def test_cluster_triggers_returns_blobs_ranked_by_strength() -> None:
    arr = _three_blob_raster()
    # q=0.5 of strictly-positive values (100 weak + 16 med + 16 bright + 1 noise)
    # lands inside the weak block so the threshold is 0.4. The mask `> 0.4`
    # keeps the 0.7 + 0.95 + 0.99 cells; min_cluster_cells=3 drops the
    # single 0.99 cell.
    points = cluster_triggers(arr, threshold_quantile=0.5, min_cluster_cells=3)
    assert len(points) == 2
    assert points[0].mean_strength > points[1].mean_strength
    assert points[0].n_cells == 16
    # First centroid sits roughly at the middle of (25..28, 25..28).
    assert 25.0 < points[0].row < 29.0
    assert 25.0 < points[0].col < 29.0


def test_cluster_triggers_drops_undersized_components() -> None:
    arr = _three_blob_raster()
    # min=2 still drops the lone 0.99 cell.
    points = cluster_triggers(arr, threshold_quantile=0.5, min_cluster_cells=2)
    assert all(p.n_cells >= 2 for p in points)
    points_strict = cluster_triggers(arr, threshold_quantile=0.5, min_cluster_cells=20)
    assert points_strict == []


def test_cluster_triggers_handles_empty_input() -> None:
    assert cluster_triggers(np.zeros((20, 20))) == []
    assert cluster_triggers(np.full((20, 20), np.nan)) == []


def test_cluster_triggers_validates_inputs() -> None:
    arr = np.zeros((10, 10))
    with pytest.raises(ValueError):
        cluster_triggers(arr[0])  # 1-D
    with pytest.raises(ValueError):
        cluster_triggers(arr, threshold_quantile=0.0)
    with pytest.raises(ValueError):
        cluster_triggers(arr, threshold_quantile=1.0)
    with pytest.raises(ValueError):
        cluster_triggers(arr, min_cluster_cells=0)
    with pytest.raises(ValueError):
        cluster_triggers(arr, connectivity=5)
    with pytest.raises(ValueError, match="cycle_period_s shape"):
        cluster_triggers(arr, cycle_period_s=np.zeros((9, 9)))


def test_cluster_triggers_populates_cycle_period_when_supplied() -> None:
    """When cycle_period_s is supplied, each TriggerPoint gets the
    mean cycle period over its cluster (excluding +inf cells).
    """
    arr = _three_blob_raster()
    cycle = np.full_like(arr, np.inf)
    # Bright blob: short cycle (100 s); medium blob: long (3000 s).
    cycle[15:19, 15:19] = 3000.0
    cycle[25:29, 25:29] = 100.0
    points = cluster_triggers(
        arr,
        threshold_quantile=0.5,
        min_cluster_cells=3,
        cycle_period_s=cycle,
    )
    assert len(points) == 2
    # Brightest first (the 0.95 blob → 100 s cycle).
    assert points[0].mean_cycle_period_s == pytest.approx(100.0)
    assert points[1].mean_cycle_period_s == pytest.approx(3000.0)


def test_cluster_triggers_cycle_period_optional() -> None:
    """Without cycle_period_s, mean_cycle_period_s remains None."""
    arr = _three_blob_raster()
    points = cluster_triggers(arr, threshold_quantile=0.5, min_cluster_cells=3)
    assert all(p.mean_cycle_period_s is None for p in points)


def test_cluster_triggers_leak_weighted_cycle_period_supplied() -> None:
    """leak_weights ⇒ per-cluster τ is leak-weighted, not arithmetic mean.

    Construct one cluster with two distinct (leak, τ) cells that are
    both in the q50 mask. The leak-weighted mean is dominated by the
    high-leak cell, the arithmetic mean is not. With leak ∈ {10, 90},
    τ ∈ {60, 600}: leak-weighted = (10·60 + 90·600) / 100 = 546 s;
    arithmetic = (60 + 600) / 2 = 330 s. Assert the supplied path
    returns 546 and the legacy path (no leak_weights) returns 330.
    """
    arr = np.zeros((10, 10), dtype=np.float64)
    cycle = np.full_like(arr, np.inf)
    leak = np.zeros_like(arr)

    # Single cluster with two cells (rows 2 and 3, col 5).
    arr[2, 5] = 0.9
    arr[3, 5] = 0.9
    arr[4, 5] = 0.9  # 3-cell cluster to clear min_cluster_cells default
    leak[2, 5] = 10.0
    leak[3, 5] = 90.0
    leak[4, 5] = 50.0  # third cell — adds a verifiable middle entry
    cycle[2, 5] = 60.0
    cycle[3, 5] = 600.0
    cycle[4, 5] = 300.0
    # Tail noise so q50 ≈ a low value the cluster cells clear.
    arr[8:10, 0:8] = 0.1

    weighted = cluster_triggers(
        arr,
        threshold_quantile=0.5,
        min_cluster_cells=3,
        cycle_period_s=cycle,
        leak_weights=leak,
    )
    assert len(weighted) == 1
    expected_weighted = (10.0 * 60.0 + 90.0 * 600.0 + 50.0 * 300.0) / (
        10.0 + 90.0 + 50.0
    )
    assert weighted[0].mean_cycle_period_s == pytest.approx(expected_weighted)

    unweighted = cluster_triggers(
        arr,
        threshold_quantile=0.5,
        min_cluster_cells=3,
        cycle_period_s=cycle,
    )
    expected_arithmetic = (60.0 + 600.0 + 300.0) / 3.0
    assert unweighted[0].mean_cycle_period_s == pytest.approx(expected_arithmetic)
    # Sanity: the two paths really do disagree.
    assert weighted[0].mean_cycle_period_s != pytest.approx(
        unweighted[0].mean_cycle_period_s
    )


def test_cluster_triggers_leak_weights_shape_validated() -> None:
    """leak_weights with a mismatched shape raises ValueError."""
    arr = _three_blob_raster()
    with pytest.raises(ValueError, match="leak_weights shape"):
        cluster_triggers(arr, leak_weights=np.zeros((9, 9)))


def test_cluster_triggers_leak_weighted_skips_zero_weight_cells() -> None:
    """Cells with leak_weights == 0 must not contribute to the τ mean.

    A 3-cell cluster where the third cell has leak=0 should give the
    same leak-weighted mean as a 2-cell cluster of the same two
    contributing cells. Guards against accidentally re-introducing
    those cells with τ-only weighting.
    """
    arr = np.zeros((10, 10), dtype=np.float64)
    cycle = np.full_like(arr, np.inf)
    leak = np.zeros_like(arr)

    arr[2, 5] = 0.9
    arr[3, 5] = 0.9
    arr[4, 5] = 0.9
    leak[2, 5] = 10.0
    leak[3, 5] = 90.0
    leak[4, 5] = 0.0  # zero-leak cell, excluded from weighted mean
    cycle[2, 5] = 60.0
    cycle[3, 5] = 600.0
    cycle[4, 5] = 9999.0  # would skew the mean if not skipped
    arr[8:10, 0:8] = 0.1

    points = cluster_triggers(
        arr,
        threshold_quantile=0.5,
        min_cluster_cells=3,
        cycle_period_s=cycle,
        leak_weights=leak,
    )
    expected = (10.0 * 60.0 + 90.0 * 600.0) / (10.0 + 90.0)
    assert points[0].mean_cycle_period_s == pytest.approx(expected)


def test_write_kmz_round_trip(tmp_path: Path) -> None:
    points = [
        TriggerPoint(row=10.5, col=20.5, mean_strength=0.91, n_cells=16),
        TriggerPoint(row=30.0, col=15.0, mean_strength=0.72, n_cells=12),
    ]
    transform = from_origin(380000.0, 500000.0, 2.0, 2.0)
    out = write_kmz(
        points, tmp_path / "trig.kmz", transform=transform, crs=CRS.from_epsg(27700)
    )
    assert out.exists()

    # KMZ is a zip of doc.kml — verify the placemarks made it through.
    with zipfile.ZipFile(out) as kmz:
        assert "doc.kml" in kmz.namelist()
        kml_bytes = kmz.read("doc.kml")
    text = kml_bytes.decode("utf-8")
    # simplekml emits `<Placemark id="...">` (with an id attribute), so
    # count the closing tag instead.
    assert text.count("</Placemark>") == 2
    # Rank 1 is the brightest.
    assert "<name>1</name>" in text
    assert "<name>2</name>" in text
    # No cycle-period data was supplied, so the description does not
    # mention it.
    assert "Cycle period:" not in text


def test_write_kmz_includes_cycle_period_when_present(tmp_path: Path) -> None:
    points = [
        TriggerPoint(
            row=10.5,
            col=20.5,
            mean_strength=0.91,
            n_cells=16,
            mean_cycle_period_s=180.0,  # 3 min
        ),
        TriggerPoint(
            row=30.0,
            col=15.0,
            mean_strength=0.72,
            n_cells=12,
            mean_cycle_period_s=4800.0,  # 80 min
        ),
    ]
    transform = from_origin(380000.0, 500000.0, 2.0, 2.0)
    out = write_kmz(
        points, tmp_path / "trig.kmz", transform=transform, crs=CRS.from_epsg(27700)
    )
    with zipfile.ZipFile(out) as kmz:
        text = kmz.read("doc.kml").decode("utf-8")
    # Both placemarks should carry a Cycle period: line, formatted in
    # the most pilot-readable unit (min for 180–3600 s, hr above).
    assert text.count("Cycle period:") == 2
    assert "Cycle period: 3.0 min" in text
    assert "Cycle period: 1.3 hr" in text


def test_write_kmz_requires_crs(tmp_path: Path) -> None:
    points = [TriggerPoint(row=0, col=0, mean_strength=0.5, n_cells=5)]
    transform = Affine.identity()
    with pytest.raises(ValueError):
        write_kmz(points, tmp_path / "no_crs.kmz", transform=transform, crs=None)


def test_write_kmz_rejects_empty(tmp_path: Path) -> None:
    transform = Affine.identity()
    with pytest.raises(ValueError):
        write_kmz([], tmp_path / "empty.kmz", transform=transform, crs="EPSG:27700")
