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


def test_write_kmz_requires_crs(tmp_path: Path) -> None:
    points = [TriggerPoint(row=0, col=0, mean_strength=0.5, n_cells=5)]
    transform = Affine.identity()
    with pytest.raises(ValueError):
        write_kmz(points, tmp_path / "no_crs.kmz", transform=transform, crs=None)


def test_write_kmz_rejects_empty(tmp_path: Path) -> None:
    transform = Affine.identity()
    with pytest.raises(ValueError):
        write_kmz([], tmp_path / "empty.kmz", transform=transform, crs="EPSG:27700")
