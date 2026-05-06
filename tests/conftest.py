"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin


@pytest.fixture
def synthetic_dem_path(tmp_path: Path) -> Path:
    """A 256×256 synthetic DEM in EPSG:27700 with a Gaussian hill.

    Useful as a standin for real LIDAR fixtures during early development.
    """
    n = 256
    cell = 1.0
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    cx = cy = (n - 1) / 2.0
    sigma = n / 6.0
    elevation = 400.0 + 80.0 * np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)
    )
    # Drop a few nodata cells to exercise the NaN conversion.
    elevation[0, 0] = np.nan
    nodata = -9999.0
    raw = np.where(np.isnan(elevation), nodata, elevation).astype(np.float32)

    path = tmp_path / "synthetic_hill.tif"
    transform = from_origin(400000.0, 450000.0, cell, cell)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(27700),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(raw, 1)
    return path
