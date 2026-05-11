"""Shared test fixtures."""

from __future__ import annotations

import matplotlib

# Use a non-interactive backend so viz tests run headless under CI and
# don't try to open a display. Must happen before pyplot is imported.
matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rasterio  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402


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


@pytest.fixture
def synthetic_lcm_path(tmp_path: Path) -> Path:
    """A 256×256 synthetic UKCEH LCM raster aligned with ``synthetic_dem_path``.

    Four quadrants covering the most physically distinct DATA.md
    surfaces:

    * NW: class 9  (heather)
    * NE: class 11 (bog) — the load-bearing wet-ground entry
    * SW: class 12 (inland rock)
    * SE: class 14 (freshwater)

    Same transform / CRS as ``synthetic_dem_path`` so the pair can be
    fed into ``run_model`` without reprojection drift.
    """
    n = 256
    cell = 1.0
    classes = np.full((n, n), 255, dtype=np.uint8)
    half = n // 2
    classes[:half, :half] = 9
    classes[:half, half:] = 11
    classes[half:, :half] = 12
    classes[half:, half:] = 14
    nodata = 255  # outside the 1..21 UKCEH range

    path = tmp_path / "synthetic_lcm.tif"
    transform = from_origin(400000.0, 450000.0, cell, cell)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(27700),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(classes, 1)
    return path


@pytest.fixture
def wild_boar_fell_fixture_path() -> Path:
    """Path to the 256×256 1 m DTM fixture over Wild Boar Fell's east edge.

    Crop from the Environment Agency LIDAR Composite (2022, 1 m), centred
    on (376050, 498700) BNG — the convex break at the lip of the plateau
    above the Mallerstang scarp. Used to validate the I/O pipeline against
    real terrain rather than purely synthetic data.
    """
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "fixtures"
        / "wild_boar_fell_east_256_1m.tif"
    )
