"""Tests for thermal_model.io.mosaic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from thermal_model.io import mosaic_dems, read_dem


def _write_tile(
    path: Path,
    *,
    origin_x: float,
    origin_y: float,
    cell: float,
    n: int,
    fill: float,
    nodata: float = -3.4028234663852886e38,
    crs: CRS | None = None,
) -> None:
    """Write a small synthetic single-band float32 GeoTIFF."""
    if crs is None:
        crs = CRS.from_epsg(27700)
    transform = from_origin(origin_x, origin_y, cell, cell)
    arr = np.full((n, n), fill, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


@pytest.fixture
def four_tile_grid(tmp_path: Path) -> list[Path]:
    """A 2x2 grid of 4 tiles, each 8x8 cells at 1 m, distinguishable
    fill values so the mosaic can be checked quadrant-by-quadrant.
    """
    n = 8
    cell = 1.0
    # Layout (BNG-like, north-up): rows top-down by northing, cols
    # left-right by easting.
    #   NW (E=0,  N=16)  fill=10        NE (E=8,  N=16)  fill=20
    #   SW (E=0,  N=8 )  fill=30        SE (E=8,  N=8 )  fill=40
    cases = [
        ("nw.tif", 0.0, 16.0, 10.0),
        ("ne.tif", 8.0, 16.0, 20.0),
        ("sw.tif", 0.0, 8.0, 30.0),
        ("se.tif", 8.0, 8.0, 40.0),
    ]
    paths = []
    for name, ox, oy, fill in cases:
        p = tmp_path / name
        _write_tile(p, origin_x=ox, origin_y=oy, cell=cell, n=n, fill=fill)
        paths.append(p)
    return paths


def test_mosaic_combines_four_quadrants(
    four_tile_grid: list[Path], tmp_path: Path
) -> None:
    out = tmp_path / "mosaic.tif"
    result = mosaic_dems(four_tile_grid, out)
    assert result == out

    dem = read_dem(out)
    assert dem.shape == (16, 16)
    assert dem.cell_size_m == 1.0
    assert dem.crs == CRS.from_epsg(27700)

    elev = dem.elevation_m
    # Top-left 8x8 block is NW (fill=10), etc. Row 0 is the top.
    np.testing.assert_array_equal(elev[:8, :8], 10.0)
    np.testing.assert_array_equal(elev[:8, 8:], 20.0)
    np.testing.assert_array_equal(elev[8:, :8], 30.0)
    np.testing.assert_array_equal(elev[8:, 8:], 40.0)


def test_mosaic_rewrites_nodata_to_minus_9999(
    four_tile_grid: list[Path], tmp_path: Path
) -> None:
    # Inject a nodata cell into one tile, then verify the mosaic
    # exposes -9999 (not the float32 sentinel) on disk.
    nw = four_tile_grid[0]
    with rasterio.open(nw, "r+") as ds:
        a = ds.read(1)
        a[0, 0] = ds.nodata
        ds.write(a, 1)

    out = tmp_path / "mosaic.tif"
    mosaic_dems(four_tile_grid, out)

    with rasterio.open(out) as ds:
        assert ds.nodata == -9999.0
        assert ds.read(1)[0, 0] == -9999.0


def test_mosaic_refuses_overwrite_by_default(
    four_tile_grid: list[Path], tmp_path: Path
) -> None:
    out = tmp_path / "mosaic.tif"
    mosaic_dems(four_tile_grid, out)
    with pytest.raises(FileExistsError):
        mosaic_dems(four_tile_grid, out)


def test_mosaic_overwrite_flag_replaces_existing(
    four_tile_grid: list[Path], tmp_path: Path
) -> None:
    out = tmp_path / "mosaic.tif"
    mosaic_dems(four_tile_grid, out)
    # Should succeed second time with overwrite=True.
    mosaic_dems(four_tile_grid, out, overwrite=True)


def test_mosaic_rejects_crs_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    _write_tile(a, origin_x=0.0, origin_y=8.0, cell=1.0, n=8, fill=1.0)
    _write_tile(
        b,
        origin_x=8.0,
        origin_y=8.0,
        cell=1.0,
        n=8,
        fill=2.0,
        crs=CRS.from_epsg(4326),
    )
    with pytest.raises(ValueError, match="CRS mismatch"):
        mosaic_dems([a, b], tmp_path / "out.tif")


def test_mosaic_rejects_cell_size_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    _write_tile(a, origin_x=0.0, origin_y=8.0, cell=1.0, n=8, fill=1.0)
    _write_tile(b, origin_x=8.0, origin_y=8.0, cell=2.0, n=4, fill=2.0)
    with pytest.raises(ValueError, match="cell size mismatch"):
        mosaic_dems([a, b], tmp_path / "out.tif")


def test_mosaic_requires_at_least_one_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no input paths"):
        mosaic_dems([], tmp_path / "out.tif")


def test_mosaic_single_tile_produces_copy(
    tmp_path: Path, four_tile_grid: list[Path]
) -> None:
    out = tmp_path / "single.tif"
    mosaic_dems([four_tile_grid[0]], out)
    dem = read_dem(out)
    assert dem.shape == (8, 8)
    np.testing.assert_array_equal(dem.elevation_m, 10.0)
