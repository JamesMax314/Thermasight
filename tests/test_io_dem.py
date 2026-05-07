"""Round-trip tests for DEM I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermal_model.io import read_dem, write_raster_like


def test_read_dem_converts_nodata_to_nan(synthetic_dem_path: Path) -> None:
    dem = read_dem(synthetic_dem_path)
    assert dem.elevation_m.shape == (256, 256)
    assert dem.cell_size_m == 1.0
    assert np.isnan(dem.elevation_m[0, 0])
    # Everything else should be finite.
    finite = np.isfinite(dem.elevation_m)
    assert finite.sum() == 256 * 256 - 1


def test_write_raster_like_round_trips(
    tmp_path: Path, synthetic_dem_path: Path
) -> None:
    dem = read_dem(synthetic_dem_path)
    out_path = tmp_path / "out.tif"
    write_raster_like(out_path, dem.elevation_m, reference=dem)

    reloaded = read_dem(out_path)
    assert reloaded.shape == dem.shape
    assert reloaded.cell_size_m == dem.cell_size_m
    assert reloaded.crs == dem.crs

    a = dem.elevation_m
    b = reloaded.elevation_m
    mask = np.isfinite(a) & np.isfinite(b)
    np.testing.assert_allclose(a[mask], b[mask], rtol=0, atol=1e-3)
    assert np.isnan(b[0, 0])


def test_read_dem_resamples_to_coarser_resolution(synthetic_dem_path: Path) -> None:
    # 256x256 at 1 m -> request 4 m -> 64x64.
    dem = read_dem(synthetic_dem_path, target_cell_size_m=4.0)
    assert dem.shape == (64, 64)
    assert dem.cell_size_m == pytest.approx(4.0)
    # The transform spans the same world extent as the original
    # (256 m x 256 m), preserved across the resample.
    rows, cols = dem.shape
    assert cols * dem.cell_size_m == pytest.approx(256.0)
    assert rows * abs(dem.transform.e) == pytest.approx(256.0)


def test_read_dem_resample_to_native_size_is_a_noop(synthetic_dem_path: Path) -> None:
    dem = read_dem(synthetic_dem_path, target_cell_size_m=1.0)
    assert dem.shape == (256, 256)
    assert dem.cell_size_m == 1.0


def test_read_dem_resample_does_not_pollute_with_nodata_sentinel(
    synthetic_dem_path: Path,
) -> None:
    # The synthetic fixture stores nodata as -9999. After bilinear
    # downsampling, no output cell should land near the sentinel — that
    # would mean the resampler averaged real values with -9999 instead
    # of either skipping them or being masked off.
    dem = read_dem(synthetic_dem_path, target_cell_size_m=4.0)
    finite = dem.elevation_m[np.isfinite(dem.elevation_m)]
    # Source elevations are 400-480 m (Gaussian hill on a 400 m
    # baseline). Anything sentinel-polluted would land well outside.
    assert finite.min() > 300.0
    assert finite.max() < 600.0


def test_read_dem_refuses_to_upsample(synthetic_dem_path: Path) -> None:
    with pytest.raises(ValueError, match="finer than the source"):
        read_dem(synthetic_dem_path, target_cell_size_m=0.5)


def test_read_dem_rejects_non_positive_target(synthetic_dem_path: Path) -> None:
    with pytest.raises(ValueError, match="target_cell_size_m"):
        read_dem(synthetic_dem_path, target_cell_size_m=0.0)


def test_read_dem_real_lidar_fixture(wild_boar_fell_fixture_path: Path) -> None:
    dem = read_dem(wild_boar_fell_fixture_path)
    assert dem.shape == (256, 256)
    assert dem.cell_size_m == 1.0
    assert dem.crs is not None and dem.crs.to_epsg() == 27700

    finite = dem.elevation_m[np.isfinite(dem.elevation_m)]
    # Wild Boar Fell summit is 708 m; the surrounding plateau and east
    # scarp sit roughly 600–710 m. Allow some slack for nearby drops.
    assert 500.0 < finite.min() < 700.0
    assert 690.0 < finite.max() < 720.0
    # The window straddles the plateau lip, so meaningful relief is required.
    assert finite.max() - finite.min() > 30.0
