"""Round-trip tests for DEM I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
