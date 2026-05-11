"""Tests for ``thermal_model.io.land_cover_wms`` (mocked HTTP)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from thermal_model.io import DEM, fetch_lcm_for_dem
from thermal_model.io.land_cover_wms import (
    UKCEH_LCM_PALETTE,
    _build_get_map_url,
    _decode_png_to_classes,
)


def _bng_dem(rows: int = 64, cols: int = 64, cell: float = 10.0) -> DEM:
    """A bare DEM dataclass in EPSG:27700 (no I/O needed)."""
    transform = from_origin(370000.0, 500000.0, cell, cell)
    return DEM(
        elevation_m=np.full((rows, cols), 400.0, dtype=np.float64),
        transform=transform,
        crs=CRS.from_epsg(27700),
        cell_size_m=cell,
    )


def _make_palette_png(class_grid: np.ndarray) -> bytes:
    """Encode a 2-D class-code grid into a 3-band RGB PNG via the palette."""
    inv_palette: dict[int, tuple[int, int, int]] = {
        code: rgb for rgb, code in UKCEH_LCM_PALETTE.items()
    }
    h, w = class_grid.shape
    rgb = np.zeros((3, h, w), dtype=np.uint8)
    for code in np.unique(class_grid):
        if int(code) not in inv_palette:
            continue
        r, g, b = inv_palette[int(code)]
        mask = class_grid == code
        rgb[0][mask] = r
        rgb[1][mask] = g
        rgb[2][mask] = b
    buf = BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="PNG",
        height=h,
        width=w,
        count=3,
        dtype="uint8",
    ) as dst:
        dst.write(rgb)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_decode_png_round_trip() -> None:
    """A palette-encoded PNG round-trips back to the original class grid."""
    grid = np.array(
        [
            [11, 11, 9, 9],
            [11, 12, 12, 9],
            [4, 4, 14, 14],
            [4, 21, 21, 14],
        ],
        dtype=np.int16,
    )
    png_bytes = _make_palette_png(grid)
    decoded = _decode_png_to_classes(png_bytes)
    assert decoded.shape == grid.shape
    np.testing.assert_array_equal(decoded, grid.astype(np.int16))


def test_decode_png_unknown_rgb_warns() -> None:
    """An RGB not in the palette becomes ``-1`` and emits a UserWarning."""
    grid = np.array([[11, 11], [11, 11]], dtype=np.int16)
    png_bytes = _make_palette_png(grid)
    # Read it back and overwrite a single pixel with an off-palette colour.
    with rasterio.open(BytesIO(png_bytes)) as src:
        rgb = src.read()
    rgb[0, 0, 0] = 200
    rgb[1, 0, 0] = 200
    rgb[2, 0, 0] = 200  # off-palette grey
    buf = BytesIO()
    with rasterio.open(
        buf, "w", driver="PNG", height=2, width=2, count=3, dtype="uint8"
    ) as dst:
        dst.write(rgb)
    bad_png = buf.getvalue()

    with pytest.warns(UserWarning, match="did not match any UKCEH palette"):
        decoded = _decode_png_to_classes(bad_png)
    assert decoded[0, 0] == -1
    assert decoded[0, 1] == 11


def test_build_get_map_url_has_required_params() -> None:
    url = _build_get_map_url(
        base_url="https://example.invalid/foo",
        layer="LC.10m.GB",
        bbox_27700=(370000.0, 480000.0, 380000.0, 490000.0),
        width=1000,
        height=1000,
        version="1.3.0",
        crs="EPSG:27700",
        fmt="image/png",
        style="",
    )
    assert url.startswith("https://example.invalid/foo?")
    for token in (
        "service=WMS",
        "version=1.3.0",
        "request=GetMap",
        "layers=LC.10m.GB",
        "crs=EPSG%3A27700",
        "bbox=370000.0%2C480000.0%2C380000.0%2C490000.0",
        "width=1000",
        "height=1000",
        "format=image%2Fpng",
    ):
        assert token in url, token


def test_fetch_lcm_for_dem_uses_cache_on_second_call(tmp_path: Path) -> None:
    dem = _bng_dem(rows=64, cols=64, cell=10.0)
    grid = np.full((64, 64), 11, dtype=np.int16)  # all bog
    png_bytes = _make_palette_png(grid)

    with patch(
        "thermal_model.io.land_cover_wms.urllib.request.urlopen",
        return_value=_FakeResponse(png_bytes),
    ) as mock_open:
        first = fetch_lcm_for_dem(dem, cache_dir=tmp_path / "cache")
        assert mock_open.call_count == 1
        # Second call must hit the on-disk cache, not the network.
        second = fetch_lcm_for_dem(dem, cache_dir=tmp_path / "cache")
        assert mock_open.call_count == 1

    assert first.classes.shape == (64, 64)
    np.testing.assert_array_equal(first.classes, second.classes)
    assert first.crs.to_epsg() == 27700
    assert first.cell_size_m == pytest.approx(10.0)


def test_fetch_lcm_for_dem_no_cache_always_fetches(tmp_path: Path) -> None:
    dem = _bng_dem()
    grid = np.full((64, 64), 11, dtype=np.int16)
    png_bytes = _make_palette_png(grid)

    with patch(
        "thermal_model.io.land_cover_wms.urllib.request.urlopen",
        return_value=_FakeResponse(png_bytes),
    ) as mock_open:
        fetch_lcm_for_dem(dem, cache_dir=tmp_path / "cache", use_cache=False)
        fetch_lcm_for_dem(dem, cache_dir=tmp_path / "cache", use_cache=False)
        assert mock_open.call_count == 2
    # Cache directory was never written to.
    assert not (tmp_path / "cache").exists()


def test_fetch_lcm_rejects_oversize_bbox(tmp_path: Path) -> None:
    """A bbox whose 10 m raster would exceed 2048 px raises NotImplementedError."""
    # 50 km × 50 km at 10 m → 5000×5000 pixels.
    transform = from_origin(370000.0, 500000.0, 1.0, 1.0)
    big_dem = DEM(
        elevation_m=np.zeros((50000, 50000), dtype=np.float32),
        transform=transform,
        crs=CRS.from_epsg(27700),
        cell_size_m=1.0,
    )
    with pytest.raises(NotImplementedError, match="2048"):
        fetch_lcm_for_dem(big_dem, cache_dir=tmp_path / "cache")


def test_fetch_lcm_requires_dem_crs(tmp_path: Path) -> None:
    transform = from_origin(0.0, 0.0, 10.0, 10.0)
    no_crs_dem = DEM(
        elevation_m=np.zeros((10, 10), dtype=np.float32),
        transform=transform,
        crs=None,
        cell_size_m=10.0,
    )
    with pytest.raises(ValueError, match="reference.crs"):
        fetch_lcm_for_dem(no_crs_dem, cache_dir=tmp_path / "cache")
