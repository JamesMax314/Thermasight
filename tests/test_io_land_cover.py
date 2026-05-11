"""Tests for ``thermal_model.io.land_cover``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from thermal_model.io import (
    DALES_LCM_ABSORPTIVITY,
    UKCEH_LCM_ABSORPTIVITY,
    LandCover,
    absorptivity_from_land_cover,
    read_dem,
    read_land_cover,
)
from thermal_model.physics import DEFAULT_ABSORPTIVITY


def _write_lcm(
    path: Path,
    classes: np.ndarray,
    *,
    transform=None,
    crs=None,
    nodata: int | None = 255,
) -> None:
    if transform is None:
        transform = from_origin(400000.0, 450000.0, 1.0, 1.0)
    if crs is None:
        crs = CRS.from_epsg(27700)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=classes.shape[0],
        width=classes.shape[1],
        count=1,
        dtype=classes.dtype.name,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(classes, 1)


def test_read_land_cover_round_trip(synthetic_lcm_path: Path) -> None:
    lcm = read_land_cover(synthetic_lcm_path)
    assert isinstance(lcm, LandCover)
    assert lcm.classes.shape == (256, 256)
    assert lcm.classes.dtype == np.int16
    assert lcm.cell_size_m == pytest.approx(1.0)
    assert lcm.crs.to_epsg() == 27700
    # No nodata pixels in the synthetic fixture (all four quadrants
    # filled), so no -1 sentinel.
    assert (lcm.classes == -1).sum() == 0
    # Quadrants come back exactly as written.
    assert np.all(lcm.classes[:128, :128] == 9)
    assert np.all(lcm.classes[:128, 128:] == 11)
    assert np.all(lcm.classes[128:, :128] == 12)
    assert np.all(lcm.classes[128:, 128:] == 14)


def test_absorptivity_uniform_lcm_matches_lookup(
    tmp_path: Path, synthetic_dem_path: Path
) -> None:
    """A uniform-class LCM under DALES_LCM_ABSORPTIVITY returns a uniform α array."""
    classes = np.full((256, 256), 11, dtype=np.uint8)  # all bog
    lcm_path = tmp_path / "all_bog.tif"
    _write_lcm(lcm_path, classes)
    dem = read_dem(synthetic_dem_path)
    lcm = read_land_cover(lcm_path)

    alpha = absorptivity_from_land_cover(lcm, dem, lookup=DALES_LCM_ABSORPTIVITY)

    finite = np.isfinite(dem.elevation_m)
    assert np.all(alpha[finite] == pytest.approx(0.40))
    # NaN propagates from the DEM's NaN cell.
    assert np.isnan(alpha[~finite]).all()


def test_absorptivity_empty_default_lookup_falls_back(
    tmp_path: Path, synthetic_dem_path: Path
) -> None:
    """Empty UKCEH_LCM_ABSORPTIVITY → every cell hits unknown_fill (0.80).

    Regression guard for the deferred-LUT product decision (the
    operator authors the production 21-class table later; until then
    --land-cover behaves like a scalar α=0.80 run, which is the safe
    default).
    """
    assert UKCEH_LCM_ABSORPTIVITY == {}, (
        "UKCEH_LCM_ABSORPTIVITY is reserved as the operator-authored "
        "production LUT and must remain empty until that work lands."
    )
    classes = np.full((256, 256), 11, dtype=np.uint8)  # all bog
    lcm_path = tmp_path / "all_bog.tif"
    _write_lcm(lcm_path, classes)
    dem = read_dem(synthetic_dem_path)
    lcm = read_land_cover(lcm_path)

    # No warning is emitted on this happy path: the empty LUT has no
    # known codes, so warning logic short-circuits.
    alpha = absorptivity_from_land_cover(lcm, dem)

    finite = np.isfinite(dem.elevation_m)
    assert np.all(alpha[finite] == pytest.approx(DEFAULT_ABSORPTIVITY))


def test_absorptivity_unknown_class_fallback_warns(
    tmp_path: Path, synthetic_dem_path: Path
) -> None:
    """An unknown class code produces unknown_fill *and* a UserWarning."""
    classes = np.full((256, 256), 11, dtype=np.uint8)
    classes[0:10, 0:10] = 99  # unknown
    lcm_path = tmp_path / "with_unknown.tif"
    _write_lcm(lcm_path, classes)
    dem = read_dem(synthetic_dem_path)
    lcm = read_land_cover(lcm_path)

    with pytest.warns(UserWarning, match="class codes outside the"):
        alpha = absorptivity_from_land_cover(lcm, dem, lookup=DALES_LCM_ABSORPTIVITY)

    # Bog cells → 0.40
    assert alpha[200, 200] == pytest.approx(0.40)
    # Unknown sliver → unknown_fill default = 0.80
    assert alpha[5, 5] == pytest.approx(DEFAULT_ABSORPTIVITY)


def test_absorptivity_lcm_nodata_falls_back_to_unknown_fill_not_nan(
    tmp_path: Path, synthetic_dem_path: Path
) -> None:
    """LCM nodata where DEM is finite → α = unknown_fill, not NaN.

    This is the load-bearing routing-preservation contract from the
    plan: a sliver of unclassified land must not silently NaN-zero
    the heating-weighted leaky-bucket routing.
    """
    classes = np.full((256, 256), 11, dtype=np.uint8)
    classes[100:110, 100:110] = 255  # nodata sentinel
    lcm_path = tmp_path / "with_nodata.tif"
    _write_lcm(lcm_path, classes, nodata=255)
    dem = read_dem(synthetic_dem_path)
    lcm = read_land_cover(lcm_path)

    with pytest.warns(UserWarning, match="class codes outside the"):
        alpha = absorptivity_from_land_cover(lcm, dem, lookup=DALES_LCM_ABSORPTIVITY)

    # The nodata block sits on finite DEM cells (DEM is a Gaussian hill
    # with a single NaN at [0, 0]). Their α must be the fallback, not
    # NaN — anywhere finite-DEM ∧ LCM-nodata.
    block = alpha[100:110, 100:110]
    assert np.all(np.isfinite(block))
    assert np.all(block == pytest.approx(DEFAULT_ABSORPTIVITY))


def test_absorptivity_dem_nan_propagates(
    tmp_path: Path, synthetic_dem_path: Path
) -> None:
    classes = np.full((256, 256), 11, dtype=np.uint8)
    lcm_path = tmp_path / "all_bog.tif"
    _write_lcm(lcm_path, classes)
    dem = read_dem(synthetic_dem_path)
    lcm = read_land_cover(lcm_path)

    alpha = absorptivity_from_land_cover(lcm, dem, lookup=DALES_LCM_ABSORPTIVITY)
    # The synthetic DEM has a single NaN cell at (0, 0).
    assert np.isnan(alpha[0, 0])
    # Other cells are finite α.
    assert np.isfinite(alpha[10, 10])


def test_absorptivity_override_lookup(tmp_path: Path, synthetic_dem_path: Path) -> None:
    classes = np.full((256, 256), 11, dtype=np.uint8)
    lcm_path = tmp_path / "all_bog.tif"
    _write_lcm(lcm_path, classes)
    dem = read_dem(synthetic_dem_path)
    lcm = read_land_cover(lcm_path)

    alpha = absorptivity_from_land_cover(lcm, dem, lookup={11: 0.99})
    assert alpha[100, 100] == pytest.approx(0.99)


def test_absorptivity_override_unknown_fill(
    tmp_path: Path, synthetic_dem_path: Path
) -> None:
    classes = np.full((256, 256), 99, dtype=np.uint8)  # all unknown
    lcm_path = tmp_path / "all_unknown.tif"
    _write_lcm(lcm_path, classes)
    dem = read_dem(synthetic_dem_path)
    lcm = read_land_cover(lcm_path)

    with pytest.warns(UserWarning, match="class codes outside the"):
        alpha = absorptivity_from_land_cover(
            lcm, dem, lookup=DALES_LCM_ABSORPTIVITY, unknown_fill=0.30
        )
    finite = np.isfinite(dem.elevation_m)
    assert np.all(alpha[finite] == pytest.approx(0.30))


def test_absorptivity_shape_matches_reference(
    synthetic_lcm_path: Path, synthetic_dem_path: Path
) -> None:
    dem = read_dem(synthetic_dem_path)
    lcm = read_land_cover(synthetic_lcm_path)
    alpha = absorptivity_from_land_cover(lcm, dem, lookup=DALES_LCM_ABSORPTIVITY)
    assert alpha.shape == dem.shape
    assert alpha.dtype == np.float64


def test_absorptivity_rejects_out_of_range_unknown_fill(
    synthetic_lcm_path: Path, synthetic_dem_path: Path
) -> None:
    dem = read_dem(synthetic_dem_path)
    lcm = read_land_cover(synthetic_lcm_path)
    with pytest.raises(ValueError, match="unknown_fill"):
        absorptivity_from_land_cover(lcm, dem, unknown_fill=1.5)


def test_absorptivity_rejects_out_of_range_lookup_value(
    synthetic_lcm_path: Path, synthetic_dem_path: Path
) -> None:
    dem = read_dem(synthetic_dem_path)
    lcm = read_land_cover(synthetic_lcm_path)
    with pytest.raises(ValueError, match=r"class 11"):
        absorptivity_from_land_cover(lcm, dem, lookup={11: 1.5})


def test_absorptivity_warns_on_crs_mismatch(
    tmp_path: Path, synthetic_dem_path: Path
) -> None:
    """LCM in a different CRS warns and still returns finite α."""
    classes = np.full((128, 128), 11, dtype=np.uint8)
    transform = from_origin(0.0, 60.0, 0.001, 0.001)  # nominal lat/lon
    lcm_path = tmp_path / "wgs84_bog.tif"
    _write_lcm(lcm_path, classes, transform=transform, crs=CRS.from_epsg(4326))
    dem = read_dem(synthetic_dem_path)

    with pytest.warns(UserWarning):
        lcm = read_land_cover(lcm_path)
    with pytest.warns(UserWarning, match="CRS"):
        alpha = absorptivity_from_land_cover(lcm, dem, lookup=DALES_LCM_ABSORPTIVITY)

    # We don't assert content here — the WGS84 footprint doesn't overlap
    # the BNG DEM in any meaningful way, so most cells should fall back
    # to unknown_fill. We just want to ensure the call doesn't raise.
    assert alpha.shape == dem.shape
    finite = np.isfinite(dem.elevation_m)
    assert np.all(np.isfinite(alpha[finite]))


def test_absorptivity_resampling_preserves_classes(
    tmp_path: Path,
) -> None:
    """A coarser DEM resamples the LCM via nearest-neighbour, not bilinear.

    Build an LCM with a sharp 3-class boundary and a DEM that's exactly
    half the LCM resolution. Each output cell must equal one of the three
    input class codes — never an interpolated value.
    """
    n = 64
    cell_lcm = 1.0
    classes = np.full((n, n), 9, dtype=np.uint8)
    classes[:, : n // 3] = 4
    classes[:, 2 * n // 3 :] = 12
    transform = from_origin(400000.0, 450000.0, cell_lcm, cell_lcm)
    lcm_path = tmp_path / "stripes.tif"
    _write_lcm(lcm_path, classes, transform=transform)

    # Coarse DEM: half the resolution, same origin.
    cell_dem = 2.0
    dem_n = n // 2
    elevation = np.full((dem_n, dem_n), 100.0, dtype=np.float32)
    dem_path = tmp_path / "coarse_dem.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=dem_n,
        width=dem_n,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(27700),
        transform=from_origin(400000.0, 450000.0, cell_dem, cell_dem),
        nodata=-9999.0,
    ) as dst:
        dst.write(elevation, 1)
    dem = read_dem(dem_path)
    lcm = read_land_cover(lcm_path)

    alpha = absorptivity_from_land_cover(lcm, dem, lookup=DALES_LCM_ABSORPTIVITY)

    # Only the three class α values should appear (plus possibly the
    # heather class unknown_fill if a cell falls outside; here all three
    # classes are in the lookup so no unknown_fill expected).
    expected_alphas = {0.75, 0.80, 0.85}
    observed = set(np.unique(alpha).tolist())
    assert observed.issubset(expected_alphas), observed


def test_landcover_dataclass_shape_property() -> None:
    classes = np.zeros((10, 20), dtype=np.int16)
    transform = from_origin(0.0, 0.0, 1.0, 1.0)
    lc = LandCover(
        classes=classes,
        transform=transform,
        crs=CRS.from_epsg(27700),
        cell_size_m=1.0,
    )
    assert lc.shape == (10, 20)
