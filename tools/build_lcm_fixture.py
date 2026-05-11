"""Generate the Wild Boar Fell LCM fixture aligned with the DEM fixture.

Produces ``data/fixtures/wild_boar_fell_east_256_lcm.tif`` — a synthetic
*but plausible* categorical land-cover raster sharing the DEM fixture's
transform and CRS exactly. Used by tests and CLI smoke runs that need a
non-trivial LCM tile without depending on a UKCEH download.

Layout (256×256, 1 m cells, EPSG:27700):

* class  9 (heather) — upper slopes, default
* class 11 (bog)     — plateau interior (NW quadrant inset)
* class 12 (inland rock) — scarp lip diagonal stripe
* class  4 (improved grassland) — lower SE corner
* class 14 (freshwater)  — small tarn (round patch)
* class 99 (unknown)     — small sliver to exercise the fallback warning

Run from the repo root with the ``thermals`` env active::

    python tools/build_lcm_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dem_path = repo_root / "data" / "fixtures" / "wild_boar_fell_east_256_1m.tif"
    out_path = repo_root / "data" / "fixtures" / "wild_boar_fell_east_256_lcm.tif"

    with rasterio.open(dem_path) as src:
        transform = src.transform
        crs = src.crs
        height = src.height

    n = height
    classes = np.full((n, n), 9, dtype=np.uint8)  # heather default

    # Bog: NW plateau interior
    classes[20:120, 20:140] = 11
    # Inland rock: diagonal stripe approximating the scarp lip
    yy, xx = np.indices((n, n))
    scarp = np.abs(yy - (n - xx) - 30) < 10
    classes[scarp] = 12
    # Improved grassland: lower-right valley
    classes[180:, 160:] = 4
    # Small freshwater tarn
    cy, cx = 200, 60
    rr = (yy - cy) ** 2 + (xx - cx) ** 2
    classes[rr <= 16**2] = 14
    # Unknown-class sliver to exercise the fallback warning path
    classes[5:8, 100:140] = 99

    nodata = 255
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(classes, 1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
