"""DEM read/write helpers.

Conventions (see CLAUDE.md §6):

* Internally, nodata is represented as ``np.nan``; rasterio's ``nodata``
  sentinel is converted on read and restored on write.
* Cell size is read from the affine transform and exposed alongside the
  array — never assumed by callers.
* The CRS is read from the file and warned on if missing; computations
  assume a projected CRS with metres as units (typically EPSG:27700).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass(frozen=True)
class DEM:
    """A digital elevation model loaded into memory.

    Attributes
    ----------
    elevation_m : np.ndarray
        2-D array of elevations in metres, with nodata as ``np.nan``.
    transform : Affine
        Affine transform mapping pixel (col, row) to projected (x, y).
    crs : rasterio.crs.CRS | None
        Coordinate reference system. ``None`` if the source file lacked one.
    cell_size_m : float
        Cell size in metres (assumes square pixels in a metric CRS).
    """

    elevation_m: np.ndarray
    transform: Affine
    crs: Any
    cell_size_m: float

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self.elevation_m.shape
        return rows, cols


def read_dem(path: str | Path) -> DEM:
    """Read a single-band DEM GeoTIFF into a :class:`DEM`.

    Nodata pixels are converted to ``np.nan``. The CRS is checked and a
    warning is emitted if it is missing or not metric.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"{path}: expected single-band DEM, got {src.count} bands")
        elevation = src.read(1, masked=False).astype(np.float64, copy=True)
        nodata = src.nodata
        if nodata is not None:
            elevation[elevation == nodata] = np.nan

        crs = src.crs
        if crs is None:
            warnings.warn(f"{path}: no CRS on raster", stacklevel=2)
        elif not crs.is_projected:
            warnings.warn(
                f"{path}: CRS {crs.to_string()} is not projected; "
                "computations assume metric units",
                stacklevel=2,
            )

        a, _, _, _, e, _ = (
            src.transform.a,
            src.transform.b,
            src.transform.c,
            src.transform.d,
            src.transform.e,
            src.transform.f,
        )
        cell_size_m = float(abs(a))
        if not np.isclose(abs(a), abs(e)):
            warnings.warn(
                f"{path}: non-square pixels ({a}, {e}); using x cell size",
                stacklevel=2,
            )

        return DEM(
            elevation_m=elevation,
            transform=src.transform,
            crs=crs,
            cell_size_m=cell_size_m,
        )


def write_raster_like(
    path: str | Path,
    array: np.ndarray,
    reference: DEM,
    *,
    nodata: float = -9999.0,
    dtype: str = "float32",
) -> None:
    """Write ``array`` as a GeoTIFF using ``reference``'s transform and CRS.

    NaN values in ``array`` are written as ``nodata``.
    """
    path = Path(path)
    if array.shape != reference.elevation_m.shape:
        raise ValueError(
            f"array shape {array.shape} does not match reference "
            f"{reference.elevation_m.shape}"
        )
    out = np.where(np.isnan(array), nodata, array).astype(dtype, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype=dtype,
        crs=reference.crs,
        transform=reference.transform,
        nodata=nodata,
        compress="deflate",
        predictor=3 if dtype.startswith("float") else 2,
    ) as dst:
        dst.write(out, 1)
