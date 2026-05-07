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
from rasterio.enums import Resampling
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


def read_dem(
    path: str | Path,
    *,
    target_cell_size_m: float | None = None,
) -> DEM:
    """Read a single-band DEM GeoTIFF into a :class:`DEM`.

    Nodata pixels are converted to ``np.nan``. The CRS is checked and a
    warning is emitted if it is missing or not metric.

    Parameters
    ----------
    path : str or Path
        Path to a single-band DEM GeoTIFF.
    target_cell_size_m : float, optional
        If provided and coarser than the source, bilinearly resample on
        read so the returned DEM has the requested cell size. Useful for
        diagnostic plots over large mosaics where the cast-shadow march
        and other per-cell operations would otherwise be slow at full
        resolution. The output transform is rebuilt to match the new
        cell size; downstream code reads ``cell_size_m`` from the DEM
        and so keeps working unchanged.

        Refusing to *upsample* is intentional: a finer-than-source DEM
        is a fabrication, and the typical use case is "go coarser to
        save time."

    Returns
    -------
    DEM
        Loaded (and possibly resampled) DEM.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"{path}: expected single-band DEM, got {src.count} bands")

        src_cell_size_m = float(abs(src.transform.a))
        if not np.isclose(abs(src.transform.a), abs(src.transform.e)):
            warnings.warn(
                f"{path}: non-square pixels "
                f"({src.transform.a}, {src.transform.e}); using x cell size",
                stacklevel=2,
            )

        out_shape = (src.height, src.width)
        out_transform = src.transform
        if target_cell_size_m is not None:
            if target_cell_size_m <= 0:
                raise ValueError(
                    f"target_cell_size_m must be positive, got {target_cell_size_m}"
                )
            if target_cell_size_m + 1e-9 < src_cell_size_m:
                raise ValueError(
                    f"target_cell_size_m {target_cell_size_m} is finer than the "
                    f"source cell size {src_cell_size_m}; refusing to upsample"
                )
            scale = src_cell_size_m / target_cell_size_m
            new_height = max(1, int(round(src.height * scale)))
            new_width = max(1, int(round(src.width * scale)))
            if (new_height, new_width) != (src.height, src.width):
                out_shape = (new_height, new_width)
                out_transform = src.transform * Affine.scale(
                    src.width / new_width, src.height / new_height
                )

        if out_shape == (src.height, src.width):
            elevation = src.read(1, masked=False).astype(np.float64, copy=True)
            nodata = src.nodata
            if nodata is not None:
                elevation[elevation == nodata] = np.nan
        else:
            elevation = src.read(
                1,
                out_shape=out_shape,
                resampling=Resampling.bilinear,
                masked=False,
            ).astype(np.float64, copy=True)
            # Bilinear resampling treats the nodata sentinel as a
            # regular value and so pollutes interior averages near
            # nodata edges. Re-read the validity mask at the target
            # resolution with nearest-neighbour and stamp NaN through
            # the mask, which is the conventional GDAL-derived recipe.
            if src.nodata is not None:
                valid = (
                    src.read_masks(
                        1, out_shape=out_shape, resampling=Resampling.nearest
                    )
                    > 0
                )
                elevation[~valid] = np.nan

        crs = src.crs
        if crs is None:
            warnings.warn(f"{path}: no CRS on raster", stacklevel=2)
        elif not crs.is_projected:
            warnings.warn(
                f"{path}: CRS {crs.to_string()} is not projected; "
                "computations assume metric units",
                stacklevel=2,
            )

        cell_size_m = float(abs(out_transform.a))

        return DEM(
            elevation_m=elevation,
            transform=out_transform,
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
