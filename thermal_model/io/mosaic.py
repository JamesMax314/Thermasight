"""Mosaic adjacent DEM tiles into a single GeoTIFF.

Wraps :func:`rasterio.merge.merge` and applies the project's on-disk
conventions: ``nodata = -9999.0``, deflate-compressed float32 with a
horizontal-differencing predictor, tiled output for fast random
access. Inputs must share a CRS and a cell size; the function checks
this up front so a misaligned source is reported clearly rather than
producing a silently-wrong mosaic.

The ``rasterio.merge`` "last source wins" rule applies on overlap.
For adjacent EA LIDAR tiles in EPSG:27700 the overlap is at most one
pixel along shared edges and the values agree, so the rule is
effectively a no-op.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import rasterio
from rasterio.merge import merge


def mosaic_dems(
    inputs: Iterable[str | Path],
    output_path: str | Path,
    *,
    nodata: float = -9999.0,
    compress: str | None = "deflate",
    overwrite: bool = False,
) -> Path:
    """Mosaic a set of single-band DEM GeoTIFFs into one raster.

    Parameters
    ----------
    inputs : iterable of str or Path
        Paths to single-band DEM GeoTIFFs. Must share a CRS and cell
        size; positions are taken from each file's affine transform.
    output_path : str or Path
        Where to write the mosaic. Parent directories are created as
        needed.
    nodata : float, default ``-9999.0``
        Output nodata sentinel. Source nodata pixels become this value
        on disk.
    compress : str or None, default ``"deflate"``
        GeoTIFF compression. ``None`` to disable.
    overwrite : bool, default ``False``
        Overwrite ``output_path`` if it exists; otherwise raise
        :class:`FileExistsError`.

    Returns
    -------
    Path
        The resolved output path.
    """
    paths = [Path(p) for p in inputs]
    if not paths:
        raise ValueError("no input paths supplied")
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass overwrite=True to clobber")

    datasets = [rasterio.open(p) for p in paths]
    try:
        first = datasets[0]
        crs = first.crs
        cell_x = first.transform.a
        cell_y = first.transform.e
        for ds in datasets[1:]:
            if ds.crs != crs:
                raise ValueError(
                    f"CRS mismatch: {first.name} has {crs!r}, {ds.name} has {ds.crs!r}"
                )
            if not (
                abs(ds.transform.a - cell_x) < 1e-9
                and abs(ds.transform.e - cell_y) < 1e-9
            ):
                raise ValueError(
                    f"cell size mismatch: {first.name} has "
                    f"({cell_x}, {cell_y}); {ds.name} has "
                    f"({ds.transform.a}, {ds.transform.e})"
                )
            if ds.count != 1:
                raise ValueError(f"{ds.name}: expected single-band DEM, got {ds.count}")

        mosaic, out_transform = merge(datasets, nodata=nodata)
    finally:
        for ds in datasets:
            ds.close()

    band = mosaic[0]  # rasterio.merge returns (bands, h, w)
    height, width = band.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": band.dtype.name,
        "crs": crs,
        "transform": out_transform,
        "nodata": nodata,
    }
    if compress:
        profile["compress"] = compress
        profile["predictor"] = 3 if band.dtype.kind == "f" else 2
        profile["tiled"] = True
        profile["blockxsize"] = 512
        profile["blockysize"] = 512

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(band, 1)

    return output_path
