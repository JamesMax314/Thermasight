"""UKCEH Land Cover Map ingestion + class-code → absorptivity lookup.

Conventions match :mod:`thermal_model.io.dem`:

* CRS is read from rasterio and a mismatch warns rather than errors —
  consistent with ``read_dem``'s tone (CLAUDE.md §6 rule 3).
* Internally the categorical raster carries the sentinel ``-1`` for
  nodata (NaN doesn't fit in an integer dtype). The downstream
  :func:`absorptivity_from_land_cover` then translates that into the
  ``unknown_fill`` α — *not* into NaN — because a sliver of unclassified
  land must not silently zero the heating-weighted leaky-bucket routing
  in :func:`thermal_model.physics.run_model`. Wet ground is dead ground;
  *no-data ground* is just unknown ground (CLAUDE.md §5).
* Lookup is fully vectorised via a 256-entry LUT; no per-cell Python
  loops (CLAUDE.md §6 rule 2).

The production 21-class UKCEH lookup ``UKCEH_LCM_ABSORPTIVITY`` is
intentionally empty until the operator authors the values. A minimal
:data:`DALES_LCM_ABSORPTIVITY` covering the inland-Yorkshire surfaces
that actually appear in our validation areas is provided for tests and
the Phase 4 Mallerstang re-render.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from thermal_model.io.dem import DEM
from thermal_model.physics.heating import DEFAULT_ABSORPTIVITY

#: Internal nodata sentinel for categorical class rasters. Chosen to be
#: non-negative-int-safe and outside the 0–255 byte range so it can't
#: collide with a UKCEH class code.
_LCM_NODATA: int = -1


#: Production UKCEH LCM 2024 class-code → absorptivity α lookup.
#: **Deliberately empty** — the operator authors the per-class α values;
#: until then every cell falls through to the ``unknown_fill`` default
#: (``DEFAULT_ABSORPTIVITY``, 0.80). See ``docs/DATA.md`` § "UKCEH LCM
#: class-code mapping".
UKCEH_LCM_ABSORPTIVITY: dict[int, float] = {}


#: Minimal Dales-focused class-code → α lookup. Used by the Phase 4
#: validation render and by tests; not the production default. Values
#: come from ``docs/DATA.md`` and are starting estimates that the
#: operator may tune.
#:
#: Codes follow UKCEH LCM 2024:
#:   4  improved grassland
#:   7  acid grassland   (treated as the dry-grass/heather Dales default)
#:   9  heather
#:   11 bog              (the load-bearing entry — wet ground is dead ground)
#:   12 inland rock
#:   14 freshwater
#:   20 urban
#:   21 suburban
DALES_LCM_ABSORPTIVITY: dict[int, float] = {
    4: 0.75,
    7: 0.80,
    9: 0.80,
    11: 0.40,
    12: 0.85,
    14: 0.05,
    20: 0.85,
    21: 0.85,
}


@dataclass(frozen=True)
class LandCover:
    """A categorical land-cover raster loaded into memory.

    Attributes
    ----------
    classes : np.ndarray
        2-D integer class-code array. Nodata is the sentinel ``-1``;
        valid UKCEH class codes are ``1..21``.
    transform : Affine
        Affine transform mapping pixel (col, row) to projected (x, y).
    crs : rasterio.crs.CRS | None
        Coordinate reference system. ``None`` if the source file lacks
        one (a warning is emitted on read in that case).
    cell_size_m : float
        Cell size in metres (assumes square pixels in a metric CRS).
    """

    classes: np.ndarray
    transform: Affine
    crs: Any
    cell_size_m: float

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self.classes.shape
        return rows, cols


def read_land_cover(path: str | Path) -> LandCover:
    """Read a single-band categorical UKCEH LCM GeoTIFF into a :class:`LandCover`.

    Source nodata is converted to the internal sentinel ``-1``. The CRS
    is checked and a warning is emitted if it is missing or not metric.

    Parameters
    ----------
    path : str or Path
        Path to a single-band integer-coded LCM GeoTIFF.

    Returns
    -------
    LandCover
        Loaded categorical raster.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(
                f"{path}: expected single-band land-cover raster, got {src.count} bands"
            )
        if not np.isclose(abs(src.transform.a), abs(src.transform.e)):
            warnings.warn(
                f"{path}: non-square pixels "
                f"({src.transform.a}, {src.transform.e}); using x cell size",
                stacklevel=2,
            )

        raw = src.read(1, masked=False)
        # Cast to int16 unconditionally so the nodata sentinel `-1` fits
        # alongside the 1..21 UKCEH codes regardless of the on-disk dtype
        # (uint8 is most common for LCM products).
        classes = raw.astype(np.int16, copy=True)
        nodata = src.nodata
        if nodata is not None:
            classes[raw == nodata] = _LCM_NODATA

        crs = src.crs
        if crs is None:
            warnings.warn(f"{path}: no CRS on raster", stacklevel=2)
        elif not crs.is_projected:
            warnings.warn(
                f"{path}: CRS {crs.to_string()} is not projected; "
                "computations assume metric units",
                stacklevel=2,
            )

        cell_size_m = float(abs(src.transform.a))

        return LandCover(
            classes=classes,
            transform=src.transform,
            crs=crs,
            cell_size_m=cell_size_m,
        )


def absorptivity_from_land_cover(
    land_cover: LandCover,
    reference: DEM,
    *,
    lookup: Mapping[int, float] = UKCEH_LCM_ABSORPTIVITY,
    unknown_fill: float = DEFAULT_ABSORPTIVITY,
) -> np.ndarray:
    """Resample ``land_cover`` to ``reference``'s grid and look up α per cell.

    Pipeline:

    1. Nearest-neighbour reproject ``land_cover.classes`` onto the DEM's
       transform / CRS / shape using :func:`rasterio.warp.reproject`.
       Nearest-neighbour is the only correct choice for categorical
       data — bilinear or cubic would invent class codes that don't
       exist in the lookup.
    2. Build a 256-entry LUT initialised to ``unknown_fill``; overwrite
       indices that appear in ``lookup`` with their α values. The LCM
       sentinel ``-1`` and any class code ``>= 256`` map to
       ``unknown_fill`` as well — never to NaN. This is the
       routing-preservation contract described in the module docstring.
    3. Cells where ``reference.elevation_m`` is NaN propagate as NaN α.

    A ``UserWarning`` is emitted if (a) the CRSes disagree on EPSG code,
    or (b) the reprojected raster contains class codes that the
    ``lookup`` doesn't cover.

    Parameters
    ----------
    land_cover : LandCover
        Source categorical raster.
    reference : DEM
        Target grid; the returned α array matches ``reference.shape``,
        ``reference.transform``, and ``reference.crs``.
    lookup : Mapping[int, float], default ``UKCEH_LCM_ABSORPTIVITY``
        Class code → α mapping. Defaults to the empty production LUT —
        i.e. every cell falls through to ``unknown_fill``. Pass
        :data:`DALES_LCM_ABSORPTIVITY` for a non-trivial Dales-focused
        run.
    unknown_fill : float, default ``DEFAULT_ABSORPTIVITY`` (0.80)
        α value used for class codes that are not in ``lookup`` and for
        LCM nodata cells inside the reference DEM's footprint. Must lie
        in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Float64 absorptivity array, ``reference.shape``, values in
        ``[0, 1]`` with NaN where the DEM is NaN.
    """
    if not 0.0 <= float(unknown_fill) <= 1.0:
        raise ValueError(f"unknown_fill must be in [0, 1], got {unknown_fill}")
    for code, alpha_value in lookup.items():
        if not 0.0 <= float(alpha_value) <= 1.0:
            raise ValueError(
                f"lookup α for class {code} must be in [0, 1], got {alpha_value}"
            )

    src_epsg = _epsg_or_none(land_cover.crs)
    dst_epsg = _epsg_or_none(reference.crs)
    if src_epsg is not None and dst_epsg is not None and src_epsg != dst_epsg:
        warnings.warn(
            f"land cover CRS EPSG:{src_epsg} differs from DEM CRS "
            f"EPSG:{dst_epsg}; reprojecting via nearest-neighbour",
            stacklevel=2,
        )
    elif (src_epsg is None) != (dst_epsg is None):
        warnings.warn(
            "one of land_cover.crs / reference.crs is None; reprojection "
            "may be unreliable",
            stacklevel=2,
        )

    dst = np.full(reference.shape, _LCM_NODATA, dtype=np.int16)
    reproject(
        source=land_cover.classes,
        destination=dst,
        src_transform=land_cover.transform,
        src_crs=land_cover.crs,
        dst_transform=reference.transform,
        dst_crs=reference.crs,
        src_nodata=_LCM_NODATA,
        dst_nodata=_LCM_NODATA,
        resampling=Resampling.nearest,
    )

    # Build a 256-entry LUT keyed by uint8 class code. UKCEH LCM uses
    # codes 1..21; we accommodate the full byte range so an unexpected
    # value (rare, but possible from a third-party LCM) doesn't crash.
    # Negative sentinels and codes >= 256 are filtered out before the
    # LUT lookup.
    lut = np.full(256, float(unknown_fill), dtype=np.float64)
    for code, alpha_value in lookup.items():
        if 0 <= int(code) < 256:
            lut[int(code)] = float(alpha_value)

    in_range = (dst >= 0) & (dst < 256)
    safe = np.where(in_range, dst, 0).astype(np.uint8)
    alpha: np.ndarray = np.where(in_range, lut[safe], float(unknown_fill))

    # Count and warn about unknown classes (anything not in the lookup,
    # including the nodata sentinel).
    known_codes = np.array(
        [int(c) for c in lookup.keys() if 0 <= int(c) < 256], dtype=np.int64
    )
    if known_codes.size:
        # `np.isin` against a small set is fast.
        is_known = np.isin(dst, known_codes)
    else:
        is_known = np.zeros_like(dst, dtype=bool)
    unknown_count = int((~is_known).sum())
    if unknown_count > 0 and known_codes.size:
        warnings.warn(
            f"{unknown_count} cell(s) had class codes outside the "
            f"absorptivity lookup; substituting α={unknown_fill}",
            stacklevel=2,
        )

    # NaN propagation from the reference DEM.
    dem_nan = ~np.isfinite(reference.elevation_m)
    alpha = np.where(dem_nan, np.nan, alpha)

    return alpha


def _epsg_or_none(crs: Any) -> int | None:
    """Return the EPSG code as ``int`` if known, else ``None``."""
    if crs is None:
        return None
    try:
        epsg = crs.to_epsg()
    except AttributeError:
        return None
    return int(epsg) if epsg is not None else None
