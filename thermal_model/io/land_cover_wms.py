"""UKCEH Land Cover Map auto-fetch over the public WMS.

The UKCEH "Land Cover Map 2024 10 m WMS" serves *styled* raster output
only — i.e. PNG/JPEG/TIFF rendered with the 21-class colour palette.
There is no public WCS endpoint advertised, so to get class codes we
fetch the rendered PNG and reverse-map RGB → class via the hardcoded
:data:`UKCEH_LCM_PALETTE` (sampled from the WMS ``GetLegendGraphic``
output for layer ``LC.10m.GB``).

Two known palette collisions in the WMS rendering — these are coastal
classes that don't appear in the inland Yorkshire Dales validation
areas, but the resolver below maps each colour to a single (rock-side)
class code and the operator can disambiguate via a different upstream
source if/when needed:

* ``(204, 179, 0)`` → class 15 (supralittoral rock); class 16
  (supralittoral sediment) shares this colour.
* ``(255, 255, 128)`` → class 17 (littoral rock); class 18 (littoral
  sediment) shares this colour.

Caching: every successful ``GetMap`` is written to
``data/cache/lcm/<layer>/<sha1>.png`` keyed by the canonical request
URL. ``use_cache=False`` (or the ``--no-lcm-cache`` CLI flag) bypasses
both read and write.
"""

from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds

from thermal_model.io.dem import DEM
from thermal_model.io.land_cover import LandCover

#: Default WMS endpoint, layer, style, version. Override via kwargs.
LCM_WMS_DEFAULTS: dict[str, str] = {
    "base_url": (
        "https://catalogue.ceh.ac.uk/maps/688492ef-d9db-43b7-8107-3675c6150568"
    ),
    "layer": "LC.10m.GB",
    "style": "",
    "version": "1.3.0",
    "format": "image/png",
    "crs": "EPSG:27700",
}

#: WMS server-advertised cap on a single GetMap response.
_WMS_MAX_PIXELS: int = 2048

#: Native LCM resolution served by ``LC.10m.GB`` and ``LC.10m.NI``.
LCM_NATIVE_CELL_SIZE_M: float = 10.0

#: Hardcoded RGB → UKCEH class-code mapping for the "traditional colour
#: scheme" returned by ``LC.10m.GB`` / ``LC.10m.NI``. Sampled from the
#: live ``GetLegendGraphic`` (2026-05-10) and cross-checked against the
#: legend labels. Two coastal palette collisions documented in the
#: module docstring.
UKCEH_LCM_PALETTE: dict[tuple[int, int, int], int] = {
    (255, 0, 0): 1,  # Broadleaved, mixed & yew woodland
    (0, 102, 0): 2,  # Coniferous woodland
    (115, 38, 0): 3,  # Arable and horticulture
    (0, 255, 0): 4,  # Improved grassland
    (127, 229, 127): 5,  # Neutral grassland
    (112, 168, 0): 6,  # Calcareous grassland
    (153, 129, 0): 7,  # Acid grassland
    (255, 255, 0): 8,  # Fen, marsh and swamp
    (128, 26, 128): 9,  # Heather
    (230, 140, 166): 10,  # Heather grassland
    (0, 128, 115): 11,  # Bog
    (210, 210, 255): 12,  # Inland rock
    (0, 0, 128): 13,  # Saltwater
    (0, 0, 255): 14,  # Freshwater
    (204, 179, 0): 15,  # Supralittoral rock (= 16 sediment in WMS palette)
    (255, 255, 128): 17,  # Littoral rock (= 18 sediment in WMS palette)
    (128, 128, 255): 19,  # Saltmarsh
    (0, 0, 0): 20,  # Urban
    (128, 128, 128): 21,  # Suburban
}


def fetch_lcm_for_dem(
    reference: DEM,
    *,
    layer: str = LCM_WMS_DEFAULTS["layer"],
    cache_dir: str | Path = Path("data/cache/lcm"),
    use_cache: bool = True,
    timeout_s: float = 60.0,
    base_url: str = LCM_WMS_DEFAULTS["base_url"],
) -> LandCover:
    """Fetch UKCEH LCM coverage of ``reference``'s footprint over WMS.

    Issues a single ``GetMap`` request at the LCM native 10 m resolution
    in EPSG:27700 spanning ``reference``'s bounding box, decodes the
    PNG, reverse-maps RGB → class via :data:`UKCEH_LCM_PALETTE`, and
    returns a :class:`LandCover`. The DEM does not need to be in
    EPSG:27700 itself — the request is always issued in BNG so we get
    the same projection as the operator-facing coastal layers, and the
    downstream :func:`absorptivity_from_land_cover` reprojects onto the
    DEM grid.

    Parameters
    ----------
    reference : DEM
        DEM whose footprint defines the WMS request bbox.
    layer : str, default ``"LC.10m.GB"``
        WMS layer name. Use ``"LC.10m.NI"`` for Northern Ireland.
    cache_dir : str or Path, default ``"data/cache/lcm"``
        Directory for the on-disk PNG cache. Created if missing.
    use_cache : bool, default ``True``
        Read from / write to the cache. ``False`` always fetches and
        skips writing the cache file.
    timeout_s : float, default 60.0
        Socket timeout for the WMS request.
    base_url : str
        WMS endpoint base (excluding query string).

    Returns
    -------
    LandCover
        Categorical raster at 10 m resolution covering ``reference``'s
        footprint in EPSG:27700.

    Raises
    ------
    NotImplementedError
        If the bbox at 10 m exceeds the WMS-advertised
        ``MaxWidth/MaxHeight = 2048`` and chunked fetching would be
        required. (Chunked fetch is follow-up scope.)
    RuntimeError
        If the WMS returns a non-PNG response or the PNG decodes to
        something other than 3 bands.
    """
    if reference.crs is None:
        raise ValueError("reference.crs is None; cannot fetch LCM without a DEM CRS")

    bbox_27700 = _dem_bounds_in_bng(reference)
    width, height = _request_dims_at_native_res(bbox_27700)
    if width > _WMS_MAX_PIXELS or height > _WMS_MAX_PIXELS:
        raise NotImplementedError(
            f"WMS request would be {width}×{height}; the server caps at "
            f"{_WMS_MAX_PIXELS}×{_WMS_MAX_PIXELS}. Reduce the DEM bounds "
            "or downsample the DEM. Chunked WMS fetch is follow-up scope."
        )

    url = _build_get_map_url(
        base_url=base_url,
        layer=layer,
        bbox_27700=bbox_27700,
        width=width,
        height=height,
        version=LCM_WMS_DEFAULTS["version"],
        crs=LCM_WMS_DEFAULTS["crs"],
        fmt=LCM_WMS_DEFAULTS["format"],
        style=LCM_WMS_DEFAULTS["style"],
    )

    cache_path = Path(cache_dir) / layer / f"{_url_sha1(url)}.png"
    png_bytes: bytes
    if use_cache and cache_path.exists():
        png_bytes = cache_path.read_bytes()
    else:
        png_bytes = _http_get(url, timeout_s=timeout_s)
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(png_bytes)

    classes = _decode_png_to_classes(png_bytes)
    if classes.shape != (height, width):
        raise RuntimeError(
            f"WMS returned {classes.shape[1]}×{classes.shape[0]} pixels; "
            f"expected {width}×{height}. Server may have changed."
        )

    minx, miny, maxx, maxy = bbox_27700
    transform: Affine = from_bounds(
        west=minx, south=miny, east=maxx, north=maxy, width=width, height=height
    )
    return LandCover(
        classes=classes,
        transform=transform,
        crs=CRS.from_epsg(27700),
        cell_size_m=LCM_NATIVE_CELL_SIZE_M,
    )


def _dem_bounds_in_bng(reference: DEM) -> tuple[float, float, float, float]:
    """Return ``(minx, miny, maxx, maxy)`` in EPSG:27700 for the DEM footprint."""
    rows, cols = reference.shape
    # Corners of the raster footprint (pixel-edge coordinates).
    corners = [
        reference.transform * (0.0, 0.0),
        reference.transform * (float(cols), 0.0),
        reference.transform * (0.0, float(rows)),
        reference.transform * (float(cols), float(rows)),
    ]
    src_epsg = _safe_epsg(reference.crs)
    if src_epsg == 27700:
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
    else:
        from pyproj import Transformer

        transformer = Transformer.from_crs(reference.crs, "EPSG:27700", always_xy=True)
        xs = []
        ys = []
        for x, y in corners:
            tx, ty = transformer.transform(x, y)
            xs.append(tx)
            ys.append(ty)
    return (min(xs), min(ys), max(xs), max(ys))


def _request_dims_at_native_res(
    bbox_27700: tuple[float, float, float, float],
) -> tuple[int, int]:
    """Pixel width/height at 10 m for the given EPSG:27700 bbox."""
    minx, miny, maxx, maxy = bbox_27700
    width = max(1, int(round((maxx - minx) / LCM_NATIVE_CELL_SIZE_M)))
    height = max(1, int(round((maxy - miny) / LCM_NATIVE_CELL_SIZE_M)))
    return width, height


def _build_get_map_url(
    *,
    base_url: str,
    layer: str,
    bbox_27700: tuple[float, float, float, float],
    width: int,
    height: int,
    version: str,
    crs: str,
    fmt: str,
    style: str,
) -> str:
    """Build a WMS 1.3.0 GetMap URL for the given bbox + dims.

    For WMS 1.3.0 in EPSG:27700, BBOX axis order is (minx, miny, maxx,
    maxy) — easting/northing — because BNG's native axis order is x,y.
    (1.3.0's "swap to lat,lon for geographic CRSes" rule does *not*
    apply to projected metric CRSes like EPSG:27700.)
    """
    minx, miny, maxx, maxy = bbox_27700
    params = {
        "service": "WMS",
        "version": version,
        "request": "GetMap",
        "layers": layer,
        "styles": style,
        "crs": crs,
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "width": str(width),
        "height": str(height),
        "format": fmt,
        "transparent": "false",
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def _url_sha1(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _http_get(url: str, *, timeout_s: float) -> bytes:
    """Fetch ``url`` with a short user-agent string. stdlib only."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "thermasight/lcm-fetch (+research)"}
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body: bytes = resp.read()
    return body


def _decode_png_to_classes(png_bytes: bytes) -> np.ndarray:
    """Decode a PNG payload into an ``int16`` class-code raster.

    Reads the PNG via rasterio (pulls in GDAL's PNG driver), takes the
    first three bands as RGB, and reverse-maps each pixel against
    :data:`UKCEH_LCM_PALETTE` using a packed-uint32 LUT — single
    vectorised pass, no per-cell loop.
    """
    with rasterio.open(BytesIO(png_bytes)) as src:
        if src.count < 3:
            raise RuntimeError(f"WMS PNG has {src.count} bands; expected at least RGB")
        r = src.read(1).astype(np.uint32)
        g = src.read(2).astype(np.uint32)
        b = src.read(3).astype(np.uint32)

    packed = (r << 16) | (g << 8) | b

    # Sparse LUT keyed by packed RGB. 16 MiB int16 buffer is acceptable;
    # alternatively a dict-backed loop, but the LUT keeps the lookup
    # branch-free and vectorised.
    lut = np.full(1 << 24, _LCM_PALETTE_NODATA, dtype=np.int16)
    for (rr, gg, bb), code in UKCEH_LCM_PALETTE.items():
        idx = (rr << 16) | (gg << 8) | bb
        lut[idx] = code

    classes = lut[packed]

    unknown = int((classes == _LCM_PALETTE_NODATA).sum())
    if unknown:
        warnings.warn(
            f"{unknown} pixel(s) in the WMS PNG did not match any UKCEH "
            "palette entry; tagging as nodata. Likely PNG anti-aliasing "
            "edges or a palette change on the server.",
            stacklevel=2,
        )
    # Map our packed-RGB nodata sentinel to the LandCover sentinel used
    # downstream (`-1`). They're already the same int16 value, but keep
    # the substitution explicit for future-proofing.
    classes = np.where(
        classes == _LCM_PALETTE_NODATA,
        np.int16(-1),
        classes,
    ).astype(np.int16)
    return classes


# Sentinel for "RGB matched no palette entry" while decoding. Same
# numeric value as the LandCover sentinel so the post-decode `where`
# is a no-op in practice; kept as a separate name for clarity.
_LCM_PALETTE_NODATA: int = -1


def _safe_epsg(crs: Any) -> int | None:
    if crs is None:
        return None
    try:
        epsg = crs.to_epsg()
    except AttributeError:
        return None
    return int(epsg) if epsg is not None else None
