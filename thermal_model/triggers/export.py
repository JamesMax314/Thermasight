"""KMZ export of clustered trigger points.

The KMZ is the project's deliverable for XCTrack / SeeYou / Google
Earth (``CLAUDE.md`` §1). Cluster centroids in raster (row, col) →
projected (x, y) via the DEM's affine transform → WGS84 (lon, lat)
via :class:`pyproj.Transformer` → KMZ via :mod:`simplekml`.

Heavy imports (``simplekml``, ``pyproj``) are deferred to call time so
``import thermal_model.triggers`` stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rasterio.transform import Affine

    from thermal_model.triggers.cluster import TriggerPoint


def write_kmz(
    points: list[TriggerPoint],
    path: str | Path,
    *,
    transform: Affine,
    crs: Any,
    name: str = "Trigger points",
) -> Path:
    """Write a list of :class:`TriggerPoint` to a KMZ.

    Each point is placed at the WGS84 longitude/latitude derived from
    its raster centroid. The placemark name is the rank (``"1"``,
    ``"2"``, ...) and the description carries the mean strength and
    cluster size so a pilot loading the file can read both numbers off
    the placemark.

    Parameters
    ----------
    points : list of TriggerPoint
        Output of :func:`thermal_model.triggers.cluster_triggers`.
        Should already be ranked by mean strength descending; this
        function does not re-sort.
    path : str or Path
        Output ``.kmz`` path. Parent directory is created if needed.
    transform : rasterio.transform.Affine
        Affine transform from raster (col, row) → projected (x, y).
        Take this from the ``DEM`` returned by
        :func:`thermal_model.io.read_dem`.
    crs : rasterio.crs.CRS or str
        CRS of the projected coordinates produced by ``transform``.
        Anything :class:`pyproj.Transformer` can read (an
        ``rasterio.crs.CRS``, an EPSG code string, a WKT, ...).
        Required — KMZ is WGS84 by definition, so we must know what
        we're transforming *from*.
    name : str, default "Trigger points"
        Document and folder name in the KMZ.

    Returns
    -------
    Path
        The path that was written.

    Raises
    ------
    ValueError
        If ``crs`` is ``None`` (we cannot reproject without one) or
        if ``points`` is empty.
    """
    import simplekml
    from pyproj import Transformer

    if crs is None:
        raise ValueError(
            "crs is None; cannot reproject to WGS84. The DEM must carry a CRS "
            "for KMZ export."
        )
    if not points:
        raise ValueError("points is empty; nothing to export")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    kml = simplekml.Kml()
    kml.document.name = name
    folder = kml.newfolder(name=name)

    for rank, point in enumerate(points, start=1):
        # Affine maps (col, row) -> (x, y). The raster centroid is
        # in (row, col) order; pass (col + 0.5, row + 0.5) to land at
        # the centre of the cell that the centroid reports.
        x_proj, y_proj = transform * (point.col + 0.5, point.row + 0.5)
        lon, lat = transformer.transform(x_proj, y_proj)

        description_lines = [
            f"Mean trigger strength: {point.mean_strength:.3f}",
            f"Cluster size: {point.n_cells} cells",
        ]
        if point.mean_cycle_period_s is not None:
            tau = float(point.mean_cycle_period_s)
            # Render compactly in the most pilot-readable unit.
            if tau < 120.0:
                tau_str = f"{tau:.0f} s"
            elif tau < 3600.0:
                tau_str = f"{tau / 60.0:.1f} min"
            else:
                tau_str = f"{tau / 3600.0:.1f} hr"
            description_lines.append(f"Cycle period: {tau_str}")

        placemark = folder.newpoint(
            name=str(rank),
            coords=[(float(lon), float(lat))],
            description="\n".join(description_lines),
        )
        # Render rank-1 brightest; fade with rank.
        placemark.style.iconstyle.scale = 1.0
        placemark.style.iconstyle.color = simplekml.Color.red

    kml.savekmz(str(out))
    return out
