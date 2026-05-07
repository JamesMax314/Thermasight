"""Command-line entry point.

Subcommands are added in the phase that introduces them. Phase 0 ships only
the entrypoint plumbing so ``python -m thermal_model`` and the
``thermal-model`` console script exist. Phase 1 adds ``preview`` for
quick-look diagnostic plots.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

    from thermal_model.io.dem import DEM

_PREVIEW_CHOICES = ("convergence", "slope", "aspect", "curvature", "heating", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermal-model",
        description="Predict thermal sources and trigger points from LIDAR DEMs.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    preview = subparsers.add_parser(
        "preview",
        help="Render a diagnostic plot of a DEM (hillshade + overlay).",
        description=(
            "Render a hillshade-backed diagnostic plot for a DEM. Overlay one "
            "of slope, aspect, profile curvature, or the inverted-DEM flow "
            "convergence; or 'all' for a 2x2 panel."
        ),
    )
    preview.add_argument(
        "--dem",
        type=Path,
        required=True,
        help="Path to a single-band DEM GeoTIFF.",
    )
    preview.add_argument(
        "--what",
        choices=_PREVIEW_CHOICES,
        default="convergence",
        help="Which overlay to render (default: convergence).",
    )
    preview.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Write a PNG to this path instead of opening a window.",
    )
    preview.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Figure DPI (default: 120).",
    )
    preview.add_argument(
        "--datetime",
        dest="when",
        type=str,
        default=None,
        help=(
            "Timezone-aware ISO timestamp (e.g. '2026-05-06T13:00:00+01:00'). "
            "Required for --what heating; ignored otherwise."
        ),
    )
    preview.add_argument(
        "--lat",
        type=float,
        default=None,
        help=(
            "Latitude in degrees (N-positive). Defaults to the centre of the "
            "DEM, reprojected from the DEM's CRS."
        ),
    )
    preview.add_argument(
        "--lon",
        type=float,
        default=None,
        help=(
            "Longitude in degrees (E-positive). Defaults to the centre of the "
            "DEM, reprojected from the DEM's CRS."
        ),
    )
    preview.add_argument(
        "--elevation",
        type=float,
        default=None,
        help=(
            "Site elevation in metres for the clear-sky model. Defaults to "
            "the median of finite DEM cells."
        ),
    )
    preview.add_argument(
        "--linke-turbidity",
        type=float,
        default=3.0,
        help="Linke turbidity for the Ineichen-Perez clear-sky model (default 3.0).",
    )
    preview.add_argument(
        "--absorptivity",
        type=float,
        default=None,
        help=(
            "Shortwave absorptivity alpha = 1 - albedo. Defaults to the "
            "upland Dales value from docs/DATA.md (0.80)."
        ),
    )
    preview.set_defaults(func=_cmd_preview)

    mosaic = subparsers.add_parser(
        "mosaic",
        help="Stitch a set of adjacent DEM tiles into a single GeoTIFF.",
        description=(
            "Mosaic a set of single-band DEM tiles into one GeoTIFF. All "
            "inputs must share a CRS and cell size. Output uses -9999 "
            "nodata and deflate compression to match the project on-disk "
            "convention."
        ),
    )
    mosaic.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to input DEM tiles (shell globs are expanded by the shell).",
    )
    mosaic.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the mosaicked GeoTIFF.",
    )
    mosaic.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output if it exists.",
    )
    mosaic.set_defaults(func=_cmd_mosaic)
    return parser


def _cmd_preview(args: argparse.Namespace) -> int:
    # Lazy imports so `--help` doesn't pay for matplotlib + rasterio.
    import matplotlib.pyplot as plt

    from thermal_model.io import read_dem
    from thermal_model.viz import (
        plot_aspect,
        plot_convergence,
        plot_heating,
        plot_profile_curvature,
        plot_slope,
    )

    dem = read_dem(args.dem)

    if args.what == "all":
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=args.dpi)
        plot_slope(dem.elevation_m, dem.cell_size_m, ax=axes[0, 0])
        plot_aspect(dem.elevation_m, dem.cell_size_m, ax=axes[0, 1])
        plot_profile_curvature(dem.elevation_m, dem.cell_size_m, ax=axes[1, 0])
        plot_convergence(dem.elevation_m, dem.cell_size_m, ax=axes[1, 1])
        fig.suptitle(str(args.dem))
    elif args.what == "heating":
        when, lat, lon, kwargs = _resolve_heating_args(args, dem)
        fig, ax = plt.subplots(figsize=(9, 8), dpi=args.dpi)
        plot_heating(dem.elevation_m, dem.cell_size_m, when, lat, lon, ax=ax, **kwargs)
    else:
        plotter = {
            "convergence": plot_convergence,
            "slope": plot_slope,
            "aspect": plot_aspect,
            "curvature": plot_profile_curvature,
        }[args.what]
        fig, ax = plt.subplots(figsize=(8, 8), dpi=args.dpi)
        plotter(dem.elevation_m, dem.cell_size_m, ax=ax)

    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return 0


def _resolve_heating_args(
    args: argparse.Namespace, dem: DEM
) -> tuple[datetime, float, float, dict[str, Any]]:
    """Validate and fill in heating-specific args from the DEM context.

    Returns ``(when, latitude_deg, longitude_deg, kwargs_for_plot_heating)``.
    Raises ``SystemExit`` on missing inputs we can't reasonably default.
    """
    from datetime import datetime as _datetime

    from pyproj import Transformer

    from thermal_model.physics import DEFAULT_ABSORPTIVITY

    if args.when is None:
        raise SystemExit(
            "preview --what heating requires --datetime "
            "(e.g. '2026-05-06T13:00:00+01:00')"
        )
    try:
        when = _datetime.fromisoformat(args.when)
    except ValueError as exc:
        raise SystemExit(f"could not parse --datetime {args.when!r}: {exc}") from exc
    if when.tzinfo is None:
        raise SystemExit(
            f"--datetime {args.when!r} is timezone-naive; "
            "include a UTC offset such as '+01:00' or 'Z'"
        )

    lat = args.lat
    lon = args.lon
    if lat is None or lon is None:
        if dem.crs is None:
            raise SystemExit(
                "DEM has no CRS; pass --lat and --lon explicitly for --what heating"
            )
        rows, cols = dem.elevation_m.shape
        centre_x_proj, centre_y_proj = dem.transform * (cols / 2.0, rows / 2.0)
        transformer = Transformer.from_crs(dem.crs, "EPSG:4326", always_xy=True)
        centre_lon, centre_lat = transformer.transform(centre_x_proj, centre_y_proj)
        if lat is None:
            lat = float(centre_lat)
        if lon is None:
            lon = float(centre_lon)

    kwargs: dict[str, object] = {
        "linke_turbidity": float(args.linke_turbidity),
        "absorptivity": float(
            args.absorptivity if args.absorptivity is not None else DEFAULT_ABSORPTIVITY
        ),
    }
    if args.elevation is not None:
        kwargs["elevation_m"] = float(args.elevation)
    return when, float(lat), float(lon), kwargs


def _cmd_mosaic(args: argparse.Namespace) -> int:
    from thermal_model.io import mosaic_dems

    out = mosaic_dems(args.inputs, args.output, overwrite=args.overwrite)
    print(f"wrote {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    func = getattr(args, "func", None)
    if func is None:
        parser.error(f"unknown command: {args.command}")
        return 2
    rc: int = func(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
