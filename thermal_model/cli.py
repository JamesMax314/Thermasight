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

_PREVIEW_CHOICES = ("convergence", "slope", "aspect", "curvature", "all")


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
    preview.set_defaults(func=_cmd_preview)
    return parser


def _cmd_preview(args: argparse.Namespace) -> int:
    # Lazy imports so `--help` doesn't pay for matplotlib + rasterio.
    import matplotlib.pyplot as plt

    from thermal_model.io import read_dem
    from thermal_model.viz import (
        plot_aspect,
        plot_convergence,
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
