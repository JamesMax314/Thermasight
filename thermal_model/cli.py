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

_PREVIEW_CHOICES = (
    "convergence",
    "slope",
    "aspect",
    "curvature",
    "heating",
    "trigger",
    "weighted-convergence",
    "leak",
    "cycle-period",
    "all",
)
_WIND_REQUIRING_CHOICES = (
    "trigger",
    "weighted-convergence",
    "leak",
    "cycle-period",
)


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
        "--resolution",
        type=float,
        default=None,
        metavar="METRES",
        help=(
            "Target cell size in metres for diagnostic plotting. If coarser "
            "than the source DEM, the file is bilinearly resampled on read "
            "before the pipeline runs. Useful for whole-mosaic previews where "
            "the cast-shadow march would otherwise be slow at native "
            "resolution. Default: native source resolution."
        ),
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
    preview.add_argument(
        "--wind-from",
        type=float,
        default=None,
        metavar="DEG",
        help=(
            "Meteorological wind direction (degrees FROM, 0=N). Required "
            "for --what trigger and --what weighted-convergence."
        ),
    )
    preview.add_argument(
        "--wind-speed",
        type=float,
        default=None,
        metavar="MS",
        help=(
            "Wind speed in m/s. Required for --what trigger and "
            "--what weighted-convergence."
        ),
    )
    preview.add_argument(
        "--wind-tilt-k",
        type=float,
        default=0.03,
        help=(
            "Wind-tilt coefficient (s/m) for the inverted-DEM ramp. "
            "Default 0.03 (moderate). See docs/MODEL.md §3 for tuning."
        ),
    )
    preview.add_argument(
        "--smoothing-sigma",
        type=float,
        default=10.0,
        metavar="METRES",
        help=(
            "Gaussian smoothing scale (metres) applied to the DEM before "
            "wind tilt and flow routing. Default 10 m (CLAUDE.md §2)."
        ),
    )
    preview.add_argument(
        "--curvature-smoothing-sigma",
        type=float,
        default=10.0,
        metavar="METRES",
        help=(
            "Gaussian sigma (metres) for the DEM smoothing applied before "
            "curvature and slope are derived for the leaky shape functions. "
            "Suppresses single-cell LIDAR speckle in κ⁺. Independent of "
            "--smoothing-sigma (which only affects routing). Default 10 m. "
            "Pass 0 to disable (raw curvature feeds the shape functions, "
            "reproducing pre-2026-05-09 behaviour)."
        ),
    )
    preview.add_argument(
        "--min-slope",
        type=float,
        default=2.5,
        metavar="DEG",
        help=(
            "Slope (degrees) below which a cell forwards everything (no leak). "
            "Default 2.5°. See docs/MODEL.md §11."
        ),
    )
    preview.add_argument(
        "--slope-scale",
        type=float,
        default=15.0,
        metavar="DEG",
        help="Slope reference scale (degrees) for the leak shape. Default 15°.",
    )
    preview.add_argument(
        "--kappa-ref",
        type=float,
        default=0.005,
        metavar="ONEPERMETRE",
        help=(
            "Reference profile-curvature scale (1/m) for the leak shape. "
            "Default 0.005 1/m."
        ),
    )
    preview.add_argument(
        "--q-ref",
        type=float,
        default=1.0e6,
        metavar="JM2",
        help=(
            "Reference buoyancy storage capacity (J/m²) for the cycle period. "
            "Default 1e6 J/m²."
        ),
    )
    preview.add_argument(
        "--f-min",
        type=float,
        default=0.15,
        help="Skimming floor on the drain fraction (default 0.15).",
    )
    preview.add_argument(
        "--f-max",
        type=float,
        default=1.0,
        help="Maximum drain fraction (default 1.0).",
    )
    preview.add_argument(
        "--no-resolve-flats",
        dest="resolve_flats",
        action="store_false",
        help=(
            "Skip Garbrecht-Martz flat resolution between pit-fill and "
            "flow accumulation. Default: enabled (recommended). Disabling "
            "speeds up large-mosaic iteration but reintroduces the "
            "parallel-streak artefact on flat plateaus."
        ),
    )
    preview.set_defaults(func=_cmd_preview, resolve_flats=True)

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

    run = subparsers.add_parser(
        "run",
        help="Run the Phase 3 trigger-prediction pipeline on a DEM tile.",
        description=(
            "Run the full ground-level trigger pipeline (smooth → wind "
            "tilt → heating-weighted D∞ accumulation → curvature & slope "
            "filter) and write a trigger-potential GeoTIFF. Optionally "
            "cluster the result and write a KMZ of trigger points."
        ),
    )
    run.add_argument("--dem", type=Path, required=True, help="Input DEM GeoTIFF.")
    run.add_argument(
        "--resolution",
        type=float,
        default=None,
        metavar="METRES",
        help=(
            "Target cell size in metres. If coarser than the source, the "
            "DEM is bilinearly resampled on read before the pipeline runs. "
            "Useful for whole-mosaic runs where 1 m native is too slow. "
            "Default: native source resolution."
        ),
    )
    run.add_argument(
        "--datetime",
        dest="when",
        type=str,
        required=True,
        help="Timezone-aware ISO timestamp (e.g. '2026-05-06T13:00:00+01:00').",
    )
    run.add_argument(
        "--wind-from",
        type=float,
        required=True,
        metavar="DEG",
        help="Meteorological wind direction (degrees FROM, 0=N).",
    )
    run.add_argument(
        "--wind-speed",
        type=float,
        required=True,
        metavar="MS",
        help="Wind speed in m/s.",
    )
    run.add_argument(
        "--wind-tilt-k",
        type=float,
        default=0.03,
        help=(
            "Wind-tilt coefficient (s/m). Default 0.03 (moderate). "
            "See docs/MODEL.md §3."
        ),
    )
    run.add_argument(
        "--smoothing-sigma",
        type=float,
        default=10.0,
        metavar="METRES",
        help="Gaussian smoothing scale (m) before flow routing. Default 10.",
    )
    run.add_argument(
        "--curvature-smoothing-sigma",
        type=float,
        default=10.0,
        metavar="METRES",
        help=(
            "Gaussian sigma (m) for the DEM smoothing applied before curvature "
            "and slope feed the leaky shape functions. Suppresses single-cell "
            "LIDAR speckle in κ⁺. Independent of --smoothing-sigma. Default 10. "
            "Pass 0 to reproduce pre-2026-05-09 behaviour."
        ),
    )
    run.add_argument(
        "--min-slope",
        type=float,
        default=2.5,
        metavar="DEG",
        help=(
            "Slope (degrees) below which a cell forwards everything (no leak). "
            "Default 2.5°. See docs/MODEL.md §11."
        ),
    )
    run.add_argument(
        "--slope-scale",
        type=float,
        default=15.0,
        metavar="DEG",
        help="Slope reference scale (degrees) for the leak shape. Default 15°.",
    )
    run.add_argument(
        "--kappa-ref",
        type=float,
        default=0.005,
        metavar="ONEPERMETRE",
        help=(
            "Reference profile-curvature scale (1/m) for the leak shape. "
            "Default 0.005 1/m."
        ),
    )
    run.add_argument(
        "--q-ref",
        type=float,
        default=1.0e6,
        metavar="JM2",
        help=(
            "Reference buoyancy storage capacity (J/m²) for the cycle period. "
            "Default 1e6 J/m²."
        ),
    )
    run.add_argument(
        "--f-min",
        type=float,
        default=0.15,
        help="Skimming floor on the drain fraction (default 0.15).",
    )
    run.add_argument(
        "--f-max",
        type=float,
        default=1.0,
        help="Maximum drain fraction (default 1.0).",
    )
    run.add_argument(
        "--absorptivity",
        type=float,
        default=None,
        help="Shortwave absorptivity alpha. Default 0.80 (upland Dales).",
    )
    run.add_argument(
        "--linke-turbidity",
        type=float,
        default=3.0,
        help="Linke turbidity for the clear-sky model (default 3.0).",
    )
    run.add_argument(
        "--lat", type=float, default=None, help="Latitude (defaults to DEM centre)."
    )
    run.add_argument(
        "--lon", type=float, default=None, help="Longitude (defaults to DEM centre)."
    )
    run.add_argument(
        "--elevation",
        type=float,
        default=None,
        help="Site elevation (m) for the clear-sky model. Defaults to DEM median.",
    )
    run.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output trigger-potential GeoTIFF path.",
    )
    run.add_argument(
        "--kmz",
        type=Path,
        default=None,
        help="Optional KMZ output for clustered trigger points.",
    )
    run.add_argument(
        "--leak-out",
        type=Path,
        default=None,
        help=(
            "Optional GeoTIFF for the absolute trigger leak (W/m²). The "
            "primary --out raster is rank-normalised in [0, 1] for "
            "backward-compatible display; this flag exposes the absolute "
            "leak field for cross-tile comparison."
        ),
    )
    run.add_argument(
        "--cycle-period-out",
        type=Path,
        default=None,
        help=(
            "Optional GeoTIFF for the buoyancy cycle period τ (seconds). "
            "Cells with no triggering carry +inf — translated to the "
            "raster nodata sentinel on write."
        ),
    )
    run.add_argument(
        "--cluster-quantile",
        type=float,
        default=0.95,
        help="Trigger-potential percentile threshold for clustering (default 0.95).",
    )
    run.add_argument(
        "--min-cluster-cells",
        type=int,
        default=3,
        help="Minimum cells in a trigger cluster (default 3).",
    )
    run.add_argument(
        "--no-resolve-flats",
        dest="resolve_flats",
        action="store_false",
        help=(
            "Skip Garbrecht-Martz flat resolution between pit-fill and "
            "flow accumulation. Default: enabled. Disabling speeds up "
            "large-mosaic iteration at the cost of streak artefacts."
        ),
    )
    run.set_defaults(func=_cmd_run, resolve_flats=True)

    return parser


def _cmd_preview(args: argparse.Namespace) -> int:
    # Lazy imports so `--help` doesn't pay for matplotlib + rasterio.
    import matplotlib.pyplot as plt

    from thermal_model.io import read_dem
    from thermal_model.viz import (
        plot_aspect,
        plot_convergence,
        plot_cycle_period,
        plot_heating,
        plot_leak,
        plot_profile_curvature,
        plot_slope,
        plot_trigger_potential,
        plot_weighted_convergence,
    )

    dem = read_dem(args.dem, target_cell_size_m=args.resolution)

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
    elif args.what in _WIND_REQUIRING_CHOICES:
        when, lat, lon, kwargs = _resolve_heating_args(args, dem)
        if args.wind_from is None or args.wind_speed is None:
            raise SystemExit(
                f"preview --what {args.what} requires --wind-from and --wind-speed"
            )
        kwargs.update(
            {
                "wind_from_deg": float(args.wind_from),
                "wind_speed_ms": float(args.wind_speed),
                "wind_tilt_k": float(args.wind_tilt_k),
                "smoothing_sigma_m": float(args.smoothing_sigma),
                "curvature_smoothing_sigma_m": float(args.curvature_smoothing_sigma),
                "min_slope_deg": float(args.min_slope),
                "resolve_flats": bool(args.resolve_flats),
            }
        )
        # leak / cycle-period plotters accept the new leak-shape params;
        # the older trigger / weighted-convergence plotters do not, so
        # only pass them through for the relevant plotters.
        if args.what in ("leak", "cycle-period"):
            kwargs.update(
                {
                    "slope_scale_deg": float(args.slope_scale),
                    "kappa_ref": float(args.kappa_ref),
                    "q_ref": float(args.q_ref),
                    "f_min": float(args.f_min),
                    "f_max": float(args.f_max),
                }
            )
        wind_plotter = {
            "trigger": plot_trigger_potential,
            "weighted-convergence": plot_weighted_convergence,
            "leak": plot_leak,
            "cycle-period": plot_cycle_period,
        }[args.what]
        fig, ax = plt.subplots(figsize=(9, 8), dpi=args.dpi)
        wind_plotter(dem.elevation_m, dem.cell_size_m, when, lat, lon, ax=ax, **kwargs)
    else:
        terrain_plotter = {
            "convergence": plot_convergence,
            "slope": plot_slope,
            "aspect": plot_aspect,
            "curvature": plot_profile_curvature,
        }[args.what]
        fig, ax = plt.subplots(figsize=(8, 8), dpi=args.dpi)
        terrain_plotter(dem.elevation_m, dem.cell_size_m, ax=ax)

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


def _cmd_run(args: argparse.Namespace) -> int:
    from thermal_model.io import read_dem
    from thermal_model.io.dem import write_raster_like
    from thermal_model.physics import run_model
    from thermal_model.triggers import cluster_triggers, write_kmz

    dem = read_dem(args.dem, target_cell_size_m=args.resolution)
    when, lat, lon, helper_kwargs = _resolve_heating_args(args, dem)

    result = run_model(
        dem.elevation_m,
        dem.cell_size_m,
        when,
        lat,
        lon,
        wind_from_deg=float(args.wind_from),
        wind_speed_ms=float(args.wind_speed),
        wind_tilt_k=float(args.wind_tilt_k),
        smoothing_sigma_m=float(args.smoothing_sigma),
        curvature_smoothing_sigma_m=float(args.curvature_smoothing_sigma),
        min_slope_deg=float(args.min_slope),
        slope_scale_deg=float(args.slope_scale),
        kappa_ref=float(args.kappa_ref),
        q_ref=float(args.q_ref),
        f_min=float(args.f_min),
        f_max=float(args.f_max),
        absorptivity=helper_kwargs["absorptivity"],
        elevation_m=helper_kwargs.get("elevation_m"),
        linke_turbidity=helper_kwargs["linke_turbidity"],
        resolve_flats=bool(args.resolve_flats),
    )

    write_raster_like(args.out, result.trigger_potential, dem)
    print(f"wrote {args.out}")

    if args.leak_out is not None:
        write_raster_like(args.leak_out, result.leak, dem)
        print(f"wrote {args.leak_out}")

    if args.cycle_period_out is not None:
        # +inf cells (no trigger) are translated to NaN before write so
        # the GeoTIFF's nodata sentinel applies; cells that *do* trigger
        # carry their finite cycle period.
        import numpy as np

        cycle_finite = np.where(
            np.isfinite(result.cycle_period_s), result.cycle_period_s, np.nan
        )
        write_raster_like(args.cycle_period_out, cycle_finite, dem)
        print(f"wrote {args.cycle_period_out}")

    if args.kmz is not None:
        # Cluster on absolute leak (W/m²), not the rank-normalised
        # trigger raster, so the KMZ description carries physically
        # meaningful strength values that compare across tiles.
        # Cycle period is averaged over each cluster and embedded.
        points = cluster_triggers(
            result.leak,
            threshold_quantile=float(args.cluster_quantile),
            min_cluster_cells=int(args.min_cluster_cells),
            cycle_period_s=result.cycle_period_s,
        )
        if not points:
            print(f"no trigger clusters survived; skipping {args.kmz}")
        else:
            write_kmz(
                points,
                args.kmz,
                transform=dem.transform,
                crs=dem.crs,
                name=f"{args.dem.stem} triggers",
            )
            print(f"wrote {args.kmz} ({len(points)} clusters)")
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
