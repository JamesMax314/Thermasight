# Thermasight

A computational model that predicts paragliding thermal sources and trigger
points from LIDAR digital elevation models.

The core idea: rising air on real terrain follows the same gradient logic as
falling water on inverted terrain. So thermal convergence is computed as
hydrological flow accumulation on a flipped DEM. See `docs/MODEL.md` for the
full conceptual model, and `CLAUDE.md` for project conventions.

## Quick start

```bash
conda env create -f environment.yml
conda activate thermals
pip install -e .
pytest
```

## Mosaic LIDAR tiles

EA LIDAR Composite ships as 5 km × 5 km blocks; for whole-hill convergence
analysis, stitch a set of adjacent blocks into a single mosaic. Inputs must
share a CRS and cell size.

```bash
# Mosaic the 12 Wild Boar / Mallerstang tiles (15 km × 20 km, 1 m, ~600 MB)
python -m thermal_model mosaic \
  --inputs data/raw/*/*.tif \
  --output data/processed/mallerstang_wildboar_1m.tif
```

Output is deflate-compressed float32 with `nodata=-9999`. Pass `--overwrite`
to replace an existing output. `python -m thermal_model mosaic --help` lists
all options.

## Preview a tile

The `preview` subcommand renders a hillshade-backed diagnostic plot of a DEM.
Useful for sanity-checking the convergence map and morphometric fields against
real terrain.

```bash
# Open a window with the convergence overlay (the headline diagnostic)
python -m thermal_model preview --dem data/fixtures/wild_boar_fell_east_256_1m.tif

# Pick a view: convergence | slope | aspect | curvature | heating |
#              weighted-convergence | trigger | all
python -m thermal_model preview --dem <path> --what slope

# Save a 2×2 panel to a PNG instead of opening a window
python -m thermal_model preview --dem <path> --what all --save out.png --dpi 150
```

### Heating-field preview

`--what heating` runs the full Phase 2 solar pipeline (slope and aspect →
sun position → clear-sky irradiance → slope projection → cast-shadow mask
→ heating field) and overlays the per-cell heating in W/m² on the
hillshade with elevation contours.

It needs a timezone-aware ISO timestamp. Latitude and longitude default to
the DEM's centre, reprojected from the DEM's CRS, so for the project's
BNG-native LIDAR you only have to pass `--datetime`:

```bash
python -m thermal_model preview \
  --dem data/fixtures/wild_boar_fell_east_256_1m.tif \
  --what heating \
  --datetime "2026-05-06T13:00:00+01:00" \
  --save heating.png
```

Override defaults if needed:

```bash
python -m thermal_model preview --dem <path> --what heating \
  --datetime "2026-05-06T13:00:00+01:00" \
  --lat 54.20 --lon -2.30 \
  --elevation 600 \
  --linke-turbidity 3.0 \
  --absorptivity 0.80
```

`--linke-turbidity` controls the Ineichen-Perez clear-sky model
(default 3.0 — temperate clear day; 2 is very clean cold air, 5 is
hazy). `--absorptivity` is the surface absorptivity α = 1 − albedo
(default 0.80, dry grass / heather; bog is ~0.4 — see `docs/DATA.md`).

### Trigger and weighted-convergence previews (Phase 3)

`--what weighted-convergence` runs the heating-weighted D∞ flow
accumulation on the inverted, wind-tilted DEM (the routing surface
for the Phase 3 pipeline). `--what trigger` then multiplies the
rank-normalised convergence by the rank-normalised positive profile
curvature and a minimum-slope mask to give the trigger-potential
raster on `[0, 1]`. Both require `--wind-from` (degrees, met
convention) and `--wind-speed` (m/s); `--datetime` is mandatory for
the heating component.

```bash
# Headline trigger overlay for a SW summer afternoon.
python -m thermal_model preview \
  --dem data/processed/mallerstang_wildboar_1m.tif \
  --what trigger \
  --datetime "2026-07-15T13:00:00+01:00" \
  --wind-from 225 --wind-speed 6 \
  --resolution 5.0 \
  --save outputs/mallerstang_trigger.png

# The intermediate that the trigger raster is built from.
python -m thermal_model preview \
  --dem data/processed/mallerstang_wildboar_1m.tif \
  --what weighted-convergence \
  --datetime "2026-07-15T13:00:00+01:00" \
  --wind-from 225 --wind-speed 6 \
  --resolution 5.0 \
  --save outputs/mallerstang_wconv.png
```

Tunables that affect routing on these previews:

* `--wind-tilt-k` (default 0.03 s/m) — tilt coefficient. `k × |u|` is
  the dimensionless fractional slope added to the smoothed DEM
  before inversion. See `docs/MODEL.md` §3 for the 0.01 (light) /
  0.03 (moderate) / 0.05 (strong) envelope.
* `--smoothing-sigma` (default 10 m) — Gaussian sigma applied to the
  DEM before wind tilt and flow routing.
* `--min-slope` (default 2.5°) — kills flat-summit and valley-floor
  artefacts.
* `--no-resolve-flats` — skips Garbrecht-Martz flat resolution
  between pit-fill and flow accumulation. Default is to *enable* it
  (recommended); turn off only for fast iteration on large mosaics
  where the streak artefact is acceptable.

The same lat/lon/elevation/turbidity/absorptivity flags as the
heating preview are accepted.

## Run the full pipeline

For the headline deliverables — a trigger-potential GeoTIFF plus
optionally a KMZ of clustered trigger points for XCTrack / SeeYou /
Google Earth — use the `run` subcommand:

```bash
python -m thermal_model run \
  --dem data/processed/mallerstang_wildboar_1m.tif \
  --resolution 5.0 \
  --datetime "2026-07-15T13:00:00+01:00" \
  --wind-from 225 --wind-speed 6 \
  --wind-tilt-k 0.03 \
  --out outputs/mallerstang_trigger_2026-07-15_1300.tif \
  --kmz outputs/mallerstang_trigger_2026-07-15_1300.kmz
```

The GeoTIFF carries the float32 trigger-potential raster on `[0, 1]`
in the source CRS (typically EPSG:27700 for EA LIDAR), with NaN
written as `-9999`. The KMZ contains one placemark per cluster,
named by rank (1 = brightest), with mean strength and cell count in
the description. Clustering uses 8-connected components on cells
where `T > q95` of strictly-positive `T`, dropping clusters smaller
than `--min-cluster-cells` (default 3); tune via
`--cluster-quantile` and `--min-cluster-cells`. Skip `--kmz` to get
the GeoTIFF only.

`python -m thermal_model run --help` lists all options.

### `--resolution` for big mosaics

The cast-shadow horizon scan dominates pipeline cost (>95%) and
scales worse than linearly with cell count. On the full 15 km × 20 km
Mallerstang mosaic at native 1 m resolution the heating render takes
~25 minutes; at 5 m resolution it takes ~40 seconds and the
diagnostic picture is essentially the same.

```bash
# Whole-mosaic heating at 5 m — interactive turnaround.
python -m thermal_model preview \
  --dem data/processed/mallerstang_wildboar_1m.tif \
  --what heating \
  --datetime "2026-05-06T13:00:00+01:00" \
  --resolution 5.0 \
  --save heating_5m.png
```

`--resolution` accepts any cell size in metres ≥ the source's; finer
than the source is rejected (no upsampling). Works on every `--what`
variant, not just `heating`. The resample uses bilinear
interpolation with a separate validity-mask reread to keep nodata
edges from polluting averages.

`python -m thermal_model preview --help` lists all options.

## Status

Phase 3 (wind tilt + ground-level triggers) feature-complete. See
`docs/ROADMAP.md` for the phased plan.
