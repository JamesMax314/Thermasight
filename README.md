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

# Pick a single view: convergence | slope | aspect | curvature | heating | all
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

Phase 2 (solar + heating) feature-complete. See `docs/ROADMAP.md` for
the phased plan.
