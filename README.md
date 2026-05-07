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

## Preview a tile

The `preview` subcommand renders a hillshade-backed diagnostic plot of a DEM.
Useful for sanity-checking the convergence map and morphometric fields against
real terrain.

```bash
# Open a window with the convergence overlay (the headline diagnostic)
python -m thermal_model preview --dem data/fixtures/wild_boar_fell_east_256_1m.tif

# Pick a single view: convergence | slope | aspect | curvature | all
python -m thermal_model preview --dem <path> --what slope

# Save a 2×2 panel to a PNG instead of opening a window
python -m thermal_model preview --dem <path> --what all --save out.png --dpi 150
```

`python -m thermal_model preview --help` lists all options.

## Status

Phase 1 (terrain morphometrics + inverted-DEM convergence). See
`docs/ROADMAP.md` for the phased plan.
