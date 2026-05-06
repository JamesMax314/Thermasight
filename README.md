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

## Status

Phase 0 (repo skeleton + I/O). See `docs/ROADMAP.md` for the phased plan.
