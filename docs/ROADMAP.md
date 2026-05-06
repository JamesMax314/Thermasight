# Roadmap

**Current phase: Phase 0 — repo skeleton, I/O, fixtures, CI.**

Update this header when a phase completes. Do not skip phases. Do not start
the next phase until its predecessor's gate passes.

---

## Phase 0 — Skeleton

- [x] Repository layout per `CLAUDE.md §3`.
- [x] `environment.yml` with the conda-managed geospatial stack.
- [x] `pyproject.toml` declaring the package + ruff/mypy config.
- [x] `.gitignore` excluding raw and processed LIDAR data.
- [x] DEM read/write with NaN nodata convention.
- [x] CLI entrypoint plumbing (`python -m thermal_model`).
- [x] Test harness with synthetic-DEM fixture and round-trip test.
- [x] CI workflow (lint + tests on a conda-forge environment).
- [x] First real LIDAR fixture under `data/fixtures/`
  (`wild_boar_fell_east_256_1m.tif`, EA Composite 2022 1 m).

**Gate**: `pytest`, `ruff check`, and `mypy thermal_model` all pass in CI.

## Phase 1 — Terrain morphometrics + inverted-DEM convergence

- [ ] Slope, aspect, and profile curvature (Horn's method).
- [ ] Pit-fill on inverted DEM.
- [ ] D∞ flow accumulation via `richdem` with a numpy fallback.
- [ ] Diagnostic plots overlaid on hillshade.
- [ ] Property tests: rotation/scaling invariants.

**Gate**: convergence raster agrees with `docs/VALIDATION.md` on three
independent test tiles. Document the comparison in `docs/VALIDATION.md`.

## Phase 2 — Solar + heating

- [ ] Sun position + clear-sky irradiance via `pvlib`.
- [ ] Hillshade and cast-shadow mask via horizon scan.
- [ ] Heating field $H = I \cdot \alpha \cdot s$.
- [ ] Coupling $P = \sqrt{H \cdot C}$ with $(p, q)$ exposed.

## Phase 3 — Wind drift + triggers

- [ ] Single-vector drift via sub-pixel `ndimage.shift`.
- [ ] Profile-curvature trigger detector + DBSCAN clustering.
- [ ] GeoTIFF + KMZ export of trigger points.
- [ ] CLI subcommands: `run`, `preview`.

## Phase 4 — Land cover + time-of-day

- [ ] UKCEH land cover ingestion + absorptivity table.
- [ ] Time-of-day weighting between heating and convergence.
- [ ] CLI flags for land cover and time-window sweeps.

## Phase 5 — Real physics

- [ ] WindNinja-driven terrain-aware wind field.
- [ ] Lagrangian plume model replacing the hydrological analogy.
- [ ] Comparison with Phase 1 convergence map as the validation step.
