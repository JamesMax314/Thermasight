# Roadmap

**Current phase: Phase 1 — terrain morphometrics + inverted-DEM convergence.**

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

- [x] Slope, aspect, and profile curvature (Horn / Zevenbergen & Thorne).
- [x] Pit-fill on inverted DEM (priority-flood, Barnes et al. 2014).
- [x] D∞ flow accumulation via `richdem` with a numpy fallback
  (`thermal_model/physics/flow.py`: `dinf_flow_directions` and
  `flow_accumulation`; Tarboton 1997, eight-facet selection,
  topological pass in descending elevation; weights, NaN nodata,
  auto-selecting backend; reference implementation is the numpy path).
- [x] Diagnostic plots overlaid on hillshade
  (`thermal_model/viz/`: Lambertian `hillshade`, generic `plot_overlay`,
  and `plot_convergence` / `plot_slope` / `plot_aspect` /
  `plot_profile_curvature` for sanity checks. Phase 2 will introduce a
  separate, physical hillshade in `solar/` for cast-shadow analysis).
  Drivable from the CLI via `python -m thermal_model preview` (pulled
  forward from Phase 3 since the diagnostic plots are not useful
  without a way to run them on a tile).
- [ ] Property tests: rotation/scaling invariants.
- [x] DEM mosaic pipeline (`thermal_model/io/mosaic.py`: `mosaic_dems()`
  wrapping `rasterio.merge.merge` with project on-disk conventions:
  `nodata=-9999`, deflate-compressed tiled float32. Validates CRS and
  cell-size consistency across inputs before merging. Drivable from
  `python -m thermal_model mosaic`. Pulled in from CLAUDE.md §9 to
  assemble whole-hill tiles from EA LIDAR 5 km blocks for the Phase 1
  validation gate; the 256 m fixture used for I/O testing is too small
  to validate convergence against any single thermal location).

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
- [x] CLI subcommand: `preview` (pulled forward to Phase 1 alongside
  the diagnostic plots).
- [ ] CLI subcommand: `run`.

## Phase 4 — Land cover + time-of-day

- [ ] UKCEH land cover ingestion + absorptivity table.
- [ ] Time-of-day weighting between heating and convergence.
- [ ] CLI flags for land cover and time-window sweeps.

## Phase 5 — Real physics

- [ ] WindNinja-driven terrain-aware wind field.
- [ ] Lagrangian plume model replacing the hydrological analogy.
- [ ] Comparison with Phase 1 convergence map as the validation step.
