# Thermal Prediction Model — Project Memory

This file is loaded into context at the start of every Claude Code session.
Read it carefully. The conceptual model in §2 is the most important thing in
this document; do not lose it across refactors.

---

## 1. What this project is

A computational model that predicts paragliding **thermal sources** (where
hot-air bubbles are generated) and **trigger points** (where they detach
from terrain and become useful lift) from LIDAR digital elevation models
of the Yorkshire Dales (and eventually elsewhere).

Inputs:
- LIDAR DEM (1 m or 2 m GeoTIFF, British National Grid / EPSG:27700)
- Date, time, latitude, longitude
- Wind direction and speed (single vector for now; gridded later)
- Optional: land cover raster, satellite-derived surface temperature

Outputs:
- Heating field (W/m²) per cell
- Thermal convergence raster (the "inverted treacle" map, optionally
  wind-tilted — see §2)
- Trigger-potential raster: the primary product, predicting where on
  the ground thermals source and trigger
- Trigger point set (vector + raster)
- Diagnostic plots and exportable GeoTIFFs / KMZ for use in XCTrack /
  SeeYou / Google Earth

The model predicts **ground-level thermal source and trigger
locations** — not where a thermal is once it is airborne. Anything
about a parcel's drift after detachment is out of scope for the main
pipeline (see `@docs/model_correction.md`).

The user is the primary operator and is an experienced paraglider flying
the Dales; **assume domain knowledge in their requests** and ask before
adding "safety" features that would clutter the output.

---

## 2. Core conceptual model — DO NOT FORGET

The model rests on one analogy: **rising air on real terrain follows the
same gradient logic as falling water on inverted terrain.**

Concretely:

```
thermal_convergence(dem)  ≡  flow_accumulation( max(dem) - dem )
```

So the heart of the engine is hydrological flow accumulation run on a
flipped DEM. Convex features on the real terrain (spurs, cliff lips,
shoulders) become concave on the inverted DEM, so flow pools there —
which is exactly where rising air is predicted to converge and release.

Everything else in the pipeline is decoration on this core. The
pipeline runs in this order (see `@docs/model_correction.md` §6 for
the full block diagram):

1. **Smooth the raw DEM** with a Gaussian (σ ≈ 10–25 m) to suppress
   sub-thermal noise before any flow routing.
2. **Wind tilt** — add a linear ramp `k · wind_speed · (col_m·sin θ −
   row_m·cos θ)` to the *smoothed* DEM, where θ is the wind-to
   bearing. This biases the inverted-DEM flow accumulation toward the
   lee side of features, which is where boundary-layer warm air
   actually pools. **Tilt before inversion, never after.**
3. **Convergence** — invert the wind-tilted smoothed DEM, pit-fill,
   and run D∞ flow accumulation. The result is the ground-level
   convergence of surface warm air.
4. **Heating field** — slope × aspect × sun × surface albedo × cast
   shadows, computed from the **raw** DEM (not the smoothed or tilted
   one — real geometry drives shadows and gradients).
5. **Energy** — geometric mean of independently normalised heating and
   convergence: `sqrt(norm(H) · norm(C))`. Both must be non-negligible
   for a cell to score; either-zero collapses correctly.
6. **Trigger filter** — multiply by normalised positive profile
   curvature (convex breaks) and a minimum-slope mask (~2.5° to kill
   flat-summit and valley-floor artefacts). Profile curvature is read
   from the **raw** DEM. The output is the trigger-potential raster
   on `[0, 1]`.

Wind enters the model **only** via the tilt in step 2. The model
does **not** drift thermals after detachment — that was a previous
formulation that has been superseded (see
`@docs/model_correction.md` §4 and §8). Any reintroduction of an
in-air drift step requires explicit operator approval.

If a proposed change weakens the connection between flow accumulation
on the (tilted) inverted DEM and the convergence map, **stop and
discuss with the user before implementing**. That mapping is the
whole project.

Detailed math and derivations are in `@docs/MODEL.md`; the
ground-level reformulation and the wind-tilt mechanism are in
`@docs/model_correction.md` and that document supersedes any
conflicting wording elsewhere. Read both files when working on
anything in `thermal_model/physics/`.

---

## 3. Repository layout

```
thermal_model/
  io/              # DEM and raster I/O, reprojection, tiling
  terrain/         # slope, aspect, curvature, ridge detection
  solar/           # sun position, irradiance, hillshade, cast shadows
  physics/         # the inverted-treacle engine + heating + wind tilt
  triggers/        # trigger-point detection and clustering
  viz/             # matplotlib previews, GeoTIFF export, KMZ export
  cli.py           # `python -m thermal_model ...` entrypoint
tests/             # pytest, mirrors the package layout
docs/
  MODEL.md         # math + physics, including planned plume model
  DATA.md          # data sources, CRS conventions, tiling strategy
  VALIDATION.md    # known thermal locations for ground-truthing
  ROADMAP.md       # phased build plan with checkboxes
data/
  raw/             # untouched LIDAR tiles (gitignored)
  processed/       # mosaicked / reprojected DEMs (gitignored)
  fixtures/        # tiny test tiles checked in for unit tests
notebooks/         # exploratory Jupyter notebooks (gitignored except templates)
```

Keep modules small and pure. Functions take arrays + parameters, return
arrays. Stateful "scene" objects live only in `physics/scene.py`.

---

## 4. Tech stack and library choices

- **Python 3.11+**
- **numpy, scipy** — array math, gradients, ndimage shifts
- **rasterio** — all GeoTIFF I/O. Never use GDAL Python bindings directly.
- **richdem** — flow accumulation. **Always use D∞**, not D8, except in
  the pure-numpy fallback in `physics/_fallback.py`.
- **pvlib** — solar position and clear-sky irradiance. Don't reinvent.
- **pyproj** — CRS transforms, used inside `io/` only.
- **matplotlib** — plotting only. No seaborn, no plotly in the core.
- **simplekml** — KMZ export of trigger points for Google Earth.
- **pytest** + **hypothesis** — testing. Property-based tests for
  geometric invariants (see §7).

Do **not** add: tensorflow, pytorch, geopandas (use plain shapely +
fiona via rasterio), pandas (numpy is enough for raster work), Django,
Flask. If you think you need any of these, ask first.

Optional, behind feature flags:
- **WindNinja** for terrain-aware wind fields (huge upgrade; later phase)
- **xarray** + **dask** for out-of-core processing of large mosaics

---

## 5. Domain knowledge the model must respect

These are facts about thermals and Yorkshire Dales terrain that should
shape design decisions. **Do not 'simplify' the model in ways that
violate these.**

- **The trigger location is on the ground, downwind of the heat
  source.** A south-facing rocky bowl warms the boundary-layer air, but
  the air pools and detaches at a convex break some distance to the
  lee — a spur shoulder, cliff lip, or ridge crest. The model finds
  this *ground-level* location; in-air parcel drift after detachment
  is out of scope.
- **Convex breaks matter more than steep slopes.** A gentle slope ending
  in a sudden cliff is a stronger trigger than a uniformly steep face.
  Profile curvature is the right proxy.
- **Lee-side triggering is the primary wind effect.** Boundary-layer
  warm air is swept downwind and pools on the lee side of ridges and
  spurs. The model handles this by tilting the DEM along the wind-to
  direction before inverting and flow-accumulating (see §2 step 2 and
  `@docs/model_correction.md` §4). Lee-wave uplift, mechanical rotors,
  and valley-wind convergence lines are *not* modelled — those are
  known limitations rather than bugs.
- **Aspect dominates in the morning, terrain dominates in the afternoon.**
  Early-day sources are tightly tied to SE/S/SW-facing slopes. By
  mid-afternoon the whole massif is warm and trigger geometry takes over.
  Time-of-day-aware weighting between heating and convergence is on the
  roadmap.
- **Wet ground is dead ground.** Bog and wet peat absorb solar energy
  into evaporation, not into the air column. If land cover is available,
  weight bog/wet-peat absorption far below dry grass and bare rock.
- **The Dales are limestone scarps and gritstone edges.** Expect strong
  triggers along edges (Malham, Gordale, Pen-y-ghent's east face,
  Ingleborough's west face). These are good validation targets.

`@docs/VALIDATION.md` has a list of known reliable thermal locations
with grid references. Use these for sanity-checking new model versions.

---

## 6. Operating principles for the agent

When working on this project:

1. **Validate the convergence map first.** Before adding more physics,
   confirm that `thermal_convergence(dem)` lights up on the known thermal
   spots in `@docs/VALIDATION.md` for at least three test tiles. If it
   doesn't, no amount of solar / wind sophistication will help — fix the
   core first.
2. **Prefer raster operations over loops.** This is numerical raster
   code. A nested Python loop over pixels is almost always wrong.
3. **CRS is sacred.** Every raster carries a CRS. Never assume EPSG:27700
   silently — read it from rasterio and warn on mismatch. All
   computations happen in a projected CRS with metres as units.
4. **Cell size is a parameter, never a constant.** Slope, curvature,
   the wind-tilt ramp, and flow-accumulation thresholds all depend on
   it. Functions that depend on cell size must take it as an argument.
5. **NaN is the nodata sentinel internally.** Convert from rasterio's
   nodata on read; convert back on write.
6. **Small test tiles checked in.** Add a 256×256 fixture under
   `data/fixtures/` for any new feature. CI must run on these without
   needing the full LIDAR archive.
7. **Plots are diagnostics, not deliverables.** The deliverable is
   GeoTIFF + KMZ. Plotting code lives in `viz/` and is never imported
   from `physics/` or `terrain/`.
8. **Ask before adding GUIs, web apps, or Streamlit dashboards.** This
   is a library + CLI. UI is out of scope unless the user explicitly
   requests it.
9. **Be honest about uncertainty.** If the model says a hill is a strong
   trigger and the user knows it isn't, the model is wrong. Log and
   investigate; do not silently tune thresholds to match.

---

## 7. Conventions

- **Style**: black (line length 88), ruff (`select = ["E","F","I","N","UP","B"]`),
  mypy strict on `thermal_model/`, lenient elsewhere.
- **Docstrings**: NumPy style. Every public function has one.
- **Type hints**: required on all public functions; encouraged elsewhere.
- **Naming**: arrays carry their meaning in the name. `dem`, `slope_rad`,
  `aspect_rad`, `irradiance_wm2`, `thermal_potential`. Never `data`,
  `arr`, `x`, `result`.
- **Units in suffixes** when there is ambiguity: `_m`, `_deg`, `_rad`,
  `_ms`, `_wm2`.
- **Tests**: every public function has at least one unit test. Use
  hypothesis for things like "rotating the DEM rotates the slope by the
  same angle" and "scaling cell size by k scales the wind-tilt ramp's
  pixel-space displacement by 1/k".

---

## 8. Commands

Dependencies are managed with **conda** (env name: `thermals`). Runtime
and dev dependencies live in `environment.yml`; the package itself is
installed editable via pip. `pyproject.toml` defines the package and its
build system but does **not** declare runtime deps — those are conda's
job, because most of the geospatial stack (rasterio, richdem, pyproj,
GDAL) ships C/C++ extensions that conda-forge handles cleanly and pip
does not. `mamba` may be substituted anywhere `conda` appears for speed.

```bash
# Setup (one-time)
conda env create -f environment.yml          # creates `thermals`
conda activate thermals
pip install -e .                             # install the package itself

# Update after environment.yml changes
conda env update -f environment.yml --prune

# Always activate before running anything
conda activate thermals

# Test
pytest                              # full suite
pytest -k convergence               # one area
pytest --hypothesis-show-statistics # property-based test stats

# Lint / format / type check
ruff check . && ruff format --check .
mypy thermal_model

# Run the model on a tile
python -m thermal_model run \
  --dem data/processed/penyghent_1m.tif \
  --datetime "2026-05-06T13:00:00+01:00" \
  --wind-from 225 --wind-speed 4 \
  --wind-tilt-k 0.03 \
  --out outputs/penyghent_2026-05-06_1300.tif \
  --kmz outputs/penyghent_2026-05-06_1300.kmz

# Quick visual preview of a tile
python -m thermal_model preview --dem <path> --datetime <iso>
```

If a command above doesn't exist yet, create it as part of the work
that needs it; don't stub it.

---

## 9. Data

- LIDAR: Environment Agency LIDAR Composite, 1 m resolution, free
  download from https://environment.data.gov.uk/survey. Tiles arrive as
  GeoTIFFs in EPSG:27700 (British National Grid). Mosaicking and
  reprojection live in `io/mosaic.py`.
- Land cover: UKCEH Land Cover Map (free for non-commercial). Optional
  input. See `@docs/DATA.md` for the absorption lookup table.
- Validation: `@docs/VALIDATION.md` — known thermal locations from
  pilot logs and XContest tracks.

Raw and processed data live under `data/` and are gitignored. Only the
small fixtures under `data/fixtures/` are checked in.

---

## 10. Roadmap and current phase

The project is built in phases, each gated on validation. See
`@docs/ROADMAP.md` for full detail with checkboxes. The current phase
is recorded at the top of that file — read it before starting new work
and update it when a phase completes.

High level:

1. **Phase 0** — repo skeleton, I/O, fixtures, CI.
2. **Phase 1** — terrain morphometrics, inverted-DEM flow accumulation,
   visual validation against `@docs/VALIDATION.md`. **Gate: convergence
   map must agree with known thermal spots on 3+ tiles.**
3. **Phase 2** — solar position, irradiance, hillshade, heating field.
4. **Phase 3** — wind tilt of the inverted-DEM convergence,
   trigger-potential pipeline (energy × convex curvature × slope
   mask), trigger-point clustering, KMZ export. See
   `@docs/model_correction.md` for the corrected formulation that
   supersedes the original "wind drift" framing.
5. **Phase 4** — land cover integration, time-of-day weighting, CLI.
6. **Phase 5** — terrain-aware wind (WindNinja) replacing the empirical
   wind-tilt coefficient `k`, and a Lagrangian plume model alongside
   the hydrological analogy for cross-validation. Big jump in physical
   realism; only after phases 1–4 are solid.

Do not skip phases. Do not start phase 2 until phase 1 has passed its
validation gate, even if it seems easy.

Any additional features and the details of implimentation should be recoreded in Roadmap before each commit.

## Git workflow

- Never commit directly to `main` or `develop`
- Create a feature branch before starting any new screen or feature:
  `git checkout -b feat/description-of-feature`
- Commit after each logical unit of work — not at end of session
- Commit messages follow Conventional Commits:
  `feat: add location selector to nav bar`
  `fix: correct refrigerant GWP calculation for R404A`
  `test: add unit tests for commuting tCO2e formula`
- Always run `ruff check .`, `ruff format --check .`, `mypy thermal_model`,
  and `pytest` before committing. All four must pass.
- Never commit with failing tests, lint errors, format diffs, or type errors
- Write a meaningful commit message — not "wip" or "update"
