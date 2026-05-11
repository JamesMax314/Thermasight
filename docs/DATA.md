# Data sources and conventions

## DEM (required)

* **Source**: Environment Agency LIDAR Composite, 1 m resolution, free at
  https://environment.data.gov.uk/survey.
* **Format**: GeoTIFF, single-band float32 elevation in metres.
* **CRS**: EPSG:27700 (British National Grid). All internal computations
  assume a projected CRS with metres as units.
* **Tiling**: Raw tiles land in `data/raw/`. Mosaicking and reprojection
  produce `data/processed/<area>_<res>m.tif`. Both directories are
  gitignored.

## Land cover (Phase 4 — wired 2026-05-10)

* **Source**: UKCEH Land Cover Map 2024, 10 m resolution.
* **Free access**: public WMS (no login). See "UKCEH LCM WMS access"
  below.
* **Use**: lookup table mapping class code → surface absorptivity
  $\alpha$. The absorptivity then drives the per-cell weight on the
  heating-weighted leaky-bucket flow accumulation in
  `thermal_model.physics.run_model`. See `MODEL.md` §11 (production
  formulation) and `model_correction.md` §3 (heating-as-flow-weight
  rationale).

### UKCEH LCM WMS access

```
Base URL : https://catalogue.ceh.ac.uk/maps/688492ef-d9db-43b7-8107-3675c6150568
Layers   : LC.10m.GB   (Great Britain, traditional palette)
           LC.10m.NI   (Northern Ireland)
           *_colourBlind variants exist; not used here
Style    : traditional (default)
CRS      : EPSG:27700 supported directly (no reprojection needed)
MaxSize  : 2048 × 2048 px per GetMap call
```

The Phase 4 WMS plumbing lives in
`thermal_model/io/land_cover_wms.py`:

* `fetch_lcm_for_dem(dem)` — issues a single ``GetMap`` request
  spanning the DEM's footprint at 10 m, decodes the PNG via rasterio,
  and reverse-maps RGB → class code using the hardcoded
  ``UKCEH_LCM_PALETTE`` (sampled from the live legend graphic on
  2026-05-10).
* Caching: every fetched PNG is keyed by SHA1 of the canonical URL
  and stored under ``data/cache/lcm/<layer>/<sha1>.png``. Default on;
  bypass with ``use_cache=False`` (CLI: ``--no-lcm-cache``).
* Oversize bbox (> 2048 px at 10 m) raises ``NotImplementedError``
  with a clear message; chunked WMS fetch is follow-up scope.

Two known palette collisions in the WMS rendering (no impact for
inland-Dales work, but documented for honesty):

* RGB ``(204, 179, 0)`` → class 15 (supralittoral rock); class 16
  (supralittoral sediment) shares this colour.
* RGB ``(255, 255, 128)`` → class 17 (littoral rock); class 18
  (littoral sediment) shares this colour.

### UKCEH LCM class-code mapping

The production 21-class ``UKCEH_LCM_ABSORPTIVITY`` lookup in
``thermal_model/io/land_cover.py`` is **deliberately empty** until the
operator authors α values for every class. Until then, every cell with
a known class code falls through to ``DEFAULT_ABSORPTIVITY`` (0.80) —
identical to a scalar-α run, which is the safe default.

For the Phase 4 Mallerstang validation render and for tests, a minimal
Dales-focused lookup is shipped:

| Code | UKCEH class                     | α    | Notes                              |
|------|---------------------------------|------|------------------------------------|
| 4    | Improved grassland              | 0.75 |                                    |
| 7    | Acid grassland                  | 0.80 | Treated as the dry-grass default   |
| 9    | Heather                         | 0.80 |                                    |
| 11   | Bog                             | 0.40 | **Load-bearing: wet ground is dead ground** |
| 12   | Inland rock                     | 0.85 |                                    |
| 14   | Freshwater                      | 0.05 |                                    |
| 20   | Urban                           | 0.85 |                                    |
| 21   | Suburban                        | 0.85 |                                    |

These are starting estimates and will be tuned alongside the full
21-class authoring work. The single most important entry to get right
is bog/wet peat: it absorbs incoming solar energy into evaporation
rather than the air column, so heating-weighted convergence collapses
over bog patches — exactly the Phase 4 effect the validation render
is meant to expose.

For reference, the broader per-surface absorptivity envelope (from
the original Phase 2 placeholder table, retained for operator
authoring of the production LUT):

| Surface                 | $\alpha$ (1 − albedo) | Notes                          |
|-------------------------|-----------------------|--------------------------------|
| Bare rock / scree       | 0.85                  | Strong heater                  |
| Dry grass / heather     | 0.80                  | Default upland Dales surface   |
| Improved grassland      | 0.75                  |                                |
| Coniferous woodland     | 0.90                  | Dark, but moisture-buffered    |
| Broadleaf woodland      | 0.85                  |                                |
| Bog / wet peat          | 0.40                  | Energy goes to evaporation     |
| Open water              | 0.05                  | Effectively dead               |
| Built / hardstanding    | 0.85                  |                                |

See `CLAUDE.md §5` for the physics rationale.

## Validation data

See `docs/VALIDATION.md` for known reliable thermal locations used as
ground truth during Phase 1 validation.

## Fixtures

`data/fixtures/` holds tiny rasters (≤ 256×256) checked into git so CI can
run without the full LIDAR archive. Synthetic fixtures generated in tests
do not need to live here; only real-terrain clips do.
