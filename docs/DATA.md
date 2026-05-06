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

## Land cover (optional)

* **Source**: UKCEH Land Cover Map (free for non-commercial use).
* **Use**: lookup table mapping class → surface absorptivity $\alpha$.

### Absorptivity lookup (placeholder)

The numbers below are starting estimates and will be tuned during Phase 4.

| Class                   | $\alpha$ (1 − albedo) | Notes                          |
|-------------------------|-----------------------|--------------------------------|
| Bare rock / scree       | 0.85                  | Strong heater                  |
| Dry grass / heather     | 0.80                  | Default upland Dales surface   |
| Improved grassland      | 0.75                  |                                |
| Coniferous woodland     | 0.90                  | Dark, but moisture-buffered    |
| Broadleaf woodland      | 0.85                  |                                |
| Bog / wet peat          | 0.40                  | Energy goes to evaporation     |
| Open water              | 0.05                  | Effectively dead               |
| Built / hardstanding    | 0.85                  |                                |

See `CLAUDE.md §5` — wet ground is dead ground; the bog row is the
single most important entry to get right.

## Validation data

See `docs/VALIDATION.md` for known reliable thermal locations used as
ground truth during Phase 1 validation.

## Fixtures

`data/fixtures/` holds tiny rasters (≤ 256×256) checked into git so CI can
run without the full LIDAR archive. Synthetic fixtures generated in tests
do not need to live here; only real-terrain clips do.
