# TODO

## Empirical Validation
- We need to use real flight logs and wind dtat to tune the empirical model parameters. these can be scraped from XC contest. If we pick a common busy launch point e.g. brevant chamonix then we will have tens of paragliders searching the same area for a large period of time and can construct a thermal map from this.

## Drafting follow-up (Phase 3.2 landed 2026-05-11)
- Operator Mallerstang re-render under canonical SW summer afternoon
  conditions to compare `draft_potential` against `leak`: confirm
  that the SW spur shoulders of Wild Boar Fell now reach the trigger
  raster, Mallerstang Edge scarp still dominates, summit-plateau
  interior stays dim. Write up in `docs/VALIDATION.md`.
- Sensitivity sweep σ ∈ {0, 25, 50, 75, 100, 150} m to validate the
  75 m default empirically.