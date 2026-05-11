# `docs/`

Documentation lives in two layers, each with a different audience.

## Narrative — read these by hand

* [`MODEL.md`](MODEL.md) — the physics and math behind the model.
  §1–§4 + §11 are the canonical current production formulation;
  §5–§8 and §10 are preserved predecessor sections.
* [`model_correction.md`](model_correction.md) — the ground-level
  trigger-prediction reformulation. Read together with `MODEL.md`.
* [`ROADMAP.md`](ROADMAP.md) — phased build plan with the current
  phase pinned at the top.
* [`VALIDATION.md`](VALIDATION.md) — known thermal locations and the
  validation log, including the 2026-05-11 Ullswater real-flight
  ground-truth entry.
* [`DATA.md`](DATA.md) — data sources, CRS conventions, the land-cover
  absorptivity lookup table.
* [`TODO.md`](TODO.md) — outstanding follow-ups not yet roadmapped.

## API reference — auto-generated

The `thermal_model/` package is documented in NumPy-style docstrings
on every public function and class. To build a browsable HTML
reference from them:

```bash
conda activate thermals
make docs
```

This drops a static site under `docs/api/` (gitignored). Open
`docs/api/index.html` in a browser. While iterating on docstrings,
`make docs-serve` runs a live-reload preview at
http://localhost:8080. `make docs-clean` wipes the generated tree.

The build uses [`pdoc`](https://pdoc.dev) with NumPy docstring
parsing and MathJax for inline LaTeX. No config files; the docstrings
as written are the entire input.
