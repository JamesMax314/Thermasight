# `docs/pdf/` — typeset PDF source

`thermasight.qmd` is the source for **The Thermasight Model**, a
typeset PDF that walks through the physics, the pipeline, and the
intermediates for the Ullswater validation render. It's the
read-through companion to the auto-generated API reference under
`docs/api/` (built via `make docs` at the repo root).

## Build

```bash
conda activate thermals
bash docs/pdf/build.sh
```

The script:

1. Regenerates the Ullswater figure set via
   `outputs/ullswater_p34_render.py` if it isn't already on disk
   (this requires `data/processed/ullswater_1m.tif`).
2. Symlinks `docs/pdf/figures/` → `outputs/Ullswater/` so the .qmd
   can reference figures by relative path.
3. Runs `quarto render thermasight.qmd --to pdf`.

The output PDF, the figures symlink, and Quarto's working directory
are all gitignored. The .qmd source and `build.sh` are checked in.

## Editing

The document is plain markdown plus LaTeX inline / block math. Edit
`thermasight.qmd` directly; rerun `build.sh` to typeset. Inline math
uses `$...$`, block math `$$...$$`, cross-references `@fig-leak` /
`@eq-conservation` / `@sec-leak`. Quarto's docs at <https://quarto.org>
have the full reference.

If the Ullswater render parameters change (canonical date/time,
wind, smoothing scales), update the prose in `thermasight.qmd` to
match.
