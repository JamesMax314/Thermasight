#!/usr/bin/env bash
# Build the typeset Thermasight model PDF.
#
# 1. Regenerate the Ullswater figure set if it isn't on disk
#    (requires `data/processed/ullswater_1m.tif`).
# 2. Symlink `docs/pdf/figures/` -> `outputs/Ullswater/` so the .qmd
#    can reference figures by relative path.
# 3. Run `quarto render` to typeset the PDF.
#
# Run with the `thermals` conda env active.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ULLSWATER_DIR="$REPO/outputs/Ullswater"
RENDER_SCRIPT="$REPO/outputs/ullswater_p34_render.py"
FIGURES_LINK="$HERE/figures"

cd "$REPO"

# 1. Regenerate figures if missing.
if [ ! -f "$ULLSWATER_DIR/ullswater_compare_5m.png" ]; then
  echo "→ Ullswater figures missing; running $RENDER_SCRIPT..."
  python "$RENDER_SCRIPT"
else
  echo "→ Using existing Ullswater figures under $ULLSWATER_DIR"
fi

# 2. Symlink figures into docs/pdf/ for relative-path resolution.
if [ ! -e "$FIGURES_LINK" ]; then
  ln -s "../../outputs/Ullswater" "$FIGURES_LINK"
  echo "→ Created symlink $FIGURES_LINK -> ../../outputs/Ullswater"
fi

# 3. Typeset.
echo "→ Typesetting $HERE/thermasight.qmd..."
quarto render "$HERE/thermasight.qmd" --to pdf

echo "→ Done: $HERE/thermasight.pdf"
