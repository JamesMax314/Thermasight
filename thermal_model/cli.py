"""Command-line entry point.

Subcommands are added in the phase that introduces them. Phase 0 ships only
the entrypoint plumbing so ``python -m thermal_model`` and the
``thermal-model`` console script exist.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermal-model",
        description="Predict thermal sources and trigger points from LIDAR DEMs.",
    )
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
