"""Entry point: python -m closure_cli <command>

Commands:
    identity  — Compare two files (Gilgamesh). Full static verification.
    observer  — Monitor two streams (Seer). Escalates to Gilgamesh on drift.
    seeker    — Classify every record (Enkidu). Deep real-time analysis.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .identity import build_parser as identity_parser
from .observer import build_parser as observer_parser
from .seeker import build_parser as seeker_parser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="closure",
        description="Closure — geometric data verification.",
    )
    parser.add_argument("--version", action="version", version=f"closure-cli {__version__}")

    subparsers = parser.add_subparsers(dest="command")
    identity_parser(subparsers)
    observer_parser(subparsers)
    seeker_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
