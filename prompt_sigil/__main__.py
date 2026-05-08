"""CLI: render a markdown prompt as an SVG sigil.

Usage:
    python -m prompt_sigil INPUT.md [-o OUTPUT.svg] [-t TITLE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .parse import parse_markdown
from .render import render_sigil


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prompt_sigil")
    p.add_argument("input", type=Path, help="Path to markdown prompt file")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output SVG path (default: out/<stem>.svg)")
    p.add_argument("-t", "--title", default=None,
                   help="Title shown at sigil centre (default: input stem)")
    args = p.parse_args(argv)

    src = args.input
    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr)
        return 2

    text = src.read_text(encoding="utf-8")
    root = parse_markdown(text)

    title = args.title if args.title is not None else src.stem
    svg = render_sigil(root, title=title)

    out = args.output or Path("out") / f"{src.stem}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")

    sys.stdout.buffer.write(f"wrote {out}\n".encode("utf-8"))
    sys.stdout.buffer.write(
        (
            f"  {root.total_tokens()} tokens, "
            f"{sum(1 for _ in _walk(root))} sections, "
            f"{root.total_imperatives()} imperatives\n"
        ).encode("utf-8")
    )
    return 0


def _walk(sec):
    for c in sec.children:
        yield c
        yield from _walk(c)


if __name__ == "__main__":
    raise SystemExit(main())
