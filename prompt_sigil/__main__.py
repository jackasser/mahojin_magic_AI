"""CLI: render or compress a markdown prompt.

Usage:
    python -m prompt_sigil INPUT.md [-o OUTPUT.svg] [-t TITLE]
    python -m prompt_sigil compress INPUT.md [--ratio 0.5] [-o OUT.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compress import compress
from .parse import parse_markdown
from .render import render_sigil


def _walk(sec):
    for c in sec.children:
        yield c
        yield from _walk(c)


def _cmd_render(args) -> int:
    src: Path = args.input
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


def _cmd_compress(args) -> int:
    src: Path = args.input
    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr)
        return 2
    text = src.read_text(encoding="utf-8")
    root = parse_markdown(text)

    importance = None
    if args.importance:
        importance = json.loads(Path(args.importance).read_text(encoding="utf-8"))
        # normalise keys
        importance = {str(k).lower(): float(v) for k, v in importance.items()}

    compressed, stats = compress(root, ratio=args.ratio, token_importance=importance)

    out_md = args.output or Path("out") / f"{src.stem}.min.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(compressed, encoding="utf-8")

    sys.stdout.buffer.write(f"wrote {out_md}\n".encode("utf-8"))
    sys.stdout.buffer.write((stats.report() + "\n").encode("utf-8"))

    if args.render_pair:
        # Side-by-side sigils for visual diff
        orig_svg = render_sigil(root, title=src.stem)
        comp_root = parse_markdown(compressed)
        comp_svg = render_sigil(comp_root, title=src.stem + ".min")
        a = Path("out") / f"{src.stem}.sigil.svg"
        b = Path("out") / f"{src.stem}.min.sigil.svg"
        a.write_text(orig_svg, encoding="utf-8")
        b.write_text(comp_svg, encoding="utf-8")
        sys.stdout.buffer.write(f"wrote {a}\nwrote {b}\n".encode("utf-8"))

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prompt_sigil")
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("render", help="render markdown as sigil SVG (default)")
    pr.add_argument("input", type=Path)
    pr.add_argument("-o", "--output", type=Path, default=None)
    pr.add_argument("-t", "--title", default=None)
    pr.set_defaults(func=_cmd_render)

    pc = sub.add_parser("compress", help="compress prompt while preserving sigil semantics")
    pc.add_argument("input", type=Path)
    pc.add_argument("--ratio", type=float, default=0.5,
                    help="target prose retention ratio (0..1). Protected content always survives.")
    pc.add_argument("-o", "--output", type=Path, default=None)
    pc.add_argument("--importance", type=Path, default=None,
                    help="JSON file: {token: weight}. Used to bias retention "
                         "(e.g. hidden-state norms from transformers.js).")
    pc.add_argument("--render-pair", action="store_true",
                    help="also write original and compressed sigils for visual diff")
    pc.set_defaults(func=_cmd_compress)

    # Backward-compat: bare path means render. argparse subparsers are strict
    # about the first positional, so we detect "no known subcommand" ourselves.
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] not in {"render", "compress", "-h", "--help"}:
        return _cmd_render(pr.parse_args(raw))
    args = p.parse_args(raw)
    if args.cmd is None:
        p.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
