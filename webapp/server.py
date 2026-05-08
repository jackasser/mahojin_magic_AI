"""Tiny stdlib HTTP server for the prompt-sigil web app.

Serves static files from webapp/static and one POST endpoint
/api/layout that returns the deterministic layout for an input markdown.
"""
from __future__ import annotations

import http.server
import json
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from prompt_sigil.parse import parse_markdown  # noqa: E402
from prompt_sigil.analyze import Layout, compute_layout  # noqa: E402

STATIC = ROOT / "static"

# Last layout produced by /api/layout — cached so /api/project can reuse the
# same TF-IDF basis + PCA for live token projection. Single-user demo, so a
# single global is fine.
_layout_lock = threading.Lock()
_current_layout: Optional[Layout] = None


def _sub_count(sec) -> int:
    return len(sec.children) + sum(_sub_count(c) for c in sec.children)


def compute_response(text: str, title: str) -> dict:
    global _current_layout
    root = parse_markdown(text)
    layout = compute_layout(root)
    if layout is None:
        return {"error": "empty input"}
    with _layout_lock:
        _current_layout = layout

    sections = []
    for i, s in enumerate(layout.sections):
        sections.append({
            "heading": s.heading,
            "depth": s.depth,
            "tokens": s.total_tokens(),
            "imperatives": s.total_imperatives(),
            "bullets": s.bullets,
            "code_blocks": s.code_blocks,
            "refs": s.urls + s.refs,
            "subsections": _sub_count(s),
            "pos_2d": [float(layout.section_2d[i, 0]), float(layout.section_2d[i, 1])],
        })

    sentences = []
    for i in range(len(layout.sentence_texts)):
        unit = layout.sentence_texts[i]
        sentences.append({
            "text": unit[:280],
            "section_idx": int(layout.sentence_section_idx[i]),
            "pos_2d": [float(layout.sentence_2d[i, 0]), float(layout.sentence_2d[i, 1])],
            "tokens": len(unit.split()),
        })

    return {
        "title": title,
        "sections": sections,
        "sentences": sentences,
        "similarity": layout.similarity.tolist(),
        "totals": {
            "tokens": root.total_tokens(),
            "sections": len(sections),
            "imperatives": root.total_imperatives(),
            "bullets": root.total_bullets(),
            "refs": root.total_refs(),
        },
    }


def project_text(text: str) -> dict:
    """Project arbitrary text into the cached layout's PCA space.

    Uses the same fitted TfidfVectorizer + PCA, so the new point lands in
    the same coordinate frame as the prompt's sentences. Returns the 2D
    position plus cosine similarity to each section.
    """
    with _layout_lock:
        layout = _current_layout
    if layout is None:
        return {"error": "no layout computed yet — POST /api/layout first"}
    text = (text or "").strip()
    if not text:
        return {"error": "empty text"}

    X = layout.vectorizer.transform([text])
    if layout.pca is not None:
        pos = layout.pca.transform(X.toarray())[0]
        pos_2d = [float(pos[0]), float(pos[1])]
    else:
        pos_2d = [0.0, 0.0]

    if layout.section_X is not None and layout.section_X.shape[0] > 0:
        sims = cosine_similarity(X, layout.section_X).flatten()
        section_similarity = [float(s) for s in sims]
        top = int(np.argmax(sims))
        top_score = float(sims[top])
    else:
        section_similarity = []
        top = -1
        top_score = 0.0

    return {
        "pos_2d": pos_2d,
        "section_similarity": section_similarity,
        "top_section": top,
        "top_score": top_score,
    }


PROJECT_ROOT = ROOT.parent
EXAMPLES_DIR = (PROJECT_ROOT / "examples").resolve()
MEMO_FILE = (PROJECT_ROOT / "memo.md").resolve()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self) -> None:
        from urllib.parse import unquote
        if self.path.startswith("/files/"):
            rel = unquote(self.path[len("/files/"):])
            full = (PROJECT_ROOT / rel).resolve()
            allowed = (
                full == MEMO_FILE
                or (
                    full.is_file()
                    and EXAMPLES_DIR in full.parents
                    and full.suffix.lower() == ".md"
                )
            )
            if not allowed:
                self.send_error(403)
                return
            data = full.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path not in ("/api/layout", "/api/project"):
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return
        try:
            if self.path == "/api/layout":
                text = str(data.get("text", ""))
                title = str(data.get("title", "prompt"))
                response = compute_response(text, title)
            else:  # /api/project
                response = project_text(str(data.get("text", "")))
        except Exception as e:  # noqa: BLE001
            response = {"error": f"{type(e).__name__}: {e}"}
        payload = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.buffer.write(
            (f"{self.address_string()} - {fmt % args}\n").encode("utf-8", "replace")
        )


def main() -> None:
    port = 8765
    addr = ("127.0.0.1", port)
    sys.stdout.buffer.write(
        f"prompt-sigil webapp on http://{addr[0]}:{port}\n".encode("utf-8")
    )
    sys.stdout.buffer.write(f"static dir: {STATIC}\n".encode("utf-8"))
    sys.stdout.flush()
    with http.server.ThreadingHTTPServer(addr, Handler) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()
