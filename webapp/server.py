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
from prompt_sigil.compress import compress as compress_prompt  # noqa: E402

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


# In-process cache for tensor-scope models. HF transformers loads lazily on
# first request and we hold them in memory thereafter so subsequent forwards
# are sub-second on CPU for small models.
_hf_models: dict = {}
_hf_lock = threading.Lock()


def _load_hf_model(model_id: str):
    with _hf_lock:
        if model_id in _hf_models:
            return _hf_models[model_id]
        # Local imports — keep optional dep contained.
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        import torch
        tok = AutoTokenizer.from_pretrained(model_id)
        # Prefer CausalLM so we get lm_head for logit-lens decoding. Fall back
        # to bare encoder for BERT-family embedding models.
        mdl = None
        head_kind = "none"
        try:
            mdl = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True)
            head_kind = "causal_lm"
        except Exception:
            mdl = AutoModel.from_pretrained(model_id, output_hidden_states=True)
            head_kind = "encoder"
        mdl.eval()
        _hf_models[model_id] = (tok, mdl, torch, head_kind)
        return _hf_models[model_id]


def hidden_states(data: dict) -> dict:
    """Extract per-layer per-token hidden states using HF transformers (PyTorch).

    This bypasses the ONNX export limitation that causes browser-side
    extraction to drop intermediate layers. Returns a dense float32 list of
    all layers so the client can fit a single PCA basis and animate over
    the layer axis — the "true" layer sweep.

    Body: {text, model_id, max_tokens? (default 256)}
    """
    text = str(data.get("text", "")).strip()
    model_id = str(data.get("model_id", "")).strip()
    max_tokens = int(data.get("max_tokens", 256))
    if not text:
        return {"error": "text required"}
    if not model_id:
        return {"error": "model_id required"}
    try:
        tok, mdl, torch, head_kind = _load_hf_model(model_id)
    except Exception as e:  # noqa: BLE001
        return {"error": f"model load failed: {type(e).__name__}: {e}"}

    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=True,
    )
    with torch.no_grad():
        out = mdl(**enc, output_hidden_states=True, return_dict=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        return {"error": "model did not expose hidden_states (unusual for HF transformers)"}
    # hs is a tuple of tensors of shape [1, seq, hidden]
    num_layers = len(hs)
    seq_len = int(hs[0].shape[1])
    hidden_dim = int(hs[0].shape[2])

    # ------------------------------------------------------------------
    # Logit lens — apply the model's output head (or input embedding
    # transposed, when tied) to every layer's hidden state. Each (layer,
    # token) cell gets the top-K most likely tokens at that depth, which
    # is the most direct way to read what the model is "thinking" at
    # each step. nostalgebraist 2020.
    # ------------------------------------------------------------------
    top_k = 3
    lens_layers = []
    lm_head_w = None
    if head_kind == "causal_lm" and hasattr(mdl, "get_output_embeddings"):
        out_emb = mdl.get_output_embeddings()
        if out_emb is not None and hasattr(out_emb, "weight"):
            lm_head_w = out_emb.weight  # [vocab, hidden]
    if lm_head_w is None and hasattr(mdl, "get_input_embeddings"):
        # Tied weights / embedding-only models: use input embedding as the
        # logit lens decoder. This is what nostalgebraist's original probe
        # does for tied-weight models.
        in_emb = mdl.get_input_embeddings()
        if in_emb is not None and hasattr(in_emb, "weight"):
            lm_head_w = in_emb.weight  # [vocab, hidden]
    if lm_head_w is not None:
        with torch.no_grad():
            for layer in hs:
                # h: [1, seq, hidden] → logits: [seq, vocab]
                logits = layer[0].to(lm_head_w.dtype) @ lm_head_w.t()
                topv, topi = torch.topk(logits, k=top_k, dim=-1)
                # Convert to probabilities for display (softmax over vocab is
                # expensive, so use top-k softmax which is the standard cheap
                # logit-lens approximation).
                tk_probs = torch.softmax(topv, dim=-1).to(torch.float32).cpu().numpy()
                tk_ids = topi.to(torch.int64).cpu().numpy().tolist()
                per_token = []
                for t in range(seq_len):
                    cands = []
                    for k in range(top_k):
                        try:
                            tok_str = tok.decode([tk_ids[t][k]], skip_special_tokens=False)
                        except Exception:  # noqa: BLE001
                            tok_str = f"#{tk_ids[t][k]}"
                        cands.append({
                            "token": tok_str,
                            "id": int(tk_ids[t][k]),
                            "prob": float(tk_probs[t][k]),
                        })
                    per_token.append(cands)
                lens_layers.append(per_token)

    layers_flat = []
    for layer in hs:
        arr = layer[0].to(torch.float32).cpu().numpy()
        layers_flat.append(arr.flatten().tolist())

    ids = enc["input_ids"][0].tolist()
    try:
        tokens = [tok.decode([i], skip_special_tokens=False) for i in ids]
    except Exception:  # noqa: BLE001
        tokens = [f"#{i}" for i in ids]

    return {
        "tokens": tokens,
        "token_ids": ids,
        "num_layers": num_layers,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "layers": layers_flat,
        "lens": lens_layers,        # [] if no decoder head available
        "head_kind": head_kind,
        "model_id": model_id,
    }


def pca_project(data: dict) -> dict:
    """Project a (N, D) matrix to (N, k) via PCA / t-SNE / UMAP.

    Body: {vectors, dim, method? = 'pca'|'tsne'|'umap'}
    Returns: {points, method, explained_variance? (PCA only)}

    PCA preserves global structure (linear); t-SNE and UMAP preserve local
    neighbourhoods at the cost of distorting global distances. For "watch
    layers separate cleanly" UMAP is usually best; for "watch a token's
    actual trajectory" PCA is honest.
    """
    from sklearn.decomposition import PCA  # local imports

    vectors = data.get("vectors") or []
    dim = int(data.get("dim", 3))
    method = str(data.get("method", "pca")).lower()
    if not vectors or not isinstance(vectors, list):
        return {"error": "vectors required"}
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        return {"error": f"vectors must be 2D, got shape {arr.shape}"}

    n, d = arr.shape
    if method == "tsne":
        from sklearn.manifold import TSNE
        # perplexity must be < n; clamp.
        perp = max(2.0, min(30.0, (n - 1) / 3.0))
        tsne = TSNE(
            n_components=min(dim, 3),
            perplexity=perp,
            random_state=42,
            init="pca",
            learning_rate="auto",
            n_iter=600 if n < 500 else 1000,
        )
        proj = tsne.fit_transform(arr)
        if proj.shape[1] < dim:
            pad = np.zeros((proj.shape[0], dim - proj.shape[1]), dtype=np.float32)
            proj = np.concatenate([proj, pad], axis=1)
        return {"points": proj.tolist(), "method": "tsne", "explained_variance": []}
    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError:
            return {"error": "umap-learn not installed (pip install umap-learn)"}
        n_neighbors = max(2, min(15, n - 1))
        reducer = umap.UMAP(
            n_components=dim,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        proj = reducer.fit_transform(arr)
        return {"points": proj.tolist(), "method": "umap", "explained_variance": []}
    # default: PCA
    if n < dim:
        pca = PCA(n_components=min(dim, n, d))
        proj = pca.fit_transform(arr)
        if proj.shape[1] < dim:
            pad = np.zeros((proj.shape[0], dim - proj.shape[1]), dtype=np.float32)
            proj = np.concatenate([proj, pad], axis=1)
        ev = list(pca.explained_variance_ratio_)
        ev += [0.0] * (dim - len(ev))
    else:
        pca = PCA(n_components=dim)
        proj = pca.fit_transform(arr)
        ev = list(pca.explained_variance_ratio_)
    return {
        "points": proj.tolist(),
        "method": "pca",
        "explained_variance": [float(v) for v in ev],
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
        if self.path not in ("/api/layout", "/api/project", "/api/compress", "/api/pca", "/api/hidden_states"):
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
            elif self.path == "/api/project":
                response = project_text(str(data.get("text", "")))
            elif self.path == "/api/pca":
                response = pca_project(data)
            elif self.path == "/api/hidden_states":
                response = hidden_states(data)
            else:  # /api/compress
                text = str(data.get("text", ""))
                ratio = float(data.get("ratio", 0.5))
                raw_imp = data.get("token_importance") or {}
                # Normalise to {lowercased token: float weight}
                importance = (
                    {str(k).lower(): float(v) for k, v in raw_imp.items()}
                    if isinstance(raw_imp, dict)
                    else None
                )
                root = parse_markdown(text)
                compressed, stats = compress_prompt(
                    root, ratio=ratio, token_importance=importance,
                )
                response = {
                    "compressed": compressed,
                    "stats": {
                        "original_tokens": stats.original_tokens,
                        "compressed_tokens": stats.compressed_tokens,
                        "ratio": stats.ratio,
                        "original_sentences": stats.original_sentences,
                        "compressed_sentences": stats.compressed_sentences,
                        "imperatives_preserved": stats.imperatives_preserved,
                        "code_blocks_preserved": stats.code_blocks_preserved,
                        "urls_preserved": stats.urls_preserved,
                        "sections": stats.sections,
                    },
                }
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
