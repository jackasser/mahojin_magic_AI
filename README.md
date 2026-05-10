# Prompt Sigil

> A visual protocol for talking to AI.
> Build prompts by drawing, see the model's thinking projected back into the
> same figure.

<p align="center">
  <img src="docs/demo.gif" alt="Prompt Sigil rotating with all 35 sections of a complex system prompt" width="520">
</p>

A 1040-token system prompt rendered into 35 angular slots, with cosine
similarities forming the inner polygon. The figure rotates slowly via CSS
keyframes so the animation continues even when the in-browser LLM blocks
the main thread.

| 1. blank | 2. place glyphs | 3. bridge | 4. conjure |
|:---:|:---:|:---:|:---:|
| ![](docs/01-blank.png) | ![](docs/02-typed.png) | ![](docs/03-bridged.png) | ![](docs/04-conjured.png) |
| empty canvas | click `◉ △ ⊞ ▽ ✦` to place typed sections (snap to rim) | click two vertices to bridge | scaffold + conjure → real sigil with k-NN polygon |

The full UI with the gold thought trail drawn after a generation:
![Prompt Sigil web app, complex prompt with thought trajectory](docs/preview.png)

## What this is

A working prototype that turns a system prompt into a deterministic
**magic-circle figure** where every visual element corresponds to a real
property of the prompt — and where the LLM's runtime output is projected
back into the same semantic plane as a glowing thought trajectory.

The aim is a **new medium for human ⇄ AI communication**:

- **Human → AI**: place typed glyphs on a circle (persona, reasoning, tools,
  limits, …) instead of writing prose. The geometry encodes intent;
  the user types almost nothing.
- **AI → Human (output)**: as the model generates, its output is projected
  into the prompt's semantic space and drawn as a trail across the figure.
  You can *see* which parts of the prompt the model is drawing from at
  each moment.
- **AI → Human (interior)**: open the **tensor scope** to peer inside the
  model itself. The hidden state of every token at every layer becomes a
  petal of a layered mandala, and a logit-lens decoder shows what word
  the model "thinks" each token is at each depth. See the representation
  morph from raw embedding into meaning, layer by layer.

You can also reduce a prompt to a **minimal sigil-equivalent form**: the
compressor keeps the contract-bearing parts (headings, code, imperatives,
URLs) verbatim and ranks remaining prose by importance, so two prompts
that produce similar sigils now also produce similar markdown.

## Why a magic circle

Magic circles are an old visual language: an outer boundary, ringed
inscriptions, an inner inscribed figure, a central seal. They are designed
for dense, parallel reading at a glance. We adopt the grammar but discard
the mysticism — every mark on this circle is a measured fact about the
prompt or the model:

- Section vertex angle = atan2 of the section's PCA-2 embedding.
- Slot width = equal (sections are parallel things, not weighted by tokens).
- Tick band density = token count.
- Inner polygon edges = pairwise cosine similarity between sections.
- Sentence cloud = each sentence as a 2D point in the same PCA basis.
- Thought dots (gold) = generated text projected through the same basis.

Same input always produces the same sigil. There is no randomness, no
"styling for effect."

## Try it

Requirements: Python 3.10+, `numpy`, `scikit-learn`. No Node, no GPU, no
build step. The tensor scope additionally needs `torch` + `transformers`
(CPU-only is fine) and optionally `umap-learn`.

```bash
pip install -r requirements.txt

# Webapp (the demo)
python webapp/server.py
# → open http://localhost:8765

# CLI (static SVG)
python -m prompt_sigil examples/complex.md
# → out/complex.svg

# CLI (compress; protected content survives any ratio)
python -m prompt_sigil compress examples/complex.md --ratio 0.4 --render-pair
# tokens     1040 -> 493  (47.4%)
# preserved  23 imperatives, 5 code blocks, 2 urls
# also writes before/after sigils for visual diff
```

The webapp runs entirely on your machine. The optional in-browser LLM
(SmolLM2 / Qwen3 / Gemma 3 / Gemma 4) is downloaded once into IndexedDB
via [transformers.js](https://github.com/huggingface/transformers.js) and
runs locally afterward — no API keys, no telemetry.

## How to use the webapp

1. Click **⌂ blank** to start with an empty canvas.
2. Pick a glyph: **◉** persona, **△** reasoning, **⊞** tools, **◇** output,
   **✦** style, **▽** limits, **○** memory, **⌬** examples.
3. Click on the figure to place a typed vertex (it snaps to the rim).
4. Click an existing vertex of the same type to cycle its preset.
5. Use **↔ bridge** to connect two vertices semantically.
6. Click **✦ scaffold** — the markdown system prompt is generated for you.
7. Click **conjure** to render the real sigil from that prompt.
8. Open **live inference**, load a model (SmolLM2-135M is the fastest first
   download), type a question, click **infer**.
9. Watch the gold thought trajectory draw itself across the circle.

You can also free-hand draw on top of an existing sigil:

- Closed shape around a section → emphasize it in the next query.
- Line through ≥2 sections → bridge them.
- Two crossing strokes near a section → suppress it.

The drawn intent is appended to your query before inference.

## Compression

Click **⊿ compress** in the sidebar (or run the CLI subcommand) to
reduce a prompt to a minimal sigil-equivalent form. The compressor:

- Keeps **verbatim**: headings, fenced code blocks (with language tag),
  URLs and file paths, and any sentence containing an imperative
  (`MUST`/`NEVER`/`必ず`/...). These are the contract-bearing parts —
  losing them changes LLM behaviour.
- **Ranks** the remaining prose by TF-IDF mass (plus an optional
  per-token importance map) and keeps the top fraction needed to hit
  the target ratio.

The output is real markdown with the original section structure intact;
it is itself a usable system prompt, not a summary. The realised ratio
can exceed the target when protected content dominates — semantic
preservation wins over hitting an exact number.

## Tensor scope

A diagnostic window into the model's interior. Open the **⌗ tensor scope**
panel and click **extract**: the server runs one forward pass with
`output_hidden_states=True` (PyTorch + HF transformers, bypassing the
ONNX-export limitation that would otherwise drop intermediate layers),
projects all `(layer × token)` hidden states through a single shared
PCA / UMAP / t-SNE basis, and renders one of two views.

**Mandala view** is the magic-circle reading. Concentric rings = layers
(outer = embedding, inner = output). Each token sits at a fixed angle
around the rim; following its radial spoke shows how its representation
travels through depth. Per-layer normalisation keeps the colours vibrant
even when one PCA axis dominates the global variance.

**Scatter view** is a 3D point cloud through PCA space, with each token's
trajectory as a polyline. Useful for spotting clusters; less for following
a single token.

In either view, every cell carries its own meaning:

- **Hover**: tooltip with token, layer, the three PC values, and the
  top-k logit-lens predictions for that cell ([nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) — apply the model's own output head to every layer's
  hidden state to read what word it would predict at that depth).
- **Click a token** to follow its spoke through every layer. The selected
  token's full D-dimensional hidden state is then drawn as a **morphing
  radial shape** below the mandala — every dim is one petal, the shape
  reshapes smoothly as the layer slider moves, and a synchronised bar
  chart shows the model's top-3 predictions reshuffling. The vector is
  the figure.
- **Click a layer ring** to jump to that layer.
- **Scroll** to zoom (anchored to the cursor), **drag** to pan,
  **double-click** to reset.

The tensor scope is intentionally decoupled from compression. It is a
viewer for human inspection, not a pipeline stage; nothing flows back
into prompt processing. (A future version may bias compression by
hidden-state activation norms via the `token_importance` hook — see
`webapp/static/index.html` for the wiring.)

## Architecture

```
prompt_sigil/   pure Python: markdown → measured tree → 2D projection
                  parse.py     tokens, imperatives, bullets, code, refs
                  analyze.py   TF-IDF + PCA-2 (deterministic), cosine sim
                  render.py    SVG output (CLI path)
                  compress.py  importance-scored extractive compression

webapp/         tiny stdlib HTTP server + single-file frontend
                  server.py    /api/{layout,project,compress,pca,
                               hidden_states,files}
                               (HF transformers used only by hidden_states
                                — keeps the base demo torch-free)
                  static/index.html
                    Sigil class       rotating SVG renderer + animations
                    LLM class         transformers.js streaming + replay
                    Sketcher class    freehand strokes + typed-glyph composer
                    TensorScope class hidden-state mandala / scatter,
                                       logit lens, vector morph, zoom-pan

examples/       sample prompts (sample.md, complex.md)
out/            generated SVGs
```

Four frontend classes, one file, no framework.

## Status

Working today (May 2026):

- Deterministic prompt → sigil rendering with rich inscriptions.
- 8-glyph composer for click-only prompt building.
- In-browser inference (SmolLM2 / Qwen3 / Gemma 3 / Gemma 4 via ONNX).
- Thought-trajectory projection of generated text into prompt PCA space.
- CSS-driven rotation that survives WASM main-thread blockage.
- Free-hand stroke interpretation (emphasize / suppress / bridge).
- One-click prompt scaffolding from typed-glyph composition.
- Extractive prompt compression that preserves sigil semantics.
- Tensor scope: per-layer per-token mandala with PCA / UMAP / t-SNE
  projection, zoom-pan, hover tooltips, and click-to-follow.
- Logit lens decoding for causal-LM scope models — read what the model
  thinks each token is at each layer.
- Selected-token morph: raw D-dim hidden state drawn as a closed radial
  shape that visibly reshapes between layers, synced with a top-k bar
  chart of the model's predictions.

Not built yet, but desired (PRs welcome — see the
[roadmap](#roadmap-help-wanted) below).

## Functional beauty (the rule we never break)

Every visual element is a 1:1 mapping to a measured property of the
input. Tables in [`CLAUDE.md`](./CLAUDE.md) and the comment block at the
top of `webapp/static/index.html` enumerate every mark. If you propose a
new visual feature, you must say what data it encodes; "looks magical"
is not an acceptable justification.

This constraint is what makes the figure read as a magic circle rather
than as decoration: dense rotational structure plus intentional emptiness
emerges from the data, never from styling.

## Roadmap (help wanted)

The most interesting open directions are listed here in priority order.
Each is a meaningful contribution; pick one and open a PR.

1. **AI proposes a sigil**. The user states a goal in one sentence; an
   LLM proposes a typed-glyph layout the user can accept or edit. Closes
   the loop on "draw the magic circle *with* the AI."

2. **Activation steering integration**. Each section becomes a steering
   vector; clicking/dragging a section modulates AI behaviour at runtime,
   not just at prompt-construction time. Closest prior art:
   [Feature Guided Activation Additions](https://arxiv.org/abs/2501.09929),
   [Conceptors](https://openreview.net/forum?id=0Yu0eNdHyV).

3. **Vertex drag**. Allow composed vertices to be dragged along the rim
   after placement so the user can curate the angular layout.

4. **Save and share**. Encode sigil state into a URL hash so a circle can
   be sent as a link, then loaded and remixed.

5. **Voice glyphs**. One-tap speak-then-attach so users can label a
   section by speaking instead of cycling presets.

6. **Multilingual presets**. The eight section types currently have
   English preset bodies. Translations welcome.

7. **Web Worker for inference**. Move transformers.js into a worker so
   the main thread stays responsive during generation. Live (not replay)
   thought-trajectory drawing becomes possible.

## Prior art

This project sits at the intersection of three traditions:

- **Mechanistic interpretability**: BertViz, exBERT, Anthropic's
  [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/),
  Distill's [Activation Atlas](https://distill.pub/2019/activation-atlas/).
  Diagnostic and rigorous, rarely beautiful.

- **Latent-space art**: Refik Anadol's
  [Unsupervised](https://refikanadol.com/works/unsupervised/),
  [Data Universe](https://www.moma.org/) at MoMA. Beautiful but the data
  is images, and the projection isn't designed to be functionally read.

- **Procedural sigil generators**:
  [Sigil Engine](https://www.sigilengine.com/),
  [Alchemy Circles Generator](https://github.com/CiaccoDavide/Alchemy-Circles-Generator).
  Parametric aesthetics, not tied to real data.

- **Human–AI co-creation**: 
  [SketchAgent (MIT/Stanford CVPR 2025)](https://news.mit.edu/2025/teaching-ai-models-to-sketch-more-like-humans-0602),
  [Real-Time AI Drawing System (arXiv 2025)](https://arxiv.org/abs/2508.19254).
  Closest to the drawing-collaboration vision.

The unique niche of this project: aesthetic + data-driven + input-deterministic.

## Contributing

Anyone is welcome. The codebase is small (≈ 2500 lines total) and uses no
framework or build pipeline; you should be able to read every line in one
sitting.

A few non-negotiables:

- **Determinism**: same input → same sigil. No randomness in the pipeline.
- **No decorative-only visual elements**: if it isn't tied to a measured
  property, it doesn't go in.
- **Stdlib-only Python server**: don't pull Flask/FastAPI; the current
  scope fits in `http.server`.
- **Single-file frontend**: don't introduce a build step.

If you want to discuss a larger change before coding, open an issue
describing the data → visual mapping you propose. Every feature negotiation
starts there.

## License

MIT — see [LICENSE](./LICENSE).

## Citation

If you use this work in research, please cite:

```
@misc{prompt-sigil-2026,
  title  = {Prompt Sigil: A Magic-Circle Protocol for Human--AI Communication},
  year   = {2026},
  url    = {https://github.com/jackasser/mahojin_magic_AI}
}
```

---

*Prompt Sigil is an open invitation: a small protocol that anyone can
extend. The goal is not a polished product but a shared visual language
that grows as more people contribute glyphs, mappings, and ways of
listening to what the AI is saying back.*
