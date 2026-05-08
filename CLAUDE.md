# Prompt Sigil — magic-circle protocol for human ⇄ AI communication

This project builds a **new visual language** for talking to LLMs. Instead of
plain text, both sides communicate through a magic-circle figure where every
mark corresponds to real data:

- A **system prompt** is rendered as a deterministic sigil (the structure of
  the prompt becomes the geometry of the figure).
- A **human composes** prompts by clicking typed glyphs onto the circle (no
  free-text needed for routine prompt structure).
- An **LLM's runtime output** is projected back into the same semantic plane
  as a thought trajectory — the user can see where the AI is "drawing from."

The long-term goal is bidirectional: humans draw → AI understands; AI thinks
→ humans see.

## Core principle: functional beauty

Every visual element must encode a real, measured property of the input.
There is no decoration, no "made it look magical." Symmetry, density, and
the magic-circle aesthetic emerge from honest projection, never from styling.

Specifically:
- **Determinism**: same input → identical sigil. Random elements are forbidden.
- **1:1 mapping**: every mark documented in `prompt_sigil/render.py` maps to a
  named property of the parsed prompt or the model's output. Don't add visual
  elements without data behind them.
- **Emergent symmetry**: section vertices are placed by semantic angle and
  laid out into equal slots; the slot-equal-angle move is *itself* a fact
  about the input ("these things are parallel"), not a styling choice.

When in doubt, refuse decoration.

## Architecture

Three components, in order of dependency:

```
prompt_sigil/   pure Python lib — markdown → sigil computation
  parse.py      regex-based markdown parser, extracts measurables (tokens,
                imperatives, bullets, code blocks, refs, depth)
  analyze.py    TF-IDF + PCA-2 projection, deterministic. Returns Layout
                (sections, sentence_2d, similarity, fitted vectorizer/pca).
  render.py     SVG generation. The CLI path. Maintained but the webapp
                duplicates much of this in JS for live updates.
  __main__.py   `python -m prompt_sigil INPUT.md` produces an SVG.

webapp/         tiny stdlib HTTP server + single-file frontend
  server.py     POST /api/layout  (parse + analyze, caches last layout)
                POST /api/project (project arbitrary text into cached PCA)
                GET  /files/...   (serves examples/* and memo.md)
  static/index.html    All UI: SVG sigil, CSS rotation, drawing tools,
                in-browser LLM via transformers.js, scaffolding from
                drawing, typed-glyph composer.

examples/       sample prompts (sample.md, complex.md)
out/            generated .svg artefacts (gitignored material)
```

The frontend has three classes:
- **Sigil** — receives a layout from the server and renders it. Owns the
  rotation group, layers, animation. CSS keyframes drive rotation so it
  survives WASM main-thread blockage.
- **LLM** — `transformers.js` wrapper. Streams generated text, then *replays*
  the buffer post-generation through `/api/project` to draw the thought
  trajectory (live projection during inference is impossible because WASM
  blocks fetch).
- **Sketcher** — human input side. Drawing modes: free-hand strokes
  (interpreted into emphasize/suppress/bridge annotations) and typed-glyph
  click composition (8 section types × 2-3 presets each).

## How to run

The webapp is the demo target. The CLI is for batch artefact generation.

```
# Webapp (preferred)
python webapp/server.py
# → http://localhost:8765
# Note: the runtime sandbox here can't bind ports; ask the user to start it.

# CLI for static SVG
python -m prompt_sigil examples/complex.md
# → out/complex.svg
```

Dependencies are minimal and installed system-wide already:
- `numpy`, `scikit-learn` (for TF-IDF + PCA + cosine_similarity)
- Stdlib only for the server (`http.server`, `json`)
- Frontend pulls `@huggingface/transformers@3.8.1` from jsdelivr

No torch, no node, no build step.

## Visual grammar (locked)

Outer boundary → inner seal, inside the rotation group:

```
R_BOUND_OUT 740   double-line boundary (frame)
R_TOKEN     720   continuous token-tick band (one tick per token)
R_DIVIDER   684   slot dividers (one per section)
R_RIM_TEXT  660   h1 heading text (rim labels)
R_VERTEX    540   inner polygon vertices (one per section)
R_CLOUD     480   sentence-cloud max radius (PCA-projected sentences)
R_SEAL      80    central seal (fixed in world space, doesn't rotate)
```

Section angle = `atan2(PC2, PC1)` of section embedding, then **sorted by
angle** and assigned **equal slots** (avoids pie-chart appearance while
preserving semantic order). The heaviest section (most tokens) is
**rotated to true north** so the figure has a data-justified orientation.

Composed (click-drawn) vertices snap to `R_VERTEX` so the result reads as
a regular magic circle rather than an arbitrary scatter.

## Drawn-glyph vocabulary (Sketcher)

Eight section types, each with a shape + colour + 2-3 preset bodies. Click
a type button, click on the figure to place; click an existing vertex of
the same type to cycle preset. See `SECTION_TYPES` in `static/index.html`.

```
◉ persona     △ reasoning   ⊞ tools     ◇ output
✦ style       ▽ limits      ○ memory    ⌬ examples
```

Free-hand strokes (legacy mode):
- closed shape around section = emphasize
- line through ≥2 sections   = bridge
- two crossing strokes near a section = suppress

`✦ scaffold` converts whatever has been placed/drawn into a markdown
prompt skeleton; the user rarely needs to type more than the user query.

## Known constraints

- **WASM inference blocks the main thread** for ~30-60s per generation on
  small models. CSS animations keep going (compositor), but JS rAF, click
  handlers, and fetch all freeze. The thought trajectory is therefore drawn
  *after* generation completes, replaying the output buffer in 7-8s.
- **TF-IDF cosine of generated text is small** (often 0.01-0.2). Thought
  positions collapse near the centre if you use raw PCA projection. We
  anchor each thought near its top-similarity section vertex instead, with
  a confidence-based radius.
- **Browser cache failures** (`Cannot put response in cache: UnknownError`)
  occasionally drop large model files; transformers.js handles this OK but
  re-download is slow.
- **Default model is `HuggingFaceTB/SmolLM2-135M-Instruct`** (~138MB) —
  smaller models render the demo fast; larger models (Qwen3-0.6B,
  gemma-3-1b, gemma-4-E2B) are available in the picker but require
  patience and stable storage.

## Conventions worth preserving

- Don't reintroduce randomness anywhere in the pipeline. Determinism is the
  marketing claim ("same prompt → same sigil"); breaking it kills the value.
- Don't add visual decoration that isn't tied to data. If you want emphasis,
  encode it via measured density / size / count, never via "looks magical."
- Keep the server stdlib-only (no Flask/FastAPI). The current scope works.
- Don't rewrite `prompt_sigil/parse.py` to use a markdown library — the
  hand-rolled regex parser handles edge cases (Japanese imperatives, the
  fenced-code state machine) the way we want.
- The frontend is one file by design. Resist the urge to split into modules.

## Open directions (in suggested priority)

1. **AI proposes a sigil** — user states a goal in one sentence; the LLM
   proposes a typed-glyph layout the user can accept/edit. Closes the loop
   on "draw the magic circle WITH the AI."
2. **Activation steering** — open-source model + nnsight or TransformerLens;
   each section becomes a steering vector; clicking/dragging a section
   modulates AI behaviour at runtime (not just at prompt construction).
3. **Vertex drag** — let composed vertices move along the rim after
   placement so the user can curate the layout.
4. **Save / share** — encode the sigil state into a URL hash so a circle
   can be sent as a link.
5. **Submission targets** — NeurIPS Creative AI track, GenProCC workshop,
   ARS Electronica. The work sits cleanly between mechanistic
   interpretability viz and generative-art demos.

## Prior art reviewed

- **BertViz / exBERT / Distill Activation Atlas** — diagnostic, not aesthetic.
- **Anthropic SAE / Mapping the Mind** — feature maps, similar spirit but
  scientific framing.
- **Refik Anadol latent-space art** — aesthetic but data is images, not
  prompts; no functional mapping.
- **Procedural sigil generators (Sigil Engine, watabou, CiaccoDavide)** —
  parametric aesthetic, not data-driven.
- **PromptViz** — system prompt → React Flow diagram. Functional but flat.
- **SketchAgent (CVPR 2025), Real-Time AI Drawing System (arXiv 2025)** —
  closest to the future drawing-collaboration vision.

This project's niche is the intersection: aesthetic + data-driven +
input-deterministic. Nothing else covers all three.
