# Announcement drafts

Drafts for posting the project to different channels. Each is calibrated to
the platform's norms (length, register, technical depth). Edit before
posting; do not paste verbatim.

---

## X / Twitter (≤ 280 chars)

> Built a visual protocol for talking to AI.
>
> System prompts → magic-circle figures (every mark = real measured data).
> The LLM's output is projected back into the same plane as a thought trail.
>
> Click glyphs to compose; almost no typing.
>
> https://github.com/jackasser/mahojin_magic_AI

---

## X / Twitter — alt (more provocative)

> What if you didn't write prompts as text — you drew them as a magic circle?
>
> Same input → same sigil. The model's thinking traces a gold path across
> the figure as it reads. ~2500 lines, no framework, runs locally.
>
> https://github.com/jackasser/mahojin_magic_AI

---

## Bluesky (≤ 300 chars)

> A small experiment: what if a system prompt looked like a magic circle?
>
> Every section becomes a vertex placed by semantic angle. Generated text
> is projected back into the same plane as a glowing trail. Clicks, not
> typing, build the prompt.
>
> Open source, local LLM, no API keys.
>
> github.com/jackasser/mahojin_magic_AI

---

## Hacker News — Show HN

> **Show HN: Prompt Sigil — a magic-circle protocol for human ⇄ AI talk**
>
> I've been playing with the idea that a system prompt is more legible as a
> figure than as text — that the magic-circle iconography (outer boundary,
> inscriptions, inner inscribed polygon, central seal) is actually a dense
> parallel-reading interface that we should re-use.
>
> The result is **Prompt Sigil**: a deterministic renderer that turns any
> markdown prompt into a magic-circle figure where every visual element
> corresponds to a real measured property of the prompt (token counts,
> imperative density, cosine similarity between sections, PCA-2 of TF-IDF
> sentence embeddings). Same input, same sigil, no randomness.
>
> The fun part: as a local LLM (SmolLM2 / Qwen3 / Gemma 3 / Gemma 4 in the
> browser via transformers.js) generates a response, its output is projected
> back into the prompt's PCA basis and drawn on top as a gold thought trail.
> You can see which sections of the prompt the model is "drawing from" at
> each moment.
>
> The flip side: a click-only composer where you place 8 typed glyphs
> (persona / reasoning / tools / output / style / limits / memory / examples)
> on the figure, optionally bridge them, and a markdown system prompt is
> generated for you. You barely type at all.
>
> Stack: ~2500 lines total. Pure-stdlib Python server (no Flask/FastAPI),
> single-file vanilla-JS frontend (no build step), numpy + scikit-learn
> for the math, transformers.js for in-browser inference. MIT.
>
> The README has a 4-panel walkthrough; the demo runs entirely on your
> machine after a one-time ~138MB model download.
>
> Repo: https://github.com/jackasser/mahojin_magic_AI
>
> The project sits between mechanistic-interpretability viz (BertViz,
> Anthropic's SAE work) and generative latent-space art (Anadol-style),
> but constrained by a strict rule: every visual element must encode a
> measurable property of the input. No decoration.
>
> Roadmap and contribution areas are in the README — drag-to-edit, AI-proposed
> sigils, activation-steering integration, web-worker inference, multilingual
> presets. PRs welcome.

---

## Reddit /r/MachineLearning — [P] Project

> **[P] Prompt Sigil — visualising prompts and LLM cognition as a single
> magic-circle figure**
>
> A working prototype that does two things in one figure:
>
> 1. **Prompt → sigil.** Markdown system prompt is parsed (sections,
>    imperatives, bullets, code blocks, refs), embedded with TF-IDF + PCA-2,
>    and rendered as a deterministic magic-circle figure. Section vertices
>    are placed by semantic angle and assigned equal angular slots; cosine
>    similarities form an inner inscribed polygon; every token of the prompt
>    becomes a tick on the rim band.
>
> 2. **LLM output → trajectory in the same plane.** A local LLM (in-browser,
>    via transformers.js) streams tokens; chunks are projected through the
>    *same* TF-IDF + PCA basis and drawn as gold dots connected by a thought
>    trail. You can see where in the prompt the model is drawing from.
>
> Plus a click-based composer (8 typed glyphs, 2-3 preset bodies each) so
> you can build a system prompt by placing shapes on a circle without
> writing prose.
>
> Why magic-circle aesthetic specifically: it's an old visual language for
> dense parallel reading. We reuse the grammar (boundary, inscriptions,
> inner polygon, seal) but every mark must encode a real measurable
> property of the input. No decoration.
>
> Stack is small: stdlib Python server, single-file vanilla-JS frontend,
> numpy/sklearn for the math. ~2500 lines. MIT.
>
> Repo (with screenshots and 4-panel walkthrough):
> https://github.com/jackasser/mahojin_magic_AI
>
> Open contribution areas in the README:
> - AI-proposed sigil layout from a one-line goal
> - Activation-steering integration (each section becomes a steering vector)
> - Web-worker inference so the trajectory can be drawn live, not on replay
> - Vertex drag, save-and-share via URL hash
> - Multilingual section presets

---

## Discord / Slack short blurb

> just open-sourced a thing — system prompts visualised as magic circles
> (every mark = real measured data), and the LLM's output draws a gold
> trail across the same figure as it generates. you click glyphs to
> compose; almost no typing.
>
> github.com/jackasser/mahojin_magic_AI

---

## Headline alternatives (pick one)

- "A magic-circle protocol for human ⇄ AI communication"
- "Prompts as figures, not paragraphs"
- "What if your system prompt looked like a magic circle?"
- "Click glyphs, not words: a visual protocol for LLMs"
- "Magic circles for the age of language models"

---

## What to attach

- `docs/preview.png` — full-UI hero shot (best for OG card / Twitter card)
- `docs/02-typed.png` — clean shot of typed-glyph composition (best for the
  "click to compose" pitch)
- `docs/04-conjured.png` — minimal black-line magic circle (best for the
  aesthetic / functional-beauty pitch)
- A 5-10s GIF of a generation cycle with the gold trail drawing — recommended
  but not required, post will land without it
