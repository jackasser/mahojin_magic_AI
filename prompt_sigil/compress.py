"""Compress a markdown prompt while preserving sigil-equivalent semantics.

Compression principle: drop sentences that contribute little to the sigil's
measurables, keep everything that is load-bearing for LLM behaviour.

Always-preserved (verbatim):
  - headings (sigil rim labels)
  - code blocks (truth-bearing, often part of the contract)
  - URLs and file paths (references the model needs)
  - sentences containing imperatives (MUST/NEVER/必ず ...)

Score-ranked (top-k retained per section to hit target ratio):
  - TF-IDF mass: distinctive vocabulary survives, filler doesn't
  - optional external token_importance map: per-token weights from any source
    (e.g. hidden-state norms from transformers.js in the webapp). Keys are
    lowercased tokens; values are float weights summed per sentence.

Output is a real markdown file with the original section structure intact,
so the compressed prompt is itself a usable system prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from .analyze import flatten
from .parse import BULLET_RE, IMPERATIVE_RE, PATH_RE, URL_RE, Section

_TOKEN_RE = re.compile(r"\b\w[\w'-]*\b", re.UNICODE)
_SENT_END_RE = re.compile(r"(?<=[.!?。!?])\s+")


def _split_units(text: str) -> List[str]:
    """Split prose into sentences without confusing soft line wraps for breaks.

    Markdown source often wraps long sentences across lines. The analyzer's
    splitter (which is fine for visualisation) breaks on every \\n; for
    compression that produces dangling fragments. Here we rejoin soft wraps
    paragraph-by-paragraph, then split on sentence terminators only.
    """
    units: List[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # Bullets are independent sentences; wrapped continuation lines
        # (indented) attach to whatever came before them.
        lines = paragraph.splitlines()
        chunks: List[str] = []
        buf: List[str] = []
        last_was_bullet = False

        def flush() -> None:
            if buf:
                chunks.append(" ".join(s.strip() for s in buf if s.strip()))
                buf.clear()

        for ln in lines:
            if BULLET_RE.match(ln):
                flush()
                buf.append(BULLET_RE.sub("", ln))
                last_was_bullet = True
            elif ln.startswith((" ", "\t")) and (last_was_bullet or buf):
                # Continuation of the previous bullet or paragraph line.
                buf.append(ln)
            else:
                if last_was_bullet:
                    flush()
                    last_was_bullet = False
                buf.append(ln)
        flush()

        for chunk in chunks:
            for sent in _SENT_END_RE.split(chunk):
                sent = sent.strip()
                if len(sent) > 2:
                    units.append(sent)
    return units


@dataclass
class CompressionStats:
    original_tokens: int
    compressed_tokens: int
    original_sentences: int
    compressed_sentences: int
    sections: int
    imperatives_preserved: int
    code_blocks_preserved: int
    urls_preserved: int

    @property
    def ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens

    def report(self) -> str:
        return (
            f"  tokens     {self.original_tokens:>5} -> {self.compressed_tokens:<5} "
            f"({self.ratio * 100:.1f}%)\n"
            f"  sentences  {self.original_sentences:>5} -> {self.compressed_sentences:<5}\n"
            f"  preserved  {self.imperatives_preserved} imperatives, "
            f"{self.code_blocks_preserved} code blocks, {self.urls_preserved} urls\n"
            f"  sections   {self.sections}"
        )


def _sentence_score(
    sentence: str,
    tfidf_vec,
    vectorizer,
    token_importance: Optional[Dict[str, float]],
) -> float:
    """Higher score = more load-bearing. Pure function of measurable signals."""
    score = 0.0
    if IMPERATIVE_RE.search(sentence):
        score += 1000.0  # always retained, but rank above non-imperatives anyway
    if URL_RE.search(sentence) or PATH_RE.search(sentence):
        score += 50.0
    if vectorizer is not None and tfidf_vec is not None:
        score += float(tfidf_vec.sum()) * 10.0
    if token_importance:
        toks = _TOKEN_RE.findall(sentence.lower())
        score += sum(token_importance.get(t, 0.0) for t in toks)
    # Slight penalty for very long sentences so the compressor prefers density
    score -= 0.1 * len(sentence)
    return score


def _is_protected(sentence: str) -> bool:
    """Sentences that must survive at any compression ratio."""
    return bool(
        IMPERATIVE_RE.search(sentence)
        or URL_RE.search(sentence)
        or PATH_RE.search(sentence)
    )


def _emit_section(
    sec: Section,
    keep_per_section: Dict[int, List[int]],
    sentences_per_section: Dict[int, List[str]],
    sec_index: Dict[int, int],
    out: List[str],
) -> Tuple[int, int]:
    """Walk a section, emit kept content. Returns (kept_tokens, kept_sentences)."""
    kept_tokens = 0
    kept_sentences = 0
    if sec.heading and sec.heading != "(untitled)":
        out.append("#" * sec.depth + " " + sec.heading)
        out.append("")
    idx = sec_index.get(id(sec))
    if idx is not None:
        keep_set = set(keep_per_section.get(idx, []))
        sents = sentences_per_section.get(idx, [])
        # Iterate body blocks in document order; for prose, keep only ranked
        # sentences; for code, always emit verbatim.
        sent_cursor = 0
        for kind, lang, content in sec.body:
            if kind == "code":
                fence = "```" + (lang or "")
                out.append(fence)
                out.append(content)
                out.append("```")
                out.append("")
                continue
            # prose block: split into sentences, keep those ranked in this section
            block_sents = _split_units(content)
            for bs in block_sents:
                # Match by index across all prose sentences of this section
                if sent_cursor < len(sents) and sents[sent_cursor] == bs:
                    if sent_cursor in keep_set:
                        out.append(bs)
                        out.append("")
                        kept_tokens += len(bs.split())
                        kept_sentences += 1
                    sent_cursor += 1
    for child in sec.children:
        kt, ks = _emit_section(
            child, keep_per_section, sentences_per_section, sec_index, out
        )
        kept_tokens += kt
        kept_sentences += ks
    return kept_tokens, kept_sentences


def compress(
    root: Section,
    ratio: float = 0.5,
    token_importance: Optional[Dict[str, float]] = None,
) -> Tuple[str, CompressionStats]:
    """Compress the prompt to roughly `ratio` of its original prose token count.

    Code blocks, URLs, and imperative sentences bypass the ratio (they are
    always retained), so the realised ratio may exceed `ratio` for prompts
    dense in protected content. That's the correct behaviour: we prefer
    semantic preservation over hitting an exact number.
    """
    sections = flatten(root)
    if not sections:
        return "", CompressionStats(0, 0, 0, 0, 0, 0, 0, 0)

    # Per-section sentence list (prose only, in document order).
    sentences_per_section: Dict[int, List[str]] = {}
    sec_index: Dict[int, int] = {}
    for i, s in enumerate(sections):
        sec_index[id(s)] = i
        sents: List[str] = []
        for kind, _, content in s.body:
            if kind == "prose":
                sents.extend(_split_units(content))
        sentences_per_section[i] = sents

    all_sentences = [s for sents in sentences_per_section.values() for s in sents]
    vectorizer = None
    if all_sentences:
        try:
            vectorizer = TfidfVectorizer(
                max_features=4000,
                stop_words="english",
                ngram_range=(1, 1),
                min_df=1,
                token_pattern=r"(?u)\b\w[\w'-]*\b",
            )
            vectorizer.fit(all_sentences)
        except ValueError:
            vectorizer = None

    keep_per_section: Dict[int, List[int]] = {}
    for i, sents in sentences_per_section.items():
        if not sents:
            keep_per_section[i] = []
            continue
        if vectorizer is not None:
            try:
                tfidf = vectorizer.transform(sents)
            except ValueError:
                tfidf = None
        else:
            tfidf = None
        scored = []
        for j, s in enumerate(sents):
            row = tfidf[j] if tfidf is not None else None
            scored.append((j, _sentence_score(s, row, vectorizer, token_importance)))
        # Always keep protected sentences; rank the rest.
        protected = [j for j, _ in scored if _is_protected(sents[j])]
        unprotected = [(j, sc) for j, sc in scored if j not in protected]
        unprotected.sort(key=lambda kv: -kv[1])
        target = max(len(protected), int(round(len(sents) * ratio)))
        slack = max(0, target - len(protected))
        kept = sorted(set(protected + [j for j, _ in unprotected[:slack]]))
        keep_per_section[i] = kept

    out_lines: List[str] = []
    kept_tokens, kept_sentences = 0, 0
    for child in root.children:
        kt, ks = _emit_section(
            child, keep_per_section, sentences_per_section, sec_index, out_lines
        )
        kept_tokens += kt
        kept_sentences += ks

    # Add code-block tokens (verbatim preserved) into kept_tokens for honesty.
    code_blocks = 0
    urls = 0
    imperatives = 0
    for s in sections:
        for kind, _, content in s.body:
            if kind == "code":
                code_blocks += 1
                kept_tokens += len(content.split())
        urls += s.urls
        imperatives += s.total_imperatives() if s is sections[0] else s.imperatives

    original_tokens = root.total_tokens()
    original_sentences = sum(len(v) for v in sentences_per_section.values())

    text = "\n".join(out_lines).rstrip() + "\n"
    stats = CompressionStats(
        original_tokens=original_tokens,
        compressed_tokens=kept_tokens,
        original_sentences=original_sentences,
        compressed_sentences=kept_sentences,
        sections=len(sections),
        imperatives_preserved=root.total_imperatives(),
        code_blocks_preserved=code_blocks,
        urls_preserved=sum(s.urls for s in sections),
    )
    return text, stats
