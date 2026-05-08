"""Compute a deterministic 2D layout of the prompt in semantic space.

We vectorize at the *sentence* level (TF-IDF), fit PCA-2, and project sections
through the same basis. The result:

  - sentence_2d  : every meaningful line of text as a point in semantic space
  - section_2d   : section centroids in the same space
  - similarity   : pairwise cosine similarity between sections (for edges)

Position around the final sigil = atan2 of these coordinates, so semantically
close items end up adjacent on the circle. The pattern is real, not styled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .parse import Section


@dataclass
class Layout:
    sections: List[Section]
    section_2d: np.ndarray            # (n_sections, 2)
    sentence_2d: np.ndarray           # (n_sentences, 2)
    sentence_section_idx: np.ndarray  # (n_sentences,)
    sentence_texts: List[str]
    similarity: np.ndarray            # (n_sections, n_sections)
    vectorizer: object = None         # fitted TfidfVectorizer (for live projection)
    pca: object = None                # fitted PCA (for live projection)
    section_X: object = None          # section TF-IDF matrix (for live similarity)


def flatten(sec: Section, results: List[Section] | None = None) -> List[Section]:
    if results is None:
        results = []
    for c in sec.children:
        results.append(c)
        flatten(c, results)
    return results


def _section_corpus(sec: Section) -> str:
    parts = [sec.heading, sec.text]
    for c in sec.children:
        parts.append(_section_corpus(c))
    return "\n".join(parts)


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。!?])\s+|\n+")
_LIST_PREFIX_RE = re.compile(r"^[\s\-*+•]+")


def _split_units(text: str) -> List[str]:
    units: List[str] = []
    for chunk in _SENT_SPLIT_RE.split(text):
        chunk = _LIST_PREFIX_RE.sub("", chunk).strip()
        if len(chunk) > 2:
            units.append(chunk)
    return units


def compute_layout(root: Section) -> Layout | None:
    sections = flatten(root)
    if not sections:
        return None

    sentence_texts: List[str] = []
    sentence_section_idx: List[int] = []
    for i, s in enumerate(sections):
        if s.heading and s.heading != "(untitled)":
            sentence_texts.append(s.heading)
            sentence_section_idx.append(i)
        for unit in _split_units(s.text):
            sentence_texts.append(unit)
            sentence_section_idx.append(i)

    if not sentence_texts:
        # Fall back to one centroid per section even with empty bodies
        sentence_texts = [s.heading or " " for s in sections]
        sentence_section_idx = list(range(len(sections)))

    vec = TfidfVectorizer(
        max_features=4000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w[\w'-]*\b",
    )
    try:
        sentence_X = vec.fit_transform(sentence_texts)
    except ValueError:
        return None

    section_corpora = [_section_corpus(s) or s.heading or " " for s in sections]
    section_X = vec.transform(section_corpora)

    pca = None
    if min(sentence_X.shape) < 2:
        sentence_2d = np.zeros((sentence_X.shape[0], 2))
        section_2d = np.zeros((len(sections), 2))
    else:
        pca = PCA(n_components=2)
        sentence_2d = pca.fit_transform(sentence_X.toarray())
        section_2d = pca.transform(section_X.toarray())

    if section_X.shape[0] >= 2:
        sim = cosine_similarity(section_X)
    else:
        sim = np.ones((len(sections), len(sections)))

    return Layout(
        sections=sections,
        section_2d=section_2d,
        sentence_2d=sentence_2d,
        sentence_section_idx=np.array(sentence_section_idx, dtype=int),
        sentence_texts=sentence_texts,
        similarity=sim,
        vectorizer=vec,
        pca=pca,
        section_X=section_X,
    )


# Backward-compatible helper used by older callers.
def compute_embeddings(root: Section):
    layout = compute_layout(root)
    if layout is None:
        return {}
    out = {}
    for i, s in enumerate(layout.sections):
        v = layout.section_2d[i]
        out[id(s)] = np.array([v[0], v[1], 0.0])
    return out
