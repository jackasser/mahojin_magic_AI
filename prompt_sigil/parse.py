"""Parse a markdown prompt into a tree of sections with measurable features.

Every property captured here is a real, countable property of the source text.
No interpretation, no decoration. The renderer maps these 1:1 to visual variables.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

IMPERATIVE_PATTERNS = [
    # English imperatives
    r"\bMUST\b", r"\bMUST NOT\b", r"\bNEVER\b", r"\bALWAYS\b",
    r"\bIMPORTANT\b", r"\bCRITICAL\b", r"\bDO NOT\b", r"\bDON'T\b",
    r"\bREQUIRED\b", r"\bSHALL\b", r"\bSHOULD\b",
    # Japanese imperatives — kept on purpose so Japanese system prompts
    # measure the same way as English ones. PRs adding more languages welcome.
    r"必ず", r"絶対に", r"してはいけない", r"してはなりません",
    r"してください", r"重要", r"禁止",
]
IMPERATIVE_RE = re.compile("|".join(IMPERATIVE_PATTERNS), re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+")
PATH_RE = re.compile(r"`[^`]*?\.[A-Za-z0-9]{1,8}`|[A-Za-z]:[\\/][\w\\/.\-]+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^\s*[-*+]\s+")


@dataclass
class Section:
    heading: str
    depth: int  # 0 = root, 1 = h1, ...
    text: str = ""
    children: List["Section"] = field(default_factory=list)
    bullets: int = 0
    code_blocks: int = 0
    code_lines: int = 0
    urls: int = 0
    refs: int = 0
    tokens: int = 0
    imperatives: int = 0
    # Verbatim body blocks in document order: ("prose", "", text) or ("code", lang, body).
    # Used by the compressor to rebuild minimal markdown without losing code or order.
    body: List[Tuple[str, str, str]] = field(default_factory=list)

    def total_tokens(self) -> int:
        return self.tokens + sum(c.total_tokens() for c in self.children)

    def total_imperatives(self) -> int:
        return self.imperatives + sum(c.total_imperatives() for c in self.children)

    def total_bullets(self) -> int:
        return self.bullets + sum(c.total_bullets() for c in self.children)

    def total_refs(self) -> int:
        return self.urls + self.refs + sum(c.total_refs() for c in self.children)


def parse_markdown(text: str) -> Section:
    """Parse markdown into a Section tree. Root is a depth-0 anonymous container."""
    root = Section(heading="(root)", depth=0)
    stack: List[Section] = [root]
    buffer: List[str] = []
    in_code = False
    code_buffer: List[str] = []

    code_lang = ""

    def flush_buffer() -> None:
        if not buffer:
            return
        body = "\n".join(buffer)
        sec = stack[-1]
        sec.text += body + "\n"
        sec.tokens += len(body.split())
        for line in buffer:
            if BULLET_RE.match(line):
                sec.bullets += 1
        sec.imperatives += len(IMPERATIVE_RE.findall(body))
        sec.urls += len(URL_RE.findall(body))
        sec.refs += len(PATH_RE.findall(body))
        sec.body.append(("prose", "", body))
        buffer.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                code = "\n".join(code_buffer)
                sec = stack[-1]
                sec.code_blocks += 1
                sec.code_lines += len(code_buffer)
                sec.tokens += len(code.split())
                sec.body.append(("code", code_lang, code))
                code_buffer.clear()
                code_lang = ""
                in_code = False
            else:
                flush_buffer()
                code_lang = stripped[3:].strip()
                in_code = True
            continue
        if in_code:
            code_buffer.append(line)
            continue
        m = HEADING_RE.match(line)
        if m:
            flush_buffer()
            depth = len(m.group(1))
            heading = m.group(2)
            new_sec = Section(heading=heading, depth=depth)
            while stack and stack[-1].depth >= depth:
                stack.pop()
            stack[-1].children.append(new_sec)
            stack.append(new_sec)
            continue
        buffer.append(line)
    flush_buffer()

    # If the document had no headings, treat the whole thing as one depth-1 section
    if not root.children and root.text.strip():
        synthetic = Section(heading="(untitled)", depth=1)
        synthetic.text = root.text
        synthetic.tokens = root.tokens
        synthetic.bullets = root.bullets
        synthetic.code_blocks = root.code_blocks
        synthetic.code_lines = root.code_lines
        synthetic.urls = root.urls
        synthetic.refs = root.refs
        synthetic.imperatives = root.imperatives
        synthetic.body = list(root.body)
        root.children.append(synthetic)
        root.body = []
        root.text = ""
        root.tokens = 0
        root.bullets = 0
        root.code_blocks = 0
        root.code_lines = 0
        root.urls = 0
        root.refs = 0
        root.imperatives = 0

    return root
