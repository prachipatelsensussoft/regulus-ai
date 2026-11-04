"""AO1 (2 Nov 2026): a list the answer restates keeps its source's "and" / "or".

A regulation's list is conjunctive or disjunctive by the word before its last item: 50.24(a)(2) "Obtaining informed
consent is not feasible because: (i) ...; (ii) ...; and (iii) ..." - all three must hold. K-002 (bundle, 2 Nov) wrote
"(i) ..., (ii) ..., or (iii) ...", turning three required conditions into alternatives; the scorer marked it
PARTIALLY FAIL for exactly that, and nothing in validation compares the joining word.

The repair changes that one word and nothing else, and only when the answer's enumerated items match the items of one
supplied list one by one (each item's content words largely in its own source item). A list the answer composes
itself, or one that cannot be matched item by item, is left as written.
"""
from __future__ import annotations

import re

_LABEL = r"\((?:[ivxl]{1,8}|[a-z]|\d{1,2})\)"
# the joining word right before an enumerator label: ", or (iii)", "; and\n- (iii)"
_JOIN = re.compile(rf"([,;])(\s*)\b(and|or)\b(\s*(?:[-*•]\s*)?)({_LABEL})", re.I)
_ITEM_LABEL = re.compile(rf"(?<![\w)]){_LABEL}\s")
_WORD = re.compile(r"[a-z][a-z'-]{3,}")
_STOP = frozenset({"that", "this", "with", "from", "have", "must", "shall", "such", "each", "which", "their", "them",
                   "they", "there", "these", "those", "been", "being", "will", "would", "should", "could", "under",
                   "into", "than", "then", "also", "only", "paragraph", "section", "chapter"})
MATCH_SHARE = 0.5
MIN_ITEM_WORDS = 3
WINDOW = 900          # how far back from the joining word the enumeration may start


def _style(label: str) -> str:
    inner = label.strip("() ")
    return "digit" if inner.isdigit() else ("roman" if re.fullmatch(r"[ivx]+", inner) else "letter")


def _stem(w: str) -> str:
    w = w.lower().removesuffix("'s").removesuffix("’s")
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def source_lists(sources) -> list[tuple[str, list[set[str]]]]:
    """(conjunction, [content words of each item]) for every supplied list whose items are separate paragraphs and
    whose second-to-last item ends with "; and" / "; or" (", and" / ", or")."""
    by_parent: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for s in sources:
        m = re.match(r"21 CFR (\d+\.\d+[a-z]?)((?:\([^)\s]+\))+)$", getattr(s, "citation", "") or "")
        if not m:
            continue
        levels = re.findall(r"\([^)]+\)", m.group(2))
        by_parent.setdefault((m.group(1), "".join(levels[:-1])), []).append((levels[-1], s.text or ""))
    out = []
    for items in by_parent.values():
        if len(items) < 2:
            continue
        tail = re.search(r"[;,]\s*(and|or)\s*$", items[-2][1].rstrip())
        if tail:
            out.append((tail.group(1).lower(), [_words(t) for _, t in items]))
    return out


def fix_conjunctions(text: str, sources) -> tuple[str, int]:
    """(text, joining words changed); see the module docstring."""
    if not text or not _JOIN.search(text):
        return text, 0
    lists = source_lists(sources)
    if not lists:
        return text, 0
    changes: list[tuple[int, int, str]] = []
    for m in _JOIN.finditer(text):
        start = max(0, m.start() - WINDOW)
        before = text[start:m.start()]
        style = _style(m.group(5))
        # only labels of the final item's kind: "(2) ... because (i) ..., (ii) ..., or (iii)" is a list of three
        labels = [x for x in _ITEM_LABEL.finditer(before) if _style(x.group(0).strip()) == style]
        if not labels:
            continue
        # the enumeration: from the first label of the run before the joining word, through the last item
        first = labels[0].start()
        for k in range(len(labels) - 1, 0, -1):     # the run starts after the last sentence end before a label
            if re.search(r"[.!?]\s", before[labels[k - 1].end():labels[k].start()]):
                first = labels[k].start()
                break
        last = re.match(r"[^.;\n]{3,400}", text[m.end():])
        chunks = [before[a.end():(labels[i + 1].start() if i + 1 < len(labels) else len(before))]
                  for i, a in enumerate(labels) if a.start() >= first] + [last.group(0) if last else ""]
        items = [_words(c) for c in chunks]
        if len(items) < 2 or any(len(w) < MIN_ITEM_WORDS for w in items):
            continue
        for conj, src_items in lists:
            if len(src_items) != len(items):
                continue
            if all(len(w & s) / len(w) >= MATCH_SHARE for w, s in zip(items, src_items)):
                if conj != m.group(3).lower():
                    word = conj if m.group(3).islower() else conj.capitalize()
                    changes.append((m.start(3), m.end(3), word))
                break
    for a, b, word in sorted(changes, reverse=True):
        text = text[:a] + word + text[b:]
    return text, len(changes)
