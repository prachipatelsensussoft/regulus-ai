"""FD&C Act paragraphs (21 U.S.C. chapter 9) as a third evidence corpus, "usc".

The eCFR and guidance corpora have no statute: in the cosmetics bank 57 of 277 keys cite the FD&C Act (MoCRA
sections 604-614, the 201(i) definition, 601-603), and every such question failed. The paragraphs come from the
official U.S. Code text (scripts/build_statute_store.py -> data/processed/statute_fdca/rows.jsonl, with the source
URL and sha256 of every file in manifest.json); nothing is paraphrased.

Ranking mirrors regulus.retrieval.section_index: PostgreSQL 'english' lexemes, section-level BM25 with the heading
weighted, plus the share of the question the best paragraph covers. A statute result is binding law (corpus "usc",
retrieval method "statute") and is labelled "FD&C Act § 605(b)(1) [21 U.S.C. 364a(b)(1)]".
"""
from __future__ import annotations

import json
import re
import math
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regulus.retrieval.models import Citation, RetrievalResult

STORE = Path(__file__).resolve().parents[3] / "data" / "processed" / "statute_fdca" / "rows.jsonl"
CORPUS_USC = "usc"
METHOD_STATUTE = "statute"
K1, B, HEAD_TF = 1.2, 0.75, 3.0
SECTION_CHAR_BUDGET = 6000       # rendered characters of one promoted section placed in the prompt
BLOCK_HEADER_CHARS = 240         # SOURCE / Tier / Corpus / Citation / Heading lines of one rendered block
_GENERIC_QUESTION_WORDS = frozenset({"fd", "c", "act", "section", "sec", "requir", "say", "provid", "mean", "cover",
                                     "subsect", "fdca", "us", "u.s.c", "21", "explain", "summar", "summari", "describ"})
_LEX_SQL = "SELECT t.i, tsvector_to_array(to_tsvector('english', t.x)) AS lex FROM unnest(%(t)s::text[]) WITH ORDINALITY AS t(x, i)"


@dataclass(frozen=True)
class StatuteSection:
    fdca: str
    usc: str
    heading: str
    score: float
    best: int            # row index of the best-matching paragraph

    @property
    def label(self) -> str:
        return f"FD&C Act § {self.fdca} [21 U.S.C. {self.usc}]"


class StatuteIndex:
    def __init__(self, rows: list[dict], row_lex: list[set[str]]) -> None:
        self.rows, self.row_lex = rows, row_lex
        self.by_sec: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            self.by_sec[r["fdca_section"]].append(i)
        self.head_lex: dict[str, set[str]] = {}
        self.sec_tf: dict[str, Counter] = {}
        for s, idx in self.by_sec.items():
            self.sec_tf[s] = Counter(x for i in idx for x in self.row_lex[i])
        self._head_rows = {s: idx[0] for s, idx in self.by_sec.items()}
        n = len(self.by_sec)
        df = Counter(x for s in self.by_sec for x in set(self.sec_tf[s]) | self.head_lex.get(s, set()))
        self.idf = {x: math.log(1 + (n - c + 0.5) / (c + 0.5)) for x, c in df.items()}
        self.avg = sum(len(c) for c in self.sec_tf.values()) / max(n, 1)

    @classmethod
    def load(cls, db, path: Path = STORE) -> "StatuteIndex":
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        from regulus.rag.validator_switches import enabled

        # SC1: a paragraph's subsection caption is indexed with it ("(e) Good manufacturing practice requirements")
        texts = [" ".join(x for x in (r["heading"], r.get("top_caption") if enabled("SC1") else None, r["paragraph_heading"],
                                      r["body_text"]) if x) for r in rows]
        lex = {int(r["i"]) - 1: set(r["lex"] or ()) for r in db.query(_LEX_SQL, {"t": texts})}
        idx = cls(rows, [lex.get(i, set()) for i in range(len(rows))])
        heads = {s: rows[ix[0]]["heading"] for s, ix in idx.by_sec.items()}
        hl = {int(r["i"]) - 1: set(r["lex"] or ()) for r in db.query(_LEX_SQL, {"t": list(heads.values())})}
        idx.head_lex = {s: hl.get(k, set()) for k, s in enumerate(heads)}
        return idx

    def outline(self, fdca: str, chars: int = 420) -> str:
        """The section's top-level subsections in its own words: each one's caption ("(e) Recordkeeping") or, without a
        caption, its first words - the picker's view of what the section covers (never text that is not the Act's)."""
        parts, seen = [], set()
        for i in self.by_sec.get(fdca, []):
            r = self.rows[i]
            top = self._top(i)
            if not top or top in seen:
                continue
            seen.add(top)
            words = (r.get("top_caption") or r["paragraph_heading"]
                     or " ".join(re.sub(r"^\([^)]+\)\s*", "", r["body_text"]).split()[:10]))
            parts.append(f"{top} {words}".strip())
        text = "; ".join(parts)
        return text if len(text) <= chars else text[:chars].rsplit(";", 1)[0] + "; ..."

    def heading(self, fdca: str) -> str:
        return self.rows[self.by_sec[fdca][0]]["heading"]

    def usc(self, fdca: str) -> str:
        return self.rows[self.by_sec[fdca][0]]["usc_section"]

    def rank(self, lexemes, limit: int = 3) -> list[StatuteSection]:
        q = set(lexemes)
        total = sum(self.idf.get(x, 0.0) for x in q) or 1.0
        out = []
        for s, idx in self.by_sec.items():
            tf_c, head = self.sec_tf[s], self.head_lex.get(s, set())
            bm = 0.0
            for x in q:
                tf = min(tf_c.get(x, 0), 3) + (HEAD_TF if x in head else 0)
                if tf:
                    bm += self.idf.get(x, 0.0) * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len(tf_c) / self.avg))
            cov = [(sum(self.idf.get(x, 0.0) for x in q & (self.row_lex[i] | head)), i) for i in idx]
            best_v, best_i = max(cov) if cov else (0.0, idx[0])
            score = bm / 10 + best_v / total
            if score > 0:
                out.append(StatuteSection(s, self.usc(s), self.heading(s), score, best_i))
        out.sort(key=lambda r: (-r.score, r.fdca))
        return out[:limit]

    def _top(self, i: int) -> str:
        m = re.match(r"^\([^)]+\)", self.rows[i]["paragraph_path"] or "")
        return m.group(0) if m else ""

    def blocks(self, fdca: str, lexemes, budget: int = SECTION_CHAR_BUDGET, named_path: str | None = None,
               header: int = BLOCK_HEADER_CHARS, prefer: dict[str, float] | None = None) -> tuple[list[list[int]], list[list[int]]]:
        """(kept, left out): the section's rows grouped by top-level subsection ("(5)" with its (A)(i)-(vii) items is
        one block, so a definition is never split), in document order. A block the question names ("605(e)") is
        kept first; the rest are kept by how much of the question they cover while the rendered size (text plus
        `header` characters per block) fits `budget`. A block longer than the budget is split at its second level."""
        idx = self.by_sec.get(fdca, [])
        groups: list[list[int]] = []
        for i in idx:
            top = self._top(i)
            if groups and (self._top(groups[-1][0]) == top):
                groups[-1].append(i)
            else:
                groups.append([i])
        cost = lambda g: sum(len(self.rows[i]["body_text"]) + 1 for i in g) + header
        split: list[list[int]] = []
        for g in groups:
            if cost(g) <= budget or len(g) == 1:
                split.append(g)
                continue
            lead, cur = [g[0]], []
            for i in g[1:]:
                second = re.match(r"^(\([^)]+\)){2}", self.rows[i]["paragraph_path"] or "")
                if cur and second and not (self.rows[i]["paragraph_path"] or "").startswith(self.rows[cur[0]]["paragraph_path"]):
                    split.append(lead + cur if not split or split[-1][:1] != lead else cur)
                    cur = []
                cur.append(i)
            split.append(lead + cur if not split or split[-1][:1] != lead else cur)
        q = set(lexemes)
        named = [k for k, g in enumerate(split)
                 if named_path and any((self.rows[i]["paragraph_path"] or "").startswith(named_path) for i in g)]
        score = {k: sum(self.idf.get(x, 0.0) for x in q & set().union(*(self.row_lex[i] for i in g))) for k, g in enumerate(split)}
        specific = q - _GENERIC_QUESTION_WORDS - {fdca, self.usc(fdca)}
        if prefer:
            # AP1: blocks by their best paragraph's cross-encoder score against the question (what 910(b)(1) lists is
            # not found by word overlap), the word-overlap score breaking ties
            best = {k: max((prefer.get(self.rows[i]["paragraph_path"] or "", 0.0) for i in g), default=0.0) for k, g in enumerate(split)}
            order = named + sorted((k for k in range(len(split)) if k not in named), key=lambda k: (-best[k], -score[k], k))
        elif len(specific) < 2:
            # "What does FD&C Act section 605 require?": nothing to rank by, so the section is read from its start
            order = named + [k for k in range(len(split)) if k not in named]
        else:
            order = named + sorted((k for k in range(len(split)) if k not in named), key=lambda k: (-score[k], k))
        kept, spent = set(), 0
        for k in order:
            c = cost(split[k])
            if spent + c <= budget:
                kept.add(k)
                spent += c
        return ([split[k] for k in sorted(kept)], [split[k] for k in range(len(split)) if k not in kept])

    def paragraphs(self, fdca: str, lexemes, budget: int = SECTION_CHAR_BUDGET, named_path: str | None = None) -> list[int]:
        """Row indices of the kept blocks, flattened (see blocks)."""
        return [i for g in self.blocks(fdca, lexemes, budget, named_path)[0] for i in g]

    def block_result(self, rows: list[int], score: float = 1.0) -> RetrievalResult:
        """One evidence passage for a block: its rows' text joined, cited as its first row's paragraph."""
        first = self.rows[rows[0]]
        lines = []
        for i in rows:
            r = self.rows[i]
            text = r["body_text"]
            if r["paragraph_heading"] and not text.startswith("("):
                text = f"{r['paragraph_path'].split(')')[-2] + ')' if r['paragraph_path'] else ''} {r['paragraph_heading']}. {text}".strip()
            lines.append(text)
        top = self._top(rows[0])
        path = top if top and all((self.rows[i]["paragraph_path"] or "").startswith(top) for i in rows) else first["paragraph_path"]
        label = f"FD&C Act § {first['fdca_section']}{path} [21 U.S.C. {first['usc_section']}{path}]"
        cit = Citation(corpus=CORPUS_USC, label=label, url=first["source_url"], title="21 U.S.C.", section=first["usc_section"],
                       paragraph=path or None, heading=first["heading"])
        return RetrievalResult(corpus=CORPUS_USC, text="\n".join(lines), heading=first["heading"], relevance_score=score,
                               retrieval_method=METHOD_STATUTE, source_url=first["source_url"], citation=cit,
                               metadata={"fdca_section": first["fdca_section"], "usc_section": first["usc_section"],
                                         "paragraph_path": path, "rows": [self.rows[i]["paragraph_path"] for i in rows]},
                               key=f"usc:{first['usc_section']}:{path}:{first['ordinal']}")

    def describe(self) -> dict[str, Any]:
        return {"sections": len(self.by_sec), "paragraphs": len(self.rows)}


_LOCK = threading.Lock()
_CACHE: dict[int, StatuteIndex | None] = {}


def statute_index(db) -> StatuteIndex | None:
    """The statute index for this database connection, or None when the store has not been built."""
    with _LOCK:
        if id(db) not in _CACHE:
            _CACHE[id(db)] = StatuteIndex.load(db) if STORE.exists() else None
        return _CACHE[id(db)]
