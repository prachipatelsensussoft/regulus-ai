"""Dense (embedding) ranking of single paragraphs: every live paragraph of the cosmetics programme (21 CFR Parts 70-82
and 700-740) and every paragraph of the FD&C Act store.

Why paragraphs (23 Oct 2026, section-resolution benchmark): section-level vectors (regulus.retrieval.section_dense)
embed a section's heading and first ~900 characters, so a section whose answer sits deep inside it is invisible to them
(720.4(b) "The person filing Form FDA 2512 should ..."; FD&C Act 605(e) recordkeeping inside the adverse-event
section). Ranking paragraphs and scoring a section by its best paragraph puts the FD&C Act section that answers a
question first for 23 of 40 Act-keyed dev questions with the lexical ranking (fixed order: 1 of 40).

Each paragraph is embedded with its section heading and, for a sub-paragraph, its parent's lead-in ("(d) A certificate
shall expire ... if:" before "(2) the package is opened"). The model is the repository's pinned BAAI/bge-small-en-v1.5,
loaded from the local cache only; vectors are cached under data/processed/paragraph_dense/ by the sha256 of the texts
and the model revision (about 3,100 paragraphs, built once in about a minute on CPU).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from regulus.retrieval.section_index import COSMETIC_PARTS

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "processed" / "paragraph_dense"
TEXT_CHARS = 900
LEADIN_CHARS = 200
# "title21" adds every other live part of Title 21 (26 Oct 2026): the page reads every question against all of Title 21
# since the topic selector went, but the reranker could only read cosmetics and FD&C Act paragraphs, so its veto of a
# wrong pick never acted on a drug, device or food question. The cosmetics/Act index and its cache are unchanged (built
# first, as before); the other parts are appended after it, cached one part per file so a repaired part re-embeds only
# itself (about 63,000 paragraphs, about 20 minutes on CPU the first time: prebuild with
# `python -m regulus.retrieval.paragraph_dense`). Default "title21" since 2 Nov 2026: every evaluation run since 26 Oct
# used it (the 61.0% 300Q run included), so the app reads what was measured; "cosmetics" restores the old index.
SCOPE = os.environ.get("REGULUS_PARAGRAPH_DENSE_SCOPE", "title21").strip().lower() or "title21"

_ROWS_SQL = """
SELECT section_number, part_number, paragraph_path, ordinal, section_heading, body_text
FROM cfr_records WHERE removed_on IS NULL AND NOT is_reserved AND part_number = ANY(%(parts)s)
ORDER BY section_number, ordinal
"""
_OTHER_ROWS_SQL = """
SELECT section_number, part_number, paragraph_path, ordinal, section_heading, body_text
FROM cfr_records WHERE removed_on IS NULL AND NOT is_reserved AND part_number = ANY(%(parts)s)
ORDER BY part_number, section_number, ordinal
"""
_OTHER_PARTS_SQL = """
SELECT DISTINCT part_number FROM cfr_records WHERE removed_on IS NULL AND NOT is_reserved AND NOT (part_number = ANY(%(parts)s))
"""
# the other parts' rows are read this many parts at a time: one query for all ~65,000 rows ran into the connection's
# 15 s statement timeout under load (2 Nov), and the index then fell back to cosmetics + the Act without a word in the
# answer - every other part's paragraphs missing from section ranking and the cross-encoder's view
OTHER_PARTS_PER_QUERY = 40


@dataclass(frozen=True)
class Para:
    corpus: str          # "cfr" | "fdca"
    section: str         # "720.4" | "605"
    path: str            # "(b)(1)" ("" for an undivided section)
    ordinal: int         # cfr_records.ordinal, or the statute store row index


def _parent(path: str) -> str | None:
    m = re.match(r"^((?:\([^)]+\))+)\([^)]+\)$", path or "")
    return m.group(1) if m else None


class ParagraphDense:
    def __init__(self, paras: list[Para], vectors: np.ndarray, embedder, texts: list[str] | None = None) -> None:
        self.paras, self.vectors, self.embedder = paras, vectors, embedder
        self.texts = texts or [""] * len(paras)   # the embedded text of each paragraph (heading, lead-in, path, body)
        self.by_section: dict[str, list[int]] = {}
        self.position: dict[Para, int] = {}
        for i, p in enumerate(paras):
            self.by_section.setdefault(p.section, []).append(i)
            self.position[p] = i

    def text(self, para: Para) -> str:
        return self.texts[self.position[para]]

    @classmethod
    def build(cls, db, statute=None, embedder=None, scope: str | None = None) -> "ParagraphDense":
        paras, texts = [], []

        def add(corpus, section, heading, rows, paras=paras, texts=texts):   # rows: [(path, ordinal, body)]
            body_by_path = {p: b for p, _, b in rows}
            for path, ordinal, body in rows:
                lead = body_by_path.get(_parent(path) or "", "") if _parent(path) else ""
                text = f"{heading} " + (f"{lead[:LEADIN_CHARS]} ... " if lead else "") + f"{path} {body}"
                paras.append(Para(corpus, section, path, ordinal))
                texts.append(" ".join(text.split())[:TEXT_CHARS])

        by_sec: dict[str, list] = {}
        heads: dict[str, str] = {}
        for r in db.query(_ROWS_SQL, {"parts": sorted(COSMETIC_PARTS)}):
            by_sec.setdefault(r["section_number"], []).append((r["paragraph_path"] or "", int(r["ordinal"]), " ".join((r["body_text"] or "").split())))
            heads[r["section_number"]] = r["section_heading"] or ""
        for s, rows in by_sec.items():
            add("cfr", s, heads[s], rows)
        if statute is not None:
            from regulus.rag.validator_switches import enabled

            captions = enabled("SC1")
            for s, idx in statute.by_sec.items():
                rows = []
                for i in idx:
                    r = statute.rows[i]
                    body = " ".join((r["body_text"] or "").split())
                    if r.get("paragraph_heading") and not body.startswith("("):
                        body = f"{r['paragraph_heading']}. {body}"
                    if captions and r.get("top_caption") and r.get("top_caption") != r.get("paragraph_heading"):
                        # SC1: the subsection's caption heads each of its paragraphs - 906(e)(1)(A) is "In general"
                        # under "(e) Good manufacturing practice requirements", which nothing else in its text says
                        body = f"{r['top_caption']}: {body}"
                    rows.append((r["paragraph_path"] or "", i, body))
                add("fdca", s, f"FD&C Act {statute.heading(s)}.", rows)
        embedder = embedder or _embedder()
        desc = embedder.describe() or {}
        revision = desc.get("resolved_revision") or desc.get("revision")
        vectors = _cached(embedder, revision, texts, CACHE_DIR / "vectors_{key}.npy")
        if (scope or SCOPE) == "title21":
            try:
                vectors = cls._add_other_parts(db, embedder, revision, add, paras, texts, vectors)
            except Exception:  # noqa: BLE001 - the wider index is an aid: the cosmetics/Act index stands without it
                logging.getLogger(__name__).exception("paragraph_dense: other parts not added")
                del paras[len(vectors):], texts[len(vectors):]
        return cls(paras, vectors, embedder, texts)

    @staticmethod
    def _add_other_parts(db, embedder, revision, add, paras, texts, vectors) -> np.ndarray:
        """Every live part outside the cosmetics programme, appended in part order; each part's vectors cached on their
        own (CACHE_DIR/parts/part_<n>_<key>.npy)."""
        by_part: dict[str, dict[str, list]] = {}
        heads: dict[str, str] = {}
        others = sorted(str(r["part_number"]) for r in db.query(_OTHER_PARTS_SQL, {"parts": sorted(COSMETIC_PARTS)}))
        for k in range(0, len(others), OTHER_PARTS_PER_QUERY):
            for r in db.query(_OTHER_ROWS_SQL, {"parts": others[k:k + OTHER_PARTS_PER_QUERY]}):
                by_part.setdefault(str(r["part_number"]), {}).setdefault(r["section_number"], []).append(
                    (r["paragraph_path"] or "", int(r["ordinal"]), " ".join((r["body_text"] or "").split())))
                heads[r["section_number"]] = r["section_heading"] or ""
        blocks = [vectors]
        for part in sorted(by_part, key=lambda p: (len(p), p)):
            p_paras, p_texts = [], []
            for s, rows in by_part[part].items():
                add("cfr", s, heads[s], rows, p_paras, p_texts)
            blocks.append(_cached(embedder, revision, p_texts, CACHE_DIR / "parts" / f"part_{part}_{{key}}.npy"))
            paras += p_paras
            texts += p_texts
        return np.concatenate(blocks).astype(np.float32, copy=False)

    def similarities(self, question: str) -> np.ndarray:
        return self.vectors @ self.embedder.embed_query(question)

    def rank_sections(self, question: str, corpus: str | None = None, sims: np.ndarray | None = None) -> list[tuple[str, float, Para]]:
        """Sections by their best paragraph's similarity to the question: (section, similarity, that paragraph)."""
        sims = self.similarities(question) if sims is None else sims
        best: dict[str, tuple[float, int]] = {}
        for s, idx in self.by_section.items():
            if corpus is not None and self.paras[idx[0]].corpus != corpus:
                continue
            j = max(idx, key=lambda i: sims[i])
            best[s] = (float(sims[j]), j)
        return [(s, v, self.paras[j]) for s, (v, j) in sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))]

    def section_paragraphs(self, question: str, section: str, sims: np.ndarray | None = None) -> list[tuple[Para, float]]:
        """One section's paragraphs, most similar first."""
        sims = self.similarities(question) if sims is None else sims
        return sorted(((self.paras[i], float(sims[i])) for i in self.by_section.get(section, ())), key=lambda x: -x[1])


def _cached(embedder, revision, texts: list[str], pattern: Path) -> np.ndarray:
    """The texts' vectors, from the cache file named by their hash (`pattern` holds "{key}"), embedded and saved when
    absent."""
    key = hashlib.sha256(json.dumps([revision, texts]).encode("utf-8")).hexdigest()[:16]
    path = pattern.with_name(pattern.name.format(key=key))
    if path.exists():
        return np.load(path)
    vectors = embedder.embed_batch(texts).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, vectors)
    tmp.replace(path)
    return vectors


def _embedder():
    from regulus.retrieval.section_dense import _embedder as section_embedder

    return section_embedder()


_LOCK = threading.Lock()
_CACHE: dict[int, ParagraphDense | None] = {}


def paragraph_dense(db) -> ParagraphDense | None:
    """The paragraph index for this database connection; None when the embedding model cannot be loaded (callers then
    rank without it - it is an aid, never a requirement)."""
    from regulus.retrieval.statute import statute_index

    with _LOCK:
        if id(db) not in _CACHE:
            try:
                _CACHE[id(db)] = ParagraphDense.build(db, statute_index(db))
            except Exception:  # noqa: BLE001 - an aid only
                _CACHE[id(db)] = None
        return _CACHE[id(db)]


if __name__ == "__main__":   # prebuild the cache for REGULUS_PARAGRAPH_DENSE_SCOPE=title21 before serving
    from regulus.retrieval.db import Database
    from regulus.retrieval.statute import statute_index

    _db = Database.from_env()
    _db.connect()
    try:
        _pd = ParagraphDense.build(_db, statute_index(_db), scope="title21")
        print(f"{len(_pd.paras)} paragraphs, {len(_pd.by_section)} sections")
    finally:
        _db.close()
