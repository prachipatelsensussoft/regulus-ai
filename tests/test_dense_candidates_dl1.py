"""DL1: a general or programme-routed question gets up to DL1_MAX dense-index sections after its BM25 candidates."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from regulus.rag import section_promotion as SP
from regulus.rag import trace
from regulus.rag import validator_switches as VS
from regulus.rag.section_promotion import SectionPromoter
from regulus.retrieval.section_index import RankedSection

SECTIONS = ["111.70", "111.75", "111.80", "117.80", "820.30", "211.84"]


class _Index:
    parts = [s.split(".")[0] for s in SECTIONS]
    headings = [f"Heading {s}" for s in SECTIONS]
    subparts = [""] * len(SECTIONS)

    def rank(self, lex, limit=22):
        return [RankedSection("111.70", "111", "Heading 111.70", "", 3.0, 0),
                RankedSection("111.75", "111", "Heading 111.75", "", 2.0, 0)][:limit]


class _Dense:
    def rank_sections(self, question, corpus=None):
        # most similar first: (section, similarity, the paragraph that matched)
        return [("111.70", 0.9, SimpleNamespace(path="(a)", ordinal=1)),
                ("820.30", 0.85, SimpleNamespace(path="(b)", ordinal=7)),
                ("111.80", 0.8, SimpleNamespace(path="(c)", ordinal=4)),
                ("117.80", 0.7, SimpleNamespace(path="", ordinal=0))]


class _Db:
    def query(self, sql, params=None):
        return []


@pytest.fixture
def promoter(monkeypatch):
    import regulus.retrieval.paragraph_dense as PD
    monkeypatch.setattr(PD, "paragraph_dense", lambda db: _Dense())
    pr = SectionPromoter(db=_Db(), mode="picker", index=_Index())
    monkeypatch.setattr(pr, "_where", lambda: {s: i for i, s in enumerate(SECTIONS)})
    monkeypatch.setattr(pr, "_titles", lambda: {})
    monkeypatch.setattr(SP.SectionIndex, "question_lexemes", staticmethod(lambda db, q: {"specif"}))
    monkeypatch.setattr(SP, "ENTITY_CONSTRAINTS", False)
    for code in ("SY1", "SY2", "SY3", "SY4", "HQ1"):
        monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {code})
    return pr


def test_dl1_is_off_by_default():
    assert "DL1" not in VS.DEFAULT.split(",")


def test_the_dense_sections_the_lexical_list_lacks_are_added_with_the_paragraph_that_matched(promoter):
    out = promoter._paragraph_dense_sections("q", [RankedSection("111.70", "111", "", "", 1.0, 0)], None, limit=2)
    assert [(r.section, r.best_ordinal) for r in out] == [("820.30", 7), ("111.80", 4)]


def test_inside_a_routed_programme_only_its_parts(promoter):
    out = promoter._paragraph_dense_sections("q", [], frozenset({"111"}), limit=4)
    assert [r.section for r in out] == ["111.70", "111.80"]


def test_a_general_question_gets_them_after_its_lexical_candidates(promoter, monkeypatch):
    monkeypatch.setattr(SP, "routed_programme", lambda q: None)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"DL1"})
    _, off = promoter.candidates("Which specifications must a manufacturer establish?")
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"DL1", "TR1"})
    tr = trace.begin()
    _, on = promoter.candidates("Which specifications must a manufacturer establish?")
    assert [c["section"] for c in off] == ["111.70", "111.75"]
    assert [c["section"] for c in on] == ["111.70", "111.75", "820.30", "111.80", "117.80"]   # lexical order kept
    dl1 = next(e for e in tr.to_dict()["events"] if e["action"] == "DL1")
    assert dl1["added"][0] == ["820.30", 0.85, "(b)"]


def test_a_routed_question_stays_inside_its_programme(promoter, monkeypatch):
    supplements = SimpleNamespace(name="Dietary supplements", parts=frozenset({"111"}))
    monkeypatch.setattr(SP, "routed_programme", lambda q: supplements)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"DL1"})
    _, on = promoter.candidates("Which specifications must a dietary supplement manufacturer establish?")
    assert [c["section"] for c in on] == ["111.70", "111.75", "111.80"]
