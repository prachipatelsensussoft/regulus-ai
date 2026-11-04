"""LP1: inside a section that does not fit, paragraphs are ranked by term overlap fused with their similarity to the
question; a list's items compete as one; a child of a requested provision that the ceiling cut is named in the notice."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from regulus.rag import ContextPolicy, RagConfig, RagPipeline
from regulus.rag import validator_switches as VS
from regulus.rag.selection import _dense_rank, _required_order
from regulus.retrieval import LexicalRetriever
from tests.test_rag import FakeGenerator
from tests.test_retrieval_plan import TABLE, TableDatabase


def _ev(eid, path, text, ordinal):
    md = {"section_number": "101.9", "paragraph_path": path, "ordinal": ordinal}
    return SimpleNamespace(evidence_id=eid, text=text, citation=f"21 CFR 101.9{path}", corpus="ecfr",
                           result=SimpleNamespace(metadata=md))


@pytest.fixture
def fake_rows(monkeypatch):
    # rows of one section: (section, path) from the metadata, as the retriever records it
    import regulus.rag.selection as S
    monkeypatch.setattr(S, "_paragraph_of", lambda e: (e.result.metadata["section_number"], e.result.metadata["paragraph_path"]))


def test_lp1_is_off_by_default():
    assert "LP1" not in VS.DEFAULT.split(",")


def test_dense_rank_shares_a_rank_between_equal_values():
    assert _dense_rank([3, 0, 3, 1]) == [1, 3, 1, 2]
    assert _dense_rank([0.2, 0.9, 0.5]) == [3, 1, 2]


QUESTION = "How must the declared values be rounded on the label?"
ROWS = [
    _ev("a", "(a)", "(a) The label shall state the serving size.", 0),                       # one question term: label
    _ev("b", "(b)", "(b) Amounts are expressed to the nearest gram increment below 5 grams.", 1),   # says it, other words
    _ev("c", "(c)", "(c) The nutrition information shall be presented in a panel.", 2),
]
SIM = {"a": 0.10, "b": 0.90, "c": 0.20}


def test_off_term_overlap_decides(monkeypatch, fake_rows):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"LP1"})
    assert [e.evidence_id for e in _required_order(ROWS, QUESTION, SIM)] == ["a", "b", "c"]
    assert [e.evidence_id for e in _required_order(ROWS, QUESTION)] == ["a", "b", "c"]


def test_on_a_paragraph_that_says_it_in_other_words_ranks_first(monkeypatch, fake_rows):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"LP1"})
    assert [e.evidence_id for e in _required_order(ROWS, QUESTION, SIM)] == ["b", "a", "c"]
    assert [e.evidence_id for e in _required_order(ROWS, QUESTION)] == ["a", "b", "c"]   # no index: as before


def test_on_a_lists_items_compete_as_one(monkeypatch, fake_rows):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"LP1"})
    rows = [_ev("p", "(a)", "(a) The label must bear:", 0),
            _ev("x", "(a)(1)", "(1) the name of the food;", 1),
            _ev("y", "(a)(2)", "(2) the net quantity rounded as declared;", 2),
            _ev("z", "(a)(3)", "(3) the name of the manufacturer;", 3),
            _ev("q", "(b)", "(b) Exemptions for small packages are in paragraph (j).", 4)]
    sim = {"p": 0.3, "x": 0.1, "y": 0.8, "z": 0.1, "q": 0.6}
    order = [e.evidence_id for e in _required_order(rows, QUESTION, sim)]
    items = [order.index(k) for k in ("x", "y", "z")]
    assert max(items) - min(items) == 2                    # the three items sit together ...
    assert order.index("q") > max(items) or order.index("q") < min(items)   # ... not split by another paragraph


# ------------------------------------------------------------------- the notice names a cut child of a requested provision

pytestmark = pytest.mark.usefixtures("cpu_era_budget")
PARENT_Q = "What does 21 CFR 820.10(b) require?"


def _answer():
    long_table = [dict(r, body_text="L" * 1500) for r in TABLE]
    policy = ContextPolicy(max_total_chars=3600, hard_max_total_chars=3600, required_max_chars_per_result=4000)
    pipe = RagPipeline(LexicalRetriever(TableDatabase(long_table)), FakeGenerator("It requires records [1]."),
                       RagConfig(context=policy))
    return pipe.answer(PARENT_Q)


def test_a_cut_child_of_the_requested_paragraph_is_named_only_with_lp1(monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"LP1"})
    off = _answer()
    children_cut = [d for d in off.metadata["selection"]["dropped"] if not d.get("required")
                    and d["citation"].startswith("21 CFR 820.10(b)(") and d["reason"].startswith("hard_ceiling")]
    assert children_cut, "the fixture must cut a child of 820.10(b)"
    assert not (off.metadata.get("evidence_gap") or {}).get("left_out")   # today: nothing is said
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"LP1"})
    on = _answer()
    assert on.metadata["evidence_gap"]["left_out"] == {"21 CFR 820.10(b)": len(children_cut)}
    assert "Evidence limit: not all of the retrieved text of 21 CFR 820.10(b)" in on.answer
