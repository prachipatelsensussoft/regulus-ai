"""TF1: the protected core - five distinct provisions (one per ask first) compete first for the evidence ceiling, with
documented expansion to eight; Top-5 is an order, never a cap."""
from __future__ import annotations

import pytest

from regulus.planning import plan_query
from regulus.rag import validator_switches as VS
from regulus.rag.context import ContextPolicy
from regulus.rag.core_evidence import CORE, MAX, choose_core, question_clauses, unit_of
from regulus.rag.selection import select_evidence
from regulus.retrieval import LexicalRetriever
from tests.test_retrieval_plan import TableDatabase, row


def test_tf1_is_off_by_default():
    assert "TF1" not in VS.DEFAULT.split(",")


@pytest.mark.parametrize("question,asks", [
    ("Who is subject to Part 111?", ["Who is subject to Part 111?"]),
    ("What fees apply, how do we pay, and can a small company get relief?",
     ["What fees apply", "how do we pay", "can a small company get relief?"]),
    ("Does 117.5 exempt a farm? What records apply?", ["Does 117.5 exempt a farm?", "What records apply?"]),
])
def test_question_clauses(question, asks):
    assert question_clauses(question) == asks


def test_units_are_top_level_paragraph_families():
    assert unit_of("111.70", "(e)(2)(i)") == "111.70(e)" and unit_of("111.3", "") == "111.3"


def _rows(n, section="101.9"):
    labels = "abcdefghijkl"
    return [{"id": f"e{i}", "section": section, "path": f"({labels[i]})", "text": f"({labels[i]}) Text {i} about labels."}
            for i in range(n)]


def test_one_ask_takes_the_five_most_similar_distinct_units():
    rows = _rows(7)
    sims = {f"e{i}": s for i, s in enumerate([0.1, 0.2, 0.9, 0.3, 0.8, 0.05, 0.7])}
    first, core = choose_core(rows, ["What is required?"], [sims], "What is required?")
    assert [c["unit"] for c in core] == ["101.9(c)", "101.9(e)", "101.9(g)", "101.9(d)", "101.9(b)"]
    assert len(core) == CORE and all(c["reason"] == "top score" for c in core)
    assert first == ["e2", "e4", "e6", "e3", "e1"]   # each unit's lead (here the unit itself) first


def test_each_ask_gets_a_unit_before_any_ask_gets_a_second():
    rows = _rows(6)
    ask1 = {"e0": 0.9, "e1": 0.85, "e2": 0.8, "e3": 0.1, "e4": 0.1, "e5": 0.1}
    ask2 = {"e0": 0.1, "e1": 0.1, "e2": 0.1, "e3": 0.2, "e4": 0.7, "e5": 0.1}
    _, core = choose_core(rows, ["what fees apply", "how do we pay"], [ask1, ask2], "fees pay")
    assert [c["unit"] for c in core][:2] == ["101.9(a)", "101.9(e)"]   # ask 1's best, then ask 2's best
    assert [c["ask"] for c in core][:2] == [1, 2]


def test_a_question_with_more_asks_than_the_core_expands_to_at_most_eight_with_a_reason():
    rows = _rows(12)
    asks = [f"what about item {i} here" for i in range(10)]
    scores = [{f"e{j}": (0.9 if j == i else 0.1) for j in range(12)} for i in range(10)]
    _, core = choose_core(rows, asks, scores, " ".join(asks))
    assert len(core) == MAX
    assert [c["reason"] for c in core][CORE:] == ["expanded: the question has 10 asks"] * (MAX - CORE)


def test_a_core_paragraph_pointing_to_an_exception_brings_it():
    rows = _rows(8)
    rows[0]["text"] = "(a) Each label must bear the statement, except as provided in paragraph (h) of this section."
    sims = {f"e{i}": s for i, s in enumerate([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.01])}
    _, core = choose_core(rows, ["What must the label bear?"], [sims], "What must the label bear?")
    assert core[-1]["unit"] == "101.9(h)" and core[-1]["reason"] == "expanded: 101.9(a) points to paragraph (h)"
    assert len(core) == CORE + 1


# ------------------------------------------------------------------------------ select_evidence under a tight ceiling

TABLE = [row("101.9", f"({c})", i, text=f"({c}) " + ("Paragraph filler text about food labeling. " * 50))
         for i, c in enumerate("abcdefgh")]
# (g) answers the question in words the question does not use: only its similarity can find it
TABLE[6] = row("101.9", "(g)", 6, text="(g) Rounding: values are rounded to the nearest 5-calorie increment. " * 25)
QUESTION = "What does 21 CFR 101.9 require?"
ASKED = "How precise must the stated energy content be?"


def _select(monkeypatch, tf1):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", (VS.VALIDATOR_PRECISION | {"TF1"}) if tf1 else (VS.VALIDATOR_PRECISION - {"TF1"}))
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"CB1", "HX1", "OF1", "LP1"})
    reg = LexicalRetriever(TableDatabase(TABLE)).run_plan(plan_query(QUESTION)).registry
    sims = {e.evidence_id: (0.95 if "(g)" in e.citation else 0.2) for e in reg.evidence}
    return select_evidence(reg, ContextPolicy(hard_max_total_chars=7000), question=ASKED, core_scores=[sims])


def test_with_tf1_the_paragraph_that_answers_is_kept_and_the_core_is_recorded(monkeypatch):
    on = _select(monkeypatch, True)
    assert any("(g)" in s.evidence.citation for s in on.selected)
    assert on.diagnostics["core_units"][0]["unit"] == "101.9(g)"
    cites = [s.evidence.citation for s in on.selected]
    assert cites == sorted(cites)   # still presented in document order


def test_without_tf1_the_same_ceiling_cuts_it_and_nothing_is_recorded(monkeypatch):
    off = _select(monkeypatch, False)
    assert not any("(g)" in s.evidence.citation for s in off.selected)
    assert "core_units" not in off.diagnostics
    assert any("(g)" in d.evidence.citation for d in off.dropped)   # the drop is recorded with its reason
