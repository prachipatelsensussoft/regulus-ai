"""OF1 (2 Nov 2026): when the requested section does not fit, its top-level paragraphs compete first (outline first)."""
import pytest

from regulus.planning import plan_query
from regulus.rag import ContextPolicy, select_evidence
from regulus.rag import validator_switches
from regulus.retrieval import LexicalRetriever
from tests.test_ind_safety_deadlines import DEADLINE_QUESTION, _registry as _deadline_registry
from tests.test_retrieval_plan import TableDatabase, row

pytestmark = pytest.mark.usefixtures("cpu_era_budget")

QUESTION = "What are the labeling requirements for dietary supplements under 21 CFR 101.36?"


def _long(lead, size):
    filler = "The nutrient shall be declared on the dietary supplement label as described in this paragraph. "
    return (lead + " " + filler * (size // len(filler) + 1))[:size]


def _table():
    # shaped like 101.36: short top-level paragraphs, long deep items under (b), and (d) - its own topic - last
    return ([row("101.36", "(a)", 0, text="(a) The label of a dietary supplement shall bear nutrition labeling."),
             row("101.36", "(b)", 1, text="(b) The Supplement Facts panel shall list the following dietary ingredients:")]
            + [row("101.36", f"(b)({k})", 1 + k, text=_long(f"({k}) Dietary supplement nutrient declaration {k}.", 3600))
               for k in range(1, 7)]
            + [row("101.36", "(c)", 9, text="(c) A proprietary blend of dietary ingredients shall be listed with its total weight."),
               row("101.36", "(d)", 10, text=("(d) Source ingredients shall be identified and the part of the plant from "
                                               "which each botanical is derived shall be declared. " * 30)[:2000])])


def _registry():
    return LexicalRetriever(TableDatabase(_table())).run_plan(plan_query(QUESTION)).registry


def _switch(monkeypatch, on):
    monkeypatch.setattr(validator_switches, "VALIDATOR_PRECISION",
                        validator_switches.VALIDATOR_PRECISION - {"OF1"} | ({"OF1"} if on else set()))


def _labels(selection):
    return [s.evidence.citation for s in selection.selected]


def test_off_a_deep_list_spends_the_ceiling_and_the_last_topic_is_dropped(monkeypatch):
    _switch(monkeypatch, False)
    sel = select_evidence(_registry(), question=QUESTION)
    assert sel.overflow, "the fixture must not fit, or it tests nothing"
    assert "21 CFR 101.36(d)" not in _labels(sel)


def test_on_every_top_level_paragraph_is_supplied_in_document_order(monkeypatch):
    _switch(monkeypatch, True)
    sel = select_evidence(_registry(), question=QUESTION)
    labels = _labels(sel)
    for p in ("(a)", "(b)", "(c)", "(d)"):
        assert f"21 CFR 101.36{p}" in labels
    ordinals = [s.evidence.result.metadata["ordinal"] for s in sel.selected]
    assert ordinals == sorted(ordinals)
    assert sel.total_chars <= ContextPolicy().hard_max_total_chars


def test_a_question_asking_for_a_value_keeps_its_relevance_order(monkeypatch):
    registry = _deadline_registry("What does 21 CFR 312.32 require?")
    _switch(monkeypatch, False)
    before = _labels(select_evidence(registry, question=DEADLINE_QUESTION))
    _switch(monkeypatch, True)
    assert _labels(select_evidence(registry, question=DEADLINE_QUESTION)) == before


def test_a_top_level_that_is_not_an_outline_is_left_as_it_is():
    from regulus.rag.selection import outline_first

    order = list(_deadline_registry("What does 21 CFR 312.32 require?").evidence)
    # (a) and (b) are 3,800 characters each: more than a third of a 9,000-character ceiling
    assert outline_first(order, "What does 21 CFR 312.32 require?", 9000) == order
