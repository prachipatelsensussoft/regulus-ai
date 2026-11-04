"""PP1: a paragraph is identified by its row - its own lead and items by position where paths repeat (DEF1 definitions
sections), and labels of any case and length."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from regulus.planning import plan_query
from regulus.rag import validator_switches as VS
from regulus.rag.context import ContextPolicy
from regulus.rag.selection import (_identity, _own_children, _required_order, _true_parent, completed_by_protected_children,
                                   consolidate_evidence, is_required, select_evidence)
from regulus.retrieval import LexicalRetriever
from tests.test_retrieval_plan import TableDatabase, row

# 117.3 as DEF1 stores it: each definition's lead at "" (path_ordinal 1, 2, ...), its items at "(1)", "(2)" - repeated
DEFS = [
    row("117.3", "", 0, text="The following definitions apply to words and phrases used in this part."),
    row("117.3", "", 1, text="Farm means:", path_ordinal=1),
    row("117.3", "(1)", 2, text="(1) Primary production farm. An operation under one management in one general location."),
    row("117.3", "(2)", 3, text="(2) Secondary activities farm. An operation not located on a primary production farm."),
    row("117.3", "", 4, text="Holding means storage of food and also includes activities performed incidental to storage:",
        path_ordinal=2),
    row("117.3", "(1)", 5, text="(1) Holding includes fumigation during storage and drying or dehydrating raw "
                                "agricultural commodities when the drying does not create a distinct commodity.",
        path_ordinal=1),
    row("117.3", "(2)", 6, text="(2) Holding also includes blending of the same raw agricultural commodity.", path_ordinal=1),
]
QUESTION = "Does drying raw agricultural commodities count as holding under 21 CFR 117.3?"


def _required(table=DEFS, question=QUESTION):
    reg = LexicalRetriever(TableDatabase(table)).run_plan(plan_query(question)).registry
    return [e for e in reg.evidence if is_required(e)]


@pytest.fixture
def pp1(monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"PP1"})


@pytest.fixture
def no_pp1(monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"PP1"})


def test_pp1_is_off_by_default():
    assert "PP1" not in VS.DEFAULT.split(",")


def test_own_children_and_true_parent_follow_position_where_paths_repeat():
    rows = [("", 0), ("", 1), ("(1)", 2), ("(2)", 3), ("", 4), ("(1)", 5), ("(2)", 6)]
    assert _own_children(1, rows) == [2, 3] and _own_children(4, rows) == [5, 6]   # Farm's items, Holding's items
    assert _true_parent(5, rows) == 4 and _true_parent(2, rows) == 1                # Holding's "(1)" -> Holding's lead
    unique = [("", 0), ("(a)", 1), ("(a)(1)", 2), ("(b)", 3)]
    assert _own_children(1, unique) == [2] and _true_parent(2, unique) == 1         # unique paths: as by prefix


def test_labels_of_any_case_and_length_have_an_identity(pp1):
    ev = SimpleNamespace(citation="21 CFR 101.9(c)(1)(i)(A)")
    assert _identity(ev) == ("101.9", "(c)(1)(i)(A)")
    assert _identity(SimpleNamespace(citation="21 CFR 117.5(g)(3)(xviii)")) == ("117.5", "(g)(3)(xviii)")


def test_without_pp1_such_labels_have_none(no_pp1):
    assert _identity(SimpleNamespace(citation="21 CFR 101.9(c)(1)(i)(A)")) is None


def _sel(items):
    return SimpleNamespace(selected=tuple(items), dropped=(), policy=ContextPolicy(), total_chars=0, diagnostics={})


def _item(citation, text, key):
    ev = SimpleNamespace(citation=citation, corpus="ecfr", key=key, satisfies=(), result=SimpleNamespace(metadata={}))
    return SimpleNamespace(evidence=ev, text=text, protected=True, truncated=False, reason="exact", chars=len(text))


def test_identical_text_in_two_uppercase_paragraphs_stays_two_provisions(monkeypatch):
    a = _item("21 CFR 101.9(c)(1)(i)(A)", "(A) Not applicable.", "k1")
    b = _item("21 CFR 101.9(c)(2)(i)(A)", "(A) Not applicable.", "k2")
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"PP1"})
    _, info = consolidate_evidence(_sel([a, b]))
    assert info.get("removed")   # today: merged although they are different provisions
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"PP1"})
    sel, info = consolidate_evidence(_sel([a, b]))
    assert len(sel.selected) == 2 and not info.get("removed")


def test_a_definition_lead_travels_with_its_own_items(monkeypatch):
    req = _required()
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"PP1"})
    order = [e.result.metadata["ordinal"] for e in _required_order(req, QUESTION)]
    holding_lead, farm_lead = order.index(4), order.index(1)
    assert holding_lead < farm_lead   # "Holding means:" inherits its "(1)" (drying), Farm's lead does not


def test_a_definition_item_is_completed_by_its_own_lead_only(monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"PP1"})
    req = _required()
    sel = select_evidence(LexicalRetriever(TableDatabase(DEFS)).run_plan(plan_query(QUESTION)).registry, ContextPolicy())
    by_ord = {s.evidence.result.metadata["ordinal"]: s for s in sel.selected}
    holding_item = by_ord[5]
    # Holding's lead present: complete; take it out (Farm's lead stays): no longer complete
    assert completed_by_protected_children(holding_item, sel)
    without = SimpleNamespace(selected=tuple(s for s in sel.selected if s.evidence.result.metadata["ordinal"] not in (4, 6)),
                              dropped=tuple(SimpleNamespace(evidence=by_ord[k].evidence) for k in (4, 6)),
                              policy=sel.policy, total_chars=0, diagnostics={})
    assert not completed_by_protected_children(holding_item, without)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"PP1"})
    assert completed_by_protected_children(holding_item, without)   # today: any "" lead completes it (Farm's)
    assert len(req) == len(DEFS)
