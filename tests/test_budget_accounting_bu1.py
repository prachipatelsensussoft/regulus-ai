"""BU1: the evidence budget is charged with the SOURCE block the prompt renders, "Tier:" line included."""
from __future__ import annotations

import pytest

from regulus.planning import plan_query
from regulus.rag import validator_switches as VS
from regulus.rag.context import ContextPolicy
from regulus.rag.coverage import check_coverage
from regulus.rag.generation import build_package, render_source, tier_line
from regulus.rag.selection import select_evidence
from regulus.retrieval import LexicalRetriever
from regulus.retrieval.evidence import TIER_DISCOVERY, TIER_EXACT, TIER_STRUCTURAL, TIER_SUPPORTING
from tests.test_retrieval_plan import TableDatabase

QUESTION = "What do 820.10(b)(2) and 820.10(b)(4) require?"


def _selection_and_package():
    plan = plan_query(QUESTION)
    out = LexicalRetriever(TableDatabase()).run_plan(plan)
    sel = select_evidence(out.registry, ContextPolicy())
    return sel, build_package(plan, check_coverage(out.registry), sel)


def test_bu1_is_off_by_default():
    assert "BU1" not in VS.DEFAULT.split(",")


@pytest.mark.parametrize("tier,rids,corpus,docket,expected", [
    (TIER_EXACT, ("r1",), "ecfr", False, "Tier: EXACT (authoritative text of r1)"),
    (TIER_EXACT, ("r1",), "guidance", True, "Tier: EXACT (the requested guidance document; non-binding)"),
    (TIER_STRUCTURAL, ("r1", "r2"), "ecfr", False, "Tier: STRUCTURAL (parent or child of r1, r2; context only, not the requested text)"),
    (TIER_SUPPORTING, (), "usc", False, "Tier: STATUTE (verbatim text from the FD&C Act section this question is about; binding law)"),
    (TIER_SUPPORTING, (), "ecfr", False, "Tier: SUPPORTING (related material; does not stand in for any requested provision)"),
    (TIER_DISCOVERY, ("r2",), "ecfr", False, "Tier: DISCOVERY (found by broad search while looking for r2; not authoritative for any requested provision)"),
])
def test_the_tier_line_text_is_unchanged_by_the_refactor(tier, rids, corpus, docket, expected):
    assert tier_line(tier, rids, corpus, docket) == expected


def test_on_each_block_is_charged_exactly_what_the_prompt_renders(monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"BU1", "BA1"})
    sel, pkg = _selection_and_package()
    reqs = dict(pkg.requirements)
    rendered = {s.index: len(render_source(s, reqs)) for s in pkg.sources}
    assert rendered and {i.index: i.chars for i in sel.selected} == rendered


def test_off_the_tier_line_is_the_uncharged_difference(monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", (VS.VALIDATOR_PRECISION | {"BA1"}) - {"BU1"})
    sel, pkg = _selection_and_package()
    reqs = dict(pkg.requirements)
    for item, src in zip(sel.selected, pkg.sources):
        assert len(render_source(src, reqs)) - item.chars == len(src.role_line(reqs)) + 1   # the line and its newline
