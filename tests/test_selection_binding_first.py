"""ContextPolicy.general_evidence_order="binding_first" (Phase 2 experiment, off by default).

For a question that names no provision, every eCFR passage competes for the soft budget before any guidance
passage, and the eCFR discovery passages compete by their own score rather than by retrieval leg. The default
("tier") must leave every selection exactly as it was. No database, no model: TableDatabase stands in.
"""

import pytest

from regulus.planning import plan_query
from regulus.rag.context import ContextPolicy
from regulus.rag.selection import _optional_order, select_evidence
from regulus.retrieval.evidence import TIER_DISCOVERY, TIER_STRUCTURAL, TIER_SUPPORTING
from regulus.retrieval.query import QueryType
from tests.test_retrieval_plan import TableDatabase, row
from tests.test_retrieval_section_coverage import COVERAGE, UI12, guidance, run

# fixtures sized to overflow the CPU-era evidence budget (tests/conftest.py)
pytestmark = pytest.mark.usefixtures("cpu_era_budget")

ON = ContextPolicy(general_evidence_order="binding_first")


def _trigram(i):
    # heading look-alikes from another part, as in S-007 (111.370, 111.155 ... for a water-supply question)
    return row(f"111.{300 + i}", "", 0, sim=0.45, body_text=f"What requirements apply to item {i}? " + "y" * 120)


def _crowded_general_registry():
    """D-016 / S-007 shape: AND-matched guidance chunks (supporting), eCFR trigram look-alikes (discovery, score
    2.25) and the section-coverage rows that answer it (discovery, scores 3.17 and 3.0, placed last)."""
    db = TableDatabase(coverage=COVERAGE, guidance=[guidance(i) for i in range(3)], trigram=[_trigram(i) for i in range(10)])
    return run(UI12, db).registry


def _labels(sel):
    return [s.evidence.citation for s in sel.selected]


def _same_selection(a, b):
    """Everything but the policy itself (which names the switch when it is on)."""
    da, db_ = a.to_dict(), b.to_dict()
    da.pop("policy"), db_.pop("policy")
    return da == db_


def test_the_default_policy_and_its_fingerprint_are_unchanged():
    assert ContextPolicy().general_evidence_order == "tier"
    assert "general_evidence_order" not in ContextPolicy().to_dict()          # fingerprint fields unchanged
    assert ON.to_dict()["general_evidence_order"] == "binding_first"           # a switched-on run says so
    with pytest.raises(ValueError):
        ContextPolicy(general_evidence_order="score")


def test_default_order_is_the_previous_tier_by_tier_order():
    reg = _crowded_general_registry()
    previous = [e for tier in (TIER_STRUCTURAL, TIER_SUPPORTING, TIER_DISCOVERY) for e in reg.evidence if e.tier == tier]
    assert _optional_order(reg, False) == previous


def test_by_default_the_answering_section_is_crowded_out():
    sel = select_evidence(_crowded_general_registry(), ContextPolicy())
    assert "21 CFR 11.10(c)" not in _labels(sel)                               # the baseline failure mode
    assert sel.diagnostics.get("binding_reserved")


def test_binding_first_keeps_the_highest_scored_binding_text_within_the_same_limits():
    reg = _crowded_general_registry()
    sel = select_evidence(reg, ON)
    assert {"21 CFR 11.10(c)", "21 CFR 11.10(b)"} <= set(_labels(sel))
    assert len(sel.selected) <= ON.max_results and sel.total_chars <= ON.max_total_chars
    assert sel.diagnostics["general_evidence_order"] == "binding_first" and "binding_reserved" not in sel.diagnostics
    position = {e.evidence_id: i for i, e in enumerate(reg.evidence)}            # still presented in registry order
    assert [position[s.evidence.evidence_id] for s in sel.selected] == sorted(position[s.evidence.evidence_id] for s in sel.selected)
    assert [s.index for s in sel.selected] == list(range(1, len(sel.selected) + 1))
    assert sel.total_chars == sum(s.chars for s in sel.selected) + 2 * (len(sel.selected) - 1)
    order = _optional_order(reg, True)
    ecfr = [e for e in order if e.result.corpus == "ecfr"]
    assert order[:len(ecfr)] == ecfr                                            # every eCFR passage before guidance


def test_the_usefulness_filter_still_applies():
    reg = _crowded_general_registry()
    sel = select_evidence(reg, ON, optional_filter=lambda r: r.retrieval_method != "trigram")
    assert not any(s.evidence.retrieval_method == "trigram" for s in sel.selected)
    assert any(d.reason == "filtered" for d in sel.dropped)


def test_a_question_that_names_a_provision_is_not_affected():
    db = TableDatabase(guidance=[guidance(i) for i in range(3)])
    reg = run("What does 21 CFR 820.10 require for the quality management system?", db).registry
    assert reg.plan.requirements
    assert _same_selection(select_evidence(reg, ON), select_evidence(reg, ContextPolicy()))


def test_a_guidance_question_is_not_affected():
    q = "What does FDA guidance recommend for protecting and retaining electronic records?"
    assert plan_query(q).query_type is QueryType.GUIDANCE_SEARCH
    db = TableDatabase(coverage=COVERAGE, guidance=[guidance(i) for i in range(3)], trigram=[_trigram(i) for i in range(4)])
    reg = run(q, db).registry
    assert _same_selection(select_evidence(reg, ON), select_evidence(reg, ContextPolicy()))


def test_rr1_requested_sections_share_the_budget_round_robin(monkeypatch):
    from types import SimpleNamespace
    from regulus.rag import selection as S
    from regulus.rag import validator_switches as VS

    def ev(sec, path):
        return SimpleNamespace(evidence_id=f"{sec}{path}", sec=sec, path=path)

    a = [ev("820.3", "(a)"), ev("820.3", "(b)")]
    b = [ev("820.10", ""), ev("820.10", "(b)"), ev("820.10", "(c)")]
    monkeypatch.setattr(S, "_paragraph_of", lambda e: (e.sec, e.path))
    monkeypatch.setattr(S, "_required_order", lambda g, q: g)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"RR1"})
    got = [e.evidence_id for e in S._per_section_order(a + b, "q")]
    assert got == ["820.3(a)", "820.10", "820.3(b)", "820.10(b)", "820.10(c)"]   # M-029: 820.10 is not starved
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"RR1"})
    assert [e.evidence_id for e in S._per_section_order(a + b, "q")] == ["820.3(a)", "820.3(b)", "820.10", "820.10(b)", "820.10(c)"]


def test_rr2_the_first_section_keeps_its_list_and_the_companions_share(monkeypatch):
    from types import SimpleNamespace
    from regulus.rag import selection as S
    from regulus.rag import validator_switches as VS

    def ev(sec, path):
        return SimpleNamespace(evidence_id=f"{sec}{path}", sec=sec, path=path)

    first = [ev("111.210", "(a)"), ev("111.210", "(b)"), ev("111.210", "(c)")]
    c1 = [ev("111.205", "(a)"), ev("111.205", "(b)")]
    c2 = [ev("111.123", "(a)"), ev("111.123", "(b)")]
    monkeypatch.setattr(S, "_paragraph_of", lambda e: (e.sec, e.path))
    monkeypatch.setattr(S, "_required_order", lambda g, q: g)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", (VS.VALIDATOR_PRECISION | {"RR2"}) - {"RR1"})
    got = [e.evidence_id for e in S._per_section_order(first + c1 + c2, "q")]
    assert got[:3] == ["111.210(a)", "111.210(b)", "111.210(c)"]   # S-018: the list is never halved
    assert got[3:] == ["111.205(a)", "111.123(a)", "111.205(b)", "111.123(b)"]


def test_ba1_a_block_is_charged_for_what_the_prompt_renders_not_the_url(monkeypatch):
    from types import SimpleNamespace
    from regulus.rag import selection as S
    from regulus.rag import validator_switches as VS
    monkeypatch.setattr(S, "render_source", lambda probe: "SOURCE 1\nCitation: 21 CFR 117.405(a)\nURL: " + "u" * 90 + "\nText:\nabc")
    ev = SimpleNamespace(result=None)
    monkeypatch.setattr(S, "ContextSource", lambda **kw: None)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"BA1"})
    full = S._block_chars(ev, "abc", 1, False)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"BA1"})
    assert full - S._block_chars(ev, "abc", 1, False) == len("URL: " + "u" * 90) + 1   # the URL line is not charged
