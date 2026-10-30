"""Stages B-F of the experimental advanced retrieval (regulus.retrieval.advanced, OFF unless passed to RagPipeline).

The base selection is never reordered or reduced: evidence is only appended, for a planned question element no
selected eCFR source addresses, in a bounded reserve inside the hard ceilings. Measured reason (2026-10-20): every
change that reordered evidence inside the fixed context exchanged one required provision for another."""

import pytest

from regulus.planning import plan_query
from regulus.rag import ContextPolicy, select_evidence
from regulus.rag.selection import _block_chars, _clean, _truncate
from regulus.retrieval import LexicalRetriever
from regulus.retrieval import sql as sql_module
from regulus.retrieval.advanced import GAP_MAX_SOURCES, STATE_COVERED, STATE_FILLED, AdvancedRetrieval
from regulus.retrieval.evidence import TIER_DISCOVERY, TIER_SUPPORTING, Evidence, EvidenceRegistry
from regulus.retrieval.models import METHOD_FULL_TEXT, METHOD_TRIGRAM
from tests.test_retrieval_plan import guidance_row, row

# fixtures sized to overflow the CPU-era evidence budget (tests/conftest.py)
pytestmark = pytest.mark.usefixtures("cpu_era_budget")

Q = ("We're a small cosmetic pigment maker and want FDA to list a new color additive for our makeup line. "
     "Can we propose it ourselves, and how must the petition be submitted?")
PART71 = dict(part_title="Color Additive Petitions", section_heading="Petitions.")
A = row("71.1", "(a)", 1, text="(a) Any interested person may propose the listing of a color additive. Such proposal "
                               "shall be made in a petition in the form prescribed in paragraph (c) of this section.", **PART71)
C = row("71.1", "(c)", 3, text="(c) Petitions shall include the following data and be submitted in the following form:", **PART71)
C1 = row("71.1", "(c)(1)", 4, text="(1) The name and all pertinent information concerning the color additive.", **PART71)


def ecfr(n, r, tier, method=METHOD_FULL_TEXT):
    return Evidence(f"e{n}", LexicalRetriever._ecfr_result(r, method, 1.0, {"similarity": 0.9}), tier)


def guidance(n, text):
    return Evidence(f"e{n}", LexicalRetriever._guidance_result(guidance_row(n, body_text=text), METHOD_FULL_TEXT, 1.0, {}), TIER_SUPPORTING)


def registry(question=Q, *evidence):
    evidence = evidence or (guidance(1, "Citizen petitions to FDA must be submitted ... color " + "x " * 450),
                            ecfr(2, A, TIER_SUPPORTING), ecfr(3, C, TIER_DISCOVERY, METHOD_TRIGRAM))
    return EvidenceRegistry(plan=plan_query(question), evidence=tuple(evidence), requirement_state={}, diagnostics={})


POLICY = ContextPolicy(max_total_chars=1500, max_results=8, binding_reserve_chars=0)


class FakeDb:
    def query(self, statement, params=None):
        if statement == sql_module.ECFR_EXACT_SECTION:
            return [dict(C, match_kind="exact"), dict(C1, match_kind="descendant")]
        return []


def extend(flags, reg=None, question=Q):
    reg = reg or registry(question)
    base = select_evidence(reg, POLICY)
    adv = AdvancedRetrieval(FakeDb(), {k: True for k in flags})
    out, trace = adv.extend(base, reg, question, POLICY, block_chars=_block_chars, clean=_clean, truncate=_truncate)
    return base, out, trace


def keys(sel):
    return [s.evidence.key for s in sel.selected]


def test_with_every_stage_off_the_selection_is_unchanged_and_traced():
    base, out, trace = extend([])
    assert keys(out) == keys(base) and out.total_chars == base.total_chars
    assert out.diagnostics["advanced"] is trace and not trace["added"]


def test_an_uncovered_element_is_filled_after_the_untouched_base_sources():
    base, out, trace = extend(["gap_fill"])
    assert keys(out)[: len(base.selected)] == keys(base)                     # base order and members unchanged
    added = out.selected[len(base.selected):]
    assert [s.evidence.result.citation.label for s in added] == ["21 CFR 71.1(c)"]
    assert added[0].index == len(base.selected) + 1 and added[0].reason == "advanced_gap"
    states = {e["id"]: e["state"] for e in trace["elements"]}
    assert states["ask2"] == STATE_FILLED


def test_guidance_never_counts_as_covering_an_element():
    _, _, trace = extend([])
    assert {e["id"]: e["state"] for e in trace["elements"]}["ask2"] != STATE_COVERED


def test_a_selected_lead_in_is_completed_with_its_own_subparagraphs():
    reg = registry(Q, ecfr(1, A, TIER_SUPPORTING), ecfr(2, C, TIER_SUPPORTING))
    _, out, trace = extend(["gap_fill", "structure"], reg)
    labels = [s.evidence.result.citation.label for s in out.selected]
    assert "21 CFR 71.1(c)(1)" in labels and labels.count("21 CFR 71.1(c)") == 1
    assert any(a["for"] == "completion" and a["kept"] for a in trace["added"])
    child = next(s for s in out.selected if s.evidence.result.citation.label == "21 CFR 71.1(c)(1)")
    assert child.evidence.tier == TIER_SUPPORTING                            # the lead-in's own tier


@pytest.mark.parametrize("question,why", [
    ("What does FAA Part 107 require for a remote drone pilot certificate?", "another agency"),
    ("What does 21 CFR 71.1 require for a color additive petition?", "names a provision"),
])
def test_other_agencies_and_named_provisions_are_left_alone(question, why):
    base, out, trace = extend(["gap_fill", "structure"], registry(question), question)
    assert keys(out) == keys(base) and why in trace["skipped"]


def test_the_reserve_bounds_what_is_appended():
    _, out, trace = extend(["gap_fill", "structure"])
    assert sum(1 for a in trace["added"] if a["kept"]) <= GAP_MAX_SOURCES
    assert out.total_chars <= POLICY.hard_max_total_chars and len(out.selected) <= POLICY.hard_max_results


def test_a_lead_in_is_completed_with_all_its_subparagraphs_or_not_at_all():
    """Measured on the answer gate (2026-10-21): a list cut to its first three items was answered as if complete."""
    class ManyChildren(FakeDb):
        def query(self, statement, params=None):
            if statement == sql_module.ECFR_EXACT_SECTION:
                return [dict(C, match_kind="exact")] + [dict(row("71.1", f"(c)({n})", 3 + n, text=f"({n}) item {n}.", **PART71),
                                                             match_kind="descendant") for n in range(1, 6)]
            return []
    reg = registry(Q, ecfr(1, A, TIER_SUPPORTING), ecfr(2, C, TIER_SUPPORTING))
    base = select_evidence(reg, POLICY)
    out, trace = AdvancedRetrieval(ManyChildren(), {"gap_fill": True, "structure": True}).extend(
        base, reg, Q, POLICY, block_chars=_block_chars, clean=_clean, truncate=_truncate)
    children = [s for s in out.selected if s.evidence.result.citation.label.startswith("21 CFR 71.1(c)(")]
    assert children == [] and any(not a["kept"] and "do not all fit" in a.get("why", "") for a in trace["added"])


def test_a_row_with_a_malformed_paragraph_path_is_never_added():
    bad = row("1114.7", "(i)(ii)(iii)(iv)(v)(vi)(4)", 9, text="(4) Petition submitted for a color additive petition.", **PART71)
    reg = registry(Q, guidance(1, "x " * 500), ecfr(2, A, TIER_SUPPORTING), ecfr(3, bad, TIER_DISCOVERY, METHOD_TRIGRAM))
    _, out, trace = extend(["gap_fill"], reg)
    assert all("1114.7" not in s.evidence.result.citation.label for s in out.selected)
