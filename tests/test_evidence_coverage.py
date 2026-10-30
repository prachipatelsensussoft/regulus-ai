"""Stage 3: coverage, requirement-aware selection, protected context.

Uses the Stage 2 in-memory TableDatabase so the full path
plan -> run_plan -> registry -> coverage -> selection -> context runs
without PostgreSQL or a model.
"""

import json

import pytest

from regulus.planning import plan_query
from regulus.rag import ContextPolicy, RagConfig, RagPipeline, check_coverage, render_selection, select_evidence
from regulus.rag.coverage import (
    STATE_COMPLETE,
    STATE_INCOMPLETE,
    STATE_NO_EXPLICIT_REQUIREMENTS,
    STATUS_DISCOVERY_ONLY,
    STATUS_EXACT,
    STATUS_STRUCTURAL_ONLY,
    STATUS_UNRESOLVED,
)
from regulus.rag.selection import DROP_HARD_CEILING_CHARS, DROP_MAX_RESULTS, DROP_SOFT_BUDGET, REASON_REQUIRED_DOCKET, REASON_REQUIRED_EXACT
from regulus.retrieval import LexicalRetriever
from regulus.retrieval.evidence import TIER_DISCOVERY, TIER_EXACT, TIER_STRUCTURAL, TIER_SUPPORTING
from tests.test_rag import FakeGenerator
from tests.test_retrieval_plan import TABLE, TableDatabase, guidance_row, row

FOUR = [f"21 CFR 820.10(b)({i})" for i in range(1, 5)]


def registry_for(question, db=None, **kw):
    return LexicalRetriever(db or TableDatabase()).run_plan(plan_query(question), **kw).registry


def labels(items):
    return [getattr(i, "citation", None) or i.evidence.citation for i in items]


def selected_labels(sel):
    return [s.evidence.citation for s in sel.selected]


# ---------------------------------------------------------------- coverage


def test_single_exact_requirement_is_complete_and_protected():
    reg = registry_for("What does 820.10(b)(2) require?")
    cov = check_coverage(reg)
    assert cov.state == STATE_COMPLETE and cov.complete and cov.explicit
    r1 = cov.for_requirement("r1")
    assert r1.status == STATUS_EXACT and r1.satisfied and len(r1.exact_evidence_ids) == 1 and len(r1.structural_evidence_ids) == 1
    sel = select_evidence(reg)
    assert selected_labels(sel)[0] == "21 CFR 820.10(b)(2)" and sel.selected[0].protected and sel.selected[0].reason == REASON_REQUIRED_EXACT
    bundle = render_selection(sel)
    assert bundle.sources[0].protected and bundle.sources[0].tier == TIER_EXACT and bundle.sources[0].requirement_ids == ("r1",)
    assert "Citation: 21 CFR 820.10(b)(2)" in bundle.text


def test_four_exact_requirements_are_complete_and_all_protected():
    reg = registry_for("What do 820.10(b)(1) through 820.10(b)(4) require?")
    cov = check_coverage(reg)
    assert cov.state == STATE_COMPLETE and cov.counts()["exact_requirements"] == 4 and cov.missing_ids == ()
    sel = select_evidence(reg)
    assert selected_labels(sel)[:4] == FOUR and all(s.protected for s in sel.selected[:4])
    assert [s.requirement_ids for s in sel.selected[:4]] == [("r1",), ("r2",), ("r3",), ("r4",)]


def test_small_top_k_cannot_lose_exact_requirements():
    """top_k is the retrieval remainder bound and max_results the soft context bound; neither touches required evidence."""
    high = [row(f"801.{i}", "", 0, rank=3.0) for i in range(1, 4)]
    reg = registry_for("820.10(b)(1) through (b)(4)", TableDatabase(full_text=high), top_k=2)
    sel = select_evidence(reg, ContextPolicy(max_results=2))
    assert selected_labels(sel)[:4] == FOUR and check_coverage(reg).complete
    assert sel.diagnostics["required_count"] == 4 and sel.diagnostics["soft_budget_exceeded_by_required"]
    assert all(d.reason == DROP_MAX_RESULTS and not d.required for d in sel.dropped)


def test_supporting_evidence_with_higher_score_does_not_displace_exact():
    high = [row("801.1", "(a)", 1, rank=9.0)]  # score 450 > 100
    reg = registry_for("820.10(b)(2)", TableDatabase(full_text=high))
    assert reg.evidence[0].tier == TIER_EXACT
    sel = select_evidence(reg, ContextPolicy(max_results=1))
    assert selected_labels(sel) == ["21 CFR 820.10(b)(2)"]
    assert [d.evidence.citation for d in sel.dropped if d.evidence.tier == TIER_SUPPORTING] == ["21 CFR 801.1(a)"]


def test_missing_requirement_with_parent_is_incomplete_and_not_promoted():
    reg = registry_for("What does 820.10(b)(5) require?")
    cov = check_coverage(reg)
    assert cov.state == STATE_INCOMPLETE and not cov.complete and cov.missing_labels == ("21 CFR 820.10(b)(5)",)
    r1 = cov.for_requirement("r1")
    assert r1.status == STATUS_STRUCTURAL_ONLY and r1.exact_evidence_ids == () and len(r1.structural_evidence_ids) == 1
    assert "parent 21 CFR 820.10(b) exists" in r1.note
    sel = select_evidence(reg)
    assert sel.diagnostics["required_count"] == 0 and not any(s.protected for s in sel.selected)
    assert selected_labels(sel) == ["21 CFR 820.10(b)"] and sel.selected[0].evidence.tier == TIER_STRUCTURAL


def test_partial_multi_requirement_coverage():
    reg = registry_for("820.10(b)(1), (b)(2), (b)(9) and (b)(4)")
    cov = check_coverage(reg)
    assert cov.state == STATE_INCOMPLETE and cov.satisfied_ids == ("r1", "r2", "r4") and cov.missing_ids == ("r3",)
    assert cov.counts() == {"total_requirements": 4, "exact_requirements": 3, "structural_only_requirements": 1,
                            "supporting_only_requirements": 0, "discovery_only_requirements": 0, "unresolved_requirements": 0}
    sel = select_evidence(reg)
    assert selected_labels(sel)[:3] == [FOUR[0], FOUR[1], FOUR[3]] and all(s.protected for s in sel.selected[:3])
    assert "21 CFR 820.10(b)(9)" not in selected_labels(sel)


def test_wholly_missing_section_is_unresolved_or_discovery_only():
    db = TableDatabase(trigram=[row("314.99", "(a)", 1, sim=0.8)])
    cov = check_coverage(registry_for("What are the requirements of 21 CFR 314.999?", db))
    assert cov.state == STATE_INCOMPLETE and cov.for_requirement("r1").status == STATUS_DISCOVERY_ONLY
    cov = check_coverage(registry_for("What are the requirements of 21 CFR 314.999?", TableDatabase()))
    assert cov.for_requirement("r1").status == STATUS_UNRESOLVED and not cov.satisfied_ids


def test_structural_parent_is_deduplicated_across_requirements():
    reg = registry_for("820.10(b)(1) through (b)(4)")
    sel = select_evidence(reg)
    parents = [s for s in sel.selected if s.evidence.citation == "21 CFR 820.10(b)"]
    assert len(parents) == 1 and parents[0].requirement_ids == ("r1", "r2", "r3", "r4") and not parents[0].protected
    bundle = render_selection(sel)
    assert bundle.text.count("Citation: 21 CFR 820.10(b)\n") == 1


def test_long_required_evidence_is_kept_whole_and_truncation_is_explicit():
    long_table = [dict(r, body_text=("R" * 2500) if r["paragraph_path"] == "(b)(2)" else r["body_text"]) for r in TABLE]
    reg = registry_for("820.10(b)(2)", TableDatabase(long_table))
    sel = select_evidence(reg)  # default: max_chars_per_result 1200 but required cap 4000
    req = sel.selected[0]
    assert req.protected and not req.truncated and len(req.text) == 2500
    assert sel.diagnostics["required_truncated"] == [] and not sel.overflow
    # a required source longer than the required cap is truncated, and says so
    sel2 = select_evidence(reg, ContextPolicy(required_max_chars_per_result=1000))
    assert sel2.selected[0].protected and sel2.selected[0].truncated and sel2.diagnostics["required_truncated"] == [reg.evidence[0].evidence_id]
    assert sel2.overflow and render_selection(sel2).sources[0].truncated


def test_total_budget_sacrifices_optional_before_required():
    big_optional = [row(f"801.{i}", "", 0, rank=1.0, body_text="G" * 1100) for i in range(1, 5)]
    reg = registry_for("820.10(b)(1) through (b)(4)", TableDatabase(full_text=big_optional))
    policy = ContextPolicy(max_results=8, max_chars_per_result=1200, max_total_chars=1500)
    sel = select_evidence(reg, policy)
    assert selected_labels(sel)[:4] == FOUR
    assert sel.diagnostics["required_dropped"] == []
    # required evidence fits the soft budget; optional evidence only spends what is left, squeezed or dropped
    optional_kept = [s for s in sel.selected if not s.protected]
    assert [s.evidence.tier for s in optional_kept][:1] == [TIER_STRUCTURAL]
    assert all(s.truncated for s in optional_kept if s.evidence.tier == TIER_SUPPORTING)
    assert any(d.reason == DROP_SOFT_BUDGET for d in sel.dropped) and {d.reason for d in sel.dropped} <= {DROP_SOFT_BUDGET, DROP_MAX_RESULTS}
    assert sel.total_chars <= policy.max_total_chars


@pytest.mark.usefixtures("cpu_era_budget")   # fixtures sized to overflow the CPU-era evidence budget (tests/conftest.py)
def test_required_evidence_above_the_hard_ceiling_is_dropped_explicitly_not_silently():
    long_table = [dict(r, body_text="L" * 1500) for r in TABLE]
    reg = registry_for("820.10(b)(1) through (b)(4)", TableDatabase(long_table))
    policy = ContextPolicy(max_total_chars=3600, hard_max_total_chars=3600, required_max_chars_per_result=4000)
    sel = select_evidence(reg, policy)
    kept = [s.evidence.citation for s in sel.selected if s.protected]
    assert kept == FOUR[:2]
    assert [d.evidence.citation for d in sel.dropped if d.required] == FOUR[2:]
    assert all(d.reason == DROP_HARD_CEILING_CHARS for d in sel.dropped if d.required)
    assert sel.overflow and sel.diagnostics["required_dropped"] == [reg.evidence[2].evidence_id, reg.evidence[3].evidence_id]


def test_structural_evidence_is_not_subject_to_the_method_filter_but_discovery_is():
    # the section row found by trigram at similarity 0.36 is still the parent of the requested paragraph
    db = TableDatabase(trigram=[row("820.10", "", 0, sim=0.36), row("172.820", "", 0, sim=0.36)])
    reg = registry_for("820.10(b)(2) and 820.999", db)
    pipeline = RagPipeline(LexicalRetriever(db), FakeGenerator("x"))
    sel = select_evidence(reg, optional_filter=pipeline.is_useful)
    assert "21 CFR 820.10" in selected_labels(sel)
    assert [d.evidence.citation for d in sel.dropped if d.reason == "filtered"] == ["21 CFR 172.820"]


def test_general_query_has_no_explicit_requirements_and_keeps_legacy_context():
    db = TableDatabase(full_text=[row(f"801.{i}", "", 0, rank=0.5) for i in range(1, 5)])
    reg = registry_for("What are the labeling requirements for prescription drugs?", db)
    cov = check_coverage(reg)
    assert cov.state == STATE_NO_EXPLICIT_REQUIREMENTS and cov.complete and not cov.explicit and cov.requirements == ()
    sel = select_evidence(reg)
    assert sel.diagnostics["required_count"] == 0 and len(sel.selected) == 4 and not any(s.protected for s in sel.selected)
    assert selected_labels(sel) == ["21 CFR 801.1", "21 CFR 801.2", "21 CFR 801.3", "21 CFR 801.4"]


def test_docket_evidence_is_required_for_a_docket_question_but_never_satisfies_a_cfr_requirement():
    db = TableDatabase(docket=[guidance_row(0), guidance_row(1)])
    reg = registry_for("Summarise FDA-2016-D-2565", db)
    cov = check_coverage(reg)
    assert cov.state == STATE_NO_EXPLICIT_REQUIREMENTS
    sel = select_evidence(reg)
    assert len(sel.protected) == 2 and all(s.reason == REASON_REQUIRED_DOCKET for s in sel.protected)
    # mixed question: the CFR requirement still needs its own exact row
    db = TableDatabase(table=[r for r in TABLE if r["paragraph_path"] != "(b)(2)"], docket=[guidance_row(0)], guidance=[guidance_row(1, rank=0.4)])
    reg = registry_for("What does 21 CFR 820.10(b)(2) require, see FDA-2016-D-2565", db)
    cov = check_coverage(reg)
    assert cov.state == STATE_INCOMPLETE and cov.missing_labels == ("21 CFR 820.10(b)(2)",)


def test_citation_indices_match_rendered_sources():
    reg = registry_for("820.10(b)(2) and 820.10(b)(4)", TableDatabase(guidance=[guidance_row(0, rank=0.3)]))
    pipeline = RagPipeline(LexicalRetriever(TableDatabase(guidance=[guidance_row(0, rank=0.3)])), FakeGenerator("See [1] and [2]; also [3]."))
    out = pipeline.answer("What do 820.10(b)(2) and 820.10(b)(4) require?")
    assert [s.index for s in out.context.sources] == list(range(1, len(out.context.sources) + 1))
    assert [c.label for c in out.citations][:2] == ["21 CFR 820.10(b)(2)", "21 CFR 820.10(b)(4)"]
    assert all(c.cited_in_answer for c in out.citations[:2])
    assert f"SOURCE {len(out.context.sources)}" in out.context.text
    assert len(out.citations) <= len(out.context.sources)


def test_selection_is_deterministic():
    db_a = TableDatabase(full_text=[row(f"801.{i}", "", 0, rank=0.5) for i in range(1, 4)])
    db_b = TableDatabase(full_text=[row(f"801.{i}", "", 0, rank=0.5) for i in range(1, 4)])
    a = select_evidence(registry_for("820.10(b)(1) through (b)(4)", db_a))
    b = select_evidence(registry_for("820.10(b)(1) through (b)(4)", db_b))
    assert [s.to_dict() for s in a.selected] == [s.to_dict() for s in b.selected]
    assert a.to_dict() == b.to_dict()
    json.dumps(a.to_dict())


def test_context_policy_ceilings_validate():
    # a hard ceiling never sits below the soft budget it bounds (legacy configs with huge soft budgets stay valid)
    p = ContextPolicy(max_total_chars=100000, hard_max_total_chars=3000)
    assert p.hard_max_total_chars == 100000 and p.hard_max_results == 24
    with pytest.raises(ValueError):
        ContextPolicy(required_max_chars_per_result=0)
    assert ContextPolicy().to_dict()["hard_max_total_chars"] == 9000


# ---------------------------------------------------------------- pipeline


def test_pipeline_complete_coverage_answers_normally_with_all_provisions_in_context():
    gen = FakeGenerator("(b)(1) [1] (b)(2) [2] (b)(3) [3] (b)(4) [4].")
    out = RagPipeline(LexicalRetriever(TableDatabase()), gen, RagConfig(top_k=4, context=ContextPolicy(max_results=4))).answer(
        "List all requirements in 21 CFR 820.10(b)(1) through (b)(4)."
    )
    assert out.coverage.state == STATE_COMPLETE and out.metadata["coverage_action"] == "answer" and out.model_called
    assert [s.result.citation.label for s in out.context.sources[:4]] == FOUR
    assert out.metadata["coverage"]["complete"] and out.metadata["selection"]["required_count"] == 4
    assert not out.answer.startswith("Not found")


def test_pipeline_partial_coverage_answers_supported_part_and_names_the_rest():
    gen = FakeGenerator("(b)(1) requires UDI [1].")
    out = RagPipeline(LexicalRetriever(TableDatabase()), gen).answer("What do 820.10(b)(1) and 820.10(b)(9) require?")
    assert out.coverage.state == STATE_INCOMPLETE and out.metadata["coverage_action"] == "answer_partial" and out.model_called
    assert out.answer.startswith("(b)(1) requires UDI [1].") and "Not found in the available eCFR data: 21 CFR 820.10(b)(9)" in out.answer
    prompt = gen.messages[0][1]["content"]
    assert "NOTE: Requested provision(s) not among the sources: 21 CFR 820.10(b)(9)" in prompt
    assert "Citation: 21 CFR 820.10(b)(9)" not in prompt


def test_pipeline_unsupported_explicit_request_is_deterministic_and_skips_the_model():
    gen = FakeGenerator("21 CFR 820.10(b)(5) requires invented things.")
    out = RagPipeline(LexicalRetriever(TableDatabase()), gen).answer("What does 21 CFR 820.10(b)(5) require?")
    assert not out.model_called and gen.messages == []
    assert out.metadata["coverage_action"] == "refuse_unsupported"
    assert out.answer.startswith("The available eCFR data does not contain 21 CFR 820.10(b)(5)")
    assert "invented" not in out.answer
    # the structural parent was still selected, so the context is not empty and the citation list reflects it
    assert not out.no_context and out.context.sources[0].result.citation.label == "21 CFR 820.10(b)"


def test_pipeline_incomplete_behavior_model_keeps_the_legacy_path():
    gen = FakeGenerator("model text")
    out = RagPipeline(LexicalRetriever(TableDatabase()), gen, RagConfig(incomplete_behavior="model")).answer("What does 21 CFR 820.10(b)(5) require?")
    assert out.model_called and out.answer == "model text" and out.metadata["coverage_action"] == "answer"
    with pytest.raises(ValueError):
        RagConfig(incomplete_behavior="ignore")


def test_pipeline_general_question_is_not_treated_as_incomplete():
    db = TableDatabase(full_text=[row("801.1", "", 0, rank=0.5)])
    out = RagPipeline(LexicalRetriever(db), FakeGenerator("text")).answer("What are the labeling requirements for drugs?")
    assert out.coverage.state == STATE_NO_EXPLICIT_REQUIREMENTS and out.metadata["coverage_action"] == "answer" and out.model_called


def test_pipeline_grounding_gate_still_runs_first():
    from tests.test_grounding import FakeIndex
    from regulus.grounding import GroundingConfig, GroundingGate

    gate = GroundingGate(FakeIndex(), GroundingConfig(enabled=True), snapshot_date="2026-09-03")
    gen = FakeGenerator("x")
    out = RagPipeline(LexicalRetriever(TableDatabase()), gen, grounding=gate).answer("What are the specific requirements of 21 CFR 314.999?")
    assert not out.model_called and out.metadata["grounding"]["action"] == "refuse"
    assert out.coverage.state == STATE_INCOMPLETE and out.metadata["coverage_action"] is None


def test_legacy_retriever_path_has_no_coverage():
    from tests.test_rag import E, FakeRetriever

    out = RagPipeline(FakeRetriever([E()]), FakeGenerator()).answer("What does 21 CFR 820.10(b) say?")
    assert out.coverage is None and out.selection is None and out.metadata["coverage"] is None and out.metadata["coverage_action"] is None
    assert out.context.sources[0].tier is None and not out.context.sources[0].protected
