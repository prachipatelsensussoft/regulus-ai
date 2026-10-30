"""Stage 4: deterministic answer validation, citation tracing, and the
pipeline's handling of answers that fail validation.

FakeGenerator answers are scripted so each failure mode is exercised on
purpose; the evidence path underneath is the real Stage 1-3 code on the
in-memory table.
"""

import json

import pytest

from regulus.inference.generate import GenerationResult
from regulus.rag import RagConfig, RagPipeline
from regulus.rag.citations import ROLE_AUTHORITATIVE, ROLE_CITED, ROLE_CONTEXT, build_answer_citations, citation_support
from regulus.rag.generation import GenerationBudget, UNVALIDATED_GENERAL, UNVALIDATED_PREFIX
from regulus.rag.validation import (
    ASSERTED_MISSING_PROVISION,
    CITATION_MISMATCH,
    DEGENERATE_OUTPUT,
    DISCOVERY_CITED_FOR_REQUIREMENT,
    EMPTY_OUTPUT,
    INVALID_CITATION,
    MALFORMED_CITATION,
    REFUSAL_WITH_EVIDENCE,
    REQUIREMENT_NOT_ADDRESSED,
    STATUS_CITATION_INVALID,
    STATUS_DEGENERATE,
    STATUS_EMPTY,
    STATUS_EVIDENCE_MISMATCHED,
    STATUS_INCOMPLETE,
    STATUS_UNSUPPORTED,
    STATUS_VALID,
    TRUNCATED,
    UNCITED_STATEMENT,
    UNSUPPORTED_FIGURE,
    PROVISION_CITATION_MISMATCH,
    LOW_OVERLAP,
    strip_invalid_citations,
    validate_answer,
)
from regulus.retrieval import LexicalRetriever
from regulus.retrieval.evidence import TIER_EXACT
from tests.test_generation_contract import package_for
from tests.test_retrieval_plan import TABLE, TableDatabase, guidance_row, row

TWO = "What do 820.10(b)(2) and 820.10(b)(4) require?"
FOUR = "What do 820.10(b)(1) through 820.10(b)(4) require?"


def gen(response, tokens=40, max_tokens=None, finished=True):
    return GenerationResult("p", response, 500, tokens, 1.0, finished=finished)


class ScriptedGenerator:
    """Returns one scripted answer and records the budget it was given."""

    def __init__(self, response, finished=True, tokens=40):
        self.response, self.finished, self.tokens = response, finished, tokens
        self.calls: list[dict] = []

    def generate(self, messages, max_new_tokens=None):
        self.calls.append({"messages": messages, "max_new_tokens": max_new_tokens})
        return GenerationResult("p", self.response, 500, self.tokens, 1.0, finished=self.finished)

    def describe(self):
        return {"model_id": "scripted"}


def pipeline(db=None, response="x", **cfg):
    g = ScriptedGenerator(response)
    return RagPipeline(LexicalRetriever(db or TableDatabase()), g, RagConfig(**cfg)), g


# ------------------------------------------------------------ completeness


def test_complete_two_provision_answer_is_valid():
    pkg = package_for(TWO)
    r = validate_answer("21 CFR 820.10(b)(2): requires traceability procedures [1]. 21 CFR 820.10(b)(4): advisory notices handled as required [2].", pkg, gen("x"))
    assert r.status == STATUS_VALID and r.ok and r.addressed == ("r1", "r2") and r.unaddressed == () and r.cited == (1, 2)
    assert r.diagnostics["cited_tiers"] == {"1": "exact", "2": "exact"}


def test_omitted_provision_is_incomplete():
    r = validate_answer("21 CFR 820.10(b)(2): requires traceability procedures [1].", package_for(TWO), gen("x"))
    assert r.status == STATUS_INCOMPLETE and r.unaddressed == ("r2",) and REQUIREMENT_NOT_ADDRESSED in r.codes()
    assert r.problems[0].requirement_id == "r2"


def test_an_optional_provision_left_out_is_a_warning_not_an_error():
    # SP2: a second pick added as further evidence (GenerationPackage.optional_labels) never withholds the answer
    from dataclasses import replace
    pkg = package_for(TWO)
    label = pkg.requirement_labels["r2"]
    r = validate_answer("21 CFR 820.10(b)(2): requires traceability procedures [1].", replace(pkg, optional_labels=frozenset({label})), gen("x"))
    assert r.ok and r.unaddressed == ("r2",)
    assert [p.severity for p in r.problems if p.code == REQUIREMENT_NOT_ADDRESSED] == ["warning"]


def test_arbitrary_textual_order_is_accepted():
    r = validate_answer("Under (b)(4), advisory notices [2]. Under (b)(2), traceability [1].", package_for(TWO), gen("x"))
    assert r.ok and set(r.addressed) == {"r1", "r2"}


def test_range_answer_using_short_paragraph_forms():
    pkg = package_for(FOUR)
    r = validate_answer("(b)(1) UDI [1]. (b)(2) traceability [2]. (b)(3) complaints [3]. (b)(4) advisory notices [4].", pkg, gen("x"))
    assert r.ok and r.addressed == ("r1", "r2", "r3", "r4")
    r = validate_answer("(b)(1) UDI [1]. (b)(2) traceability [2]. (b)(4) advisory notices [4].", pkg, gen("x"))
    assert r.status == STATUS_INCOMPLETE and r.unaddressed == ("r3",)


def test_addressed_by_citing_the_exact_source_without_naming_it():
    r = validate_answer("The first requires a UDI system [1]. The second requires traceability [2].", package_for("820.10(b)(1) and (b)(2)"), gen("x"))
    assert r.ok and r.addressed == ("r1", "r2")


def test_partial_range_coverage_negated_mention_is_fine_and_positive_mention_is_unsupported():
    pkg = package_for("What do 820.10(b)(1) through 820.10(b)(5) require?")
    good = "(b)(1) [1]. (b)(2) [2]. (b)(3) [3]. (b)(4) [4]. 21 CFR 820.10(b)(5) was not found in the available eCFR data."
    r = validate_answer(good, pkg, gen("x"))
    assert r.ok
    bad = "(b)(1) [1]. (b)(2) [2]. (b)(3) [3]. (b)(4) [4]. 21 CFR 820.10(b)(5) requires annual audits of the QMS [5]."
    r = validate_answer(bad, pkg, gen("x"))
    assert r.status == STATUS_UNSUPPORTED and ASSERTED_MISSING_PROVISION in r.codes() and r.problems[-1].requirement_id == "r5"


def test_refusal_with_exact_evidence_is_flagged():
    r = validate_answer("The available regulatory sources do not provide enough information to answer this.", package_for(TWO), gen("x"))
    assert r.status == STATUS_INCOMPLETE and REFUSAL_WITH_EVIDENCE in r.codes() and r.answer_class == "refusal"


# ---------------------------------------------------------------- citations


def test_invalid_citation_number():
    pkg = package_for(TWO)
    r = validate_answer("(b)(2) traceability [1]. (b)(4) advisory notices [17].", pkg, gen("x"))
    assert r.status == STATUS_CITATION_INVALID and r.invalid_citations == (17,) and INVALID_CITATION in r.codes()
    assert strip_invalid_citations("(b)(4) advisory notices [17].", (17,)) == "(b)(4) advisory notices."


def test_citation_to_wrong_source_is_a_mismatch():
    pkg = package_for(TWO)  # SOURCE 1 = (b)(2), SOURCE 2 = (b)(4), SOURCE 3 = (b) parent
    r = validate_answer("21 CFR 820.10(b)(2) requires traceability [2]. 21 CFR 820.10(b)(4) covers advisory notices [1].", pkg, gen("x"))
    assert r.status == STATUS_EVIDENCE_MISMATCHED
    assert {p.code for p in r.errors} == {CITATION_MISMATCH, PROVISION_CITATION_MISMATCH}
    assert [p.requirement_id for p in r.errors if p.code == CITATION_MISMATCH] == ["r1", "r2"]


def test_markers_after_the_full_stop_attach_to_their_own_sentence():
    pkg = package_for(TWO)
    # the marker follows the period: it belongs to the (b)(2) statement, not to the (b)(4) one
    r = validate_answer("21 CFR 820.10(b)(2): traceability procedures. [1]\n\n21 CFR 820.10(b)(4): advisory notices. [2]", pkg, gen("x"))
    assert r.ok and r.addressed == ("r1", "r2")
    # the real failure from the audit: (b)(4) described from a guidance chunk and cited to it
    db = TableDatabase(guidance=[guidance_row(0, rank=0.4)])
    pkg = package_for(TWO, db)
    sup = next(s.index for s in pkg.sources if s.tier == "supporting")
    r = validate_answer(f"21 CFR 820.10(b)(2): traceability. [1]\n\n21 CFR 820.10(b)(4): a UDI system for every device. [{sup}]", pkg, gen("x"))
    assert r.status == STATUS_EVIDENCE_MISMATCHED
    prob = next(p for p in r.problems if p.code == CITATION_MISMATCH)
    assert prob.requirement_id == "r2" and prob.sentence.startswith("21 CFR 820.10(b)(4)")


def test_marker_after_a_closing_quote_attaches_to_its_own_sentence():
    from regulus.rag.validation import attach_trailing_markers

    assert attach_trailing_markers('He said "chapter." [3]\nNext') == 'He said "chapter [3]."\nNext'
    pkg = package_for(FOUR)
    text = '21 CFR 820.10(b)(2): text "ends here." [2]\n21 CFR 820.10(b)(3) (eCFR, binding, verbatim): body [3].\n21 CFR 820.10(b)(4) (eCFR, binding, verbatim): body [4].\n21 CFR 820.10(b)(1): x [1].'
    assert validate_answer(text, pkg, gen('x')).ok


def test_pipeline_replaces_a_mismatched_statement_with_verbatim_exact_text():
    db = TableDatabase(guidance=[guidance_row(0, rank=0.4)])
    fake = "21 CFR 820.10(b)(2): traceability procedures. [1]\n\n21 CFR 820.10(b)(4): a UDI system for every device. [4]"
    # the statement cites a related (optional) source; Stage 6 leaves those out when the exact evidence is
    # complete, so the Stage 5 context is kept here to exercise the mismatch repair itself
    out = RagPipeline(LexicalRetriever(db), ScriptedGenerator(fake), RagConfig(optional_context="always")).answer(TWO)
    assert out.metadata["validation"]["initial"]["status"] == STATUS_EVIDENCE_MISMATCHED
    assert out.metadata["validation"]["repair"] == "mismatched_statements_replaced"
    assert "UDI system for every device" not in out.answer and "traceability procedures" in out.answer
    assert "21 CFR 820.10(b)(4) (eCFR, binding, verbatim): 820.10(b)(4) body text [2]." in out.answer
    assert out.validation.ok and out.validation.addressed == ("r1", "r2")


def test_structural_citation_for_its_requirement_is_accepted():
    pkg = package_for(TWO)
    r = validate_answer("21 CFR 820.10(b)(2) sits under the ISO clause list [3] and requires traceability [1]. 21 CFR 820.10(b)(4) covers advisory notices [2].", pkg, gen("x"))
    assert r.ok


def test_supporting_citation_in_a_general_answer_is_fine():
    db = TableDatabase(full_text=[row("801.1", "", 0, rank=0.5), row("801.2", "", 0, rank=0.4)])
    r = validate_answer("Labeling must carry the name [1] and the quantity [2].", package_for("What must a drug label carry?", db), gen("x"))
    assert r.ok and r.diagnostics["explicit"] is False


def test_discovery_citation_cannot_support_a_requested_provision():
    db = TableDatabase(trigram=[row("172.820", "", 0, sim=0.7)])
    pkg = package_for("What do 820.10(b)(2) and 820.999 require?", db)
    disc = next(s.index for s in pkg.sources if s.tier == "discovery")
    r = validate_answer(f"21 CFR 820.10(b)(2) requires traceability [{disc}].", pkg, gen("x"))
    assert r.status == STATUS_UNSUPPORTED and DISCOVERY_CITED_FOR_REQUIREMENT in r.codes()


def test_malformed_citation_syntax_is_a_warning():
    r = validate_answer("(b)(2) traceability [1-2]. (b)(4) advisory notices [2].", package_for(TWO), gen("x"))
    assert MALFORMED_CITATION in r.codes() and r.diagnostics["warnings"] == 1


def test_uncited_obligation_statement_is_a_warning_only():
    r = validate_answer("21 CFR 820.10(b)(2) requires traceability procedures. 21 CFR 820.10(b)(4) requires handling advisory notices [2].", package_for(TWO), gen("x"))
    assert r.ok and UNCITED_STATEMENT in r.codes() and r.problems[0].requirement_id == "r1"


def test_answer_citations_list_only_cited_and_authoritative_sources():
    pkg = package_for(TWO, TableDatabase(guidance=[guidance_row(0, rank=0.4)]))
    answer = "21 CFR 820.10(b)(2) requires traceability [1]. 21 CFR 820.10(b)(4) covers advisory notices."
    r = validate_answer(answer, pkg, gen("x"))
    cites = build_answer_citations(answer, pkg, r.addressed)
    assert [(c.label, c.role, c.cited_in_answer) for c in cites] == [("21 CFR 820.10(b)(2)", ROLE_CITED, True), ("21 CFR 820.10(b)(4)", ROLE_AUTHORITATIVE, False)]
    assert cites[0].tier == "exact" and cites[0].requirement_ids == ("r1",)
    assert not any("Guidance" in c.label for c in cites)  # in context, not cited, not authoritative: not listed
    support = citation_support(answer, pkg)
    assert support["by_tier"] == {"exact": 1, "structural": 0, "supporting": 0, "discovery": 0, "invalid": 0}
    assert support["exact_source_cited_for"] == {"r1": True, "r2": False}


def test_over_citation_is_visible_in_citation_support():
    pkg = package_for("820.10(b)(2)", TableDatabase(guidance=[guidance_row(0, rank=0.4), guidance_row(1, rank=0.3)]))
    support = citation_support("21 CFR 820.10(b)(2) requires traceability [1][2][3][4].", pkg)
    assert support["by_tier"]["exact"] == 1 and support["by_tier"]["structural"] == 1 and support["by_tier"]["supporting"] == 2


def test_figures_absent_from_every_source_are_unsupported():
    pkg = package_for("What penalty does 820.10(b)(3) specify?")
    r = validate_answer("21 CFR 820.10(b)(3) specifies a penalty amount of $1,000 for late notification [1].", pkg, gen("x"))
    assert r.status == STATUS_UNSUPPORTED and UNSUPPORTED_FIGURE in r.codes() and "1000" in r.problems[0].detail
    # numbers that do appear in a source or in the requested label are fine
    r = validate_answer("21 CFR 820.10(b)(3) body text [1].", pkg, gen("x"))
    assert UNSUPPORTED_FIGURE not in r.codes()
    out = RagPipeline(LexicalRetriever(TableDatabase()), ScriptedGenerator("21 CFR 820.10(b)(3) specifies a penalty of $1,000 [1].")).answer("What penalty does 820.10(b)(3) specify?")
    assert out.metadata["validation"]["repair"] == "extractive_replacement" and "$1,000" not in out.answer


def test_named_provision_must_be_cited_to_its_own_source():
    pkg = package_for("What does 21 CFR 820.10 say?")  # every row of the section is exact for r1
    idx = {s.citation: s.index for s in pkg.sources}
    wrong = f"21 CFR 820.10(b)(1) states that manufacturers must document traceability procedures [{idx['21 CFR 820.10(b)(2)']}]."
    r = validate_answer(wrong, pkg, gen("x"))
    assert r.status == STATUS_EVIDENCE_MISMATCHED and PROVISION_CITATION_MISMATCH in r.codes()
    right = f"21 CFR 820.10(b)(1) states that manufacturers must document a UDI system [{idx['21 CFR 820.10(b)(1)']}]."
    # attributed to the requested provision when the sentence is about exactly one
    pkg2 = package_for(TWO)
    r2 = validate_answer("21 CFR 820.10(b)(4): a UDI system as in 21 CFR 820.10(b)(1) [1].", pkg2, gen("x"))
    assert PROVISION_CITATION_MISMATCH in r2.codes() and next(p for p in r2.problems if p.code == PROVISION_CITATION_MISMATCH).requirement_id == "r2"
    assert validate_answer(right, pkg, gen("x")).ok
    parent_ok = f"21 CFR 820.10(b)(1) sits under paragraph (b) [{idx['21 CFR 820.10(b)']}]."
    assert PROVISION_CITATION_MISMATCH not in validate_answer(parent_ok, pkg, gen("x")).codes()


def test_low_overlap_with_the_cited_exact_source_is_a_warning_only():
    pkg = package_for(TWO)
    r = validate_answer("21 CFR 820.10(b)(2): 820.10(b)(2) body text [1]. 21 CFR 820.10(b)(4): the submitter must assign a unique device identifier to every medical device before distribution [2].", pkg, gen("x"))
    assert LOW_OVERLAP in r.codes() and all(p.severity == "warning" for p in r.problems if p.code == LOW_OVERLAP)
    # the same sentence is also an unsupported claim (no supplied source has its words), an error since the
    # 16 Oct 2026 severity experiment: that, not the low overlap, is what makes the report fail
    assert {p.code for p in r.errors} == {"unsupported_claim"} and r.status == STATUS_UNSUPPORTED
    r = validate_answer("21 CFR 820.10(b)(4): 820.10(b)(4) body text [2].", pkg, gen("x"))
    assert LOW_OVERLAP not in r.codes()


# ------------------------------------------------------------- robustness


def test_truncation_is_detected_from_the_finish_flag_and_from_the_cap():
    pkg = package_for(TWO)
    r = validate_answer("(b)(2) traceability [1]. (b)(4) advisory", pkg, gen("x", finished=False))
    assert r.truncated and TRUNCATED in r.codes() and r.status == STATUS_INCOMPLETE
    r = validate_answer("(b)(2) traceability [1]. (b)(4) advisory notices [2].", pkg, GenerationResult("p", "x", 1, pkg.max_new_tokens, 1.0))
    assert r.truncated and r.finished is None
    r = validate_answer("(b)(2) traceability [1]. (b)(4) advisory notices [2].", pkg, gen("x", finished=True))
    assert not r.truncated and r.ok


def test_empty_and_degenerate_outputs():
    pkg = package_for(TWO)
    assert validate_answer("", pkg, gen("")).status == STATUS_EMPTY
    loop = " ".join(["The available regulatory sources do not provide enough information to answer this."] * 4)
    r = validate_answer(loop, pkg, gen(loop))
    assert r.status == STATUS_DEGENERATE and DEGENERATE_OUTPUT in r.codes()
    ngram_loop = "part 806 of this chapter " * 20
    assert validate_answer(ngram_loop, pkg, gen(ngram_loop)).status == STATUS_DEGENERATE


def test_validation_is_deterministic():
    pkg = package_for(FOUR)
    a = validate_answer("(b)(1) [1]. (b)(2) [2]. (b)(3) [3].", pkg, gen("x"))
    b = validate_answer("(b)(1) [1]. (b)(2) [2]. (b)(3) [3].", pkg, gen("x"))
    assert a.to_dict() == b.to_dict()
    json.dumps(a.to_dict())


# ---------------------------------------------------------------- pipeline


def test_pipeline_passes_the_budget_and_uses_the_grounded_prompt():
    p, g = pipeline(response="21 CFR 820.10(b)(2): traceability [1].\n21 CFR 820.10(b)(4): advisory notices [2].")
    out = p.answer(TWO)
    b = out.package.budget
    evidence = b.evidence_tokens(sum(len(s.text) for s in out.package.sources if s.tier == TIER_EXACT))
    expected = min(max(b.base_tokens + b.per_unit_tokens * 2, evidence, b.min_tokens), b.hard_max_tokens)
    assert g.calls[0]["max_new_tokens"] == expected and out.package.max_new_tokens == expected
    user = g.calls[0]["messages"][1]["content"]
    assert "REQUESTED PROVISIONS" in user and "Tier: EXACT (authoritative text of r1)" in user
    assert out.validation.ok and out.metadata["validation"]["repair"] is None and out.metadata["generation"]["contract"] == "grounded_v1"
    assert [c.role for c in out.citations] == [ROLE_CITED, ROLE_CITED]
    assert "validation_ms" in out.timings_ms and out.metadata["citation_support"]["by_tier"]["exact"] == 2


def test_pipeline_replaces_an_unsupported_answer_with_verbatim_exact_text():
    fake = "21 CFR 820.10(b)(2) requires traceability [1]. 21 CFR 820.999 requires a written recall plan [4]."
    p, g = pipeline(TableDatabase(trigram=[row("172.820", "", 0, sim=0.7)]), response=fake)
    out = p.answer("What do 21 CFR 820.10(b)(2) and 21 CFR 820.999 require?")
    assert out.metadata["validation"]["initial"]["status"] == STATUS_UNSUPPORTED
    assert out.metadata["validation"]["repair"] == "extractive_replacement"
    assert out.answer.startswith(UNVALIDATED_PREFIX)
    assert "21 CFR 820.10(b)(2) (eCFR, binding, verbatim): 820.10(b)(2) body text [1]." in out.answer
    assert "21 CFR 820.999: not found in the available eCFR data." in out.answer
    assert "recall plan" not in out.answer and out.answer.count("not found in the available eCFR data") == 1
    assert out.validation.ok and out.metadata["coverage_action"] == "answer_partial"
    assert [c.role for c in out.citations] == [ROLE_CITED]


def test_pipeline_appends_omitted_provisions_verbatim():
    p, g = pipeline(response="21 CFR 820.10(b)(1): UDI system [1]. 21 CFR 820.10(b)(2): traceability [2].")
    out = p.answer(FOUR)
    assert out.metadata["validation"]["initial"]["unaddressed"] == ["r3", "r4"]
    assert out.metadata["validation"]["repair"] == "appended_missing_provisions"
    assert out.answer.startswith("21 CFR 820.10(b)(1): UDI system [1].")
    assert "21 CFR 820.10(b)(3) (eCFR, binding, verbatim): 820.10(b)(3) body text [3]." in out.answer
    assert out.validation.ok and out.validation.addressed == ("r1", "r2", "r3", "r4")
    assert [c.label for c in out.citations] == [f"21 CFR 820.10(b)({i})" for i in range(1, 5)]


def test_pipeline_removes_invalid_citations_and_notes_truncation():
    g = ScriptedGenerator("21 CFR 820.10(b)(2): traceability [9]. 21 CFR 820.10(b)(4): advisory", finished=False)
    out = RagPipeline(LexicalRetriever(TableDatabase()), g).answer(TWO)
    assert out.metadata["validation"]["initial"]["status"] == STATUS_CITATION_INVALID
    assert "[9]" not in out.answer
    # truncation is stated, not hidden, and is not dressed up as a finished answer
    assert "Incomplete: generation stopped at the token limit" in out.answer
    assert out.metadata["validation"]["repair"] == "invalid_citations_removed+truncation_noted"
    assert out.validation.status == STATUS_INCOMPLETE and out.validation.truncated  # truncation cannot be repaired away


def test_pipeline_refusal_with_evidence_becomes_extractive():
    p, g = pipeline(response="The available regulatory sources do not provide enough information to answer this.")
    out = p.answer("What does 21 CFR 820.10(b)(2) require?")
    assert out.metadata["validation"]["repair"] == "extractive_replacement" and "820.10(b)(2) body text [1]." in out.answer


def test_pipeline_degenerate_general_answer_gets_a_safe_message():
    db = TableDatabase(full_text=[row("801.1", "", 0, rank=0.5)])
    loop = "Labels must carry the name " + "of the product and the name " * 12 + "of the product."  # Stage 8: an in-sentence loop
    p, g = pipeline(db, response=loop)
    out = p.answer("What must a drug label carry?")
    # Stage 5: the answer is withheld by the safe fallback builder, which names the sources but quotes none
    assert out.answer.startswith(UNVALIDATED_GENERAL.rstrip(".")) and "801.1 body" not in out.answer
    assert out.metadata["validation"]["repair"] == "unvalidated_answer_withheld" and out.citations == []
    assert out.metadata["fallback_type"] == "unvalidated_answer_withheld" and out.metadata["raw_model_output"] == loop


def test_pipeline_annotate_and_off_modes():
    fake = "21 CFR 820.10(b)(2): traceability [1]."
    p, g = pipeline(response=fake, validation_behavior="annotate")
    out = p.answer(TWO)
    assert out.answer == fake and out.validation.status == STATUS_INCOMPLETE and out.metadata["validation"]["repair"] is None
    p, g = pipeline(response=fake, validation_behavior="off")
    out = p.answer(TWO)
    assert out.answer == fake and out.validation is None and out.metadata["validation"] is None
    with pytest.raises(ValueError):
        RagConfig(validation_behavior="maybe")


def test_pipeline_deterministic_paths_do_not_call_the_model_or_cite():
    p, g = pipeline(response="21 CFR 820.10(b)(5) requires invented audits.")
    out = p.answer("What does 21 CFR 820.10(b)(5) require?")
    assert not out.model_called and g.calls == [] and out.citations == [] and out.validation is None
    assert out.answer.startswith("The available eCFR data does not contain 21 CFR 820.10(b)(5)")
    out = p.answer("What are the specific requirements of 21 CFR 314.999?")
    assert not out.model_called and "does not contain 21 CFR 314.999" in out.answer


def test_pipeline_partial_range_keeps_the_unresolved_statement_after_validation():
    p, g = pipeline(response="(b)(1) [1]. (b)(2) [2]. (b)(3) [3]. (b)(4) [4].")
    out = p.answer("What do 820.10(b)(1) through 820.10(b)(5) require?")
    assert out.validation.ok and out.answer.endswith("Not found in the available eCFR data: 21 CFR 820.10(b)(5). The answer above covers only the provisions that were retrieved.")
    assert out.metadata["coverage_action"] == "answer_partial" and len(out.citations) == 4


def test_legacy_retriever_path_is_unchanged():
    from tests.test_rag import E, FakeGenerator, FakeRetriever

    out = RagPipeline(FakeRetriever([E()]), FakeGenerator("Answer text [1].")).answer("What does 21 CFR 820.10(b) say?")
    assert out.package is None and out.validation is None and out.metadata["generation"] is None
    assert [c.role for c in out.citations] == [ROLE_CONTEXT] and out.citations[0].cited_in_answer


def test_generator_without_budget_parameter_still_works():
    from tests.test_rag import FakeGenerator

    out = RagPipeline(LexicalRetriever(TableDatabase()), FakeGenerator("21 CFR 820.10(b)(2): traceability [1].")).answer("What does 21 CFR 820.10(b)(2) require?")
    assert out.model_called and out.validation.ok and out.metadata["generation"]["finished"] is None


def test_config_description_includes_stage4_settings():
    d = RagConfig(generation_budget=GenerationBudget(hard_max_tokens=300)).to_dict()
    assert d["generation_budget"]["hard_max_tokens"] == 300 and d["validation_behavior"] == "repair"


# ------------------------------------------------------------ CL1 citations without content (29 Oct)


def test_cl1_a_list_of_citations_that_states_nothing():
    from regulus.rag.validation import citations_only
    assert citations_only("21 CFR 820.10(b) [1]\n21 CFR 820.10(b)(1) [2]\n21 CFR 820.45(a) [7]")
    assert not citations_only("21 CFR 820.10(b)(1) [2]: requires a documented system to assign unique device identifiers.\n"
                              "21 CFR 820.45 [6]")
    assert not citations_only("21 CFR 820.10(b) [1]")   # one line: a short answer, not a list
    # terse statements say something; pointers to a citation do not (30 Oct, promoting CL1)
    assert not citations_only("21 CFR 820.10(b)(2): traceability procedures. [1]\n\n21 CFR 820.10(b)(4): advisory notices. [2]")
    assert citations_only("See 21 CFR 820.45(a)(1) [8]\nSee also 21 CFR 820.45(b) [9]\nUnder part 820 [3]")


def test_cl1_rejects_a_contentless_answer_only_when_on(monkeypatch):
    from regulus.rag import validator_switches as VS
    bare = "21 CFR 820.10(b)(2) [1]\n21 CFR 820.10(b)(4) [2]"
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"CL1"})
    assert "degenerate_output" in validate_answer(bare, package_for(TWO), gen(bare)).codes()
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"CL1"})
    assert "degenerate_output" not in validate_answer(bare, package_for(TWO), gen(bare)).codes()
