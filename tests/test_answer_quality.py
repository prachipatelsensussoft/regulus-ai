"""Answer-quality Stage 3: question analysis, answer profiles, scope handling,
answer normalisation and quality flags, all behind answer_shape="structured".

The default ("adaptive") pipeline must be unchanged: every Stage 3 behaviour is
tested both ways. In-memory eCFR table, scripted generators: no PostgreSQL, no model.
"""

from __future__ import annotations

import re

import pytest

from regulus.planning import plan_query
from regulus.planning.intent import (
    SCOPE_IN,
    SCOPE_MIXED,
    SCOPE_OUT,
    TYPE_APPLICABILITY,
    TYPE_CITATION,
    TYPE_COMPARISON,
    TYPE_DEFINITION,
    TYPE_EXCEPTION,
    TYPE_EXPLANATION,
    TYPE_OUT_OF_SCOPE,
    TYPE_PROCEDURE,
    TYPE_REQUIREMENT,
    TYPE_SCENARIO,
    analyze_question,
)
from regulus.rag import RagConfig, RagPipeline
from regulus.rag.answer_plan import PROFILE_TAIL, PROFILES, WEAK_EVIDENCE, profile_for
from regulus.rag.answer_quality import mixed_scope_note, normalize_answer, out_of_scope_answer, quality_flags
from regulus.rag.generation import GROUNDED_SYSTEM_PROMPT, build_grounded_messages, render_package
from regulus.rag.validation import validate_answer
from regulus.retrieval import LexicalRetriever
from tests.test_answer_shape import SYNTHESIS, TABLE, TEXT, gen
from tests.test_answer_validation import ScriptedGenerator
from tests.test_retrieval_plan import TableDatabase, guidance_row, row


def analysis(q):
    return analyze_question(plan_query(q))


def pipe(response="x", mode="structured", db=None, **kw):
    g = ScriptedGenerator(response, **kw)
    return RagPipeline(LexicalRetriever(db or TableDatabase(TABLE)), g, RagConfig(answer_shape=mode)), g


# ------------------------------------------------------------ classification


@pytest.mark.parametrize("question, kind", [
    ("What does 21 CFR 820.10(b)(2) require?", TYPE_REQUIREMENT),
    ("What is Part 821?", TYPE_DEFINITION),
    ("What is a quality management system?", TYPE_DEFINITION),
    ("Explain 21 CFR 820.10(b)(3).", TYPE_EXPLANATION),
    ("Summarize 21 CFR 803.13.", TYPE_EXPLANATION),
    ("How does a manufacturer submit a 510(k)?", TYPE_PROCEDURE),
    ("Does 21 CFR 820.10(b)(2) require annual traceability audits?", TYPE_APPLICABILITY),
    ("Who must comply with 21 CFR 820.10(d)?", TYPE_APPLICABILITY),
    ("What are the exemptions from 21 CFR 807.81(a)?", TYPE_EXCEPTION),
    ("Compare 21 CFR 820.10(c) and 820.10(d).", TYPE_COMPARISON),
    ("If a manufacturer's device supports life, what does 21 CFR 820.10(d) require?", TYPE_SCENARIO),
    ("Which provision of 21 CFR 820.10 refers to part 806?", TYPE_CITATION),
    ("Cite the provision that requires manufacturers to notify FDA of complaints.", TYPE_CITATION),
    ("What does the EU Medical Device Regulation require for UDI?", TYPE_OUT_OF_SCOPE),
])
def test_each_question_type(question, kind):
    a = analysis(question)
    assert a.question_type == kind, a.to_dict()
    assert a.signals


def test_yes_no_is_recognised_and_premise_questions_are_applicability():
    a = analysis("Does 21 CFR 820.10(b)(2) require annual traceability audits?")
    assert a.is_yes_no and a.question_type == TYPE_APPLICABILITY
    assert not analysis("What does 21 CFR 820.10(b)(2) require?").is_yes_no


def test_unknown_questions_default_to_explanation_not_refusal():
    a = analysis("Tell me something.")
    assert a.question_type == TYPE_EXPLANATION and a.scope_status == SCOPE_IN


def test_ambiguous_wording_follows_the_documented_precedence():
    # comparison beats requirement; citation lookup beats requirement
    assert analysis("Compare what 21 CFR 820.10(b)(2) and 820.10(b)(4) require.").question_type == TYPE_COMPARISON
    assert analysis("Which section requires traceability under 21 CFR 820?").question_type == TYPE_CITATION


def test_classification_is_deterministic():
    q = "Does 21 CFR 820.10(b)(2) require annual traceability audits?"
    assert analysis(q) == analysis(q) and analysis(q).to_dict() == analysis(q).to_dict()


# ------------------------------------------------------------ scope (conservative by evidence)


@pytest.mark.parametrize("question", [
    "What does the EU Medical Device Regulation require for UDI?",
    "What does EU MDR Article 10 require of manufacturers?",
    "What does Health Canada require for device licensing?",
])
def test_named_non_us_frameworks_are_out_of_scope(question):
    assert analysis(question).scope_status == SCOPE_OUT


@pytest.mark.parametrize("question", [
    # benchmark-style regulatory questions with no "regulatory" word: must NOT be declined
    "What must an importer evaluate about a foreign supplier's performance?",
    "What are the responsibilities of the quality control unit?",
    "What performance standard applies to laser products?",
    "How must net quantity of contents be declared?",
    # a U.S. reference keeps a foreign word in scope
    "How does the FDA treat devices already CE marked in the European Union?",
])
def test_regulatory_questions_without_obvious_vocabulary_stay_in_scope(question):
    assert analysis(question).scope_status == SCOPE_IN


def test_general_knowledge_is_answered_under_review_not_refused_by_rule():
    a = analysis("What is the boiling point of water in Celsius?")
    assert a.scope_status == SCOPE_IN and "no_regulatory_vocabulary" in a.signals


def test_mixed_questions_isolate_the_unrelated_part():
    a = analysis("What does 21 CFR 820.10(b)(3) require, and what is the population of Paris?")
    assert a.scope_status == SCOPE_MIXED and a.out_of_scope_part == "what is the population of Paris"
    # a follow-up that refers back is part of the regulatory question
    b = analysis("How must advisory notices be handled under Part 820, and which section establishes that requirement?")
    assert b.scope_status == SCOPE_IN


# ------------------------------------------------------------ profiles and prompt


def test_each_type_has_a_profile_and_every_profile_keeps_the_shared_habits():
    for kind in ("requirement", "definition", "explanation", "procedure", "applicability", "exception_condition",
                 "scenario", "citation_lookup", "comparison"):
        text = profile_for(kind)
        assert text.endswith(PROFILE_TAIL)
        assert "\n" not in text and not re.search(r"^\s*[-*]\s", text, re.M)  # prose only: nothing list-like to copy


def test_yes_no_profile_forbids_agreeing_with_an_unsupported_premise():
    text = profile_for("applicability", yes_no=True)
    assert text.startswith('Begin with Yes, No, or "The sources do not say"')
    assert "instead of agreeing with it" in text


def test_weak_evidence_and_mixed_caveats_come_first():
    text = profile_for("explanation", weak_evidence=True, out_of_scope_part="what is the population of Paris")
    assert text.startswith(WEAK_EVIDENCE) and '"what is the population of Paris"' in text


def test_structured_prompt_carries_the_profile_and_keeps_grounding_rules():
    p, g = pipe(SYNTHESIS)
    p.answer("What does 21 CFR 820.10(b)(2) require?")
    system, user = g.calls[0]["messages"][0]["content"], g.calls[0]["messages"][1]["content"]
    assert system == GROUNDED_SYSTEM_PROMPT                                   # grounding rules untouched
    assert "ANSWER FORMAT: " + PROFILES["requirement"] in user
    assert user.count("ANSWER FORMAT:") == 1


def test_the_default_prompt_is_byte_identical_without_stage3():
    p_adaptive, ga = pipe(SYNTHESIS, mode="adaptive")
    p_adaptive.answer("What does 21 CFR 820.10(b)(2) require?")
    p_struct, gs = pipe(SYNTHESIS, mode="structured")
    p_struct.answer("What does 21 CFR 820.10(b)(2) require?")
    a, s = ga.calls[0]["messages"][1]["content"], gs.calls[0]["messages"][1]["content"]
    assert "Begin with one sentence" not in a
    # the only difference is the profile inserted at the start of the ANSWER FORMAT line
    assert s.replace(PROFILES["requirement"] + " " + PROFILE_TAIL + " ", "") == a


def test_prompt_is_deterministic_and_has_no_benchmark_leakage():
    p1, g1 = pipe(SYNTHESIS)
    p2, g2 = pipe(SYNTHESIS)
    q = "Does 21 CFR 820.10(b)(2) require annual traceability audits?"
    p1.answer(q)
    p2.answer(q)
    user = g1.calls[0]["messages"][1]["content"]
    assert user == g2.calls[0]["messages"][1]["content"]
    for leak in ("expected_terms", "expected_behavior", "key_answer_elements", "S5-", "answer key"):
        assert leak not in user


def test_prompt_growth_is_small():
    p_a, ga = pipe(SYNTHESIS, mode="adaptive")
    p_s, gs = pipe(SYNTHESIS, mode="structured")
    q = "Does 21 CFR 820.10(b)(2) require annual traceability audits?"
    p_a.answer(q)
    p_s.answer(q)
    grow = len(gs.calls[0]["messages"][1]["content"]) - len(ga.calls[0]["messages"][1]["content"])
    assert 0 < grow < 500  # a few sentences, not a template


# ------------------------------------------------------------ out of scope and mixed


def test_named_foreign_framework_is_declined_without_the_model():
    p, g = pipe("anything")
    out = p.answer("What does the EU Medical Device Regulation require for UDI?")
    assert not out.model_called and g.calls == []
    assert out.metadata["coverage_action"] == "refuse_out_of_scope"
    assert out.answer == out_of_scope_answer(analysis("What does the EU Medical Device Regulation require for UDI?").scope_reason)
    assert out.citations == []


def test_adaptive_mode_without_the_answerability_gate_does_not_decline_by_scope():
    # Stage 3 behaviour, kept reachable with answerability="off"; with the Stage 5 gate (default)
    # scope is decided in every mode (tests/test_answerability.py)
    g = ScriptedGenerator("anything")
    p = RagPipeline(LexicalRetriever(TableDatabase(TABLE)), g, RagConfig(answer_shape="adaptive", answerability="off"))
    out = p.answer("What does the EU Medical Device Regulation require for UDI?")
    assert out.metadata["coverage_action"] != "refuse_out_of_scope"


def test_mixed_question_answers_the_regulatory_part_and_states_the_rest():
    answer = "21 CFR 820.10(b)(3) requires notifying FDA of complaints under part 803 [1]."
    p, g = pipe(answer)
    out = p.answer("What does 21 CFR 820.10(b)(3) require, and what is the population of Paris?")
    assert out.model_called and out.answer.endswith(mixed_scope_note("what is the population of Paris"))
    assert "the sources do not cover" in g.calls[0]["messages"][1]["content"]


# ------------------------------------------------------------ normalisation


def test_leading_markers_move_onto_the_sentence_they_label():
    text, info = normalize_answer("[1] 21 CFR 820.10(c) requires design controls. It applies to class II.\n[2] 820.10(d) requires traceability.", "q?")
    assert text == "21 CFR 820.10(c) requires design controls [1]. It applies to class II.\n820.10(d) requires traceability [2]."
    assert info["markers_moved"] == 2 and info["changed"]


def test_normalisation_changes_no_words():
    raw = "- [3] Advisory notices shall be handled under part 806.\n[4][5] Manufacturers must notify FDA"
    text, _ = normalize_answer(raw, "q?")
    bare = lambda t: re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s+", " ", re.sub(r"\[\d+\]", "", t))).strip()
    assert bare(text) == bare(raw)  # only markers moved: the words and punctuation are the same


def test_a_restated_question_is_removed_only_when_an_answer_follows():
    q = "What is the capital of France?"
    assert normalize_answer("What is the capital of France? [1] The sources do not say.", q)[0] == "The sources do not say."
    assert normalize_answer("What is the capital of France? [1]", q)[0] == "What is the capital of France? [1]"  # nothing else to keep


def test_normalisation_happens_before_validation_and_fixes_attribution():
    # line-leading markers, as the model writes them: the validator would attach [2] to the (c) sentence
    answer = "[1] 21 CFR 820.10(c) requires design and development controls for class II devices.\n[2] 21 CFR 820.10(d) applies to devices that support or sustain life, as appropriate."
    p, _ = pipe(answer)
    out = p.answer("Compare 21 CFR 820.10(c) and 820.10(d).")
    assert out.answer.startswith("21 CFR 820.10(c) requires design and development controls for class II devices [1].")
    assert out.metadata["normalization"]["markers_moved"] == 2
    p2, _ = pipe(answer, mode="adaptive")
    assert p2.answer("Compare 21 CFR 820.10(c) and 820.10(d).").answer == answer  # default: text untouched


# ------------------------------------------------------------ validation, repair, citations


def test_validation_and_repair_still_run_in_structured_mode():
    bad = "21 CFR 820.10(b)(2) requires traceability procedures under part 821, if applicable [9]."
    p, _ = pipe(bad)
    out = p.answer("What does 21 CFR 820.10(b)(2) require?")
    v = out.metadata["validation"]
    assert v["initial"]["status"] == "citation_invalid" and v["repair"] == "invalid_citations_removed"
    assert v["final"]["status"] == "valid" and "[9]" not in out.answer


def test_citations_stay_traceable_and_none_are_invented():
    good = "21 CFR 820.10(b)(2) requires documented traceability procedures under part 821, if applicable [1]."
    p, _ = pipe(good)
    out = p.answer("What does 21 CFR 820.10(b)(2) require?")
    assert [c.label for c in out.citations] == ["21 CFR 820.10(b)(2)"]
    assert all(c.index in {s.index for s in out.package.sources} for c in out.citations)


# ------------------------------------------------------------ quality flags


def _flags(answer, question, mode="structured", response=None):
    p, _ = pipe(response or answer, mode=mode)
    out = p.answer(question)
    return {f["flag"] for f in out.metadata["quality_flags"] or []}, out


def test_premise_affirmed_without_support_is_caught_and_not_shown():
    # Stage 7: validation now catches the affirmed premise (unsupported_premise), so it never reaches the
    # answer; the identified provision is quoted with a statement that it does not state the premise
    flags, out = _flags("21 CFR 820.10(b)(2) requires annual traceability audits [1].",
                        "Does 21 CFR 820.10(b)(2) require annual traceability audits?")
    assert "unsupported_premise" in out.metadata["validation"]["initial"]["codes"]
    assert "requires annual traceability audits" not in out.answer and "does not state" in out.answer
    assert out.metadata["fallback_type"] == "premise_not_established"


def test_a_proper_no_answer_is_not_flagged():
    flags, _ = _flags("No. The sources do not say that 21 CFR 820.10(b)(2) requires annual audits; it requires traceability "
                      "procedures under part 821, if applicable [1].", "Does 21 CFR 820.10(b)(2) require annual traceability audits?")
    assert not flags & {"unsupported_claim", "missing_direct_answer"}


def test_missing_citation_and_weak_structure_are_flagged():
    long = ("21 CFR 820.10(b)(2) concerns traceability. Manufacturers keep written procedures describing how "
            "products move through production. Each lot receives an identifier that links materials to finished goods. "
            "Records show which suppliers provided components and when those components arrived. Staff review the "
            "records during audits and correct gaps promptly. Distribution logs name the consignees that received "
            "each shipment. When a problem appears, investigators follow the trail backward to the source. Training "
            "ensures employees understand why accurate entries matter. Management reviews the system periodically "
            "and approves changes before use. Electronic systems must protect entries from alteration. Paper forms "
            "are signed and dated by the responsible person. Retention periods depend on the expected life of the "
            "product. Retrieval must be possible within a reasonable time when regulators ask. Complaints are "
            "linked to affected lots so recalls can be scoped. Outsourced activities remain the manufacturer's "
            "responsibility under the quality agreement. Periodic checks confirm that labels match the records "
            "kept for every batch shipped.")
    flags, _ = _flags(long, "What does 21 CFR 820.10(b)(2) require?")
    assert {"missing_citation", "weak_structure"} <= flags


def test_quality_flags_exist_only_in_structured_mode():
    _, out = _flags("x [1].", "What does 21 CFR 820.10(b)(2) require?", mode="adaptive")
    assert out.metadata["quality_flags"] is None and out.metadata["question_analysis"] is None


def test_flags_are_signals_with_evidence():
    flags = quality_flags("A [1].", None, analysis("What does 21 CFR 820.10(b)(2) require?"),
                          {"codes": ["qualifier_dropped"]}, coverage_action="answer", model_called=True, question="q")
    assert flags == [{"flag": "missing_condition", "detail": "validator: qualifier_dropped"}]


def test_weak_evidence_prompt_and_disclosure_flag():
    db = TableDatabase(TABLE, or_ecfr=[row("801.1", "", 0, text="Labeling of devices.", rank=0.1)],
                       or_guidance=[guidance_row(0, rank=0.1)])
    p, g = pipe("Water boils when it is heated enough [1].", db=db)
    out = p.answer("What is the boiling point of water in Celsius?")
    if out.model_called:
        assert out.package.weak_evidence == all(s.tier == "discovery" for s in out.package.sources)
        if out.package.weak_evidence:
            assert WEAK_EVIDENCE in g.calls[0]["messages"][1]["content"]
            assert "uncertainty_not_disclosed" in {f["flag"] for f in out.metadata["quality_flags"]}


# ------------------------------------------------------------ streaming in structured mode


@pytest.mark.usefixtures("delivered_as_drafted")   # the final is the validated draft itself
def test_structured_streaming_draft_final_and_equivalence():
    from tests.test_streaming import StreamingGenerator

    answer = "[1] 21 CFR 820.10(b)(2) requires documented traceability procedures under part 821, if applicable."
    q = "What does 21 CFR 820.10(b)(2) require?"
    events = []
    streamed = RagPipeline(LexicalRetriever(TableDatabase(TABLE)), StreamingGenerator(answer),
                           RagConfig(answer_shape="structured")).answer(q, on_event=lambda k, d: events.append((k, d)))
    plain = RagPipeline(LexicalRetriever(TableDatabase(TABLE)), ScriptedGenerator(answer),
                        RagConfig(answer_shape="structured")).answer(q)
    kinds = [k for k, _ in events]
    draft = "".join(d["text"] for k, d in events if k == "delta")
    assert kinds.count("delta") > 3 and kinds[-1] == "generation_finished"   # draft streamed, then generation ends
    assert draft == answer                                                   # the draft is the raw text ...
    assert streamed.answer != draft and streamed.answer.endswith("if applicable [1].")  # ... the final is normalised + validated
    assert streamed.answer == plain.answer and streamed.metadata["validation"] == plain.metadata["validation"]
    assert [c.to_dict() for c in streamed.citations] == [c.to_dict() for c in plain.citations]


def test_structured_streaming_errors_still_propagate():
    from tests.test_streaming import StreamingGenerator

    with pytest.raises(RuntimeError, match="crashed"):
        RagPipeline(LexicalRetriever(TableDatabase(TABLE)), StreamingGenerator("a b c d e", fail_after=2),
                    RagConfig(answer_shape="structured")).answer("What does 21 CFR 820.10(b)(2) require?", on_event=lambda *a: None)
