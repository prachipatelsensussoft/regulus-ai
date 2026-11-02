"""Regression tests for the adaptive answer-length policy.

The failure this policy exists to eliminate is an answer cut short by an
artificial limit. The failure it must not introduce is padding. So these
tests check two different things and never confuse them:

* the **allowance** scales with what the question actually needs, and is
  never the thing that ends a complete answer;
* the **instruction** asks for the shortest complete answer and nowhere
  prescribes a length, so brevity can never cost a condition or an exception.

They run on the in-memory table used by the Stage 2/3 tests: no PostgreSQL
and no model.
"""

from __future__ import annotations

import re

import pytest

from regulus.config import AnswerLengthConfig, ConfigError, load_settings
from regulus.inference.generate import GenerationResult
from regulus.rag import RagConfig, RagPipeline
from regulus.rag.generation import (
    GROUNDED_SYSTEM_PROMPT,
    POLICY_ADAPTIVE,
    POLICY_FIXED,
    GenerationBudget,
    build_grounded_messages,
    render_package,
)
from regulus.rag.validation import STATUS_INCOMPLETE, TRUNCATED, validate_answer
from regulus.retrieval import LexicalRetriever
from regulus.retrieval.evidence import TIER_EXACT
from tests.test_answer_validation import ScriptedGenerator
from tests.test_generation_contract import package_for
from tests.test_retrieval_plan import TableDatabase, row

SIMPLE = "What does 21 CFR 820.10(b)(2) require?"
MULTI = "What do 820.10(b)(2) and 820.10(b)(4) require?"
BROAD = "What does 21 CFR 820.10 require?"
COMPREHENSIVE = "What do 21 CFR 820.10(b)(1) through 820.10(b)(4) require?"
GENERAL = "What is a 510(k)?"

# the legacy policy this change replaced, kept here as the thing that must
# never come back: 64 base + 64 per provision, capped at 448
LEGACY = GenerationBudget(base_tokens=64, per_unit_tokens=64, range_bonus_tokens=32,
                          general_tokens=160, max_units=6, hard_max_tokens=448,
                          min_tokens=1, evidence_multiple=0.01, policy=POLICY_FIXED)


def exact_chars(pkg) -> int:
    return sum(len(s.text) for s in pkg.sources if s.tier == TIER_EXACT)


# ------------------------------------------------------- A: short answers


def test_a_simple_question_gets_a_modest_allowance_and_a_conciseness_instruction():
    pkg = package_for(SIMPLE)
    b = pkg.budget
    # room to answer, nowhere near the ceiling: the allowance is not a target
    assert b.min_tokens <= pkg.max_new_tokens < b.hard_max_tokens
    text = render_package(pkg)
    assert "ANSWER FORMAT:" in text
    assert "Use the minimum amount of text a complete and accurate answer needs" in GROUNDED_SYSTEM_PROMPT


def test_a_simple_question_is_not_given_more_room_than_a_comprehensive_one():
    simple = package_for(SIMPLE).max_new_tokens
    comprehensive = package_for(COMPREHENSIVE).max_new_tokens
    assert simple < comprehensive


def test_the_instruction_never_prescribes_a_length():
    """A sentence or word count is a preferred length in disguise, and the
    model obeys it by dropping qualifiers. Nothing may state one."""
    # Spelled-out counts ("two to four plain sentences") and explicit numeric
    # limits ("at most 150 words"). Written narrowly on purpose: a loose
    # pattern matches the digits inside the citation example 820.10(b)(2).
    counted = re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|a few|several)\b"
        r"(?:\s+(?:to|or)\s+\w+)?\s+(?:\w+\s+)?(sentences|words|paragraphs)\b"
        r"|\b(at most|no more than|up to|maximum of|limit of)\s+\d+\s+(sentences|words|tokens|paragraphs)\b",
        re.IGNORECASE)
    assert not counted.search(GROUNDED_SYSTEM_PROMPT)
    for question in (SIMPLE, MULTI, BROAD, COMPREHENSIVE, GENERAL):
        db = TableDatabase(full_text=[row("801.1", "", 0, rank=0.5)]) if question == GENERAL else None
        text = render_package(package_for(question, db))
        fmt = text.split("ANSWER FORMAT:")[1]
        assert not counted.search(fmt), f"{question}: {fmt[:120]}"


def test_the_instruction_forbids_dropping_qualifiers_to_stay_short():
    for needle in ("condition", "exception", "applicability or scope limit", "effective date",
                   "cross-reference", "required action", "distinction between subsections",
                   "never stop before the answer is complete"):
        assert needle in GROUNDED_SYSTEM_PROMPT, needle


# --------------------------------------------- B: naturally longer answers


def test_b_allowance_grows_with_the_number_of_requested_provisions():
    sizes = [package_for(q).max_new_tokens for q in (SIMPLE, MULTI, COMPREHENSIVE)]
    assert sizes == sorted(sizes), sizes
    assert sizes[-1] > sizes[0]


def test_b_allowance_grows_with_the_amount_of_evidence_even_for_one_provision():
    """Two questions, one provision each, different amounts of authoritative
    text: the denser one gets more room without anyone choosing a number."""
    budget = GenerationBudget(min_tokens=1)          # take the floor out of the way
    short_text = TableDatabase()
    long_row = [dict(r, body_text=r["body_text"] * 60) if r.get("paragraph_path") == "(b)(2)" else r
                for r in __import__("tests.test_retrieval_plan", fromlist=["TABLE"]).TABLE]
    dense = package_for(SIMPLE, TableDatabase(long_row), budget=budget)
    sparse = package_for(SIMPLE, short_text, budget=budget)
    assert exact_chars(dense) > exact_chars(sparse)
    assert dense.max_new_tokens > sparse.max_new_tokens


def test_b_the_legacy_fixed_policy_would_have_been_tighter():
    """The regression this change fixes, stated as an assertion."""
    for question in (SIMPLE, MULTI, COMPREHENSIVE):
        adaptive = package_for(question).max_new_tokens
        legacy = package_for(question, budget=LEGACY).max_new_tokens
        assert adaptive > legacy, question


# ------------------------------------------------ C: broad question, concise


def test_c_a_broad_section_question_is_not_forced_into_a_subsection_dump():
    pkg = package_for(BROAD)
    fmt = render_package(pkg).split("ANSWER FORMAT:")[1]
    # the section is answered as one provision, not as a per-provision list of children
    assert "leave out" in fmt or "no more" in fmt or "do not pad" in fmt
    assert pkg.max_new_tokens <= pkg.budget.hard_max_tokens


def test_c_units_are_capped_so_a_wide_section_cannot_run_away():
    pkg = package_for(BROAD)
    b = pkg.budget
    structural = b.base_tokens + b.per_unit_tokens * b.max_units
    assert pkg.max_new_tokens <= max(structural, b.evidence_tokens(exact_chars(pkg)), b.min_tokens)
    assert pkg.max_new_tokens <= b.hard_max_tokens


# ------------------------------------------ D: explicit comprehensive question


def test_d_a_comprehensive_question_gets_room_for_every_subsection():
    pkg = package_for(COMPREHENSIVE)
    exact = [s for s in pkg.sources if s.tier == TIER_EXACT]
    assert len(exact) == 4
    # enough room to restate all four provisions, which the retired policy
    # (64 + 64 * 4 = 320) did not reliably give
    assert pkg.max_new_tokens >= pkg.budget.evidence_tokens(exact_chars(pkg))
    assert pkg.max_new_tokens > package_for(SIMPLE).max_new_tokens


def test_d_every_requested_provision_keeps_its_own_block_in_the_format():
    fmt = render_package(package_for(COMPREHENSIVE)).split("ANSWER FORMAT:")[1]
    assert "for each requested provision, in the order listed" in fmt
    assert "citation label on its own line" in fmt
    assert "do not compress one that carries conditions or exceptions" in fmt


# ------------------------------------------------------ E: generation ceiling


def test_e_truncation_is_detected_from_the_end_of_turn_token():
    pkg = package_for(MULTI)
    unfinished = GenerationResult("p", "21 CFR 820.10(b)(2): traceability [1]. 21 CFR 820.10(b)(4): advis",
                                  500, pkg.max_new_tokens, 1.0, finished=False)
    report = validate_answer(unfinished.response, pkg, unfinished)
    assert report.truncated and TRUNCATED in report.codes()
    assert report.status == STATUS_INCOMPLETE


def test_e_a_finished_answer_at_the_same_length_is_not_called_truncated():
    """The distinction is the end-of-turn token, not the token count."""
    pkg = package_for(MULTI)
    finished = GenerationResult("p", "21 CFR 820.10(b)(2): traceability [1]. 21 CFR 820.10(b)(4): advisory notices [2].",
                                500, pkg.max_new_tokens, 1.0, finished=True)
    report = validate_answer(finished.response, pkg, finished)
    assert not report.truncated


def test_e_the_report_names_which_limit_was_reached():
    pkg = package_for(MULTI)
    at_allowance = GenerationResult("p", "partial [1]", 500, pkg.max_new_tokens, 1.0, finished=False)
    message = " ".join(p.detail for p in validate_answer(at_allowance.response, pkg, at_allowance).problems
                       if p.code == TRUNCATED)
    assert "this question's allowance" in message

    tight = package_for(MULTI, budget=GenerationBudget(hard_max_tokens=64))
    hit_ceiling = GenerationResult("p", "partial [1]", 500, tight.max_new_tokens, 1.0, finished=False)
    message = " ".join(p.detail for p in validate_answer(hit_ceiling.response, tight, hit_ceiling).problems
                       if p.code == TRUNCATED)
    assert "the hard safety ceiling" in message


def test_e_a_truncated_answer_is_never_presented_as_complete():
    g = ScriptedGenerator("21 CFR 820.10(b)(2): traceability [1]. 21 CFR 820.10(b)(4): advis", finished=False)
    out = RagPipeline(LexicalRetriever(TableDatabase()), g).answer(MULTI)
    # since the 16 Oct severity fix a cut draft is not delivered with a note: the requested provisions' verified text is
    # shown instead (P2, 43 withheld drafts: salvaging cut drafts gained no PASS and delivered wrong answers). Either way
    # the cut answer is never presented as complete
    assert ("Incomplete: generation stopped at the token limit" in out.answer
            or out.metadata.get("fallback_type") == "verbatim_requested_provision")
    assert "advis" not in out.answer.replace("advisory", "")                 # the cut fragment never reaches the page
    assert out.validation.truncated and out.validation.status == STATUS_INCOMPLETE
    assert "truncation_noted" in (out.metadata["validation"]["repair"] or "")


def test_e_truncation_handling_does_not_duplicate_content():
    """The repair completes a cut answer from the exact sources; it must never
    append a second copy of what the model already said."""
    g = ScriptedGenerator("21 CFR 820.10(b)(2): traceability [1]. 21 CFR 820.10(b)(4): advis", finished=False)
    out = RagPipeline(LexicalRetriever(TableDatabase()), g).answer(MULTI)
    assert out.answer.count("21 CFR 820.10(b)(2): traceability [1].") <= 1
    assert out.answer.count("Incomplete: generation stopped") <= 1
    assert out.answer.count("820.10(b)(2) body text") <= 1
    paragraphs = [p.strip() for p in out.answer.split("\n") if p.strip()]
    assert len(paragraphs) == len(set(paragraphs)), out.answer


def test_e_the_ceiling_is_generous_relative_to_any_allowance_the_policy_asks_for():
    defaults = GenerationBudget()
    assert defaults.hard_max_tokens >= 1024
    widest = max(package_for(q).max_new_tokens for q in (SIMPLE, MULTI, BROAD, COMPREHENSIVE))
    assert widest < defaults.hard_max_tokens, "the ceiling must not be the thing that sets answer length"


# --------------------------------------------------------- F: citations


def test_f_a_longer_answer_keeps_valid_citations():
    pkg = package_for(COMPREHENSIVE)
    long_answer = "\n".join(
        f"21 CFR 820.10(b)({i}): the provision requires the stated controls [{i}]." for i in range(1, 5)
    )
    report = validate_answer(long_answer, pkg, GenerationResult("p", long_answer, 500, 300, 1.0, finished=True))
    assert report.ok, report.codes()
    assert not report.invalid_citations
    assert set(report.addressed) == {"r1", "r2", "r3", "r4"}


def test_f_the_citation_contract_is_unchanged_by_the_length_policy():
    for needle in ("End each substantive statement with the number of the source it comes from",
                   "Cite only sources you actually used"):
        assert needle in GROUNDED_SYSTEM_PROMPT, needle
    for question in (SIMPLE, MULTI, COMPREHENSIVE):
        fmt = render_package(package_for(question)).split("ANSWER FORMAT:")[1]
        assert "source number in brackets" in fmt


def test_f_citations_survive_the_whole_pipeline_at_length():
    answer = "\n".join(
        f"21 CFR 820.10(b)({i}): the provision requires the stated controls [{i}]." for i in range(1, 5)
    )
    g = ScriptedGenerator(answer, tokens=300)
    out = RagPipeline(LexicalRetriever(TableDatabase()), g).answer(COMPREHENSIVE)
    assert out.validation.ok
    assert [c.index for c in out.citations][:4] == [1, 2, 3, 4]
    assert all(c.label for c in out.citations)


# ----------------------------------------------------------- configuration


def test_the_policy_is_configurable_in_one_place():
    settings = load_settings(use_dotenv=False)
    assert settings.answer_length.policy == POLICY_ADAPTIVE
    budget = GenerationBudget.from_config(settings.answer_length)
    assert budget.hard_max_tokens == settings.answer_length.max_new_tokens
    assert budget.policy == POLICY_ADAPTIVE
    assert budget.hard_max_tokens >= 1024, "the shipped ceiling is deliberately generous"


def test_the_configured_ceiling_is_not_the_configured_answer_length():
    settings = load_settings(use_dotenv=False)
    budget = GenerationBudget.from_config(settings.answer_length)
    assert budget.min_tokens < budget.hard_max_tokens
    assert budget.general_tokens < budget.hard_max_tokens
    assert budget.base_tokens + budget.per_unit_tokens < budget.hard_max_tokens


def test_environment_overrides_reach_the_budget(monkeypatch):
    monkeypatch.setenv("REGULUS_MAX_NEW_TOKENS", "1536")
    monkeypatch.setenv("REGULUS_ANSWER_LENGTH_POLICY", "fixed")
    settings = load_settings(use_dotenv=False)
    assert settings.answer_length.max_new_tokens == 1536
    assert GenerationBudget.from_config(settings.answer_length).policy == POLICY_FIXED


def test_a_nonsense_policy_is_rejected_rather_than_ignored():
    with pytest.raises(ConfigError):
        AnswerLengthConfig(policy="short").validate()
    with pytest.raises(ConfigError):
        AnswerLengthConfig(min_tokens=4096, max_new_tokens=512).validate()
    with pytest.raises(ConfigError):
        AnswerLengthConfig(max_new_tokens=0).validate()


def test_the_fixed_policy_reproduces_the_pre_adaptive_behaviour():
    """Kept so the change can be measured against what it replaced."""
    fixed = GenerationBudget(policy=POLICY_FIXED, base_tokens=64, per_unit_tokens=64,
                             general_tokens=160, min_tokens=1)
    assert package_for(SIMPLE, budget=fixed).max_new_tokens == 64 + 64
    assert package_for(MULTI, budget=fixed).max_new_tokens == 64 + 64 * 2


def test_the_budget_is_reported_so_a_short_answer_can_be_explained():
    pkg = package_for(MULTI)
    d = pkg.to_dict()
    assert d["max_new_tokens"] == pkg.max_new_tokens
    assert d["budget"]["policy"] == POLICY_ADAPTIVE
    assert d["budget"]["hard_max_tokens"] == pkg.budget.hard_max_tokens
    assert "min_tokens" in d["budget"] and "evidence_multiple" in d["budget"]


# ------------------------------------------------ AL1: the allowance keeps pace with the evidence (29 Oct)


def test_al1_a_forty_paragraph_answer_is_no_longer_capped_at_ten_units(monkeypatch):
    """300Q S-024: 8 sections in 40 paragraph blocks were allowed 1,696 tokens (10 units) and cut mid-sentence."""
    from types import SimpleNamespace

    from regulus.rag import validator_switches as VS
    from regulus.rag.pipeline import answer_budget

    plan, cov = SimpleNamespace(kind="sections"), SimpleNamespace(explicit=True, satisfied_ids=tuple(f"r{i}" for i in range(8)))
    base = GenerationBudget()
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"AL1"})
    assert answer_budget(base) is base
    assert base.tokens_for(plan, cov, 40) == 1696
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"AL1"})
    wide = answer_budget(base)
    assert wide.tokens_for(plan, cov, 40) == 96 + 160 * 18
    assert wide.tokens_for(plan, cov, 3) == base.tokens_for(plan, cov, 3)   # a short answer's allowance is unchanged
    assert wide.hard_max_tokens == 3072
