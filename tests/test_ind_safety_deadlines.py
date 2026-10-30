"""IND safety-reporting deadlines: 21 CFR 312.32(c)(1) is 15 calendar days, (c)(2) is 7.

The reported failure was an answer that confused the two: it put the 7-calendar-day deadline on
paragraph (c)(1)(i) and cited 312.32(a), the Definitions paragraph. Four separate defects were on
the path of that one answer, and this file holds one group of tests per defect plus the six
answer scenarios the fix has to decide correctly.

    selection      the paragraphs of a requested section competed for the 9,000-character hard
                   ceiling in DOCUMENT order, so the six (a) definition paragraphs took 4,416 of
                   it and (c)(2) - 298 characters, and the whole answer to the question - was
                   dropped. It is now a relevance-ordered competition, and only that: the sources
                   are still presented in document order, and nothing changes while everything
                   fits or while the question has no terms in common with the section.
    answerability  a question asking for a "deadline" was classified as asking for a calendar
                   DATE, which "7 calendar days" is not, so a correct answer was withheld.
    evaluator      a claim that puts (c)(1)'s figure on (c)(2)'s subject matter aligns to (c)(1),
                   agrees with it about 15 days, and was abstained on. The content gate already
                   names (c)(2) as the source holding the claim's distinguishing terms, so the
                   claim is now compared with it and the numeric conflict decides it.
    fabrication    the figure check ignored numbers of fewer than two digits, so an invented
                   "7 calendar days" was not reported when no supplied source stated a 7-day
                   period.

Regulatory text: VERBATIM 21 CFR 312.32 from the read-only eCFR snapshot. Answers: CONSTRUCTED
for testing - they are not regulatory text and several are deliberately wrong.
"""

from __future__ import annotations

import pytest

from regulus.accuracy import answer as AN
from regulus.rag import ContextPolicy, select_evidence
from regulus.rag.evidence_values import evidence_has_value, requested_value_kind, selection_value_kind
from regulus.rag.selection import (DROP_HARD_CEILING_CHARS, _match_keys, _referenced_paths, _relevance,
                                   _required_order, is_required)
from regulus.planning import plan_query
from regulus.retrieval import LexicalRetriever
from tests.test_accuracy_claim_extraction import judge
from tests.test_retrieval_plan import TableDatabase, row

# fixtures sized to overflow the CPU-era evidence budget (tests/conftest.py)
pytestmark = pytest.mark.usefixtures("cpu_era_budget")

# ------------------------------------------------------------------ the provision, verbatim

C1_TEXT = ("(c)(1) IND safety reports. The sponsor must notify FDA and all participating investigators (i.e., all "
           "investigators to whom the sponsor is providing drug under its INDs or under any investigator's IND) in an "
           "IND safety report of potential serious risks, from clinical trials or any other source, as soon as "
           "possible, but in no case later than 15 calendar days after the sponsor determines that the information "
           "qualifies for reporting under paragraph (c)(1)(i), (c)(1)(ii), (c)(1)(iii), or (c)(1)(iv) of this section. "
           "In each IND safety report, the sponsor must identify all IND safety reports previously submitted to FDA "
           "concerning a similar suspected adverse reaction, and must analyze the significance of the suspected "
           "adverse reaction in light of previous, similar reports or any other relevant information.")
C2_TEXT = ("(2) Unexpected fatal or life-threatening suspected adverse reaction reports. The sponsor must also notify "
           "FDA of any unexpected fatal or life-threatening suspected adverse reaction as soon as possible but in no "
           "case later than 7 calendar days after the sponsor's initial receipt of the information.")
DEFINITION_TEXT = ("Life-threatening adverse event or life-threatening suspected adverse reaction. An adverse event "
                   "or suspected adverse reaction is considered “life-threatening” if, in the view of either "
                   "the investigator or sponsor, its occurrence places the patient or subject at immediate risk of "
                   "death. It does not include an adverse event or suspected adverse reaction that, had it occurred "
                   "in a more severe form, might have caused death.")

C1 = ("21 CFR 312.32(c)(1)", C1_TEXT)
C2 = ("21 CFR 312.32(c)(2)", C2_TEXT)
DEFINITION = ("21 CFR 312.32(a)", DEFINITION_TEXT)

DEADLINE_QUESTION = ("What is the difference between the 15-calendar-day and 7-calendar-day IND safety reporting "
                     "deadlines in 21 CFR 312.32?")

# A section whose paragraphs cannot all fit inside hard_max_total_chars, shaped like 21 CFR 312.32
# itself: long paragraphs the question did not ask about come first in document order, and the
# short paragraph that answers it - (c)(2), 298 characters - comes last. In the real section the
# six (a) definitions took 4,416 of the 9,000 characters before any requirement text was reached.
def _pad(lead, size):
    filler = "An adverse event or suspected adverse reaction is considered as described in this paragraph. "
    return (lead + " " + filler * (size // len(filler) + 1))[:size]


def _table():
    return [
        row("312.32", "(a)", 0, text=_pad("(a) Definitions. The following definitions of terms apply to this section:"
                                          f" {DEFINITION_TEXT}", 3800)),
        row("312.32", "(b)", 1, text=_pad("(b) Review of safety information. The sponsor must promptly review all "
                                          "information relevant to the safety of the drug.", 3800)),
        row("312.32", "(c)(1)", 2, text=C1_TEXT),
        row("312.32", "(c)(1)(i)", 3, text=_pad("(i) Serious and unexpected suspected adverse reaction. The sponsor "
                                                "must report any suspected adverse reaction.", 1500)),
        row("312.32", "(c)(2)", 4, text=C2_TEXT),
    ]


def _registry(question):
    return LexicalRetriever(TableDatabase(_table())).run_plan(plan_query(question)).registry


def _labels(selection):
    return [s.evidence.citation for s in selection.selected]


# ------------------------------------------------------------------ selection


def test_the_paragraph_that_answers_the_question_is_not_dropped_for_the_definitions():
    """The defect: (c)(2) states the 7-calendar-day deadline and was dropped as overflow."""
    registry = _registry("What does 21 CFR 312.32 require?")
    before = select_evidence(registry)
    after = select_evidence(registry, question=DEADLINE_QUESTION)
    assert before.overflow, "the fixture must not fit inside the hard ceiling, or it tests nothing"
    assert "21 CFR 312.32(c)(2)" not in _labels(before)
    assert "21 CFR 312.32(c)(2)" in _labels(after)
    assert "21 CFR 312.32(c)(1)" in _labels(after), "the 15-day rule must survive alongside the 7-day rule"


def test_the_dropped_definitions_are_recorded_not_lost_silently():
    after = select_evidence(_registry("What does 21 CFR 312.32 require?"), question=DEADLINE_QUESTION)
    dropped = {d.evidence.citation for d in after.dropped if d.required}
    assert dropped, "paragraphs that did not fit must still be reported"
    assert all(d.reason == DROP_HARD_CEILING_CHARS for d in after.dropped if d.required)


def test_selected_sources_stay_in_document_order():
    """Relevance decides what is kept, never the order a reader or the model sees."""
    after = select_evidence(_registry("What does 21 CFR 312.32 require?"), question=DEADLINE_QUESTION)
    ordinals = [s.evidence.result.metadata["ordinal"] for s in after.selected]
    assert ordinals == sorted(ordinals)


def test_the_ceilings_are_still_respected():
    policy = ContextPolicy()
    after = select_evidence(_registry("What does 21 CFR 312.32 require?"), question=DEADLINE_QUESTION)
    assert after.total_chars <= policy.hard_max_total_chars
    assert len(after.selected) <= policy.hard_max_results


@pytest.mark.parametrize("question", [
    None,                               # no question: the previous behaviour
    "Summarize 21 CFR 312.32.",         # no salient terms beyond the section number
    "What do zebra quarantine kennels require under 21 CFR 312.32?",   # terms, but none in the section
])
def test_without_a_relevant_question_the_selection_is_byte_identical(question):
    registry = _registry("What does 21 CFR 312.32 require?")
    before = select_evidence(registry)
    after = select_evidence(registry, question=question)
    assert _labels(after) == _labels(before)
    assert after.total_chars == before.total_chars


def test_evidence_that_all_fits_is_unchanged_by_the_question():
    small = [row("312.32", "(c)(1)", 0, text=C1_TEXT), row("312.32", "(c)(2)", 1, text=C2_TEXT)]
    registry = LexicalRetriever(TableDatabase(small)).run_plan(plan_query("What does 21 CFR 312.32 require?")).registry
    before = select_evidence(registry)
    after = select_evidence(registry, question=DEADLINE_QUESTION)
    assert not before.overflow and _labels(after) == _labels(before) and after.total_chars == before.total_chars


# ------------------------------------------------------------------ answerability


def test_a_deadline_may_be_a_period_not_only_a_calendar_date():
    assert requested_value_kind("What is the IND safety reporting deadline under 21 CFR 312.32?") == "deadline"
    assert evidence_has_value("deadline", [C2_TEXT]) is True
    assert evidence_has_value("deadline", [DEFINITION_TEXT]) is False


@pytest.mark.parametrize("question", [
    "By what date must a manufacturer comply with the amended Part 820?",
    "What is the compliance date for the Quality Management System Regulation?",
    "What is the effective date of 21 CFR 820.35?",
])
def test_a_question_that_asks_for_a_calendar_date_still_asks_for_one(question):
    assert requested_value_kind(question) == "date"
    assert evidence_has_value("date", [C2_TEXT]) is False, "7 calendar days is not a compliance date"


# ------------------------------------------------------------------ the six answer scenarios

QUESTION = "What are the IND safety reporting deadlines in 21 CFR 312.32?"

CORRECT_15 = ("The sponsor must submit an IND safety report no later than 15 calendar days after determining the "
              "information qualifies for reporting. [1]")
CORRECT_7 = ("The sponsor must notify FDA of any unexpected fatal or life-threatening suspected adverse reaction no "
             "later than 7 calendar days after initial receipt of the information. [2]")
REVERSED_7 = ("The sponsor must submit an IND safety report no later than 7 calendar days after determining the "
              "information qualifies for reporting. [1]")
REVERSED_15 = ("The sponsor must notify FDA of any unexpected fatal or life-threatening suspected adverse reaction no "
               "later than 15 calendar days after initial receipt of the information. [2]")


def _finals(assessment):
    return [c["support"] for c in assessment.checked["claims"]]


def _substantive(assessment):
    return next(d["state"] for d in assessment.to_dict()["dimensions"] if d["dimension"] == "substantive_correctness")


def test_both_deadlines_stated_correctly_are_credited():
    """Neither statement may be contradicted and neither deadline may be called unsupported.

    With heading-less harness sources this asserts only that nothing FAILS: without the section
    heading the relevance check sees no "IND safety report" wording in (c)(2). See
    `test_both_deadlines_against_production_shaped_sources_pass` for the production-shaped case."""
    a = judge(QUESTION, f"{CORRECT_15}\n\n{CORRECT_7}", [C1, C2])
    assert AN.CONTRADICTED not in _finals(a)
    assert _substantive(a) != "fail"
    assert AN.SUPPORTED in _finals(a)
    assert all(c["support_basis"] != "operator_conflict" for c in a.checked["claims"])


def test_both_deadlines_against_production_shaped_sources_pass():
    headed = [(c, t, "IND safety reporting.") for c, t in (C1, C2)]
    a = judge(QUESTION, f"{CORRECT_15}\n\n{CORRECT_7}", headed)
    assert all(c["relevance"]["verdict"] == "relevant" for c in a.checked["claims"])
    assert a.to_dict()["answer_verdict"]["verdict"] == "correct"


def test_the_two_deadlines_reversed_is_a_substantive_error():
    a = judge(QUESTION, f"{REVERSED_7}\n\n{REVERSED_15}", [C1, C2])
    assert AN.CONTRADICTED in _finals(a)
    assert _substantive(a) == "fail"
    assert a.to_dict()["answer_verdict"]["verdict"] == "incorrect"
    assert "7" in a.to_dict()["answer_verdict"]["reason"] and "15" in a.to_dict()["answer_verdict"]["reason"]


def test_the_right_deadline_under_the_wrong_subsection_does_not_pass():
    """The 7-day rule is (c)(2)'s; presented as (c)(1)'s it must not be credited as correct."""
    a = judge(QUESTION, CORRECT_7.replace("[2]", "[1]"), [C1, C2])
    assert a.to_dict()["answer_verdict"]["verdict"] != "correct"
    assert _substantive(a) == "fail"


def test_the_right_citation_with_the_wrong_deadline_is_a_substantive_error():
    a = judge(QUESTION, REVERSED_15, [C1, C2])
    assert AN.CONTRADICTED in _finals(a)
    assert _substantive(a) == "fail"


def test_a_deadline_error_fails_an_answer_that_is_otherwise_correct():
    """Correct information elsewhere in the answer must not excuse the wrong deadline."""
    a = judge(QUESTION, f"The sponsor must promptly review all safety information. [1]\n\n{CORRECT_15}\n\n{REVERSED_15}",
              [C1, C2])
    finals = _finals(a)
    assert AN.CONTRADICTED in finals, "the reversed deadline must be contradicted"
    assert AN.SUPPORTED in finals, "the correct 15-day statement is still credited"
    assert _substantive(a) == "fail"
    assert a.to_dict()["answer_verdict"]["verdict"] == "incorrect"


def test_a_deadline_absent_from_the_evidence_is_reported_as_unsupported():
    """The original failure: (c)(2) was missing, and the answer asserted a 7-day deadline anyway."""
    from regulus.rag.validation import _periods, _supplied_figures
    supplied = _supplied_figures(C1_TEXT) | _supplied_figures(DEFINITION_TEXT)
    invented = ("the 7-calendar-day deadline applies to reports that qualify under paragraph (c)(1)(i) of this section")
    assert _periods(invented) == {"7"}
    assert "7" not in supplied, "no supplied source states a 7-day period"
    assert "15" in supplied


def test_a_period_the_evidence_spells_out_is_not_called_invented():
    from regulus.rag.validation import _periods, _supplied_figures
    assert _periods("within 7 calendar days") <= _supplied_figures("no later than seven calendar days after receipt")


# ==================================================================== Priority 1: synonyms and
# cross-references.
#
# Shared wording is not the only way a question names what it wants. "What are the IND safety
# reporting TIMEFRAMES" asks for exactly the paragraph the first fix found for "deadline", but
# scored (c)(2) at one shared word and the (c)(1) family at three, so the answer lost the ceiling
# again. Selection now prefers a paragraph that STATES a value of the kind asked for, and keeps a
# cross-referenced list whole so the conditions a deadline runs from arrive with it.

C1_II = ("(ii) Findings from other studies. The sponsor must report any findings from epidemiological studies, pooled "
         "analysis of multiple studies, or clinical studies, whether or not conducted under an IND, that suggest a "
         "significant risk in humans exposed to the drug, in an IND safety report as required by this section.")
C1_III = ("(iii) Findings from animal or in vitro testing. The sponsor must report any findings from animal or in "
          "vitro testing that suggest a significant risk in humans exposed to the drug, in an IND safety report as "
          "required by this section, whether or not conducted by the sponsor.")
C1_I = ("(i) Serious and unexpected suspected adverse reaction. The sponsor must notify FDA of any suspected adverse "
        "reaction that is both serious and unexpected.")
C1_IV = ("(iv) Increased rate of occurrence of serious suspected adverse reactions. The sponsor must notify FDA of any "
         "clinically important increase in the rate of a serious suspected adverse reaction.")

SYNONYMS = [
    "What are the IND safety reporting timeframes under 21 CFR 312.32?",
    "What are the time limits for IND safety reporting under 21 CFR 312.32?",
    "How quickly must IND safety reports be submitted under 21 CFR 312.32?",
    "What is the reporting timeline for IND safety reports under 21 CFR 312.32?",
]
NO_VALUE_ASKED = [
    "Summarize 21 CFR 312.32.",
    "What are the definitions in 21 CFR 312.32?",
    "What does 21 CFR 312.32 require?",
    "What must a sponsor report in an IND safety report under 21 CFR 312.32?",
]


def _xref_table():
    """21 CFR 312.32's shape where the cross-reference matters: definitions that crowd the ceiling,
    (c)(1) stating the 15-day clock and naming the four conditions it runs from, two of those
    conditions sharing the question's words and two not, and (c)(2) stating the 7-day rule last."""
    return [
        row("312.32", "(a)", 0, text=_pad("(a) Definitions. The following definitions of terms apply to this section:"
                                          f" {DEFINITION_TEXT}", 4000)),
        row("312.32", "(b)", 1, text=_pad("(b) Review of safety information. The sponsor must promptly review all "
                                          "information relevant to the safety of the drug.", 3400)),
        row("312.32", "(c)(1)", 2, text=C1_TEXT),
        row("312.32", "(c)(1)(i)", 3, text=C1_I),
        row("312.32", "(c)(1)(ii)", 4, text=C1_II),
        row("312.32", "(c)(1)(iii)", 5, text=C1_III),
        row("312.32", "(c)(1)(iv)", 6, text=C1_IV),
        row("312.32", "(c)(2)", 7, text=C2_TEXT),
        row("312.32", "(e)", 8, text=_pad("(e) Disclaimer. A safety report or other information submitted by a sponsor "
                                          "does not necessarily reflect a conclusion.", 500)),
    ]


def _xref_registry():
    return LexicalRetriever(TableDatabase(_xref_table())).run_plan(plan_query("What does 21 CFR 312.32 require?")).registry


def test_the_cross_reference_fixture_overflows_or_it_tests_nothing():
    assert select_evidence(_xref_registry()).overflow


# -------------------------------------------------- a synonym finds the answer-bearing paragraph


@pytest.mark.parametrize("question", SYNONYMS)
def test_a_synonym_for_the_deadline_selects_the_paragraph_that_states_it(question):
    """(c)(2) states the 7-calendar-day rule and shares almost none of these questions' words."""
    registry = _xref_registry()
    before = select_evidence(registry)
    after = select_evidence(registry, question=question)
    assert "21 CFR 312.32(c)(2)" not in _labels(before)
    assert "21 CFR 312.32(c)(2)" in _labels(after)
    assert "21 CFR 312.32(c)(1)" in _labels(after), "the 15-day rule must survive alongside it"


@pytest.mark.parametrize("question", SYNONYMS)
def test_a_synonym_is_recognised_for_ranking_but_does_not_widen_the_gate(question):
    """Selection may prefer period-bearing paragraphs on these wordings; the answerability gate,
    which WITHHOLDS an answer when the evidence states no value of the kind asked for, may not."""
    assert selection_value_kind(question) == "duration_or_frequency"
    assert requested_value_kind(question) is None


def test_the_gate_and_the_ranking_agree_on_the_wordings_the_gate_does_act_on():
    for question in ["What is the reporting deadline under 21 CFR 312.32?",
                     "How soon must a sponsor report an unexpected fatal reaction?",
                     "Within how many days must a sponsor submit an IND safety report?",
                     "By what date must a manufacturer comply with the amended Part 820?"]:
        assert selection_value_kind(question) == requested_value_kind(question) is not None


@pytest.mark.parametrize("question", NO_VALUE_ASKED)
def test_a_question_asking_for_no_value_does_not_engage_the_value_rank(question):
    """The rank and the cross-reference propagation both hang off this one condition, so a question
    that asks for no value is ranked on term overlap alone, exactly as before this change."""
    assert selection_value_kind(question) is None


def test_the_cross_reference_propagation_applies_only_when_a_value_is_asked_for():
    """Measured on the real corpus: with the propagation unconditional, "What follow-up does
    21 CFR 312.32 require?" lost (d)(1) - "the sponsor must promptly investigate all safety
    information it receives" - to make room for the (c)(1) reporting conditions. Ranked by term
    overlap alone, a list already stays together on document order, so the repair is confined to
    the case that needs it."""
    required = [e for e in _xref_registry().evidence if is_required(e)]
    value = _required_order(required, "What are the IND safety reporting timeframes under 21 CFR 312.32?")
    plain = _required_order(required, "What are the IND safety reporting requirements under 21 CFR 312.32?")
    low_overlap = ["21 CFR 312.32(c)(1)(i)", "21 CFR 312.32(c)(1)(iv)"]

    def rank(order, citation):
        return [e.citation for e in order].index(citation)

    for citation in low_overlap:
        assert rank(value, citation) < rank(plain, citation), (
            f"{citation} is pulled forward by the reference only when a value is asked for")


# -------------------------------------------------- a cross-referenced list is kept whole

LIST = ["21 CFR 312.32(c)(1)(i)", "21 CFR 312.32(c)(1)(ii)", "21 CFR 312.32(c)(1)(iii)", "21 CFR 312.32(c)(1)(iv)"]


@pytest.mark.parametrize("question", SYNONYMS + [DEADLINE_QUESTION])
def test_the_conditions_a_deadline_runs_from_are_kept_as_a_unit(question):
    """(c)(1)'s clock runs from a determination "under paragraph (c)(1)(i), (c)(1)(ii), (c)(1)(iii),
    or (c)(1)(iv)". Keeping two of those four and dropping the others states the rule against half
    its conditions, which is how a deadline gets attached to the wrong provision."""
    kept = _labels(select_evidence(_xref_registry(), question=question))
    present = [c for c in LIST if c in kept]
    assert present == LIST, f"the list must be whole, got {present}"


def test_without_the_cross_reference_two_of_the_four_conditions_are_lost():
    """What the propagation repairs: ranked on their own words, (ii) and (iii) survive and (i) and
    (iv) do not. Guards the reason the rule exists, not the rule's implementation."""
    registry = _xref_registry()
    question = SYNONYMS[0]
    keys, kind = _match_keys(question), selection_value_kind(question)
    rows = [e for e in registry.evidence if e.citation in LIST]
    scores = {e.citation: _relevance(e.text, keys) for e in rows}
    assert scores["21 CFR 312.32(c)(1)(ii)"] > scores["21 CFR 312.32(c)(1)(i)"]
    assert scores["21 CFR 312.32(c)(1)(iii)"] > scores["21 CFR 312.32(c)(1)(iv)"]
    assert kind is not None


# -------------------------------------------------- the reference parser


def test_a_paragraph_reference_list_is_read():
    assert _referenced_paths(C1_TEXT) == {"(c)(1)(i)", "(c)(1)(ii)", "(c)(1)(iii)", "(c)(1)(iv)"}


def test_an_exception_names_the_paragraph_it_defers_to():
    assert _referenced_paths("(a) Except as provided in paragraph (d) of this section, you must comply.") == {"(d)"}
    assert _referenced_paths("as required by paragraphs (a) and (b) of this section") == {"(a)", "(b)"}


def test_a_reference_to_another_section_is_not_a_paragraph_of_this_one():
    assert _referenced_paths("must be reported under 21 CFR 312.33(a) of this chapter") == frozenset()
    assert _referenced_paths("in accordance with 211.180(c)") == frozenset()


def test_text_that_references_nothing_yields_nothing():
    assert _referenced_paths(C2_TEXT) == frozenset()
    assert _referenced_paths("") == frozenset()


# -------------------------------------------------- the ceilings and the presentation order hold


@pytest.mark.parametrize("question", SYNONYMS + NO_VALUE_ASKED + [DEADLINE_QUESTION])
def test_the_ceilings_and_document_order_hold_for_every_wording(question):
    policy = ContextPolicy()
    sel = select_evidence(_xref_registry(), question=question)
    assert sel.total_chars <= policy.hard_max_total_chars
    assert len(sel.selected) <= policy.hard_max_results
    ordinals = [s.evidence.result.metadata["ordinal"] for s in sel.selected]
    assert ordinals == sorted(ordinals), "relevance decides what is kept, never the order shown"


# ==================================================================== Priority 2: generation.
#
# With (c)(2) in the evidence, "How quickly must IND safety reports be submitted?" is still
# answered with the 15-day rule only. The cause is in the prompt, not the evidence: the question
# gets the FOCUSED answer shape, whose instruction reads "state only what the requested provision
# says about the focus ... Do not describe the rest of the provision", and its focus terms are the
# question's own words - quickly, ind, safety, reports, submitted. (c)(1) carries four of them and
# (c)(2) one, so the model states (c)(1) and leaves the distinction out, although system rule 10
# says never to drop "a distinction between subsections". The 15-day figure also appears in three
# sources against one for 7 days.
#
# Every prompt change measured to surface the 7-day rule made the 0.6B model either dump the whole
# section or INVENT which reports each period applies to - "a serious and unexpected suspected
# adverse reaction ... no later than 7 calendar days", the (c)(1)(i) category given (c)(2)'s
# period. An omission was traded for a confident wrong statement, so no generation change ships.
# These tests pin what can be pinned without the model: that the evidence reaches it, and that the
# evaluator fails the wrong statements the rejected prompts produced.

C1_I_VERBATIM = ("21 CFR 312.32(c)(1)(i)", "(i) Serious and unexpected suspected adverse reaction. The sponsor must "
                 "report any suspected adverse reaction that is both serious and unexpected. The sponsor must report an "
                 "adverse event as a suspected adverse reaction only if there is evidence to suggest a causal "
                 "relationship between the drug and the adverse event, such as:")
HOW_QUICKLY = "How quickly must IND safety reports be submitted under 21 CFR 312.32?"
SEVEN_DAY = ("What is the reporting deadline for an unexpected fatal or life-threatening suspected adverse reaction "
             "under 21 CFR 312.32?")


@pytest.mark.parametrize("question", [HOW_QUICKLY, SYNONYMS[0], SEVEN_DAY, DEADLINE_QUESTION])
def test_both_deadline_paragraphs_reach_the_model(question):
    """Generation cannot state what it is not given: for every deadline wording, (c)(1) and (c)(2)
    are both in the selected evidence, with the conditions (c)(1)'s clock runs from."""
    kept = _labels(select_evidence(_xref_registry(), question=question))
    assert "21 CFR 312.32(c)(1)" in kept and "21 CFR 312.32(c)(2)" in kept
    assert [c for c in LIST if c in kept] == LIST


def test_the_misassignment_a_rejected_prompt_produced_is_contradicted():
    """Observed verbatim in the ablation: (c)(1)(i)'s category given (c)(2)'s 7-day period, cited to
    (c)(2). It agrees with its cited source on the number, so only the subject matter gives it away."""
    answer = ("The sponsor must report a serious and unexpected suspected adverse reaction as soon as possible, but in "
              "no case later than 7 calendar days after the sponsor's initial receipt of the information. [3]")
    a = judge(HOW_QUICKLY, answer, [C1, C1_I_VERBATIM, C2])
    assert a.checked["claims"][0]["support"] == AN.CONTRADICTED
    assert _substantive(a) == "fail"
    assert "15" in a.checked["claims"][0]["support_reason"] and "7" in a.checked["claims"][0]["support_reason"]


def test_the_seven_day_rule_given_the_fifteen_day_period_is_contradicted():
    answer = ("The sponsor must notify FDA of any unexpected fatal or life-threatening suspected adverse reaction no "
              "later than 15 calendar days after initial receipt of the information. [3]")
    a = judge(HOW_QUICKLY, answer, [C1, C1_I_VERBATIM, C2])
    assert a.checked["claims"][0]["support"] == AN.CONTRADICTED
    assert _substantive(a) == "fail"


def test_a_correct_seven_day_statement_is_supported_by_its_source():
    answer = ("The sponsor must notify FDA of any unexpected fatal or life-threatening suspected adverse reaction no "
              "later than 7 calendar days after the sponsor's initial receipt of the information. [3]")
    claim = judge(HOW_QUICKLY, answer, [C1, C1_I_VERBATIM, C2]).checked["claims"][0]
    assert claim["support"] == AN.SUPPORTED
    assert claim["support_basis"] != "operator_conflict"


# Production sources carry their SECTION heading ("IND safety reporting."), and the relevance check
# reads heading and text together. The `judge` harness builds sources WITHOUT a heading unless one is
# given, and without it a correct (c)(2) statement looked off-topic for "How quickly ...?" - which was
# recorded here, and in three reports, as an evaluator limitation. Against the real package it is
# relevant and correct (Priority 2A, work/relevance_trace.py). These sources are production-shaped.
HEADED = [(c, t, "IND safety reporting.") for c, t in (C1, C1_I_VERBATIM, C2)]


def test_a_correct_seven_day_answer_to_how_quickly_is_relevant_and_correct():
    answer = ("The sponsor must notify FDA of any unexpected fatal or life-threatening suspected adverse reaction no "
              "later than 7 calendar days after the sponsor's initial receipt of the information. [3]")
    a = judge(HOW_QUICKLY, answer, HEADED)
    claim = a.checked["claims"][0]
    assert claim["relevance"]["verdict"] == "relevant" and claim["support"] == AN.SUPPORTED
    assert _substantive(a) == "pass"


def test_relevance_reads_the_section_heading_a_production_source_carries():
    """Why the harness and production disagreed: the same statement, with and without the heading."""
    answer = ("The sponsor must notify FDA of any unexpected fatal or life-threatening suspected adverse reaction no "
              "later than 7 calendar days after the sponsor's initial receipt of the information. [3]")
    bare = judge(HOW_QUICKLY, answer, [C1, C1_I_VERBATIM, C2]).checked["claims"][0]["relevance"]["verdict"]
    headed = judge(HOW_QUICKLY, answer, HEADED).checked["claims"][0]["relevance"]["verdict"]
    assert (bare, headed) == ("irrelevant", "relevant")


def test_a_reversed_deadline_is_still_contradicted_against_production_shaped_sources():
    """The heading makes a statement relevant; it never makes a wrong one correct."""
    answer = ("The sponsor must notify FDA of any unexpected fatal or life-threatening suspected adverse reaction no "
              "later than 15 calendar days after initial receipt of the information. [3]")
    a = judge(HOW_QUICKLY, answer, HEADED)
    assert a.checked["claims"][0]["support"] == AN.CONTRADICTED and _substantive(a) == "fail"


def test_with_the_gpu_budget_the_whole_section_fits(monkeypatch):
    # CB1/HX1/BA1 (promoted 30 Oct): the 15,000-character / 48-source ceiling without URL charges holds every required
    # paragraph of 21 CFR 312.32 - the overflow the tests above exercise no longer happens for this section
    from regulus.rag import validator_switches as VS
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"CB1", "HX1", "BA1"})
    selection = select_evidence(_registry("What does 21 CFR 312.32 require?"), question=DEADLINE_QUESTION)
    assert not selection.overflow and not [d for d in selection.dropped if d.required]
