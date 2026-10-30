"""Completeness of a FOCUSED answer: which requirement groups a narrow question actually asks about.

Found 2026-10-13 (artifacts/accuracy/completeness_focus_gap_*): a correct answer to "What is the
reporting deadline for an unexpected fatal or life-threatening suspected adverse reaction under 21 CFR
312.32?" - "no later than 7 calendar days after the sponsor's initial receipt of the information" -
FAILED completeness for not covering 21 CFR 312.32(d), follow-up. (d)(3) carries four of the question's
eight topic words (report, suspected, adverse, reaction), which met the on-topic threshold; all four
also appear in (c), the group the answer covered. The words that say WHICH deadline is meant -
unexpected, fatal, life-threatening - appear in (c) alone. A group is now demanded only if it also
carries one of the question's words that tell the on-topic groups apart (evaluator._focus_gap).

That change alone would have let an answer that drops when the 7-day clock starts PASS 100, since
nothing compared a time limit's anchor. So a claim stating a time limit its evidence anchors to an event
("after the sponsor's initial receipt of the information"), without any anchor of its own, now drops a
condition of that limit and is at most partially supported (operators.dropped_time_anchor).

Measured before adoption: of the 122 focused recorded acceptance answers, the focus gap demands a group
on 12 (11 of them hand-scored incomplete, O=0) before and after; across all 299 recorded answers one
score changes - U-010, correct (100) -> partial (40), hand N=1 O=0.

Evidence: VERBATIM 21 CFR 312.32 from the read-only eCFR snapshot (tests/fixtures/accuracy_312_32_rows.json,
snapshot date and content hash per paragraph), served through the real plan, retrieval, package and
validator on the PRODUCTION configuration. Answers: CONSTRUCTED, scripted in place of the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from regulus.accuracy import answer as AN
from regulus.accuracy import operators as OP
from regulus.accuracy.review import FAIL, PASS
from tests.test_accuracy_pass_integrity import completeness, production
from tests.test_retrieval_plan import row

# scripted model drafts: the model-drafted path (tests/conftest.py)
pytestmark = pytest.mark.usefixtures("model_drafted_path")

_F = json.loads((Path(__file__).parent / "fixtures" / "accuracy_312_32_rows.json").read_text(encoding="utf-8"))
_TEXT = {p["paragraph"]: p["text"] for p in _F["paragraphs"]}
# one row per paragraph, in document order: SOURCE 1 (a), 2 (b), 3 (c)(1), 4 (c)(1)(i), 5 (c)(2),
# 6 (d), 7 (d)(1), 8 (d)(2), 9 (d)(3)
S312 = [row("312.32", p["paragraph"], i, text=p["text"], section_heading=_F["section_heading"])
        for i, p in enumerate(_F["paragraphs"], 1)]


def body(paragraph):
    """A paragraph's operative text without its designation and heading."""
    text = _TEXT[paragraph]
    if text.startswith("("):
        text = text.split(") ", 1)[1] if paragraph in ("(d)(1)", "(d)(2)", "(d)(3)", "(c)(1)(i)") else text
    if paragraph in ("(b)", "(c)(1)", "(c)(2)"):
        text = text.split(". ", 1)[1]
    return text


SEVEN = ("What is the reporting deadline for an unexpected fatal or life-threatening suspected adverse reaction "
         "under 21 CFR 312.32?")
BROAD_FOCUSED = "What are the IND safety reporting requirements under 21 CFR 312.32?"
BROAD_SECTION = "What does 21 CFR 312.32 require?"
FOLLOW_UP = "What follow-up does 21 CFR 312.32 require after an IND safety report?"
COMPARE = ("What is the difference between the 15-calendar-day and 7-calendar-day IND safety reporting deadlines in "
           "21 CFR 312.32?")
CORRECT_SEVEN = ("The reporting deadline for an unexpected fatal or life-threatening suspected adverse reaction is no "
                 "later than 7 calendar days after the sponsor's initial receipt of the information [5].")


def _mode(out):
    return (out.metadata.get("answer_shape") or {}).get("mode")


def _claims(d):
    return d["checked"]["claims"]


# ---------------------------------------------------------------- 1. the correct seven-day answer


def test_a_correct_seven_day_answer_passes():
    v, d, out = production(SEVEN, CORRECT_SEVEN, S312)
    assert _mode(out) == "focused"
    assert completeness(d)["state"] == "pass"
    assert (v["status"], v["live_check_score"]) == (PASS, 100)


def test_follow_up_is_not_demanded_of_a_question_about_the_seven_day_deadline():
    _, d, _ = production(SEVEN, CORRECT_SEVEN, S312)
    assert "312.32(d)" not in completeness(d)["detail"]


# ---------------------------------------------------------------- 2. a material condition dropped


def test_a_seven_day_answer_that_drops_when_the_clock_starts_does_not_pass():
    """The period is right; the event it runs from is gone. Before this change it FAILED, but only for
    the unrelated (d) demand - exactly as the correct answer did - so nothing told the two apart."""
    answer = ("An unexpected fatal or life-threatening suspected adverse reaction must be reported to FDA no later than "
              "7 calendar days [5].")
    v, d, _ = production(SEVEN, answer, S312)
    assert v["status"] == FAIL
    claim = _claims(d)[0]
    assert claim["support"] != AN.SUPPORTED
    assert "from a stated event" in claim["support_reason"]


def test_the_trigger_kept_in_other_words_is_not_called_dropped():
    answer = ("An unexpected fatal or life-threatening suspected adverse reaction must be reported to FDA within 7 "
              "calendar days of the sponsor first receiving the information [5].")
    _, d, _ = production(SEVEN, answer, S312)
    assert "from a stated event" not in _claims(d)[0]["support_reason"]


# ---------------------------------------------------------------- 3. broad questions


def test_a_broad_focused_question_still_demands_every_group_it_is_about():
    """"IND safety reporting requirements": every topic word is shared by the on-topic groups, so no word
    distinguishes them and the rule is exactly what it was - (d) is still demanded."""
    v, d, out = production(BROAD_FOCUSED, f"{body('(c)(1)')} [3] {body('(c)(2)')} [5]", S312)
    assert _mode(out) == "focused" and v["status"] == FAIL
    assert completeness(d)["state"] == "fail" and "312.32(d)" in completeness(d)["detail"]


def test_a_question_about_the_whole_section_is_a_synthesis_the_focus_gap_does_not_judge():
    _, _, out = production(BROAD_SECTION, f"{body('(c)(1)')} [3]", S312)
    assert _mode(out) == "synthesis"
    assert body("(b)")[:40] in out.answer, "the pipeline appends uncovered groups to a synthesis before assessment"


def test_a_section_answer_missing_follow_up_and_the_seven_day_rule_does_not_pass():
    """Was a strict xfail (identical on the evaluator before 2026-10-13): answered with (c)(1) alone, the
    pipeline appends (a) and (b) but never the 7-day rule or follow-up, and the synthesis was PASS 100.
    Fixed by evaluator._synthesis_gap; see tests/test_accuracy_completeness_synthesis.py."""
    v, d, _ = production(BROAD_SECTION, f"{body('(c)(1)')} [3]", S312)
    assert v["status"] != PASS
    assert completeness(d)["state"] == "fail"
    assert "312.32(c)(2)" in completeness(d)["detail"]


# ---------------------------------------------------------------- 4. a question about follow-up


def test_a_follow_up_question_answered_without_follow_up_fails():
    v, d, _ = production(FOLLOW_UP, f"{body('(c)(1)')} [3]", S312)
    assert v["status"] == FAIL
    assert completeness(d)["state"] == "fail" and "312.32(d)" in completeness(d)["detail"]


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN, pre-existing and unchanged: a correct follow-up answer still fails completeness. The question's word "
    "'follow-up' does not match the regulation's 'followup', so the only word left to tell the groups apart is "
    "'after' - which (c)(1) also contains ('15 calendar days after the sponsor determines') - and (c) is demanded. "
    "Strict: a fix flips it."))
def test_a_correct_follow_up_answer_passes():
    answer = f"{body('(d)(1)')} [7] {body('(d)(2)')} [8] {body('(d)(3)')} [9]"
    v, _, _ = production(FOLLOW_UP, answer, S312)
    assert v["status"] == PASS


# ---------------------------------------------------------------- 5. comparing both deadlines


def test_a_comparison_answer_states_both_deadlines_without_contradiction():
    v, d, _ = production(COMPARE, f"{body('(c)(1)')} [3] {body('(c)(2)')} [5]", S312)
    assert all(c["support"] == AN.SUPPORTED for c in _claims(d))
    assert "312.32(d)" not in completeness(d)["detail"]
    assert next(x for x in d["dimensions"] if x["dimension"] == "substantive_correctness")["state"] != "fail"


def test_a_comparison_that_reverses_the_deadlines_is_still_contradicted():
    reversed_answer = ("The sponsor must notify FDA of any unexpected fatal or life-threatening suspected adverse "
                       "reaction as soon as possible but in no case later than 15 calendar days after the sponsor's "
                       "initial receipt of the information [5].")
    v, d, _ = production(COMPARE, reversed_answer, S312)
    assert v["status"] == FAIL and _claims(d)[0]["support"] == AN.CONTRADICTED


# ---------------------------------------------------------------- the anchor check itself

C2 = _TEXT["(c)(2)"]


@pytest.mark.parametrize("claim,dropped", [
    ("It must be reported no later than 7 calendar days.", True),
    ("It must be reported within 7 calendar days [5].", True),
    ("It must be reported no later than 7 calendar days after the sponsor's initial receipt of the information.", False),
    ("It must be reported within 7 days of receiving the information.", False),        # paraphrased anchor: accepted
    ("It must be reported no later than 7 calendar days [5] after initial receipt.", False),
    ("The 7-calendar-day deadline applies to fatal or life-threatening reactions.", False),  # a label, not a limit
    ("It must be reported no later than 15 calendar days.", False),                    # another period: not this check
])
def test_a_time_limit_without_the_event_it_runs_from(claim, dropped):
    assert (OP.dropped_time_anchor(claim, C2) is not None) is dropped


def test_an_unanchored_period_in_the_evidence_asks_for_no_anchor():
    assert OP.dropped_time_anchor("It must be kept for 2 years.", "Records must be kept for 2 years.") is None


def test_a_dropped_anchor_is_an_omission_never_a_conflict():
    c = OP.compare("The sponsor must notify FDA no later than 7 calendar days.", C2)
    assert any(o["operator"] == OP.OMISSION_TIME_ANCHOR for o in c.omissions)
    assert not c.conflicted
