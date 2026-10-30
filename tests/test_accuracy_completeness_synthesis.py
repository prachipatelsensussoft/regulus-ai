"""Completeness of a SYNTHESIS of a named provision ("What does 21 CFR 312.32 require?").

Found 2026-10-13 (artifacts/accuracy/synthesis_completeness_*): answered with 21 CFR 312.32(c)(1) alone,
the synthesis was assessed PASS 100 although it never states the 7-day rule of (c)(2) or the follow-up
duties of (d). Two separate paths led there:

  * with every paragraph in the evidence, the pipeline's group-coverage check counted a group as present
    if ANY member was cited, named, word-matched or cross-referenced - (c)(1) "covered" (c)(2) as a member
    of group (c), and (d)(3) "covered" itself because the answer mentions "paragraph (c)", which (d)(3)
    cross-references - and the evaluator, which never checked a synthesis, passed every supported answer;
  * in production the question has no words to rank paragraphs by, so the hard ceiling drops (c)(2) and
    (d)(1)-(3); repair then appends the bare heading "(d) Followup." for group (d), and the answer is
    complete against what remained.

evaluator._synthesis_gap checks every top-level obligation of the named provision (obligations nested under
another obligation are its detail, so not every paragraph is required), and treats obligations the ceiling
kept out of the evidence as making completeness impossible to establish. Measured: 40 named sections,
answered completely (every top-level obligation verbatim) and with one obligation left out (143 answers) -
before, 6 of the 143 incomplete answers were not a full PASS; after, 132, with all 40 complete answers still
PASS. Of the 299 recorded acceptance answers two scores change, NC-013 and NC-018, both hand-scored N=0 O=0.

Evidence: VERBATIM 21 CFR 312.32 (tests/fixtures/accuracy_312_32_rows.json) and 211.25, through the real
plan, retrieval, package, validator and repair on the PRODUCTION configuration. Answers: CONSTRUCTED.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from regulus.accuracy import evaluator as E
from regulus.accuracy.review import FAIL, NOT_EVALUATED, PASS
from tests.test_accuracy_completeness_focus import (
    BROAD_SECTION, CORRECT_SEVEN, FOLLOW_UP, S312, SEVEN, _F, _mode, body)
from tests.test_accuracy_pass_integrity import C, S211, completeness, production
from tests.test_retrieval_plan import row

# SOURCE numbers in S312: 1 (a), 2 (b), 3 (c)(1), 4 (c)(1)(i), 5 (c)(2), 6 (d), 7 (d)(1), 8 (d)(2), 9 (d)(3)


def m(*paragraphs):
    index = {"(b)": 2, "(c)(1)": 3, "(c)(2)": 5, "(d)(1)": 7, "(d)(2)": 8, "(d)(3)": 9}
    return " ".join(f"{body(p)} [{index[p]}]" for p in paragraphs)


COMPLETE = m("(b)", "(c)(1)", "(c)(2)", "(d)(1)", "(d)(2)", "(d)(3)")


# ---------------------------------------------------------------- 1-4. broad synthesis


def test_a_synthesis_of_c1_alone_is_not_a_full_pass_although_repair_appends_a_and_b():
    v, d, out = production(BROAD_SECTION, m("(c)(1)"), S312)
    assert _mode(out) == "synthesis"
    assert body("(b)")[:40] in out.answer, "repair still appends the missing groups it detects"
    assert v["status"] == FAIL and completeness(d)["state"] == "fail"
    for missing in ("312.32(c)(2)", "312.32(d)(1)", "312.32(d)(2)", "312.32(d)(3)"):
        assert missing in completeness(d)["detail"]


def test_a_complete_synthesis_passes():
    v, d, _ = production(BROAD_SECTION, COMPLETE, S312)
    assert completeness(d)["state"] == "pass"
    assert (v["status"], v["live_check_score"]) == (PASS, 100)


def test_a_synthesis_without_the_seven_day_rule_fails():
    """(c)(2) sits in the same top-level group as (c)(1); a group-level check could not see it go."""
    v, d, _ = production(BROAD_SECTION, m("(b)", "(c)(1)", "(d)(1)", "(d)(2)", "(d)(3)"), S312)
    assert v["status"] == FAIL
    assert completeness(d)["state"] == "fail" and "312.32(c)(2)" in completeness(d)["detail"]


def test_a_synthesis_without_follow_up_fails():
    v, d, _ = production(BROAD_SECTION, m("(b)", "(c)(1)", "(c)(2)"), S312)
    assert v["status"] == FAIL
    assert completeness(d)["state"] == "fail" and "312.32(d)" in completeness(d)["detail"]


def test_individually_supported_statements_do_not_make_an_incomplete_synthesis_complete():
    _, d, _ = production(BROAD_SECTION, m("(b)", "(c)(1)", "(c)(2)"), S312)
    assert all(c["support"] == "supported" for c in d["checked"]["claims"])
    assert next(x for x in d["dimensions"] if x["dimension"] == "substantive_correctness")["state"] == "pass"
    assert completeness(d)["state"] == "fail", "supported and correct, yet incomplete"


# ---------------------------------------------------------------- the ceiling withheld the provision's obligations

PAD = "This definition applies to the terms used in this section as they are described here. "
_TEXT = {p["paragraph"]: p["text"] for p in _F["paragraphs"]}


def _crowded_table():
    """21 CFR 312.32 as production sees it for this question: long text ahead of the requirements fills the
    hard ceiling in document order, so (c)(2) and follow-up never reach the evidence."""
    a = _TEXT["(a)"] + " " + PAD * 50
    b = _TEXT["(b)"] + " " + PAD * 50
    rows = [("(a)", a[:4000]), ("(b)", b[:3900])] + [(p, _TEXT[p]) for p in
                                                     ("(c)(1)", "(c)(1)(i)", "(c)(2)", "(d)", "(d)(1)", "(d)(2)", "(d)(3)")]
    return [row("312.32", p, i, text=t, section_heading=_F["section_heading"]) for i, (p, t) in enumerate(rows, 1)]


@pytest.mark.usefixtures("cpu_era_budget")   # fixtures sized to overflow the CPU-era evidence budget (tests/conftest.py)
def test_obligations_the_ceiling_kept_out_make_completeness_impossible_to_establish():
    table = _crowded_table()
    _, _, probe = production(BROAD_SECTION, "x", table)
    kept = {s.citation for s in probe.package.sources}
    assert "21 CFR 312.32(c)(2)" not in kept, "the fixture must withhold the 7-day rule, or it tests nothing"
    assert "21 CFR 312.32(c)(2)" in {c for c, _ in probe.package.withheld}
    # an answer complete against what WAS kept: every kept source, verbatim
    answer = " ".join(f"{s.text.strip()} [{s.index}]" for s in probe.package.sources)
    v, d, _ = production(BROAD_SECTION, answer, table)
    assert v["status"] != PASS
    assert completeness(d)["state"] in ("not_established", "fail")


@pytest.mark.usefixtures("cpu_era_budget")   # fixtures sized to overflow the CPU-era evidence budget (tests/conftest.py)
def test_the_withheld_rows_never_reach_the_prompt():
    from regulus.rag.generation import render_package
    _, _, probe = production(BROAD_SECTION, "x", _crowded_table())
    prompt = render_package(probe.package)
    assert probe.package.withheld and _TEXT["(c)(2)"][:60] not in prompt


# ---------------------------------------------------------------- 5-7. focused questions are unchanged


def test_the_focused_seven_day_answer_still_passes():
    v, d, out = production(SEVEN, CORRECT_SEVEN, S312)
    assert _mode(out) == "focused" and (v["status"], v["live_check_score"]) == (PASS, 100)


def test_a_seven_day_answer_without_its_starting_event_still_fails():
    answer = ("An unexpected fatal or life-threatening suspected adverse reaction must be reported to FDA no later than "
              "7 calendar days [5].")
    v, _, _ = production(SEVEN, answer, S312)
    assert v["status"] == FAIL


def test_a_wrong_deadline_for_the_fatal_category_still_fails(monkeypatch):
    # the evaluator's own check, on a delivered answer: without FC2 the wrong figure reaches it
    from regulus.rag import validator_switches as VS
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"FC2"})
    v, d, _ = production(SEVEN, CORRECT_SEVEN.replace("7 calendar", "15 calendar"), S312)
    assert v["status"] == FAIL and d["checked"]["claims"][0]["support"] == "contradicted"


def test_with_fc2_a_wrong_deadline_is_never_delivered(monkeypatch):
    # 30 Oct, promoting FC2: the 15-day figure under the 7-day paragraph (c)(2) is caught before delivery and the
    # provision is quoted instead - the wrong answer never reaches the reader (the evaluator then scores a quotation)
    from regulus.rag import validator_switches as VS
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"FC2"})
    v, d, out = production(SEVEN, CORRECT_SEVEN.replace("7 calendar", "15 calendar"), S312)
    codes = [p["code"] for p in out.metadata["validation"]["initial"]["problems"]]
    assert "figure_misattributed" in codes and out.metadata["fallback_type"] is not None
    wrong = CORRECT_SEVEN.replace("7 calendar", "15 calendar").split(" [")[0]
    # the model's sentence is never shown; the quoted regulation may itself say "15 calendar days" ((c)(1))
    assert wrong not in out.answer and v["status"] != PASS


def test_a_focused_question_is_not_asked_for_the_whole_section():
    """The synthesis check never runs on a focused plan: the narrow answer is not asked for (b) or (d)."""
    _, d, out = production(SEVEN, CORRECT_SEVEN, S312)
    assert _mode(out) == "focused" and "requires" not in completeness(d)["detail"]


# ---------------------------------------------------------------- 9. 211.25 controls


def test_a_211_25_synthesis_completed_by_repair_still_passes():
    v, d, out = production("What does 21 CFR 211.25 require?", f"{C} [3]", S211)
    assert _mode(out) == "synthesis" and (v["status"], v["live_check_score"]) == (PASS, 100)


# ---------------------------------------------------------------- the unit rule itself

def _pkg(texts):
    """A synthesis plan over one requested section; texts: {paragraph: text}, in document order."""
    sources, members = [], []
    for i, (para, text) in enumerate(texts.items(), 1):
        sources.append(SimpleNamespace(index=i, citation=f"21 CFR 9.9{para}", text=text, tier="exact",
                                       citation_fields={"paragraph": para}))
        members.append(i)
    group = SimpleNamespace(kind="requirement", requirement_ids=("r1",), members=tuple(members))
    plan = SimpleNamespace(shape=SimpleNamespace(mode="synthesis"), groups=(group,))
    return SimpleNamespace(answer_plan=plan, requirements=(("r1", "21 CFR 9.9"),), sources=tuple(sources), withheld=())


def _gap(texts, answer):
    return E._synthesis_gap(_pkg(texts), answer, (), (), ())


def test_an_obligation_reproduced_verbatim_is_represented():
    texts = {"(a)": "You must keep records of every lot.", "(b)": "You must test each batch before release."}
    missing, maybe, withheld = _gap(texts, "You must keep records of every lot. You must test each batch before release.")
    assert (missing, maybe, withheld) == ([], [], [])


def test_a_shared_stock_phrase_is_not_a_reproduction():
    long_b = ("You must report it as soon as possible, but in no case later than 15 calendar days after the "
              "determination is made, and identify every earlier report of a similar reaction in the same submission.")
    texts = {"(a)": "You must notify FDA as soon as possible, but in no case later than 15 calendar days after receipt.",
             "(b)": long_b}
    missing, _, _ = _gap(texts, texts["(a)"])
    assert missing == ["21 CFR 9.9(b)"]


def test_the_detail_under_an_obligation_is_not_a_separate_unit():
    texts = {"(a)": "You must keep these records:", "(a)(1)": "The lot number must be recorded.",
             "(a)(2)": "The date must be recorded."}
    missing, _, _ = _gap(texts, "The lot number must be recorded.")
    assert missing == [], "the lead-in is represented by one of its items; its items are not units of their own"


def test_a_bare_heading_states_no_obligation_and_stands_in_for_none():
    texts = {"(a)": "You must keep records of every lot.", "(d)": "(d) Followup.",
             "(d)(1)": "The sponsor must promptly investigate all safety information it receives."}
    missing, _, _ = _gap(texts, "You must keep records of every lot. (d) Followup.")
    assert missing == ["21 CFR 9.9(d)(1)"]


@pytest.mark.parametrize("mode,requirements", [("focused", (("r1", "21 CFR 9.9"),)), ("synthesis", ())])
def test_the_check_applies_only_to_a_synthesis_of_a_named_provision(mode, requirements):
    pkg = _pkg({"(a)": "You must keep records."})
    pkg.answer_plan.shape.mode = mode
    pkg.requirements = requirements
    assert E._synthesis_gap(pkg, "", (), (), ()) is None


def test_withheld_obligations_are_reported_but_withheld_definitions_are_not():
    pkg = _pkg({"(a)": "You must keep records."})
    pkg.withheld = (("21 CFR 9.9(c)", "The sponsor must report within 7 days."), ("21 CFR 9.9(x)", "Lot means a batch."))
    assert E._synthesis_gap(pkg, "You must keep records.", (), (), ())[2] == ["21 CFR 9.9(c)"]
