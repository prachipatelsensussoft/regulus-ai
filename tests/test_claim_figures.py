"""FC2 (29 Oct): a figure stated under one paragraph's citation that only another supplied paragraph contains."""
from types import SimpleNamespace

import pytest

from regulus.rag import validator_switches as VS
from regulus.rag.claim_validation import FIGURE_MISATTRIBUTED, check_claims

# 117.201(c)(2)(i) is a lead-in; its children carry the deadlines (300Q F-016)
ROWS = [(16, "(c)(2)(i)", "(i) Submitted to FDA initially:"),
        (17, "(c)(2)(i)(A)", "(A) By December 17, 2018, for a facility that begins operations before September 17, 2018;"),
        (18, "(c)(2)(i)(B)", "(B) Before beginning operations, for a facility that begins operations after September 17, 2018; or"),
        (19, "(c)(2)(i)(C)", "(C) By July 31 of the applicable calendar year, when the status of a facility changes")]
PKG = SimpleNamespace(sources=[SimpleNamespace(index=i, text=t, citation=f"21 CFR 117.201{p}", requirement_ids=("r1",),
                                               tier="exact") for i, p, t in ROWS], answer_style="restate")
IDENT = lambda s: ("21", "117.201", s.citation.split("117.201")[1])
RELATED = lambda a, b: a[:2] == b[:2] and (a[2].startswith(b[2]) or b[2].startswith(a[2]))


def codes(sentence):
    return [f.code for f in check_claims([sentence], PKG, question="What must a qualified facility submit?",
                                         identity=IDENT, related=RELATED)]


@pytest.fixture(autouse=True)
def fc2_on(monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"FC2"})


def test_a_deadline_under_the_wrong_sibling_is_misattributed():
    assert FIGURE_MISATTRIBUTED in codes("21 CFR 117.201(c)(2)(i)(B) [18]: Submitted by July 31 of the applicable year.")


def test_the_same_deadline_under_its_own_paragraph_is_not():
    assert FIGURE_MISATTRIBUTED not in codes("21 CFR 117.201(c)(2)(i)(C) [19]: Submitted by July 31 of the applicable year.")


def test_a_lead_in_cited_for_its_childs_figure_is_tolerated():
    assert FIGURE_MISATTRIBUTED not in codes("21 CFR 117.201(c)(2)(i) [16]: Submitted initially by December 17, 2018.")


def test_paragraph_designators_and_section_numbers_are_not_figures():
    assert FIGURE_MISATTRIBUTED not in codes("21 CFR 117.201(c)(2)(i)(B) [18]: Before beginning operations (see 117.201(c)(2)(i)(C)).")


def test_off_without_the_switch(monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"FC2"})
    assert FIGURE_MISATTRIBUTED not in codes("21 CFR 117.201(c)(2)(i)(B) [18]: Submitted by July 31 of the applicable year.")


def test_each_cited_line_is_checked_on_its_own():
    """List lines ending in ':' or '; or' reach the check as one sentence; the parent cited on the first line must not
    cover a figure the last line misplaces (F-016 replay)."""
    merged = ("21 CFR 117.201(c)(2)(i) [16]: Initially, by December 17, 2018; or\n\n"
              "21 CFR 117.201(c)(2)(i)(B) [18]: Submitted by July 31 of the applicable year.")
    assert FIGURE_MISATTRIBUTED in codes(merged)


# ------------------------------------------------------------ FD1 false denials (29 Oct)

from regulus.rag.claim_validation import false_denial  # noqa: E402

HCTP = SimpleNamespace(index=3, citation="21 CFR 820.1(a)(4)", text=(
    "(4) HCT/Ps. The provisions of this part apply to manufacturers of human cells, tissues, and cellular and tissue-based "
    "products (HCT/Ps), as defined in § 1271.3(d) of this chapter, that are devices (subject to premarket review or "
    "notification, or exempt from notification)."))
BLOOD = SimpleNamespace(index=2, citation="21 CFR 820.1(a)(3)", text=(
    "(3) Blood and blood components. Manufacturers of blood and blood components used for transfusion or for further "
    "manufacturing are not subject to this part."))
REPORTING = SimpleNamespace(index=4, citation="21 CFR 820.10(b)(3)", text="(3) For Clause 8.2.3 in ISO 13485, Reporting "
                            "to regulatory authorities, the manufacturer must comply with part 803 complaint reporting.")
TRACE = SimpleNamespace(index=5, citation="21 CFR 820.10(b)(2)", text="(2) Traceability procedures under part 821.")


def test_fd1_flags_a_denial_the_supplied_text_contradicts():
    hit = false_denial("Additionally, the regulations do not apply to human cells, tissues, and cellular and tissue-based "
                       "products (HCT/Ps) that are regulated as devices", [HCTP, BLOOD])
    assert hit and hit[0] == 3


@pytest.mark.parametrize("sentence", [
    "Manufacturers of blood and blood components used for transfusion are not subject to Part 820",   # the text says so
    "The provisions explicitly state that the requirements of this part do not apply to such manufacturers",  # anaphor
    "No provision requires a Chief Quality Officer to personally sign every batch record",              # NC phrasing
    "FDA may refuse to file any application that does not contain human cells data",                    # relative clause
])
def test_fd1_leaves_supported_or_unscoped_denials_alone(sentence):
    assert false_denial(sentence, [HCTP, BLOOD]) is None


def test_fd1_a_denial_about_a_named_provision_is_checked_against_it_only():
    s = "21 CFR 820.10(b)(2) [5]: Requires traceability procedures; this does not address complaint reporting to regulatory authorities"
    assert false_denial(s, [REPORTING, TRACE], cited=[TRACE]) is None
    assert false_denial("Part 820 does not address complaint reporting to regulatory authorities", [REPORTING, TRACE]) is not None


def test_fd1_the_regulations_own_exclusion_is_not_a_false_denial():
    # 30 Oct, promoting FD1: a verbatim 21 CFR 312.32 answer was failed - its quotation of the (a) definition's exclusion
    # was read as a denial of (c)(1)
    a = SimpleNamespace(index=1, citation="21 CFR 312.32(a)", text=(
        "Life-threatening adverse event or life-threatening suspected adverse reaction. ... It does not include an adverse "
        "event or suspected adverse reaction that, had it occurred in a more severe form, might have caused death."))
    c1 = SimpleNamespace(index=3, citation="21 CFR 312.32(c)(1)", text=(
        "In each IND safety report, the sponsor must identify all IND safety reports previously submitted to FDA "
        "concerning a similar suspected adverse reaction, and must analyze the significance of the suspected adverse "
        "reaction in light of previous, similar reports or any other relevant information."))
    quoted = ("It does not include an adverse event or suspected adverse reaction that, had it occurred in a more severe "
              "form, might have caused death [1].")
    assert false_denial(quoted, [a, c1]) is None
    # the answer's own denial is still checked (M-001)
    assert false_denial("Additionally, the regulations do not apply to human cells, tissues, and cellular and tissue-based "
                        "products (HCT/Ps) regulated as devices [3].", [HCTP, BLOOD]) is not None
