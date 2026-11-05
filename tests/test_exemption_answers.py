"""EQ1 (5 Nov 2026): a question asking for a regulation's exemptions is answered by quoting its frame sections verbatim
(regulus.rag.exemption_answers). The question TYPE - phrasings and look-alikes - and the composer. No database, no model."""
from dataclasses import dataclass

import pytest

from regulus.rag.exemption_answers import LEAD, asks_exemptions, compose

ASKS = [
    "Who is subject to 21 CFR Part 117 and what are the principal exemptions?",
    "Which farms and produce are covered by the produce safety rule, and what exemptions apply?",
    "What corrections and removals are exempt from reporting under Part 806 but still require records?",
    "What are the exemptions in Part 117?",
    "Which facilities are exempt from the preventive controls requirements?",
    "What exemptions apply to small businesses under Part 111?",
    "Which tobacco products are exempt from the premarket requirements?",
    "Who is excluded from the records access requirements?",
    "What produce is not covered by the produce safety rule?",
    "What are the main exemptions from the hazard analysis requirements?",
]
LOOKALIKES = [
    "How may FDA withdraw a qualified facility exemption under Part 117?",                    # procedure
    "How may a manufacturer obtain an exemption or variance from a requirement of Part 820?",
    "What is the difference between a Substantial Equivalence report, an Exemption from SE request, and a PMTA?",  # a name
    "When is an Investigational Device Exemption required and what are the significant risk criteria?",
    "Does Part 820 require ISO 13485 certification, and does a certificate exempt the manufacturer from inspection?",
    "How does a prescription device qualify for exemption from the adequate directions for use requirement?",
    "What records must a qualified facility keep?",
    "What does 21 CFR 117.5 say?",
    "Is a qualified facility subject to subpart C?",
    "What are the small-package labeling requirements in Part 101?",
    "Which records are excluded from 610(a)?",                                  # one named provision (cosmetics follow-ups)
    "What products are excluded from 'suntanning preparations'?",              # a defined term
    "What exemptions apply under 21 CFR 117.5?",
    "Which devices are exempt under section 510(m)?",
    "What is excluded from the definition of a farm?",
]


@pytest.mark.parametrize("question", ASKS)
def test_eq1_catches_the_question_type(question):
    assert asks_exemptions(question)


@pytest.mark.parametrize("question", LOOKALIKES)
def test_eq1_leaves_look_alikes(question):
    assert not asks_exemptions(question)


@dataclass
class _Src:
    index: int
    citation: str
    text: str
    heading: str = ""
    is_docket: bool = False


class _Pkg:
    def __init__(self, sources):
        self.sources = sources


SOURCES = [
    _Src(1, "21 CFR 117.260(a)", "If an order under § 117.254 is received, ...",
         "Compliance with, or revocation of, an order to withdraw a qualified facility exemption."),   # not a frame
    _Src(2, "21 CFR 117.1(a)", "(a) The criteria and definitions in this part apply in determining whether a food is adulterated.",
         "Applicability and status."),
    _Src(3, "21 CFR 117.5(a)", "(a) Except as provided by subpart E of this part, subparts C and G of this part do not apply to "
         "a qualified facility.", "Exemptions."),
    _Src(4, "21 CFR 117.5(b)", "(b) Subparts C and G of this part do not apply with respect to activities that are subject to "
         "part 123 of this chapter.", "Exemptions."),
]
Q = "Who is subject to 21 CFR Part 117 and what are the principal exemptions?"


def test_eq1_quotes_the_frame_sections_verbatim_with_their_citations():
    lines = compose(_Pkg(SOURCES), Q).split("\n")
    assert lines[0] == LEAD
    assert lines[1] == ("21 CFR 117.1(a) (Applicability and status) (eCFR, binding, verbatim): (a) The criteria and definitions "
                        "in this part apply in determining whether a food is adulterated [2].")
    assert lines[2].startswith("21 CFR 117.5(a) (Exemptions) (eCFR, binding, verbatim): (a) Except as provided") and lines[2].endswith("[3].")
    assert lines[3].startswith("21 CFR 117.5(b) (eCFR, binding, verbatim): (b) Subparts C and G") and lines[3].endswith("[4].")
    assert len(lines) == 4                     # 117.260 is not a frame section


def test_eq1_declines_without_a_section_that_states_an_exemption():
    assert compose(_Pkg(SOURCES[:2]), Q) is None            # an Applicability section alone answers no exemption


def test_eq1_declines_for_other_questions():
    assert compose(_Pkg(SOURCES), "How may FDA withdraw a qualified facility exemption under Part 117?") is None


def test_eq1_quotes_every_supplied_passage_of_a_long_section():
    # F-001 (5 Nov, blind): a 3,000-character cap cut 117.5 at (g)(1), before the on-farm exemptions the key names
    long = [_Src(10 + k, f"21 CFR 117.5({chr(97 + k)})", f"({chr(97 + k)}) " + "Subparts C and G of this part do not apply. " * 20,
                 "Exemptions.") for k in range(8)]
    lines = compose(_Pkg(long), Q).split("\n")
    assert len(lines) == 9 and lines[-1].startswith("21 CFR 117.5(h) ")


def test_eq1_leaves_out_a_list_lead_whose_items_were_not_supplied():
    # Y-001 (5 Nov): the budget ended 112.2 on "(B) Will only sell to another entity that agrees, in writing, it will either:"
    src = SOURCES + [_Src(5, "21 CFR 117.5(c)", "(c) The following activities are exempt:", "Exemptions.")]
    lines = compose(_Pkg(src), Q).split("\n")
    assert not any("117.5(c)" in line for line in lines) and lines[-1].startswith("21 CFR 117.5(b) ")


def test_eq1_cites_a_paragraph_the_quoted_text_names():
    # Y-001 (5 Nov): 112.2(b)(4) names "§ 112.2(b)(3)(i) or (ii)" - the validator asks their sources among its markers
    src = [_Src(1, "21 CFR 112.2(b)", "(b) Produce is eligible for exemption under the following conditions: x.",
                "What produce is not covered by this part?"),
           _Src(2, "21 CFR 112.2(b)(3)(i)", "(i) Annually obtain written assurance.", "What produce is not covered by this part?"),
           _Src(3, "21 CFR 112.2(b)(3)(ii)", "(ii) Annually obtain other assurance.", "What produce is not covered by this part?"),
           _Src(4, "21 CFR 112.2(b)(4)", "(4) An entity providing a written assurance under § 112.2(b)(3)(i) or (ii) must act "
                "consistently with the assurance.", "What produce is not covered by this part?")]
    lines = compose(_Pkg(src), "What produce is not covered by the produce safety rule?").split("\n")
    assert lines[-1].endswith("assurance [4][2][3].")
