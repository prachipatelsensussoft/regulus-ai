"""RI1 (5 Nov 2026): a drug or device establishment that refuses, delays, denies or limits an FDA inspection - answered
from FD&C Act § 501's opening and 501(j), and for a device 21 CFR 820.1(c), verbatim. The question TYPE: phrasings that
must fire (with their product) and look-alikes that must not. No database, no model."""
from dataclasses import dataclass

import pytest

from regulus.rag.refused_inspection import asks_refused_inspection, compose

PHRASINGS = [
    ("What does Part 820 provide with respect to foreign manufacturers that refuse to permit an FDA inspection?", "device"),
    ("What happens if a foreign medical device manufacturer refuses an FDA inspection?", "device"),
    ("Is a device adulterated if its manufacturer delays an FDA inspection?", "device"),
    ("Can FDA refuse entry of devices from a plant that denied an inspection?", "device"),
    ("Under the QMSR, what is the effect of a manufacturer refusing an FDA inspection?", "device"),
    ("What is the consequence when a drug manufacturer limits an FDA inspection?", "drug"),
    ("What happens to drugs made at a facility that refuses to permit FDA inspection?", "drug"),
    ("If an API manufacturer won't allow FDA inspectors in, what does the FD&C Act provide?", "drug"),
    ("Are drugs adulterated when the establishment's inspection is delayed or limited by the owner?", "drug"),
    ("What does the FD&C Act say about a pharmaceutical plant that obstructs an FDA inspection?", "drug"),
]
LOOKALIKES = [
    "Can FDA inspect a foreign device manufacturer?",                                 # no refusal
    "How does FDA schedule inspections of drug plants?",
    "What records must a device manufacturer provide in advance of an inspection?",
    "What happens if a food facility refuses an FDA inspection?",                      # other provisions govern
    "What happens if a cosmetic manufacturer refuses an FDA inspection?",
    "What happens if a dietary supplement firm refuses inspection?",
    "Can a tobacco manufacturer deny an FDA inspection?",
    "What does a manufacturer have to do if FDA refuses its 510(k)?",
    "What happens if a manufacturer refuses an FDA inspection?",                      # no product named: not assumed
    "Which drug inspection observations go on a Form 483?",
]


@pytest.mark.parametrize("question,kind", PHRASINGS)
def test_ri1_catches_the_question_type_and_its_product(question, kind):
    assert asks_refused_inspection(question) == kind


@pytest.mark.parametrize("question", LOOKALIKES)
def test_ri1_leaves_look_alikes(question):
    assert asks_refused_inspection(question) is None


@dataclass
class _Src:
    index: int
    citation: str
    text: str
    is_docket: bool = False


class _Pkg:
    def __init__(self, sources):
        self.sources = sources


LEAD = _Src(2, "FD&C Act § 501 [21 U.S.C. 351]", "A drug or device shall be deemed to be adulterated—")
J = _Src(3, "FD&C Act § 501(j) [21 U.S.C. 351(j)]",
         "(j) Delayed, denied, or limited inspection; refusal to permit entry or inspection\nIf it is a drug or device and it "
         "has been manufactured, processed, packed, or held in any factory, warehouse, or establishment and the owner, "
         "operator, or agent of such factory, warehouse, or establishment delays, denies, or limits an inspection, or "
         "refuses to permit entry or inspection.\nFor purposes of paragraph (a)(2)(B), the term \"current good "
         "manufacturing practice\" includes the implementation of oversight and controls over the manufacture of drugs.")
REG = _Src(1, "21 CFR 820.1(c)", "(c) Foreign manufacturers. A device that is imported or offered for import into the United "
           "States is subject to refusal of admission to the United States under section 801(a) of the Federal Food, Drug, "
           "and Cosmetic Act if, among other things, it appears to be adulterated as set forth in the Federal Food, Drug, "
           "and Cosmetic Act and its implementing regulations.")
Q = "What happens if a foreign medical device manufacturer refuses an FDA inspection?"


def test_ri1_device_answer_quotes_501_its_j_and_820_1_c():
    lines = compose(_Pkg([REG, LEAD, J]), Q, "device").split("\n")
    assert lines[1] == "FD&C Act § 501 [21 U.S.C. 351] (FD&C Act, binding, verbatim): A drug or device shall be deemed to be adulterated— [2]"
    assert lines[2].startswith("FD&C Act § 501(j) [21 U.S.C. 351(j)] (FD&C Act, binding, verbatim): If it is a drug or device")
    assert lines[2].endswith("refuses to permit entry or inspection. [3]")   # 501's closing text, stored under (j), is not quoted
    assert lines[3] == "For an imported device, 21 CFR 820.1(c) provides, verbatim:"
    assert lines[4].startswith("21 CFR 820.1(c) (eCFR, binding, verbatim): (c) Foreign manufacturers.") and lines[4].endswith("[1].")


def test_ri1_drug_answer_is_the_act_text_alone():
    out = compose(_Pkg([LEAD, J]), "What happens to drugs made at a facility that refuses to permit FDA inspection?", "drug")
    assert len(out.split("\n")) == 3 and "820.1" not in out


@pytest.mark.parametrize("sources,kind", [([REG, J], "device"), ([REG, LEAD], "device"), ([LEAD, J], "device"), ([J], "drug")])
def test_ri1_declines_without_its_evidence(sources, kind):
    assert compose(_Pkg(sources), Q, kind) is None
