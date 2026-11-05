"""GP1 (5 Nov 2026): "which FDA guidance requires X" is re-asked as "which regulation requires X" and led by the premise
note from 21 CFR 10.115(d)(1); a device question naming a quality-system subject in words is routed to its former Part 820
section for QM1 (regulus.rag.qmsr.former_by_subject). The question TYPE, not one benchmark wording: phrasings that must
fire and look-alikes that must not. No database (a stub), no model."""
import pytest

from regulus.rag.guidance_premise import PREMISE_LEAD, premise_note, premise_rewrite
from regulus.rag.qmsr import former_by_subject

PREMISE_PHRASINGS = [
    ("Which FDA guidance requires a device manufacturer to qualify its suppliers?",
     "Which regulation requires a device manufacturer to qualify its suppliers?"),
    ("What guidance document requires drug manufacturers to have a stability testing program?",
     "What regulation requires drug manufacturers to have a stability testing program?"),
    ("Which FDA guidance mandates written food safety plans for food facilities?",
     "Which regulation mandates written food safety plans for food facilities?"),
    ("Under which FDA guidance must device makers maintain complaint files?",
     "Under which regulation must device makers maintain complaint files?"),
    ("Which guidance obligates cosmetic manufacturers to list ingredients on the label?",
     "Which regulation obligates cosmetic manufacturers to list ingredients on the label?"),
    ("What FDA guidance makes it mandatory to validate sterilization processes for devices?",
     "What regulation makes it mandatory to validate sterilization processes for devices?"),
    ("Which FDA guidance document imposes the requirement for a written stability program?",
     "Which regulation imposes the requirement for a written stability program?"),
    ("According to which FDA guidance are manufacturers required to qualify suppliers?",
     "According to which regulation are manufacturers required to qualify suppliers?"),
    ("Which FDA guidance says device manufacturers must have CAPA procedures?",
     "Which regulation says device manufacturers must have CAPA procedures?"),
    ("Which of FDA's guidance documents requires 510(k) holders to report changes?",
     "Which regulation requires 510(k) holders to report changes?"),
]
PREMISE_LOOKALIKES = [
    "Which FDA guidance covers supplier qualification for devices?",       # asks for a guidance document: it exists
    "What does FDA's guidance on process validation recommend?",
    "Which regulation requires a written stability testing program?",
    "Is there FDA guidance on cybersecurity for medical devices?",
    "What does the 510(k) guidance require in a submission?",              # a named guidance's content
    "Which guidance documents does FDA list for 510(k) submissions?",
    "What guidance has FDA issued on supplier controls?",
    "Which FDA guidance describes how to qualify suppliers?",
    "Does FDA guidance on sterilization apply to reusable devices?",
    "What does FDA recommend in its guidance on human factors?",
]


@pytest.mark.parametrize("question,rewritten", PREMISE_PHRASINGS)
def test_gp1_reasks_the_guidance_premise_as_the_regulation(question, rewritten):
    assert premise_rewrite(question) == rewritten


@pytest.mark.parametrize("question", PREMISE_LOOKALIKES)
def test_gp1_leaves_questions_that_ask_for_a_guidance_document(question):
    assert premise_rewrite(question) is None


class _DB:
    def __init__(self, body):
        self.body = body

    def query(self, sql, params):
        assert params == {"s": "10.115", "p": "(d)"}
        return [{"body_text": self.body}] if self.body is not None else []


D_TEXT = ("(d) Are you or FDA required to follow a guidance document? (1) No. Guidance documents do not establish legally "
          "enforceable rights or responsibilities. They do not legally bind the public or FDA.")


def test_gp1_note_quotes_10_115_d_1_as_the_database_holds_it():
    assert premise_note(_DB(D_TEXT)) == PREMISE_LEAD.format(
        quote="Guidance documents do not establish legally enforceable rights or responsibilities. "
              "They do not legally bind the public or FDA.")


@pytest.mark.parametrize("body", [None, "(d) Are you or FDA required to follow a guidance document?", ""])
def test_gp1_note_is_left_out_without_that_text(body):
    assert premise_note(_DB(body)) is None
    assert premise_note(None) is None


LIVE_820 = {"820.1", "820.3", "820.7", "820.10", "820.35", "820.45"}
SUBJECTS = [
    ("Which regulation requires a device manufacturer to qualify its suppliers?", [("820.50", "(a)")]),
    ("Under which regulation must device makers maintain complaint files?", [("820.198", "")]),
    ("Which regulation says device manufacturers must have CAPA procedures?", [("820.100", "")]),
    ("Which regulation requires device manufacturers to train their personnel?", [("820.25", "")]),
    ("What regulation makes it mandatory to validate sterilization processes for devices?", [("820.75", "")]),
    ("Which regulation mandates design controls for medical devices?", [("820.30", "")]),
    ("Which regulation requires device manufacturers to calibrate test equipment?", [("820.72", "")]),
    ("Which regulation requires device makers to control nonconforming product?", [("820.90", "")]),
    ("Which regulation requires device manufacturers to perform quality audits?", [("820.22", "")]),
    ("Which regulation requires device manufacturers to have management review?", [("820.20", "")]),
    # premarket, other domains, or no single subject: left to the normal pipeline
    ("Which regulation requires 510(k) holders to report changes?", []),
    ("Which regulation requires device establishments to register with FDA?", []),
    ("Which regulation requires drug manufacturers to have a written stability testing program?", []),
    ("Which regulation requires food facilities to verify their suppliers?", []),
    ("Which regulation requires a device manufacturer to keep records?", []),
]


@pytest.mark.parametrize("question,expected", SUBJECTS)
def test_gp1_device_subject_routes_to_the_former_section(question, expected):
    assert former_by_subject(question, LIVE_820) == expected
