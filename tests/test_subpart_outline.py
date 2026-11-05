"""SP4O (5 Nov 2026): a pure overview question about one large subpart is answered with the subpart's outline. The gate
(regulus.rag.section_promotion.subpart_overview) decides the question TYPE - the subpart as a whole and nothing narrower -
and the composer (regulus.rag.subpart_outline) quotes each section's supplied passages. No database, no model."""
from dataclasses import dataclass

import pytest

from regulus.rag.section_promotion import SUBPART_OUTLINE_NOTE, subpart_overview
from regulus.rag.subpart_outline import compose, outline_request

P211 = "Part 211 - Current Good Manufacturing Practice for Finished Pharmaceuticals"
P50 = "Part 50 - Protection of Human Subjects"
P112 = "Part 112 - Standards for the Growing, Harvesting, Packing, and Holding of Produce for Human Consumption"
P507 = "Part 507 - Current Good Manufacturing Practice, Hazard Analysis, and Risk-Based Preventive Controls for Food for Animals"
P4 = "Part 4 - Regulation of Combination Products"
CHILDREN = "Additional Safeguards for Children in Clinical Investigations"

OVERVIEWS = [   # the subpart as a whole: outlined
    ("Current Good Manufacturing Practice", P507, ["Personnel.", "Plant and grounds."],
     "What current good manufacturing practice requirements apply to the manufacturing, processing, packing, and holding of animal food?"),
    (CHILDREN, P50, ["IRB duties.", "Wards."], "What additional safeguards apply to children enrolled in a clinical investigation of a drug or device?"),
    ("Agricultural Water", P112, ["Agricultural water: general."], "What requirements apply to agricultural water under the produce safety rule?"),
    ("Postmarketing Safety Reporting for Combination Products", P4, ["What is the scope of this subpart?"],
     "What postmarket safety reporting requirements apply to a combination product applicant?"),
    # held out (not among the questions the rule was fit on)
    ("Production and Process Controls", P211, ["Written procedures; deviations."], "What production and process control requirements apply under Part 211?"),
    ("Packaging and Labeling Control", P211, ["Labeling issuance."], "What packaging and labeling control requirements apply to drug products?"),
    ("Agricultural Water", P112, ["Agricultural water: general."], "What agricultural water requirements apply to farms?"),
    ("Informed Consent of Human Subjects", P50, ["General requirements for informed consent."],
     "What informed consent requirements apply to human subjects in clinical investigations?"),
]
NARROWER = [   # one section, or something the subpart title does not cover: not outlined
    ("Laboratory Controls", P211, ["General requirements.", "Testing and release for distribution."],
     "What are the general laboratory control requirements under Part 211?"),                     # 211.160 (D-019, PASS)
    ("Records and Reports", P211, ["General requirements.", "Equipment cleaning and use log."],
     "What are the general requirements for records and reports under Part 211?"),                # 211.180 (D-024, PASS)
    ("Buildings and Facilities", P211, ["Design and construction features."],
     "What building and facility requirements apply to aseptic processing areas?"),               # 211.42 (D-005, PASS)
    ("Warning Statements", "Part 740 - Cosmetic Product Warning Statements", ["Products for which warning statements are required."],
     "What warning statement requirements apply to cosmetics whose safety has not been substantiated?"),   # 740.10 (C-003)
    ("Supply-Chain Program", P507, ["Requirement to establish and implement a supply-chain program."],
     "What supply-chain program requirements apply to an animal food facility, and when is the receiving facility responsible?"),
    (CHILDREN, P50, ["IRB duties."], "What must an IRB find before approving research involving children that presents greater than minimal risk?"),
    ("Agricultural Water", P112, ["Agricultural water: general."], "How often must agricultural water be tested under the produce safety rule?"),
    ("Holding and Distribution", P211, ["Warehousing procedures."], "What are the warehousing procedures required for drug products?"),
]


@pytest.mark.parametrize("title,part_title,headings,question", OVERVIEWS)
def test_sp4o_outlines_a_pure_overview_of_the_subpart(title, part_title, headings, question):
    assert subpart_overview(title, part_title, headings, question)


@pytest.mark.parametrize("title,part_title,headings,question", NARROWER)
def test_sp4o_leaves_a_question_that_narrows_to_one_section(title, part_title, headings, question):
    assert not subpart_overview(title, part_title, headings, question)


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
    _Src(1, "21 CFR 112.41", "All agricultural water must be safe and of adequate sanitary quality for its intended use.",
         "What requirements apply to the quality of agricultural water?"),
    _Src(2, "21 CFR 112.44(a)", "(a) You must test the quality of water as follows:", "What specific microbial quality criteria apply?"),
    _Src(3, "21 CFR 112.44(a)(1)", "(1) There must be no detectable generic E. coli in 100 mL of agricultural water."),
    _Src(4, "21 CFR 112.47(a)", "(a) You may meet the requirements related to agricultural water testing required under §§ "
         "112.43(c)(4)(ii) and 112.44 using:", "Who must perform the tests required under this subpart?"),
]
MEMBERS = ["112.41", "112.44", "112.47"]


def test_sp4o_outline_quotes_each_section_with_its_items_and_cross_references():
    lines = compose(_Pkg(SOURCES), "What requirements apply to agricultural water?", "Agricultural Water", MEMBERS).split("\n")
    assert lines[0].startswith("The Agricultural Water subpart of 21 CFR Part 112 sets out these requirements section by section")
    assert lines[1].startswith("21 CFR 112.41 (What requirements apply to the quality of agricultural water?) (eCFR, binding, verbatim): All")
    assert lines[2] == "21 CFR 112.44(a) (What specific microbial quality criteria apply?) (eCFR, binding, verbatim): (a) You must test the quality of water as follows: [2]."
    assert lines[3].startswith("21 CFR 112.44(a)(1) (eCFR, binding, verbatim): (1) There must be")   # the lead-in's item follows it
    assert lines[4].endswith("112.43(c)(4)(ii) and 112.44 using: [4][2].")                               # 112.44 cited to its own text


def test_sp4o_outline_declines_when_too_few_sections_are_supplied():
    assert compose(_Pkg(SOURCES[:1]), "What requirements apply to agricultural water?", "Agricultural Water", MEMBERS) is None


def test_sp4o_outline_request_reads_the_promotion_note():
    class _Pr:
        notes = ("reranked: 112.42", f"{SUBPART_OUTLINE_NOTE}: Agricultural Water: 112.41, 112.44, 112.47")
    assert outline_request(_Pr()) == ("Agricultural Water", MEMBERS)
