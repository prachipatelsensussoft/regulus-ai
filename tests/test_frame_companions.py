"""AE1 (5 Nov 2026): a question asking who or what a regulation covers, or what it exempts, gets the sections that frame
the regulation. The question TYPE (phrasings and look-alikes) and which section headings are frames - calibrated on the
exemption headings that occur across Title 21. No database, no model."""
import pytest

from regulus.rag.section_promotion import _FRAME_SECTION, applies_asked

ASKS = [
    "Who is subject to 21 CFR Part 117 and what are the principal exemptions?",
    "Which farms and produce are covered by the produce safety rule, and what exemptions apply?",
    "Who must comply with the Foreign Supplier Verification Programs regulation?",
    "To whom does Part 11 apply?",
    "Which facilities are exempt from the preventive controls requirements?",
    "What exemptions apply to small businesses under Part 111?",
    "Are cosmetics exempt from registration under MoCRA?",
    "Who is subject to the records access requirements?",
    "What is the applicability of Part 820 to component manufacturers?",
    "Which tobacco products are exempt from substantial equivalence?",
]
DOES_NOT_ASK = [
    "What records must a food facility keep under Part 117?",
    "How long must batch records be retained?",
    "What is a qualified individual under Part 117?",
    "What must a hazard analysis include?",
    "What labeling is required for a medical device?",
    "Does FDA inspect foreign facilities?",
    "What does 21 CFR 117.130 require?",
    "Which definitions apply to Part 820?",
]
FRAMES = [
    "Exemptions.", "Exemptions and variances.", "Exemptions and alternatives.", "Scope.", "Purpose and scope.",
    "Applicability and status.", "Who is subject to this subpart?", "Who is excluded from all or part of the regulations in this subpart?",
    "To what foods do the requirements in this subpart apply?", "What food is covered by this part?",
    "Which farms are subject to the requirements of this part?",
    "Which farms are eligible for a qualified exemption and associated modified requirements based on average monetary value?",
]
NOT_FRAMES = [
    "Applicability of subparts C, D, and G of this part to a facility solely engaged in the storage of unexposed packaged food.",
    "Exemptions from required label statements.",          # one topic's exemption, not the regulation's frame
    "Exemption from premarket notification.",
    "Exemptions for device establishments.",
    "Definitions.", "Records required for this subpart.", "Personnel.",
]


@pytest.mark.parametrize("question", ASKS)
def test_ae1_catches_the_question_type(question):
    assert applies_asked(question)


@pytest.mark.parametrize("question", DOES_NOT_ASK)
def test_ae1_leaves_other_questions(question):
    assert not applies_asked(question)


@pytest.mark.parametrize("heading", FRAMES)
def test_ae1_frame_headings(heading):
    assert _FRAME_SECTION.match(heading)


@pytest.mark.parametrize("heading", NOT_FRAMES)
def test_ae1_carve_outs_and_topic_exemptions_are_not_frames(heading):
    assert not _FRAME_SECTION.match(heading)
