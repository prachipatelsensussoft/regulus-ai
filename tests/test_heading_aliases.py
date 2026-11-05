"""SY4H (5 Nov 2026): an alias whose expansion is too common in body text to search on is matched against its home part's
headings instead (regulus.retrieval.acronyms.heading_aliases). No database, no model."""
import pytest

from regulus.retrieval.acronyms import ALIAS_HOMES, heading_aliases

PREMARKET = [("premarket notification", ("807",))]


@pytest.mark.parametrize("question", [
    "When must a 510(k) be submitted for a change to a legally marketed device?",
    "Is a 510 (k) required for a modified device?",
])
def test_sy4h_510k_is_matched_against_part_807_headings(question):
    assert heading_aliases(question) == PREMARKET


@pytest.mark.parametrize("question", [
    "What is required in a premarket notification 510(k) submission?",   # already spelled out
    "What must a PMA application contain under Part 814?",                # 814's headings say "PMA" itself
    "What are the FSVP recordkeeping requirements?",                      # specific alias: SY4 searches it in the text
    "What does section 510 of the Act require?",                          # not "510(k)"
])
def test_sy4h_leaves_other_questions(question):
    assert heading_aliases(question) == []


def test_sy4h_homes_only_where_the_headings_use_the_expansion():
    assert ALIAS_HOMES == {"premarket notification": ("807",)}
