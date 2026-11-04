"""Paragraph labels longer than 4 characters (4 Nov 2026, cfr migration 0011): the eCFR numbers paragraphs (xviii) ..
(xxxvii), which the database now stores ("21 CFR 175.300(b)(3)(xxviii)") instead of chaining them under the previous
label. Every regex that reads a paragraph label accepts a roman numeral of 5-8 letters besides the 1-4 character labels
it read before, and still not an ordinary word in parentheses ("(except)", "(listed)")."""
from types import SimpleNamespace

from regulus.rag import answer_plan, answerability, claim_validation, conversation, selection, shape_validation


def test_an_evidence_citation_with_a_long_label_is_a_provision():
    assert selection._identity(SimpleNamespace(citation="21 CFR 117.5(g)(3)(xxiii)")) == ("117.5", "(g)(3)(xxiii)")
    assert selection._identity(SimpleNamespace(citation="21 CFR 117.5(g)(3)(xvii)")) == ("117.5", "(g)(3)(xvii)")
    assert selection._identity(SimpleNamespace(citation="21 CFR 117.5(except)")) is None


def test_cross_references_and_reserved_paragraphs_read_long_labels_but_not_words():
    assert selection._REF_TOKEN_RE.findall("paragraphs (b)(3)(xxviii) and (xxxiii), except (listed) items") == [
        "b", "3", "xxviii", "xxxiii"]
    assert selection._RESERVED_RE.match("(xxviii) [Reserved]")
    assert not selection._RESERVED_RE.match("(except) [Reserved]")


def test_answer_text_labels_with_long_numerals_are_recognised():
    assert claim_validation._LABEL.sub("", "21 CFR 175.300(b)(3)(xxviii): Silicones") == " Silicones"
    assert shape_validation._LEAD_LABEL.match("- 21 CFR 117.5(g)(3)(xxiii) exempts").group(1) == "(g)(3)(xxiii)"
    assert shape_validation._LABEL_RE.findall("117.5(g)(3)(xxiii) and (except)") == ["117.5(g)(3)(xxiii)"]
    assert answer_plan._TOKEN_RE.findall("(xxxiii) Miscellaneous (except) (A)") == ["xxxiii", "A"]
    assert answer_plan._HEADING_RE.match("(xxiii) Waxes. More").groups() == ("xxiii", "Waxes")


def test_a_question_pointing_at_a_long_label_paragraph():
    assert conversation._PARAGRAPH_POINTER.findall("what does paragraph (b)(3)(xxviii) say? and paragraph (except) one") == [
        "(b)(3)(xxviii)"]
    assert answerability._AMBIGUOUS_RE.search("what does paragraph (xviii) require")
