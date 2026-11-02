"""CN1 (2 Nov 2026): citations written as provision labels ("[4.2(a)]") are normalised to [n] markers before validation."""
from types import SimpleNamespace

from regulus.rag.validation import _label_citations, resolve_label_citations


def _src(i, section, paragraph, text):
    return SimpleNamespace(index=i, corpus="ecfr", citation=f"21 CFR {section}{paragraph}", text=text,
                           citation_fields={"title": "21", "section": section, "paragraph": paragraph})


PKG = SimpleNamespace(sources=[
    _src(1, "4.1", "", "This subpart applies to combination products and establishes which current good manufacturing "
                       "practice requirements apply to these products."),
    _src(2, "4.2", "", "Combination product has the meaning set forth in § 3.2(e) of this chapter. Constituent part is a "
                       "drug, device, or biological product that is part of a combination product."),
    _src(3, "4.3", "(a)", "The current good manufacturing practice requirements in parts 210 and 211 of this chapter apply "
                          "to a combination product that includes a drug constituent part other than a medical gas."),
    _src(4, "4.3", "(b)", "The current good manufacturing practice requirements in part 820 of this chapter apply to a "
                          "combination product that includes a device constituent part."),
])


def test_labels_read_as_provisions():
    assert _label_citations("4.3(a), (b), (e)") == ["21 CFR 4.3(a)", "21 CFR 4.3(b)", "21 CFR 4.3(e)"]
    assert _label_citations("21 CFR 4.2") == ["21 CFR 4.2"]
    assert _label_citations("§ 4.3(a)(1), (2)") == ["21 CFR 4.3(a)(1)", "21 CFR 4.3(a)(2)"]


def test_a_label_naming_a_supplied_provision_becomes_its_marker():
    text, n = resolve_label_citations("Constituent part means a drug, device, or biological product [4.2]. Device "
                                      "constituent parts follow part 820 [4.3(a), (b)].", PKG)
    assert n == 2 and "[2]" in text and "[3][4]" in text and "[4." not in text


def test_a_label_naming_an_absent_paragraph_points_at_the_source_of_its_words():
    # X-001 (2 Nov): 4.2 has no paragraph (a); the sentence is 4.3(a)'s - provision_citation_mismatch then sees it
    text, n = resolve_label_citations("21 CFR 4.2 defines that a drug constituent part other than a medical gas is "
                                      "subject to the requirements in parts 210 and 211 [4.2(a)].", PKG)
    assert n == 1 and text.endswith("parts 210 and 211 [3].")


def test_a_label_with_no_supporting_source_is_dropped():
    text, n = resolve_label_citations("Records must be kept for seven years [4.9(c)].", PKG)
    assert n == 1 and text == "Records must be kept for seven years."


def test_markers_and_plain_text_are_untouched():
    text = "The requirements in part 820 apply to a device constituent part [4]. See 21 CFR 4.3(b) for details."
    assert resolve_label_citations(text, PKG) == (text, 0)


def _act(i, path, text):
    return SimpleNamespace(index=i, corpus="usc", citation=f"FD&C Act § 905{path} [21 U.S.C. 387e{path}]", text=text,
                           citation_fields={})


ACT = SimpleNamespace(sources=[
    _src(1, "1100.200", "", "This subpart sets out definitions for tobacco products."),
    _act(2, "(c)", "Every person who first engages in the manufacture of a tobacco product shall register."),
    _act(3, "(d)", "Every person required to register shall immediately register any additional establishment."),
])


def test_act_labels_become_markers():
    # T-007 (2 Nov): "[21 U.S.C. 387e(c)]" and the typo "[21.U.S.C. 387e(d)]" were never counted as citations
    text, n = resolve_label_citations("A manufacturer must register [21 U.S.C. 387e(c)]. Any additional establishment "
                                      "is registered too [21.U.S.C. 387e(d)]. See also [FD&C Act § 905(c)].", ACT)
    assert n == 3 and text == ("A manufacturer must register [2]. Any additional establishment is registered too [3]. "
                               "See also [2].")


def test_a_supplied_sources_own_label_is_not_a_citation():
    text = "FD&C Act § 905(c) [21 U.S.C. 387e(c)] requires registration [2]."
    assert resolve_label_citations(text, ACT) == (text, 0)
