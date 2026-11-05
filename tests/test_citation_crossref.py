"""CM3 (5 Nov 2026): the citation_mismatch check does not fire when the cited paragraph's own text names the provision the
sentence mentions - the sentence restates the cited paragraph's cross-reference (CM1's rule for provision_citation_mismatch,
extended). No database, no model."""
import pytest

from regulus.rag.validation import _cited_text_names


@pytest.mark.parametrize("label,texts", [
    ("21 CFR 117.254", ["If an entity receives an order under § 117.254 to withdraw a qualified facility exemption, it must"]),
    ("21 CFR 112.44", ["(a) You may meet the requirements related to agricultural water testing required under §§ 112.43(c)(4)(ii) and 112.44 using:"]),
    ("21 CFR 806.20", ["unrelated text", "Records shall be kept as required by § 806.20(b)."]),
])
def test_cm3_cited_paragraph_names_the_provision(label, texts):
    assert _cited_text_names(label, texts)


@pytest.mark.parametrize("label,texts", [
    ("21 CFR 117.254", ["If an entity receives an order under § 117.2540 ..."]),   # another number
    ("21 CFR 117.254", ["... under § 1117.254 ..."]),
    ("21 CFR 117.254", ["Subparts C and G of this part do not apply."]),
    ("Part 117", ["§ 117.254"]),                                                     # no section in the label
])
def test_cm3_does_not_excuse_a_real_mis_citation(label, texts):
    assert not _cited_text_names(label, texts)
