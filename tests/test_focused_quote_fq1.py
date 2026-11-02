from types import SimpleNamespace

from regulus.rag import validator_switches
from regulus.rag.fallback import focused_quote
from regulus.retrieval.evidence import TIER_EXACT


def _package(n):
    sources = [SimpleNamespace(is_docket=False, tier=TIER_EXACT, requirement_ids=("r1",), corpus="ecfr", index=i,
                               citation=f"21 CFR 201.1({chr(96 + i)})",
                               text=f"({chr(96 + i)}) The label must bear the name and place of business of the distributor {i}.")
               for i in range(1, n + 1)]
    return SimpleNamespace(sources=sources)


def test_the_focused_quote_keeps_three_paragraphs_by_default(monkeypatch):
    monkeypatch.setattr(validator_switches, "VALIDATOR_PRECISION", validator_switches.VALIDATOR_PRECISION - {"FQ1"})
    text = focused_quote(_package(6), "What must the label's name and place of business statement include?")
    assert text.count("verbatim):") == 3


def test_fq1_quotes_more_of_the_requested_provision(monkeypatch):
    monkeypatch.setattr(validator_switches, "VALIDATOR_PRECISION", validator_switches.VALIDATOR_PRECISION | {"FQ1"})
    text = focused_quote(_package(6), "What must the label's name and place of business statement include?")
    assert text.count("verbatim):") == 6   # G-001 (27 Oct): 201.1(a), (h)(1), (j) quoted, (b)-(g) scored as omitted


def test_ls3_reads_requirements_apply_questions_about_a_part():
    from regulus.rag.list_answers import _ASKS_REQUIREMENTS_APPLY
    assert _ASKS_REQUIREMENTS_APPLY.search("What sanitation requirements apply to the physical plant and grounds under Part 111?")
    assert not _ASKS_REQUIREMENTS_APPLY.search("Do the sanitation requirements of Part 111 apply to a retailer?")


# ------------------------------------------------------------------ LQ1: a list is quoted whole or not at all

def _rows(*rows):
    sources = [SimpleNamespace(is_docket=False, tier=TIER_EXACT, requirement_ids=("r1",), corpus="ecfr", index=i,
                               citation=f"21 CFR {c}", text=t) for i, (c, t) in enumerate(rows, 1)]
    return SimpleNamespace(sources=sources)


STREAMLINED = _rows(
    ("4.4(a)", "(a) For single entity or co-packaged combination products, compliance with current good manufacturing "
               "practice requirements shall be achieved through an operating system that is demonstrated to comply with:"),
    ("4.4(a)(1)", "(1) The specifics of each set of current good manufacturing practice regulations listed under § 4.3 "
                  "as they apply to each constituent part included in the combination product; or"),
    ("4.4(a)(2)", "(2) The provisions specified in paragraph (b) of this section."),
    ("4.4(b)", "(b) If the combination product includes a device constituent part and a drug constituent part, other "
               "provisions of the device quality system regulation must also be shown to have been satisfied."),
)


def _with(monkeypatch, *on):
    monkeypatch.setattr(validator_switches, "VALIDATOR_PRECISION",
                        (validator_switches.VALIDATOR_PRECISION - {"LQ1", "FQ1"}) | set(on))


def test_lq1_quotes_every_item_of_an_or_list(monkeypatch):
    q = "What is the streamlined approach to current good manufacturing practice for a co-packaged combination product?"
    _with(monkeypatch)
    before = focused_quote(STREAMLINED, q, limit=2)
    _with(monkeypatch, "LQ1")
    after = focused_quote(STREAMLINED, q, limit=4)
    assert "4.4(a)(2)" in after and "4.4(a)(1)" in after and "4.4(a) " in after
    assert "4.4(a)(2)" not in before          # the 29 Oct defect: "...; or" quoted without (a)(2)


def test_lq1_brings_the_items_of_a_lead_in(monkeypatch):
    pkg = _rows(("4.102(c)(1)", "(1) If your combination product contains a device constituent part, you must submit:"),
                ("4.102(c)(1)(i)", "(i) Five-day reports as required by § 803.53 of this chapter;"),
                ("4.102(c)(1)(ii)", "(ii) Malfunction reports as required by § 803.50 of this chapter; and"),
                ("4.102(c)(1)(iii)", "(iii) Correction or removal reports as required by part 806 of this chapter."),
                ("4.100(a)", "(a) This subpart identifies postmarketing safety reporting requirements for applicants."))
    _with(monkeypatch, "LQ1")
    text = focused_quote(pkg, "What must a combination product with a device constituent part submit?", limit=8)
    for p in ("(c)(1)(i)", "(c)(1)(ii)", "(c)(1)(iii)"):
        assert f"21 CFR 4.102{p} " in text


def test_lq1_skips_a_list_that_does_not_fit_rather_than_quote_its_lead_in(monkeypatch):
    long = "x " * 400
    pkg = _rows(("1.1(a)", "(a) The label of a product must state the name of the distributor of the product."),
                ("1.1(b)", "(b) The label of a product must list the following:"),
                ("1.1(b)(1)", f"(1) The distributor name {long};"),
                ("1.1(b)(2)", f"(2) The distributor place {long}."))
    _with(monkeypatch, "LQ1")
    text = focused_quote(pkg, "What must the label of a product state about the distributor?", limit=8, max_chars=900)
    assert "1.1(a) " in text and "1.1(b) " not in text and "1.1(b)(1)" not in text


def test_lq1_members_of_paragraphs_that_open_no_list():
    from regulus.rag.fallback import list_members

    members = list_members([(s, s.text) for s in STREAMLINED.sources])
    assert members[0] == {0, 1, 2} and members[1] == {0, 1, 2} and members[2] == {2} and members[3] == {3}
