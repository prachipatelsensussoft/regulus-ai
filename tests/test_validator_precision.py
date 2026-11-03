"""Validator and answer fixes A1-A4, G1, E1, S1, C82 (23 Oct 2026, regulus.rag.validator_switches): each changes only the
answers it targets. No database, no model."""
import re
from types import SimpleNamespace

from regulus.rag import validator_switches
from regulus.rag.claim_validation import absence_supported, scoped_absence
from regulus.rag.generation import PackagedSource
from regulus.rag.validation import (ValidationProblem, ValidationReport, _figures, _strip_supplied_citations,
                                    excise_unsupported_sentences, repoint_named_citations)


def _src(index, section, paragraph, text, tier="exact"):
    return PackagedSource(index=index, tier=tier, requirement_ids=("r1",), protected=True, corpus="ecfr",
                          citation=f"21 CFR {section}{paragraph}", heading=None, url=None, text=text, truncated=False,
                          evidence_id=f"e{index}", key=f"k{index}",
                          citation_fields={"title": "21", "section": section, "paragraph": paragraph})


def _report(*problems):
    return ValidationReport(status="unsupported", problems=tuple(problems), addressed=(), unaddressed=(), cited=(),
                            invalid_citations=(), truncated=False, finished=True, generated_tokens=None,
                            max_new_tokens=None, answer_class="answer")


def _unsupported(sentence):
    return ValidationProblem("unsupported_claim", "error", "only 10% of its content words appear in any supplied source",
                             sentence=sentence)


# ------------------------------------------------------------ switches


def test_the_gated_fixes_are_on_by_default():
    # every fix that passed the exact replay and a blind A/B; SA1 and ST1 stay off (27 Oct); SP2 joined on 2 Nov only
    # together with SP2F, the floor that removed its lost PASS answers; OC1 was rejected (30 Oct)
    assert validator_switches.VALIDATOR_PRECISION == frozenset(validator_switches.DEFAULT.split(",")) or \
        "REGULUS_VALIDATOR_PRECISION" in __import__("os").environ
    default = set(validator_switches.DEFAULT.split(","))
    assert not {"SA1", "ST1", "OC1"} & default
    assert "SP2" not in default or "SP2F" in default


# ------------------------------------------------------------ A1 citations are not figures


def test_a1_a_supplied_citation_is_not_a_figure_but_an_invented_one_still_is():
    supplied = "shall conform to the requirements of § 74.705 (a)(1) and (b) of this chapter; FD&C Act § 605(e)"
    assert _figures("conform to §74.705(a)(1) and (b) [1].") == {"74"}                  # the old false figure
    assert _figures(_strip_supplied_citations("conform to §74.705(a)(1) and (b) [1].", supplied)) == set()
    assert _figures(_strip_supplied_citations("conform to § 74.799(a) [1].", supplied)) == {"74"}
    assert _figures(_strip_supplied_citations("keep records 6 years under § 605(e) [2]", supplied)) == set()


def test_a1_reads_only_cfr_citations_as_citations():
    # FD&C Act section numbers stay figures until the misstatement checks ship: the false figure 605 was the only
    # finding on a misleading draft (FDCA605-Q1, blind A/B 23 Oct), which must stay withheld
    supplied = "FD&C Act § 605(e) Each responsible person shall maintain records; § 74.1328 (a)(1)"
    assert _figures(_strip_supplied_citations("medical records beyond what is specified in § 605.", supplied)) == {"605"}
    assert _figures(_strip_supplied_citations("conform to §74.1328(a)(1) [1].", supplied)) == set()


def test_a1_never_touches_amounts():
    assert _figures(_strip_supplied_citations("the fee is $75 per petition", "§ 70.19 fees")) == {"75"}
    assert _figures(_strip_supplied_citations("not more than 0.6 percent", "not more than 0.5 percent")) == set()


# ------------------------------------------------------------ A2 scoped absence statements

GMP_USE = ("(c) Uses and restrictions. Disodium EDTA-copper may be safely used in amounts consistent with good "
           "manufacturing practices in the coloring of shampoos which are cosmetics.")
SPEC = "Free copper, not more than 100 parts per million. Lead (as Pb), not more than 20 parts per million."


def test_a2_scoped_absence_is_recognized_and_unscoped_denial_is_not():
    assert scoped_absence("These provisions set no limit on the amount of guanine that may be used.")
    assert scoped_absence("21 CFR 73.2120 does not specify any maximum concentration.")
    assert scoped_absence("No specific amount limit is stated in 21 CFR 82.1330.")
    assert not scoped_absence("There is no limit on how much guanine you can use.")
    assert not scoped_absence("FDA does not regulate this product.")


def test_a2_checks_the_kind_of_limit_against_the_uses_text():
    s = "These provisions impose no specific limits on the amount of disodium EDTA-copper that can be added."
    assert absence_supported(s, [SPEC, GMP_USE])                       # purity limits are not use amounts
    assert not absence_supported(s, [SPEC, "(c) It may be used in an amount not to exceed 0.5 percent by weight."])
    assert not absence_supported(s, ["(c) It may be safely used subject to the restrictions in paragraph (b)."])
    assert not absence_supported(s, [SPEC])                             # no uses text: nothing to verify against
    assert not absence_supported("These provisions say nothing more.", [GMP_USE])   # no kind of limit named


# ------------------------------------------------------------ A3 excision of unsupported filler

DRAFT = ("Disodium EDTA-copper may be used to tint shampoos, but it is not permitted in body washes [1]. "
         "The use in shampoos is limited to amounts consistent with good manufacturing practice [1]. "
         "The labeling must conform to the requirements of section 70.25 [2]. "
         "The use in body washes is not addressed in any provision, so it is not permitted under current regulations.")
QUESTION = "Can we use disodium EDTA-copper to tint a body wash and a shampoo?"


def test_a3_drops_an_uncited_unsupported_sentence_whose_point_is_already_made():
    filler = "The use in body washes is not addressed in any provision, so it is not permitted under current regulations."
    out = excise_unsupported_sentences(DRAFT, _report(_unsupported(filler)), QUESTION)
    assert out is not None and filler not in out[0] and out[1] == 1
    assert out[0].startswith("Disodium EDTA-copper may be used to tint shampoos")


def test_a3_never_drops_the_verdict_sentence_or_a_cited_claim():
    first = "Disodium EDTA-copper may be used to tint shampoos, but it is not permitted in body washes [1]."
    assert excise_unsupported_sentences(DRAFT, _report(_unsupported(first)), QUESTION) is None
    cited = "The labeling must conform to the requirements of section 70.25 [2]."
    assert excise_unsupported_sentences(DRAFT, _report(_unsupported(cited)), QUESTION) is None


def test_a3_keeps_the_only_sentence_that_answers_part_of_the_question():
    draft = ("Records must be kept for two years after creation [1]. They must be available to FDA for inspection [1]. "
             "No provision requires holding lots in quarantine until verified.")
    q = "What records must we keep, and must we hold lots in quarantine until verification?"
    only = "No provision requires holding lots in quarantine until verified."
    assert excise_unsupported_sentences(draft, _report(_unsupported(only)), q) is None


def test_a3_leaves_a_draft_whose_unsupported_sentences_are_its_answer():
    draft = ("21 CFR 700.27 governs prohibited cattle materials in cosmetics [1]. "
             "None of these provisions address contamination by rodent droppings. "
             "The recycled resin is not governed by any requirement in this section. "
             "Rodent droppings may still raise concerns.")
    probs = [_unsupported(s) for s in ("None of these provisions address contamination by rodent droppings.",
                                       "The recycled resin is not governed by any requirement in this section.",
                                       "Rodent droppings may still raise concerns.")]
    assert excise_unsupported_sentences(draft, _report(*probs), "Which provisions cover rodent droppings and resin?") is None


def test_a3_needs_every_error_to_be_excisable():
    filler = "The use in body washes is not addressed in any provision, so it is not permitted under current regulations."
    fig = ValidationProblem("unsupported_figure", "error", "figure(s) not found in any supplied source: 74")
    assert excise_unsupported_sentences(DRAFT, _report(_unsupported(filler), fig), QUESTION) is None


# ------------------------------------------------------------ A4 citation numbers re-pointed to the named provision


def test_a4_repoints_a_marker_to_the_one_source_that_is_the_named_provision():
    pkg = SimpleNamespace(sources=(
        _src(1, "71.30", "(a)", "(a) Objections and hearings relating to color additive regulations shall be governed by parts 10 and 12."),
        _src(2, "71.30", "(b)", "(b) The fees specified in § 70.19 of this chapter shall be applicable.")))
    text = "21 CFR 71.30(b) requires that the fees specified in §70.19 be applicable when filing an objection to a color additive regulation [1]."
    out, n = repoint_named_citations(text, pkg)
    assert n == 1 and out.endswith("regulation [2].")


def test_a4_leaves_a_correct_or_uncertain_citation_alone():
    pkg = SimpleNamespace(sources=(
        _src(1, "71.30", "(a)", "(a) Objections and hearings relating to color additive regulations shall be governed by parts 10 and 12."),
        _src(2, "71.30", "(b)", "(b) The fees specified in § 70.19 of this chapter shall be applicable.")))
    right = "21 CFR 71.30(b) makes the fees in §70.19 applicable [2]."
    assert repoint_named_citations(right, pkg) == (right, 0)
    unrelated = "21 CFR 71.30(b) says that cosmetics must carry a warning statement about aerosols and flammability [1]."
    assert repoint_named_citations(unrelated, pkg) == (unrelated, 0)      # the words are not the named provision's


def test_a2_does_not_vouch_for_a_missing_deadline():
    # a deadline usually lives in another section than the one cited (P-010: 812.40 cited, 812.46 sets the timeframe)
    s = "These provisions set no specific timeframes for monitoring the investigation."
    assert scoped_absence(s) and not absence_supported(s, ["(a) Sponsors shall select qualified investigators."])


def test_a3_never_trims_a_cut_off_draft():
    filler = "The use in body washes is not addressed in any provision, so it is not permitted under current regulations."
    rep = _report(_unsupported(filler))
    cut = ValidationReport(**{**rep.__dict__, "truncated": True})
    unfinished = ValidationReport(**{**rep.__dict__, "finished": False})
    assert excise_unsupported_sentences(DRAFT, cut, QUESTION) is None
    assert excise_unsupported_sentences(DRAFT, unfinished, QUESTION) is None


def test_a3_leaves_a_draft_whose_verdict_says_the_evidence_does_not_govern():
    draft = ("A retailer cannot establish compliance under 21 CFR 700.27 because this regulation does not govern talc "
             "storage [1]. Records must be retained for 2 years [2]. Storage rules are not part of these records.")
    only = "Storage rules are not part of these records."
    assert excise_unsupported_sentences(draft, _report(_unsupported(only)), "What storage records prove our talc compliance?") is None


# ------------------------------------------------------------ G1 quotations survive duplicate removal

CAUTION = ("Caution—Use only as directed. Excessive use or prolonged exposure may cause irritation to skin and urinary "
           "tract. Discontinue use if rash, redness, or itching occurs. Consult your physician if irritation persists. "
           "Keep out of reach of children.")


def test_g1_a_quotation_repeated_from_its_source_keeps_every_sentence():
    from regulus.inference.repetition_text import drop_duplicate_sentences
    draft = ('The label must include the caution: "Use only as directed. Excessive use or prolonged exposure may cause '
             'irritation to skin and urinary tract. Discontinue use if rash, redness, or itching occurs. Consult your '
             'physician if irritation persists." [2]\n\nThe label must bear the following caution:\n' + CAUTION[:-1] + " [3].")
    cut, n = drop_duplicate_sentences(draft)                     # without the sources: the old behaviour
    assert n >= 2 and "Excessive use" not in cut.split("\n")[-1]
    kept, m = drop_duplicate_sentences(draft, [CAUTION])
    assert m == 0 and kept == draft


def test_g1_an_ordinary_repeat_is_still_removed():
    from regulus.inference.repetition_text import drop_duplicate_sentences
    text = ("Records must be kept for two years [1]. They must be available to FDA [1].\n"
            "Records must be kept for two years [1]. Lots need no quarantine [2].")
    assert drop_duplicate_sentences(text, [CAUTION]) == (
        "Records must be kept for two years [1]. They must be available to FDA [1].\nLots need no quarantine [2].", 1)


# ------------------------------------------------------------ E1 obligation strength


def test_e1_flags_a_duty_its_source_only_requests_or_recommends():
    from regulus.rag.claim_validation import strengthened_obligation
    requested = "(a) The manufacturer, packer, or distributor is requested to file Form FDA 2512 for each cosmetic product."
    assert strengthened_obligation("21 CFR 720.1 requires the manufacturer, packer, or distributor to file Form FDA 2512 "
                                   "for each cosmetic product [1].", [requested])
    should = "Changes in ingredients or brand name should be shown by filing an amended Form FDA 2512 within 60 days."
    assert strengthened_obligation("An amended Form FDA 2512 must be filed within 60 days of a brand name change [1].", [should])


def test_e1_leaves_conditions_of_a_permission_and_real_duties_alone():
    from regulus.rag.claim_validation import strengthened_obligation
    may_if = ("A batch of a straight color listed in this subpart may be certified for use only in drugs and cosmetics "
              "if such batch conforms to the requirements of 82.5.")
    assert strengthened_obligation("21 CFR 82.1050 requires that a batch conform to the requirements of 82.5 to be "
                                   "certified for drugs and cosmetics [1].", [may_if]) is None
    shall = "The label shall bear adequate directions for safe use and the following caution."
    assert strengthened_obligation("The label must bear adequate directions for safe use and the caution [1].", [shall]) is None
    assert strengthened_obligation("Objections or requests for a hearing require a filing fee of $250 [1].",
                                   ["Objections and requests for public hearing shall be accompanied by a fee of $250."]) is None


# ------------------------------------------------------------ S1 status notes


def test_s1_a_note_follows_only_the_provisions_the_answer_cites():
    from regulus.rag.status_notes import status_notes
    srcs = (_src(1, "710.3", "", "(a) Form FD-2511 ... is obtainable on request."),
            _src(2, "740.18", "(a)", "(a) The principal display panel ... shall bear ..."),
            _src(3, "701.3", "(a)", "(a) The label shall bear a declaration of ingredients."))
    assert [n["id"] for n in status_notes("Use Form FD-2511 [1].", srcs)] == ["vcrp-closed"]
    assert [n["id"] for n in status_notes("The warning applies [2] and ingredients are declared [3].", srcs)] == ["740.18-stayed"]
    assert status_notes("Ingredients are declared [3].", srcs) == []
    assert status_notes("Use Form FD-2511.", srcs) == []           # nothing cited, nothing appended
    act = SimpleNamespace(index=4, citation="FD&C Act § 606(a) [21 U.S.C. 364b(a)]", heading=None, corpus="usc")
    assert [n["id"] for n in status_notes("FDA shall issue GMP rules [4].", (act,))] == ["fdca606-gmp-not-issued"]


# ------------------------------------------------------------ C82 Part 82 conformity listings composed from their text

RED28 = "The color additive D&C Red No. 28 shall conform in identity and specifications to the requirements of § 74.1328 (a)(1) and (b) of this chapter."
GEN_C = ("A batch of a straight color listed in this subpart may be certified, in accordance with the provisions of this "
         "part, for use only in drugs and cosmetics, if such batch conforms to the requirements of § 82.5 and to the "
         "specifications set forth in this subpart for such color.")


def _pkg82(*sources, requested=("21 CFR 82.1328", "21 CFR 82.1050")):
    srcs = []
    for i, (section, text) in enumerate(sources, 1):
        s = _src(i, section, "", text)
        srcs.append(PackagedSource(**{**s.__dict__, "heading": "D&C Red No. 28." if section == "82.1328" else "General."}))
    return SimpleNamespace(sources=tuple(srcs), requirements=tuple((f"r{i}", r) for i, r in enumerate(requested, 1)))


def test_c82_composes_a_conformity_listing_from_its_text():
    from regulus.rag.color_listing import compose_certification
    out = compose_certification(_pkg82(("82.1328", RED28), ("82.1050", GEN_C)),
                                "We make bath bombs colored with D&C Red No. 28. What does 82.1328 require, and does it restrict use?")
    assert out.splitlines() == [
        "21 CFR 82.1328 contains one requirement: D&C Red No. 28 shall conform in identity and specifications to the "
        "requirements of 21 CFR 74.1328(a)(1) and (b) [1].",
        "21 CFR 82.1328 states no use restriction [1].",
        f"21 CFR 82.1050 states: \"{GEN_C}\" [2]."]


def test_c82_stays_out_of_other_questions_and_other_listings():
    from regulus.rag.color_listing import compose_certification
    pkg = _pkg82(("82.1328", RED28), ("82.1050", GEN_C))
    assert compose_certification(pkg, "Can we use D&C Red No. 28 in a lipstick?") is None            # a use question
    assert compose_certification(pkg, "What does 82.1328 require of D&C Red No. 30?") is None         # another color
    assert compose_certification(_pkg82(("82.1328", RED28)), "What does 82.1328 require of D&C Red No. 28?") is None
    limited = _pkg82(("82.1328", RED28), ("82.1328", "(b) It may be used only in externally applied cosmetics."), ("82.1050", GEN_C))
    assert compose_certification(limited, "What does 82.1328 require of D&C Red No. 28?") is None    # says more


def test_e1_leaves_a_duty_another_supplied_source_states():
    # FDCA613-Q2: "registration is required" cited to 710.1's voluntary request, while 607 (supplied) mandates it
    from regulus.rag.claim_validation import strengthened_obligation
    s = "Registration is required for any owner or operator of a facility that manufactures cosmetic products [1]."
    voluntary = "The owner or operator of a cosmetic product establishment is requested to register each such establishment."
    act = "Every person that owns or operates a facility that manufactures cosmetic products shall register each facility."
    assert strengthened_obligation(s, [voluntary])
    assert strengthened_obligation(s, [voluntary], [act]) is None


def test_c82_a_provision_named_inside_a_verbatim_quote_of_the_cited_source_is_the_source_speaking():
    from regulus.rag.validation import _without_cited_quotes
    gen50 = ("A batch of a straight color listed in this subpart may be certified, in accordance with the provisions of "
             "the regulations in this part, for use in food, drugs, and cosmetics, if such batch conforms to the "
             "requirements of § 82.5 and to the specifications in this subpart set forth for such color.")
    quoted = f'21 CFR 82.50 states: "{gen50}" [2].'
    assert "82.5 " not in _without_cited_quotes(quoted, [gen50]).replace("82.50", "")
    moved = f'21 CFR 82.50 states: "{gen50[:-1]} [2].".'           # the marker moved inside the quote
    assert "82.5 " not in _without_cited_quotes(moved, [gen50]).replace("82.50", "")
    paraphrase = '21 CFR 82.50 says the batch "must meet the limits of § 82.5 for lead and arsenic" [2].'
    assert _without_cited_quotes(paraphrase, [gen50]) == paraphrase      # not the source's words: still checked


def test_c82_an_fdc_listing_says_which_82_5_limits_govern_it():
    from regulus.rag.color_listing import compose_certification
    y5 = "The color additive FD&C Yellow No. 5 shall conform in identity and specifications to the requirements of § 74.705 (a)(1) and (b) of this chapter."
    gen_b = ("A batch of a straight color listed in this subpart may be certified, in accordance with the provisions of the "
             "regulations in this part, for use in food, drugs, and cosmetics, if such batch conforms to the requirements "
             "of § 82.5 and to the specifications in this subpart set forth for such color.")
    srcs = [("82.705", "", y5), ("82.50", "", gen_b), ("82.5", "(b)(1)", "In the case of a straight color listed in subpart B:"),
            ("82.5", "(b)(1)(i)", "Lead (as Pb), not more than 0.001 percent."),
            ("82.5", "(b)(2)", "In the case of a straight color listed in subpart C or D:")]
    pkg = SimpleNamespace(sources=tuple(PackagedSource(**{**_src(i, sec, par, t).__dict__, "heading": "FD&C Yellow No. 5." if sec == "82.705" else None})
                                        for i, (sec, par, t) in enumerate(srcs, 1)),
                          requirements=(("r1", "21 CFR 82.705"), ("r2", "21 CFR 82.50"), ("r3", "21 CFR 82.5")))
    out = compose_certification(pkg, "We supply FD&C Yellow No. 5. What specifications must a batch meet for certification?")
    assert "FD&C Yellow No. 5 is a straight color listed in subpart B, so the subpart B limits in 21 CFR 82.5(b)(1) apply to it [3][4]." in out
    assert '21 CFR 82.5(b)(2) states: "In the case of a straight color listed in subpart C or D:" [5].' in out


# ------------------------------------------------------------ P1 practices the regulations do not address


def test_p1_detects_the_practice_questions_and_stays_out_of_others():
    from regulus.rag.practice_silence import asked_practices, asks_records
    keys = lambda q: [p.key for p in asked_practices(q)]
    assert keys("How often must we calibrate our scales and pH meters for cosmetic batches?") == ["calibration"]
    assert keys("Do we legally need a licensed pest-control contractor for our small cosmetics plant?") == ["pest_control"]
    assert keys("Do our mixers and filling machines for lotions need to be FDA 'certified'?") == ["equipment_certification"]
    assert asks_records("What pest-control records must a cosmetics contract manufacturer keep, and for how long?")
    assert keys("How do I clean the mixer for our lotion?") == []                         # asks no requirement
    assert keys("Must our mixers for lotions be cleaned daily?") == []                    # equipment, not certification
    assert keys("We make a face cream and an SPF 30 moisturizer on one kettle: cleaning and calibration rules?") == []
    assert keys("How often must we calibrate our scales?") == []                          # not about cosmetics


class _FakeDB:
    def __init__(self, texts):
        self.texts = texts

    def query(self, sql, params):
        return [{"section_number": "700.1", "body_text": t} for t in self.texts]


def test_p1_silence_is_a_search_of_the_text_not_an_assumption():
    from types import SimpleNamespace as NS
    from regulus.rag.practice_silence import PRACTICES, silent
    cal = next(p for p in PRACTICES if p.key == "calibration")
    act = NS(rows=[{"body_text": "A cosmetic shall be deemed to be adulterated if ..."}], by_sec={"601": [0]})
    assert silent(_FakeDB(["The term cosmetic product means a finished cosmetic."]), act, cal)
    assert not silent(_FakeDB(["Each scale shall be calibrated at least once a year."]), act, cal)   # a provision added later


def _act(index, cite, text):
    return SimpleNamespace(index=index, citation=f"FD&C Act § {cite} [21 U.S.C. x]", text=text, heading=None, corpus="usc")


def test_p1_composes_only_from_the_quoted_act_text():
    from regulus.rag.practice_silence import compose, notes_for, PRACTICES
    srcs = (_act(1, "606(a)", "The Secretary shall by regulation establish good manufacturing practices for facilities."),
            _act(2, "606(c)", "The Secretary shall publish a notice of proposed rulemaking not later than 2 years after."),
            _act(5, "601", "A cosmetic shall be deemed to be adulterated—"),
            _act(3, "601(c)", "(c) If it has been prepared, packed, or held under insanitary conditions."),
            _act(4, "601(f)", "(f) If it has been manufactured under conditions that do not meet the good manufacturing practice requirements."))
    pkg = SimpleNamespace(sources=srcs)
    cal = [p for p in PRACTICES if p.key == "calibration"]
    out = compose(pkg, "How often must we calibrate our scales for cosmetic batches?", notes_for(cal))
    lines = out.splitlines()
    assert lines[0].startswith("A search of 21 CFR Parts 700-740 and FD&C Act sections 601-614") and "calibration" in lines[0]
    assert lines[1] == 'FD&C Act § 606(a) states: "The Secretary shall by regulation establish good manufacturing practices for facilities." [1].'
    assert all(re.search(r"\[\d\]\.$", ln) for ln in lines[1:])
    # records asked, but 610(a) not supplied: nothing is composed (the model answers as before)
    assert compose(pkg, "What calibration records must we keep for our cosmetic batches?", notes_for(cal)) is None
    assert compose(pkg, "How often must we calibrate our scales for cosmetic batches?", ()) is None   # not this promotion


# ------------------------------------------------------------ ND1 a stray decline sentence


def test_stray_decline_removed_from_a_cited_answer():
    from regulus.rag.pipeline import drop_stray_decline

    text = ("No filing fee is required [1].  \n\nThe available regulatory sources do not provide enough information "
            "to answer this.\n\nStatus note (FDA): closed.")
    out, n = drop_stray_decline(text)
    assert n == 1 and out == "No filing fee is required [1].  \n\nStatus note (FDA): closed."


def test_bare_or_scoped_decline_is_kept():
    from regulus.rag.pipeline import drop_stray_decline

    bare = "The available regulatory sources do not provide enough information to answer this."
    assert drop_stray_decline(bare) == (bare, 0)
    uncited = "Filing is voluntary. " + bare
    assert drop_stray_decline(uncited) == (uncited, 0)
    scoped = ("Section 610 covers records [1]. The available regulatory sources do not provide enough information to "
              "answer whether FDA can demand pricing data.")
    assert drop_stray_decline(scoped) == (scoped, 0)


# ------------------------------------------------------------ MF1 an amount with zero cents


def test_a_whole_dollar_amount_is_the_sources_amount_with_zero_cents(monkeypatch):
    from regulus.rag import validation as V

    monkeypatch.setattr(V, "precision_fix", lambda fix: fix == "MF1")
    assert V._figures("at most $75 per member per day") == V._figures("not to exceed $75.00 per member per day")
    assert V._figures("10.0 percent") == {"10"} and V._figures("10.05 percent") == {"10.05"}
    monkeypatch.setattr(V, "precision_fix", lambda fix: False)
    assert V._figures("$75") != V._figures("$75.00")


# ------------------------------------------------------------ EX1 "not necessary" is an exemption


def test_an_exemption_restated_from_not_necessary_is_not_invented():
    from regulus.rag import claim_validation as CV

    assert CV._EXEMPT_SUPPORT.search("he may list such color additive and/or that certification procedure is not necessary")
    assert CV._EXEMPT_SUPPORT.search("batches thereof are exempt from the certification")
    assert not CV._EXEMPT_SUPPORT.search("The color additive shall conform to the specifications.")


# ------------------------------------------------------------ BN2 the verifier's score


def test_bn2_prefers_the_complete_draft_and_penalises_denials():
    from types import SimpleNamespace as NS

    from regulus.rag.pipeline import answer_score

    pkg = NS(sources=[NS(index=i, requirement_ids=("r1",)) for i in (1, 2, 3)] + [NS(index=4, requirement_ids=())])
    full = "The deposit is $2,600 [1]. Withdrawal before filing refunds less $600 [2]. Within 30 days [3]."
    part = "The deposit is $2,600 [1]. See the definitions [4]."
    denial = full + " The regulation does not address cosmetic use."
    assert answer_score(full, pkg) > answer_score(part, pkg)
    assert answer_score(denial, pkg) < answer_score(full, pkg)


def test_af1_a_section_named_with_its_paragraph_is_supplied(monkeypatch):
    from regulus.rag import validation as V

    src = "Tallow renders a cosmetic adulterated under section 601(c) of the act."
    monkeypatch.setattr(V, "precision_fix", lambda fix: fix == "AF1")
    assert "601" in V._supplied_figures(src)
    monkeypatch.setattr(V, "precision_fix", lambda fix: False)
    assert "601" not in V._supplied_figures(src)


def test_e2b_a_duty_another_supplied_source_states_itself():
    from types import SimpleNamespace as N
    from regulus.rag.claim_validation import duty_stated
    s = ("A manufacturer subject to 21 CFR part 820 must establish and maintain a quality management system that is "
         "appropriate for its specific device.")
    must = N(citation="21 CFR 820.1(a)", text="Any manufacturers engaged in the design, manufacture, packaging, labeling, "
             "storage, installation, or servicing of a finished device must establish and maintain a quality management "
             "system that is appropriate for its specific device(s).")
    encouraged = N(citation="21 CFR 820.1(a)(2)", text="This regulation does not apply to manufacturers of components or "
                   "parts of finished devices, but such manufacturers are encouraged to consider provisions of this regulation.")
    assert duty_stated(s, [encouraged, must])
    assert not duty_stated(s, [encouraged])


def test_e2b_a_paragraph_under_a_duty_lead_in_states_the_duty():
    from types import SimpleNamespace as N
    from regulus.rag.claim_validation import duty_stated
    s = ("A manufacturer subject to 21 CFR part 820 must document a quality management system that complies with the "
         "applicable requirements of ISO 13485.")
    lead = N(citation="21 CFR 820.10", text="A manufacturer subject to this part as described by § 820.1(a) must:")
    a = N(citation="21 CFR 820.10(a)", text="(a) Document. Document a quality management system that complies with the "
          "applicable requirements of ISO 13485 and other applicable requirements of this part; and")
    assert duty_stated(s, [lead, a])
    assert not duty_stated(s, [a])


def test_ar1_a_computed_figure_is_supported_only_when_its_arithmetic_holds():
    from regulus.rag.validation import verified_arithmetic
    supplied = {"2000", "0.45", "288"}
    assert verified_arithmetic("the fee is 2,000 pounds × $0.45 per pound = $900.", supplied) == {"900"}
    assert verified_arithmetic("2,000 x $0.45 = $950", supplied) == set()           # wrong arithmetic
    assert verified_arithmetic("3,000 × $0.45 = $1,350", supplied) == set()         # an operand nobody supplied


def test_c3_a_sibling_naming_the_cited_paragraph_states_its_condition(monkeypatch):
    from types import SimpleNamespace as N
    from regulus.rag import validator_switches as VS
    from regulus.rag.claim_validation import check_claims
    a = N(index=1, citation="21 CFR 740.2(a)", tier="exact", requirement_ids=("r1",), is_docket=False, text=(
        "(a) A warning statement shall appear on the label in bold type on contrasting background, but in no case may "
        "the letters be less than 1/16 inch in height, unless an exemption pursuant to paragraph (b) of this section is established."))
    b = N(index=2, citation="21 CFR 740.2(b)", tier="exact", requirement_ids=("r1",), is_docket=False, text=(
        "(b) If the label of any cosmetic package is too small to accommodate the information, the Commissioner may "
        "establish by regulation an acceptable alternative method, e.g., type size smaller than 1/16 inch in height."))
    pkg = N(sources=[a, b], answer_style="explain", question="How must warnings look?", requirements=())
    ident = lambda s: ("21", "740.2", s.citation.split("740.2")[1])
    rel = lambda x, y: x[:2] == y[:2] and (x[2].startswith(y[2]) or y[2].startswith(x[2]))
    s = "A smaller type size is allowed only if an exemption is established by regulation for a label too small [2]."
    codes = lambda: {f.code for f in check_claims([s], pkg, identity=ident, related=rel)}
    assert "condition_not_in_cited_source" in codes()
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"C3"})
    assert "condition_not_in_cited_source" not in codes()


# ------------------------------------------------------------ EX2 "do not apply" and the section heading


def test_ex2_an_exemption_restated_from_do_not_apply_is_cited_to_its_own_paragraph(monkeypatch):
    from types import SimpleNamespace as N
    from regulus.rag import validator_switches as VS
    from regulus.rag.claim_validation import check_claims
    # full300e F-002: 117.5(b) cited for "exempt"; another supplied 117.5 paragraph says "exempt" - withheld
    b = N(index=1, citation="21 CFR 117.5(b)", tier="exact", requirement_ids=("r1",), is_docket=False, heading=None, text=(
        "(b) Subparts C and G of this part do not apply with respect to activities that are subject to part 123 of this "
        "chapter (Fish and Fishery Products) at a facility if you are required to comply with, and are in compliance with, part 123."))
    g = N(index=2, citation="21 CFR 117.5(g)", tier="exact", requirement_ids=("r1",), is_docket=False, heading=None, text=(
        "(g) The exemption in paragraph (g)(1) applies to a facility solely engaged in the storage of raw agricultural commodities."))
    pkg = N(sources=[b, g], answer_style="explain", question="What exemptions apply?", requirements=())
    s = "Activities subject to part 123 are exempt from subparts C and G if the facility is in compliance with part 123 [1]."
    codes = lambda: {f.code for f in check_claims([s], pkg)}
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"EX2"})
    assert "condition_not_in_cited_source" in codes()
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"EX2"})
    assert "condition_not_in_cited_source" not in codes()
    # an "exempt" claim cited to a paragraph that states no exemption at all is still flagged
    plain = N(index=3, citation="21 CFR 117.10(a)", tier="exact", requirement_ids=("r1",), is_docket=False, heading=None,
              text="(a) Disease control. Any person who appears to have an illness must be excluded from operations.")
    pkg.sources.append(plain)
    assert "condition_not_in_cited_source" in {f.code for f in check_claims(
        ["Persons with an illness are exempt from operations under the disease control rule [3]."], pkg)}


def test_ex2_the_cited_paragraphs_section_heading_states_the_exemption(monkeypatch):
    from types import SimpleNamespace as N
    from regulus.rag import validator_switches as VS
    from regulus.rag.claim_validation import check_claims
    d = N(index=1, citation="21 CFR 117.5(d)(1)", tier="exact", requirement_ids=("r1",), is_docket=False,
          heading="§ 117.5 Exemptions.", text="(d)(1) Subpart B of this part applies only to the microbiological hazards.")
    o = N(index=2, citation="21 CFR 117.5(a)", tier="exact", requirement_ids=("r1",), is_docket=False, heading=None,
          text="(a) Except as provided by subpart E of this part, subparts C and G of this part do not apply to a qualified facility.")
    pkg = N(sources=[d, o], answer_style="explain", question="What exemptions apply?", requirements=())
    s = "Under this exemption, Subpart B applies only to the microbiological hazards [1]."
    codes = lambda: {f.code for f in check_claims([s], pkg)}
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"EX2"})
    assert "condition_not_in_cited_source" in codes()
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"EX2"})
    assert "condition_not_in_cited_source" not in codes()


# ------------------------------------------------------------ SR1 sentence remediation in place of a withheld answer


_SR1_DRAFT = ("Part 211 sets the minimum current good manufacturing practice for drug products [1]. "
              "It applies to drug products for administration to humans or animals [1]. "
              "Positron emission tomography drugs are excluded unless otherwise specified [1]. "
              "The regulations in parts 210 and 211 supplement each other [2].")
_SR1_BAD = "Positron emission tomography drugs are excluded unless otherwise specified [1]."


def test_sr1_removes_a_cited_sentence_the_validator_rejected():
    from regulus.rag.validation import excise_flagged_sentences, excise_unsupported_sentences
    # D-001's shape (full300e FAIL, withheld): one sentence's condition is in another source, not the cited one
    rep = _report(ValidationProblem("condition_not_in_cited_source", "error", "condition 'unless' is in a supplied source, "
                                    "but not in the one(s) cited", sentence=_SR1_BAD))
    out = excise_flagged_sentences(_SR1_DRAFT, rep, "What is the scope of Part 211?")
    assert out is not None and _SR1_BAD not in out[0] and out[1] == 1 and "supplement each other [2]" in out[0]
    # A3 alone keeps refusing cited errors and other codes: its behaviour is unchanged
    assert excise_unsupported_sentences(_SR1_DRAFT, rep, "What is the scope of Part 211?") is None


def test_sr1_keeps_a3s_guards():
    from regulus.rag.validation import excise_flagged_sentences
    q = "What is the scope of Part 211?"
    first = _SR1_DRAFT.split(" [1]. ")[0] + " [1]."
    # never the verdict (first) sentence
    assert excise_flagged_sentences(_SR1_DRAFT, _report(ValidationProblem("unsupported_claim", "error", "x", sentence=first)), q) is None
    # an error that is not one sentence's (truncation, degeneracy) leaves the draft alone
    assert excise_flagged_sentences(_SR1_DRAFT, _report(ValidationProblem("truncated", "error", "stopped")), q) is None
    # more than a third of the substantive sentences flagged: the draft is off its evidence, not trimmed
    s2 = "It applies to drug products for administration to humans or animals [1]."
    rep = _report(ValidationProblem("unsupported_claim", "error", "x", sentence=s2),
                  ValidationProblem("condition_not_in_cited_source", "error", "x", sentence=_SR1_BAD))
    assert excise_flagged_sentences(_SR1_DRAFT, rep, q) is None
    # the only sentence speaking to a word of the question stays: removing it would drop part of the answer
    q2 = "Are positron emission tomography drugs excluded?"
    rep = _report(ValidationProblem("condition_not_in_cited_source", "error", "x", sentence=_SR1_BAD))
    assert excise_flagged_sentences(_SR1_DRAFT, rep, q2) is None


def test_sr1_locates_an_unsupported_figure_by_its_number():
    from regulus.rag.validation import excise_flagged_sentences
    draft = ("Records must be kept for 1 year after the expiration date [1]. Records may be original or true copies [2]. "
             "Electronic records must comply with part 11 [3]. A copy must reach FDA within 45 days [1].")
    rep = _report(ValidationProblem("unsupported_figure", "error", "figure(s) not found in any supplied source: 45"))
    out = excise_flagged_sentences(draft, rep, "How long must records be kept?")
    assert out is not None and "45 days" not in out[0] and "1 year" in out[0]
    # the figure inside a source marker is not a sentence's figure
    rep = _report(ValidationProblem("unsupported_figure", "error", "figure(s) not found in any supplied source: 3"))
    assert excise_flagged_sentences(draft.replace("45 days [1]", "45 days"), rep, "How long?") is None
