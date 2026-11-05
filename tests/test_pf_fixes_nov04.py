"""4 Nov 2026 fixes for 300Q PARTIALLY FAIL answers (all off until measured): DI1 and MP1 (regulus.rag.answer_fixes),
SP4 (named SB1 until 4 Nov) and VS1 (regulus.rag.section_promotion). No database, no model."""
from dataclasses import dataclass

import pytest

import regulus.rag.validator_switches as VS
from regulus.rag import section_promotion as SP
from regulus.rag.answer_fixes import drop_denial_opener, resolve_letter_markers
from regulus.rag.section_promotion import Promotion, subpart_requirements_named

# ------------------------------------------------------------ DI1

M018_Q = "How does Part 820 extend the standard's traceability requirements for implantable devices beyond the standard's own scope?"
M018_A = ("Part 820 does not extend traceability requirements for implantable devices beyond the scope of the standard's own "
          "provisions. Instead, it specifies that manufacturers of devices that support or sustain life must comply with "
          "Clause 7.5.9.2 of ISO 13485 [1].")


def test_di1_drops_the_opener_the_next_sentence_contradicts():
    text, dropped = drop_denial_opener(M018_A, M018_Q)
    assert dropped.startswith("Part 820 does not extend")
    assert text.startswith("Part 820 specifies that manufacturers of devices that support or sustain life")
    assert text.endswith("[1].")


def test_di1_names_the_section_when_the_next_sentence_does():
    q = "What does Part 820 provide with respect to foreign manufacturers that refuse to permit an FDA inspection?"
    a = ("Part 820 does not provide specific requirements regarding foreign manufacturers that refuse to permit an FDA "
         "inspection. Instead, Section 820.1(c) states that a device imported may be refused admission [1].")
    text, dropped = drop_denial_opener(a, q)
    assert dropped and text.startswith("Section 820.1(c) states")


def test_di1_leaves_a_denial_that_is_the_answer():
    # a negative control asks WHICH provision requires something: the denial is the answer
    q = "Which provision of 21 CFR Part 11 requires biometric-only authentication for electronic signatures?"
    a = "Part 11 does not require biometric-only authentication. Instead, 11.200(a) covers non-biometric signatures [1]."
    assert drop_denial_opener(a, q) == (a, None)
    # the denied verb is not the question's
    q = "How does Part 820 extend the standard's traceability requirements?"
    a = "Part 820 does not require certification. Instead, it requires traceability [1]."
    assert drop_denial_opener(a, q) == (a, None)
    # no "Instead" / "Rather": a plain negative statement stays
    a = "Part 820 does not extend the requirement to all devices. It applies to devices that support or sustain life [1]."
    assert drop_denial_opener(a, M018_Q) == (a, None)
    # a denial later in the answer stays
    a = "Clause 7.5.9.2 applies [1]. Part 820 does not extend it further. Instead, it relies on the standard [1]."
    assert drop_denial_opener(a, M018_Q) == (a, None)


# ------------------------------------------------------------ MP1


@dataclass
class _Src:
    index: int
    citation: str
    text: str = ""


SOURCES = [_Src(1, "21 CFR 820.45"), _Src(2, "21 CFR 820.45(a)"), _Src(8, "21 CFR 820.45(b)"), _Src(9, "21 CFR 820.45(c)")]


def test_mp1_points_a_letter_marker_at_its_paragraph():
    text = "The release of labeling for use must be documented in accordance with Clause 4.2.5 of ISO 13485 [b]."
    out, n = resolve_letter_markers(text, SOURCES)
    assert n == 1 and out.endswith("ISO 13485 [8].")


def test_mp1_leaves_ambiguous_unsupplied_and_ordinary_brackets():
    two = SOURCES + [_Src(12, "21 CFR 820.35(b)")]
    text = "Records must be kept [b]."
    assert resolve_letter_markers(text, two) == (text, 0)                      # two sections have a (b): ambiguous
    named = "Under 21 CFR 820.35, servicing records must be kept [b]."
    assert resolve_letter_markers(named, two) == ("Under 21 CFR 820.35, servicing records must be kept [12].", 1)
    assert resolve_letter_markers("See [d].", SOURCES) == ("See [d].", 0)       # no (d) supplied
    assert resolve_letter_markers("Use [1] and a[b] array.", SOURCES) == ("Use [1] and a[b] array.", 0)


# ------------------------------------------------------------ SP4 (was SB1)


def test_sb1_needs_the_subpart_title_followed_by_requirements():
    assert subpart_requirements_named("Holding and Distribution", "What holding and distribution requirements apply under Part 211?")
    assert subpart_requirements_named("Health and Hygiene", "What personnel qualification, training, health and hygiene "
                                      "requirements apply under the produce safety rule?")
    # the title only occurring is not enough (E-008 PASS must not change)
    assert not subpart_requirements_named("Electronic Signatures",
                                          "What are the requirements for the uniqueness and certification of electronic signatures?")
    # a one-word title never qualifies
    assert not subpart_requirements_named("Records", "What records requirements apply under Part 112?")


class _SubpartDB:
    def __init__(self, rows):
        self.rows = rows

    def query(self, sql, params=None):
        return self.rows


def _promoter(db):
    pr = SP.SectionPromoter.__new__(SP.SectionPromoter)
    pr.db = db
    return pr


def test_sb1_brings_a_small_named_subpart_only():
    rows = [{"id": 8, "title": "Holding and Distribution", "sec": "211.142", "c": 400, "o": 1},
            {"id": 8, "title": "Holding and Distribution", "sec": "211.150", "c": 409, "o": 2},
            {"id": 9, "title": "Laboratory Controls", "sec": "211.160", "c": 2000, "o": 3}]
    q = "What holding and distribution requirements apply under Part 211?"
    assert _promoter(_SubpartDB(rows)).named_small_subparts("211.150", q) == ["211.142"]
    big = [dict(r, c=SP.SP3_MAX_CHARS) for r in rows[:2]]
    assert _promoter(_SubpartDB(big)).named_small_subparts("211.150", q) == []


# ------------------------------------------------------------ VS1


class _Gen:
    retrieval_only = False
    thinking = False

    def __init__(self, reply):
        self.reply = reply

    def generate(self, msgs, max_new_tokens=None, **kw):
        return type("R", (), {"response": self.reply})()


def _veto_promoter(monkeypatch, reply, vetoed_section):
    pr = SP.SectionPromoter.__new__(SP.SectionPromoter)
    pr.mode, pr.generator, pr.db = "picker", _Gen(reply), None
    cands = [{"section": "101.4", "part": "101", "heading": "Food; designation of ingredients.", "snippet": "allergens"},
             {"section": "403", "statute": True, "usc": "343", "heading": "Misbranded food", "snippet": "allergen labeling"},
             {"section": "101.9", "part": "101", "heading": "Nutrition labeling of food.", "snippet": "nutrition"}]
    monkeypatch.setattr(SP, "RERANKER", True)
    monkeypatch.setattr(SP, "ENTITY_CONSTRAINTS", False)
    monkeypatch.setattr(pr, "prepare", lambda *a, **k: (["allergen"], cands))
    for name in ("act_definition_asked", "definitions_asked", "practice_silence"):
        monkeypatch.setattr(pr, name, lambda *a, **k: None)
    monkeypatch.setattr(pr, "rerank_veto", lambda q, c, p: next(x for x in c if x["section"] == vetoed_section))
    monkeypatch.setattr(pr, "_statute_promotion", lambda acts, *a, **k: Promotion(
        sections=(), companions=(), method="picker", statute=tuple(a for a, _ in acts)))
    monkeypatch.setattr(pr, "siblings", lambda c, cands, lex: [c["section"]])
    monkeypatch.setattr(pr, "companions", lambda s, q, lex: [])
    return pr


def test_vs1_keeps_the_regulation_when_the_veto_takes_the_pickers_own_act_section(monkeypatch):
    q = "How must major food allergens be declared on a food label?"
    pr = _veto_promoter(monkeypatch, "21 CFR 101.4, FD&C Act § 403", "403")
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", frozenset())
    off = pr.resolve(q)
    assert off.statute == ("403",) and off.sections == ()                    # today: the Act section only
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", frozenset({"VS1"}))
    on = pr.resolve(q)
    assert on.statute == ("403",) and on.sections == ("101.4",)
    assert "veto kept regulation: 101.4" in on.notes


def test_vs1_leaves_a_veto_to_an_act_section_the_picker_did_not_name(monkeypatch):
    q = "How must major food allergens be declared on a food label?"
    pr = _veto_promoter(monkeypatch, "21 CFR 101.4, 21 CFR 101.9", "403")
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", frozenset({"VS1"}))
    on = pr.resolve(q)
    assert on.sections == () and not any("veto kept" in n for n in on.notes)


# ------------------------------------------------------------ SU1


@dataclass
class _QSrc:
    index: int
    citation: str
    text: str
    is_docket: bool = False


class _Pkg:
    def __init__(self, sources):
        self.sources = sources


QX1_SOURCES = [
    _QSrc(1, "21 CFR 820.45", "In addition to the requirements of Clause 7.5.1 of ISO 13485 (incorporated by reference, see "
          "\u00a7 820.7), Control of production and service provision, each manufacturer must document and maintain procedures "
          "that provide a detailed description of the activities to ensure the integrity, inspection, storage, and operations "
          "for labeling and packaging during the customary conditions of processing, storage, handling, distribution, and where "
          "appropriate, use of the device."),
    _QSrc(2, "21 CFR 820.45(a)", "(a) The manufacturer must ensure labeling and packaging has been examined for accuracy prior "
          "to release or storage, where applicable, to include the following:"),
    _QSrc(8, "21 CFR 820.45(b)", "(b) The release of the labeling for use must be documented in accordance with Clause 4.2.5 "
          "of ISO 13485."),
    _QSrc(9, "21 CFR 820.10", "A manufacturer subject to this part as described by \u00a7 820.1(a) must:"),
]


def test_su1_answers_a_supplement_question_with_the_sections_own_text():
    from regulus.rag.qmsr import compose_supplement

    out = compose_supplement(_Pkg(QX1_SOURCES), "What does 21 CFR 820.45 require that the incorporated standard does not?")
    lines = out.splitlines()
    assert lines[0] == ("Under 21 CFR 820.45, which adds to ISO 13485 Clause 7.5.1, what it requires in addition to the "
                        "standard, verbatim:")
    assert [ln.split(" (eCFR")[0] for ln in lines[1:]] == ["21 CFR 820.45", "21 CFR 820.45(a)", "21 CFR 820.45(b)"]
    assert lines[1].endswith("[1].") and "820.10" not in out


def test_su1_needs_one_named_section_the_standard_and_its_in_addition_sentence():
    from regulus.rag.qmsr import asks_supplement, compose_supplement

    assert asks_supplement("What does 21 CFR 820.35 add to the record control requirements of the incorporated standard?") == "820.35"
    assert asks_supplement("Which provisions of Part 820 are FDA supplements to the incorporated standard?") is None
    assert asks_supplement("What does 21 CFR 820.45 require for labeling?") is None            # the standard not named
    assert asks_supplement("How do 820.35 and 820.45 add to ISO 13485?") is None                 # two sections
    # 820.10 does not open "In addition to ... Clause": no composed answer
    assert compose_supplement(_Pkg(QX1_SOURCES), "What does 21 CFR 820.10 require beyond ISO 13485?") is None


# ------------------------------------------------------------ SU1, part level (5 Nov)
PART_SOURCES = [
    _QSrc(3, "21 CFR 820.10(a)", "(a) Document. Document a quality management system that complies with the applicable "
          "requirements of ISO 13485 (incorporated by reference, see § 820.7) and other applicable requirements of this part; and"),
    _QSrc(4, "21 CFR 820.10(b)", "(b) Applicable regulatory requirements. Comply, as appropriate, with the other applicable "
          "regulatory requirements in this title, including, but not limited to the following, to fully comply with the listed "
          "ISO 13485 Clause:"),
    _QSrc(6, "21 CFR 820.10(e)", "(e) Enforcement. The failure to comply with any applicable requirement in this part renders a "
          "device adulterated under section 501(h) of the Federal Food, Drug, and Cosmetic Act."),
    _QSrc(7, "21 CFR 820.35", "In addition to the requirements of Clause 4.2.5 in ISO 13485 (incorporated by reference, see "
          "§ 820.7), Control of Records, the manufacturer must include the following information in certain records:"),
    QX1_SOURCES[0],
]
M029 = "Which provisions of Part 820 are FDA supplements to the incorporated standard rather than incorporation by reference?"


# the question TYPE, not M-029's wording: phrasings a user may write, and look-alikes answered elsewhere
PART_SUPPLEMENT_PHRASINGS = [
    M029,
    "Which requirements in the QMSR go beyond ISO 13485?",
    "What does FDA add to ISO 13485 under the QMSR?",
    "What are FDA's additional requirements beyond ISO 13485 in Part 820?",
    "Which parts of the QMSR are FDA-specific rather than incorporated from ISO 13485?",
    "What requirements does Part 820 impose in addition to ISO 13485?",
    "Under the Quality Management System Regulation, what does FDA require on top of ISO 13485?",
    "Which sections of 21 CFR Part 820 supplement ISO 13485:2016?",
    "Besides incorporating ISO 13485, what extra requirements does the QMSR contain?",
    "How does Part 820 go beyond the requirements of ISO 13485?",
    "Apart from ISO 13485, what additional FDA requirements are in the QMSR?",
    "What does the QMSR require in addition to the standard?",
]
PART_SUPPLEMENT_LOOKALIKES = [
    "What does 21 CFR 820.45 add to ISO 13485?",                                   # one section: SU1's own form
    "Does Part 820 require ISO 13485 certification?",                              # QC1
    "What are the CAPA requirements under Part 820?",                              # QM1
    "Which provisions of Part 820 cover complaint files?",
    "What changed when FDA amended Part 820 to the QMSR, and what is the compliance date?",   # QS1
    "Is ISO 13485 incorporated by reference into Part 820?",
    "Which ISO 13485 clause covers purchasing?",
    "What records must be kept under 21 CFR 820.35?",
    "Which provisions of Part 211 add to the batch record?",                       # not part 820
    "What does the QMSR say about labeling inspection?",
    "What additional records does Part 820 require beyond ISO 13485 for complaints?",  # one subject: its own provision
    "What labeling requirements does Part 820 add on top of ISO 13485?",
    "What additional records does Part 820 require?",                              # no comparison with the standard
    "How does Part 820 define a manufacturer?",
]


@pytest.mark.parametrize("question", PART_SUPPLEMENT_PHRASINGS)
def test_su1_part_level_catches_the_question_type(question):
    from regulus.rag.qmsr import asks_part_supplements

    assert asks_part_supplements(question)


@pytest.mark.parametrize("question", PART_SUPPLEMENT_LOOKALIKES)
def test_su1_part_level_leaves_look_alikes_to_their_own_answers(question):
    from regulus.rag.qmsr import asks_part_supplements

    assert not asks_part_supplements(question)


def test_su1_part_level_names_and_quotes_only_what_the_evidence_holds():
    from regulus.rag.qmsr import compose_part_supplements

    out = compose_part_supplements(_Pkg(PART_SOURCES), M029)
    lines = out.split("\n")
    # each lead names only the provision quoted right after it; no summary sentence cites a provision away from its text
    assert lines[0] == "Part 820 incorporates ISO 13485 by reference through 21 CFR 820.10(a) [3], verbatim:"
    assert lines[1].startswith("21 CFR 820.10(a) (eCFR, binding, verbatim): (a) Document.") and lines[1].endswith("[3].")
    assert lines[2] == "The provisions of Part 820 that add FDA requirements to the standard, each verbatim:"
    assert [ln.split(" (eCFR", 1)[0] for ln in lines[3:]] == ["21 CFR 820.10(b)", "21 CFR 820.10(e)", "21 CFR 820.35",
                                                                "21 CFR 820.45"]
    assert lines[5].endswith("[7].") and lines[6].endswith("[1].")


def test_su1_part_level_declines_without_the_incorporation_or_an_in_addition_lead():
    from regulus.rag.qmsr import compose_part_supplements

    assert compose_part_supplements(_Pkg([s for s in PART_SOURCES if s.citation != "21 CFR 820.10(a)"]), M029) is None
    assert compose_part_supplements(_Pkg([s for s in PART_SOURCES if "820.10" in s.citation]), M029) is None
    assert compose_part_supplements(_Pkg(PART_SOURCES), "What does 21 CFR 820.45 add to ISO 13485?") is None
