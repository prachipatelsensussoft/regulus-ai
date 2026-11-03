"""DR1 (30 Oct): programme routing - regulus.retrieval.programmes and its use in section promotion."""
import json
from pathlib import Path

import pytest

from regulus.retrieval.programmes import PROGRAMMES, programme_for

STATUTE_ROWS = Path(__file__).resolve().parents[1] / "data" / "processed" / "statute_fdca" / "rows.jsonl"


@pytest.mark.parametrize("question,name", [
    ("What statutory provision governs premarket tobacco product applications?", "tobacco"),                   # T-001
    ("What must a Substantial Equivalence report contain and what is a valid predicate?", "tobacco"),          # T-005
    ("What harmful and potentially harmful constituent data must be submitted?", "tobacco"),                   # PB-002
    ("How must major food allergens be declared on a food label?", "food_labeling"),                           # L-010
    ("When must a food bear a common or usual name established by regulation?", "food_labeling"),              # L-015
    ("When must a 510(k) be submitted for a change to a legally marketed device?", "device_submissions"),      # P-003
    ("How are medical devices classified and what does a 513(g) request accomplish?", "device_submissions"),   # P-008
    ("What must an IDE application contain?", "device_submissions"),
    ("What are the MQSA requirements for mammography equipment?", "radiological_health"),
    ("How is the primary mode of action of a combination product determined?", "combination_products"),
    # 2 Nov: biologics and animal food
    ("What adverse experience reporting requirements apply to licensed biological products?", "biologics"),   # B-002
    ("How are manufacturing changes to a licensed biological product reported?", "biologics"),                # B-004
    ("What data must a biologics license application include?", "biologics"),
    ("When must a BLA supplement be submitted?", "biologics"),
    ("What must a blood establishment record for each unit of whole blood?", "biologics"),
    ("What current good manufacturing practice requirements apply to the manufacturing, processing, packing, and "
     "holding of animal food?", "animal_food"),                                                             # A-001
    ("What hazard analysis and preventive controls requirements apply to an animal food facility?", "animal_food"),   # A-002
    ("What requirements apply to medicated feed manufacturing and to veterinary feed directive drugs?", "animal_food"),  # A-004
    ("How long must a VFD be kept by the feed distributor?", "animal_food"),
    ("Which labeling statements must pet food bear?", "animal_food"),
    ("May a medicated feed contain two new animal drugs?", "animal_food"),
])
def test_a_question_naming_one_programme_is_routed(question, name):
    assert programme_for(question).name == name


@pytest.mark.parametrize("question", [
    "How must net quantity of contents be declared?",                     # L-005: food, drug, device or cosmetic
    "What is the standard for substantial equivalence?",                 # P-002: 510(k) as well as tobacco
    "What must a PMA application contain under Part 814?",               # a named part decides
    "What steps must be taken to prevent allergens from cross-contact during food processing?",   # 117.80, not labeling
    "What is the principal display panel for over-the-counter drugs?",   # 201.60: shared words without "food"
    "Do OTC products claiming to help break a cigarette habit need an approved new drug application?",   # 310.544
    "What labeling requirements apply to human food by-products distributed as animal feed?",   # 507.28 and 117.95
    "Does a mammography facility need a 510(k) for its x-ray unit?",     # two programmes named
    # biologics: the bare plural is another part's scope clause (bank: 1271.330, 310.509)
    "Are biological products exempt from these rules regarding plastic packaging?",
    "Are biological products regulated under section 351 of the Public Health Service Act included in this rule?",
    "Which drugs and biological products need a medication guide?",       # a drug programme named
    "What must a biologics license application contain under Part 601?",  # a named part decides
    # animal food yields to a named part, another programme or the drug/colour/import words of another
    "What does 21 CFR Part 507 require of an animal food facility?",
    "Can a color additive be used in pet food?",
    "Must imported animal feed be registered?",
    "Can tobacco by-products be used in animal feed?",
    "What does an OTC drug label for a VFD need?",
])
def test_no_programme_or_another_one_is_not_routed(question):
    assert programme_for(question) is None


def test_ends_and_pma_are_acronyms_only():
    assert programme_for("When the study ends, what records apply?") is None
    assert programme_for("Which tobacco products are ENDS?").name == "tobacco"


def test_animal_food_is_not_kept_from_its_own_programme_by_the_animal_feed_words():
    # _OTHER names "animal food/feed" to keep the other programmes off it, and "drugs" for the drug programmes; the
    # animal-food programme's own phrases ("animal feed", "new animal drugs", "VFD drugs") do not count against it
    assert programme_for("Which nutrition labeling rules apply to animal feed?") is None   # food labeling yields
    assert programme_for("What records must a feed mill keep for medicated animal feed?").name == "animal_food"
    assert programme_for("What are the requirements for VFD drugs in animal feed?").name == "animal_food"
    assert programme_for("Which drugs may be used in medicated feed?") is None             # a drug, not an animal drug


def test_a_combination_product_naming_a_biologic_stays_a_combination_product():
    q = "How is the lead center assigned for a combination product with a biological product constituent?"
    assert programme_for(q).name == "combination_products"
    assert programme_for("Is a licensed biological product a combination product?").name == "combination_products"


def test_act_sections_are_ones_the_statute_store_holds():
    held = {json.loads(line)["fdca_section"] for line in STATUTE_ROWS.read_text(encoding="utf-8").splitlines() if line}
    for p in PROGRAMMES:
        assert set(p.act) <= held, (p.name, sorted(set(p.act) - held))
        assert len(set(p.act)) == len(p.act), p.name
    by_name = {p.name: p for p in PROGRAMMES}
    assert {"901", "903", "908", "909", "910", "911", "915", "919"} <= set(by_name["tobacco"].act)
    assert {"534", "535", "537", "538"} <= set(by_name["radiological_health"].act)
    assert by_name["biologics"].act == ()                 # PHS Act 351 is not in the store
    assert {"504", "512", "418"} <= set(by_name["animal_food"].act)


def test_mammography_is_not_offered_the_electronic_product_sections():
    q = "What records must a manufacturer of an electronic product keep?"
    assert programme_for(q).name == "radiological_health" and "537" in programme_for(q).act_for(q)
    q = "What quality assurance records must a mammography facility keep?"
    assert programme_for(q).name == "radiological_health" and programme_for(q).act_for(q) == ()


def test_prepare_offers_the_routed_programmes_act_sections(monkeypatch):
    from regulus.rag import section_promotion as SP
    from regulus.rag import validator_switches as VS

    class _Statute:
        by_sec = {s: [0] for s in ("531", "532", "534", "535", "537", "538", "504", "512", "418", "409")}

    class _Promoter(SP.SectionPromoter):
        def __init__(self):
            self.mode, self._statute_ix, self.db = "picker", _Statute(), None

        @property
        def statute(self):
            return self._statute_ix

        def named_statute(self, question, lex):
            return []

        def candidates(self, question, parts=None):
            return ["x"], [{"section": "1002.10", "part": "1002", "heading": "", "subpart": ""}]

        def statute_candidates(self, lexemes, question=None, offered=SP.STATUTE_OFFERED):
            return [{"statute": True, "section": s} for s in offered]

    monkeypatch.setattr(SP.SectionIndex, "question_lexemes", staticmethod(lambda db, q: ["x"]))
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", frozenset({"DR1"}))
    p = _Promoter()
    sections = lambda q: [c["section"] for c in p.prepare(q)[1]]   # noqa: E731
    assert sections("What reports must a manufacturer of an electronic product submit?") == ["1002.10", "534", "537", "535"]
    assert sections("What reports must a mammography facility submit?") == ["1002.10"]          # mammography: none
    assert sections("What must a biologics license application contain?") == ["1002.10"]       # biologics: none
    assert sections("Which records must a medicated feed mill keep?") == ["1002.10", "418", "504", "512"]
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", frozenset())
    assert sections("What reports must a manufacturer of an electronic product submit?") == ["1002.10"]


def test_routed_programme_is_off_unless_dr1(monkeypatch):
    from regulus.rag import section_promotion as SP
    from regulus.rag import validator_switches as VS

    q = "What must a premarket tobacco product application contain?"
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"DR1"})
    assert SP.routed_programme(q) is None
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"DR1"})
    assert SP.routed_programme(q).name == "tobacco"


def test_sub_questions_of_a_routed_question_stay_in_its_programme(monkeypatch):
    # L-015 (30 Oct): "where are those names codified?" searched every part and promoted 502.19 (animal food)
    from regulus.rag import section_promotion as SP
    from regulus.rag import validator_switches as VS

    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"DR1", "QD1"})
    monkeypatch.setattr(SP, "RERANKER", "stub")

    class Gen:
        def generate(self, messages, max_new_tokens=0, **kw):
            return type("G", (), {"response": "When must a food bear a common or usual name?\nWhere are those names codified?"})()

    pr = SP.SectionPromoter.__new__(SP.SectionPromoter)
    pr.generator, seen = Gen(), []
    monkeypatch.setattr(pr, "candidates", lambda sub, scope: (seen.append(scope), ([], [{"section": "102.5"}]))[1], raising=False)
    monkeypatch.setattr(pr, "rerank_scores", lambda sub, cfr: {"102.5": (0.9, "")}, raising=False)
    q = "When must a food bear a common or usual name established by regulation, and where are those names codified?"
    assert pr.decomposed_sections(q) == ["102.5"]
    assert seen and all(scope == ("101", "102", "104", "105") for scope in seen)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"DR1"})
    seen.clear()
    pr.decomposed_sections(q)
    assert seen and all(scope is None for scope in seen)   # without DR1: every part, as before


def test_ap2_the_named_act_subsection_is_passed_on(monkeypatch):
    from regulus.rag import section_promotion as SP
    from regulus.rag import validator_switches as VS

    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"AP2"})
    q = "How are medical devices classified and what does a 513(g) request accomplish?"   # P-008
    assert SP.named_act_path("513", q, "FD&C Act § 513") == "(g)"                       # the question names it
    assert SP.named_act_path("513", "What is a classification request?", "FD&C Act § 513(f)(2)") == "(f)(2)"   # the picker
    assert SP.named_act_path("513", "What is a classification request?", "FD&C Act § 513") == ""
    assert SP.named_act_path("513", "Under 21 CFR 513.2(a) ...", None) == ""                # a CFR number is not the Act
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"AP2"})
    assert SP.named_act_path("513", q, None) == ""


def test_qd3_another_parts_sub_question_section_needs_the_stricter_floor(monkeypatch):
    # P-012: the UDI question (promoted 830.320) got 822.3 from a sub-question searched across every part
    from regulus.rag import section_promotion as SP
    from regulus.rag import validator_switches as VS

    # the unrouted path (DR2 off): sub-questions are searched across every part
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", (VS.VALIDATOR_PRECISION | {"QD1", "QD3"}) - {"DR2"})
    monkeypatch.setattr(SP, "RERANKER", "stub")

    class Gen:
        def generate(self, messages, max_new_tokens=0, **kw):
            return type("G", (), {"response": "What must be submitted to GUDID?\nWhat must the label bear?"})()

    pr = SP.SectionPromoter.__new__(SP.SectionPromoter)
    pr.generator = Gen()
    best = {"What must be submitted to GUDID?": "822.3", "What must the label bear?": "801.20"}

    def candidates(sub, scope):   # a search scoped to parts finds only sections of those parts
        sec = best[sub]
        return [], ([{"section": sec}] if scope is None or sec.split(".")[0] in scope else [])

    monkeypatch.setattr(pr, "candidates", candidates, raising=False)
    monkeypatch.setattr(pr, "rerank_scores", lambda sub, cfr: {"822.3": (0.6, ""), "801.20": (0.8, "")}, raising=False)
    q = "What are the unique device identification requirements and what must be submitted to GUDID?"
    assert pr.decomposed_sections(q, None, ("830",)) == ["801.20"]           # 822.3 at 0.6 < 0.7 dropped
    assert pr.decomposed_sections(q, None, ("822",)) == ["822.3", "801.20"]  # its own part: the 0.5 floor
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"QD3"})
    assert pr.decomposed_sections(q, None, ("830",)) == ["822.3", "801.20"]  # off: as before
    # DR2 on (default since 3 Nov): the UDI question is routed to device labeling (801/809/830), its sub-questions are
    # searched there, and postmarket 822.3 is not reachable at all, with or without QD3
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"DR2"})
    assert pr.decomposed_sections(q, None, ("830",)) == ["801.20"]


# ------------------------------------------------------------ DR2 (3 Nov): the benchmark domains DR1 left unrouted

@pytest.mark.parametrize("question,name", [
    ("What must a seafood HACCP plan contain, and what monitoring and verification are required?", "seafood_haccp"),   # Y-009
    ("What supplier verification activities does FSVP require and how is the appropriate activity determined?", "fsvp"),  # V-004
    ("What must appear on a device label with respect to the name and place of business of the manufacturer?", "device_labeling"),  # U-001
    ("What events must a device manufacturer report under the Medical Device Reporting regulation?", "device_postmarket"),  # R-001
    ("What is the 'appropriate for the protection of the public health' standard?", "tobacco"),                          # T-002
    ("What hazard analysis and HACCP plan requirements apply to juice processors?", "juice_haccp"),
    ("What pH control requirements apply to acidified foods?", "canned_foods"),
    ("What requirements apply to agricultural water under the produce safety rule?", "produce_safety"),
    ("What labeling requirements apply to an in vitro diagnostic product?", "device_labeling"),
])
def test_dr2_routes_the_programmes_dr1_left_unrouted(question, name):
    assert programme_for(question) is None or programme_for(question).name != name   # DR1 alone does not
    assert programme_for(question, extended=True).name == name


@pytest.mark.parametrize("question", [
    "What does EU MDR Article 10 require of manufacturers?",                                   # SD-001: the European MDR
    "What should I call a seafood cocktail that has only one type of seafood?",               # 102.54 naming, not HACCP
    "Do I need to include can dimensions or volume when filing an entry for an acidified food?",  # 1.73 import entry
    "How should a receiving facility document that an importer follows the foreign supplier verification requirements?",
    "What kinds of products are considered in vitro diagnostic products?",                    # 201.119
    "Is it allowed to change the labeling of a device after approval?",                       # 814.80
])
def test_dr2_leaves_other_programmes_questions_alone(question):
    assert programme_for(question, extended=True) is None


def test_dr2_keeps_every_dr1_route():
    # an investigational device's label is 812.5's: device labeling must not take it from device submissions
    q = "Can an investigational device label claim that it is safe or effective for the purposes of investigation?"
    assert programme_for(q).name == programme_for(q, extended=True).name == "device_submissions"


def test_routed_programme_uses_dr2_only_when_switched_on(monkeypatch):
    from regulus.rag import validator_switches as VS
    from regulus.rag.section_promotion import routed_programme
    q = "What must a seafood HACCP plan contain?"
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", (VS.VALIDATOR_PRECISION | {"DR1"}) - {"DR2"})
    assert routed_programme(q) is None
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"DR2"})
    assert routed_programme(q).name == "seafood_haccp"
