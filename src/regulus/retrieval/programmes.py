"""DR1 (30 Oct, off until measured): route a question to the regulatory programme it names - its 21 CFR parts and the
FD&C Act sections that programme is issued under - before the section picker reads its candidates. The query
classification + metadata filter step of search systems; cosmetics questions have had it since 22 Oct
(section_index.COSMETIC_PARTS).

full300d (29 Oct, client keys), the five weakest domains: food labeling 13 of 16 below PASS/PP, tobacco 10/12, device
submissions 8/12, radiological health 5/6, combination products 4/4. The candidates of a tobacco question were mostly
other programmes' sections (25.35, 1.79, 16.1, 606.121) and 1114.7 was not among them; the Act sections the tobacco
keys cite (904-911) were never offered although the store holds them - the picker named "FD&C Act 910(a)(2)(B)" for
T-001 and it was discarded. Other programmes' sections were picked for X-003 (820.1 for 4.4) and P-012 (830 for 801.20).

2 Nov: biologics (Parts 600-680) and animal food (feed, medicated feed, VFD) added; the statute store's new chapter IX
and electronic-product sections (901-920, 531-538) offered to tobacco and radiological questions.

A question is routed only when it names exactly one programme by its own vocabulary and no 21 CFR part; a question
naming none ("How must net quantity of contents be declared?" - food 101.7, drug 201.62, device 801.62, cosmetic
701.13) is left as it is."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Programme:
    name: str
    parts: frozenset[str]
    act: tuple[str, ...]            # FD&C Act sections offered to the picker, when the statute store holds them
    words: re.Pattern               # case-insensitive programme vocabulary
    acronyms: re.Pattern | None     # case-sensitive ("ENDS", "PMA", "IDE" are words in lower case)
    # words other programmes use too ("principal display panel" is also an OTC drug's, 201.60): they count only beside
    # `context` ("food")
    shared: re.Pattern | None = None
    context: re.Pattern | None = None
    # the programme's own phrases that _OTHER would read as another programme ("animal feed", "VFD drugs"): taken out of
    # the question before the _OTHER check, so they do not keep the question from its own programme
    own_other: re.Pattern | None = None
    # the Act sections are not offered when the question matches this: mammography is issued under the PHS Act (MQSA,
    # 42 U.S.C. 263b), not the electronic-product sections 531-538 that the rest of the programme is
    no_act: re.Pattern | None = None
    # programmes whose vocabulary this one's includes: when both are named, the question is this one's
    subsumes: frozenset[str] = frozenset()
    # a question also matching this names another programme as well: not routed to this one
    not_with: re.Pattern | None = None

    def act_for(self, question: str) -> tuple[str, ...]:
        """The Act sections offered for this question: `act`, or none when it matches `no_act`."""
        if self.no_act is not None and self.no_act.search(question or ""):
            return ()
        return self.act


# word edges by lookaround, not \b: "510(k)" ends in a parenthesis
def _ci(*alts: str) -> re.Pattern:
    return re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.I)


def _cs(*alts: str) -> re.Pattern:
    return re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)")


_MAMMOGRAPHY = _ci(r"mammograph\w*", r"MQSA")

PROGRAMMES: tuple[Programme, ...] = (
    Programme(
        "tobacco", frozenset({"1100", "1105", "1107", "1114", "1140", "1141", "1143", "1150"}),
        # 900 definitions, 901 FDA authority and the products it applies to (the deeming rule, Part 1100), 902/903
        # adulterated/misbranded, 904 health information (ingredients, HPHCs), 905 registration and listing, SE reports,
        # 906 general controls (TPMP), 907 product standards, 908 notification and recall, 909 records and reports
        # (removals and corrections), 910 PMTA, 911 MRTP, 915 constituent testing and reporting, 919 user fees (Part
        # 1150), 920 origin labeling and tracking records. Not 912-914, 916-918, 921 (judicial review, retail outlets,
        # FTC, preemption, advisory committee, cessation drugs, reports to Congress)
        ("900", "901", "902", "903", "904", "905", "906", "907", "908", "909", "910", "911", "915", "919", "920"),
        _ci(r"tobacco", r"e-?cigarettes?", r"electronic nicotine", r"cigars?", r"cigarettes?", r"smokeless", r"nicotine",
            r"vap(?:e|es|ing)", r"modified risk", r"harmful and potentially harmful", r"deeming rule",
            # "substantial equivalence" alone is also the 510(k) standard (P-002); its report and the exemption are tobacco
            r"substantial equivalence reports?", r"exemption from (?:the )?(?:substantial equivalence|SE)"),
        _cs(r"ENDS", r"PMTAs?", r"MRTPA?s?", r"HPHCs?", r"SE reports?")),
    Programme(
        # 403A (national uniform nutrition labeling: preemption) and 403B (third-party literature) are not offered: with
        # three Act sections offered per question both would be offered with 403 to every labeling question
        "food_labeling", frozenset({"101", "102", "104", "105"}), ("403",),
        _ci(r"food labels?", r"food labeling", r"nutrition facts", r"nutrition labeling", r"nutrient content claims?",
            r"(?:qualified )?health claims?", r"structure/function claims?", r"reference amounts? customarily consumed",
            r"supplement facts", r"menu labeling",
            # allergens on a label, not cross-contact in a plant (117.135)
            r"allergens?\W+(?:\w+\W+){0,6}(?:label\w*|declar\w*)", r"(?:label\w*|declar\w*)\W+(?:\w+\W+){0,6}(?:food )?allergens?"),
        _cs(r"RACCs?"),
        shared=_ci(r"statement of identity", r"principal display panel", r"serving sizes?", r"vending machines?",
                   r"common or usual name"),
        context=_ci(r"foods?", r"beverages?", r"dietary supplements?")),
    Programme(
        "device_submissions", frozenset({"807", "812", "814", "860"}), ("510", "513", "515", "520"),
        _ci(r"510\(k\)s?", r"premarket notifications?", r"premarket approval", r"De Novo", r"investigational device(?: exemptions?)?",
            r"device classification", r"reclassification", r"classification panels?", r"humanitarian device exemptions?",
            r"513\(g\)", r"device listing"),
        _cs(r"PMAs?", r"IDEs?", r"HDEs?")),
    Programme(
        "radiological_health", frozenset({"900", "1000", "1002", "1003", "1004", "1005", "1010", "1020", "1030", "1040"}),
        # electronic product radiation control (chapter V subchapter C): 534 performance standards (Parts 1010-1040),
        # 537 inspection, records and reports (Part 1002), 535 notification and repair of defects (Parts 1003/1004),
        # 538 prohibited acts and exemptions, 532 the control programme, 531 definitions. Not 536 (imports: a question
        # naming imports is not routed, _OTHER) or 533, 539-542 (studies, enforcement, repealed, State cooperation)
        ("534", "537", "535", "538", "532", "531"),
        _ci(r"mammograph\w*", r"radiation[- ]emitting", r"electronic products?", r"electronic product radiation",
            r"laser products?", r"x-ray", r"sunlamps?", r"accession numbers?", r"microwave ovens?", r"radiation safety"),
        _cs(r"MQSA"),
        no_act=_MAMMOGRAPHY),
    Programme(
        "combination_products", frozenset({"3", "4"}), ("503",),
        _ci(r"combination products?", r"constituent parts?", r"primary mode of action", r"lead (?:agency )?center",
            r"requests? for designation", r"Office of Combination Products"),
        _cs(r"PMOA", r"RFDs?"),
        # a combination product is drug-device-biologic by definition: "a drug-biologic combination product" is one
        subsumes=frozenset({"biologics"})),
    Programme(
        # Subchapter F. The PHS Act section 351 the licensing parts are issued under is not in the statute store, and the
        # FD&C sections in their authority lines (501, 502, 505, 510) are the drug provisions: no Act sections
        "biologics", frozenset({"600", "601", "606", "607", "610", "630", "640", "660", "680"}), (),
        # "a biological product", not the bare plural: "are biological products exempt from these rules?" is another
        # part's scope clause (bank: 5 of 15 plural uses were 1271.330, 1271.390, 310.4, 310.509's own scope questions)
        _ci(r"biological product", r"licensed biological products", r"biological product deviations?",
            r"biologics licen[cs]e(?:s| applications?)?", r"licensed biologics?", r"lot release", r"blood establishments?",
            r"blood components?", r"blood products?", r"whole blood", r"source plasma", r"blood donors?", r"blood donations?",
            r"allergenic (?:products?|extracts?)"),
        _cs(r"BLAs?")),
    Programme(
        # Subchapter E's animal food parts and the medicated-feed parts of Subchapter C (225 medicated feed CGMP, 226
        # Type A medicated articles), 515 (medicated feed mill licenses) and 558 (new animal drugs for use in feed); not
        # the dosage-form new animal drug parts (510-529, 556). Act: 418 (preventive controls, Part 507), 504 (VFD
        # drugs), 512 (new animal drugs; 512(m) feed mill licenses), 409 (food additives, 409(i) animal food)
        "animal_food", frozenset({"225", "226", "500", "501", "502", "507", "509", "515", "558", "570", "571", "573", "579",
                                  "582", "584", "589"}),
        ("418", "504", "512", "409"),
        _ci(r"animal foods?", r"animal feeds?", r"pet foods?", r"medicated (?:animal )?feeds?", r"medicated articles?",
            r"veterinary feed directives?", r"feed mills?", r"livestock feeds?"),
        _cs(r"VFDs?"),
        # a question naming human food as well (by-products of human food held for animal food: 117.95 and 507.28) names
        # both food programmes
        not_with=_ci(r"human foods?"),
        # "new animal drugs" for use in feed and "VFD drugs" are this programme's, not the human drug programme's
        own_other=_ci(r"animal (?:food|feed)s?", r"(?:new )?animal drugs?", r"veterinary feed directive drugs?",
                      r"VFD drugs?")),
)

_PART_NAMED = re.compile(r"\b(?:21\s*CFR|Part|Parts|§)\s*\d", re.I)
# another product programme or the import entry filing (ACE, Part 1 Subpart D) named: routing would hide its sections.
# Not for combination products, which are drug-device-biologic by definition
_OTHER = _ci(r"drugs?", r"over-the-counter", r"imports?", r"imported", r"importers?", r"cosmetics?", r"colou?r(?:s|ed|ing)?",
             r"colou?r additives?", r"colou?rants?", r"animal (?:food|feed)", r"clinical (?:trials?|investigations?)", r"research",
             r"IRBs?")
_OTHER_ACRONYMS = _cs(r"OTC", r"IND", r"NDAs?", r"ANDAs?", r"ACE", r"D&C")


def programme_for(question: str) -> Programme | None:
    """The one programme the question names, or None (none, several, or a 21 CFR part named - the part decides)."""
    q = question or ""
    if _PART_NAMED.search(q):
        return None
    hits = [p for p in PROGRAMMES
            if p.words.search(q) or (p.acronyms is not None and p.acronyms.search(q))
            or (p.shared is not None and p.shared.search(q) and p.context is not None and p.context.search(q))]
    hits = [p for p in hits if not any(p.name in h.subsumes for h in hits)]
    if len(hits) != 1:
        return None
    p = hits[0]
    if p.not_with is not None and p.not_with.search(q):
        return None
    if p.name != "combination_products":
        rest = p.own_other.sub(" ", q) if p.own_other is not None else q
        if _OTHER.search(rest) or _OTHER_ACRONYMS.search(rest):
            return None
    return p
