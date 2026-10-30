"""Section promotion: a question that names no provision is answered as if it had named the section that answers it.

Why (cosmetics benchmark, 22 Oct 2026): a general question reaches the model as "REQUESTED PROVISIONS: none (general
question)" with related passages only, and both Qwen3 models decline it (82 of 88 on 1.7B; 20 of 29 on
Qwen3-4B-Instruct-2507 with an answer-first format). The same questions with their section named are answered, and on
4B-2507 applied to the facts. So the step that was missing is choosing the section, not writing the answer.

How:
  1. rank whole eCFR sections for the question (regulus.retrieval.section_index): top `candidates` sections;
  2. picker: the answering model reads the question and each candidate's heading, part, subpart and best-matching
     sentence, and names the one or two that answer it ("NONE" = no promotion). On 95 bank questions its first choice
     was a key section 80% of the time against 72% for the lexical top 1; the model chooses where to look and never
     supplies regulatory text - the promoted sections are fetched verbatim from the database like any named provision;
  3. companions, mechanical, from the promoted section: a defined term of its family's definitions section (70.3 for
     Parts 70-82, 700.3 for 700-740) that appears in its use or restriction sentences or in the question; 70.5(a) when
     the question names an eye-area product and the section is a color listing; and in Part 82 the "General." section of
     the same subpart. Measured against the bank keys: all needed companions for 52 of 56 questions that need them;
  4. the plan is rebuilt from the question with the citations appended ("... (21 CFR 74.2203, 70.5(a))"), so the
     model sees which provisions were resolved and every later stage treats them as requested provisions.

mode "lexical" promotes the lexical top 1 without a model call (for retrieval-only measurement); "off" does nothing.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

from regulus.retrieval.section_index import (COSMETIC_PARTS, COSMETIC_TERMS, SectionIndex, cosmetic_question, deployment_scope,
                                             section_index)
from regulus.retrieval.statute import BLOCK_HEADER_CHARS, StatuteIndex, statute_index
from regulus.rag.validator_switches import enabled as precision_fix

MODES = ("off", "lexical", "picker")
CANDIDATES = 22
CFR_CANDIDATES_WITH_STATUTE = 20
LEXICAL_CANDIDATES = 16   # cosmetics questions: lexical candidates fused with ...
DENSE_CANDIDATES = 12     # ... the dense top 12 (regulus.retrieval.section_dense)
RRF_K = 10
# a part-only question ("Under Part 710, ...") gets the dense candidates of that part as well (on since 23 Oct, after the
# section-resolution and blind answer gates; REGULUS_PART_DENSE=0 turns it off)
PART_DENSE = os.environ.get("REGULUS_PART_DENSE", "1") == "1"
# a question naming a color additive: other colors' listings vetoed, the named listing offered and, when it ranks
# first, promoted without the picker (regulus.retrieval.entities; on since 23 Oct; REGULUS_ENTITY_CONSTRAINTS=0 turns it off)
ENTITY_CONSTRAINTS = os.environ.get("REGULUS_ENTITY_CONSTRAINTS", "1") == "1"
# the offered FD&C Act sections in order of relevance instead of 201, 601 ... 614 (on since 23 Oct; REGULUS_STATUTE_RANKING=0)
STATUTE_RANKING = os.environ.get("REGULUS_STATUTE_RANKING", "1") == "1"
# a picker reply naming two FD&C Act sections promotes both (the Act budget shared); before, a second pick was kept
# only from the other corpus (on since 23 Oct; REGULUS_PICKER_SECOND_ACT=0 turns it off)
SECOND_ACT = os.environ.get("REGULUS_PICKER_SECOND_ACT", "1") == "1"
# a cross-encoder (regulus.retrieval.rerank) reads each candidate's best paragraphs; a pick it scores below
# RERANK_VETO_BELOW while another candidate scores RERANK_VETO_MARGIN higher is replaced by that one (on since 23 Oct
# with bge-m3, 0.30 / 0.15 fitted on the dev set; REGULUS_RERANKER=off turns it off; without the model files it is off)
RERANKER = (os.environ.get("REGULUS_RERANKER", "bge-m3").strip() or None) if os.environ.get("REGULUS_RERANKER", "bge-m3").strip().lower() not in ("off", "none", "0") else None
RERANK_PARAGRAPHS = 3
RERANK_VETO_BELOW = float(os.environ.get("REGULUS_RERANK_VETO_BELOW", "0.3"))
RERANK_VETO_MARGIN = float(os.environ.get("REGULUS_RERANK_VETO_MARGIN", "0.15"))
V2_TIE = 0.05   # V2: a near tie between the reranker's top and the picker's other choice
STATUTE_BUDGET = 6000   # rendered characters of statute evidence (source headers included)
CO_STATUTE_BUDGET = 2500  # the same when the Act text accompanies a promoted regulation   # fewer CFR candidates when the cosmetics chapter of the Act is offered too
# FD&C Act sections offered to the picker for a cosmetics question: chapter VI (601-614) and the 201 definitions.
# 301 (prohibited acts) and 704 (inspection) are in the store but not offered: they are long, general, and won the
# lexical ranking for questions about everything else (27/56 statute keys in the top 3).
STATUTE_OFFERED = ("201", "601", "602", "603", "604", "605", "606", "607", "608", "609", "610", "611", "612", "613", "614")
_PATH = r"((?:\([a-zA-Z0-9]{1,6}\))*)"
# explicit references to the FD&C Act: accepted as they are
_ACT_REF = re.compile(r"(?:FD&C\s+Act|FDCA|Federal\s+Food,\s+Drug,?\s+and\s+Cosmetic\s+Act)\s*(?:§|section|sec\.)?\s*(\d{3})(?!\d|\.\d)" + _PATH, re.I)
_SIGN_REF = re.compile(r"§\s*(\d{3})(?!\d|\.\d)" + _PATH)
_USC_REF = re.compile(r"21\s+U\.?S\.?C\.?\s*§?\s*(\d{3}[a-z]*)" + _PATH, re.I)
# bare "section 605", "the 610 duties", "under 607": only in a cosmetics question that names no other Act
_SECTION_WORD_REF = re.compile(r"\b(?:section|sec\.)\s*(\d{3})(?!\d|\.\d)" + _PATH, re.I)
_LED_COSMETIC_SECTION = re.compile(r"\b(?:the|under|per)\s+(60[1-9]|61[0-4])(?!\d|\.\d)" + _PATH, re.I)
_OTHER_ACT = re.compile(r"\b(?:Public\s+Health\s+Service|Regulatory\s+Flexibility|Fair\s+Packaging(?:\s+and\s+Labeling)?|"
                        r"Federal\s+Trade\s+Commission|Unfunded\s+Mandates|Paperwork\s+Reduction|Controlled\s+Substances|"
                        r"Federal\s+Meat\s+Inspection|Poultry\s+Products\s+Inspection|Toxic\s+Substances\s+Control)\s+Act\b|"
                        r"\bPHSA?\b|\bFPLA\b|\bTSCA\b|\bCSA\b", re.I)
_SUBPART_REF = re.compile(r"\bsubpart\s+([A-Z])\b")
_BIOLOGIC = re.compile(r"\b(?:biologic(?:al|s)?|blood|plasma|vaccines?|tissues?|HCT/Ps?|allergenic|potency|sterility)\b", re.I)
_SUBSECTION_OF = re.compile(r"\bsubsection\s*\(([a-z0-9]{1,4})\)\s*of\s+(?:FD&C\s+Act\s+)?(?:section|§)\s*(\d{3})", re.I)
_BARE_COSMETIC_SECTION = re.compile(r"(?<![\d.§])\b(60[1-9]|61[0-4])(?!\d|\.\d)" + _PATH)
SNIPPET_CHARS = 200
LEADIN_CHARS = 140   # a sub-paragraph's lead-in shown before it
# 16 tokens cut 33 of 91 holdout replies mid-citation ("21 CFR 740.18, FD&C Act §"), losing every second pick (23 Oct
# failure analysis); greedy decoding keeps the first pick byte-identical under a larger cap. 48 since 23 Oct
# (REGULUS_PICKER_MAX_TOKENS overrides).
PICKER_MAX_TOKENS = int(os.environ.get("REGULUS_PICKER_MAX_TOKENS", "48"))
DEFINITIONS_SECTION = "604"                                    # FD&C Act 604: "In this subchapter:" definitions
# ST1: a regulation citing the Act it implements ("section 403(i) of the act", "sections 501 and 502 of the Federal Food,
# Drug, and Cosmetic Act")
_ACT_CITED = re.compile(r"\bsections?\s+((?:\d{3}[A-Za-z]?(?:\([a-zA-Z0-9]{1,6}\))*(?:\s*,\s*|\s+(?:and|or)\s+)?)+)\s+of\s+the\s+"
                        r"(?:Federal\s+Food,\s+Drug,\s+and\s+Cosmetic\s+)?[Aa]ct\b")
SUBCHAPTER_DEFINED = frozenset(str(n) for n in range(605, 615))  # the sections 604's definitions govern
GENERIC_DEFINED_TERMS = frozenset({"cosmetic product"})        # used in every section; defines nothing a question turns on
DEFINITION_BUDGET = 2400                                      # rendered characters for 604 definitions with a promoted section
PICKER_THINK_BUDGET = 0   # a thinking model picks with its reasoning block closed: the reply is one citation

PICKER_SYSTEM = ("You select which sections of Title 21 of the Code of Federal Regulations or of the FD&C Act answer a "
                 "question. You are given the question and candidate sections, each with its heading and one sentence of its "
                 "text. Judge by what each section is about. Reply with citations only.")
# The reply names the sections ("21 CFR 71.2", "FD&C Act § 607"), not their bracket numbers: with 29 numbered
# candidates Qwen3-4B-Instruct-2507 answered "[27]" (FD&C Act 612) for questions about 607 and 613 (22 Oct).
PICKER_LAST = ("Reply with the citation of the one or two sections whose text most directly answers the question, best "
               "first, separated by a comma, written as shown (for example 21 CFR 71.2 or FD&C Act § 607). Reply NONE if no "
               "candidate answers it.")

_DEF_RE = re.compile(r"^\([a-z]{1,2}\)\s+(?:The terms?\s+)?(.+?)\s+means?\b", re.I)
# CS1: a color additive regulation (Parts 70-82) and the cosmetics sections of the Act it does not carry out
_COLOR_ADDITIVE_PART = re.compile(r"^(?:7\d|8[0-2])\.")
_MOCRA_SECTIONS = frozenset(str(n) for n in range(603, 615))
# QX1 query expansion: the restatement prompt, its length, and how many sections it may add
EXPANSION_SYSTEM = ("You restate questions about U.S. FDA regulations in the formal terms the regulations themselves use. "
                    "You never answer them.")
EXPANSION_PROMPT = ("Question: {question}\n\nRestate the question in one sentence in the formal terms a U.S. regulation on "
                    "this subject would use. Do not answer it and do not cite section numbers. Reply with the sentence only.")
EXPANSION_MAX_TOKENS = 64
EXPANSION_ADD = 4
# SY1: the question asks the agency to act ("ask FDA to", "request that FDA", "apply to FDA for")
_ASKS_AGENCY = re.compile(r"\b(?:ask|asks|asking|request|requesting|apply\s+to)\s+(?:the\s+)?(?:FDA|agency|Commissioner)\s+(?:to|for)\b"
                          r"|\b(?:ask|request)\s+that\s+(?:the\s+)?(?:FDA|agency|Commissioner)\b", re.I)
# RO1: a color listing of any product (food 73.1xxx, cosmetics 73.2xxx, devices 73.3xxx, 74.xxxx, 82.xxx) other
# than Part 82's general sections; 71.25-Q1 went to the device listing 73.3128
_ANY_LISTING = re.compile(r"^(?:7[34]\.\d{3,4}[a-z]?|82\.(?!(?:3|5|6|50|51|1050|1051|2050|2051)$)\d{3,4}[a-z]?)$")
# DF2: the FD&C Act's definition sections, and a question invoking a definition, the Act or FDA's classification
ACT_DEFINITION_SECTIONS = ("201", "604", "608")
_NAMES_ACT = re.compile(r"\b(?:fd&c act|the act\b|federal food, drug,? and cosmetic act|mocra|modernization of cosmetics)")
# SY2 (27 Oct): plain words -> the terms the regulations use for them. "Which parts of our submission become public?"
# is 71.15 (confidentiality, public disclosure), not 71.20 (publication); "used at a higher level to hide discoloration"
# is 71.22 (deceptive use, conceal damage or inferiority)
_THESAURUS = (
    (re.compile(r"\b(?:becomes?|made|make|stays?|remains?|kept|keep)\s+(?:non-?public|public|confidential|secret|private)\b|"
                r"\bpublicly\s+(?:available|released)\b|\b(?:disclosed|released)\s+to\s+the\s+public\b", re.I),
     "public disclosure confidentiality"),
    (re.compile(r"\b(?:hide|hides|hiding|hidden|disguise\w*|cover(?:s|ing)?\s+up)\b|\bto\s+mask\b|\bmasking\b", re.I),
     "conceal deception deceptive"),
)
# SY3 (27 Oct): plain words for the MoCRA exemptions - "a small brand under $1 million" is FD&C Act 612 (Small
# businesses: average gross annual sales); "our acne cream is also an OTC drug" is 613 (a cosmetic product or facility
# also subject to the drug requirements of chapter V)
_THESAURUS_SY3 = (
    (re.compile(r"\bsmall\s+(?:business|company|companies|brand|firm|maker|manufacturer)s?\b|\$\s?1\s?(?:million|m)\b|"
                r"\$\s?1,000,000\b|\b(?:annual|yearly)\s+(?:gross\s+)?(?:sales|revenue)\b", re.I),
     "small businesses average gross annual sales"),
    (re.compile(r"\b(?:also|is|as)\s+(?:an?\s+)?(?:otc|over-the-counter)?\s*drugs?\b|\bOTC\b|\bover-the-counter\b|"
                r"\bdrug\s+(?:and|&)\s+(?:a\s+)?cosmetic\b|\bsubject\s+to\s+(?:the\s+)?drug\b", re.I),
     "also subject to the requirements of subchapter V drugs exemption"),
)
# QD1: a question with several asks ("What fees apply, how do we pay, and can a small company get relief?")
_ASK_START = re.compile(r"(?:^|[,;:]\s*(?:and\s+|or\s+)?|\b(?:and|or)\s+)(?:how|what|when|which|who|where|whether|can|could|must|"
                        r"do|does|is|are|may|should)\b", re.I)


# QD2 (29 Oct, step E2): a comparison or interaction asks about two things as surely as two question marks do - "How does
# the Part 117 supply-chain program differ from the FSVP requirements for importers?" (F-022: 117 Subpart G and 1 Subpart
# L), "... and how do they interact with Part 820?" (M-042)
_COMPARES = re.compile(
    r"\bdifferences?\s+between\b|\bdiffers?\s+from\b|\bhow\s+(?:does|do|is|are)\b[^?]{0,140}?\b(?:differ\w*|different)\b"
    r"|\bcompare[sd]?\s+(?:to|with)\b|\bcomparison\s+(?:of|between)\b|\bversus\b|\bvs\.?\s"
    r"|\binteract\w*\s+with\b|\bhow\s+(?:does|do|is|are)\b[^?]{0,140}?\brelat(?:e|es|ed)\s+(?:to|with)\b", re.I)


def with_reserved(ranked: list, extra: list, cap: int) -> list:
    """`ranked` cut so that `extra` follows it within the first `cap` places (HQ1: appended after 22 candidates, its
    sections were dropped when the FD&C Act step keeps the first 20 - code review)."""
    if not extra:
        return ranked
    room = max(cap - len(extra), 0)   # cut only when ranked would push the additions past the cap
    return list(ranked[:room]) + list(extra)


def multi_part(question: str) -> bool:
    """QD1: the question asks two or more things - two question marks, or two asks in its last sentence ("What fees
    apply, how do we pay, and can a small company get relief?"); with QD2 also a comparison or an interaction."""
    q = (question or "").strip()
    if q.count("?") >= 2:
        return True
    if precision_fix("QD2") and _COMPARES.search(q):
        return True
    last = re.split(r"(?<=[.!])\s+", q)[-1]
    return len(_ASK_START.findall(last)) >= 2


QD1_SYSTEM = "You split questions about U.S. FDA regulations into the separate questions they contain. You never answer them."
QD1_PROMPT = ("Question: {question}\n\nWrite each separate question it asks on its own line, as a complete question that "
              "can be understood alone. Write at most three. Reply with the questions only.")
QD1_MAX_TOKENS = 120
QD1_SUBQUESTIONS = 3
QD1_CANDIDATES = 8
QD1_MIN_SCORE = 0.5
QD1_MAX = 2
QD2_MIN_SCORE = float(os.environ.get("REGULUS_QD2_MIN_SCORE", "0.7"))   # QD2: a cross-part sub-question's section
# EY1: a question saying the listing of its (unnamed) color does not provide for use in the area of the eye
_LISTING_SILENT_ON_EYE = re.compile(
    r"\b(?:listing|listed|regulation|it|its)\b[^.?]{0,40}?\b(?:doesn't|does\s+not|isn't|is\s+not|not)\s+(?:mention|provide|"
    r"cover|include|address|list|say|allow|permit|specif)\w*[^.?]{0,30}?\beye|\bnot\s+(?:listed|approved|permitted|provided)\s+"
    r"for\s+(?:the\s+)?(?:area\s+of\s+the\s+)?eye|\bsilent\s+on\s+eye", re.I)
# DF1 (definitions_asked): the opening of a definition paragraph, the definitions sections read, and what a question
# about the Act's own definitions says
_DEFINES_RE = re.compile(r"^\([a-z]{1,2}\)\s+(?:The\s+terms?\s+|An?\s+|The\s+)?(?P<t>[^.;:]{2,90}?)\s+"
                         r"(?:means?|is\s+(?:any|a|an|the)|refers?\s+to|applies\s+to)\b")
DEFINITION_SECTIONS = ("70.3", "700.3", "82.3")
DEFINITION_HOP_CHARS = 600
HQ1_LIMIT = int(os.environ.get("REGULUS_HQ1_LIMIT", "3"))            # HQ1: sections the question index may add
HQ1_MIN_SCORE = float(os.environ.get("REGULUS_HQ1_MIN_SCORE", "0.75"))  # ... at this cosine similarity or more
DV1_TOP = 3           # DV1: paragraphs promoted for a question naming only a small part
DV1_MIN_SCORE = 0.3
SP3_MAX_SECTIONS = 8      # SP3: a subpart this small is promoted whole with its section
SP3_MAX_CHARS = 4500
SP3_PARTS = ("111",)      # parts built of small requirement-and-records subparts (dietary supplement CGMP)
HC1_MAX = 2   # HC1: sibling sections added for topics the promoted heading misses
# MS1: further sections of the promoted section's part, best cross-encoder score first, above a floor, within a cap
MS1_MIN_SCORE = float(os.environ.get("REGULUS_MS1_MIN_SCORE", "0.7"))
MS1_MAX = int(os.environ.get("REGULUS_MS1_MAX", "2"))
MS1_MAX_CHARS = int(os.environ.get("REGULUS_MS1_MAX_CHARS", "6000"))
XC1_MIN_SCORE = float(os.environ.get("REGULUS_XC1_MIN_SCORE", "0.85"))  # XC1: a candidate of another part, this sure
XC1_MAX = 1
PL1_MAX = int(os.environ.get("REGULUS_PL1_MAX", "3"))                    # PL1: added sections allowed, paragraph-level
PL1_WHOLE_CHARS = int(os.environ.get("REGULUS_PL1_WHOLE_CHARS", "2500"))  # PL1: a section this short is added whole
MS2_POOL = int(os.environ.get("REGULUS_MS2_POOL", "12"))
AP1_TOP = int(os.environ.get("REGULUS_AP1_TOP", "12"))   # AP1: Act paragraphs scored by the cross-encoder per section   # MS2: structural neighbours scored per question
SP2O_MARGIN = float(os.environ.get("REGULUS_SP2O_MARGIN", "0.15"))   # SP2O: score lead that puts the second pick first
AR2_MIN_SCORE = float(os.environ.get("REGULUS_AR2_MIN_SCORE", "0.7"))   # AR2: a section found for an absence claim
AR2_LEXICAL = 5                                                            # AR2: lexical hits scored per topic
PC1_MIN_SCORE = float(os.environ.get("REGULUS_PC1_MIN_SCORE", "0.95"))   # PC1: cross-encoder confidence that skips the picker
PC1_MARGIN = float(os.environ.get("REGULUS_PC1_MARGIN", "0.10"))
# NAV1: table-of-contents navigation - the subpart's headings shown to the model, which names the further sections needed
NAV1_MAX = int(os.environ.get("REGULUS_NAV1_MAX", "3"))
NAV1_MAX_CHARS = int(os.environ.get("REGULUS_NAV1_MAX_CHARS", "6000"))
AU2_MIN_SCORE = float(os.environ.get("REGULUS_AU2_MIN_SCORE", "0.5"))   # AU2: an implementing regulation joins a lone Act section
NAV1_TOC_LIMIT = 60     # v2: the whole subpart (H-007's key 1.1455 was line 40 of subpart S and was cut at 30)
NAV1_SNIPPET = 120      # v2: each entry's opening text - a heading alone ("Materials examination and usage criteria",
                        # D-016's labeling key 211.122) does not say what a section covers
NAV2_LIMIT = 15   # NAV2: sections of the part outside the subpart, closest to the question, added to NAV1's contents
NAV1_MAX_TOKENS = 80   # v2: 40 cut F-012's reply mid-citation ("..., 21 CFR 117.")
NAV1_SYSTEM = ("You help find the sections of a U.S. regulation that answer a question. One section has already been "
               "chosen. Below is a numbered list of the other sections of its subpart, each with its heading and opening "
               "words. Reply with the list numbers of the sections the question ALSO needs - other steps, records, "
               "definitions, exceptions or conditions it asks about - most needed first, at most three, like: 3, 7. Use "
               "only numbers from the list. If it needs none of them, reply none.")
# SP2F: the picker's second pick (SP2) stands only when the cross-encoder scores it at least this high. Calibrated 28 Oct on
# the 58 dev second picks of the SP2 live run (0.7: 12 gains kept, 5 -> 2 losses, 0 PASS lost); on the 21 sealed ones it
# then blocked all 3 losses (2 of them PASS) and kept 3 of 5 gains
SP2F_MIN_SCORE = float(os.environ.get("REGULUS_SP2F_MIN_SCORE", "0.7"))
_HC1_GENERIC = set("""requirement requirements required requires apply applies applicable drugs product products
general regulation regulations section sections subpart provision provisions current practice manufacturing what which
under their there these those other""".split())


def _coordinated(question: str) -> list[tuple[set[str], set[str]]]:
    """HC1: the topic stems on each side of every "and" / "or" in the question, four words either way."""
    words = re.findall(r"[a-z][a-z-]*", (question or "").lower())
    return [(_topic_stems(" ".join(words[max(0, i - 4):i])), _topic_stems(" ".join(words[i + 1:i + 5])))
            for i, w in enumerate(words) if w in ("and", "or")]


_GENERIC_HEADING = re.compile(r"^\s*(?:scope|definitions?|purpose|applicability|general(?:\s+provisions|\s+requirements)?|"
                              r"authority|severability|exemptions?)\.?\s*$", re.I)
# DQ1: a term the question asks the meaning of - quoted, or "who is the X", "what is a X", "definition / meaning of X"
_ASKED_TERM = re.compile(
    r"\b(?:who|what)\s+(?:is|are)\s+(?:an?|the)\s+['\"‘“]?([a-z][a-z -]{2,40}?)['\"’”]?(?=[?,.]|\s+(?:under|in|for|and)\b)"
    r"|\b(?:definition|meaning) of (?:an?|the)?\s*['\"‘“]?([a-z][a-z -]{2,40}?)['\"’”]?(?=[?,.]|\s)"
    r"|['‘\"“]([A-Za-z][A-Za-z -]{2,40})['’\"”]", re.I)
# SN1: "under Part 117 Subpart B" - the letter is matched in capitals only, so "subpart of" is never a subpart
_NAMED_SUBPART = re.compile(r"\b[Pp]art\s+(\d{1,4})\b[^?.]{0,40}?\b[Ss]ubpart\s+([A-Z]{1,2})\b")
# ... or by its heading: "under the Part 111 personnel subpart" (Subpart B - Personnel)
_NAMED_SUBPART_HEADING = re.compile(r"\b[Pp]art\s+(\d{1,4})\s+([a-z][a-z-]+(?: [a-z][a-z-]+)?)\s+subpart\b")
# "What definitions apply to this part?" (111.3, 112.3, 117.3): a heading about its own part is never another part's twin
_OWN_PART_HEADING = re.compile(r"\bthis (?:part|subpart)\b", re.I)


def heading_phrase_in(heading: str, question: str) -> bool:
    """HM1 / LS2: the heading's lead phrase (before its first comma), two words or more, occurs in the question word for
    word - endings aside ("Rejected components" in "for rejected components")."""
    stem = lambda w: w[:6]
    lead = [stem(w) for w in re.findall(r"[a-z]+", re.split(r"[,;]", (heading or "").lower())[0])]
    if len([w for w in lead if len(w) >= 5]) < 2:
        return False
    q = [stem(w) for w in re.findall(r"[a-z]+", (question or "").lower())]
    return any(q[i:i + len(lead)] == lead for i in range(len(q) - len(lead) + 1))


def _topic_stems(text: str) -> set[str]:
    """Topic words of a question or heading, as six-letter stems ("salvaged" / "salvaging" -> "salvag")."""
    return {w[:6] for w in re.findall(r"[a-z]{5,}", (text or "").lower()) if w not in _HC1_GENERIC}


DV1_WHOLE_SECTION_CHARS = int(os.environ.get("REGULUS_DV1_WHOLE_SECTION_CHARS", "2500"))   # DV1: a section this short is supplied whole (820.7, 820.45, 820.35: M-023 mis-cited (a)(3) from its families)
DV1_MAX_SECTIONS = 12  # a part this small is answered paragraph by paragraph
# DV2: a question asking for the part's scope or its definitions as a whole, and the headings of the section that holds them
_FRAME_ASKED = {
    "scope": re.compile(r"\bscope\s+of\b", re.I),
    "definitions": re.compile(r"\b(?:which|what)\s+definitions\b|\bdefinitions\s+(?:that\s+)?apply\b", re.I),
}
_FRAME_HEADINGS = {"scope": re.compile(r"^\s*(?:scope|purpose|applicability)\b", re.I),
                   "definitions": re.compile(r"^\s*definitions?\b", re.I)}
_PHRASE_STOP = set("the and for with that this from such any are its which under part section".split())
FC1_MAX_SECTIONS = 3   # FC1: a heading phrase this rare in Parts 70-71 names the subject a fee paragraph prices
DEFINITION_GENERIC_USES = 3
_ACT_DEFINITION_RE = re.compile(r"\b(?:FD&C\s+Act|the\s+Act\b|MoCRA|Modernization|section\s+\d{3}\b|§\s*\d{3}\b|60[1-9]\b|61[0-4]\b|201\b)", re.I)
_CLASSIFY_RE = re.compile(r"\b(?:classify|classifies|treat|treats|count\s+as|counts\s+as|in\s+fda['’]?s?\s+sense)\b")
# plain words for a defined one-word term, among what a question asks to classify
_PLAIN_TERMS = {"blend": "mixture", "blends": "mixture", "carrier": "diluent", "oil": "diluent", "solvent": "diluent"}
_USE_RE = re.compile(r"\b(?:safely used|restricted to|uses and restrictions|for use (?:in|only)|may be used)\b", re.I)
_SENT_RE = re.compile(r"(?<=[.;])\s+")
# a listing that speaks to eye-area use: "area of the eye", or names eye products ("eyeliner, brush-on-brow, eye shadow")
_EYE_USE = re.compile(r"area of the eye|eye ?liner|eye ?shadow|mascara|eyebrow|brush-on-brow|eye makeup", re.I)
EYE_TERMS = frozenset({"eye", "eyelin", "eyebrow", "mascara", "shadow"})   # 21 CFR 720.4(c)(3), stemmed
COLOR_LISTING_PARTS = frozenset({"73", "74", "82"})

_XREF_SQL = """
SELECT DISTINCT x.target_citation FROM cfr_cross_references x JOIN cfr_records r ON r.id = x.from_record_id
WHERE r.section_number = %(s)s AND r.removed_on IS NULL AND x.resolvable
"""
_SNIPPET_SQL = """
SELECT section_number, ordinal, paragraph_path, body_text FROM cfr_records
WHERE removed_on IS NULL AND section_number = ANY(%(s)s) AND ordinal = ANY(%(o)s)
"""
_LEADIN_SQL = """
SELECT section_number, paragraph_path, body_text FROM cfr_records
WHERE removed_on IS NULL AND section_number = ANY(%(s)s) AND paragraph_path = ANY(%(p)s)
"""
# XP1: the cross-references of a section's rows, or of one paragraph's rows (and its sub-paragraphs)
_XREF_SQL_PATH = """
SELECT DISTINCT x.target_citation FROM cfr_cross_references x JOIN cfr_records r ON r.id = x.from_record_id
WHERE r.section_number = %(s)s AND r.removed_on IS NULL AND x.resolvable
  AND (%(p)s = '' OR r.paragraph_path = %(p)s OR r.paragraph_path LIKE %(pp)s)
"""
_SECTION_SQL = """
SELECT paragraph_path, body_text FROM cfr_records
WHERE removed_on IS NULL AND NOT is_reserved AND section_number = %(s)s ORDER BY ordinal
"""


@dataclass(frozen=True)
class Promotion:
    sections: tuple[str, ...]          # the promoted section(s), "74.2203"
    companions: tuple[str, ...]        # "70.3(v)", "70.5(a)", "82.1050"
    method: str
    statute: tuple[str, ...] = ()      # promoted FD&C Act section(s), "605"; their paragraphs become the evidence
    statute_rows: tuple[int, ...] = () # StatuteIndex row indices placed in the prompt
    statute_blocks: tuple[tuple[int, ...], ...] = ()     # the same rows grouped into evidence passages
    statute_left_out: tuple[tuple[int, ...], ...] = ()   # blocks of the promoted section(s) the budget kept out
    candidates: tuple[dict, ...] = ()
    picker_output: str | None = None
    seconds: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)
    # the cross-encoder's score (0..1) of the promoted section for the question, when the reranker read it (Q2)
    rerank: float | None = None
    # AMB1 (method "product_clarification"): {product family: ((section, heading, subpart), ...)} offered to the reader
    choices: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...] = ()

    @property
    def citations(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.sections + self.companions))

    @property
    def suffix(self) -> str:
        return " (21 CFR " + ", ".join(self.citations) + ")" if self.citations else ""

    def to_dict(self) -> dict[str, Any]:
        return {"sections": list(self.sections), "companions": list(self.companions), "method": self.method,
                "statute": list(self.statute), "statute_rows": len(self.statute_rows),
                "statute_blocks_left_out": len(self.statute_left_out),
                "citations": list(self.citations), "picker_output": self.picker_output, "seconds": round(self.seconds, 2),
                "candidates": [c["section"] for c in self.candidates], "notes": list(self.notes), "rerank": self.rerank,
                **({"choices": {k: [list(c) for c in v] for k, v in self.choices}} if self.choices else {})}


def rerank_label(c: dict) -> str:
    """A candidate's citation with where it sits, for the cross-encoder: the same color is listed word for word in the
    Foods, Drugs and Cosmetics subparts, and without the subpart the reranker could not tell 73.85 from 73.2085."""
    if c.get("statute"):
        return f"FD&C Act § {c['section']} (Federal Food, Drug, and Cosmetic Act)"
    where = c.get("part_title") or f"Part {c.get('part', '')}"
    if c.get("subpart"):
        where += f"; {c['subpart'].title()}"
    return f"21 CFR {c['section']} ({where})"


def picker_prompt(question: str, candidates: list[dict]) -> str:
    lines = [f"QUESTION: {question}", "", "CANDIDATE SECTIONS:"]
    for i, c in enumerate(candidates, 1):
        if c.get("statute"):
            lines.append(f"[{i}] FD&C Act § {c['section']} [21 U.S.C. {c['usc']}] {c['heading']} (Federal Food, Drug, and "
                         f"Cosmetic Act, statute) Covers: {c['snippet']}")
            continue
        where = c.get("part_title") or f"Part {c['part']}"
        if c.get("subpart"):
            where += f"; {c['subpart'].title()}"
        lines.append(f"[{i}] 21 CFR {c['section']} {c['heading']} ({where}) Text: \"{c['snippet']}\"")
    lines += ["", PICKER_LAST]
    return "\n".join(lines)


_CFR_NUM = re.compile(r"\b(\d{1,4}\.\d+[a-z]?)\b")
_ACT_NUM = re.compile(r"(?:§|[Ss]ection|Act)\s*(\d{3})(?!\d|\.\d)")
_CITATION_LIKE = re.compile(r"\d\.\d|CFR|§|\bAct\b|[Ss]ection|U\.S\.C", re.I)


# SA1: FD&C Act sections offered to a general (not cosmetics) question; 301 (prohibited acts) is long and general and
# won the lexical ranking for unrelated questions, as it did for cosmetics questions
GENERAL_ACT_OFFERED = 3
_ACT_NOT_OFFERED = frozenset({"301"})


_AU1_GENERIC = frozenset({"201", "301", "704", "801"})   # definitions, prohibited acts, inspection, imports: in most lists


def named_act_path(section: str, question: str, raw: str | None) -> str:
    """AP2 (30 Oct): the subsection of a picked FD&C Act section that the question writes ("what does a 513(g) request
    accomplish?") or, failing that, the picker wrote ("FD&C Act § 513(g)"), so its blocks are supplied first
    (StatuteIndex.blocks keeps a named block before ranking the rest). P-008: the picker named 513(g), the path was
    dropped, and word ranking supplied 513(b), (c) and (f) and left (g) among 54 blocks out - the model then said 513(g)
    was not in the sources. "" when neither names one, or AP2 is off."""
    if not precision_fix("AP2"):
        return ""
    rx = re.compile(rf"(?<![\d.]){re.escape(section)}((?:\([A-Za-z0-9]{{1,4}}\))+)")
    for text in (question, raw):
        m = rx.search(text or "")
        if m:
            return m.group(1)
    return ""


def routed_programme(question: str):
    """DR1 (regulus.rag.validator_switches): the one programme the question names (regulus.retrieval.programmes), or None."""
    if not precision_fix("DR1"):
        return None
    from regulus.retrieval.programmes import programme_for

    return programme_for(question)


def authority_sections(parts: list[str]) -> list[str]:
    """AU1: the FD&C Act sections these parts are issued under, in their first part's order then the next part's, from
    data/reference/part_fdca_authority.json (eCFR <AUTH> lines mapped to the Act sections held)."""
    from regulus.retrieval.part_authority import authority

    return list(dict.fromkeys(s for p in dict.fromkeys(parts) for s in authority(p)))


def general_act_offered() -> bool:
    """SA1 (regulus.rag.validator_switches): a question outside the cosmetics programme is offered the FD&C Act sections
    it is most about (tobacco 904-911, devices 510-520, food labeling 403 ...), as a cosmetics question is offered
    chapter VI. 26 Oct: 25 of the 225 300-question dev rows are keyed to the Act; until then a general question could
    reach Act text only by naming the section."""
    from regulus.rag.validator_switches import enabled

    return enabled("SA1")


def part_only_any() -> bool:
    """PO2 (regulus.rag.validator_switches): a question naming only a 21 CFR part outside the cosmetics programme
    ("What are the reserve sample requirements under Part 211?") is resolved to its section(s) inside that part, as
    "Under Part 82, ..." is. 26 Oct: 74 of the 225 300-question dev rows; inside the named part the lexical ranking puts
    a key section first for 57 and in the top two for 61, where until then the question was answered from part-wide
    retrieval and 1 of 14 scored rows passed."""
    from regulus.rag.validator_switches import enabled

    return enabled("PO2")


def second_in_part(picks: list[int], cands: list[dict], picked: dict) -> str | None:
    """SP2 (regulus.rag.validator_switches): the picker's second regulation, promoted as a companion of the first when
    it is in the same part, outside the cosmetics programme (whose companions have their own rules). 26 Oct, 35
    recorded 300-question picks: the key needs more than one section for 21; a second pick in the first pick's part was a
    key section 11 times of 14, one in another part 1 of 7; in the cosmetics bank and follow-ups 0 of 4."""
    from regulus.rag.validator_switches import enabled

    if not enabled("SP2") or picked.get("statute") or picked["part"] in COSMETIC_PARTS:
        return None
    for i in picks[1:]:
        c = cands[i - 1]
        if not c.get("statute") and c["part"] == picked["part"] and c["section"] != picked["section"]:
            return c["section"]
    return None


def parse_picks(text: str, n: int, sections: list[str] | None = None) -> list[int]:
    """Candidate positions (1-based) the picker's reply names, best first, at most two.

    The reply is asked for citations ("21 CFR 71.2", "FD&C Act § 607"); each is mapped to the candidate carrying it.
    A reply that cites something that is not a candidate gives [] - its digits are never read as positions ("21 CFR
    70.3" would have meant candidates 21 and 3). Bare positions are accepted only as "[n]" or a reply of numbers."""
    if not text or (re.search(r"\bnone\b", text, re.I) and not re.search(r"\d", text)):
        return []
    out: list[int] = []
    if sections:
        cited = _CFR_NUM.findall(text) + _ACT_NUM.findall(_CFR_NUM.sub(" ", text))
        for x in cited:
            if x in sections and sections.index(x) + 1 not in out:
                out.append(sections.index(x) + 1)
        if out or _CITATION_LIKE.search(re.sub(r"\[\d+\]", " ", text)):
            return out[:2]
    numbers = re.findall(r"\[(\d+)\]", text) or (re.findall(r"\d+", text) if re.fullmatch(r"[\d,\s\[\]]+", text.strip()) else [])
    for x in numbers:
        i = int(x)
        if 1 <= i <= n and i not in out:
            out.append(i)
    return out[:2]


COSMETIC_TERMS_GENERIC = frozenset({"cosmet", "label", "product", "compani", "manufactur", "make", "sell", "use"})
_PART_QUESTION_WORDS = frozenset({"part", "cfr", "requir", "cover", "say", "provid", "regul", "apply", "appli", "rule", "contain",
                                  "establish", "address", "set", "govern", "21", "subpart", "general", "overview"})
_SUBPART_OF = (("cosmetic", "cosmetic"), ("drug", "drug"), ("food", "food"), ("device", "device"))
_NON_COSMETIC_TERMS = {"drug": frozenset({"drug", "tablet", "capsul", "oral", "ingest"}),
                       "food": frozenset({"food", "beverag", "candi"}), "device": frozenset({"devic", "sutur", "lens", "implant"})}


# product words that say which subpart of a color listing a question is about (stemmed, PostgreSQL 'english'); used only
# when no reranker reads the listings (SectionPromoter.choose_listing)
_PRODUCT_KIND_TERMS = {
    "food": frozenset("""food beverag drink juic punch soda cola sauc snack confect candi dessert ice cereal bake bread cake
                         gum gelatin feed anim pet fish salmon shrimp trout egg yolk meat sausag poultri chicken dairi chees
                         milk yogurt season spice popcorn margarin oil-bas jam jelli frost soup gravi beer wine spirit""".split()),
    "drug": frozenset("drug tablet capsul oral ingest prescript medic pharmaceut ointment otc".split()),
    "device": frozenset("devic sutur lens implant contact catheter".split()),
}
# cosmetic product words (COSMETIC_TERMS without the color words every color question has), with the eye makeup words of
# 21 CFR 720.4(c)(3) ("eyebrow pencil", "eye shadow") - "green eye pencil" named no product kind (73.2326, 23 Oct)
_COSMETIC_PRODUCT_TERMS = (COSMETIC_TERMS - {"color", "colorant", "pigment", "lake", "fd", "tint", "glitter", "shimmer"}) | {"eye", "pencil"}


def _subpart_kind(subpart: str) -> str | None:
    s = (subpart or "").lower()
    return next((k for key, k in _SUBPART_OF if key in s), None)


class SectionPromoter:
    def __init__(self, db, generator=None, mode: str = "picker", candidates: int = CANDIDATES,
                 index: SectionIndex | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f"section_promotion must be one of {MODES}, got {mode!r}")
        self.db, self.generator, self.mode, self.limit = db, generator, mode, candidates
        self._index = index
        self._defs: dict[str, list[tuple[str, list[str]]]] | None = None
        self._part_titles: dict[str, str] | None = None

    @property
    def statute(self) -> StatuteIndex | None:
        return statute_index(self.db)

    def statute_candidates(self, lexemes, question: str | None = None, offered: tuple[str, ...] = STATUTE_OFFERED) -> list[dict]:
        st = self.statute
        if st is None:
            return []
        ranked = {r.fdca: r for r in st.rank(lexemes, limit=len(st.by_sec))}
        order = self.statute_order(question, ranked, offered) if (STATUTE_RANKING and question) else offered
        out = []
        for s in order:
            if s not in st.by_sec:
                continue
            r = ranked.get(s)
            row = st.rows[r.best if r else st.by_sec[s][0]]
            out.append({"statute": True, "section": s, "usc": st.usc(s), "heading": st.heading(s), "part": "FDCA",
                        "subpart": "", "part_title": "", "score": round(r.score, 4) if r else 0.0,
                        # the section's own subsection captions: what it covers (retrieval diagnosis, 22 Oct)
                        "snippet": st.outline(s) or " ".join(row["body_text"].split())[:SNIPPET_CHARS]})
        return out

    def rerank_scores(self, question: str, cands: list[dict]) -> dict[str, tuple[float, str]]:
        """Each candidate section's best cross-encoder score over its RERANK_PARAGRAPHS paragraphs most similar to the
        question by embedding: section -> (score 0..1, that paragraph's path). {} without the models."""
        from regulus.retrieval.paragraph_dense import paragraph_dense
        from regulus.retrieval.rerank import reranker

        pd, rr = paragraph_dense(self.db), reranker(RERANKER) if RERANKER else None
        if pd is None or rr is None:
            return {}
        sims = pd.similarities(question)
        passages, owners = [], []
        for c in cands:
            label = rerank_label(c)
            for para, _ in pd.section_paragraphs(question, c["section"], sims)[:RERANK_PARAGRAPHS]:
                passages.append(f"{label} {pd.text(para)}")
                owners.append((c["section"], para.path))
        best: dict[str, tuple[float, str]] = {}
        for (sec, path), s in zip(owners, rr.score(question, passages) if passages else ()):
            if sec not in best or s > best[sec][0]:
                best[sec] = (float(s), path)
        self.__dict__["last_rerank"] = best   # diagnostics for the caller (the harness records it)
        return best

    def rerank_order(self, question: str, cands: list[dict]) -> list[dict]:
        """RO1 (27 Oct): the regulation candidates reordered by reciprocal-rank fusion of their lexical rank and the
        cross-encoder's score before the picker reads them - the reorder-then-pick step of retrieval systems, since a
        language model choosing from a list favours what it reads first; the FD&C Act candidates keep their place
        after them. The scores are kept for the veto (no second scoring)."""
        scores = self.rerank_scores(question, cands)
        if not scores:
            return cands
        self.__dict__["_order_scores"] = (question, tuple(c["section"] for c in cands), scores)
        cfr = [c for c in cands if not c.get("statute")]
        # a question that names no color is not about one color's listing: the cross-encoder favours listings for any
        # "exempt from certification" question (27 Oct: 71.25-Q1 -> 73.1001, 71.26-Q1 -> 73.3124), so listings keep
        # their lexical place and take no reranker credit (118 dev picker questions: key first 57 -> 58, top 3 77 -> 81)
        from regulus.retrieval.entities import entity_index

        ents = entity_index(self.index)
        no_color = not ents.match(question)
        listing = {c["section"] for c in cfr if no_color and (ents.is_color_listing(c["section"]) or _ANY_LISTING.match(c["section"]))}
        scored = [c for c in cfr if c["section"] not in listing]
        by_rr = sorted(scored, key=lambda c: -scores.get(c["section"], (0.0, ""))[0])
        rr_rank = {c["section"]: i for i, c in enumerate(by_rr, 1)}
        fused = sorted(cfr, key=lambda c: -(1 / (RRF_K + cfr.index(c) + 1)
                                           + (0.0 if c["section"] in listing else 1 / (RRF_K + rr_rank[c["section"]]))))
        return fused + [c for c in cands if c.get("statute")]

    def reranked_companions(self, question: str, cands: list[dict], pr: "Promotion") -> list[str]:
        """MS1 (28 Oct): rerank, threshold, budget - the other regulation candidates of the promoted section's part whose
        cross-encoder score reaches MS1_MIN_SCORE, best first, at most MS1_MAX and MS1_MAX_CHARS characters. The picker
        names at most two sections and only its first was kept: 300Q, 27 Oct, 68 wrong answers had a key section among
        the candidates and not in the prompt, 44 of them ranked 1-3. Outside the cosmetics programme, whose companions
        have their own rules. The scores are RO1's (no second scoring when RO1 ordered the same candidates)."""
        if not pr.sections or pr.sections[0].split(".")[0] in COSMETIC_PARTS:
            return []
        kept = self.__dict__.get("_order_scores")
        scores = (kept[2] if kept and kept[0] == question and set(kept[1]) == {c["section"] for c in cands}
                  else self.rerank_scores(question, cands))
        part, have = pr.sections[0].split(".")[0], {c.split("(")[0] for c in pr.citations}
        pool = [c for c in cands if not c.get("statute") and c["part"] == part and c["section"] not in have]
        if precision_fix("MS2"):
            # MS2: the promoted section's structure joins the pool - its subpart's sections and the sections of its part
            # it cites - scored by the same cross-encoder (300Q 27 Oct: of 148 answers lacking a key section, 66 had
            # every missing section in a supplied section's subpart or cited by it)
            extra = [{"section": s, "part": part} for s in self.structural_neighbours(pr.sections[0].split("(")[0])
                     if s not in have and s not in {c["section"] for c in pool}]
            if extra:
                scores = {**scores, **self.rerank_scores(question, extra)}
                pool += extra
        out, chars = [], 0
        cap = max(MS1_MAX, PL1_MAX) if precision_fix("PL1") else MS1_MAX
        for c in sorted(pool, key=lambda c: -scores.get(c["section"], (0.0, ""))[0]):
            score, path = scores.get(c["section"], (0.0, ""))
            if score < MS1_MIN_SCORE or len(out) >= cap:
                break
            n = len(self._section_text(c["section"]))
            if precision_fix("PL1") and path and n > PL1_WHOLE_CHARS:
                # PL1 (29 Oct, step E3): a long section added as context brings the top-level paragraph its best-scored
                # paragraph sits in, with everything under it (small-to-big passage retrieval), never the whole section:
                # E1/E1b, 117.475 added whole filled the paragraph limit and pushed F-020's key 117.405 out of the prompt
                top = re.match(r"\([^)]+\)", path)
                if top is not None:
                    para = c["section"] + top.group(0)
                    n = sum(len(r["body_text"] or "") for r in self.db.query(_SECTION_SQL, {"s": c["section"]})
                            if (r["paragraph_path"] or "").startswith(top.group(0)))
                    if 0 < n and chars + n <= MS1_MAX_CHARS:
                        out.append(para)
                        chars += n
                    continue
            if chars + n <= MS1_MAX_CHARS:
                out.append(c["section"])
                chars += n
            elif precision_fix("MS3") and path:
                # MS3 (28 Oct): small-to-big at paragraph level - a section too large to add whole brings the top-level
                # paragraph its best-scored paragraph sits in, with everything under it (pf49b, client keys: L-008's
                # key is 101.9(b) and 101.12, and 101.12 - the reference-amounts tables - never fit whole)
                m3 = re.match(r"\([^)]+\)", path)
                if m3 is None:   # an unusual path: no paragraph family to take (code review: AttributeError lost the promotion)
                    continue
                top = m3.group(0)
                para = c["section"] + top
                n = sum(len(r["body_text"] or "") for r in self.db.query(_SECTION_SQL, {"s": c["section"]})
                        if (r["paragraph_path"] or "").startswith(top))
                if 0 < n and chars + n <= MS1_MAX_CHARS:
                    out.append(para)
                    chars += n
        if precision_fix("XC1"):
            out += self._cross_part_companion(cands, scores, part, have | {x.split("(")[0] for x in out}, chars)
        return out

    def _cross_part_companion(self, cands: list[dict], scores: dict, part: str, have: set, chars: int) -> list[str]:
        """XC1 (29 Oct, step E4): the best regulation candidate from a part other than the promoted section's, when the
        cross-encoder scores it at least XC1_MIN_SCORE - at most XC1_MAX, within MS1_MAX_CHARS, a long section as its best
        paragraph family. MS1 pools the promoted section's part only, so a candidate from another part was never scored
        for addition (dev 225: 8 key sections, e.g. U-001 801.1 at rank 2 / 0.99, G-006 211.137 at rank 1). Unlike XP1
        (cross-part citations, which brought 112.4 into F-020) only the question's own candidates are considered."""
        out: list[str] = []
        others = [c for c in cands if not c.get("statute") and c.get("part") != part and c["section"] not in have]
        for c in sorted(others, key=lambda c: -scores.get(c["section"], (0.0, ""))[0]):
            score, path = scores.get(c["section"], (0.0, ""))
            if score < XC1_MIN_SCORE or len(out) >= XC1_MAX:
                break
            n = len(self._section_text(c["section"]))
            label = c["section"]
            top = re.match(r"\([^)]+\)", path or "")
            if n > PL1_WHOLE_CHARS and top is not None:
                label = c["section"] + top.group(0)
                n = sum(len(r["body_text"] or "") for r in self.db.query(_SECTION_SQL, {"s": c["section"]})
                        if (r["paragraph_path"] or "").startswith(top.group(0)))
            if 0 < n and chars + n <= MS1_MAX_CHARS:
                out.append(label)
                chars += n
        return out

    def structural_neighbours(self, section: str, limit: int = MS2_POOL) -> list[str]:
        """MS2: the sections of `section`'s subpart nearest to it, then the sections of its part it cites
        (cfr_cross_references), at most `limit`."""
        part = section.split(".")[0]
        try:
            sib = [r["s"] for r in self.db.query(
                "SELECT section_number s, min(ordinal) o FROM cfr_records WHERE removed_on IS NULL AND NOT is_reserved AND "
                "subpart_id = (SELECT subpart_id FROM cfr_records WHERE section_number = %(s)s AND removed_on IS NULL AND "
                "subpart_id IS NOT NULL LIMIT 1) GROUP BY 1", {"s": section}) if r["s"] != section]
            cited = []
            for r in self.db.query(_XREF_SQL, {"s": section}):
                m = re.match(r"^21 CFR (\d+)\.(\d+[a-z]?)", r["target_citation"] or "")
                if m and m.group(1) == part and f"{m.group(1)}.{m.group(2)}" != section:
                    cited.append(f"{m.group(1)}.{m.group(2)}")
        except Exception:  # noqa: BLE001 - an aid only
            return []
        num = lambda s: float(s.split(".")[1].rstrip("abcdefghijklmnopqrstuvwxyz") or 0)
        sib.sort(key=lambda s: abs(num(s) - num(section)))
        return list(dict.fromkeys(cited + sib))[:limit]

    def act_paragraph_scores(self, question: str, fdca: str) -> dict[str, float]:
        """AP1 (28 Oct): the FD&C Act section's paragraphs most similar to the question by embedding (AP1_TOP), scored by
        the cross-encoder: paragraph path -> score. 910 is 13,500 characters and 2,500 of it fit beside a regulation;
        word overlap chose blocks without 910(b)(1)'s application contents (300Q T-001, SA1 A/B 26 Oct). {} without
        the models."""
        from regulus.retrieval.paragraph_dense import paragraph_dense
        from regulus.retrieval.rerank import reranker

        pd, rr = paragraph_dense(self.db), reranker(RERANKER) if RERANKER else None
        if pd is None or rr is None:
            return {}
        top = pd.section_paragraphs(question, fdca)[:AP1_TOP]
        if not top:
            return {}
        label = f"FD&C Act § {fdca}"
        scores = rr.score(question, [f"{label} {pd.text(para)}" for para, _ in top])
        return {para.path: float(s) for (para, _), s in zip(top, scores)}

    def absence_sections(self, question: str, topics: list[str], promotion: "Promotion", limit: int = 2) -> list[str]:
        """AR2 (28 Oct): sections of the promoted section's part that hold what a draft said the provisions do not
        specify - each topic ranked lexically inside the part, the best AR2_LEXICAL scored by the cross-encoder against
        the question and the topic, kept at AR2_MIN_SCORE or more. Active retrieval: the draft's "is not specified" is
        read as a missing passage (300Q 27 Oct: 47 model answers said so, 23% PASS+PP vs 43%; in 29 of the 37 that
        failed, a key section was not in the prompt - H-007 'retention ... is not specified', 1.1455 holds it)."""
        if not promotion.sections:
            return []
        part = promotion.sections[0].split(".")[0]
        have = {c.split("(")[0] for c in promotion.citations}
        out: list[str] = []
        for topic in topics:
            lex = SectionIndex.question_lexemes(self.db, topic)
            ranked = [r for r in self.index.rank(lex, limit=200) if r.part == part and r.section not in have
                      and r.section not in out][:AR2_LEXICAL]
            if not ranked:
                continue
            scores = self.rerank_scores(f"{question} {topic}", [{"section": r.section, "part": r.part, "subpart": r.subpart}
                                                                for r in ranked])
            best = max(ranked, key=lambda r: scores.get(r.section, (0.0, ""))[0])
            if scores.get(best.section, (0.0, ""))[0] >= AR2_MIN_SCORE:
                out.append(best.section)
            if len(out) >= limit:
                break
        return out

    def implementing_regulation(self, question: str, fdca: str, cands: list[dict]) -> str | None:
        """AU2 (28 Oct): the reverse of AU1 - an FD&C Act section promoted alone brings the best regulation candidate from
        a part issued under it (its Authority line), when the cross-encoder rates it at AU2_MIN_SCORE or more. Retrieval
        check 28 Oct, client keys: L-010 got Act 403(w) without 101.4 (Part 101 is issued under 403), P-002 Act 513(i)
        without 807.100 (Part 807 under 513)."""
        from regulus.retrieval.part_authority import _parts

        parts = {p for p, acts in _parts().items() if fdca in acts}
        pool = [c for c in cands if not c.get("statute") and c["part"] in parts]
        if not pool:
            return None
        kept = self.__dict__.get("_order_scores")
        scores = (kept[2] if kept and kept[0] == question and set(kept[1]) == {c["section"] for c in cands}
                  else self.rerank_scores(question, pool))
        best = max(pool, key=lambda c: scores.get(c["section"], (0.0, ""))[0])
        return best["section"] if scores.get(best["section"], (0.0, ""))[0] >= AU2_MIN_SCORE else None

    def cross_part_citations(self, pr: "Promotion", limit: int = 1) -> list[str]:
        """XP1 (28 Oct): the provisions in OTHER parts that the promoted provision itself cites (cfr_cross_references) -
        a promoted paragraph ("820.1(d)") by its own rows only - at most `limit`, whole when the cited section fits
        MS1_MAX_CHARS, else the cited paragraph. Retrieval check 28 Oct: M-007's key 10.30 (citizen petitions) is cited by
        820.1(d) ("... under 10.30"), which was promoted, and was never supplied."""
        first = pr.sections[0]
        m = re.match(r"^(\d+\.\d+[a-z]?)((?:\([^)]+\))*)$", first)
        if not m:
            return []
        sec, path = m.group(1), m.group(2)
        part, have = sec.split(".")[0], {c.split("(")[0] for c in pr.citations}
        try:
            rows = self.db.query(_XREF_SQL_PATH, {"s": sec, "p": path or "", "pp": (path or "") + "%"})
        except Exception:  # noqa: BLE001 - an aid only
            return []
        out = []
        for r in rows:
            c = re.match(r"^21 CFR (\d+)\.(\d+[a-z]?)((?:\([a-z0-9]+\))*)$", r["target_citation"] or "")
            if not c or c.group(1) == part or f"{c.group(1)}.{c.group(2)}" in have:
                continue
            target = f"{c.group(1)}.{c.group(2)}"
            whole = len(self._section_text(target))
            label = target if 0 < whole <= MS1_MAX_CHARS else (target + c.group(3) if c.group(3) else None)
            if label and label not in out:
                out.append(label)
            if len(out) >= limit:
                break
        return out

    def navigate_subpart(self, question: str, pr: "Promotion") -> list[str]:
        """NAV1 (28 Oct): hierarchical navigation - the promoted section's subpart, as a table of contents of its section
        headings, is shown to the model, which names the other sections the question also needs (at most NAV1_MAX, within
        NAV1_MAX_CHARS). The client's keys span a subpart where the question's words match one section: retrieval check
        28 Oct, 15 of the 17 key sections missing in the first 33 dev questions were in the promoted section's subpart,
        and the cross-encoder scored them low (111.73 at 0.03 for "a finished batch fails an identity specification")."""
        if self.generator is None or getattr(self.generator, "retrieval_only", False):
            return []
        sec = pr.sections[0].split("(")[0]
        try:
            rows = self.db.query(
                "SELECT section_number s, min(section_heading) h, min(ordinal) o, sum(length(body_text)) c, "
                "left(string_agg(body_text, ' ' ORDER BY ordinal), %(n)s) t FROM cfr_records "
                "WHERE removed_on IS NULL AND NOT is_reserved AND subpart_id = (SELECT subpart_id FROM cfr_records WHERE "
                "section_number = %(s)s AND removed_on IS NULL AND subpart_id IS NOT NULL LIMIT 1) GROUP BY 1 ORDER BY 3",
                {"s": sec, "n": NAV1_SNIPPET})
        except Exception:  # noqa: BLE001 - an aid only
            return []
        have = {c.split("(")[0] for c in pr.citations}
        toc = [r for r in rows if r["s"] not in have][:NAV1_TOC_LIMIT]
        more = self.part_neighbours(question, sec, have | {r["s"] for r in rows}) if precision_fix("NAV2") else []
        if not toc and not more:
            return []
        chosen = next((r for r in rows if r["s"] == sec), None)
        entries = toc + list(more)
        snippet = lambda r: " ".join((r.get("t") or "").split())[:NAV1_SNIPPET]
        lines = [f"QUESTION: {question}", "", f"ALREADY CHOSEN: 21 CFR {sec} {(chosen or {}).get('h') or ''}".rstrip(), "",
                 "OTHER SECTIONS:"] + [f"[{i}] 21 CFR {r['s']} {r['h'] or ''} - {snippet(r)}".rstrip(" -")
                                       for i, r in enumerate(entries, 1)]
        toc = entries
        result = self.generator.generate([{"role": "system", "content": NAV1_SYSTEM}, {"role": "user", "content": "\n".join(lines)}],
                                         max_new_tokens=NAV1_MAX_TOKENS)
        reply = (getattr(result, "response", None) or "")
        # the subpart's sections (shown or not: the model may name one beyond the list) and the listed part sections
        size = {r["s"]: int(r["c"] or 0) for r in list(rows) + list(toc) if r["s"] not in have}
        named = [toc[int(n) - 1]["s"] for n in re.findall(r"(?<![\d.])(\d{1,2})(?![\d.])", reply) if 0 < int(n) <= len(toc)]
        named += re.findall(r"\b(\d{1,4}\.\d{1,4}[a-z]?)\b", reply)
        out, chars = [], 0
        for s in dict.fromkeys(named):
            if s in size and len(out) < NAV1_MAX and chars + size[s] <= NAV1_MAX_CHARS:
                out.append(s)
                chars += size[s]
        return out

    def part_neighbours(self, question: str, section: str, exclude: set[str], limit: int = NAV2_LIMIT) -> list[dict]:
        """NAV2 (28 Oct): the sections of the promoted section's part outside its subpart that are closest to the question
        by paragraph embedding - added to NAV1's table of contents, so a key in another subpart can be named (S-005:
        111.610, the records subpart, for a personnel-records question). [{s, h, c}]; [] without the index."""
        from regulus.retrieval.paragraph_dense import paragraph_dense

        pd = paragraph_dense(self.db)
        if pd is None:
            return []
        part = section.split(".")[0]
        near = [s for s, _, _ in pd.rank_sections(question, corpus="cfr") if s.split(".")[0] == part and s not in exclude][:limit]
        if not near:
            return []
        rows = self.db.query("SELECT section_number s, min(section_heading) h, sum(length(body_text)) c FROM cfr_records "
                             "WHERE removed_on IS NULL AND NOT is_reserved AND section_number = ANY(%(s)s) GROUP BY 1", {"s": near})
        by = {r["s"]: r for r in rows}
        return [by[s] for s in near if s in by]

    def cascade_pick(self, question: str, cands: list[dict]) -> str | None:
        """PC1 (28 Oct): a model cascade - when the cross-encoder is confident (its best regulation candidate at
        PC1_MIN_SCORE or more and PC1_MARGIN ahead of the next), its two best regulation candidates are the picks and the
        language-model picker is not called (14 s a question on the 27 Oct 300Q run); otherwise None and the picker
        decides as before. The reply has the picker's form ("21 CFR 111.80, 21 CFR 111.83"), so SP2/SP2F/SP2O apply."""
        kept = self.__dict__.get("_order_scores")
        scores = kept[2] if kept and kept[0] == question and set(kept[1]) == {c["section"] for c in cands} else None
        if not scores:
            return None
        cfr = sorted((c for c in cands if not c.get("statute")), key=lambda c: -scores.get(c["section"], (0.0, ""))[0])
        if len(cfr) < 2:
            return None
        top, nxt = scores.get(cfr[0]["section"], (0.0, ""))[0], scores.get(cfr[1]["section"], (0.0, ""))[0]
        if top < PC1_MIN_SCORE or top - nxt < PC1_MARGIN:
            return None
        self.__dict__["last_cascade"] = (cfr[0]["section"], round(top, 4), round(nxt, 4))
        # marked, so the recorded picker_output shows which questions the cascade decided ("cascade 98-70: ..."); the scores
        # carry no decimal point, so the reply's parser can never read them as a CFR section
        return f"cascade {round(top * 100)}-{round(nxt * 100)}: 21 CFR {cfr[0]['section']}, 21 CFR {cfr[1]['section']}"

    def rerank_veto(self, question: str, cands: list[dict], picked: dict) -> dict | None:
        """The candidate that replaces the picker's choice, or None to keep it: the pick scores below RERANK_VETO_BELOW
        and the best-scored candidate at least RERANK_VETO_MARGIN above it."""
        kept = self.__dict__.get("_order_scores")
        scores = (kept[2] if kept and kept[0] == question and set(kept[1]) == {c["section"] for c in cands}
                  else self.rerank_scores(question, cands))
        if not scores:
            return None
        if precision_fix("RO1"):
            # the RO1 guard holds for the veto too: a question naming no color is not handed to a color listing (71.25-Q1:
            # picker 80.35, vetoed to the device-color listing 73.3128, 27 Oct final run)
            from regulus.retrieval.entities import entity_index

            ents = entity_index(self.index)
            if not ents.match(question):
                cands = [c for c in cands if not (ents.is_color_listing(c["section"]) or _ANY_LISTING.match(c["section"]))
                         or c is picked] or cands
        top = max(cands, key=lambda c: (scores.get(c["section"], (0.0, ""))[0], -cands.index(c)))
        ps, ts = scores.get(picked["section"], (0.0, ""))[0], scores.get(top["section"], (0.0, ""))[0]
        if top["section"] != picked["section"] and ps < RERANK_VETO_BELOW and ts - ps >= RERANK_VETO_MARGIN:
            return top
        return None

    def statute_order(self, question: str, ranked: dict, offered: tuple[str, ...] = STATUTE_OFFERED) -> list[str]:
        """The offered FD&C Act sections, most relevant first: the lexical ranking (StatuteIndex.rank) fused by
        reciprocal rank with each section's best paragraph by embedding (regulus.retrieval.paragraph_dense). The fixed
        order 201, 601 ... 614 put the key section first for 1 of 40 Act-keyed dev questions; this order for 23."""
        from regulus.retrieval.paragraph_dense import paragraph_dense

        lex_order = sorted(offered, key=lambda s: (-(ranked[s].score if s in ranked else 0.0), offered.index(s)))
        score = {s: 1.0 / (RRF_K + i + 1) for i, s in enumerate(lex_order)}
        pd = paragraph_dense(self.db)
        if pd is not None:
            dense = [s for s, _, _ in pd.rank_sections(question, corpus="fdca") if s in offered]
            for i, s in enumerate(dense):
                score[s] += 1.0 / (RRF_K + i + 1)
        return sorted(offered, key=lambda s: (-score[s], offered.index(s)))

    def named_statute(self, question: str, lexemes) -> list[tuple[str, str]]:
        """FD&C Act sections (with any paragraph path) the question itself cites: (section, path) pairs in order.

        Explicit references ("FD&C Act 605(e)", "§ 607", "21 U.S.C. 364a(b)") are accepted. A bare "section 610",
        "the 610 duties" or "under 607" is read as the FD&C Act only in a cosmetics question that names no other Act:
        "section 610 ... biological products" and "section 301 of the Public Health Service Act" are not (code review)."""
        st = self.statute
        if st is None:
            return []
        usc_to_fdca = {st.usc(s): s for s in st.by_sec}
        other_act = bool(_OTHER_ACT.search(question))
        found: list[tuple[str, str]] = [] if other_act else [(s, f"({p})") for p, s in _SUBSECTION_OF.findall(question) if s in st.by_sec]
        found += [(m, p) for m, p in _ACT_REF.findall(question) if m in st.by_sec]
        found += [(usc_to_fdca[m.lower()], p) for m, p in _USC_REF.findall(question) if m.lower() in usc_to_fdca]
        if not other_act:
            found += [(m, p) for m, p in _SIGN_REF.findall(question) if m in st.by_sec]
            if not found and cosmetic_question(lexemes) and not _BIOLOGIC.search(question):
                found = [(m, p) for m, p in _SECTION_WORD_REF.findall(question) + _LED_COSMETIC_SECTION.findall(question)
                         + _BARE_COSMETIC_SECTION.findall(question) if m in st.by_sec]
        out: list[tuple[str, str]] = []
        for s, p in found:
            if s not in [x for x, _ in out]:
                out.append((s, p))
        return out

    def _statute_promotion(self, sections, lexemes, method: str, t0: float, cands=(), raw=None, budget: int | None = None) -> Promotion:
        """`sections`: (section, named paragraph path or "") pairs."""
        st = self.statute
        kept, left = [], []
        budget = (budget or STATUTE_BUDGET) // max(len(sections), 1)
        asked = self.__dict__.get("_asked")
        for s, path in sections:
            prefer = self.act_paragraph_scores(asked, s) if asked and precision_fix("AP1") else None
            k, lo = st.blocks(s, lexemes, budget=budget, named_path=path or None, prefer=prefer)
            kept += [tuple(g) for g in k]
            left += [tuple(g) for g in lo]
        if any(s in SUBCHAPTER_DEFINED for s, _ in sections):
            kept += [g for g in self.definition_blocks([i for g in kept for i in g]) if g not in kept]
        return Promotion(sections=(), companions=(), method=method, statute=tuple(s for s, _ in sections),
                         statute_rows=tuple(i for g in kept for i in g), statute_blocks=tuple(kept),
                         statute_left_out=tuple(left), candidates=tuple(cands), picker_output=raw,
                         seconds=time.perf_counter() - t0)

    def practice_silence(self, question: str, lexemes, t0: float) -> Promotion | None:
        """P1 (regulus.rag.practice_silence): a cosmetics question about a manufacturing practice no general cosmetic
        provision and no cosmetics section of the Act mentions (calibration, pest control, cleaning SOPs, equipment
        certification) is answered from FD&C Act 606 and 601 (and 610 for records), pinned paragraph by paragraph;
        the search that establishes the silence runs on the database text, so a provision added later ends it."""
        from regulus.rag import practice_silence as PS
        from regulus.rag.validator_switches import enabled as precision_fix

        if not precision_fix("P1"):
            return None
        practices = PS.asked_practices(question)
        if not practices or self.statute is None or not all(PS.silent(self.db, self.statute, p) for p in practices):
            return None
        pinned = PS.statute_paths(practices, PS.asks_records(question))
        st = self.statute
        kept: list[tuple[int, ...]] = []
        for s, path in pinned:
            rows = tuple(i for i in st.by_sec.get(s, ()) if st.rows[i].get("paragraph_path") == path)
            if not rows:
                return None
            kept.append(rows)
        return Promotion(sections=(), companions=(), method="practice_silence",
                         statute=tuple(dict.fromkeys(s for s, _ in pinned)),
                         statute_rows=tuple(i for g in kept for i in g), statute_blocks=tuple(kept),
                         seconds=time.perf_counter() - t0, notes=PS.notes_for(practices))

    def definition_blocks(self, rows: list[int]) -> list[tuple[int, ...]]:
        """FD&C Act section 604 opens "In this subchapter:" - its definitions govern sections 605-614. The definitions
        of the terms the kept text uses ("serious adverse event", "responsible person", "facility"), most used first,
        within DEFINITION_BUDGET; they follow the section's own blocks, so the hard ceiling drops them first. On the
        dev questions the key elements needed them and they were never supplied (605 reports without 604(5), 22 Oct)."""
        st = self.statute
        if st is None or DEFINITIONS_SECTION not in st.by_sec:
            return []
        text = " ".join(st.rows[i]["body_text"] for i in rows).lower()
        groups: dict[str, list[int]] = {}
        for i in st.by_sec[DEFINITIONS_SECTION]:
            top = st._top(i)
            if top:
                groups.setdefault(top, []).append(i)
        scored = []
        for top, g in groups.items():
            term = (st.rows[g[0]]["paragraph_heading"] or st.rows[g[0]].get("top_caption") or "").strip().lower()
            uses = len(re.findall(rf"\b{re.escape(term)}\b", text)) if term and term not in GENERIC_DEFINED_TERMS else 0
            if uses:
                scored.append((-uses, g[0], tuple(g)))
        out, used = [], 0
        for _, _, g in sorted(scored):
            cost = sum(len(st.rows[i]["body_text"]) + 1 for i in g) + BLOCK_HEADER_CHARS
            if used + cost <= DEFINITION_BUDGET:
                out.append(g)
                used += cost
        return out

    @property
    def index(self) -> SectionIndex:
        if self._index is None:
            self._index = section_index(self.db)
        return self._index

    # -- candidates -----------------------------------------------------------
    def candidates(self, question: str, parts: tuple[str, ...] | None = None) -> tuple[list[str], list[dict]]:
        asked = question   # HQ1 matches the user's words, before SY1-SY3 add search terms (code review)
        if precision_fix("SY1") and _ASKS_AGENCY.search(question or "") and not re.search(r"\bpetition", question or "", re.I):
            # SY1: asking FDA to act is petitioning it - "Can we ask FDA to exempt it from certification?" never says
            # "petition", and 71.18 ("Petition for exemption from certification") was not among the 22 candidates
            question = f"{question} petition"
        if precision_fix("SY2"):
            # SY2: the regulations' own words for what the question says plainly (domain thesaurus, search terms only)
            extra = [terms for pattern, terms in _THESAURUS if pattern.search(question or "")]
            if precision_fix("SY3"):
                extra += [terms for pattern, terms in _THESAURUS_SY3 if pattern.search(question or "")]
            if extra:
                question = f"{question} {' '.join(extra)}"
        if precision_fix("SY4"):
            # SY4: the acronyms the regulations define, spelled out for the search ("RACC" -> "reference amount commonly
            # consumed": L-008 never had 101.12 as a candidate) - regulus.retrieval.acronyms
            from regulus.retrieval.acronyms import expansions

            spelled = expansions(question)
            if spelled:
                question = f"{question} {' '.join(spelled)}"
        lex = SectionIndex.question_lexemes(self.db, question)
        if parts:
            ranked = [r for r in self.index.rank(lex, limit=400) if r.part in parts]
            sub = _SUBPART_REF.search(question or "")
            subpart = None
            if sub:   # "subpart C of Part 82": that subpart's sections only (retrieval diagnosis: parser)
                in_sub = [r for r in ranked if (r.subpart or "").lower().startswith(f"subpart {sub.group(1).lower()} ")]
                subpart = f"subpart {sub.group(1).lower()} " if in_sub else None
                ranked = in_sub or ranked
            ranked = ranked[:self.limit]
            if PART_DENSE:
                # the dense ranking inside the named part(s) too: a section sharing no lexeme with the question was never
                # offered ("FDA Registered Facility, Reg. No." -> 710.8: 'registered' and 'registration' stem apart).
                # Appended after the lexical order, not fused into it: fusing reordered the top of the list and lost
                # two rank-1 keys of 17 (23 Oct section-resolution benchmark)
                extra = self._dense_only(question, ranked, parts, subpart)
                ranked = ranked[:max(self.limit - len(extra), 0)] + extra
        elif cosmetic_question(lex):
            # a cosmetics question: candidates from the color-additive and cosmetic parts only. Food, feed and device
            # sections were picked for cosmetics questions (117.251, 111.455, 226.30; 22 Oct); on the 277-question bank
            # the restriction loses no key section and adds 3 to the candidate lists
            ranked = [r for r in self.index.rank(lex, limit=400) if r.part in COSMETIC_PARTS][:LEXICAL_CANDIDATES]
            ranked = self._fuse_dense(question, ranked)
        elif (programme := routed_programme(asked)) is not None:
            # DR1: the same for the one programme the question names (regulus.retrieval.programmes), as deep as a general
            # question's list - the section dense index holds the cosmetics parts only, so there is no dense leg to fuse
            ranked = [r for r in self.index.rank(lex, limit=400) if r.part in programme.parts][:self.limit]
        else:
            ranked = self.index.rank(lex, limit=self.limit)
        if precision_fix("HQ1"):
            # HQ1 (step D): sections whose index-time questions match the question, appended after the list like the dense
            # leg (never fused into its order), inside the named part(s) or the cosmetics parts when the question is one
            scope = tuple(parts) if parts else (tuple(COSMETIC_PARTS) if cosmetic_question(lex) else None)
            hq = self._question_matches(asked, ranked, scope)
            ranked = with_reserved(ranked, hq, CFR_CANDIDATES_WITH_STATUTE)
        if ENTITY_CONSTRAINTS:
            try:
                ranked = self._entity_constrained(question, lex, ranked, parts)
            except Exception:  # noqa: BLE001 - an aid: without it the ranking stands
                pass
        if not ranked:
            return lex, []
        rows = self.db.query(_SNIPPET_SQL, {"s": [r.section for r in ranked], "o": sorted({r.best_ordinal for r in ranked})})
        text = {(r["section_number"], int(r["ordinal"])): (r["paragraph_path"] or "", r["body_text"] or "") for r in rows}
        # a sub-paragraph is shown with its lead-in: "(2) the package is opened" says nothing without "(d) A certificate
        # shall expire ... if:" (retrieval diagnosis: picker context)
        parents = {}
        for r in ranked:
            path = text.get((r.section, r.best_ordinal), ("", ""))[0]
            m = re.match(r"^(\([^)]+\))\(", path or "")
            if m:
                parents[(r.section, m.group(1))] = ""
        if parents:
            for row in self.db.query(_LEADIN_SQL, {"s": sorted({s for s, _ in parents}), "p": sorted({p for _, p in parents})}):
                key = (row["section_number"], row["paragraph_path"])
                if key in parents:
                    parents[key] = " ".join((row["body_text"] or "").split())

        def snippet(r):
            path, body = text.get((r.section, r.best_ordinal), ("", ""))
            body = " ".join(body.split())
            m = re.match(r"^(\([^)]+\))\(", path or "")
            lead = parents.get((r.section, m.group(1)), "") if m else ""
            if lead:
                return (lead[:LEADIN_CHARS].rstrip() + " ... " + body)[:SNIPPET_CHARS + LEADIN_CHARS]
            return body[:SNIPPET_CHARS]

        titles = self._titles()
        return lex, [{"section": r.section, "part": r.part, "heading": r.heading, "subpart": r.subpart,
                      "part_title": titles.get(r.part, ""), "score": round(r.score, 4), "snippet": snippet(r)}
                     for r in ranked]

    def _dense_only(self, question: str, ranked: list, parts: tuple[str, ...], subpart: str | None, limit: int = 4) -> list:
        """Up to `limit` sections of the named part(s) (and subpart) that the dense ranking puts in its top
        DENSE_CANDIDATES there and the lexical list lacks, in dense order."""
        from regulus.retrieval.section_dense import section_dense
        from regulus.retrieval.section_index import RankedSection

        dense_ix = section_dense(self.db)
        if dense_ix is None:
            return []
        ix, where, first = self.index, self._where(), self.__dict__["_first_ordinal"]
        inside = [s for s in dense_ix.rank(question, len(dense_ix.names)) if s in where and ix.parts[where[s]] in parts
                  and (subpart is None or (ix.subparts[where[s]] or "").lower().startswith(subpart))][:DENSE_CANDIDATES]
        have = {r.section for r in ranked}
        return [RankedSection(s, ix.parts[where[s]], ix.headings[where[s]], ix.subparts[where[s]], 0.0, first.get(s, 0))
                for s in inside if s not in have][:limit]

    def _question_matches(self, question: str, ranked: list, parts) -> list:
        """HQ1: up to HQ1_LIMIT sections the hypothetical-question index ranks at least HQ1_MIN_SCORE that `ranked` lacks."""
        from regulus.retrieval.section_index import RankedSection
        from regulus.retrieval.section_questions import section_questions

        hq = section_questions()
        if hq is None:
            return []
        ix, where, first = self.index, self._where(), self.__dict__.get("_first_ordinal") or {}
        have = {r.section for r in ranked}
        out = []
        for s, _ in hq.rank(question, HQ1_LIMIT + len(have), parts=parts, min_score=HQ1_MIN_SCORE):
            if s in have or s not in where:
                continue
            out.append(RankedSection(s, ix.parts[where[s]], ix.headings[where[s]], ix.subparts[where[s]], 0.0, first.get(s, 0)))
            if len(out) >= HQ1_LIMIT:
                break
        return out

    def _fuse_dense(self, question: str, ranked: list) -> list:
        """Lexical top LEXICAL_CANDIDATES fused with the dense top DENSE_CANDIDATES by reciprocal rank (k = RRF_K); the
        dense ranking finds paraphrased sections the words miss (unseen bank questions: key among the candidates
        129 -> 138 of 143). Without the embedding model the lexical list is returned unchanged."""
        from regulus.retrieval.section_dense import section_dense
        from regulus.retrieval.section_index import RankedSection

        dense_ix = section_dense(self.db)
        if dense_ix is None:
            return ranked
        dense = dense_ix.rank(question, DENSE_CANDIDATES)
        score: dict[str, float] = {}
        for i, r in enumerate(ranked):
            score[r.section] = score.get(r.section, 0.0) + 1.0 / (RRF_K + i + 1)
        for i, s in enumerate(dense):
            score[s] = score.get(s, 0.0) + 1.0 / (RRF_K + i + 1)
        by_section = {r.section: r for r in ranked}
        ix = self.index
        out = []
        for s in sorted(score, key=lambda x: (-score[x], x))[:self.limit]:
            if s in by_section:
                out.append(by_section[s])
            elif s in ix.sections:
                i = ix.sections.index(s)
                out.append(RankedSection(s, ix.parts[i], ix.headings[i], ix.subparts[i], 0.0, 0))
        return out

    # -- named color additives (regulus.retrieval.entities) ---------------------------------------
    def _where(self) -> dict[str, int]:
        cached = self.__dict__.get("_where_ix")
        if cached is None:
            ix = self.index
            cached = self.__dict__["_where_ix"] = {s: i for i, s in enumerate(ix.sections)}
            first: dict[str, int] = {}
            for si, o in zip(ix.row_section.tolist(), ix.row_ordinal.tolist()):
                s = ix.sections[si]
                first[s] = min(first.get(s, o), o)
            self.__dict__["_first_ordinal"] = first
        return cached

    def preferred_listing(self, sections, lex, parts: tuple[str, ...] | None = None, exact=(), order=()) -> list[str]:
        """Of one color's listing sections, the ones a question about it is answered from, best first, when no reranker
        reads them: the listing for the product kind the question names, then the Part 82 certification listing, then a
        listing without a product subpart. A question that names a food, drug or device product and no cosmetic one
        is about that kind ("carmine in our strawberry ice cream": 73.100, not the cosmetics listing 73.2087, 23 Oct
        synthetic set); otherwise the cosmetics subpart, the deployment's own (the same rule as siblings()). Within one
        kind: the exactly named color first ("paprika oleoresin" 73.345 before "paprika" 73.340), then the ranking's
        order (`order`, the candidate list), never the section number."""
        ix, where = self.index, self._where()
        lx = set(lex)
        other = {k for k, terms in _PRODUCT_KIND_TERMS.items() if lx & terms}
        wanted = set(other) if other and not (lx & _COSMETIC_PRODUCT_TERMS) else other | {"cosmetic"}
        ranked = []
        for s in sections:
            i = where.get(s)
            if i is None or (parts and ix.parts[i] not in parts):
                continue
            kind = _subpart_kind(ix.subparts[i])
            ranked.append((0 if kind in wanted else 1 if ix.parts[i] == "82" else 2 if kind is None else 3,
                           0 if (not exact or s in exact) else 1, order.index(s) if s in order else len(order), s))
        return [r[-1] for r in sorted(ranked)]

    def _listing_cands(self, sections) -> list[dict]:
        ix, where, titles = self.index, self._where(), self._titles()
        return [{"section": s, "part": ix.parts[where[s]], "heading": ix.headings[where[s]], "subpart": ix.subparts[where[s]],
                 "part_title": titles.get(ix.parts[where[s]], "")} for s in sections if s in where]

    def choose_listing(self, question: str, sections, lex, parts: tuple[str, ...] | None, exact=(), order=()) -> str | None:
        """The one listing of a named color a question is about. With the reranker it reads them - each with its subpart
        ("Subpart A - Foods", "Subpart C - Cosmetics") and its best paragraphs - and the best-scored one is chosen:
        whether 'salmon feed', 'fruit punch' or 'lip balm' is a food or a cosmetic is read from the text, not from a
        word list. Without the reranker, preferred_listing's product-kind rule."""
        where = self._where()
        pool = [s for s in sections or () if s in where and (not parts or self.index.parts[where[s]] in parts)]
        # a Part 82 listing is a one-line pointer ("shall conform to ... § 74.2331"): the companion of the listing that
        # says where the color may be used, never chosen in its place (5 dev keys lost to it, 23 Oct) - unless the
        # question asks about Part 82 only
        substantive = [s for s in pool if self.index.parts[where[s]] != "82"]
        pool = substantive or pool
        if not pool:
            return None
        lx = set(lex)
        kinds = {k for k, terms in _PRODUCT_KIND_TERMS.items() if lx & terms} | ({"cosmetic"} if lx & _COSMETIC_PRODUCT_TERMS else set())
        # a question that names one kind of product ("mascara", "ice cream") is decided by that word: the reranker reads
        # the Foods and Cosmetics listings of an exempt color as near-identical text and chose the food listing for 5
        # cosmetics questions (23 Oct). It reads the listings when the question names no kind of product, or several.
        if RERANKER and len(pool) > 1 and len(kinds) != 1:
            scores = self.rerank_scores(question, self._listing_cands(pool))
            if scores:
                return max(pool, key=lambda s: (scores.get(s, (0.0, ""))[0], -pool.index(s)))
        return (self.preferred_listing(pool, lex, parts, exact, order) or [None])[0]

    def _entity_constrained(self, question: str, lex, ranked: list, parts: tuple[str, ...] | None) -> list:
        """A question naming a color additive: a question naming a certified color (FD&C Blue No. 1) cannot mean another
        certified color's listing, so those are taken out of the candidates (FD&C Red No. 40 was picked for an FD&C Blue
        No. 1 question); an exempt color's listing is never taken out, because exempt names are loose ("pearlescent mica
        pigment", "krill meal" - vetoing them lost 12 synthetic keys, 23 Oct). Each named color's listing is offered
        when the ranking missed it (third place: offered, not forced)."""
        from regulus.retrieval.entities import entity_index, names_a_color
        from regulus.retrieval.section_index import RankedSection

        E = entity_index(self.index)
        named = [m for m in E.match(question) if names_a_color(question, m)]
        if not named:
            return ranked
        allowed = {s for m in named for s in m.sections}
        certified = any(any(k[0] == "cert" for k in m.keys) for m in named)
        kept = [r for r in ranked if not (certified and E.is_certified_listing(r.section) and r.section not in allowed)]
        have = {r.section for r in kept}
        ix, where, first = self.index, self._where(), self.__dict__["_first_ordinal"]
        for m in named:
            if have & set(m.sections):
                continue
            best = self.preferred_listing(m.sections, lex, parts, m.exact, [r.section for r in kept])
            if best:
                i = where[best[0]]
                kept.insert(min(2, len(kept)), RankedSection(best[0], ix.parts[i], ix.headings[i], ix.subparts[i], 0.0,
                                                             first.get(best[0], 0)))
                have.add(best[0])
        return kept

    def entity_pick(self, question: str, lex, cands: list[dict], parts: tuple[str, ...] | None) -> list[str] | None:
        """The sections promoted without the picker, or None: when the best-ranked regulation left after the color
        constraint is a listing of a color the question names, the listing of that exact name (not a shorter name inside
        it: "paprika oleoresin" is 73.345, not 73.340) for the question's product (choose_listing) is promoted, with the
        Part 82 certification listing of a cosmetics listing (as siblings() does); the same for a second named color.
        The picker still decides every other question."""
        from regulus.retrieval.entities import entity_index, names_a_color, normalize_heading

        E = entity_index(self.index)
        named = [m for m in E.match(question) if names_a_color(question, m)]
        top = next((c for c in cands if not c.get("statute")), None)
        if not named or top is None or not E.is_color_listing(top["section"]):
            return None
        first = next((m for m in named if top["section"] in m.sections), None)
        if first is None:
            return None
        ix, where = self.index, self._where()
        out: list[str] = []
        order = [c["section"] for c in cands if not c.get("statute")]
        for m in [first] + [m for m in named if m is not first][:1]:
            best = self.choose_listing(question, m.sections, lex, parts, m.exact, order)
            if best is None:
                continue
            chosen = [best]
            if ix.parts[where[best]] != "82":
                # the Part 82 certification listing of the same color, when the ranking offered it (as siblings() keeps it)
                head = normalize_heading(ix.headings[where[best]])
                chosen += [s for s in order if s in m.sections and ix.parts[where[s]] == "82"
                           and normalize_heading(ix.headings[where[s]]) == head][:1]
            out += [s for s in chosen if s not in out]
        return out or None

    def _titles(self) -> dict[str, str]:
        if self._part_titles is None:
            self._part_titles = {r["number"]: r["title"] or "" for r in self.db.query("SELECT number, title FROM cfr_parts")}
        return self._part_titles

    # -- companions -------------------------------------------------------------
    def _definitions(self) -> dict[str, list[tuple[str, list[str]]]]:
        if self._defs is None:
            self._defs = {}
            for sec in ("70.3", "700.3"):
                out = []
                for r in self.db.query(_SECTION_SQL, {"s": sec}):
                    m = _DEF_RE.match(r["body_text"] or "")
                    if not (m and r["paragraph_path"]):
                        continue
                    terms = [t.strip(" ,") for t in re.split(r",\s*|\s+and\s+", m.group(1))]
                    terms = [t for t in terms if len(t.split()) >= 2]
                    if terms:
                        out.append((r["paragraph_path"], terms))
                self._defs[sec] = out
        return self._defs

    def _section_text(self, section: str) -> str:
        """A section's live text, fetched once per section and promoter (it is read up to three times per question)."""
        cache = self.__dict__.setdefault("_text_cache", {})
        if section not in cache:
            if len(cache) > 256:
                cache.clear()
            cache[section] = " ".join(r["body_text"] or "" for r in self.db.query(_SECTION_SQL, {"s": section}))
        return cache[section]

    def definition_companions(self, question: str, limit: int = 4) -> list[str]:
        """For a question that asks how FDA defines something ("in FDA's sense", "How do FDA's definitions treat 'area
        of the eye' ..."): the paragraphs of the programme's definitions sections (70.3, 700.3, 82.3) whose defined term
        the question names - kept whatever section is promoted (retrieval diagnosis: promotion switched the
        defined-terms leg off). Single-word terms count only when the question quotes them."""
        from regulus.retrieval.query import _INVOKES_DEFINITION_RE, _QUOTED_TERM_RE

        if not _INVOKES_DEFINITION_RE.search(question or ""):
            return []
        q = " ".join((question or "").lower().split())
        quoted = {m.strip().lower() for m in _QUOTED_TERM_RE.findall(question or "")}
        out: list[str] = []
        for sec in ("70.3", "700.3", "82.3"):
            for path, terms in self._all_definitions(sec):
                for term in terms:
                    tl = term.lower().strip()
                    single = len(tl.split()) == 1
                    hit = (tl in quoted or any(tl in qt or qt in tl for qt in quoted)) if single else \
                          re.search(rf"(?<![a-z]){re.escape(tl)}s?(?![a-z])", q)
                    if hit and sec + path not in out:
                        out.append(sec + path)
                        break
        return out[:limit]

    def asked_definitions(self, section: str, question: str, limit: int = 2) -> list[str]:
        """DQ1 (28 Oct): the definition paragraph of a term the question asks about ("who is the 'importer'", "what is a
        qualified facility") from the promoted section's own subpart, else its part (cfr_definitions) - one paragraph,
        never a whole definitions section. 300Q 27 Oct: V-001 promoted 1.502 and never had 1.500's "Importer means";
        F-002's "qualified facility" is 117.3. "Importer" is defined in five parts, so the subpart decides."""
        terms = {t.strip().lower() for m in _ASKED_TERM.finditer(question or "") for t in m.groups() if t}
        if not terms:
            return []
        sec = section.split("(")[0]
        sql = ("SELECT lower(d.term) t, r.section_number s, r.paragraph_path p FROM cfr_definitions d JOIN cfr_records r ON "
               "r.id = d.record_id WHERE r.removed_on IS NULL AND ")
        try:
            rows = self.db.query(sql + "r.subpart_id = (SELECT subpart_id FROM cfr_records WHERE section_number = %(s)s AND "
                                 "removed_on IS NULL AND subpart_id IS NOT NULL LIMIT 1)", {"s": sec}) or \
                   self.db.query(sql + "d.part_number = %(p)s", {"p": sec.split(".")[0]})
        except Exception:  # noqa: BLE001 - an aid only
            return []
        out = [r["s"] + r["p"] for r in rows if r["p"] and r["s"] != sec
               and any(r["t"] == q or r["t"] == q.rstrip("s") for q in terms)]
        return list(dict.fromkeys(out))[:limit]

    def _defined_terms(self, sec: str) -> list[tuple[str, list[str], str]]:
        """(paragraph, the term(s) it defines in their usual forms, text) for every paragraph of a definitions section;
        no forms for one that defines nothing ("For a material otherwise meeting the definition of ..."). Terms: "The
        term X means", "A color additive is any", "The hair dye exemption in section 601(a) of the act applies to"."""
        cache = self.__dict__.setdefault("_defined_terms_cache", {})
        if sec not in cache:
            out = []
            for r in self.db.query(_SECTION_SQL, {"s": sec}):
                body, path = r["body_text"] or "", r["paragraph_path"]
                if not path:
                    continue
                m = _DEFINES_RE.match(body)
                forms: set[str] = set()
                if m:
                    term = re.sub(r"\s+(?:in\s+section\s+\S+|of\s+the\s+act)\b.*$", "", m.group("t").strip(" ,").lower())
                    forms = {term, re.sub(r"\s+(?:of\s+(?:a\s+)?cosmetic\s+products?|used\s+by\s+the\s+establishment)$", "", term)}
                    if " and " in term:   # "externally applied drugs and externally applied cosmetics"
                        halves = [h.strip() for h in term.split(" and ")]
                        forms |= set(halves) | {os.path.commonprefix(halves).strip()}
                out.append((path, sorted(f for f in forms if len(f) >= 3), body))
            cache[sec] = out
        return cache[sec]

    def definitions_asked(self, question: str, limit: int = 6) -> list[str]:
        """DF1: the definition paragraphs ("70.3(f)", "70.3(g)") of a question that asks how FDA's definitions
        classify what it describes - "Are these 'color additives' in FDA's sense?", "How do FDA's definitions classify
        the lake, the blend and the oil?" - or [] (27 Oct 2026: such questions were resolved to the listings of the
        colors they named, 73.2496 for mica, or to 82.1051, and none was answered from 70.3). A term of more than one
        word counts when the question names it; a one-word term ("lake", "package") only when quoted or among what
        the question asks to classify. With each matched definition come a definition that contains its term ("system
        of commercial distribution of a cosmetic product"), a paragraph that limits it ("For a material otherwise
        meeting the definition of color additive to be exempt ..."), and for a one-word term the short definitions of
        the terms it uses ("straight color" in the definition of lake). A question about the Act's definitions (604
        "facility", 608 "safe", MoCRA) is left to the Act. Switched by "DF1"."""
        from regulus.rag.validator_switches import enabled as precision_fix
        from regulus.retrieval.query import _INVOKES_DEFINITION_RE, _QUOTED_TERM_RE

        q = question or ""
        if not precision_fix("DF1") or not _INVOKES_DEFINITION_RE.search(q) or _ACT_DEFINITION_RE.search(q):
            return []
        low = " ".join(q.lower().split())
        quoted = [x.strip().lower() for x in _QUOTED_TERM_RE.findall(q)]
        m = _CLASSIFY_RE.search(low)
        asked = " ".join(_PLAIN_TERMS.get(w, w) for w in re.findall(r"[a-z&-]+", low[m.end():] if m else ""))

        def named(form: str) -> bool:
            # "FDA's color additive definitions" names no term: it says which definitions
            pat = rf"(?<![a-z]){re.escape(form)}s?(?![a-z])(?!\s+definitions?\b)"
            if len(form.split()) > 1:
                return bool(re.search(pat, low))
            return any(re.search(pat, x) for x in quoted) or bool(re.search(pat, asked))

        out: list[str] = []
        for sec in DEFINITION_SECTIONS:
            paras = self._defined_terms(sec)
            for path, forms, body in paras:
                hit = [f for f in forms if named(f)]
                if not hit:
                    continue
                with_it = [path] + [p2 for p2, f2, b2 in paras if p2 != path and (
                    any(f in x for f in hit for x in f2 if x != f and len(f.split()) > 1)
                    or any(re.search(rf"\bdefinition\s+of\s+(?:an?\s+|the\s+)?{re.escape(f)}\b", b2, re.I) for f in hit))]
                if all(len(f.split()) == 1 for f in hit):
                    # a term most of the section's definitions use ("cosmetic product" in 700.3) explains none of them
                    def uses(x, b):
                        return bool(re.search(rf"\b{re.escape(x)}s?\b", b, re.I))
                    with_it += [p2 for p2, f2, b2 in paras if p2 != path and len(b2) <= DEFINITION_HOP_CHARS and any(
                        len(x.split()) > 1 and uses(x, body) and sum(uses(x, b3) for p3, _, b3 in paras if p3 != p2)
                        < DEFINITION_GENERIC_USES for x in f2)]
                out += [sec + p for p in with_it if sec + p not in out]
        return out[:limit]

    def act_definition_asked(self, question: str) -> tuple[str, str] | None:
        """DF2 (27 Oct): the FD&C Act paragraph defining what a question asks something to be classified as - "Is it a
        cosmetic under the FD&C Act definition?" -> 201(i); "Is our salon a 'facility'?" -> 604(3). The terms are read
        from the Act text ('The term "cosmetic" means'), in sections 201 and 604; the longest term the question uses
        in a classification ("is it a ...", "counts as", "qualify as", "under the ... definition", a quoted term) wins.
        Switched by "DF2"."""
        if not precision_fix("DF2") or self.statute is None:
            return None
        cache = self.__dict__.setdefault("_act_terms", [])
        if not cache:
            st = self.statute
            for s in ACT_DEFINITION_SECTIONS:
                for i in st.by_sec.get(s, ()):
                    row = st.rows[i]
                    for term in re.findall(r"[Tt]he terms?\s+[\"“]([^\"”]{2,60})[\"”]", row.get("body_text") or "")[:1]:
                        cache.append((s, row.get("paragraph_path") or "", term.lower()))
        q = " ".join((question or "").lower().split())
        if not _NAMES_ACT.search(q) or re.search(r"\bpart\s+\d+|\b\d{2,4}\.\d+|\b(?:60\d|61[0-4]|201)\b", q):
            return None   # the regulations' definitions (DF1), or a question naming a regulation, are not the Act's
        mocra = "mocra" in q or "modernization" in q
        quoted = [x.strip().lower() for x in re.findall(r"[\"'‘“]([^\"'’”]{2,60})[\"'’”]", question or "")]
        best = None
        for s, path, term in cache:
            if mocra and s == "201":
                continue   # MoCRA's own definitions (604, 608) for a MoCRA question
            t = re.escape(term)
            classify = (re.search(rf"\b(?:is|are|it|they|as|an?|qualif\w+\s+as|count\w*\s+as)\s+(?:an?\s+|the\s+)?{t}s?\b[^?]{{0,60}}"
                                  rf"(?:\?|under|in\s+the|definition|sense)", q)
                        or any(re.fullmatch(rf"{t}s?", x) for x in quoted))
            if classify and (best is None or len(term) > len(best[2])):
                best = (s, path, term)
        return (best[0], best[1]) if best else None

    def _all_definitions(self, sec: str) -> list[tuple[str, list[str]]]:
        cache = self.__dict__.setdefault("_defs_all", {})
        if sec not in cache:
            out = []
            for r in self.db.query(_SECTION_SQL, {"s": sec}):
                m = _DEF_RE.match(r["body_text"] or "") or re.match(r"^\([a-z]{1,2}\)\s+(?:A|An|The)\s+(.+?)\s+is\b", r["body_text"] or "")
                if m and r["paragraph_path"]:
                    out.append((r["paragraph_path"], [x.strip(" ,") for x in re.split(r",\s*|\s+and\s+", m.group(1)) if x.strip(" ,")]))
            cache[sec] = out
        return cache[sec]

    def cross_reference_companions(self, section: str, limit: int = 2) -> list[str]:
        """Provisions the promoted section itself cites, inside the cosmetics programme (cfr_cross_references): a
        paragraph citation as it is ("710.4" -> 700.3(j)); a whole-section citation by the paragraphs of that section
        that cite back the promoted one ("70.55" -> 70.19(n), the fee for the studies 70.55 provides), or whole when it
        is short. Not for color listings, whose citations are identity and labeling boilerplate."""
        out: list[str] = []
        for r in self.db.query(_XREF_SQL, {"s": section}):
            m = re.match(r"^21 CFR (\d+)\.(\d+[a-z]?)((?:\([a-z0-9]+\))*)$", r["target_citation"] or "")
            if not m or m.group(1) not in COSMETIC_PARTS or f"{m.group(1)}.{m.group(2)}" == section:
                continue
            target, path = f"{m.group(1)}.{m.group(2)}", m.group(3)
            if path:
                cand = [target + path]
            else:
                rows = self.db.query(_SECTION_SQL, {"s": target})
                back = [r2["paragraph_path"] for r2 in rows if r2["paragraph_path"]
                        and re.search(rf"§+\s*{re.escape(section)}(?![\d])", r2["body_text"] or "")]
                cand = [target + p for p in back[:1]] or ([target] if len(rows) <= 3 else [])
            out += [c for c in cand if c not in out]
            if len(out) >= limit:
                break
        return out[:limit]

    def companions(self, section: str, question: str, lexemes) -> list[str]:
        ix = self.index
        try:
            i = ix.sections.index(section)
        except ValueError:
            return []
        part, subpart = ix.parts[i], ix.subparts[i]
        out: list[str] = []
        p = int(re.match(r"\d+", part).group(0)) if re.match(r"\d+", part) else 0
        family = "70.3" if 70 <= p <= 82 else ("700.3" if 700 <= p <= 799 else None)
        if family and family != section:
            sents = _SENT_RE.split(self._section_text(section))
            use_text = " ".join(s for s in sents if _USE_RE.search(s)).lower()
            q = (question or "").lower()
            for path, terms in self._definitions().get(family, ()):
                if any(t.lower() in use_text or t.lower() in q for t in terms):
                    out.append(family + path)
        if part in COLOR_LISTING_PARTS and set(lexemes) & EYE_TERMS and "70.3(s)" not in out:
            out.append("70.3(s)")   # the question names an eye product: what "area of the eye" covers
        if part in COLOR_LISTING_PARTS and set(lexemes) & EYE_TERMS and section != "70.5":
            # 70.5(a) (no eye-area use unless the listing provides for it) only when the listing does not itself speak to
            # the area of the eye: added to a listing that permits eye use, it led the model to deny that use (22 Oct)
            if not _EYE_USE.search(self._section_text(section)):
                out.append("70.5(a)")
        if part not in COLOR_LISTING_PARTS:
            out += [c for c in self.cross_reference_companions(section) if c not in out]
        if part == "82":
            for j, (pj, sj, hj) in enumerate(zip(ix.parts, ix.subparts, ix.headings)):
                if pj == part and sj == subpart and (hj or "").lower().startswith("general.") and ix.sections[j] != section:
                    out.append(ix.sections[j])
            # C82 (regulus.rag.validator_switches): a certification question about an FD&C (subpart B) listing also gets
            # 82.5, the general specifications 82.50 makes a condition of certification (lead, arsenic, heavy metals):
            # the 82.705 / 82.706 keys name those limits, and the composed answer can quote them only when supplied
            from regulus.rag.validator_switches import enabled as precision_fix

            if ("82.50" in out and precision_fix("C82") and re.search(r"\bcertif|\bspecification", question or "", re.I)
                    and re.search(r"shall conform in identity and specifications to the requirements of §\s*74\.",
                                  self._section_text(section))):
                out.append("82.5")   # only a conformity listing, which the C82 composer answers from these texts
        out += [c for c in self.fee_companions(section) if c not in out]
        return list(dict.fromkeys(out))

    # -- decision ---------------------------------------------------------------
    def fee_companions(self, section: str) -> list[str]:
        """FC1 (27 Oct): the paragraphs of 21 CFR 70.19 (fees for color additive services) that price what a Part 70/71
        section provides - a two-word subject of the section's heading that the fee paragraph names too, rare in Parts
        70-71 (at most FC1_MAX_SECTIONS sections use it): 70.51 "Advisory committee ..." -> 70.19(l), (m); 70.55
        "Request for scientific studies" -> 70.19(n). The keys of both name the fee; neither section cites 70.19."""
        if not precision_fix("FC1") or not re.match(r"^7[01]\.", section) or section == "70.19":
            return []
        cache = self.__dict__.setdefault("_fc1", {})
        if "df" not in cache:
            stop = set("the of and or a an to in on for by with any be is are as such that this shall will may which "
                       "under its from at not other all".split())

            def stem(w):
                return w[:-3] + "y" if w.endswith("ies") else (w[:-1] if w.endswith("s") and not w.endswith("ss") else w)

            def bigrams(t):
                ws = [stem(x) for x in re.findall(r"[a-z]+", (t or "").lower())]
                return {(a, b) for a, b in zip(ws, ws[1:]) if a not in stop and b not in stop and len(a) > 3 and len(b) > 3}

            texts: dict[str, str] = {}
            for r in self.db.query("SELECT section_number s, body_text t FROM cfr_records WHERE removed_on IS NULL "
                                   "AND part_number IN ('70', '71')"):
                texts[r["s"]] = texts.get(r["s"], "") + " " + (r["t"] or "")
            from collections import Counter

            cache["df"] = Counter(b for t in texts.values() for b in bigrams(t))
            cache["bigrams"] = bigrams
            cache["fees"] = [(r["paragraph_path"], bigrams(r["body_text"])) for r in self.db.query(_SECTION_SQL, {"s": "70.19"})
                             if r["paragraph_path"]]
        heading = next((h for s2, h in zip(self.index.sections, self.index.headings) if s2 == section), "") or ""
        subject = {b for b in cache["bigrams"](heading) if cache["df"][b] <= FC1_MAX_SECTIONS}
        return [f"70.19{path}" for path, bg in cache["fees"] if subject & bg]

    def named_companions(self, sections: list[str], question: str) -> list[str]:
        """Companions of sections the question names itself ("What does 82.1327 require?" -> 82.1050, the subpart's
        certification rule; "under § 710.4" -> 700.3(j)): the same rules as for a promoted section, no picker."""
        lex = SectionIndex.question_lexemes(self.db, question)
        out: list[str] = []
        for s in sections:
            out += [c for c in self.companions(s, question, lex) if c not in out and c not in sections]
        out += [c for c in self.definition_companions(question) if c not in out and c not in sections]
        if "70.5(a)" in out and any(_EYE_USE.search(self._section_text(s)) for s in sections):
            out.remove("70.5(a)")
        return out

    def siblings(self, picked: dict, cands: list[dict], lexemes) -> list[str]:
        """The same color listed elsewhere under the same heading (74.2254 / 82.1254 / 74.1254 "D&C Orange No. 4."): for
        a cosmetics question the Cosmetics-subpart listing and the Part 82 certification listing are both promoted, the
        Foods/Drugs/Devices listings only when the question names that kind of product."""
        if ENTITY_CONSTRAINTS:   # 74.1101 'FD&C Blue No. 1' and 74.2101 'FD&C Blue No. 1.' are one color
            from regulus.retrieval.entities import normalize_heading as norm
        else:
            norm = lambda h: (h or "").strip().lower()  # noqa: E731
        head = norm(picked["heading"])
        if picked["part"] not in COLOR_LISTING_PARTS or re.fullmatch(r"(?:general|definitions?|scope)\.?", head):
            return [picked["section"]]   # "General." in 82.50 / 82.1050 / 82.2050 names no color (code review)
        same = [c for c in cands if not c.get("statute") and c["part"] in COLOR_LISTING_PARTS
                and norm(c["heading"]) == head
                and not (picked["part"] == "82" and c["part"] == "82")]   # one color is listed once in Part 82
        lx = set(lexemes)
        wanted = {k for k, terms in _NON_COSMETIC_TERMS.items() if lx & terms}
        if cosmetic_question(lexemes) or not wanted:
            wanted.add("cosmetic")

        def keep(c):
            kind = _subpart_kind(c["subpart"])
            return c["part"] == "82" or kind is None or kind in wanted

        ordered = [picked] + [c for c in same if c["section"] != picked["section"]]
        chosen = [c["section"] for c in ordered if keep(c)] or [picked["section"]]
        return chosen[:2]

    def prepare(self, question: str, parts: tuple[str, ...] | None = None, named_only: bool = False,
                t0: float | None = None) -> Promotion | tuple[list[str], list[dict]] | None:
        """Everything resolve() decides before a pick: a Promotion of the FD&C Act section(s) the question names, None
        when nothing may be promoted, or (lexemes, candidates) to pick from. Deterministic (no model call): the
        section-resolution harness measures candidate generation through it."""
        t0 = time.perf_counter() if t0 is None else t0
        lex = SectionIndex.question_lexemes(self.db, question)
        named = self.named_statute(question, lex)
        if named:
            return self._statute_promotion(named[:2], lex, "named_statute", t0)
        if named_only:
            return None   # only a section the question names may be resolved here; nothing is guessed
        if parts and not all(p in COSMETIC_PARTS for p in parts) and not part_only_any():
            return None   # a part outside the cosmetics programme (or a misread statute number): never promoted into
        if parts and len({x for x in lex if not x.isdigit()} - _PART_QUESTION_WORDS - COSMETIC_TERMS_GENERIC) < 2:
            return None   # "What does Part 720 require?" asks about the whole part: not narrowed to one section
        lex, cands = self.candidates(question, parts)
        if cands and precision_fix("SN1"):
            cands = self.named_subpart(question, cands)
        if cands and precision_fix("QX1"):
            cands = self.expanded_candidates(question, parts, cands)
        offer_statute = self.statute is not None and cosmetic_question(lex) and not parts
        if offer_statute:
            cands = cands[:CFR_CANDIDATES_WITH_STATUTE] + self.statute_candidates(lex, question)
        elif (self.statute is not None and not parts and cands and (programme := routed_programme(question)) is not None
              and programme.act):
            # DR1: the Act sections the programme is issued under (the tobacco keys cite 904-911, which AU1 reached only
            # through a top candidate's part - T-001's were 25.35, 1.79, 1105.10)
            act = tuple(x for x in programme.act if x in self.statute.by_sec)
            cands = cands[:CFR_CANDIDATES_WITH_STATUTE] + self.statute_candidates(lex, question, act)[:GENERAL_ACT_OFFERED]
        elif self.statute is not None and not parts and cands and (general_act_offered() or precision_fix("AU1")):
            # SA1: the FD&C Act sections a general question is most about, after the regulations
            general = tuple(x for x in sorted(self.statute.by_sec) if x not in STATUTE_OFFERED and x not in _ACT_NOT_OFFERED)
            if precision_fix("AU1"):
                # AU1 (28 Oct): only the Act sections the top regulations' parts are issued under (their Authority line):
                # SA1 offered 502 for a PMTA question (T-002; Part 1114 is issued under 910) and 761 for a
                # structure/function claim (L-014)
                general = tuple(x for x in authority_sections([c["part"] for c in cands[:3] if not c.get("statute")])
                                if x in general and x not in _AU1_GENERIC)
            cands = cands[:CFR_CANDIDATES_WITH_STATUTE] + self.statute_candidates(lex, question, general)[:GENERAL_ACT_OFFERED]
        return (lex, cands) if cands else None

    def named_subpart(self, question: str, cands: list[dict]) -> list[dict]:
        """SN1 (28 Oct): a question naming "Part N Subpart X" is answered from that subpart - its candidates only, in
        their order, when it has any (a metadata filter); one named by its heading ("the Part 111 personnel subpart")
        has its candidates moved first, the rest kept after them (a boost: S-005 also asks for Subpart P's records rules). 300Q 27 Oct: "personnel under Part 117 Subpart B" promoted
        117.4 (Subpart A) for 117.10, "qualified facility under Part 117 Subpart D" 117.5 for 117.201. A subpart with no
        candidate, or one that does not exist (21 CFR 830 Subpart F, a trap question), leaves the candidates as they are."""
        m = _NAMED_SUBPART.search(question or "")
        boost = m is None   # a subpart named by its heading goes first; one named by its letter is the only choice
        if m:
            where, args = "parent_headings[3] LIKE %(h)s", {"p": m.group(1), "h": f"Subpart {m.group(2)} %"}
        else:
            m = _NAMED_SUBPART_HEADING.search(question or "")
            if not m:
                return cands
            where, args = "parent_headings[3] ~* %(h)s", {"p": m.group(1), "h": rf"^Subpart [A-Z]{{1,2}} - {m.group(2)}\M"}
        try:
            rows = self.db.query("SELECT DISTINCT section_number s FROM cfr_records WHERE removed_on IS NULL AND part_number = "
                                 f"%(p)s AND {where}", args)
        except Exception:  # noqa: BLE001 - an aid only
            return cands
        inside = {r["s"] for r in rows}
        kept = [c for c in cands if c["section"] in inside]
        # S-005 asks for the personnel subpart's records AND where the general recordkeeping rules are (Subpart P)
        return (kept + [c for c in cands if c["section"] not in inside] if boost else kept) if kept else cands

    def decomposed_sections(self, question: str, parts=None, home: tuple[str, ...] = ()) -> list[str]:
        """QD1 (27 Oct): question decomposition for a multi-part question (Question Decomposition for RAG, ACL SRW 2025):
        the model writes at most QD1_SUBQUESTIONS sub-questions; each one's lexical candidates are scored by the
        cross-encoder and its best regulation section, at QD1_MIN_SCORE or more, is returned (in sub-question order) to
        join the promoted section as a companion - a question needing several sections otherwise got one."""
        if self.generator is None or getattr(self.generator, "retrieval_only", False) or not RERANKER:
            return []
        try:
            kw = {"think_budget": 0} if getattr(self.generator, "thinking", False) is True else {}
            gen = self.generator.generate([{"role": "system", "content": QD1_SYSTEM},
                                           {"role": "user", "content": QD1_PROMPT.format(question=question)}],
                                          max_new_tokens=QD1_MAX_TOKENS, **kw)
            subs = [re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", x).strip() for x in (getattr(gen, "response", "") or "").splitlines()]
            subs = [x for x in subs if x.endswith("?") and 3 <= len(x.split()) <= 40][:QD1_SUBQUESTIONS]
        except Exception:  # noqa: BLE001 - an aid only
            return []
        # QD2: a comparison or interaction reaches beyond the part the question names - F-022 names Part 117 and needs
        # FSVP's Part 1, M-042 names Part 820 and needs Parts 807 and 830 - so its sub-questions search every part, at
        # the stricter floor MS1 and SP2F use (a section of another part must be clearly relevant)
        # (a comparison inside one part - "Under Part 111, the difference between QC and production personnel" - stays
        # in it: the other parts are searched only for a sub-question nothing in the named part answers; code review)
        across = precision_fix("QD2") and bool(_COMPARES.search(question or ""))
        kept_rerank = self.__dict__.get("last_rerank")   # the question's own scores, not the last sub-question's

        def best_for(sub, scope, floor):
            try:
                _, cands = self.candidates(sub, scope)
                cfr = [c for c in cands if not c.get("statute")][:QD1_CANDIDATES]
                scores = self.rerank_scores(sub, cfr)
            except Exception:  # noqa: BLE001
                return None
            best = max(cfr, key=lambda c: scores.get(c["section"], (0.0, ""))[0], default=None)
            score = scores.get(best["section"], (0.0, ""))[0] if best is not None else 0.0
            if (best is not None and scope is None and home and precision_fix("QD3")
                    and best["section"].split(".")[0] not in home):
                # QD3 (30 Oct): a sub-question searched across all parts adds a section of another part than the promoted
                # one's only at QD2's stricter floor - P-012 (UDI, Part 830) got 822.3 (postmarket surveillance)
                floor = max(floor, QD2_MIN_SCORE)
            return best["section"] if best is not None and score >= floor else None

        # DR1: a question routed to its programme keeps it for its sub-questions - L-015's "where are those names
        # codified?" searched every part and promoted 502.19 (animal food) for a food labeling question (30 Oct, FAIL)
        programme = routed_programme(question) if parts is None else None
        scope = tuple(sorted(programme.parts)) if programme is not None else parts
        out: list[str] = []
        for sub in subs if len(subs) >= 2 else []:
            found = best_for(sub, scope, QD1_MIN_SCORE)
            if found is None and across and parts:
                found = best_for(sub, None, QD2_MIN_SCORE)
            if found is not None and found not in out:
                out.append(found)
        if kept_rerank is not None:
            self.__dict__["last_rerank"] = kept_rerank
        return out

    def expanded_candidates(self, question: str, parts, cands: list[dict]) -> list[dict]:
        """QX1 (27 Oct): LLM query expansion (HyDE / Query2doc): the model restates the question in the regulation's own
        terms and the sections that restatement ranks first join the candidates in place of the last-ranked - "Can we ask
        FDA to exempt it from certification?" never says "petition", and 71.18 ("Petition for exemption from
        certification") was not among the 22 candidates. The picker still reads the user's question; nothing is added
        when there is no model or the reply is unusable."""
        if self.generator is None or getattr(self.generator, "retrieval_only", False):
            return cands
        try:
            kw = {"think_budget": 0} if getattr(self.generator, "thinking", False) is True else {}
            gen = self.generator.generate([{"role": "system", "content": EXPANSION_SYSTEM},
                                           {"role": "user", "content": EXPANSION_PROMPT.format(question=question)}],
                                          max_new_tokens=EXPANSION_MAX_TOKENS, **kw)
            text = " ".join((getattr(gen, "response", "") or "").split())
        except Exception:  # noqa: BLE001 - an aid only
            return cands
        if not text or len(text) > 400 or re.search(r"\d+\.\d+", text):
            return cands   # empty, rambling, or citing a section (a guessed citation never steers retrieval)
        self.__dict__["last_expansion"] = text
        _, more = self.candidates(text, parts)
        have = {c["section"] for c in cands}
        new = [c for c in more if not c.get("statute") and c["section"] not in have][:EXPANSION_ADD]
        if not new:
            return cands
        cfr = [c for c in cands if not c.get("statute")]
        return cfr[:max(0, len(cfr) - len(new))] + new + [c for c in cands if c.get("statute")]

    def part_paragraphs(self, question: str, part: str, k: int = DV1_TOP) -> list[str]:
        """DV1 (27 Oct): the paragraphs of a small 21 CFR part most relevant to the question by the cross-encoder
        ("820.10(b)(4)" for "How must advisory notices be handled under Part 820?"), at most `k`, scored at least
        DV1_MIN_SCORE, best first - paragraph-level retrieval for a question that names only the part. [] without the
        reranker."""
        from regulus.retrieval.rerank import reranker

        rr = reranker(RERANKER) if RERANKER else None
        if rr is None:
            return []
        rows = self.db.query("SELECT section_number s, paragraph_path p, section_heading h, body_text t FROM cfr_records "
                             "WHERE removed_on IS NULL AND NOT is_reserved AND part_number = %(p)s ORDER BY ordinal", {"p": part})
        # every row is scored: a definitions paragraph is one row per term under one path (820.3(b) has six; the
        # "Safety and Performance" row was the fourth and was dropped as a duplicate citation)
        items = [(f"{r['s']}{r['p'] or ''}", f"{r['h'] or ''} {r['t']}") for r in rows
                 if (r["t"] or "").strip() and "[Reserved]" not in (r["t"] or "")]
        if not items:
            return []
        try:
            scores = rr.score(question, [t for _, t in items])
        except Exception:  # noqa: BLE001 - an aid only
            return []
        # hybrid: the cross-encoder's rank fused (reciprocal rank) with the rank by two-word phrases the paragraph shares
        # with the question ("design and development" is 820.10(c)'s own phrase; the cross-encoder preferred 820.7(b))
        def phrases(t):
            w = [x for x in re.findall(r"[a-z]{3,}", (t or "").lower()) if x not in _PHRASE_STOP]
            return set(zip(w, w[1:]))
        qp = phrases(question)
        overlap = [len(qp & phrases(t)) for _, t in items]
        by_ce = {i: r for r, i in enumerate(sorted(range(len(items)), key=lambda i: -scores[i]), 1)}
        by_ph = {i: r for r, i in enumerate(sorted(range(len(items)), key=lambda i: -overlap[i]), 1)}
        fused = sorted(range(len(items)), key=lambda i: -(1 / (RRF_K + by_ce[i]) + (1 / (RRF_K + by_ph[i]) if overlap[i] else 0)))
        # kept: a confident cross-encoder score, two shared phrases, or the most phrases any paragraph shares (820.10(c)
        # shares only "design and development" and scored under DV1_MIN_SCORE)
        top_overlap = max(overlap)
        out: list[str] = []
        best = None
        for i in fused:
            if len(out) >= k:
                break
            if (scores[i] >= DV1_MIN_SCORE or overlap[i] >= 2 or (overlap[i] and overlap[i] == top_overlap)) \
                    and items[i][0] not in out:
                out.append(items[i][0])
                best = max(best or 0.0, float(scores[i]))
        self.__dict__["last_part_score"] = best   # DV2: the caller's Q2 quote decision
        whole = frozenset()
        if precision_fix("DV2"):
            # DV2 (29 Oct): "What is the scope of Part 820 ..." / "Which definitions apply to Part 820 ..." asks for the
            # part's frame as a whole - its Scope or Definitions section is supplied whole, not one paragraph family (300Q
            # full300c: M-001 got 820.1(a) and (c) without (b) and (d), M-008 820.3(b) without (a) - both PASS lost)
            frame = _FRAME_HEADINGS.get(next((kind for kind, rx in _FRAME_ASKED.items() if rx.search(question or "")), ""))
            if frame is not None:
                whole = frozenset(r["s"] for r in rows if frame.match(r["h"] or ""))
        return self.paragraph_families(out, items, whole)

    def sibling_companions(self, section: str, question: str, limit: int = HC1_MAX) -> list[str]:
        """HC1 (27 Oct): sections of the same subpart whose heading names a topic of the question that the promoted
        section's heading does not - "drug product inspection and expiration dating" promoted 211.137 (Expiration
        dating) and missed its sibling 211.134 (Drug product inspection); "returned and salvaged" missed 211.208. Drug
        CGMP keys span sibling sections: 9 of 13 imperfect drug answers lacked one (27 Oct). Most-covering first."""
        try:
            rows = self.db.query(
                "SELECT DISTINCT section_number s, section_heading h FROM cfr_records WHERE removed_on IS NULL AND NOT "
                "is_reserved AND subpart_id = (SELECT subpart_id FROM cfr_records WHERE section_number = %(s)s AND "
                "removed_on IS NULL AND subpart_id IS NOT NULL LIMIT 1)", {"s": section})
        except Exception:  # noqa: BLE001 - an aid only
            return []
        own = next((r["h"] for r in rows if r["s"] == section), None)
        if own is None:
            return []
        mine = _topic_stems(own)
        wanted: set[str] = set()
        for left, right in _coordinated(question):
            # "drug product inspection and expiration dating": the promoted heading covers one side, a sibling the other
            if mine & right and not mine & left:
                wanted |= left
            elif mine & left and not mine & right:
                wanted |= right
        if not wanted:
            return []
        scored = sorted(((len(wanted & _topic_stems(r["h"])), r["s"]) for r in rows if r["s"] != section), reverse=True)
        return [sec for n, sec in scored if n][:limit]

    def small_subpart(self, section: str) -> list[str]:
        """SP3 (27 Oct): the other sections of the promoted section's subpart, in order, when the whole subpart is small
        (at most SP3_MAX_SECTIONS sections and SP3_MAX_CHARS characters) - parent-document retrieval at subpart level.
        Part 111 is built of small subparts, each a requirement section with its records section: "What must a firm do
        when it receives a product complaint?" promoted 111.560 and the key is Subpart O, 111.553 + 111.560 + 111.570
        (13 of 15 supplement answers lacked the rest of their subpart, 27 Oct)."""
        try:
            rows = self.db.query(
                "SELECT section_number s, sum(length(body_text)) c, min(ordinal) o FROM cfr_records WHERE removed_on IS NULL "
                "AND NOT is_reserved AND subpart_id = (SELECT subpart_id FROM cfr_records WHERE section_number = %(s)s AND "
                "removed_on IS NULL AND subpart_id IS NOT NULL LIMIT 1) GROUP BY 1 ORDER BY 3", {"s": section})
        except Exception:  # noqa: BLE001 - an aid only
            return []
        if not rows or len(rows) > SP3_MAX_SECTIONS or sum(r["c"] or 0 for r in rows) > SP3_MAX_CHARS:
            return []
        return [r["s"] for r in rows if r["s"] != section]

    def heading_matches(self, section: str, question: str, limit: int = HC1_MAX) -> list[str]:
        """HM1 (27 Oct): sections of the promoted section's part whose heading's lead phrase the question contains word
        for word (two topic words or more) - title-field matching. "What are the requirements for use of approved
        components and for rejected components?" names 211.86 ("Use of approved components, ...") and 211.89 ("Rejected
        components, ..."); the picker chose 211.84 (testing and approval)."""
        part = section.split(".")[0]
        try:
            rows = self.db.query("SELECT DISTINCT section_number s, section_heading h FROM cfr_records WHERE part_number = "
                                 "%(p)s AND removed_on IS NULL AND NOT is_reserved", {"p": part})
        except Exception:  # noqa: BLE001
            return []
        out = []
        for r in sorted(rows, key=lambda r: [int(x) if x.isdigit() else 0 for x in r["s"].split(".")]):
            if r["s"] != section and not _GENERIC_HEADING.match(r["h"] or "") and heading_phrase_in(r["h"], question):
                out.append(r["s"])
        return out[:limit] if len(out) <= limit else []   # a question every heading matches names none of them

    def general_twin(self, section: str, question: str) -> str:
        """HT1 (27 Oct): a section whose heading and number another part repeats (213.192 and 211.192, "Production record
        review") is read as the larger, general part's unless the question names the specialised part's subject: 213 is
        CGMP for medical gases, and a question about production record review got its 213.192. A question naming the
        section's own part keeps it (S-002, 28 Oct: "definitions that apply to 21 CFR Part 111" moved 111.3 to 112.3)."""
        part = section.split(".")[0]
        if re.search(rf"\b(?:part|21\s*C\.?F\.?R\.?)\s*{part}\b", question or "", re.I):
            return section
        try:
            rows = self.db.query(
                "SELECT t.section_number s, t.part_number p, t.parent_headings[2] title, t.section_heading h, (SELECT "
                "count(DISTINCT x.section_number) FROM cfr_records x WHERE x.part_number = t.part_number AND x.removed_on IS "
                "NULL) n FROM cfr_records t, cfr_records o WHERE o.section_number = %(s)s AND o.removed_on IS NULL AND "
                "t.removed_on IS NULL AND NOT t.is_reserved AND t.section_heading = o.section_heading AND t.subchapter_code = "
                "o.subchapter_code AND split_part(t.section_number, '.', 2) = split_part(%(s)s, '.', 2) GROUP BY 1, 2, 3, 4",
                {"s": section})
        except Exception:  # noqa: BLE001
            return section
        if len(rows) < 2 or _GENERIC_HEADING.match(rows[0]["h"] or "") or _OWN_PART_HEADING.search(rows[0]["h"] or ""):
            return section   # "Scope." / "Definitions." repeat across unrelated parts: not twins
        mine = next((r for r in rows if r["s"] == section), None)
        best = max(rows, key=lambda r: (r["n"] or 0, -float(r["p"] or 0)))
        if mine is None or best["s"] == section:
            return section
        own_words = _topic_stems(mine["title"] or "") - _topic_stems(best["title"] or "")
        return section if own_words & _topic_stems(question) else best["s"]

    @staticmethod
    def paragraph_families(picks: list[str], items: list[tuple[str, str]], whole: frozenset = frozenset()) -> list[str]:
        """Small-to-big: each picked paragraph widened to its top-level paragraph with every paragraph under it
        (820.1(a)(4) -> 820.1(a), (a)(1)-(a)(4)), or to its whole section when the section is short (820.7: ISO 9000 in
        (a) beside ISO 13485 in (b)) - matched on the paragraph, answered from its context. Device fix check: three
        PASSes lost to a lone paragraph (M-001, M-004, M-010)."""
        size: dict[str, int] = {}
        for cite, text in items:
            sec = re.match(r"\d+\.\d+[a-z]?", cite).group(0)
            size[sec] = size.get(sec, 0) + len(text)
        out: list[str] = []
        for pick in picks:
            sec = re.match(r"\d+\.\d+[a-z]?", pick).group(0)
            top = re.match(r"\([a-z0-9]+\)", pick[len(sec):])
            if size.get(sec, 0) <= DV1_WHOLE_SECTION_CHARS or top is None or sec in whole:
                family = [sec]
            else:
                head = sec + top.group(0)
                family = [c for c, _ in items if c == head or c.startswith(head + "(")]
            out += [c for c in dict.fromkeys(family) if c not in out]
        return out

    def anchored(self, question: str, cfr: tuple[str, ...] = (), statute: tuple[str, ...] = ()) -> Promotion | None:
        """A follow-up resolved to the provisions its conversation is about (regulus.rag.conversation): those sections
        are promoted, method "conversation", with the companions this question's own words call for - no ranking, no
        picker. None when the question names an FD&C Act section itself or nothing anchored exists (the caller then
        resolves it as it would on its own); a cosmetic practice the regulations are silent on (P1) is answered as such."""
        if self.mode == "off":
            return None
        t0 = time.perf_counter()
        lex = SectionIndex.question_lexemes(self.db, question)
        if self.named_statute(question, lex):
            return None
        silence = self.practice_silence(question, lex, t0)
        if silence is not None:
            return silence
        known = [s for s in cfr if s in set(self.index.sections)]
        acts = [s for s in statute if self.statute is not None and s in self.statute.by_sec]
        if known:
            pr = self._cfr_promotion(known, question, lex, (), None, None, "conversation", t0)
            return pr
        if acts:
            return self._statute_promotion([(s, "") for s in acts[:2]], lex, "conversation", t0)
        return None

    def statute_references(self, sections: list[str], limit: int = 2) -> list[tuple[str, str]]:
        """ST1: the FD&C Act sections (with the cited subsection) that these 21 CFR sections' own text cites - "section
        403(i) of the act", "sections 501 and 502 of the Federal Food, Drug, and Cosmetic Act" - that the Act store
        holds, in order of first citation, at most `limit`."""
        st = self.statute
        if st is None:
            return []
        out: list[tuple[str, str]] = []
        for sec in sections:
            for m in _ACT_CITED.finditer(self._section_text(sec)):
                for num, path in re.findall(r"(\d{3}[A-Za-z]?)((?:\([a-zA-Z0-9]{1,6}\))*)", m.group(1)):
                    if num in st.by_sec and num not in [x for x, _ in out]:
                        out.append((num, path))
                    if len(out) >= limit:
                        return out
        return out

    def statute_companions(self, question: str, sections: list[str]) -> Promotion | None:
        """ST1: the cited FD&C Act text as a companion of the regulation (method "statute_reference"), or None."""
        refs = self.statute_references(sections)
        if not refs:
            return None
        t0 = time.perf_counter()
        lex = SectionIndex.question_lexemes(self.db, question)
        return self._statute_promotion(refs, lex, "statute_reference", t0, budget=CO_STATUTE_BUDGET)

    def ranks_any(self, question: str, cfr: tuple[str, ...] = (), statute: tuple[str, ...] = (), top: int = 8) -> bool:
        """Whether the question's own words rank one of these sections among their first `top` candidates (no model
        call)."""
        # FD&C Act sections are not checked: the Act's general sections (601 adulteration) lead the ranking of most
        # cosmetics questions, so ranking says nothing about whether a question is about them
        if not cfr:
            return False
        lex, cands = self.candidates(question)
        return any(c["section"] in cfr for c in cands[:top])

    def heading_words(self, cfr: tuple[str, ...]) -> list[str]:
        """The headings of these sections (for the follow-up's shared-subject test)."""
        ix = self.index
        pos = {s: i for i, s in enumerate(ix.sections)}
        return [ix.headings[pos[s]] or "" for s in cfr if s in pos]

    def resolve(self, question: str, parts: tuple[str, ...] | None = None, named_only: bool = False) -> Promotion | None:
        """`parts`: the question names only these parts ("Under Part 82, ..."); the answer is looked for inside them."""
        self.__dict__["_asked"] = question   # AP1 ranks the FD&C Act paragraphs supplied against it
        if self.mode == "off":
            return None
        t0 = time.perf_counter()
        if self.mode == "picker" and not named_only and (self.generator is None or getattr(self.generator, "retrieval_only", False)):
            # no model to pick with (retrieval-only harness): nothing is promoted, nothing is ranked - except an FD&C
            # Act section the question names, which needs no pick
            lex = SectionIndex.question_lexemes(self.db, question)
            named = self.named_statute(question, lex)
            return self._statute_promotion(named[:2], lex, "named_statute", t0) if named else None
        prepared = self.prepare(question, parts, named_only, t0)
        if not isinstance(prepared, tuple):
            return prepared
        lex, cands = prepared
        if not parts and precision_fix("AMB1") and deployment_scope() != "cosmetics":
            # AMB1: the same provision written for several product types, and the question names none of them - the
            # reader is asked which (regulus.rag.product_clarification), not given one type's rule chosen for them
            from regulus.rag.product_clarification import choices as product_choices, sections as choice_sections

            found = product_choices(question, cands, self.index)
            if found:
                return Promotion(sections=choice_sections(found), companions=(), method="product_clarification",
                                 candidates=tuple(cands), seconds=time.perf_counter() - t0,
                                 notes=(f"product types: {', '.join(found)}",),
                                 choices=tuple((k, tuple((c.section, c.heading, c.subpart) for c in v)) for k, v in found.items()))
        raw, other, notes, second_act, second_cfr = None, None, (), None, None
        act_defined = self.act_definition_asked(question) if not parts else None
        if act_defined:
            # DF2: the Act paragraph that defines what the question asks something to be classified as
            return self._statute_promotion([act_defined], lex, "act_definition", t0, cands)
        defined = self.definitions_asked(question) if not parts else None
        if defined:
            # DF1: a question asking how the definitions classify what it describes is answered from those definitions
            return Promotion(sections=tuple(defined), companions=(), method="definitions", candidates=tuple(cands),
                             seconds=time.perf_counter() - t0)
        try:
            entity = self.entity_pick(question, lex, cands, parts) if ENTITY_CONSTRAINTS else None
        except Exception:  # noqa: BLE001 - an aid: without it the picker decides
            entity = None
        if entity is not None:
            return self._cfr_promotion(entity, question, lex, cands, None, None, "entity", t0)
        if _LISTING_SILENT_ON_EYE.search(question or "") and precision_fix("EY1"):
            # EY1: no color named, and the question says its listing does not provide for eye-area use: the rule for
            # that is 70.5(a), not the listing the picker guesses for "a certified FD&C red" (74.2340, 27 Oct)
            return Promotion(sections=("70.5(a)",), companions=("70.3(s)",), method="eye_area_rule",
                             candidates=tuple(cands), seconds=time.perf_counter() - t0)
        silence = self.practice_silence(question, lex, t0)
        if silence is not None:
            return silence
        if self.mode == "lexical":
            first = next((c for c in cands if not c.get("statute")), None)
            if first is None:
                return None
            picked_c = first
        else:
            if self.generator is None or getattr(self.generator, "retrieval_only", False):
                return None   # no model to pick with (retrieval-only harness): nothing is promoted
            if RERANKER and precision_fix("RO1"):
                cands = self.rerank_order(question, cands)
            raw = self.cascade_pick(question, cands) if precision_fix("PC1") else None
            if raw is None:
                msgs = [{"role": "system", "content": PICKER_SYSTEM}, {"role": "user", "content": picker_prompt(question, cands)}]
                kw = {"think_budget": PICKER_THINK_BUDGET} if getattr(self.generator, "thinking", False) is True else {}
                result = self.generator.generate(msgs, max_new_tokens=PICKER_MAX_TOKENS, **kw)
                raw = (getattr(result, "response", None) or "").strip()
            picks = parse_picks(raw, len(cands), [c["section"] for c in cands])
            if not picks:
                return None
            picked_c = cands[picks[0] - 1]   # first choice; a second choice is used only from the other corpus
            other = next((cands[i - 1] for i in picks[1:] if bool(cands[i - 1].get("statute")) != bool(picked_c.get("statute"))), None)
            if SECOND_ACT and picked_c.get("statute"):
                # "FD&C Act § 604, FD&C Act § 605": both Act sections, the Act budget shared (605-Q2 lost 605 to 604)
                second_act = next((cands[i - 1] for i in picks[1:] if cands[i - 1].get("statute")
                                   and cands[i - 1]["section"] != picked_c["section"]), None)
            if RERANKER:
                vetoed = self.rerank_veto(question, cands, picked_c)
                if vetoed is not None and precision_fix("V2"):
                    # V2 (27 Oct): in a near tie the picker's own other choice wins over the reranker's top - 71.6-Q1:
                    # picker 71.2, then 71.6; reranker 71.20 0.913, 71.6 0.897; the veto chose 71.20
                    scored = self.__dict__.get("last_rerank") or {}
                    top = scored.get(vetoed["section"], (0.0, ""))[0]
                    own = next((cands[i - 1] for i in picks[1:] if not cands[i - 1].get("statute")
                                and cands[i - 1]["section"] != vetoed["section"]
                                and scored.get(cands[i - 1]["section"], (0.0, ""))[0] >= top - V2_TIE), None)
                    if own is not None:
                        vetoed = own
                if vetoed is not None:
                    notes = (f"rerank veto: {picked_c['section']} -> {vetoed['section']}",)
                    picked_c, other = vetoed, (None if other is not None and other["section"] == vetoed["section"] else other)
                    if precision_fix("VB1") and other is not None and bool(other.get("statute")) == bool(picked_c.get("statute")):
                        # VB1: the second pick stands only from the other corpus. A veto from 720.1 to FD&C Act 606 kept the
                        # picker's Act 612 as the "regulation" and promoted "21 CFR 612" (FDCA606-Q2 declined, 27 Oct)
                        other = None
            second_cfr = second_in_part(picks, cands, picked_c)
            if second_cfr is not None and precision_fix("SP2F"):
                # SP2F (28 Oct): SP2 live, 3 of 3 losses were second picks outside the key (F-004 117.135, S-005 111.140,
                # V-001 1.505); a second pick the cross-encoder does not also rate is not added
                kept = self.__dict__.get("_order_scores")
                scores = (kept[2] if kept and kept[0] == question and set(kept[1]) == {c["section"] for c in cands}
                          else self.rerank_scores(question, cands))
                if scores and scores.get(second_cfr, (0.0, ""))[0] < SP2F_MIN_SCORE:
                    notes, second_cfr = notes + (f"second pick {second_cfr} below the reranker floor",), None
        if picked_c.get("statute"):
            acts = [(picked_c["section"], named_act_path(picked_c["section"], question, raw))] + (
                [(second_act["section"], named_act_path(second_act["section"], question, raw))]
                if second_act is not None and second_act["section"] != picked_c["section"] else [])
            pr = self._statute_promotion(acts, lex, self.mode, t0, cands, raw)
            if other is not None:   # FD&C Act section first, a regulation second: the regulation is promoted with it
                sec = self.siblings(other, cands, lex)
                comps = [c for s in sec for c in self.companions(s, question, lex)]
                pr = replace(pr, sections=tuple(sec), companions=tuple(dict.fromkeys(c for c in comps if c not in sec)))
            elif precision_fix("AU2") and not pr.sections:
                implementing = self.implementing_regulation(question, picked_c["section"], cands)
                if implementing:
                    pr = replace(pr, sections=(implementing,), notes=pr.notes + (f"implementing regulation: {implementing}",))
            return replace(pr, notes=pr.notes + notes)
        if second_cfr is not None and precision_fix("SP2O"):
            # SP2O (28 Oct): the picker's two same-part picks led by the one the cross-encoder rates clearly higher - Z-001
            # (SP2 live) picked 900.4 (accreditation bodies' duties) before 900.11 (facility certification), and the answer
            # was written from the first; the reranker veto only acts on a far larger gap
            kept = self.__dict__.get("_order_scores")
            scores = (kept[2] if kept and kept[0] == question and set(kept[1]) == {c["section"] for c in cands}
                      else self.rerank_scores(question, cands))
            first, second = scores.get(picked_c["section"], (0.0, ""))[0], scores.get(second_cfr, (0.0, ""))[0]
            swap = next((c for c in cands if c["section"] == second_cfr), None)
            if swap is not None and second >= first + SP2O_MARGIN:
                notes = notes + (f"second pick leads: {second_cfr} ({second:.2f}) over {picked_c['section']} ({first:.2f})",)
                picked_c, second_cfr = swap, picked_c["section"]
        pr = self._cfr_promotion(self.siblings(picked_c, cands, lex), question, lex, cands, raw, other, self.mode, t0)
        scored = self.__dict__.get("last_rerank") if RERANKER and raw is not None else None
        if scored and picked_c["section"] in scored:
            pr = replace(pr, rerank=round(scored[picked_c["section"]][0], 4))
        if second_cfr is not None and second_cfr not in pr.citations:
            pr = replace(pr, companions=pr.companions + (second_cfr,), notes=pr.notes + (f"second pick: {second_cfr}",))
        if precision_fix("MS1") and raw is not None:
            extra = self.reranked_companions(question, cands, pr)
            if extra:
                pr = replace(pr, companions=pr.companions + tuple(extra), notes=pr.notes + (f"reranked: {', '.join(extra)}",))
        if precision_fix("XP1") and pr.sections:
            cited = self.cross_part_citations(pr)
            if cited:
                pr = replace(pr, companions=pr.companions + tuple(cited), notes=pr.notes + (f"cross-part: {', '.join(cited)}",))
        if precision_fix("NAV1") and raw is not None and pr.sections:
            nav = self.navigate_subpart(question, pr)
            if nav:
                pr = replace(pr, companions=pr.companions + tuple(nav), notes=pr.notes + (f"navigated: {', '.join(nav)}",))
        if precision_fix("QD1") and multi_part(question):
            cited = {c.split("(")[0] for c in pr.citations}   # a paragraph PL1 added is its section (code review)
            home = tuple(dict.fromkeys(x.split(".")[0] for x in pr.sections))
            extra = [s2 for s2 in self.decomposed_sections(question, parts, home) if s2.split("(")[0] not in cited][:QD1_MAX]
            if extra:
                pr = replace(pr, companions=pr.companions + tuple(extra), notes=pr.notes + (f"sub-questions: {', '.join(extra)}",))
        return replace(pr, notes=pr.notes + notes) if notes else pr

    def _cfr_promotion(self, picked: list[str], question: str, lex, cands, raw, other, method: str, t0: float) -> Promotion:
        """The Promotion of picked regulation(s): their companions, and a second FD&C Act pick as co-statute."""
        if precision_fix("HT1"):
            picked = list(dict.fromkeys(self.general_twin(s, question) for s in picked))
        comps: list[str] = []
        for s in picked:
            comps += [c for c in self.companions(s, question, lex) if c not in comps]
        if precision_fix("HC1"):
            for s in picked:
                comps += [c for c in self.sibling_companions(s, question) if c not in comps and c not in picked]
        if precision_fix("SP3") and picked and picked[0].split(".")[0] in SP3_PARTS:
            from regulus.rag.list_answers import _ASKS_CONTENTS

            if not _ASKS_CONTENTS.search(question or ""):   # a list question is answered by its own section's list (LS1)
                comps += [c for c in self.small_subpart(picked[0]) if c not in comps and c not in picked]
        if precision_fix("HM1") and picked:
            comps += [c for c in self.heading_matches(picked[0], question) if c not in comps and c not in picked]
        comps += [c for c in self.definition_companions(question) if c not in comps and c not in picked]
        if precision_fix("DQ1") and picked:
            comps += [c for c in self.asked_definitions(picked[0], question) if c not in comps and c not in picked]
        if "70.5(a)" in comps and any(_EYE_USE.search(self._section_text(s)) for s in picked):
            comps.remove("70.5(a)")   # one promoted listing provides for eye-area use (code review: Part 82 sibling)
        if (other is not None and other.get("statute") and precision_fix("CS1") and picked
                and _COLOR_ADDITIVE_PART.match(picked[0]) and other["section"] in _MOCRA_SECTIONS):
            # CS1: Parts 70-82 carry out section 721 of the Act; MoCRA's 603-614 (product listing, safety
            # substantiation, labeling contact) were co-promoted to 5 of 16 Part 70 petition questions (27 Oct), their
            # text filling the budget and once restated in the answer
            other = None
        if other is not None and other.get("statute"):
            # a regulation first and an FD&C Act section second (registration rule 710 with Act 607): both, the Act text
            # as additional evidence with a smaller budget (retrieval diagnosis: 11 unseen questions need both)
            st = self._statute_promotion([(other["section"], "")], lex, method, t0, (), None, budget=CO_STATUTE_BUDGET)
            return Promotion(sections=tuple(picked), companions=tuple(c for c in comps if c not in picked), method=method,
                             statute=st.statute, statute_rows=st.statute_rows, statute_blocks=st.statute_blocks,
                             statute_left_out=st.statute_left_out, candidates=tuple(cands), picker_output=raw,
                             seconds=time.perf_counter() - t0)
        return Promotion(sections=tuple(picked), companions=tuple(c for c in comps if c not in picked), method=method,
                         candidates=tuple(cands), picker_output=raw, seconds=time.perf_counter() - t0)
