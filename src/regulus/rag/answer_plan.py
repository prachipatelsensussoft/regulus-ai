"""Evidence planning: how the selected evidence should be organised for the
question's answer shape (Stage 12).

Retrieval decides what evidence is relevant; this module decides how that
evidence is organised; generation decides how it is expressed. Nothing here
removes evidence. Every source of the GenerationPackage is placed in exactly
one slot of the AnswerPlan (the lead-in, one group, context or related), so
`AnswerPlan.all_indices()` always equals the package's source indices: a
synthesized answer can drop a paragraph, but the plan never drops a source.

Hierarchy comes from existing eCFR identity (section, paragraph path):

    section-level request     groups = top-level paragraphs; descendants are
                              attached to their parent; the section row is the
                              lead-in
    paragraph-level request   one group per requested provision, with its
                              children; structural parents are context
    part-level request        one group per section
    no CFR request            one group per document (guidance) or section

Each source gets a role (lead_in, requirement, subordinate, consequence,
exception, definition, context, related) and attributes that a row can carry
whatever its role: qualifiers ("if applicable", "as appropriate", ...),
applicability (the devices or persons a requirement is limited to) and
cross-references (part 830, Clause 7.5.9.1, section 501(h)). All of it is
read from the source text by fixed patterns; nothing is paraphrased.

`format_line()` turns the plan into the one ANSWER FORMAT instruction the model
sees (see the note above it for why it is prose, not a block). It says nothing
about length: that remains Stage 11's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from regulus.planning.shape import (
    MODE_COMPARISON,
    MODE_DIRECT,
    MODE_ENUMERATION,
    MODE_FOCUSED,
    MODE_SYNTHESIS,
    AnswerShape,
)
from regulus.retrieval.evidence import TIER_EXACT, TIER_STRUCTURAL

if TYPE_CHECKING:  # annotation only
    from regulus.planning.plan import Requirement

ROLE_LEAD_IN = "lead_in"            # a section row that introduces its paragraphs ("A manufacturer ... must:")
ROLE_REQUIREMENT = "requirement"    # a requirement heading a group
ROLE_SUBORDINATE = "subordinate"    # a requirement under a parent requirement
ROLE_CONSEQUENCE = "consequence"    # what non-compliance leads to
ROLE_EXCEPTION = "exception"        # a carve-out from a requirement
ROLE_DEFINITION = "definition"      # a defined term
ROLE_CONTEXT = "context"            # a structural parent of a requested provision
ROLE_RELATED = "related"            # supporting or discovery material
MATERIAL_ROLES = (ROLE_REQUIREMENT, ROLE_SUBORDINATE, ROLE_CONSEQUENCE, ROLE_EXCEPTION, ROLE_DEFINITION)

GROUP_REQUIREMENT = "requirement"
GROUP_CONSEQUENCE = "consequence"
GROUP_SIDE = "comparison_side"
GROUP_DOCUMENT = "document"

# (canonical phrase, pattern) - the canonical phrase is what the prompt asks the answer to keep
QUALIFIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("if applicable", re.compile(r"\bif applicable\b", re.I)),
    ("where applicable", re.compile(r"\bwhere applicable\b", re.I)),
    ("as applicable", re.compile(r"\bas applicable\b", re.I)),
    ("as appropriate", re.compile(r"\bas appropriate\b", re.I)),
    ("where appropriate", re.compile(r"\bwhere appropriate\b", re.I)),
    ("to the extent", re.compile(r"\bto the extent\b", re.I)),
    ("unless", re.compile(r"\bunless\b", re.I)),
    ("except", re.compile(r"\bexcept(?:ing)?\b(?! as provided)", re.I)),
    ("in addition to", re.compile(r"\bin addition to\b", re.I)),
    ("including, but not limited to", re.compile(r"\bincluding,? but not limited to\b", re.I)),
)
# wording that preserves each qualifier in an answer (used by shape validation)
QUALIFIER_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "if applicable": ("if applicable", "where applicable", "when applicable", "as applicable", "if it applies", "where it applies", "applicable"),
    "where applicable": ("if applicable", "where applicable", "when applicable", "as applicable", "applicable"),
    "as applicable": ("if applicable", "where applicable", "when applicable", "as applicable", "applicable"),
    "as appropriate": ("as appropriate", "where appropriate", "when appropriate", "appropriate"),
    "where appropriate": ("as appropriate", "where appropriate", "when appropriate", "appropriate"),
    "to the extent": ("to the extent", "extent"),
    "unless": ("unless", "except"),
    "except": ("except", "unless", "exempt", "other than"),
}
# qualifiers whose loss changes whether a requirement applies (checked in validation)
CONDITIONAL_QUALIFIERS = tuple(QUALIFIER_EQUIVALENTS)

_CROSS_REF_RE = re.compile(
    r"\bparts?\s+\d{1,4}\b"
    r"|§+\s*\d{1,4}\.\d{1,4}[a-z]?(?:\((?:[a-z0-9]{1,4}|[ivxl]{5,8})\))*"
    r"|\bsection\s+\d{2,4}\([a-z]\)"
    r"|\bClauses?\s+\d+(?:\.\d+)+(?:(?:,\s*(?:and\s+)?|\s+and\s+)\d+(?:\.\d+)+)*",
    re.I,
)
_APPLICABILITY_RES = (
    re.compile(r"\bManufacturers of (.{5,260}?)(?:,)?\s+(?:must|shall)\b", re.S),
    re.compile(r"\bA manufacturer subject to (this part[^:;]{0,80}?)\s+(?:must|shall)\b", re.S),
    re.compile(r"\b(?:This|These) (?:section|paragraph|requirements?) appl(?:y|ies) (?:only )?to (.{5,200}?)[.;]", re.S),
)
_CONSEQUENCE_RE = re.compile(r"\b(adulterated|misbranded|subject to regulatory action|civil money penalt|renders? (?:a|the) device)\b", re.I)
_EXCEPTION_RE = re.compile(r"\b(?:does|do|shall) not apply\b|\bexempt(?:ed|ion|ions)?\b|^\s*(?:\([a-z0-9]+\)\s*)?Exceptions?\b", re.I)
_DEFINITION_RE = re.compile(r"^\s*(?:\([a-z0-9]+\)\s*)?[A-Z][^.]{0,80}?\bmeans\b", re.S)
_HEADING_RE = re.compile(r"^\s*\(([a-z0-9]{1,4}|[ivxl]{5,8})\)\s+([A-Z][A-Za-z ,'—-]{1,70}?)\.\s")
_TOKEN_RE = re.compile(r"\(([A-Za-z0-9]{1,4}|[ivxl]{5,8})\)")


def _tokens(path: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(path or ""))


def _path(tokens: Iterable[str]) -> str:
    return "".join(f"({t})" for t in tokens)


def extract_qualifiers(text: str) -> tuple[str, ...]:
    return tuple(name for name, pat in QUALIFIER_PATTERNS if pat.search(text or ""))


def extract_cross_references(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for m in _CROSS_REF_RE.finditer(text or ""):
        ref = re.sub(r"\s+", " ", m.group(0).strip())
        ref = re.sub(r"^parts?", lambda x: x.group(0).lower(), ref, flags=re.I)
        if ref not in out:
            out.append(ref)
    return tuple(out)


APPLICABILITY_MAX_CHARS = 170


def extract_applicability(text: str) -> str | None:
    """Who or what a requirement is limited to. A long limit is cut at a clause
    boundary, never mid-word, and marked as shortened; the full text is in the source."""
    for pat in _APPLICABILITY_RES:
        m = pat.search(text or "")
        if m:
            phrase = re.sub(r"\s+", " ", m.group(1)).strip(" ,")
            if len(phrase) <= APPLICABILITY_MAX_CHARS:
                return phrase
            cut = phrase.find(",", 30)
            if 0 < cut <= APPLICABILITY_MAX_CHARS:
                return phrase[:cut] + " (as further limited in the source)"
            return phrase[: phrase.rfind(" ", 0, APPLICABILITY_MAX_CHARS)] + " (as further limited in the source)"
    return None


def extract_heading(text: str) -> str | None:
    """The paragraph's own eCFR heading ("(b) Applicable regulatory requirements. ..."), when it has one."""
    m = _HEADING_RE.match(text or "")
    if not m:
        return None
    heading = m.group(2).strip()
    return heading if 1 <= len(heading.split()) <= 8 else None


# ------------------------------------------------------------------ model


@dataclass(frozen=True)
class PlannedSource:
    index: int
    citation: str
    tier: str
    role: str
    section: str | None
    paragraph: str
    heading: str | None
    qualifiers: tuple[str, ...] = ()
    applicability: str | None = None
    cross_references: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()

    @property
    def material(self) -> bool:
        return self.role in MATERIAL_ROLES

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "citation": self.citation, "tier": self.tier, "role": self.role,
            "section": self.section, "paragraph": self.paragraph, "heading": self.heading,
            "qualifiers": list(self.qualifiers), "applicability": self.applicability,
            "cross_references": list(self.cross_references), "requirement_ids": list(self.requirement_ids),
        }


@dataclass(frozen=True)
class EvidenceGroup:
    key: str                          # "820.10(b)", "r1", "doc:<citation>"
    label: str                        # "21 CFR 820.10(b)" or a document title
    heading: str | None               # the head row's own heading, if any
    kind: str                         # requirement | consequence | comparison_side | document
    head: int | None                  # SOURCE index of the group's head row
    members: tuple[int, ...]          # every SOURCE index in the group, head first, then document order
    requirement_ids: tuple[str, ...] = ()

    @property
    def items(self) -> tuple[int, ...]:
        """Members other than the head: the group's subordinate items."""
        return tuple(i for i in self.members if i != self.head)

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "heading": self.heading, "kind": self.kind, "head": self.head,
                "members": list(self.members), "items": list(self.items), "requirement_ids": list(self.requirement_ids)}


@dataclass(frozen=True)
class AnswerPlan:
    shape: AnswerShape
    mode: str                                  # the mode the plan was built for (shape.mode unless adjusted)
    lead_in: tuple[int, ...]
    groups: tuple[EvidenceGroup, ...]
    context: tuple[int, ...]
    related: tuple[int, ...]
    sources: tuple[PlannedSource, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)
    # The time periods a value question needs, when the requested provision states more than one
    # (`value_entries`). Empty for every other question, which then renders exactly as before.
    values: tuple["ValueEntry", ...] = field(default_factory=tuple)

    def source(self, index: int) -> PlannedSource | None:
        return next((s for s in self.sources if s.index == index), None)

    def group_of(self, index: int) -> EvidenceGroup | None:
        return next((g for g in self.groups if index in g.members), None)

    def all_indices(self) -> tuple[int, ...]:
        """Every source index the plan places, in plan order. Equal to the package's indices."""
        out = list(self.lead_in)
        for g in self.groups:
            out.extend(g.members)
        out.extend(self.context)
        out.extend(self.related)
        return tuple(out)

    def material_items(self) -> tuple[int, ...]:
        """Sources an exhaustive answer must represent: every material member of every group."""
        return tuple(i for g in self.groups for i in g.members if (s := self.source(i)) is not None and s.material)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape.to_dict(), "mode": self.mode, "lead_in": list(self.lead_in),
            "groups": [g.to_dict() for g in self.groups], "context": list(self.context), "related": list(self.related),
            "sources": [s.to_dict() for s in self.sources], "notes": list(self.notes),
            **({"values": [v.to_dict() for v in self.values]} if self.values else {}),
        }


# ------------------------------------------------------------------ value plan
#
# A focused question that asks for a TIME PERIOD of a provision stating more than one gets, from
# the pipeline, the periods paired with what each applies to - the model is never asked to work
# that pairing out. Measured on 21 CFR 312.32 with the production engine (greedy): asked "How
# quickly must IND safety reports be submitted?", the focused instruction ("state only what the
# provision says about the focus") made the model state (c)(1)'s 15 days alone, because (c)(1)
# shares four of the question's words and (c)(2) one. Every instruction that instead asked the
# model to give "each period and what it applies to" made it invent the pairing, and invent it
# wrongly ("a serious and unexpected suspected adverse reaction ... 7 calendar days"). Handed the
# pairing, in each source's own words, it stated both rules correctly, with their own triggers and
# citations, in about a fifth of the length.

_PERIOD_KINDS = ("deadline", "duration_or_frequency")
_PERIOD_NUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|sixty|ninety)"
_PERIOD_RE = re.compile(rf"\b({_PERIOD_NUM})[\s-]+((?:calendar|working|business)[\s-]+)?(days?|weeks?|months?|years?|hours?)\b", re.I)
# A paragraph designation, however many tokens, before the paragraph's own heading: "(c)(1) IND
# safety reports. The sponsor ...". `extract_heading` reads a single token only and is used for
# group headings elsewhere, so it is left as it is.
_DESIGNATION_RE = re.compile(r"^\s*(?:\([A-Za-z0-9]{1,4}\)\s*)+")
_OWN_HEADING_RE = re.compile(r"^([A-Z][A-Za-z ,'\u2014-]{1,90}?)\.\s")
_SENTENCE_END_RE = re.compile(r"(?<=[a-z0-9)])\.(?=\s|$)")
_CLAUSE_BREAK_RE = re.compile(r"[,;:]\s+|(?<=[.])\s+")
_LEADING_CONJUNCTION_RE = re.compile(r"^(?:but|and|or|then|so)\s+", re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]+")
# Question words that name the VALUE asked for rather than a condition it applies to.
_VALUE_WORDS = frozenset({"deadline", "deadlines", "timeframe", "timeframes", "time", "limit", "limits", "timeline",
                          "timelines", "quickly", "soon", "long", "often", "rapidly", "promptly", "calendar-day",
                          "calendar", "day", "days", "period", "periods", "frequency", "frequently", "within", "many"})
NAMED_CONDITION_SHARE = 0.6  # share of the question's condition words a heading must carry to be the one it names


@dataclass(frozen=True)
class ValueEntry:
    value: str            # "7 calendar days"
    index: int            # the SOURCE stating it
    condition: str        # that paragraph's own eCFR heading: what the period applies to
    clause: str           # the source's own words around the period, through the end of its sentence

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "index": self.index, "condition": self.condition, "clause": self.clause}


def _paragraph_heading(text: str) -> str | None:
    body = _DESIGNATION_RE.sub("", text or "", count=1)
    if body == (text or ""):
        return None
    m = _OWN_HEADING_RE.match(body)
    heading = m.group(1).strip() if m else None
    return heading if heading and 1 <= len(heading.split()) <= 10 else None


def _periods(text: str) -> list[tuple[str, re.Match[str]]]:
    out: list[tuple[str, re.Match[str]]] = []
    for m in _PERIOD_RE.finditer(text or ""):
        unit = m.group(3).lower().rstrip("s") + "s"
        convention = re.sub(r"[\s-]+", " ", m.group(2) or "").strip().lower()
        value = " ".join(x for x in (m.group(1).lower(), convention, unit) if x)
        if value not in (v for v, _ in out):
            out.append((value, m))
    return out


def _value_clause(text: str, match: re.Match[str]) -> str:
    """From the clause boundary before the period to the end of its sentence, verbatim: the event it
    runs from and any cross-reference after it come with it."""
    end = _SENTENCE_END_RE.search(text, match.end())
    starts = [b.end() for b in _CLAUSE_BREAK_RE.finditer(text, 0, match.start())]
    clause = text[starts[-1] if starts else 0: end.start() if end else len(text)].strip()
    return _LEADING_CONJUNCTION_RE.sub("", clause)


def _stems(text: str) -> set[str]:
    return {w.lower()[:5] for w in _WORD_RE.findall(text or "")}


def value_entries(value_kind: str | None, shape: AnswerShape | None, sources) -> tuple[ValueEntry, ...]:
    """The periods a focused period question needs, each with what it applies to - or () to leave the
    prompt exactly as it was.

    Fires only for a FOCUSED question asking for a time period whose requested provision's EXACT
    sources state at least two DIFFERENT periods. Each distinct period is represented once, by the
    first source in document order that states it, and every one of those sources must carry its
    own paragraph heading - the heading is what the period is said to apply to, so a period without
    one makes the plan abstain rather than describe it in words the pipeline made up.

    A question that names one of those conditions is already about one period, and the focused
    answer handles it: "the reporting deadline for an unexpected fatal or life-threatening suspected
    adverse reaction" names (c)(2)'s heading and gets () - its prompt is unchanged. Words the
    question shares with the SECTION's heading ("IND safety reporting") name the whole section, not
    one condition in it, so they do not narrow anything.
    """
    if value_kind not in _PERIOD_KINDS or shape is None or shape.mode != MODE_FOCUSED:
        return ()
    exact = [s for s in sources if s.tier == TIER_EXACT and getattr(s, "corpus", "ecfr") == "ecfr"]
    first: dict[str, tuple[object, re.Match[str]]] = {}
    for s in exact:
        for value, m in _periods(s.text):
            first.setdefault(value, (s, m))
    if len(first) < 2:
        return ()
    entries = []
    for value, (s, m) in first.items():
        heading = _paragraph_heading(s.text)
        if heading is None:
            return ()
        entries.append(ValueEntry(value, s.index, heading, _value_clause(s.text, m)))
    umbrella = set().union(*(_stems(s.heading or "") for s in exact))
    condition_words = [t for t in shape.focus_terms if t not in _VALUE_WORDS and t.lower()[:5] not in umbrella]
    if condition_words:
        named = [e for e in entries
                 if sum(1 for t in condition_words if t.lower()[:5] in _stems(e.condition)) / len(condition_words)
                 >= NAMED_CONDITION_SHARE]
        if len(named) == 1:
            return ()
    return tuple(entries)


# ------------------------------------------------------------------ build


def _identity(source) -> tuple[str | None, tuple[str, ...]]:
    f = source.citation_fields or {}
    if source.corpus != "ecfr" or not f.get("section"):
        return None, ()
    return f.get("section"), _tokens(f.get("paragraph") or "")


def _role(source, section: str | None, tokens: tuple[str, ...], *, head: bool) -> str:
    if source.tier not in (TIER_EXACT, TIER_STRUCTURAL):
        return ROLE_RELATED
    text = source.text or ""
    if section is not None and not tokens and source.tier == TIER_EXACT and text.rstrip().endswith((":", "following")):
        return ROLE_LEAD_IN
    if _CONSEQUENCE_RE.search(text):
        return ROLE_CONSEQUENCE
    if _EXCEPTION_RE.search(text):
        return ROLE_EXCEPTION
    if _DEFINITION_RE.search(text):
        return ROLE_DEFINITION
    return ROLE_REQUIREMENT if head else ROLE_SUBORDINATE


def build_answer_plan(shape: AnswerShape, sources, requirements: "tuple[Requirement, ...]") -> AnswerPlan:
    """Organise the package's sources for `shape`. Every source is placed exactly once."""
    by_index = {s.index: s for s in sources}
    ident = {s.index: _identity(s) for s in sources}
    notes: list[str] = []
    lead_in: list[int] = []
    context: list[int] = []
    related: list[int] = []
    groups: list[EvidenceGroup] = []
    heads: dict[int, bool] = {}
    placed: set[int] = set()

    def place(i: int, bucket: list[int]) -> None:
        if i not in placed:
            placed.add(i)
            bucket.append(i)

    def ecfr_label(section: str, tokens: tuple[str, ...]) -> str:
        return f"21 CFR {section}{_path(tokens)}"

    requested_paths: dict[str, tuple[str | None, tuple[str, ...], str]] = {
        r.id: (r.section, _tokens(r.paragraph_path), r.level) for r in requirements
    }
    paragraph_reqs = [r for r in requirements if r.level == "paragraph"]
    section_reqs = [r for r in requirements if r.level == "section"]
    part_reqs = [r for r in requirements if r.level == "part"]

    # 1. paragraph-level requests: one group per requested provision, children attached
    #    (a child that was itself requested keeps its own group)
    requested_rows = {(r.section, _tokens(r.paragraph_path)) for r in paragraph_reqs}
    for r in paragraph_reqs:
        sec, want, _ = requested_paths[r.id]
        members: list[int] = []
        head = None
        for s in sources:
            if s.index in placed:
                continue
            ssec, toks = ident[s.index]
            if ssec != sec:
                continue
            if toks == want and s.tier == TIER_EXACT:
                head = s.index
                members.insert(0, s.index)
            elif len(toks) > len(want) and toks[: len(want)] == want and not (s.tier == TIER_EXACT and (ssec, toks) in requested_rows):
                members.append(s.index)
        if not members:
            continue
        placed.update(members)
        heads.update({i: i == head for i in members})
        head_src = by_index.get(head) if head is not None else None
        groups.append(EvidenceGroup(
            key=r.canonical, label=r.canonical, heading=extract_heading(head_src.text) if head_src else None,
            kind=GROUP_SIDE if shape.mode == MODE_COMPARISON else GROUP_REQUIREMENT,
            head=head, members=tuple(members), requirement_ids=(r.id,),
        ))

    # 2. section-level requests: top-level paragraphs as groups, descendants attached, section row as lead-in.
    #    In a comparison each requested section is one side.
    for r in section_reqs:
        sec = r.section
        rows = [s for s in sources if s.index not in placed and ident[s.index][0] == sec and s.tier in (TIER_EXACT, TIER_STRUCTURAL)]
        if shape.mode == MODE_COMPARISON and rows:
            members = [s.index for s in rows]
            placed.update(members)
            heads.update({i: n == 0 for n, i in enumerate(members)})
            groups.append(EvidenceGroup(key=r.canonical, label=r.canonical, heading=None, kind=GROUP_SIDE,
                                        head=members[0], members=tuple(members), requirement_ids=(r.id,)))
            continue
        top: dict[str, list[int]] = {}
        for s in rows:
            toks = ident[s.index][1]
            if not toks:
                if _role(s, sec, toks, head=True) == ROLE_LEAD_IN:
                    place(s.index, lead_in)
                    heads[s.index] = False
                else:  # a section with no paragraphs: the section row is the requirement itself
                    top.setdefault("", []).append(s.index)
                continue
            top.setdefault(toks[0], []).append(s.index)
        for first, members in top.items():
            head = next((i for i in members if len(ident[i][1]) <= 1), None)
            ordered = ([head] if head is not None else []) + [i for i in members if i != head]
            placed.update(ordered)
            heads.update({i: i == head for i in ordered})
            head_src = by_index.get(head) if head is not None else None
            kind = GROUP_CONSEQUENCE if head_src is not None and _CONSEQUENCE_RE.search(head_src.text or "") else GROUP_REQUIREMENT
            groups.append(EvidenceGroup(
                key=f"{sec}({first})" if first else sec, label=ecfr_label(sec, (first,) if first else ()),
                heading=extract_heading(head_src.text) if head_src else None, kind=kind, head=head,
                members=tuple(ordered), requirement_ids=(r.id,),
            ))

    # 3. part-level requests: one group per section
    for r in part_reqs:
        by_section: dict[str, list[int]] = {}
        for s in sources:
            ssec, toks = ident[s.index]
            if s.index in placed or ssec is None or not ssec.startswith(f"{r.part}.") or s.tier != TIER_EXACT:
                continue
            by_section.setdefault(ssec, []).append(s.index)
        for sec, members in by_section.items():
            placed.update(members)
            heads.update({i: n == 0 for n, i in enumerate(members)})
            groups.append(EvidenceGroup(key=sec, label=f"21 CFR {sec}", heading=None, kind=GROUP_REQUIREMENT,
                                        head=members[0], members=tuple(members), requirement_ids=(r.id,)))

    # 4. whatever is left: structural parents are context; exact rows with no request
    #    structure (a requested guidance document) group by document; the rest is related
    docs: dict[str, list[int]] = {}
    for s in sources:
        if s.index in placed:
            continue
        if s.tier == TIER_STRUCTURAL:
            place(s.index, context)
        elif s.tier == TIER_EXACT or not requirements:
            docs.setdefault(s.citation, []).append(s.index)
            placed.add(s.index)
        else:
            place(s.index, related)
    for label, members in docs.items():
        heads.update({i: n == 0 for n, i in enumerate(members)})
        groups.append(EvidenceGroup(key=f"doc:{label}", label=label, heading=None, kind=GROUP_DOCUMENT,
                                    head=members[0], members=tuple(members),
                                    requirement_ids=tuple(dict.fromkeys(r for i in members for r in by_index[i].requirement_ids))))

    planned = []
    for s in sources:
        sec, toks = ident[s.index]
        if s.index in lead_in:
            role = ROLE_LEAD_IN
        elif s.index in context:
            role = ROLE_CONTEXT
        elif s.index in related:
            role = ROLE_RELATED
        else:
            role = _role(s, sec, toks, head=heads.get(s.index, False))
            if role == ROLE_LEAD_IN:  # only the section row of a section request is a lead-in
                role = ROLE_REQUIREMENT if heads.get(s.index, False) else ROLE_SUBORDINATE
        planned.append(PlannedSource(
            index=s.index, citation=s.citation, tier=s.tier, role=role, section=sec, paragraph=_path(toks),
            heading=extract_heading(s.text), qualifiers=extract_qualifiers(s.text),
            applicability=extract_applicability(s.text), cross_references=extract_cross_references(s.text),
            requirement_ids=tuple(s.requirement_ids),
        ))

    mode = shape.mode
    if shape.focus_terms:
        hits = [s.index for s in sources if any(t in (s.text or "").lower() for t in shape.focus_terms)]
        if not hits:
            notes.append(f"none of the evidence mentions {', '.join(shape.focus_terms)}; the answer covers the requested provision as a whole")
            mode = MODE_SYNTHESIS
    missing = set(by_index) - set(placed)
    if missing:  # cannot happen by construction; recorded rather than lost if it ever does
        related.extend(sorted(missing))
        notes.append(f"sources {sorted(missing)} had no place in the plan and were listed as related")
    return AnswerPlan(shape=shape, mode=mode, lead_in=tuple(lead_in), groups=tuple(groups), context=tuple(context),
                      related=tuple(related), sources=tuple(planned), notes=tuple(notes))


# ------------------------------------------------------------------ render
#
# Where the plan sits in the prompt matters more than how it is worded. Live
# runs (2026-09-24) showed the 0.6B model reproducing whatever list-like text it
# read last: a bulleted plan block after the evidence was copied verbatim
# (coverage 8/8 -> 2/8 on the 820.10 overview), and so was a list of items or
# conditions inside the ANSWER FORMAT line. So the per-question plan is rendered
# as indented lines under REQUESTED PROVISIONS, before the evidence, with
# sources named by their citation markers ([4]); the ANSWER FORMAT line stays
# short and list-free, like the pre-Stage-12 lines the model follows well.
# Nothing here states a length.


def _marks(indices: Iterable[int]) -> str:
    return "".join(f"[{i}]" for i in indices)


def _short(plan: AnswerPlan, g: EvidenceGroup) -> str:
    """"(b) Applicable regulatory requirements" inside one section, else the full label."""
    sections = {s.section for s in plan.sources if s.section}
    label = g.label
    if len(sections) == 1 and label.startswith("21 CFR ") and "(" in label and g.kind != GROUP_SIDE:
        label = label[label.index("("):]
    return f"{label} {g.heading}" if g.heading else label


def _conditions(plan: AnswerPlan, groups: Iterable[EvidenceGroup]) -> list[str]:
    out: list[str] = []
    for g in groups:
        for i in g.members:
            s = plan.source(i)
            if s is None:
                continue
            for q in s.qualifiers:
                if q in CONDITIONAL_QUALIFIERS:
                    out.append(f'"{q}" [{i}]')
        head = plan.source(g.head) if g.head is not None else None
        if head is not None and head.applicability:
            out.append(f"{_short(plan, g)} applies only to {head.applicability} [{g.head}]")
    return out


def _keep_sentence(plan: AnswerPlan, groups: Iterable[EvidenceGroup]) -> str:
    conditions = _conditions(plan, groups)
    rule = 'Never turn "if applicable" or "as appropriate" into an unconditional requirement.'
    if not conditions:
        return rule
    return "Keep these conditions with the requirements they limit: " + "; ".join(conditions) + ". " + rule


_SHORT_TEXT = 70


_OBLIGATION_CLAUSE_RE = re.compile(r"[^,.;:]*\b(?:must|shall)\b[^.;:]*", re.I)
_HANDLE_WORDS = 30


def _handle(plan: AnswerPlan, i: int, text: str) -> str:
    """A source-derived handle for one item, so a small model has something to
    say about it: its own text when short, else its obligation clause (the
    words around "must"/"shall", cut at a word boundary), plus any condition.
    Always source wording, never a paraphrase."""
    s = plan.source(i)
    body = re.sub(r"^\s*\([A-Za-z0-9]{1,4}\)\s*", "", text or "").strip().rstrip(";:,. ")
    body = re.sub(r"[;,]?\s*and$", "", body).strip()
    if s is None:
        return f"[{i}]"
    if body and len(body) <= _SHORT_TEXT:
        return f"{body} [{i}]"
    m = _OBLIGATION_CLAUSE_RE.search(body)
    clause = m.group(0).strip() if m else ""
    words = clause.split()
    if len(words) > _HANDLE_WORDS:
        clause = " ".join(words[:_HANDLE_WORDS]) + " ..."
    parts = [clause] if clause else [r for r in s.cross_references if r.lower().startswith("part")]
    parts += [q for q in s.qualifiers if q in CONDITIONAL_QUALIFIERS and q not in clause]
    return f"{', '.join(p for p in parts if p)} [{i}]"


_KEEP = ('Keep every condition such as "if applicable" or "as appropriate", and every limit on which devices or '
         'persons a requirement applies to, with the requirement it limits; never make a conditional requirement unconditional.')


def plan_lines(plan: AnswerPlan, texts: dict[int, str] | None = None, focus: bool = True) -> list[str]:
    """The plan as indented lines under REQUESTED PROVISIONS, near the top of the
    prompt. Lists live here, never in the final instruction: the 0.6B model
    reproduces a list that sits in the last line it reads."""
    texts = texts or {}
    mode, groups = plan.mode, plan.groups
    title = {MODE_SYNTHESIS: "synthesized overview (not exhaustive)", MODE_FOCUSED: "focused on the requested provision",
             MODE_ENUMERATION: "every item, listed separately (exhaustive)", MODE_COMPARISON: "comparison",
             MODE_DIRECT: "direct answer"}[mode]
    lines = [f"  answer shape: {title}"]
    if focus and plan.shape.focus_terms and mode == MODE_FOCUSED:
        lines.append(f"  focus: {', '.join(plan.shape.focus_terms)}")
    if plan.values:
        lines.append("  periods stated, each with the reports it applies to and when it runs from: "
                     + "; ".join(f"{v.condition}: {v.clause} [{v.index}]" for v in plan.values))
    if mode == MODE_SYNTHESIS:
        order = []
        for g in [g for g in groups if g.kind != GROUP_DOCUMENT] or list(groups):
            part = f"{_short(plan, g)} {_marks([g.head]) if g.head is not None else ''}".strip()
            if g.items:
                part += " with sub-requirements " + ", ".join(_handle(plan, i, texts.get(i, "")) for i in g.items)
            order.append(part)
        lines.append("  requirement groups, in order: " + "; ".join(order))
    elif mode == MODE_ENUMERATION:
        order = [f"{s.citation} [{i}]" for g in groups for i in g.members if (s := plan.source(i)) is not None]
        lines.append("  items, in order: " + "; ".join(order))
    elif mode == MODE_COMPARISON:
        lines.append("  sides: " + "; ".join(
            f"{g.label} {_marks([g.head]) if g.head is not None else ''}{(' with ' + _marks(g.items)) if g.items else ''}".strip()
            for g in groups))
    elif mode == MODE_FOCUSED:
        items = [i for g in groups for i in g.items]
        if items:
            lines.append("  its sub-requirements: " + ", ".join(_handle(plan, i, texts.get(i, "")) for i in items))
    conditions = _conditions(plan, groups)
    if conditions:
        lines.append("  conditions to keep: " + "; ".join(conditions))
    return lines


_FORMAT_BY_MODE = {
    MODE_SYNTHESIS: (
        "ANSWER FORMAT: a synthesized overview in plain sentences. Start with what the requested provision requires "
        "overall, then write one paragraph for each of its requirement groups, saying in that paragraph what each of its "
        "sub-requirements requires instead of giving each source a paragraph of its own. Do not leave a group out. End "
        "each statement with the numbers of the sources it uses, like [4][5]. " + _KEEP
    ),
    MODE_ENUMERATION: (
        "ANSWER FORMAT: for each item, in the order listed, write its citation label on its own line followed by plain "
        "sentences stating what it requires, taken from its own source and ending with the source number in brackets. "
        "Keep distinct items separate and include every one. " + _KEEP
    ),
    MODE_FOCUSED: (
        "ANSWER FORMAT: plain sentences stating what the requested provision requires, taken from its EXACT source and "
        "its sub-requirements, each ending with the source number in brackets. Do not describe other paragraphs of the "
        "section. " + _KEEP
    ),
    MODE_COMPARISON: (
        "ANSWER FORMAT: a comparison in plain sentences. First state what each side requires and whom it applies to, "
        "citing its own sources; then state how they differ and what, if anything, they share. Do not describe provisions "
        "that are not part of the comparison. " + _KEEP
    ),
    MODE_DIRECT: (
        "ANSWER FORMAT: begin by stating directly what was asked about, then add only the points from the sources needed "
        "for that statement to be accurate, in plain sentences, each ending with the numbers of the sources it uses. " + _KEEP
    ),
}


# RagConfig.answer_style="apply" (22 Oct 2026). The restating lines above ask for "plain sentences stating what the
# requested provision requires", and both Qwen3 models do exactly that: the provision is copied paragraph by paragraph
# and the question is never answered ("Is FD&C Green No. 3 permitted in the eye shadow?" - not said). Asked to answer
# first and apply the provisions to the facts, Qwen3-4B-Instruct-2507 states "permitted in the blush, not permitted in
# the eye shadow under 70.5(a), all batches certified" from the same evidence. Prose only, no list to copy.
APPLY_FORMAT = (
    "ANSWER FORMAT: First answer the QUESTION itself, applying the provisions to the facts it describes: for each thing "
    "it asks about, say whether it is permitted or not, required or not, or give the period, amount or place, and which "
    "provision decides it. Where a provision makes the outcome depend on a condition, a determination or an exception, "
    "say what it depends on instead of giving a flat yes or no. When the question asks about a limit, restriction or "
    "requirement that the provisions supplied do not impose, say that these provisions impose none; never say that FDA "
    "lacks an authority or that no regulation anywhere requires something. If the provisions settle only part of "
    "the question, answer that part and say which part they leave open, instead of declining. Then go through the "
    "requested provisions paragraph by paragraph and state, in one sentence each, every requirement they set that bears "
    "on the question, such as identity and specifications, uses and restrictions, labeling and certification, each with "
    "its own paragraph citation. Say each point once. End every sentence with its source number in brackets. Keep every "
    "condition and exception with the requirement it limits; never make a conditional requirement unconditional."
)


# DA1 (2 Nov 2026, off until measured): the first part of APPLY_FORMAT asks the model to answer the question; Qwen3-4B
# fills it with the provisions' requirements and then states them again paragraph by paragraph - 90 of 184 model
# answers of 29 Oct say their content twice (S-005, S-010, F-019), which is length the reader does not need and minutes
# of generation on this Mac. Removing the repeat afterwards lost 3 PASS (DD1, E-002's only closed-system definition), so
# the model is asked not to write it: the first part is the direct answer only, the requirements stay with their
# paragraphs. Nothing the format asks to be covered is dropped.
DIRECT_ANSWER_ONLY = (" Keep this first part to the direct answer, in at most three sentences: the decision, period, "
                      "amount or place and the provision that decides it, not the requirements themselves, which belong "
                      "to the paragraph-by-paragraph part that follows.")
_DIRECT_AFTER = "instead of declining."


def apply_format() -> str:
    """APPLY_FORMAT, with DA1's limit on its first part when that fix is on."""
    from regulus.rag.validator_switches import enabled

    if not enabled("DA1"):
        return APPLY_FORMAT
    head, sep, tail = APPLY_FORMAT.partition(_DIRECT_AFTER)
    return head + sep + DIRECT_ANSWER_ONLY + tail


def format_line(plan: AnswerPlan, profile: str | None = None, focus: bool = True) -> str:
    """The shape-specific ANSWER FORMAT line: short, list-free, no length.
    `profile` (answer-quality Stage 3) is a lead instruction for the question type,
    placed first; without it the line is exactly the Stage 12 line. `focus=False`
    drops the topic restriction of a focused question (generation.short_requested_provision)."""
    if plan.values:
        line = ("ANSWER FORMAT: the question asks for a time period, and the requested provision states more than one. In "
                "plain sentences, give each period listed under \"periods stated\" in the words given there, with the "
                "reports it applies to and when it runs from, each sentence ending with that period's own source number in "
                "brackets. Do not give a period for any other reports, and do not describe the rest of the provision. " + _KEEP)
    elif focus and plan.mode == MODE_FOCUSED and plan.shape.focus_terms:
        line = ("ANSWER FORMAT: plain sentences stating only what the requested provision says about the focus, each "
                "ending with the source number in brackets. Do not describe the rest of the provision. " + _KEEP)
    else:
        line = _FORMAT_BY_MODE[plan.mode]
    if profile:
        line = "ANSWER FORMAT: " + profile + " " + line[len("ANSWER FORMAT: "):]
    return line


# ---------------------------------------------------------------- answer profiles (Stage 3)
#
# One short lead instruction per question type (regulus.planning.intent), put in
# front of the answer-shape line. Prose only, no headings to copy and no lists:
# the 0.6B model reproduces lists and templates it reads last (see the note above
# plan_lines). Every profile ends with the same two habits the baseline lacked:
# no restating the question, and each source number at the end of the sentence
# it supports.

PROFILE_TAIL = "Do not restate the question. Put each source number at the end of the sentence it supports."
PROFILES = {
    "requirement": "Begin with one sentence that directly states what is required.",
    "definition": "Begin with a one-sentence definition taken from the sources, then add only what is needed to understand it here.",
    "explanation": "Begin with the main point in one sentence, then explain it from the sources.",
    "procedure": ("State what is required. Give numbered steps only where the sources describe steps, in the sources' order, "
                  "and add no step the sources do not give."),
    "applicability": "Begin by stating to whom or to what the requirement applies, as the sources define it.",
    "applicability_yes_no": ("Begin with Yes, No, or \"The sources do not say\", decided only by what the sources state. If the "
                             "sources do not mention something the question assumes, say so instead of agreeing with it."),
    "exception_condition": "Begin with the direct answer, then give each condition or exception exactly as the sources state it.",
    "comparison": "",
    "scenario": ("Begin with the answer for the situation described, then explain why from the sources, and say which "
                 "parts are your inference rather than what the sources state."),
    "citation_lookup": ("Begin with one sentence that names the provision answering the question and says what it provides, "
                        "ending with that provision's source number."),
}
WEAK_EVIDENCE = ("The sources below were found by a broad search and may not answer the question. If they do not, say that "
                 "the available regulatory sources do not provide enough information, and do not answer from general knowledge.")


def profile_for(question_type: str, *, yes_no: bool = False, weak_evidence: bool = False,
                out_of_scope_part: str | None = None) -> str:
    """The Stage 3 lead instruction for a question: its type's profile, plus the
    weak-evidence and mixed-scope caveats when they apply, plus the shared habits."""
    key = "applicability_yes_no" if (yes_no and question_type in ("applicability", "requirement", "exception_condition")) else question_type
    parts = [WEAK_EVIDENCE] if weak_evidence else []
    if out_of_scope_part:
        parts.append(f"Answer only the regulatory part of the question; the sources do not cover \"{out_of_scope_part}\".")
    parts.append(PROFILES.get(key, PROFILES["explanation"]))
    parts.append(PROFILE_TAIL)
    return " ".join(p for p in parts if p)
