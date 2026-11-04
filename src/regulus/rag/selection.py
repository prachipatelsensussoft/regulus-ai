"""Evidence selection for the generation context (Stage 3).

Turns an EvidenceRegistry into the ordered, bounded set of sources the
model will see. The selector is requirement-aware and works in phases:

    1. required evidence   exact rows that ARE a requested provision (and the
                           chunks of a requested docket): protected, kept in
                           requirement order, full text up to the required
                           per-source cap, bounded only by the hard ceilings
    1b. material evidence  Stage 12, only when the caller passes `material`:
                           evidence it marks as a material part of a requested
                           provision (its child paragraphs) is kept like
                           required evidence - bounded by the hard ceilings,
                           not the soft budget - but is not protected and
                           keeps its tier
    2. structural evidence parent/child paragraphs of requested provisions
    3. supporting evidence the question's full-text hits
    4. discovery evidence  broad-method hits for unresolved requirements

Phases 2 to 4 are optional evidence: they spend whatever the soft budget
(ContextPolicy.max_results / max_chars_per_result / max_total_chars) leaves
after the required evidence, and are dropped before any required source.
Required evidence may exceed the soft budget up to the hard ceilings
(hard_max_results / hard_max_total_chars); a required source that does not
fit even there is dropped with an explicit `overflow` record, never
silently. Every decision is recorded in the selection diagnostics.

No scores are combined across tiers: a score only orders optional evidence
inside its tier, which the registry already did.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from regulus.rag.context import ORDER_BINDING_FIRST, ContextPolicy, ContextSource, TRUNCATION_MARK, render_source
from regulus.rag.evidence_values import evidence_has_value, selection_value_kind
from regulus.retrieval.evidence import TIER_DISCOVERY, TIER_EXACT, TIER_STRUCTURAL, TIER_SUPPORTING, Evidence, EvidenceRegistry
from regulus.retrieval.models import METHOD_EXACT_DOCKET, RetrievalResult
from regulus.retrieval.query import QueryType, fallback_terms

REASON_REQUIRED_EXACT = "required_exact"          # tier exact, satisfies a requirement
REASON_REQUIRED_DOCKET = "required_docket"        # the requested guidance document
REASON_STRUCTURAL = "structural_context"
REASON_SUPPORTING = "supporting"
REASON_DISCOVERY = "discovery"
REASON_SUBORDINATE = "subordinate_of_requested"  # Stage 12: a child of a requested provision

HX1_MAX_RESULTS = 48   # HX1: required paragraphs that may be supplied, under the unchanged character ceiling
CB1_MAX_TOTAL_CHARS = int(__import__("os").environ.get("REGULUS_CB1_MAX_TOTAL_CHARS", "15000"))   # CB1: hard ceiling for the GPU model
DROP_HARD_CEILING_CHARS = "hard_ceiling_chars"    # required evidence exceeded hard_max_total_chars
DROP_HARD_CEILING_COUNT = "hard_ceiling_count"    # required evidence exceeded hard_max_results
DROP_SOFT_BUDGET = "soft_budget"                  # optional evidence: max_total_chars exhausted
DROP_MAX_RESULTS = "max_results"                  # optional evidence: max_results reached
DROP_FILTERED = "filtered"                        # optional evidence excluded by the pipeline's usefulness rule

MIN_SQUEEZED_CHARS = 200  # an optional source is shortened to fit only if at least this much text remains

_PREFIX = 6  # a question term matches a passage on its first this-many characters ("reporting" ~ "reports")

# A paragraph pointing at other paragraphs: "under paragraph (c)(1)(i), (c)(1)(ii), (c)(1)(iii),
# or (c)(1)(iv) of this section", "except as provided in paragraph (d)". Used to keep what a
# paragraph refers to with it; see `_referenced_paths`.
_REF_CUE_RE = re.compile(r"\bparagraphs?\b", re.I)
_REF_RUN_RE = re.compile(r"(?:\(\s*[A-Za-z0-9]{1,4}\s*\)|[\s,;]|\band\b|\bor\b|\bthrough\b|\bto\b)+", re.I)
_REF_SPLIT_RE = re.compile(r"[,;]|\band\b|\bor\b|\bthrough\b|\bto\b", re.I)
_REF_TOKEN_RE = re.compile(r"\(\s*([A-Za-z0-9]{1,4})\s*\)")


@dataclass(frozen=True)
class SelectedEvidence:
    index: int                      # 1-based SOURCE number in the prompt
    evidence: Evidence
    text: str                       # cleaned, possibly truncated text placed in the context
    truncated: bool
    protected: bool
    reason: str
    chars: int                      # rendered block length charged to the budget

    @property
    def requirement_ids(self) -> tuple[str, ...]:
        e = self.evidence
        return tuple(dict.fromkeys(e.satisfies + e.structural_for + e.discovered_for))

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "evidence_id": self.evidence.evidence_id,
            "key": self.evidence.key,
            "citation": self.evidence.citation,
            "tier": self.evidence.tier,
            "requirement_ids": list(self.requirement_ids),
            "protected": self.protected,
            "reason": self.reason,
            "chars": len(self.text),
            "block_chars": self.chars,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class DroppedEvidence:
    evidence: Evidence
    reason: str
    required: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence.evidence_id,
            "citation": self.evidence.citation,
            "tier": self.evidence.tier,
            "requirement_ids": list(dict.fromkeys(self.evidence.satisfies + self.evidence.structural_for + self.evidence.discovered_for)),
            "reason": self.reason,
            "required": self.required,
        }


@dataclass(frozen=True)
class Selection:
    selected: tuple[SelectedEvidence, ...]
    dropped: tuple[DroppedEvidence, ...]
    policy: ContextPolicy
    total_chars: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def protected(self) -> tuple[SelectedEvidence, ...]:
        return tuple(s for s in self.selected if s.protected)

    @property
    def overflow(self) -> bool:
        """True when required evidence had to be dropped or truncated."""
        return bool(self.diagnostics.get("required_dropped")) or bool(self.diagnostics.get("required_truncated"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [s.to_dict() for s in self.selected],
            "dropped": [d.to_dict() for d in self.dropped],
            "policy": self.policy.to_dict(),
            "total_chars": self.total_chars,
            "overflow": self.overflow,
            **self.diagnostics,
        }


def is_required(evidence: Evidence) -> bool:
    """Exact evidence for an explicit requirement, or the requested docket's chunks."""
    if evidence.tier != TIER_EXACT:
        return False
    return bool(evidence.satisfies) or evidence.retrieval_method == METHOD_EXACT_DOCKET


def _reason(evidence: Evidence) -> str:
    if evidence.tier == TIER_EXACT:
        return REASON_REQUIRED_EXACT if evidence.satisfies else REASON_REQUIRED_DOCKET
    return {TIER_STRUCTURAL: REASON_STRUCTURAL, TIER_SUPPORTING: REASON_SUPPORTING}.get(evidence.tier, REASON_DISCOVERY)


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    cut = max(limit - len(TRUNCATION_MARK), 0)
    return text[:cut].rstrip() + TRUNCATION_MARK, True


def _block_chars(evidence: Evidence, text: str, index: int, truncated: bool) -> int:
    """Length of the rendered SOURCE block, what the budget is charged with."""
    probe = ContextSource(index=index, result=evidence.result, text=text, truncated=truncated)
    block = render_source(probe)
    from regulus.rag.validator_switches import enabled as _switch

    if _switch("BA1"):
        # BA1 (28 Oct): the grounded prompt (generation.render_source) renders no URL line, but every block was charged
        # for one (~95 characters of "URL: https://www.ecfr.gov/current/title-21/..." a paragraph) - with 24-40
        # paragraphs, 2,000-4,000 characters of the ceiling spent on text the model never sees; chosen companions were
        # dropped as hard_ceiling_chars (retrieval check 28 Oct: F-002 117.3, Z-005 1040.11)
        block = "\n".join(line for line in block.split("\n") if not line.startswith("URL: "))
    if _switch("BU1"):
        # BU1 (4 Nov): the prompt's block also carries a "Tier:" line (generation.render_source) that the budget never
        # charged - 40 to 110 characters a source, so 48 paragraphs could put the SOURCES block 2,000-5,000 characters
        # past the ceiling that was meant to bound it
        from regulus.rag.generation import tier_line
        from regulus.retrieval.models import METHOD_EXACT_DOCKET

        rids = tuple(dict.fromkeys(evidence.satisfies + evidence.structural_for + evidence.discovered_for))
        block += "\n" + tier_line(evidence.tier, rids, evidence.corpus, evidence.retrieval_method == METHOD_EXACT_DOCKET)
    return len(block)


def _match_keys(question: str) -> tuple[str, ...]:
    """Prefix keys for the question's salient terms.

    The terms are the retrieval layer's own (`regulus.retrieval.query.fallback_terms`), so
    selection and retrieval agree on what a question is about. A hyphenated term is split into
    its parts ("7-calendar-day" -> "calendar", "day"); a bare number or a section number is
    dropped, because it discriminates nothing inside the section it names.
    """
    keys: list[str] = []
    for term in fallback_terms(question or ""):
        for part in term.strip('"').split("-"):
            part = part.strip(".()").lower()
            if len(part) < 3 or part.replace(".", "").isdigit():
                continue
            key = part[:_PREFIX]
            if key not in keys:
                keys.append(key)
    return tuple(keys)


def _referenced_paths(text: str) -> frozenset[str]:
    """The paragraph paths this paragraph points at, as "(c)(1)(i)" strings.

    Only the explicit "paragraph(s) ..." form, and only the paths themselves: a reference to
    another SECTION carries its own section number and is not one of these. Deliberately literal -
    it is used to keep a cross-referenced list together, not to resolve citations.
    """
    found: set[str] = set()
    for cue in _REF_CUE_RE.finditer(text or ""):
        run = _REF_RUN_RE.match(text, cue.end())
        if run is None:
            continue
        for piece in _REF_SPLIT_RE.split(run.group(0)):
            tokens = _REF_TOKEN_RE.findall(piece)
            if tokens:
                found.add("".join(f"({t})" for t in tokens))
    return frozenset(found)


def _relevance(text: str, keys: tuple[str, ...]) -> int:
    """How many of the question's terms this passage contains."""
    low = (text or "").lower()
    return sum(1 for k in keys if k in low)


def _paragraph_of(evidence: Evidence) -> tuple[str, str] | None:
    """(section, paragraph path) of an eCFR row, from the row itself.

    The retriever's own metadata, not the citation label: `_identity` parses the label with a
    lowercase-only paragraph pattern and so cannot read "(c)(1)(i)(A)".
    """
    md = evidence.result.metadata
    section = md.get("section")
    if evidence.corpus != "ecfr" or not section:
        return None
    return (str(section), str(md.get("paragraph_path") or ""))


def _one_section(required: list[Evidence]) -> bool:
    """True when every required row is a paragraph of the SAME eCFR section.

    This is the only case `_required_order` may reorder. The registry hands required evidence
    over in requirement order and then document order, and for a single requested section that
    document order is all there is - nothing in it reflects the question. For a requested PART it
    is different: the rows are section stems the retriever has already ranked by relevance to the
    question (C1), and a second, cruder ranking here would override a better one. Measured over
    the 300-question benchmark, reordering part-level evidence dropped the very sections the
    question named (21 CFR 111.105 for "what must quality control personnel do", 111.210 for
    "what must the master manufacturing record include"), so it is left alone.
    """
    ids = [_paragraph_of(e) for e in required]
    return all(i is not None for i in ids) and len({i[0] for i in ids}) == 1


def _required_order(required: list[Evidence], question: str | None) -> list[Evidence]:
    """The order in which required evidence competes for the hard ceiling.

    Document order spends the ceiling on whatever comes first, which for a whole requested
    section means the paragraphs the question did not ask about: answering "the 7-calendar-day
    deadline in 21 CFR 312.32" spent 4,416 of 9,000 characters on the (a) definitions and then
    dropped (c)(2), the 298-character paragraph that states that deadline. Ordering the
    competition by relevance keeps the paragraphs the question is about.

    Shared wording alone is not enough. A question can name the KIND of value it wants without
    using any of the regulation's words: asked for the IND safety reporting "timeframes" of 21 CFR
    312.32, term overlap scored (c)(2) - the 7-calendar-day rule, and the answer - at one word and
    the (c)(1) family at three, because (c)(2) says "notify FDA" rather than "IND safety report".
    So a paragraph that STATES a value of the kind asked for is preferred over one that merely
    shares words, and term overlap then orders each group. Only four of this section's 25
    paragraphs state a time period, and (c)(2) is one of them.

    A lead-in inherits both signals from its own subparagraphs, so a parent is never ranked below
    a child and a kept fragment always arrives with the text that introduces it. Document order
    breaks every tie, so with no question, no salient terms, no value asked for, no overlap, or
    evidence that is not one section's paragraphs, the order - and therefore the selected set - is
    exactly what it was before.
    """
    keys = _match_keys(question or "")
    kind = selection_value_kind(question or "")
    if (not keys and kind is None) or len(required) < 2:
        return required
    if not _one_section(required):
        return _per_section_order(required, question)
    own = [_relevance(e.text, keys) for e in required]
    states = [1 if kind and evidence_has_value(kind, [e.text]) else 0 for e in required]
    paths = [_paragraph_of(e)[1] for e in required]  # one section: _one_section checked it
    best, has = list(own), list(states)
    from regulus.rag.validator_switches import enabled as _switch

    rows = [(paths[k], _ordinal(e)) for k, e in enumerate(required)]
    if _switch("PP1") and len(set(paths)) < len(paths) and all(o is not None for _, o in rows):
        # PP1 (4 Nov): where paths repeat (a definitions section, DEF1: each definition's lead at "" and its items at
        # "(1)", "(2)" ...), a row inherits from its OWN subparagraphs - after it, before the next row of its path - so
        # "Holding means:" travels with its items, and Farm's "(1)" does not inherit from Holding's "(1)(i)"
        children = [_own_children(i, rows) for i in range(len(rows))]
    else:
        children = [[j for j, other in enumerate(paths) if j != i and path and other.startswith(path + "(")]
                    for i, path in enumerate(paths)]   # j is a subparagraph of i
    for i, kids in enumerate(children):
        for j in kids:
            best[i] = max(best[i], own[j])
            has[i] = max(has[i], states[j])
    # What a paragraph POINTS AT belongs with it: the 15-calendar-day clock of 21 CFR
    # 312.32(c)(1) runs from the sponsor's determination "under paragraph (c)(1)(i), (c)(1)(ii),
    # (c)(1)(iii), or (c)(1)(iv)", and ranking those four against each other on incidental word
    # overlap kept (ii) and (iii) and dropped (i) and (iv) - half of one list, which invites
    # exactly the misattribution this ranking exists to prevent. A referenced paragraph therefore
    # inherits the referring paragraph's term score, so a cross-referenced list is kept or dropped
    # as a unit, and an exception ("except as provided in paragraph (d)") comes with its rule.
    #
    # Only when a value kind was asked for, because that is the only case this repairs. The value
    # flag ranks ahead of term overlap and so can separate members of one list; plain term overlap
    # breaks its ties by document order, which already keeps a list together. Measured: with the
    # propagation unconditional, "What follow-up does 21 CFR 312.32 require?" - which asks for no
    # value - dropped (d)(1), "the sponsor must promptly investigate all safety information it
    # receives", to make room for the (c)(1) reporting conditions. Worse evidence for that
    # question, so the propagation does not apply to it.
    #
    # The value flag itself is NOT propagated: a condition does not state the value asked for, and
    # spreading the flag would flatten the very signal that finds the answer-bearing paragraph.
    if kind is not None:
        referenced = [_referenced_paths(e.text) for e in required]
        inherited = list(best)
        for i, refs in enumerate(referenced):
            for j, path in enumerate(paths):
                if j != i and path and path in refs:
                    inherited[j] = max(inherited[j], best[i])
        best = inherited
    if not any(best) and not any(has):
        return required
    return [e for _, _, _, e in sorted(((-has[i], -best[i], i, e) for i, e in enumerate(required)))]


def _per_section_order(required: list[Evidence], question: str | None) -> list[Evidence]:
    """Several named sections - a promoted section with its companions ("21 CFR 701.3" with 701.30): each section's
    paragraphs compete in relevance order (as one section's would), and the sections keep the order they came in.

    Nothing is ranked ACROSS sections, so part-level evidence (one row per section, already ranked by the retriever,
    see _one_section) is unchanged. Measured 22 Oct 2026: on the 352-question retrieval baseline every record is
    identical; on the 277-question cosmetics bank 13 prompts change, one gains its key paragraphs (701.3(m) and (r),
    which document order left behind 701.3(a)-(c)(3)) and none loses one. A section whose rows arrive in more than
    one run is left as it is."""
    ids = [_paragraph_of(e) for e in required]
    if any(i is None for i in ids):
        return required
    groups: list[tuple[str, list[Evidence]]] = []
    for e, i in zip(required, ids):
        if groups and groups[-1][0] == i[0]:
            groups[-1][1].append(e)
        else:
            groups.append((i[0], [e]))
    if len({g[0] for g in groups}) != len(groups):
        return required
    ordered = [(_required_order(g, question) if len(g) > 1 else g) for _, g in groups]
    from regulus.rag.validator_switches import enabled as _switch

    base = _sections_order(ordered, _switch)
    if _switch("CF1") and len(ordered) > 1:
        # CF1: each companion section's best paragraph is guaranteed, after the first section's best one
        floor = [o[0] for o in ordered[1:] if len(_clean(o[0].text)) <= CF1_MAX_PARAGRAPH_CHARS]
        head = [ordered[0][0]]
        taken = {e.evidence_id for e in head + floor}
        return head + floor + [e for e in base if e.evidence_id not in taken]
    return base


# CF1 (2 Nov 2026, off until measured): companion floor. RR2 lets the first section take every paragraph before its
# companions share what is left; a long first section then leaves them nothing. F-002: 117.5 took all 35 slots and the
# companions 117.3 (what a qualified facility is) and 117.201 reached the model with no paragraph - scored "no 117.3
# definition or 117.201 content"; A-004: 558.6 took 48 and Part 225 (medicated feed CGMP) none. Each companion keeps
# its one most relevant paragraph - the diversity floor of MMR-style packing - and the first section fills the rest as
# before. A companion whose best paragraph is longer than CF1_MAX_PARAGRAPH_CHARS gets no floor (it would crowd out).
CF1_MAX_PARAGRAPH_CHARS = 1500


def _sections_order(ordered: list[list[Evidence]], _switch) -> list[Evidence]:
    """The sections' paragraphs in competition order under RR2 / RR1 (or the sections in turn)."""
    if _switch("RR2") and len(ordered) > 2:
        # RR2 (28 Oct): RR1 halved S-018's 111.210 list by sharing the budget with its companions (full 300, 15:50) -
        # the first (requested or promoted) section keeps all its paragraphs first; only the sections after it share
        # what is left round-robin (M-029: 820.10 behind 820.3, 820.35, 820.45, 820.1, 820.7 got its lead-in only)
        rest: list[Evidence] = []
        for rank in range(max(len(o) for o in ordered[1:])):
            rest += [o[rank] for o in ordered[1:] if rank < len(o)]
        return ordered[0] + rest
    if _switch("RR1") and len(ordered) > 1:
        # RR1 (28 Oct): round-robin across the requested sections - each section's best paragraph, then each one's
        # second, and so on - so no section is left with its lead-in only when the ceilings bind (diversity-aware
        # allocation, as MMR). 300Q M-029: 820.3, 820.35, 820.45, 820.1 and 820.7 took the budget in arrival order and
        # 820.10, whose (b)-(e) are the key, got its one-line lead-in. A section's own paragraphs keep their order.
        out: list[Evidence] = []
        for rank in range(max(len(o) for o in ordered)):
            out += [o[rank] for o in ordered if rank < len(o)]
        return out
    return [e for o in ordered for e in o]


# OF1 (2 Nov 2026, off until measured): outline first. When the requested section does not fit, its top-level
# paragraphs compete ahead of the rest, so every topic the section covers reaches the model before any topic's detail.
# 101.36 (82 paragraphs, 31,000 characters) kept 28: deep items such as (b)(2)(i)(A) and the (c) proprietary-blend
# lead-in, but not (d), the source ingredient and plant-part rule (L-013, PARTIALLY FAIL: "silent on proprietary blends
# and part-of-plant"). 29 Oct 300Q: a key section was cut by the ceiling in 23 non-PASS answers and 3 PASS.
# Only for a question that asks for no particular value (a deadline or amount keeps the relevance order built for it,
# 312.32(c)(2)), and only when the section's top level is an outline - at most OF1_SHARE of the ceiling (a definitions
# section, whose top level is all of it, is left as it is).
OF1_SHARE = 1 / 3
_TOP_LEVEL = re.compile(r"^\([^()]+\)$")


def outline_first(order: list[Evidence], question: str | None, ceiling: int) -> list[Evidence]:
    """OF1: the first section's top-level paragraphs moved ahead, in their relative order; otherwise `order`."""
    if not order or selection_value_kind(question or "") is not None:
        return order
    ids = [_paragraph_of(e) for e in order]
    if ids[0] is None:
        return order
    cost = {e.evidence_id: _block_chars(e, _clean(e.text), 1, False) for e in order}
    if sum(cost.values()) + 2 * (len(order) - 1) <= ceiling:
        return order          # everything fits: the order changes nothing
    section = ids[0][0]
    top = [e for e, i in zip(order, ids) if i is not None and i[0] == section and _TOP_LEVEL.match(i[1] or "")]
    if len(top) < 2 or sum(cost[e.evidence_id] + 2 for e in top) > OF1_SHARE * ceiling:
        return order
    lead = {e.evidence_id for e in top}
    return top + [e for e in order if e.evidence_id not in lead]


def select_evidence(
    registry: EvidenceRegistry,
    policy: ContextPolicy | None = None,
    *,
    optional_filter: Callable[[RetrievalResult], bool] | None = None,
    material: Callable[[Evidence], bool] | None = None,
    question: str | None = None,
) -> Selection:
    """Deterministic, requirement-aware selection. Same inputs, same output.

    `question`, when given, decides only which required evidence is kept if it does not all fit
    inside the hard ceilings (`_required_order`); the selected sources are always presented in
    document order. Without it, or when nothing has to be dropped, the result is unchanged."""
    policy = policy or ContextPolicy()
    selected: list[SelectedEvidence] = []
    dropped: list[DroppedEvidence] = []
    used = 0
    required_truncated: list[str] = []

    def gap() -> int:
        return 2 if selected else 0  # blank line between blocks

    # Phase 1: required evidence. It competes for the hard ceiling in `_required_order` - document
    # order, or relevance to the question when one is given - and is then presented in document
    # order. A block's length depends on its index only through its "SOURCE n" line, so the kept
    # set occupies the same total either way and the accounting below is order-independent.
    from regulus.rag.validator_switches import enabled as _switch

    if _switch("HX1") and policy.hard_max_results < HX1_MAX_RESULTS:
        # HX1 (27 Oct): the character ceiling is the budget; 24 paragraphs cut 111.260's batch-record list in half (S-020)
        from dataclasses import replace as _replace

        policy = _replace(policy, hard_max_results=HX1_MAX_RESULTS)
    if _switch("CB1") and policy.hard_max_total_chars < CB1_MAX_TOTAL_CHARS:
        # CB1 (28 Oct): the 9,000-character hard ceiling was set "to keep the prompt bounded on this CPU" (0.6B on CPU);
        # Qwen3-4B runs on the GPU. Retrieval check 28 Oct: F-020's MS1 companion 117.405 was chosen and dropped as
        # hard_ceiling_chars behind 117.410's 22 paragraphs, each block carrying ~220 characters of header (SOURCE, Corpus,
        # Citation, Heading, URL) - half the budget was headers
        from dataclasses import replace as _replace

        policy = _replace(policy, hard_max_total_chars=CB1_MAX_TOTAL_CHARS)
    registry_order = [e for e in registry.evidence if is_required(e)]
    position = {e.evidence_id: i for i, e in enumerate(registry_order)}
    competition = _required_order(registry_order, question)
    if _switch("OF1"):
        competition = outline_first(competition, question, policy.hard_max_total_chars)
    keep: list[tuple[Evidence, str, bool]] = []
    for e in competition:
        text, truncated = _truncate(_clean(e.text), policy.required_max_chars_per_result)
        chars = _block_chars(e, text, len(keep) + 1, truncated)
        if len(keep) >= policy.hard_max_results:
            dropped.append(DroppedEvidence(e, DROP_HARD_CEILING_COUNT, required=True))
            continue
        if used + (2 if keep else 0) + chars > policy.hard_max_total_chars:
            dropped.append(DroppedEvidence(e, DROP_HARD_CEILING_CHARS, required=True))
            continue
        used += (2 if keep else 0) + chars
        keep.append((e, text, truncated))
    for e, text, truncated in sorted(keep, key=lambda k: position[k[0].evidence_id]):
        if truncated:
            required_truncated.append(e.evidence_id)
        index = len(selected) + 1
        selected.append(SelectedEvidence(index, e, text, truncated, True, _reason(e),
                                         _block_chars(e, text, index, truncated)))
    used = sum(s.chars for s in selected) + 2 * max(len(selected) - 1, 0)
    required_by_relevance = [e.evidence_id for e in competition] != [e.evidence_id for e in registry_order]

    required_count = len(selected)
    required_chars = used

    # Phase 1b: material parts of requested provisions, same ceilings as required evidence
    material_ids: list[str] = []
    if material is not None:
        for e in registry.evidence:
            if is_required(e) or not material(e):
                continue
            text, truncated = _truncate(_clean(e.text), policy.required_max_chars_per_result)
            index = len(selected) + 1
            chars = _block_chars(e, text, index, truncated)
            if len(selected) >= policy.hard_max_results:
                dropped.append(DroppedEvidence(e, DROP_HARD_CEILING_COUNT, required=False))
                material_ids.append(e.evidence_id)
                continue
            if used + gap() + chars > policy.hard_max_total_chars:
                dropped.append(DroppedEvidence(e, DROP_HARD_CEILING_CHARS, required=False))
                material_ids.append(e.evidence_id)
                continue
            used += gap() + chars
            selected.append(SelectedEvidence(index, e, text, truncated, False, REASON_SUBORDINATE, chars))
            material_ids.append(e.evidence_id)
    handled = set(material_ids)

    # Phase 1c (RC2): binding text first claim on part of the soft budget, for a question that names
    # nothing (see ContextPolicy.binding_reserve_chars). Same per-source cap, filter and limits as
    # below; the sources are put back in registry order afterwards, so only membership changes.
    # With general_evidence_order="binding_first" every eCFR passage already competes first, so the
    # reserve would only pre-empt that order and is not applied.
    first_optional = len(selected)
    reserved_ids: list[str] = []
    reserve = min(policy.binding_reserve_chars, policy.max_total_chars)
    binding_first = policy.general_evidence_order == ORDER_BINDING_FIRST and _binding_reserve_applies(registry)
    if reserve > 0 and _binding_reserve_applies(registry) and not binding_first:
        limit = min(policy.max_total_chars, policy.hard_max_total_chars)
        spent = 0
        for tier in (TIER_STRUCTURAL, TIER_SUPPORTING, TIER_DISCOVERY):
            for e in registry.evidence:
                if e.tier != tier or is_required(e) or e.evidence_id in handled or e.result.corpus != "ecfr":
                    continue
                if tier != TIER_STRUCTURAL and optional_filter is not None and not optional_filter(e.result):
                    continue  # recorded as filtered by the loop below
                if len(selected) >= policy.max_results or len(selected) >= policy.hard_max_results:
                    continue
                text, truncated = _truncate(_clean(e.text), policy.max_chars_per_result)
                index = len(selected) + 1
                chars = _block_chars(e, text, index, truncated)
                if spent + chars > reserve or used + gap() + chars > limit:
                    continue
                spent += chars
                used += gap() + chars
                selected.append(SelectedEvidence(index, e, text, truncated, False, _reason(e), chars))
                reserved_ids.append(e.evidence_id)
                handled.add(e.evidence_id)

    # Phases 2-4: optional evidence within the soft budget, never displacing the above
    for e in _optional_order(registry, binding_first):
        if is_required(e) or e.evidence_id in handled:
            continue
        # structural evidence is identity-based (a parent or child of a requested
        # provision); the usefulness rule targets method-based hits only
        if e.tier != TIER_STRUCTURAL and optional_filter is not None and not optional_filter(e.result):
            dropped.append(DroppedEvidence(e, DROP_FILTERED, required=False))
            continue
        if len(selected) >= policy.max_results or len(selected) >= policy.hard_max_results:
            dropped.append(DroppedEvidence(e, DROP_MAX_RESULTS, required=False))
            continue
        text, truncated = _truncate(_clean(e.text), policy.max_chars_per_result)
        index = len(selected) + 1
        chars = _block_chars(e, text, index, truncated)
        limit = min(policy.max_total_chars, policy.hard_max_total_chars)
        if used + gap() + chars > limit:
            remaining = limit - used - gap()
            header = chars - len(text)
            if remaining - header >= MIN_SQUEEZED_CHARS:
                text, truncated = _truncate(text, remaining - header)
                chars = _block_chars(e, text, index, True)
                truncated = True
            else:
                dropped.append(DroppedEvidence(e, DROP_SOFT_BUDGET, required=False))
                continue
        used += gap() + chars
        selected.append(SelectedEvidence(index, e, text, truncated, False, _reason(e), chars))

    if reserved_ids or binding_first:
        # registry order, as without the reserve; a block's length depends on its index only through
        # its "SOURCE n" line and the indices stay 1..n, so the total charged is unchanged
        position = {e.evidence_id: i for i, e in enumerate(registry.evidence)}
        optional = sorted(selected[first_optional:], key=lambda s: position[s.evidence.evidence_id])
        selected = selected[:first_optional] + [
            replace(s, index=first_optional + i + 1, chars=_block_chars(s.evidence, s.text, first_optional + i + 1, s.truncated))
            for i, s in enumerate(optional)
        ]

    # exact evidence that is not required (docket handled above) cannot exist; any other tier is optional
    diagnostics = {
        "required_count": required_count,
        "required_chars": required_chars,
        "optional_count": len(selected) - required_count,
        **({"material_selected": [s.evidence.evidence_id for s in selected if s.reason == REASON_SUBORDINATE]} if material is not None else {}),
        "required_dropped": [d.evidence.evidence_id for d in dropped if d.required],
        "required_truncated": required_truncated,
        "optional_dropped": [d.evidence.evidence_id for d in dropped if not d.required],
        "soft_budget_exceeded_by_required": required_chars > policy.max_total_chars or required_count > policy.max_results,
        **({"required_order": "relevance"} if required_by_relevance else {}),
        **({"binding_reserved": reserved_ids} if reserved_ids else {}),
        **({"general_evidence_order": ORDER_BINDING_FIRST} if binding_first else {}),
        "tier_counts": {t: sum(1 for s in selected if s.evidence.tier == t) for t in (TIER_EXACT, TIER_STRUCTURAL, TIER_SUPPORTING, TIER_DISCOVERY)},
    }
    return Selection(tuple(selected), tuple(dropped), policy, used, diagnostics)


_SENTENCE_END = (".", '."', ".)", "?", "!")


def _optional_order(registry: EvidenceRegistry, binding_first: bool) -> list[Evidence]:
    """The order in which optional evidence competes for the soft budget.

    Default: tier by tier (structural, supporting, discovery), registry order within a tier - so every
    passage that matched all the question's words, a 0.03-scored guidance chunk included, precedes every
    OR / trigram / section-coverage passage, and the section-coverage rows the registry puts last stay last.

    binding_first (ContextPolicy.general_evidence_order, a question that names nothing): every eCFR passage
    before every guidance passage; within the eCFR, structural and supporting in registry order, then the
    discovery passages by their own score, highest first (ties keep registry order). Discovery scores share
    one 0-5 band (full_text_or, trigram, section coverage), so this ranks a section that covers the question's
    weight (D-016's 211.125 at 5.0) ahead of a heading that merely resembles it. Guidance keeps its own order
    after the eCFR. Nothing is added or removed here: the same candidates, filter and limits."""
    tiers = (TIER_STRUCTURAL, TIER_SUPPORTING, TIER_DISCOVERY)
    if not binding_first:
        return [e for tier in tiers for e in registry.evidence if e.tier == tier]
    ecfr = [e for e in registry.evidence if e.result.corpus == "ecfr"]
    other = [e for e in registry.evidence if e.result.corpus != "ecfr"]
    return ([e for e in ecfr if e.tier == TIER_STRUCTURAL] + [e for e in ecfr if e.tier == TIER_SUPPORTING]
            + sorted((e for e in ecfr if e.tier == TIER_DISCOVERY), key=lambda e: -e.score)
            + [e for tier in tiers for e in other if e.tier == tier])


def _binding_reserve_applies(registry: EvidenceRegistry) -> bool:
    """RC2 applies to a question that names no provision and no docket and is not about guidance:
    every source it gets is optional, and the eCFR is the binding text. A question that names a
    provision keeps exactly the selection it had; a guidance question keeps guidance first."""
    plan = getattr(registry, "plan", None)
    return (plan is not None and not plan.requirements and plan.docket is None
            and plan.query_type is not QueryType.GUIDANCE_SEARCH)


def is_complete_sentence(text: str) -> bool:
    """True when a provision's text ends a sentence. A lead-in ("shall be in terms of:") or a chained
    fragment ("...; or, in the absence thereof,") needs the text around it to be read correctly."""
    return (text or "").rstrip().endswith(_SENTENCE_END)


def _ordinal(evidence) -> int | None:
    """The row's place in its section (cfr_records.ordinal), or None when the result does not carry it."""
    o = (getattr(evidence.result, "metadata", None) or {}).get("ordinal")
    return int(o) if o is not None else None


def _parent_path(path: str) -> str | None:
    """"(b)(1)" -> "(b)"; "(1)" -> "" (the section's or the definition's lead); "" -> None."""
    return path[:path.rfind("(")] if path else None


def _own_children(i: int, rows: list[tuple[str, int]]) -> list[int]:
    """PP1: the rows that are row i's subparagraphs by position, not only by path - after it, under its path, and
    before the next row that has its path (a repeated path: several definitions' items are all "(1)", "(2)" ...).
    `rows`: (paragraph path, ordinal) of one section's rows."""
    path, at = rows[i]
    later_same = [o for p, o in rows if p == path and o > at]
    stop = min(later_same) if later_same else None
    return [j for j, (p, o) in enumerate(rows) if j != i and o > at and (stop is None or o < stop)
            and (p.startswith(path + "(") if path else p != "")]


def _true_parent(i: int, rows: list[tuple[str, int]]) -> int | None:
    """PP1: row i's parent - the nearest row BEFORE it whose path is its parent path ("Holding means:" for the "(1)"
    that follows it, never Farm's lead three definitions earlier)."""
    path, at = rows[i]
    parent = _parent_path(path)
    if parent is None:
        return None
    before = [(o, j) for j, (p, o) in enumerate(rows) if p == parent and o < at]
    return max(before)[1] if before else None


def _identity(evidence) -> tuple[str, str] | None:
    """(section, paragraph path) of an eCFR row, from its citation label ("21 CFR 830.20(b)(1)")."""
    from regulus.rag.validator_switches import enabled as _switch

    # PP1: labels of any case and up to 8 characters - "(A)", "(xviii)"; without it such a row has no identity and
    # consolidate_evidence may merge its text with another provision's identical text
    label = r"\([A-Za-z0-9]{1,8}\)" if _switch("PP1") else r"\([a-z0-9]{1,4}\)"
    m = re.match(rf"^\d+ CFR (\d{{1,4}}\.\d{{1,4}}[a-z]?)((?:{label})*)$", evidence.citation or "")
    if m:
        return (m.group(1), m.group(2))
    # an FD&C Act passage ("FD&C Act § 201(v)(1) [21 U.S.C. 321(v)(1)]") is a provision too: never merged as a near-duplicate
    m = re.match(r"^FD&C Act § (\d+[A-Za-z]*(?:-\d+)?)((?:\([A-Za-z0-9]{1,6}\))*) \[", evidence.citation or "")
    return ("FDCA " + m.group(1), m.group(2)) if m else None


def completed_by_protected_children(item, selection: Selection) -> bool:
    """A requested provision whose text is not a complete sentence reads correctly when the chain it belongs
    to is present: its own subparagraphs (a lead-in, "A unique device identifier (UDI) must:") or its parent
    lead-in (a list item, "(a) Be issued under ...;") among the protected evidence. Related material is then
    not needed (Stage 8: 21 CFR 830.20 kept an off-target guidance chunk, and the model looped on it). A
    fragment whose chain is NOT present (S5-032, S5-101: requested list items without their lead-in) keeps
    the full context, as in Stage 6."""
    own = _identity(item.evidence)
    if own is None:
        return False
    section, path = own
    from regulus.rag.validator_switches import enabled as _switch

    if _switch("PP1"):
        # PP1: the chain is the item's own - its parent and children by position among every retrieved row of the
        # section (selected and dropped), so a "(1)" is completed by its own definition's lead, not by any lead
        pool = [s.evidence for s in selection.selected] + [d.evidence for d in selection.dropped]
        same = [(e, _identity(e), _ordinal(e)) for e in pool]
        same = [(e, idt[1], o) for e, idt, o in same if idt is not None and idt[0] == section and o is not None]
        if _ordinal(item.evidence) is not None and any(e is item.evidence for e, _, _ in same):
            rows = [(p, o) for _, p, o in same]
            me = next(k for k, (e, _, _) in enumerate(same) if e is item.evidence)
            protected = {id(s.evidence) for s in selection.selected if s.protected and s is not item}
            chain = _own_children(me, rows) + ([p] if (p := _true_parent(me, rows)) is not None else [])
            return any(id(same[k][0]) in protected for k in chain)
    for o in selection.selected:
        if o is item or not o.protected or (oid := _identity(o.evidence)) is None or oid[0] != section:
            continue
        if oid[1].startswith(path + "(") or (path.startswith(oid[1] + "(") or (path and not oid[1])):
            return True
    return False


def required_only(selection: Selection) -> Selection:
    """Only the required (protected) evidence, re-indexed 1..n.

    Stage 6 (docs/generation_quality_stage6.md): used when every requested provision has
    exact evidence that is complete text on its own (see RagPipeline.build_bundle for the
    conditions): related supporting/discovery sources and unrequested parents or children
    are left out, because the 0.6B model is distracted by related material it does not
    need. Also the "perfect evidence" controlled condition of the evaluations."""
    kept = [s for s in selection.selected if s.protected]
    items = tuple(SelectedEvidence(i, s.evidence, s.text, s.truncated, True, s.reason, s.chars) for i, s in enumerate(kept, start=1))
    dropped = selection.dropped + tuple(DroppedEvidence(s.evidence, "optional_not_needed_exact_evidence_complete", False)
                                        for s in selection.selected if not s.protected)
    return Selection(items, dropped, selection.policy, sum(s.chars for s in items),
                     {**selection.diagnostics, "required_only": True, "optional_count": 0})


# ---------------------------------------------------------------- evidence consolidation (Stage 8)

_RESERVED_RE = re.compile(r"^\W*(?:\([a-z0-9]{1,4}\)\s*)*\[?\s*reserved\s*\]?\W*$", re.I)


def _normalised(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _trigrams(text: str) -> set[tuple[str, ...]]:
    w = _normalised(text).split()
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


NEAR_DUPLICATE_MIN = 0.9  # share of the shorter chunk's word 3-grams found in the other


def consolidate_evidence(selection: Selection) -> tuple[Selection, dict[str, Any]]:
    """One evidence item per distinct passage, before the prompt is built (Stage 8).

    * a content-free "[Reserved]" row is dropped, unless it is the only exact evidence of a requested
      provision (the answer must then be able to say the provision is reserved);
    * identical text (after normalising case, spacing and punctuation) is kept once WHEN it is the same
      provision, or is not a CFR provision (guidance chunks); identical wording in two different provisions
      (21 CFR 117.180(d) and 507.53(d)) stays: they are different rules;
    * near-duplicate guidance chunks (>= 90 % of the shorter chunk's 3-grams in the other) are kept once;
      CFR provisions are never merged by similarity (21 CFR 111.155(a), 111.160(a), 111.165(a) share
      wording but are different provisions).
    The first occurrence in selection order (the strongest) is kept; every removal is recorded.
    """
    kept: list[SelectedEvidence] = []
    removed: list[dict[str, Any]] = []
    sole_exact: dict[str, int] = {}
    for s in selection.selected:
        for rid in s.evidence.satisfies:
            sole_exact[rid] = sole_exact.get(rid, 0) + 1
    for s in selection.selected:
        ident = _identity(s.evidence)
        norm = _normalised(s.text)
        if _RESERVED_RE.match(s.text or "") and not any(sole_exact.get(rid) == 1 for rid in s.evidence.satisfies):
            removed.append({"key": s.evidence.key, "citation": s.evidence.citation, "reason": "reserved (no content)"})
            continue
        duplicate_of = None
        for k in kept:
            k_ident = _identity(k.evidence)
            if _normalised(k.text) == norm and (ident is None or k_ident is None or ident == k_ident):
                duplicate_of = k
                break
            if ident is None and k_ident is None and s.evidence.corpus == k.evidence.corpus != "ecfr":
                a, b = _trigrams(s.text), _trigrams(k.text)
                if a and b and len(a & b) / min(len(a), len(b)) >= NEAR_DUPLICATE_MIN:
                    duplicate_of = k
                    break
        if duplicate_of is not None:
            removed.append({"key": s.evidence.key, "citation": s.evidence.citation, "reason": "duplicate", "kept": duplicate_of.evidence.key})
            continue
        kept.append(s)
    if not removed:
        return selection, {"removed": []}
    items = tuple(SelectedEvidence(i, s.evidence, s.text, s.truncated, s.protected, s.reason, s.chars) for i, s in enumerate(kept, start=1))
    dropped = selection.dropped + tuple(DroppedEvidence(next(s.evidence for s in selection.selected if s.evidence.key == r["key"]),
                                                         "consolidated_" + ("reserved" if r["reason"].startswith("reserved") else "duplicate"), False)
                                        for r in removed)
    return (Selection(items, dropped, selection.policy, sum(s.chars for s in items), {**selection.diagnostics, "consolidated": removed}),
            {"removed": removed})
