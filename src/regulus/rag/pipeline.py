"""RagPipeline: question -> Retriever -> context -> Generator -> answer + citations.

Citations never come from the model. They are the authoritative Citation
objects of the retrieval results that were placed in the context; the model
may refer to them as [n]. Sources with the same citation label (several
chunks of one guidance document) are reported once with all their indices.
"""

from __future__ import annotations

import inspect
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from regulus.inference.types import GenerationResult
from regulus.planning import QueryPlan, plan_query
from regulus.planning.plan import KIND_GENERAL
from regulus.planning.ranges import expand_section_ranges
from regulus.retrieval.domain import GENERAL_KNOWLEDGE
from regulus.planning.intent import SCOPE_MIXED, SCOPE_OUT, TYPE_CITATION, QuestionAnalysis, analyze_question
from regulus.planning.shape import MODE_COMPARISON, MODE_ENUMERATION, MODE_FOCUSED, AnswerShape, analyze_shape
from regulus.rag import answerability as gate
from regulus.rag.citation_repair import repoint_citations
from regulus.rag.conversation import turn_anchor
from regulus.rag.answer_quality import mixed_scope_note, normalize_answer, out_of_scope_answer, quality_flags, strip_plan_labels
from regulus.rag.context import ContextBundle, ContextPolicy, build_context, render_selection
from regulus.rag.citations import ROLE_CONTEXT, RagCitation, build_answer_citations, citation_support
from regulus.rag.coverage import STATE_INCOMPLETE, CoverageReport, check_coverage
from regulus.rag.generation import (
    APPLY_RETRY_SYSTEM_PROMPT,
    UNVALIDATED_GENERAL,
    UNVALIDATED_PREFIX,
    GenerationBudget,
    GenerationPackage,
    build_grounded_messages,
    build_package,
    definitions_not_applied,
    extractive_answer,
    extractive_items,
    verbatim_requested,
)
from regulus.rag.fallback import (
    FB_COMPOSED_LIST,
    FB_COMPOSED_LISTING,
    FB_DEFINITIONS_QUOTED,
    FB_COMPOSED_DEFINITIONS,
    FB_COMPOSED_PREMISE,
    FB_COMPOSED_QMSR,
    FB_COMPOSED_RECORDS,
    FB_FOCUSED_QUOTE,
    FB_PRODUCT_CLARIFICATION,
    FB_PREMISE_NOT_ESTABLISHED,
    FB_PROVISION_NOT_FOUND,
    FB_COMPOSED_SILENCE,
    FB_UNVALIDATED,
    FOCUSED_QUOTE_LEAD,
    FOCUSED_QUOTE_LEAD_PICKED,
    blocked_answer,
    focused_quote,
    premise_not_established,
    provision_is_identified,
    quoting_allowed,
    unvalidated_answer,
)
from regulus.rag.selection import (
    Selection,
    completed_by_protected_children,
    consolidate_evidence,
    is_complete_sentence,
    required_only,
    select_evidence,
)
from regulus.rag.shape_validation import ENUMERATION_INCOMPLETE, MATERIAL_GROUP_UNCOVERED
from regulus.inference.repetition_text import drop_duplicate_sentences, repeated_sentence
from regulus.rag.claim_validation import INVENTED_CONDITION, PARAGRAPH_MISATTRIBUTION, UNSUPPORTED_PREMISE
from regulus.rag.validation import (
    CITATION_MISMATCH,
    INVALID_CITATION,
    PROVISION_CITATION_MISMATCH,
    REFUSAL_WITH_EVIDENCE,
    STATUS_CITATION_INVALID,
    STATUS_DEGENERATE,
    STATUS_EMPTY,
    STATUS_EVIDENCE_MISMATCHED,
    STATUS_UNSUPPORTED,
    ValidationReport,
    attach_trailing_markers,
    excise_unsupported_sentences,
    repoint_named_citations,
    strip_invalid_citations,
    validate_answer,
)
from regulus.rag.validator_switches import enabled as precision_fix
if TYPE_CHECKING:  # annotation only; importing it would load torch
    from regulus.rag.generator import Generator
from regulus.rag.prompt import (NO_CONTEXT_ANSWER, build_messages, coverage_note, incomplete_coverage_answer,
                                unresolved_provisions_statement, withheld_evidence_statement)
from regulus.retrieval.base import Filters
from regulus.retrieval.evidence import REL_CHILD, TIER_EXACT, TIER_STRUCTURAL, Evidence, EvidenceRegistry
from regulus.retrieval.lexical import LexicalRetriever, SearchResponse
from regulus.retrieval.models import METHOD_TRIGRAM, RetrievalResult
from regulus.traceability import evidence as evidence_trace

_CITE_RE = re.compile(r"\[(\d{1,2})\]")
# Stage 5: final validation states that mean the shown text is not grounded (a repair that failed)
_UNGROUNDED = (STATUS_EMPTY, STATUS_DEGENERATE, STATUS_UNSUPPORTED, STATUS_EVIDENCE_MISMATCHED)
# Stage 5: blocked answers keep the coverage_action earlier stages recorded for the same situation
_BLOCK_ACTIONS = {"out_of_scope": "refuse_out_of_scope", "provision_not_found": "refuse_unsupported", "no_evidence": None,
                  "product_clarification": "ask_product_type"}


@dataclass(frozen=True)
class RagConfig:
    top_k: int = 8
    corpus: str | None = None
    # Trigram-only hits at the pg_trgm floor (similarity 0.3 -> score 1.5) are
    # mostly headings sharing a common word. Provisional cut-off, not tuned on
    # the benchmark: documented in docs/lexical_rag.md.
    min_trigram_similarity: float = 0.4
    # "refuse": deterministic no-context answer without calling the model (default,
    # conservative). "model": send the empty-sources prompt to the model.
    no_context_behavior: str = "refuse"
    # Stage 9: a retriever may mark results as contextual_only when the query
    # named an identifier (CFR section/part, docket) that has no live record.
    # Such passages are related material, not evidence the identifier exists;
    # by default they are kept out of the grounding context so the answer
    # falls back to the deterministic not-found response.
    exclude_contextual_only: bool = True
    context: ContextPolicy = field(default_factory=ContextPolicy)
    # Stage 3: what to do when explicit requirements lack exact evidence.
    # "partial": answer from the supported provisions only and state which were
    # not found; when none is supported, answer deterministically without the
    # model. "model": send whatever was selected to the model (legacy).
    incomplete_behavior: str = "partial"
    # Stage 4: per-question token budget and what to do with a generated answer
    # that fails deterministic validation. "repair": replace or complete it from
    # the exact sources; "annotate": keep it and report; "off": skip validation.
    generation_budget: GenerationBudget = field(default_factory=GenerationBudget)
    validation_behavior: str = "repair"
    # Stage 12: "adaptive" derives an AnswerShape from the question and gives
    # the model the evidence organised for it (docs/adaptive_answer_shape.md);
    # "off" restores the previous prompt and selection, for measurement.
    # "structured" (answer-quality Stage 3, docs/answer_quality.md) adds question
    # analysis, answer profiles, deterministic scope handling, answer
    # normalisation and quality flags on top of "adaptive".
    answer_shape: str = "adaptive"
    # Stage 5 (docs/answerability_gate_stage5.md): "gate" decides answerability before
    # generation, blocks unanswerable questions without calling the model and uses the
    # safe fallback builder; "off" restores the Stage 4 behaviour, for measurement.
    answerability: str = "gate"
    # Stage 6 (docs/generation_quality_stage6.md): "incomplete_only" gives the model only the
    # protected evidence (exact + structural) when every requested provision has exact evidence,
    # and keeps optional supporting/discovery sources only when something requested is missing
    # or no provision was requested; "always" restores the Stage 5 context, for measurement.
    optional_context: str = "incomplete_only"
    # Stage 7 (docs/retrieval_validation_stage7.md): "general_knowledge" declines a question that names no
    # provision, uses no word of the database's own regulatory vocabulary and matches no passage
    # (regulus.retrieval.domain); "off" disables it. Acts only when the pipeline is given a ScopePolicy.
    scope_policy: str = "general_knowledge"
    # Stage 7: "on" = the retriever built by regulus.retrieval.factory adds semantic search as a fallback for
    # insufficient lexical evidence (regulus.retrieval.fallback); "off" = lexical only.
    semantic_fallback: str = "on"
    # Stage 8 (docs/generation_control_stage8.md): "on" = drop content-free [Reserved] rows and duplicate passages
    # before the prompt (regulus.rag.selection.consolidate_evidence); "off" = previous behaviour.
    evidence_consolidation: str = "on"
    # 22 Oct 2026 (regulus.rag.section_promotion): a question that names no provision is answered as if it named the
    # section that answers it. "picker" = the answering model chooses among the top-ranked sections; "lexical" = the
    # top-ranked section, no model call; "off" = previous behaviour (the library default; production turns it on).
    section_promotion: str = "off"
    # 22 Oct 2026: "apply" = the ANSWER FORMAT line asks for the answer to the question first, applied to its facts,
    # then the supporting provision wording (regulus.rag.answer_plan.APPLY_FORMAT); "restate" = previous lines.
    answer_style: str = "restate"

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.no_context_behavior not in ("refuse", "model"):
            raise ValueError("no_context_behavior must be 'refuse' or 'model'")
        if self.incomplete_behavior not in ("partial", "model"):
            raise ValueError("incomplete_behavior must be 'partial' or 'model'")
        if self.validation_behavior not in ("repair", "annotate", "off"):
            raise ValueError("validation_behavior must be 'repair', 'annotate' or 'off'")
        if self.answer_shape not in ("adaptive", "structured", "off"):
            raise ValueError("answer_shape must be 'adaptive', 'structured' or 'off'")
        if self.answerability not in ("gate", "off"):
            raise ValueError("answerability must be 'gate' or 'off'")
        if self.optional_context not in ("incomplete_only", "always"):
            raise ValueError("optional_context must be 'incomplete_only' or 'always'")
        if self.scope_policy not in ("general_knowledge", "off"):
            raise ValueError("scope_policy must be 'general_knowledge' or 'off'")
        if self.evidence_consolidation not in ("on", "off"):
            raise ValueError("evidence_consolidation must be 'on' or 'off'")
        if self.semantic_fallback not in ("on", "off"):
            raise ValueError("semantic_fallback must be 'on' or 'off'")
        if self.section_promotion not in ("off", "lexical", "picker"):
            raise ValueError("section_promotion must be 'off', 'lexical' or 'picker'")
        if self.answer_style not in ("restate", "apply"):
            raise ValueError("answer_style must be 'restate' or 'apply'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "retriever": "LexicalRetriever",
            "top_k": self.top_k,
            "corpus": self.corpus or "all",
            "min_trigram_similarity": self.min_trigram_similarity,
            "no_context_behavior": self.no_context_behavior,
            "exclude_contextual_only": self.exclude_contextual_only,
            "incomplete_behavior": self.incomplete_behavior,
            "generation_budget": self.generation_budget.to_dict(),
            "validation_behavior": self.validation_behavior,
            "answer_shape": self.answer_shape,
            "answerability": self.answerability,
            "optional_context": self.optional_context,
            "scope_policy": self.scope_policy,
            "semantic_fallback": self.semantic_fallback,
            "evidence_consolidation": self.evidence_consolidation,
            "context": self.context.to_dict(),
            # recorded only when on, so the fingerprints of earlier configurations are unchanged
            **({"section_promotion": self.section_promotion} if self.section_promotion != "off" else {}),
            **({"answer_style": self.answer_style} if self.answer_style != "restate" else {}),
        }


# DV1: the small parts its paragraph promotion is measured on (REGULUS_DV1_PARTS, comma-separated)
DV1_PARTS = frozenset(p.strip() for p in os.environ.get("REGULUS_DV1_PARTS", "820,11").split(",") if p.strip())
# AR2: what a draft says the provisions do not specify ("The retention period ... is not specified in the available
# sources", "The provisions do not specify renewal requirements")
_ABSENCE = [re.compile(r"\b(?:provisions?|sources?|regulations?|text|section|part)\s+(?:do|does)\s+not\s+(?:specify|state|provide|"
                       r"address|establish|include)\s+(?P<t>[^.;:\n]{3,90})", re.I),
            re.compile(r"(?P<t>[A-Za-z][^.;:\n]{3,90}?)\s+(?:is|are)\s+not\s+(?:specified|stated|addressed|provided|established)\b", re.I)]


def absence_topics(text: str, limit: int = 2) -> list[str]:
    """AR2: the topics a draft says the supplied provisions do not specify, at most `limit`."""
    out: list[str] = []
    for rx in _ABSENCE:
        for m in rx.finditer(text or ""):
            topic = re.sub(r"^(?:the|any|a|an)\s+|\s+(?:in|within|by) the (?:available|supplied|provided|retrieved)\b.*$", "",
                           m.group("t").strip(), flags=re.I)
            if len(topic.split()) >= 2 and topic.lower() not in (x.lower() for x in out):
                out.append(topic)
    return out[:limit]


@dataclass
class RagAnswer:
    question: str
    answer: str
    citations: list[RagCitation]
    retrieved_results: list[RetrievalResult]
    context: ContextBundle
    no_context: bool
    model_called: bool
    timings_ms: dict[str, float]
    input_tokens: int | None
    output_tokens: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: EvidenceRegistry | None = None  # Stage 2 registry (None on the legacy path)
    coverage: CoverageReport | None = None    # Stage 3 coverage (None on the legacy path)
    selection: Selection | None = None        # Stage 3 selection (None on the legacy path)
    package: GenerationPackage | None = None  # Stage 4 generation contract (None on the legacy path)
    validation: ValidationReport | None = None  # Stage 4 final validation (None when not generated/validated)
    # The exact evidence placed in the prompt (regulus.traceability). Kept out of
    # to_dict()/diagnostics so existing artifacts keep their shape; None means the
    # evidence was not captured, never that it was empty.
    evidence_snapshot: evidence_trace.EvidenceSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved": [
                {"rank": i, "corpus": r.corpus, "citation": r.citation.label, "url": r.source_url, "score": round(r.relevance_score, 4), "method": r.retrieval_method, "key": r.key}
                for i, r in enumerate(self.retrieved_results, start=1)
            ],
            "context": self.context.to_dict(),
            "no_context": self.no_context,
            "model_called": self.model_called,
            "timings_ms": self.timings_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": self.metadata,
        }

    def diagnostics(self) -> dict[str, Any]:
        """Compact per-question record for the benchmark (retrieval vs generation)."""
        d = self.to_dict()
        out = {k: d[k] for k in ("retrieved", "context", "citations", "no_context", "model_called", "timings_ms")}
        out["retrieval"] = dict(self.metadata.get("retrieval", {}))
        out["grounding"] = self.metadata.get("grounding")
        # Stage 3 keys are added only when present so legacy runs keep their shape
        for key in ("coverage", "coverage_action", "generation", "validation", "citation_support", "answer_shape",
                    "question_analysis", "quality_flags"):
            if self.metadata.get(key) is not None:
                out[key] = self.metadata[key]
        if self.package is not None:  # Stage 5 gate record (the legacy path keeps its shape)
            for key in ("answerability", "generation_called", "generation_blocked", "block_reason", "fallback_type", "fallback_reason"):
                out[key] = self.metadata.get(key)
        return out


@dataclass
class RagGenerationResult(GenerationResult):
    """GenerationResult plus RAG diagnostics, so the existing evaluation runner
    can store retrieval information without changing its interface."""

    diagnostics: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)


def _statute_notes(promotion) -> tuple[str, ...]:
    """The prompt's NOTE when only part of a resolved FD&C Act section fits the evidence (the general search's notes
    are about evidence the statute replaced)."""
    if not promotion.statute_left_out:
        return ()
    return (f"Only part of FD&C Act § {', '.join(promotion.statute)} is supplied below ({len(promotion.statute_left_out)} "
            "subsection(s) left out); do not state what the left-out text requires.",)


def may_promote(question: str, plan: QueryPlan) -> str | None:
    """Whether section promotion may run for this question: None (no), "named_statute" (only an FD&C Act section the
    question itself names may be resolved) or "any". Never when a deterministic refusal decided from the question
    itself would otherwise apply (code review, 22 Oct): a named non-U.S. framework or a mixed-scope question, a
    provision pointed at without being identified ("paragraph (b)", "this section"), a request for FDA guidance or a
    docket, another CFR title ("16 CFR", "40 CFR"). A plan with requirements is promoted only when every requirement
    is a whole 21 CFR part of the color-additive and cosmetic programme (Parts 70-82, 700-740) and the question is not
    a citation lookup: "Under Part 82, what specifications ..." is narrowed to the section; "What does Part 11 require?"
    is not."""
    from regulus.rag.answerability import _AMBIGUOUS_RE
    from regulus.retrieval.section_index import COSMETIC_PARTS

    if plan.docket is not None or plan.guidance_requested or _OTHER_CFR_TITLE.search(question):
        return None
    analysis = analyze_question(plan)
    if analysis.scope_status in (SCOPE_OUT, SCOPE_MIXED):
        return None
    statute_named = bool(_FDCA_CUE.search(question) or re.search(r"§\s*\d{3}(?!\d|\.\d)", question))
    if plan.requirements:
        if analysis.question_type == TYPE_CITATION or not all(r.reference.section is None and r.reference.title == "21"
                                                               for r in plan.requirements):
            return None
        if all(r.reference.part in COSMETIC_PARTS for r in plan.requirements):
            return "any"
        from regulus.rag.section_promotion import part_only_any

        if part_only_any() and not any((r.raw or "").lstrip().startswith("§") for r in plan.requirements):
            return "any"   # PO2: a part outside the cosmetics programme, resolved inside that part
        # "FD&C Act § 607" / "21 U.S.C. § 364a" are read as CFR parts: only the statute check may run
        return "named_statute" if statute_named and all((r.raw or "").lstrip().startswith("§") for r in plan.requirements) else None
    if _AMBIGUOUS_RE.search(question):
        # "what does subsection (b) of section 605 require": identified only if the question names the section itself
        return "named_statute" if statute_named else None
    return "any"


# a follow-up's own words "point to" an anchored section when it is among their first this many candidate sections
FOLLOWUP_RANK_TOP = int(os.environ.get("REGULUS_FOLLOWUP_RANK_TOP", "3"))
# section resolutions whose result is certain enough for an answer composed from it (Q1, FP1, RS1): the question's
# conversation, an exact color-additive name, an FD&C Act section the question names
_CONFIDENT_RESOLUTION = frozenset({"conversation", "entity", "named_statute", "statute_reference"})
# Q2: the reranker score a picker-chosen section needs before its paragraphs may stand in for a withheld answer. 26 Oct, 25
# recorded withheld answers quoted: every regression and 9 of 11 wrong-section quotes scored below 0.6, 6 of 11 gains above
Q2_MIN_RERANK = 0.6
# completeness findings that do not apply to a composed answer about one point of a provision (FP1, RS1)
# DV1: a question asking WHICH provisions of a part do something (M-029, "Which provisions of Part 820 are FDA
# supplements ...") spans the part; narrowed to its three best paragraphs it was declined
_ASKS_WHICH_PROVISIONS = re.compile(r"\b(?:which|what)\s+(?:provisions|sections|paragraphs|requirements)\s+of\b", re.I)
# DV2: a question about an FDA guidance document itself (NC-022: "Summarize FDA's final guidance 'QMSR Certification Body
# Accreditation Program'" - a document that does not exist) is not answered from the regulation's paragraphs
# DV2: a question about other parts and how they relate to the named one (M-042: "Which parts of Title 21 govern device
# establishment registration, listing, and unique device identification, and how do they interact with Part 820?" - the
# answer is Parts 807 and 830, never in Part 820's paragraphs)
_ASKS_OTHER_PARTS = re.compile(r"\b(?:which|what)\s+(?:other\s+)?parts\s+(?:of\s+(?:title\s+21|21\s+CFR|the\s+(?:CFR|regulations?))|"
                               r"(?:govern|apply|cover|regulate)\b)|\binteract\w*\s+with\s+(?:21\s+CFR\s+)?part\s+\d", re.I)
_GUIDANCE_DOCUMENT = re.compile(r"\b(?:final|draft)\s+guidance\b|\bguidance\s+(?:document\s+)?['\"‘“]"
                                r"|\bsummari[sz]e\b[^?.]{0,40}\bguidance\b", re.I)
_FOCUSED_COMPLETENESS_CODES = frozenset({"requirement_not_addressed", "material_group_uncovered", "enumeration_incomplete"})
_OTHER_CFR_TITLE = re.compile(r"\b(?!21\b)\d{1,2}\s+C\.?F\.?R\b", re.I)
_FDCA_CUE = re.compile(r"\bFD&C\s+Act\b|\bFDCA\b|\bFederal\s+Food,\s+Drug,?\s+and\s+Cosmetic\s+Act\b|\b21\s+U\.?S\.?C\b", re.I)


_NOTE_BY_CODE = {
    "unsupported_figure": "{detail}. Give only periods, amounts and numbers that the EVIDENCE states, in its words; do not add or combine them.",
    "unsupported_claim": "This sentence is not supported by the EVIDENCE: \"{sentence}\". Remove it, or state only what a source says.",
    "requirement_not_addressed": "{detail}. Answer the question from that provision.",
    "refusal_with_evidence": "The draft declined, but the EVIDENCE contains the provision the question is about. Answer from it.",
    "invented_condition": "{detail}: \"{sentence}\". Keep only conditions a source states.",
    "condition_not_in_cited_source": "{detail}: \"{sentence}\". Cite the source that states the condition.",
    "provision_citation_mismatch": "{detail}: \"{sentence}\". Cite the source whose Citation is the provision named.",
    "citation_mismatch": "{detail}. Cite the EXACT source of the provision named.",
    "paragraph_misattribution": "{detail}. Attribute each statement to the paragraph it comes from.",
    "invalid_citation": "{detail}. Use only the source numbers shown in the EVIDENCE.",
    "enumeration_incomplete": "{detail}. Include it, briefly, where it bears on the question.",
    "truncated": "The draft ran too long and was cut off. Answer concisely: each point once.",
    "degenerate_output": "The draft repeated itself ({detail}). Say each point once.",
    "unsupported_premise": "{detail}. Do not affirm what no source states.",
    "obligation_strengthened": "The draft {detail}: \"{sentence}\". State it with the source's own word, not as a requirement.",
}


def _quote_sources(package: GenerationPackage | None) -> list[str] | None:
    """The supplied source texts duplicate removal must not cut a verbatim quotation of (fix G1), or None."""
    return [s.text for s in package.sources] if package is not None and precision_fix("G1") else None


def correction_notes(report: ValidationReport, package: GenerationPackage, limit: int = 6) -> list[str]:
    """The validator's error-level findings, worded for the model (its own detail text, never a new claim)."""
    out: list[str] = []
    for p in report.problems:
        if p.severity != "error":
            continue
        tmpl = _NOTE_BY_CODE.get(p.code, "{detail}.")
        note = tmpl.format(detail=(p.detail or "").rstrip("."), sentence=" ".join((p.sentence or "").split())[:220])
        if note not in out:
            out.append(note)
    return out[:limit]


def _COSMETIC_PARTS():
    from regulus.retrieval.section_index import COSMETIC_PARTS

    return COSMETIC_PARTS


def _without_part_requirements(plan: QueryPlan, keep_none: bool = False) -> QueryPlan:
    """The plan without its part-level requirements (renumbered), when sections were resolved inside those parts.
    `keep_none`: the result may have no requirements left (the part was a misread statute section)."""
    from regulus.planning.plan import Requirement, _plan_kind

    kept = [r for r in plan.requirements if r.reference.section is not None]
    if len(kept) == len(plan.requirements) or (not kept and not keep_none):
        return plan
    reqs = tuple(Requirement(id=f"r{i}", order=i, reference=r.reference, raw=r.raw, origin=r.origin, mention=r.mention)
                 for i, r in enumerate(kept, start=1))
    return replace(plan, requirements=reqs, kind=_plan_kind(plan.query_type, reqs, plan.scan))


# the model's whole answer is the system prompt's decline sentence (rule 9), once or repeated (see _BARE_REFUSAL_RE)
REFUSAL_RETRY_NOTE = ("\n\nNOTE: The EVIDENCE above was retrieved for this question and contains the provision it is about. "
                      "Answer from it as the ANSWER FORMAT says. Where the provision does not settle a point, say so for "
                      "that point only.")
_BARE_REFUSAL_RE = re.compile(r"^(?:\s*The available regulatory sources do not provide enough information to answer "
                              r"this(?: question)?\.\s*)+$")
# CV1: a cited sentence its sources support with probability below this is checked again; FR1: facts-first prompts
CV1_MIN_SUPPORT = float(os.environ.get("REGULUS_CV1_MIN_SUPPORT", "0.2"))
_CV1_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"“])|\n+")
FR1_MAX_TOKENS = 400
FR1_PROMPT = ("\n\nBEFORE ANSWERING: list the facts the SOURCES state that bear on the QUESTION, one per line, each in the "
              "source's own words and ending with its source number in brackets. Include the conditions each fact depends on. "
              "Do not answer the question yet and do not add anything the SOURCES do not say.")
FR1_ANSWER = "\n\nNow answer the QUESTION from these facts and the SOURCES, following the ANSWER FORMAT."
# BN2: the verifier's score of a validated draft - the requested provision's paragraphs it cites (completeness, as the
# rubric scores it) minus its denials ("the regulation does not address ...", 70.45-Q1 said so of its own subject)
_DENIAL = re.compile(r"\b(?:does|do)\s+not\s+(?:address|cover|mention|establish|apply\s+to)\b|\bnot\s+addressed\b|"
                     r"\bno\s+provision\b|\bdoes\s+not\s+(?:specify|state)\b", re.I)


def answer_score(text: str, package) -> float:
    requested = {s.index for s in package.sources if getattr(s, "requirement_ids", None)}
    cited = {int(m) for m in re.findall(r"\[(\d{1,3})\]", text or "")}
    denials = sum(1 for sent in re.split(r"(?<=[.!?])\s+", text or "") if _DENIAL.search(sent))
    return 0.5 * len(cited & requested) - 1.0 * denials


PC1_MIN_SCORE = 0.5   # PC1: an uncited paragraph this relevant to the question is quoted after the answer
PC1_MAX = 3
PC1_LEAD = "Other paragraphs of {section} that bear on the question, verbatim:"
TK1_MIN_TOKENS = 640   # TK1: the least room an applied answer gets
# AL1: the answer allowance counts up to this many required units and may reach this many tokens (was 10 units / 2,048:
# set when the prompt was capped near 2,600 tokens; CB1/HX1 now supply up to 48 paragraphs and 15,000 characters)
AL1_MAX_UNITS = int(os.environ.get("REGULUS_AL1_MAX_UNITS", "18"))
AL1_MAX_TOKENS = int(os.environ.get("REGULUS_AL1_MAX_TOKENS", "3072"))
# CX1: a draft with at least this many different error codes skips the correction pass (300Q full300c: 31 of 32 such
# passes were rejected - a rejected pass leaves the answer exactly as a skipped one - and the one accepted stayed FAIL)
CX1_MAX_CODES = int(os.environ.get("REGULUS_CX1_MAX_CODES", "3"))
# LM1: name the closest sources before the evidence when there are at least this many; at most LM1_TOP of them
LM1_MIN_SOURCES = 6
LM1_TOP = 3
LM1_MIN_SCORE = 0.3
# OC1: the added sections that are context, not provisions the answer must restate - the reranker's (MS1/MS2), absence
# retrieval's (AR2), navigation's and cross-part citations. The picker's own second pick (SP2) stays a requested provision.
OC1_NOTES = ("reranked: ", "absence retrieval: ", "navigated: ", "cross-part: ", "paragraph companions: ")


def companions_as_context(package: GenerationPackage, plan: QueryPlan, labels: frozenset[str]) -> GenerationPackage:
    """OC1 (29 Oct): sections added as context stay in the evidence, as SUPPORTING sources, and leave the requested
    provisions and the answer plan. 300Q full300c: in all 122 answers with added sections they were requested units the
    model restated paragraph by paragraph - S-024's 8 sections in 40 paragraphs were cut at the token limit (PASS lost),
    S-006 restated 111.20 beside the 111.15 it was asked about. The token allowance is left as it is (a ceiling, never a
    target). Nothing changes when every requested provision is such a section."""
    from regulus.rag.answer_plan import build_answer_plan
    from regulus.retrieval.evidence import TIER_SUPPORTING

    drop = {rid for rid, canon in package.requirements if canon in labels}
    if not drop or len(drop) == len(package.requirements):
        return package
    kept_reqs = tuple(r for r in plan.requirements if r.id not in drop)
    sources = tuple(replace(s, tier=TIER_SUPPORTING, requirement_ids=(), protected=False)
                    if s.requirement_ids and set(s.requirement_ids) <= drop else
                    replace(s, requirement_ids=tuple(r for r in s.requirement_ids if r not in drop))
                    for s in package.sources)
    answer_plan = package.answer_plan
    if answer_plan is not None:
        rebuilt = build_answer_plan(answer_plan.shape, sources, kept_reqs)
        answer_plan = replace(rebuilt, values=answer_plan.values) if answer_plan.values else rebuilt
    return replace(package, sources=sources, answer_plan=answer_plan,
                   requirements=tuple(r for r in package.requirements if r[0] not in drop),
                   satisfied_ids=tuple(i for i in package.satisfied_ids if i not in drop),
                   context_ids=tuple(sorted(drop)))


def answer_budget(budget: GenerationBudget) -> GenerationBudget:
    """AL1 (29 Oct): the answer allowance keeps pace with the evidence. 300Q full300c: 29 answers stopped at the 1,696-token
    allowance (10 units x 160 + 96), none in the run before CB1/HX1; S-024 (8 sections, 40 paragraphs) lost its PASS cut
    mid-sentence. Decoding is greedy, so an answer that finished within the old allowance is unchanged."""
    if not precision_fix("AL1"):
        return budget
    return replace(budget, max_units=max(budget.max_units, AL1_MAX_UNITS),
                   hard_max_tokens=max(budget.hard_max_tokens, AL1_MAX_TOKENS))


def cosmetics_conversation(context) -> bool:
    """CT1: every provision the conversation's anchor names is in the color additive and cosmetics programme (21 CFR
    Parts 70-82 and 700-799, FD&C Act 601-614)."""
    a = getattr(context, "anchor", None)
    if not a:
        return False
    parts = [int(s.split(".")[0]) for s in a.cfr] + [int(p) for p in a.parts if p.isdigit()]
    return (all(70 <= p <= 82 or 700 <= p <= 799 for p in parts)
            and all(s.isdigit() and 601 <= int(s) <= 614 for s in a.statute))


# DV1: a question about the QMSR that names no part ("If a requirement of ISO 13485 conflicts with the FD&C Act ...")
_QMSR_NAMED = re.compile(r"\bISO\s*13485\b|\bQMSR\b|\bquality\s+management\s+system\s+regulation\b", re.I)
# FU2D: a question whether a named color may be used is a question for that color's own listing
_USE_OF_COLOR = re.compile(r"\b(?:can|may|could)\s+we\s+use\b|\buse\s+[^?]{0,60}?\bas\s+the\s+colou?rant\b|\bto\s+tint\b", re.I)
_STRAY_DECLINE_RE = re.compile(r"[ \t]*The available regulatory sources do not provide enough information to answer "
                               r"this(?: question)?\.[ \t]*")
_CITED_RE = re.compile(r"\[\d{1,3}\]")


def drop_stray_decline(text: str) -> tuple[str, int]:
    """The rule-9 decline sentence removed from an answer that, without it, still states something with a citation
    marker; an answer that is only the decline (or cites nothing) is returned unchanged. -> (text, sentences removed)"""
    found = _STRAY_DECLINE_RE.findall(text or "")
    if not found or not _CITED_RE.search(_STRAY_DECLINE_RE.sub(" ", text)):
        return text, 0
    lines = []
    for line in text.split("\n"):
        if _STRAY_DECLINE_RE.search(line):
            line = _STRAY_DECLINE_RE.sub(" ", line).strip()
            if not line:
                continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), len(found)


class RagPipeline:
    def __init__(self, retriever: LexicalRetriever, generator: Generator, config: RagConfig | None = None, grounding=None,
                 scope_policy=None, premise_check=None, advanced=None) -> None:
        """`grounding` is an optional Stage 13 GroundingGate. It is None by
        default, so the Stage 6/7 pipeline is unchanged. When supplied it runs
        after retrieval and context construction and can replace the generated
        answer with a deterministic refusal; retrieval itself is never altered,
        which keeps runs comparable across stages."""
        self.retriever = retriever
        self.generator = generator
        self.config = config or RagConfig()
        self.grounding = grounding
        # Stage 7: regulus.retrieval.domain.ScopePolicy (None = no general-knowledge policy)
        self.scope_policy = scope_policy
        # experimental (2026-10-20): regulus.grounding.premise.PremiseCheck; None = off, the pipeline unchanged
        self.premise_check = premise_check
        # experimental (2026-10-21): regulus.retrieval.advanced.AdvancedRetrieval; None = off, the pipeline unchanged
        self.advanced = advanced
        self._promoter = None   # regulus.rag.section_promotion.SectionPromoter, built on first use

    def section_promoter(self):
        """The section promoter of config.section_promotion, or None when off or without a database."""
        if self.config.section_promotion == "off":
            return None
        if self._promoter is None:
            db = getattr(self.retriever, "db", None)
            if db is None:
                return None
            from regulus.rag.section_promotion import SectionPromoter

            self._promoter = SectionPromoter(db, self.generator, mode=self.config.section_promotion)
        self._promoter.generator = self.generator
        return self._promoter

    def promotion_request(self, question: str, plan: QueryPlan, require_model: bool = True):
        """Whether section promotion runs for this question, decided once for answer() and for the section-resolution
        harness (regulus.evaluation.section_resolution), so both gate alike: (promoter or None, allowed - see
        may_promote -, the cosmetics parts a part-only question names or None, error). `require_model=False` (harness
        only) keeps the promoter when the generator is a retrieval-only stub."""
        allowed = may_promote(question, plan)
        promoter = self.section_promoter() if allowed else None
        error = None
        if require_model and promoter is not None and promoter.mode == "picker" and getattr(self.generator, "retrieval_only", False):
            promoter, error = None, "skipped: no model to pick with (retrieval-only harness)"
        if promoter is not None and allowed == "any" and self.scope_policy is not None \
                and not self.scope_policy.vocabulary.domain_terms(question):
            promoter = None   # no word of the regulatory vocabulary ("What is the capital of France?"): never promoted
        parts = tuple(dict.fromkeys(r.reference.part for r in plan.requirements)) if plan.requirements else None
        return promoter, allowed, parts, error

    # -- steps ---------------------------------------------------------------

    def retrieve(self, question: str, plan: QueryPlan | None = None, reference_targets: tuple[str, ...] = ()) -> tuple[SearchResponse, EvidenceRegistry | None]:
        """Stage 2: plan-driven retrieval when the retriever supports it.

        The plan is built once per question (here if the caller did not pass
        one) and handed to `run_plan`, so retrieval never re-parses the
        question. Retrievers without `run_plan` (hybrid, semantic, test fakes)
        keep the query-level path and return no registry.
        """
        run_plan = getattr(self.retriever, "run_plan", None)
        if run_plan is None:
            return self.retriever.run(question, corpus=self.config.corpus, top_k=self.config.top_k, filters=Filters()), None
        plan = plan if plan is not None else plan_query(question)
        kwargs = {}
        if reference_targets and "reference_targets" in inspect.signature(run_plan).parameters:
            kwargs["reference_targets"] = tuple(reference_targets)
        out = run_plan(plan, corpus=self.config.corpus, top_k=self.config.top_k, filters=Filters(), **kwargs)
        return out.response, out.registry

    def is_useful(self, r: RetrievalResult) -> bool:
        """Usefulness rule for optional evidence: drop trigram-only hits below the
        cut-off and (by default) contextual-only passages. A section whose specific heading is in the question
        (lexical heading corroboration) is kept: its heading, not its trigram score, names it as the provision asked
        about."""
        if (r.metadata or {}).get("heading_match"):
            return True
        if r.retrieval_method == METHOD_TRIGRAM and float(r.metadata.get("similarity", 0.0)) < self.config.min_trigram_similarity:
            return False
        if self.config.exclude_contextual_only and r.metadata.get("contextual_only"):
            return False
        return True

    def select_useful(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Legacy path: keep the retriever's own ordering, apply is_useful()."""
        return [r for r in results if self.is_useful(r)]

    def shape_for(self, plan: QueryPlan) -> AnswerShape | None:
        """Stage 12: the answer shape, or None when the setting is off."""
        return analyze_shape(plan) if self.config.answer_shape in ("adaptive", "structured") else None

    def analysis_for(self, plan: QueryPlan) -> QuestionAnalysis | None:
        """Answer-quality Stage 3: the question analysis, only in "structured" mode."""
        return analyze_question(plan) if self.config.answer_shape == "structured" else None

    def question_view(self, plan: QueryPlan) -> tuple[AnswerShape | None, QuestionAnalysis | None]:
        """The answer shape and question analysis the pipeline acts on: the ONE place
        they are decided, so anything that rebuilds a prompt (evaluation) rebuilds it exactly."""
        shape, analysis = self.shape_for(plan), self.analysis_for(plan)
        if analysis is not None and shape is not None and analysis.question_type == TYPE_CITATION and shape.mode == MODE_ENUMERATION:
            # "Which provision ...?" asks for one provision; naming a second part in the question
            # ("... refers to part 806") must not turn it into an exhaustive list of every paragraph
            shape = replace(shape, mode=MODE_FOCUSED, exhaustive=False, signals=shape.signals + ("citation_lookup_not_enumeration",),
                            rationale="a citation lookup: name the provision that answers, not every provision")
        return shape, analysis

    def complete_plan(self, plan: QueryPlan) -> QueryPlan:
        """Stage 1, corpus-aware: fill in a cross-section range the parser could not.

        "21 CFR 50.50 through 50.56" parses to its two endpoints, because which
        sections lie between them is a fact about the corpus, not about the
        identifier (regulus.planning.ranges). A retriever that can list a part's
        sections finishes the range here, once, before anything reads the plan, so
        retrieval, coverage, selection and the prompt all see the same requirements.
        Retrievers that cannot (hybrid, semantic, test fakes) leave the plan alone.
        """
        sections_in_part = getattr(self.retriever, "sections_in_part", None)
        if sections_in_part is None:
            return plan
        return expand_section_ranges(plan, sections_in_part)[0]

    def reference_lookup_plan(self, plan: QueryPlan) -> tuple[QueryPlan, tuple[str, ...]]:
        """Stage 7: a citation lookup whose only named provisions are the ones being REFERRED TO
        ("cite the provision that requires ... meeting part 803 criteria") asks for an unnamed provision.
        With a retriever that can search by content (the semantic fallback), it is re-planned as a content
        search constrained to evidence that mentions the referenced provision. Without one, the plan is
        unchanged and the Stage 5 gate declines it (WRONG_PROVISION)."""
        if self.config.answerability != "gate" or "reference_targets" not in inspect.signature(
                getattr(self.retriever, "run_plan", lambda *a, **k: None)).parameters:
            return plan, ()
        targets = tuple(gate.reference_targets(plan))
        if not targets or len(targets) != len(plan.requirements) or analyze_question(plan).question_type != TYPE_CITATION:
            return plan, ()
        return replace(plan, requirements=(), kind=KIND_GENERAL,
                       notes=plan.notes + (f"reference lookup: {', '.join(targets)} is referred to, not asked for; searched by content",)), targets

    @staticmethod
    def is_subordinate(evidence: Evidence) -> bool:
        """A child paragraph of a requested provision: part of what was asked for."""
        return evidence.tier == TIER_STRUCTURAL and REL_CHILD in evidence.relations.values()

    def select(self, registry: EvidenceRegistry, shape: AnswerShape | None = None, question: str | None = None,
               promotion=None) -> Selection:
        """Stage 3 selection; with a Stage 12 shape, children of requested
        provisions are kept like required evidence instead of competing for the
        soft budget. Stage 5 calls this too, so it measures what the pipeline uses.

        `question` only decides which required evidence survives the hard ceilings when a
        requested provision has more paragraphs than fit; see `selection._required_order`."""
        material = self.is_subordinate if shape is not None else None
        selection = select_evidence(registry, self.config.context, optional_filter=self.is_useful, material=material,
                                    question=question)
        if self.advanced is not None:
            from regulus.rag.selection import _block_chars, _clean, _truncate
            selection, _ = self.advanced.extend(selection, registry, question or registry.plan.question, self.config.context,
                                                block_chars=_block_chars, clean=_clean, truncate=_truncate)
        if promotion is not None and promotion.statute_rows:
            selection = self.statute_selection(promotion, selection)
        return selection

    def statute_selection(self, promotion, selection: Selection) -> Selection:
        """A question resolved to FD&C Act section(s): their text, verbatim, grouped by subsection and in document order,
        is the evidence (the general search's guidance and eCFR passages are not about the provision the question asks).
        Blocks the budget kept out are recorded as dropped required evidence, so the answer carries the gap notice."""
        from regulus.rag.selection import DROP_HARD_CEILING_CHARS, DroppedEvidence, SelectedEvidence, _block_chars, _clean, _truncate
        from regulus.retrieval.evidence import TIER_SUPPORTING
        from regulus.retrieval.statute import statute_index

        st = statute_index(self.retriever.db)
        # statute only: its text is the evidence. With a promoted regulation as well: the Act text follows the
        # regulation's evidence (co-promotion), which is kept
        companion = promotion.method == "statute_reference"   # ST1: Act text cited by the requested regulation
        keep = list(selection.selected) if (promotion.citations or companion) else []
        chosen, dropped, used = list(keep), [], (selection.total_chars if keep else 0)
        for n, rows in enumerate(promotion.statute_blocks, start=1):
            result = st.block_result(list(rows))
            evidence = Evidence(f"s{n}", result, TIER_SUPPORTING)
            text, truncated = _truncate(_clean(result.text), self.config.context.required_max_chars_per_result)
            chars = _block_chars(evidence, text, len(chosen) + 1, truncated)
            if used + 2 + chars > self.config.context.hard_max_total_chars:
                if companion:
                    continue   # ST1: an optional companion that does not fit is left out, not reported as missing
                dropped.append(DroppedEvidence(evidence, DROP_HARD_CEILING_CHARS, required=True))
                continue
            used += 2 + chars
            chosen.append(SelectedEvidence(len(chosen) + 1, evidence, text, truncated, True, "statute_promotion", chars))
        for k, rows in enumerate(() if companion else promotion.statute_left_out, start=1):
            dropped.append(DroppedEvidence(Evidence(f"x{k}", st.block_result(list(rows)), TIER_SUPPORTING),
                                           DROP_HARD_CEILING_CHARS, required=True))
        if len(chosen) == len(keep):
            return selection
        return Selection(tuple(chosen), tuple(selection.dropped if keep else ()) + tuple(dropped), selection.policy, used,
                         {**selection.diagnostics, "statute_promotion": list(promotion.statute)})

    def build_bundle(self, response: SearchResponse, registry: EvidenceRegistry | None, shape: AnswerShape | None = None,
                     question: str | None = None, promotion=None) -> tuple[ContextBundle, CoverageReport | None, Selection | None]:
        """Stage 3: coverage -> selection -> rendering when a registry exists;
        the legacy filter + build_context otherwise."""
        if registry is None:
            return build_context(self.select_useful(response.results), self.config.context, notes=response.notes), None, None
        coverage = check_coverage(registry)
        selection = self.select(registry, shape, question, promotion=promotion)
        if (self.config.optional_context == "incomplete_only" and coverage.explicit and coverage.complete
                and not (shape is not None and shape.mode == MODE_COMPARISON)
                and all(is_complete_sentence(s.text) or completed_by_protected_children(s, selection)
                        for s in selection.selected if s.protected and s.evidence.tier == TIER_EXACT)):
            # Stage 6: the exact evidence is all the model needs, when it is complete text on its own.
            # Kept in full (the Stage 5 context): a comparison, which contrasts each side's applicability and
            # conditions carried by the parent/child paragraphs (AS-05 overstated a scope without them), and a
            # requested provision whose text is a lead-in or a chained fragment, which the model misreads
            # without its parent and siblings (S5-011, S5-032, S5-101). docs/generation_quality_stage6.md
            selection = required_only(selection)
        if self.config.evidence_consolidation == "on":
            # Stage 8: one evidence item per distinct passage before the prompt (regulus.rag.selection)
            selection, _ = consolidate_evidence(selection)
        notes = list(response.notes)
        if coverage.state == STATE_INCOMPLETE:
            notes.append(coverage_note(coverage.missing_labels))
        return render_selection(selection, notes=notes), coverage, selection

    def build_citations(self, bundle: ContextBundle, answer: str) -> list[RagCitation]:
        mentioned = {int(m) for m in _CITE_RE.findall(answer)}
        grouped: dict[str, list] = {}
        for s in bundle.sources:
            grouped.setdefault(s.result.citation.label, []).append(s)
        out = []
        for label, group in grouped.items():
            first = group[0]
            indices = tuple(s.index for s in group)
            out.append(
                RagCitation(
                    index=first.index,
                    indices=indices,
                    label=label,
                    corpus=first.result.corpus,
                    url=first.result.source_url,
                    cited_in_answer=any(i in mentioned for i in indices),
                    citation=first.result.citation.to_dict(),
                    tier=first.tier,
                    requirement_ids=first.requirement_ids,
                    role=ROLE_CONTEXT,
                )
            )
        return sorted(out, key=lambda c: c.index)

    @staticmethod
    def _capture_evidence(question, bundle: ContextBundle, package: GenerationPackage | None,
                          coverage: CoverageReport | None, prompt_sent: str | None):
        """The exact evidence of this answer, from the same frozen object the prompt
        was rendered from, checked against the prompt actually sent. Observational:
        it reads, never changes, and a failure yields None (not captured)."""
        if package is not None:
            from regulus.rag.generation import SOURCES_CLOSE, SOURCES_OPEN
            sources, point = package.sources, evidence_trace.CAPTURE_PACKAGE
        else:
            from regulus.rag.prompt import SOURCES_CLOSE, SOURCES_OPEN
            sources, point = bundle.sources, evidence_trace.CAPTURE_CONTEXT
        return evidence_trace.capture_safely(
            sources, capture_point=point,
            prompt_sources=evidence_trace.sources_block(prompt_sent, SOURCES_OPEN, SOURCES_CLOSE),
            model_called=prompt_sent is not None,
            question=package.question if package is not None else question,
            requirements=package.requirements if package is not None else (),
            satisfied_ids=coverage.satisfied_ids if coverage is not None else (),
            missing_labels=coverage.missing_labels if coverage is not None else (),
            coverage_explicit=bool(coverage.explicit) if coverage is not None else False)

    # -- Stage 4: generation contract, validation, repair --------------------

    def correction_pass(self, messages, draft: str, package: GenerationPackage, report: ValidationReport, question: str,
                        analysis) -> tuple[tuple[str, Any, ValidationReport] | None, dict[str, Any]]:
        """answer_style "apply": a draft the validator rejected is returned to the model ONCE with what was wrong, as a
        reviewer would hand it back ("the figure 270 appears in no source", "the answer does not address 21 CFR
        70.50"). The corrected text goes through the same post-processing and the same validation, and is used only
        if it validates; otherwise the original draft and the normal repair path (verbatim quote or withheld) stand."""
        notes = correction_notes(report, package)
        if not notes:
            return None, {"attempted": False}
        # rule 9 says the EVIDENCE contains the provision only when one was supplied (requested or resolved)
        system = APPLY_RETRY_SYSTEM_PROMPT if (package.satisfied_ids or package.resolved) else messages[0]["content"]
        user = (messages[-1]["content"] + "\n\nYOUR DRAFT ANSWER:\n" + (draft or "(empty)") + "\n\nPROBLEMS FOUND IN THE DRAFT:\n"
                + "\n".join(f"- {n}" for n in notes) + "\n\nWrite the corrected answer to the QUESTION now, from the EVIDENCE only, "
                "following the ANSWER FORMAT. Output only the answer.")
        msgs = [{"role": "system", "content": system}] + list(messages[1:-1]) + [{**messages[-1], "content": user}]
        t0 = time.perf_counter()
        gen2 = self._generate(msgs, package.max_new_tokens, None)
        text = (gen2.response or "").strip()
        if not repeated_sentence(text):
            text, _ = drop_duplicate_sentences(text, _quote_sources(package))
        text, _ = strip_plan_labels(text)
        if analysis is not None:
            text, _ = normalize_answer(text, question)
        text, _ = repoint_citations(text, [(s.index, s.text) for s in package.sources])
        if precision_fix("A4"):
            text, _ = repoint_named_citations(text, package)
        report2 = validate_answer(text, package, gen2)
        # problem codes only: the notes quote rejected draft sentences, which must not reach the page or the history
        info = {"attempted": True, "codes_before": list(dict.fromkeys(p.code for p in report.problems if p.severity == "error")),
                "accepted": report2.ok, "seconds": round(time.perf_counter() - t0, 2), "codes_after": list(report2.codes())}
        return ((text, gen2, report2) if report2.ok else None), info

    def claim_support(self, text: str, package) -> list[tuple[str, float]]:
        """CV1: each cited sentence of `text` with the entailment model's probability that the sources it cites support
        it (regulus.rag.entailment); verbatim quotations, status notes and sentences under 6 words are not checked.
        [] without the model."""
        from regulus.rag.entailment import entailment_checker

        checker = entailment_checker()
        if checker is None:
            return []
        by_index = {s.index: s for s in package.sources}
        pairs, claims = [], []
        for sentence in _CV1_SENTENCE.split(text or ""):
            marks = [int(m) for m in re.findall(r"\[(\d{1,3})\]", sentence) if int(m) in by_index]
            claim = re.sub(r"\s*\[\d{1,3}\]", "", sentence).strip()
            if (not marks or len(claim.split()) < 6 or "(eCFR, binding, verbatim)" in sentence
                    or claim.lower().startswith(("status note", "evidence limit"))):
                continue
            pairs.append((" ".join(by_index[i].text or "" for i in marks), claim))
            claims.append(sentence.strip())
        try:
            return list(zip(claims, checker.support_many(pairs))) if pairs else []
        except Exception:  # noqa: BLE001 - an aid only
            return []

    def verify_claims(self, messages, draft: str, package, question: str, analysis):
        """CV1 (Chain-of-Verification with an entailment model): a validated answer's sentences the cited sources do not
        support (MiniCheck probability below CV1_MIN_SUPPORT) go back to the model ONCE to be corrected or removed; the
        revision is kept only if it validates and its weakest sentence is better supported, else the flagged sentences
        are dropped when cited statements remain. -> ((text, report) | None, info)"""
        scored = self.claim_support(draft, package)
        flagged = [c for c, p in scored if p < CV1_MIN_SUPPORT]
        info = {"checked": len(scored), "flagged": len(flagged),
                "min_support": round(min((p for _, p in scored), default=1.0), 3)}
        if not flagged:
            return None, info
        user = (messages[-1]["content"] + "\n\nYOUR DRAFT ANSWER:\n" + draft + "\n\nA FACT CHECK OF EACH SENTENCE AGAINST THE "
                "SOURCE IT CITES FOUND THESE SENTENCES NOT SUPPORTED BY THAT SOURCE:\n" + "\n".join(f"- {c}" for c in flagged)
                + "\n\nRead each one against the SOURCE it cites. Correct it to what that source says and cite the source that "
                "states it, or remove it. Keep the rest of the answer. Output only the answer.")
        msgs = [{"role": "system", "content": messages[0]["content"]}] + list(messages[1:-1]) + [{**messages[-1], "content": user}]
        gen2 = self._generate(msgs, package.max_new_tokens, None)
        text = (gen2.response or "").strip()
        if not repeated_sentence(text):
            text, _ = drop_duplicate_sentences(text, _quote_sources(package))
        text, _ = strip_plan_labels(text)
        if analysis is not None:
            text, _ = normalize_answer(text, question)
        text, _ = repoint_citations(text, [(s.index, s.text) for s in package.sources])
        report2 = validate_answer(text, package, gen2)
        rescored = self.claim_support(text, package)
        new_min = min((p for _, p in rescored), default=1.0)
        info.update(revised_min_support=round(new_min, 3), revised_ok=report2.ok)
        if report2.ok and new_min > info["min_support"]:
            info["outcome"] = "revised"
            return (text, report2), info
        kept = draft
        for c in flagged:
            kept = kept.replace(c, "")
        kept = re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", kept)).strip()
        if _CITED_RE.search(kept):
            report3 = validate_answer(kept, package, gen2)
            if report3.ok:
                info["outcome"] = "unsupported_removed"
                return (kept, report3), info
        info["outcome"] = "kept"
        return None, info

    def facts_first_answer(self, messages, package, question: str, analysis):
        """BN2: the facts-first draft (FR1's two steps) post-processed and validated like the plain one ->
        (text, gen, report), or None when the model listed no cited facts."""
        facts = self._generate([messages[0], {**messages[-1], "content": messages[-1]["content"] + FR1_PROMPT}],
                               FR1_MAX_TOKENS, None)
        listed = (facts.response or "").strip()
        if not listed or not _CITED_RE.search(listed):
            return None
        msgs = list(messages[:-1]) + [{**messages[-1], "content": messages[-1]["content"]
                                       + "\n\nFACTS THE SOURCES STATE ABOUT THE QUESTION (listed first):\n" + listed + FR1_ANSWER}]
        gen2 = self._generate(msgs, package.max_new_tokens, None)
        text = (gen2.response or "").strip()
        if not repeated_sentence(text):
            text, _ = drop_duplicate_sentences(text, _quote_sources(package))
        text, _ = strip_plan_labels(text)
        if analysis is not None:
            text, _ = normalize_answer(text, question)
        text, _ = repoint_citations(text, [(s.index, s.text) for s in package.sources])
        if precision_fix("A4"):
            text, _ = repoint_named_citations(text, package)
        return text, gen2, validate_answer(text, package, gen2)

    def _generate(self, messages, max_new_tokens: int | None, on_text=None):
        """Call the generator with the per-question budget when it accepts one,
        and with the streaming text callback when there is one and it accepts it."""
        fn = self.generator.generate
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = {}
        kwargs = {}
        if "max_new_tokens" in params and max_new_tokens:
            kwargs["max_new_tokens"] = max_new_tokens
        if on_text is not None and "on_text" in params:
            kwargs["on_text"] = on_text
        return fn(messages, **kwargs)

    def repair_answer(self, text: str, package: GenerationPackage, report: ValidationReport, *, quote: bool = True,
                      unquotable: frozenset[str] = frozenset(), sources=None) -> tuple[str, str | None]:
        """Deterministic repair from the exact sources; never invents content.

        empty / degenerate / unsupported / refusal with evidence -> the answer is
        replaced by the verbatim exact text of the requested provisions (or a
        safe message when there are none); invalid markers are removed;
        requested provisions the model left out are appended verbatim; a
        truncated answer is marked as such.

        Stage 5 (the pipeline passes these when the answerability gate is on):
        `quote` False -> a replaced answer is withheld with a safe message instead of
        quoting sources that were not shown to answer the question; `unquotable`
        requirement ids (part-level requests) are never appended verbatim, because
        their first matching passage is not what was asked; `sources` lets the safe
        message name the sources consulted.
        """
        codes = report.codes()
        replace_all = report.status in (STATUS_EMPTY, STATUS_DEGENERATE, STATUS_UNSUPPORTED) or REFUSAL_WITH_EVIDENCE in codes
        if replace_all:
            if quote and package.satisfied_ids:
                return f"{UNVALIDATED_PREFIX}\n{verbatim_requested(package)}", "extractive_replacement"
            if sources is not None:
                return unvalidated_answer(package, sources, quote=False)[0], FB_UNVALIDATED
            return UNVALIDATED_GENERAL, "safe_message"
        repairs: list[str] = []
        if report.status == STATUS_CITATION_INVALID:
            text = strip_invalid_citations(text, report.invalid_citations)
            repairs.append("invalid_citations_removed")
        def locate(normalized: str, sentence: str) -> str | None:
            # the report's sentences come from the unrepaired text; invalid markers may since have been stripped
            for candidate in (sentence, strip_invalid_citations(sentence, report.invalid_citations)):
                if candidate and candidate in normalized:
                    return candidate
            return None

        # Stage 7: a statement presented as provision X but worded from another provision is replaced IN PLACE by
        # X's own verbatim text (its source is known: the finding's citation)
        misattributed = [p for p in report.problems if p.code == PARAGRAPH_MISATTRIBUTION and p.severity == "error" and p.sentence]
        if misattributed:
            normalized = attach_trailing_markers(text)
            for prob in misattributed:
                found = locate(normalized, prob.sentence)
                if found and prob.citation is not None:
                    normalized = normalized.replace(found, extractive_items(package, (prob.citation,)) + "\n")
            text = normalized.strip()
            repairs.append("misattributed_statements_replaced")
        # a statement about a requested provision that cites unrelated sources is
        # removed and the provision's verbatim exact text takes its place
        mismatched = [p for p in report.problems if p.code in (CITATION_MISMATCH, PROVISION_CITATION_MISMATCH) and p.sentence]
        if mismatched:
            normalized = attach_trailing_markers(text)
            for prob in mismatched:
                found = locate(normalized, prob.sentence)
                if found:
                    normalized = normalized.replace(found, "")
            text = re.sub(r"[ \t]{2,}", " ", normalized).strip()
            affected = tuple(dict.fromkeys(p.requirement_id for p in mismatched if p.requirement_id)) or tuple(package.satisfied_ids)
            affected = tuple(r for r in affected if r not in unquotable)
            if affected:
                text = (text.rstrip() + "\n" if text.strip() else "") + extractive_answer(package, affected)
            repairs.append("mismatched_statements_replaced")
        # Stage 7: a statement carrying a condition that no supplied source contains is removed (nothing is added
        # in its place: the condition has no source to restore)
        # Only when the rest of the answer had no substantive error: removing sentences from an answer that is
        # failing for another reason could leave a plausible remainder that no check can see is wrong.
        invented = [p for p in report.problems if p.code == INVENTED_CONDITION and p.severity == "error" and p.sentence]
        substantive = {p.code for p in report.errors} - {INVENTED_CONDITION, INVALID_CITATION}
        if invented and not substantive:
            normalized = attach_trailing_markers(text)
            for prob in invented:
                found = locate(normalized, prob.sentence)
                if found:
                    normalized = normalized.replace(found, "")
            text = re.sub(r"[ \t]{2,}", " ", normalized).strip()
            repairs.append("invented_conditions_removed")
        missing = tuple(r for r in report.unaddressed if not any(p.requirement_id == r for p in mismatched) and r not in unquotable)
        if missing:
            text = (text.rstrip() + "\n" if text.strip() else "") + extractive_answer(package, missing)
            repairs.append("appended_missing_provisions")
        # Stage 12: items of an explicit enumeration, or whole requirement groups of a
        # synthesis, that the answer left out - appended verbatim from their sources
        items = tuple(dict.fromkeys(p.citation for p in report.problems if p.code == ENUMERATION_INCOMPLETE and p.citation is not None))
        plan = package.answer_plan
        for p in report.problems:
            if p.code == MATERIAL_GROUP_UNCOVERED and plan is not None and (g := plan.group_of(p.citation)) is not None:
                items += tuple(i for i in g.members if i not in items)
        if items:
            text = (text.rstrip() + "\n" if text.strip() else "") + extractive_items(package, items)
            repairs.append("appended_missing_items")
        if report.truncated:
            # Never present a severed answer as a finished one. Any requested
            # provision the cut removed has already been appended verbatim
            # above, so this marks what is left: prose that stopped mid-thought.
            text = text.rstrip() + (
                "\n(Incomplete: generation stopped at the token limit before this answer finished, "
                "so wording after this point is missing.)"
            )
            repairs.append("truncation_noted")
        return text, "+".join(repairs) if repairs else None

    def with_statute_references(self, question: str, plan: QueryPlan, promotion):
        """ST1: `promotion` with the cited Act sections added (a new "statute_reference" promotion when the question named
        its regulation itself), or `promotion` unchanged."""
        promoter = self.section_promoter()
        primary = list(promotion.sections) if promotion is not None and promotion.sections else [
            r.reference.section for r in plan.requirements if r.reference.title == "21" and r.reference.section]
        if promoter is None or not primary:
            return promotion
        try:
            extra = promoter.statute_companions(question, list(dict.fromkeys(primary))[:3])
        except Exception:  # noqa: BLE001 - an aid only
            return promotion
        if extra is None:
            return promotion
        if promotion is None:
            return extra
        return replace(promotion, statute=extra.statute, statute_rows=extra.statute_rows, statute_blocks=extra.statute_blocks,
                       statute_left_out=extra.statute_left_out)

    def compose_from_identified(self, package, question: str, plan: QueryPlan, promotion) -> tuple[str | None, str | None]:
        """FP1 / RS1 (regulus.rag.validator_switches): a confirmation question asserting a figure the identified provision
        does not state (regulus.rag.premise_figures), or a records question about a section that requires none
        (regulus.rag.records_silence) - both decided from the database text, never by the model. (None, None) otherwise."""
        sections = [r.reference.section for r in plan.requirements if r.reference.title == "21" and r.reference.section]
        if any(r.reference.section is None for r in plan.requirements):
            return None, None
        labels = [f"21 CFR {s}" for s in dict.fromkeys(sections)]
        labels += [f"FD&C Act § {s}" for s in (promotion.statute if promotion is not None else ())]
        if precision_fix("FP1"):
            from regulus.rag.premise_figures import compose as compose_premise

            text = compose_premise(package, question, labels)
            if text:
                return text, FB_COMPOSED_PREMISE
        if precision_fix("RS1") and sections and not (promotion is not None and promotion.statute):
            from regulus.rag.records_silence import asks_records, compose as compose_records

            promoter = self.section_promoter()
            if promoter is not None and asks_records(question):
                try:
                    texts = {f"21 CFR {s}": promoter._section_text(s) for s in dict.fromkeys(sections)}
                except Exception:  # noqa: BLE001 - without the full text the silence cannot be established
                    return None, None
                text = compose_records(package, question, texts)
                if text:
                    return text, FB_COMPOSED_RECORDS
        return None, None

    def repair_with_fallback(self, text: str, package: GenerationPackage, report: ValidationReport, gen, *, question: str,
                             plan: QueryPlan, shape: AnswerShape | None, analysis: QuestionAnalysis | None, sources,
                             focus_quote: bool = False, focus_lead: str | None = None) -> dict[str, Any]:
        """Repair a generated answer that failed validation, then (Stage 5, answerability="gate") never
        show a result that is still ungrounded: it is replaced by the safe fallback builder. The one
        implementation used by `answer()` and by the evaluation harness.

        Returns text, repair, repair_output (the repair's own text), final (validation report),
        fallback_type and fallback_reason."""
        gate_on = self.config.answerability == "gate"
        quote = quoting_allowed(question, plan, shape, analysis, package) if gate_on else True
        premise = [p for p in report.problems if p.code == UNSUPPORTED_PREMISE and p.severity == "error"]
        if gate_on and premise and provision_is_identified(plan, package):
            # Stage 7: a yes/no answer affirmed what the identified provision does not state
            terms = re.findall(r"'([^']+)'", premise[0].detail) or [premise[0].detail]
            text, fallback_type = premise_not_established(package, terms)
            return {"text": text, "repair": fallback_type, "repair_output": text, "final": validate_answer(text, package, gen),
                    "fallback_type": fallback_type, "fallback_reason": f"the answer affirmed a premise the provision does not state: {premise[0].detail}"}
        kwargs = ({"quote": quote, "unquotable": frozenset(r.id for r in plan.requirements if r.reference.section is None),
                   "sources": sources} if gate_on else {})
        text, repair = self.repair_answer(text, package, report, **kwargs)
        repair_output = text if repair is not None else None
        final = validate_answer(text, package, gen) if repair is not None else report
        fallback_type = fallback_reason = None
        if repair == FB_UNVALIDATED:
            fallback_type, fallback_reason = FB_UNVALIDATED, f"the generated answer failed validation: {report.status}"
        elif gate_on and (final.status in _UNGROUNDED or REFUSAL_WITH_EVIDENCE in final.codes()):
            failed_status = final.status
            text, fallback_type = unvalidated_answer(package, sources, quote=quote)
            fallback_reason = f"the repaired answer still failed validation: {failed_status}"
            repair = (repair + "+" if repair else "") + "failed_repair_fallback"
            final = validate_answer(text, package, gen)
        if fallback_type == FB_UNVALIDATED and focus_quote:
            # Q1: the governing provision's relevant paragraphs, verbatim, instead of "not shown". Grounded by
            # construction (the EXACT text of the identified provision, each paragraph under its own citation); the
            # answer validator, which requires every requested provision and list item to be addressed, is recorded
            # but does not decide - a quote is partial on purpose
            quoted = focused_quote(package, question, **({"lead": focus_lead} if focus_lead else {}))
            if quoted:
                text, final, fallback_type = quoted, validate_answer(quoted, package, gen), FB_FOCUSED_QUOTE
                fallback_reason = f"{fallback_reason}; the governing provision is quoted instead"
                repair = (repair + "+" if repair else "") + FB_FOCUSED_QUOTE
        return {"text": text, "repair": repair, "repair_output": repair_output, "final": final,
                "fallback_type": fallback_type, "fallback_reason": fallback_reason}

    # -- main entry ----------------------------------------------------------

    def relevant_uncited(self, answer: str, package, question: str) -> str | None:
        """PC1 (27 Oct): the paragraphs of the answer's single requested 21 CFR section (not a color listing - LC1 does
        those) that no citation of the answer points to and that the cross-encoder scores at least PC1_MIN_SCORE for
        the question, at most PC1_MAX, verbatim and cited (70.19-Q1 left out (h), the full deposit on resubmission
        after withdrawal; 71.1-Q1 left out (e), the signature)."""
        from regulus.rag.color_listing import LISTING_RE
        from regulus.rag.generation import _verbatim, whole_sentences
        from regulus.rag.section_promotion import RERANKER
        from regulus.retrieval.rerank import reranker

        labels = [label for _, label in package.requirements if re.match(r"^21 CFR \d+\.\d+[a-z]?$", label)]
        if len(labels) != 1 or LISTING_RE.match(labels[0]):
            return None
        own = [s for s in package.sources if re.match(rf"^{re.escape(labels[0])}(?:\(|$)", s.citation or "")]
        cited = {int(m) for m in re.findall(r"\[(\d{1,3})\]", answer or "")}
        cited_labels = {s.citation for s in own if s.index in cited}
        if not cited_labels:
            return None
        todo = [s for s in own if s.index not in cited and not any(s.citation.startswith(c) or c.startswith(s.citation)
                                                                  for c in cited_labels) and len(s.text or "") >= 60]
        rr = reranker(RERANKER) if RERANKER and todo else None
        if rr is None:
            return None
        try:
            scores = rr.score(question, [s.text for s in todo])
        except Exception:  # noqa: BLE001 - an aid only
            return None
        keep = [s for sc, s in sorted(zip(scores, todo), key=lambda x: -x[0]) if sc >= PC1_MIN_SCORE][:PC1_MAX]
        keep = sorted(keep, key=lambda s: s.index)
        lines = [f"{s.citation} (eCFR, binding, verbatim): {_verbatim(whole_sentences(s.text), s.index)}"
                 for s in keep if whole_sentences(s.text)]
        return "\n".join([PC1_LEAD.format(section=labels[0])] + lines) if lines else None

    def small_part_paragraphs(self, question: str, plan):
        """DV1 (27 Oct): (plan, promotion) for a question that names only one small 21 CFR part (at most DV1_MAX_SECTIONS
        live sections - Part 820, the QMSR, has six) or, naming none, ISO 13485 or the QMSR (Part 820); None otherwise.
        "How must advisory notices be handled under Part 820?" read the whole part, the budget cut it to scattered
        paragraphs and 820.10(b)(4) was not among them (device baseline: 3 of 12 'not addressed' or withheld)."""
        from regulus.rag.section_promotion import DV1_MAX_SECTIONS, Promotion

        reqs = plan.requirements
        part = None
        if reqs and all(r.reference.section is None and r.reference.title == "21" for r in reqs) \
                and len({r.reference.part for r in reqs}) == 1:
            part = reqs[0].reference.part
        elif not reqs and _QMSR_NAMED.search(question or ""):
            part = "820"
        if not part or _ASKS_WHICH_PROVISIONS.search(question or ""):
            return None   # "which provisions of Part 820 are ...": answered across the part, not from three paragraphs
        if precision_fix("DV2") and (_GUIDANCE_DOCUMENT.search(question or "") or _ASKS_OTHER_PARTS.search(question or "")):
            return None
        if part not in DV1_PARTS:
            # the parts DV1 was measured on (28 Oct retrieval check, 225 dev, client keys: +7 complete in Part 820 - M-005,
            # M-009, M-015..M-018, M-042 - Part 11 unchanged); in Part 806 its three paragraphs dropped R-005's 806.10(b)
            # ("10 working days"), which the whole section had supplied
            return None
        promoter = self.section_promoter()
        if promoter is None:
            return None
        try:
            live = self.live_part_sections(part) or set()
            if not live or len(live) > DV1_MAX_SECTIONS:
                return None
            paras = promoter.part_paragraphs(question, part)
        except Exception:  # noqa: BLE001 - an aid only
            return None
        if not paras:
            return None
        promoted = _without_part_requirements(plan_query(question + " (21 CFR " + ", ".join(paras) + ")"))
        if not promoted.requirements:
            return None
        # DV2: the paragraphs' best cross-encoder score, so Q2 may quote them for an answer that cannot be shown (M-026: the
        # draft failing validation was withheld, where the picker's promotion of the same 820.45 had been quoted - PASS)
        score = promoter.__dict__.get("last_part_score") if precision_fix("DV2") else None
        return promoted, Promotion(sections=tuple(paras), companions=(), method="part_paragraphs", rerank=score)

    def nonexistent_provision_answer(self, question: str, coverage) -> str | None:
        """NX1 (29 Oct): a requested provision the database shows does not exist is said not to exist, with any premise
        term its section never uses (regulus.rag.nonexistence); None otherwise, and the caller keeps its wording."""
        db = getattr(self.retriever, "db", None)
        if not precision_fix("NX1") or db is None or coverage is None or not coverage.missing_labels:
            return None
        from regulus.rag.nonexistence import statement

        try:
            return statement(db, question, coverage.missing_labels)
        except Exception:  # noqa: BLE001 - an aid: the existing wording stands
            return None

    def closest_sources(self, question: str, package) -> tuple[int, ...]:
        """LM1: the (at most LM1_TOP) sources the cross-encoder scores highest for the question, at least LM1_MIN_SCORE,
        in score order; () without the reranker. 70.19-Q2: of 18 fee paragraphs the model read the new-petition deposit
        (c) for an amendment, which (i) prices, in the middle of the evidence."""
        from regulus.rag.section_promotion import RERANKER
        from regulus.retrieval.rerank import reranker

        rr = reranker(RERANKER) if RERANKER else None
        if rr is None:
            return ()
        try:
            scores = rr.score(question, [s.text or "" for s in package.sources])
        except Exception:  # noqa: BLE001 - an aid only
            return ()
        ranked = sorted(zip(scores, (s.index for s in package.sources)), key=lambda x: -x[0])
        return tuple(i for sc, i in ranked[:LM1_TOP] if sc >= LM1_MIN_SCORE)

    def read_followup(self, question: str, plan: QueryPlan, context):
        """How a question asked later in a conversation is read against it (regulus.rag.conversation.resolve), with
        the signals that need the database: whether it names an FD&C Act section or a color additive the anchored
        sections do not list, and whether its own words rank an anchored section. None without a context."""
        from regulus.rag.conversation import resolve as resolve_followup, snowball_stemmer, subject_words

        if not context:
            return None
        anchor = context.anchor
        promoter = self.section_promoter()
        names = bool(plan.requirements) or bool(plan.docket)
        other_entity = ranked = False
        heading_words: set[str] = set()
        stem = None
        if promoter is not None and anchor:
            try:
                if precision_fix("FS2"):
                    stem = snowball_stemmer(promoter.db, [question] + [t.question for t in context.turns]
                                            + list(promoter.heading_words(anchor.cfr)))
                from regulus.retrieval.entities import entity_index
                from regulus.retrieval.section_index import SectionIndex

                lex = SectionIndex.question_lexemes(promoter.db, question)
                names = names or bool(promoter.named_statute(question, lex))
                ents = entity_index(promoter.index)
                colors = ents.match(question)
                other_entity = bool(colors) and not any(set(c.sections) & set(anchor.cfr) for c in colors)
                if (other_entity and precision_fix("FU2D") and not any(ents.is_color_listing(s) for s in anchor.cfr)
                        and not _USE_OF_COLOR.search(question)):
                    # FU2D: a color named in a conversation about a general section (82.51 lakes, 70.25 labeling) is
                    # asked about under it - "How would we name a calcium lake of FD&C Blue No. 1?" after 82.51
                    other_entity = False
                if not (names or other_entity):
                    ranked = promoter.ranks_any(question, anchor.cfr, anchor.statute, top=FOLLOWUP_RANK_TOP)
                    heading_words = set().union(*(subject_words(h, stem) for h in promoter.heading_words(anchor.cfr)))
            except Exception:  # noqa: BLE001 - an aid: without the signals the text alone decides
                pass
        fu = resolve_followup(question, context, names_provision=names, other_entity=other_entity,
                              anchor_ranked=ranked, heading_words=heading_words, stem=stem)
        if (precision_fix("CQ1") and fu is not None and anchor and not names and not other_entity
                and fu.reason != "product_type_chosen"   # AMB1: the reader's choice is applied as given, never rewritten
                and context.earlier_question and self.generator is not None
                and not getattr(self.generator, "retrieval_only", False)):
            fu = self.rewrite_followup(question, context, fu, promoter)
        return fu

    def rewrite_followup(self, question: str, context, fu, promoter):
        """CQ1 (regulus.rag.conversation, conversational query rewriting): the model rewrites the follow-up into a
        standalone question; one naming the anchored provision is answered as a question about it (the earlier
        question shown as context), one left unchanged as a new question. On any failure the word rules' reading
        stands."""
        from regulus.rag.conversation import (REWRITE_MAX_TOKENS, Anchor, FollowUp, read_rewrite, rewrite_messages)

        anchor = context.anchor
        label = anchor.labels()[0]
        try:
            heading = (promoter.heading_words(anchor.cfr[:1]) or [None])[0] if promoter is not None and anchor.cfr else None
            kw = {"think_budget": 0} if getattr(self.generator, "thinking", False) is True else {}
            gen = self.generator.generate(rewrite_messages(question, context.earlier_question[:400], label, heading),
                                          max_new_tokens=REWRITE_MAX_TOKENS, **kw)
            read = read_rewrite(getattr(gen, "response", "") or "", question, anchor)
        except Exception:  # noqa: BLE001 - an aid: the word rules decide
            return fu
        if read is None:
            return replace(fu, reason=fu.reason + "+rewrite_unusable")
        text, continues = read
        if continues:
            return FollowUp(True, "rewrite", text, anchor, context.earlier_question[:400], fu.cues)
        return FollowUp(False, "rewrite_new_topic", question, Anchor(), None, fu.cues)

    def live_part_sections(self, part: str) -> set[str] | None:
        """The sections of a 21 CFR part that have live, unreserved text in the database (cached), or None."""
        cache = self.__dict__.setdefault("_live_sections", {})
        if part not in cache:
            db = getattr(self.retriever, "db", None)
            try:
                rows = db.query("SELECT DISTINCT section_number FROM cfr_records WHERE part_number = %(p)s AND "
                                "removed_on IS NULL AND NOT is_reserved", {"p": part}) if db is not None else None
                cache[part] = {r["section_number"] for r in rows} if rows else None
            except Exception:  # noqa: BLE001 - without the database's answer nothing is said about a section's status
                cache[part] = None
        return cache[part]

    def part_mentions(self, part: str, pattern: str) -> bool | None:
        """Whether any live text of the 21 CFR part matches the (PostgreSQL, case-insensitive) pattern; None unknown."""
        db = getattr(self.retriever, "db", None)
        if db is None:
            return None
        try:
            rows = db.query("SELECT 1 FROM cfr_records WHERE part_number = %(p)s AND removed_on IS NULL AND NOT is_reserved "
                            "AND body_text ~* %(r)s LIMIT 1", {"p": part, "r": pattern})
        except Exception:  # noqa: BLE001 - unknown: nothing is said
            return None
        return bool(rows)

    def qmsr_request(self, question: str, plan: QueryPlan) -> tuple[list, list]:
        """QM1: (former part 820 sections the question names, ISO 13485 clauses it names that do not exist)."""
        if not precision_fix("QM1"):
            return [], []
        from regulus.rag import qmsr

        live = self.live_part_sections("820")
        if not live:
            return [], []
        former = qmsr.former_refs(plan, live)
        if precision_fix("QM2"):
            former += [f for f in qmsr.former_by_name(question, live) if f not in former]
        unknown = qmsr.unknown_clauses(question)
        about_820 = former or any(r.reference.part == "820" for r in plan.requirements) or re.search(
            r"\bpart\s+820\b|\bQMSR\b|quality management system regulation", question, re.I)
        return former, (unknown if about_820 else [])

    @staticmethod
    def qmsr_plan(question: str, plan: QueryPlan, former: list) -> QueryPlan:
        """The plan with the former sections replaced by the live sections that carry their subject."""
        from regulus.planning.plan import Requirement, _plan_kind
        from regulus.rag import qmsr

        gone = {s for s, _ in former}
        widened = plan_query(question + " (21 CFR " + ", ".join(qmsr.successors(former)) + ")")
        # the former sections, and a whole "Part 820" (its first matching passages are not the answer), give way to
        # the live sections named above
        kept = [r for r in widened.requirements if r.reference.section not in gone
                and not (r.reference.part == "820" and r.reference.section is None)]
        reqs = tuple(Requirement(id=f"r{i}", order=i, reference=r.reference, raw=r.raw, origin=r.origin, mention=r.mention)
                     for i, r in enumerate(kept, start=1))
        return replace(widened, requirements=reqs, kind=_plan_kind(widened.query_type, reqs, widened.scan))

    def answer(self, question: str, on_event=None, context=None) -> RagAnswer:
        """Answer a question (see _answer_once). AR2: a validated draft that says the provisions do not specify
        something the question asks gets one more pass with the section of the promoted part that holds it."""
        first = self._answer_once(question, on_event=on_event, context=context)
        if not precision_fix("AR2") or context is not None:
            return first
        try:
            topics = absence_topics(first.answer)
            promo = (first.metadata or {}).get("section_promotion") or {}
            promoter = self.__dict__.get("_promoter")
            if not topics or not promo.get("sections") or promoter is None:
                return first
            from regulus.rag.section_promotion import Promotion

            prev = Promotion(sections=tuple(promo["sections"]), companions=tuple(promo.get("companions") or ()), method="")
            extra = promoter.absence_sections(question, topics, prev)
            if not extra:
                return first
            self.__dict__["_ar2_extra"] = tuple(extra)
            try:
                second = self._answer_once(question, on_event=None, context=None)
            finally:
                self.__dict__.pop("_ar2_extra", None)
            ok = ((second.metadata or {}).get("validation") or {}).get("final", {}).get("ok")
            if ok and len(absence_topics(second.answer)) < len(topics) and \
                    (second.metadata or {}).get("fallback_type") == (first.metadata or {}).get("fallback_type"):
                second.metadata["absence_retrieval"] = {"topics": topics, "added": list(extra), "kept": True}
                return second
            first.metadata["absence_retrieval"] = {"topics": topics, "added": list(extra), "kept": False}
            return first
        except Exception as exc:  # noqa: BLE001 - an aid: the first answer stands
            (first.metadata or {})["absence_retrieval"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            return first

    def _answer_once(self, question: str, on_event=None, context=None) -> RagAnswer:
        """Answer a question. `on_event(kind, data)`, when given, is told about
        progress as it happens - "context" (the sources selected), "generation_started",
        "delta" (draft text of the one generation, as produced), "generation_finished" -
        so a caller can stream. It never changes the answer: validation and repair
        still run on the complete text, and the returned RagAnswer is the final
        result. An exception raised by `on_event` aborts the call (used to cancel
        generation when a streaming client disconnects)."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if context and precision_fix("CT1") and cosmetics_conversation(context):
            from regulus.retrieval.section_index import _FOCUS, deployment_scope, focus

            if _FOCUS.get() is None and deployment_scope() is None:
                # CT1: a conversation about cosmetics provisions keeps its product area for its follow-ups ("Who must
                # permit access?" read across Title 21 went to drug, device or food rules: 15 of 75, 26 Oct)
                with focus("cosmetics"):
                    return self.answer(question, on_event=on_event, context=context)
        emit = on_event or (lambda kind, data: None)
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        asked = question
        spelling = None
        if precision_fix("SP1") and getattr(self.retriever, "db", None) is not None:
            # SP1: a misspelled word corrected against the regulations' own vocabulary ("carmin", "anatto"), as a
            # search engine's "did you mean" does (regulus.retrieval.spelling); the user's words stay the asked question
            try:
                from regulus.retrieval.spelling import vocabulary

                corrected, spelling = vocabulary(self.retriever.db).correct(question)
                if spelling:
                    question = corrected
            except Exception:  # noqa: BLE001 - an aid: the question is answered as asked
                spelling = None
        plan = plan_query(question)  # Stage 1: the one and only parse of the question
        # a follow-up (`context`: regulus.rag.conversation.ConversationContext) is read against its conversation; without
        # a context nothing below changes
        followup = self.read_followup(question, plan, context) if context else None
        if followup is not None and followup.question != question:
            question = followup.question   # "paragraph (b)" / "this section" written out as the anchored section
            plan = plan_query(question)
        anchored = followup is not None and followup.anchored and not plan.requirements
        # QM1: a section of the former Quality System regulation (or an ISO 13485 clause that does not exist) - the
        # question is re-planned onto the live part 820 section that now carries the subject (regulus.rag.qmsr)
        qmsr_former, qmsr_unknown = self.qmsr_request(question, plan)
        # QC1: a question whether Part 820 requires ISO 13485 certification is answered from 820.10 (regulus.rag.qmsr)
        qmsr_cert = False
        if precision_fix("QC1") and not (qmsr_former or qmsr_unknown):
            from regulus.rag import qmsr as _qmsr

            if _qmsr.asks_certification(question) and self.live_part_sections("820"):
                qmsr_cert = self.part_mentions("820", r"certif") is False   # confirmed silent, not unknown
        if qmsr_former or qmsr_unknown or qmsr_cert:
            plan = self.qmsr_plan(question, plan, qmsr_former)
        own_plan = plan   # the question's own plan: every scope decision is made on it, never on a promoted one
        promotion = None
        promotion_error = None
        named_companions_added: list[str] = []
        dv1 = self.small_part_paragraphs(question, plan) if (precision_fix("DV1") and not anchored) else None
        if dv1 is not None:
            # DV1: a question naming only a small part (or ISO 13485 / the QMSR) is answered from that part's most
            # relevant paragraphs, promoted as if it had cited them
            plan, promotion = dv1
            promoter, allowed, parts, promotion_error = None, None, None, None
        else:
            promoter, allowed, parts, promotion_error = self.promotion_request(question, plan)
        if anchored and promoter is None and allowed == "any":
            promoter = self.section_promoter()   # the conversation names the domain a short follow-up leaves out
        if promoter is not None:
            # the section(s) that answer a question naming none - or naming only a 21 CFR part of the cosmetics
            # programme ("Under Part 82, ...") - are resolved, and the plan is rebuilt as if the question cited them
            part_only = bool(plan.requirements)
            try:
                promotion = (promoter.anchored(question, followup.anchor.cfr, followup.anchor.statute)
                             if anchored and allowed == "any" else None)
                if promotion is None:
                    promotion = promoter.resolve(question, parts=parts, named_only=allowed == "named_statute")
                extra = self.__dict__.get("_ar2_extra")
                if promotion is not None and extra:
                    add = tuple(s for s in extra if s not in promotion.citations)
                    promotion = replace(promotion, companions=promotion.companions + add,
                                        notes=promotion.notes + (f"absence retrieval: {', '.join(add)}",))
            except Exception as exc:  # noqa: BLE001 - promotion is an aid: on any failure the question is answered as before
                promotion = None
                promotion_error = f"{type(exc).__name__}: {exc}"[:300]
            if promotion is not None and promotion.citations:
                promoted = plan_query(question + promotion.suffix)
                if part_only:
                    promoted = _without_part_requirements(promoted)
                if promoted.requirements:
                    plan = promoted
                else:
                    promotion = None
            elif promotion is not None and promotion.statute_rows:
                if part_only and not all((r.raw or "").lstrip().startswith("§") for r in plan.requirements):
                    promotion = None   # a real 21 CFR part is asked for as well: not replaced by the statute (code review)
                else:
                    # "FD&C Act § 607" is read by the CFR parser as "21 CFR part 607": the statute section replaces it
                    plan = _without_part_requirements(plan, keep_none=True) if part_only else plan
            elif promotion is not None:
                promotion = None
        elif (self.config.section_promotion != "off" and plan.requirements and plan.docket is None
              and all(r.reference.section is not None and r.reference.title == "21" and r.reference.part in _COSMETIC_PARTS()
                      for r in plan.requirements)
              and analyze_question(plan).scope_status not in (SCOPE_OUT, SCOPE_MIXED)):
            # the question names cosmetics-programme sections itself: their companions are added as for a promoted section
            named_promoter = self.section_promoter()
            if named_promoter is not None:
                try:
                    named = [r.reference.section for r in plan.requirements]
                    comps = named_promoter.named_companions(named, question)
                    if comps:
                        widened = plan_query(question + " (21 CFR " + ", ".join(comps) + ")")
                        if len(widened.requirements) > len(plan.requirements):
                            plan = widened
                            promotion_error = None
                            promotion = None
                            named_companions_added = comps
                except Exception as exc:  # noqa: BLE001 - an aid only
                    promotion_error = f"named companions: {type(exc).__name__}: {exc}"[:300]
        if precision_fix("ST1") and not (promotion is not None and promotion.statute):
            # ST1 (regulus.rag.validator_switches): the FD&C Act text the requested regulation itself cites, as companion
            # evidence after the regulation's own (default off until its live A/B)
            promotion = self.with_statute_references(question, plan, promotion)
        timings["promotion_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        plan = self.complete_plan(plan)  # ... and the one and only corpus lookup it needs
        original_plan = plan if promotion is None else self.complete_plan(own_plan)
        plan, reference_targets = self.reference_lookup_plan(plan)
        # Stage 12 shape + Stage 3 question analysis (None unless "structured"), decided in one place
        shape, analysis = self.question_view(plan)
        timings["plan_ms"] = round((time.perf_counter() - t0) * 1000, 2)  # part of retrieval_ms
        response, registry = self.retrieve(question, plan, reference_targets)
        # scope is decided from the question's OWN plan: a reference lookup still names a provision
        scope_decision = (self.scope_policy.decide(question, original_plan, response.diagnostics, results=bool(response.results))
                          if self.scope_policy is not None and self.config.scope_policy == "general_knowledge" else None)
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t1 = time.perf_counter()
        bundle, coverage, selection = self.build_bundle(response, registry, shape, question, promotion=promotion)
        timings["context_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        emit("context", {"no_context": bundle.is_empty, "sources": [
            {"index": s.index, "citation": s.result.citation.label, "corpus": s.result.corpus, "url": s.result.source_url,
             "tier": s.tier} for s in bundle.sources]})

        no_context = bundle.is_empty
        model_called = False
        input_tokens = output_tokens = None

        # Stage 13 grounding gate: decided from the verified database and from
        # structural identifier comparison, never from similarity.
        grounding_decision = None
        if self.grounding is not None:
            verification, pre = self.grounding.check_identifier(question)
            grounding_decision = pre
            if not pre.refused:
                post = self.grounding.check_context(verification, [s.result for s in bundle.sources])
                if post.checked:
                    grounding_decision = post
        premise = self.premise_check.decide(question) if self.premise_check is not None else None
        # Stage 3: explicit requirements without exact evidence. With nothing
        # supported the answer is deterministic (the model is not asked to
        # describe a provision it was not given); with some supported the model
        # answers from those and the unresolved ones are stated afterwards.
        t_pkg = time.perf_counter()
        package = (build_package(plan, coverage, selection,
                                 notes=(tuple(bundle.notes) + _statute_notes(promotion) if (promotion is not None and promotion.method == "statute_reference")
                                        else _statute_notes(promotion) if (promotion is not None and promotion.statute) else bundle.notes), budget=answer_budget(self.config.generation_budget), answer_shape=shape,
                                 question_analysis=analysis)
                   if coverage is not None and selection is not None else None)
        if package is not None and self.config.answer_style != "restate":
            package = replace(package, answer_style=self.config.answer_style)
        if package is not None and promotion is not None and promotion.statute:
            from regulus.retrieval.statute import statute_index

            st = statute_index(self.retriever.db)
            package = replace(package, resolved=tuple(f"FD&C Act § {s} [21 U.S.C. {st.usc(s)}] {st.heading(s)}"
                                                      for s in promotion.statute))
        if package is not None and followup is not None and followup.earlier_question:
            package = replace(package, earlier_question=followup.earlier_question)
        if package is not None and precision_fix("ST2") and package.sources:
            # ST2: the dated status of a supplied provision (stayed, program closed, rule never issued) in the prompt
            from regulus.rag.status_notes import supplied_status_notes

            st_notes = [n["text"] for n in supplied_status_notes(package.sources)]
            if st_notes:
                package = replace(package, notes=tuple(package.notes) + tuple(st_notes))
        if package is not None and precision_fix("TK1") and package.answer_style == "apply":
            # TK1: an answer applying provisions to the question's facts gets room to finish (27 Oct: 70.55-Q1 cut at 256
            # and 740.2-Q1 at 416 tokens, both withheld as truncated); a short answer still stops when it is done
            package = replace(package, max_new_tokens=min(max(package.max_new_tokens, TK1_MIN_TOKENS),
                                                          package.budget.hard_max_tokens))
        if package is not None and precision_fix("FX1") and package.answer_style == "apply" and package.sources:
            # FX1: a verified example of a complete answer to a similar question about another provision (form only)
            try:
                from regulus.rag import answer_examples as AX
                from regulus.retrieval.paragraph_dense import paragraph_dense

                pd = paragraph_dense(self.retriever.db) if getattr(self.retriever, "db", None) is not None else None
                ex = AX.choose(getattr(pd, "embedder", None), question, AX.covered(package))
                if ex is not None:
                    package = replace(package, example={"id": ex["id"], "question": ex["question"], "covers": ex["covers"]})
            except Exception:  # noqa: BLE001 - an aid only
                pass
        if package is not None and precision_fix("LM1") and len(package.sources) >= LM1_MIN_SOURCES:
            closest = self.closest_sources(question, package)
            if closest:
                package = replace(package, closest=closest)   # recorded in the package's to_dict
        if package is not None and promotion is not None:
            # SP2: the picker's second regulation is further evidence; an answer that leaves it out is not withheld. The
            # same for the sections MS1/MS2 ("reranked: a, b") and AR2 ("absence retrieval: a") add (28 Oct: F-020's
            # MS1 companion 117.405 was made a requirement, the draft failed on it and only 117.410 was quoted)
            added = [n.split(": ", 1)[1] for n in promotion.notes
                     if n.startswith(("second pick: ", "reranked: ", "absence retrieval: ", "navigated: ", "cross-part: ",
                                      "sub-questions: ", "paragraph companions: "))]
            extra = frozenset(f"21 CFR {s.strip()}" for group in added for s in group.split(",") if s.strip())
            if extra:
                # an added section none of whose text reached the context is not a requested provision at all: listed,
                # prompt rule 5 has the model announce it "not found in the available eCFR data" (28 Oct, client keys:
                # L-003 PASS-level answer marked down for an "irrelevant 101.3 remark", S-020 "implies 111.255 and
                # 111.140 were not found", G-010 "207.49: not found")
                unsupplied = {rid for rid, canon in package.requirements if canon in extra and rid not in package.satisfied_ids}
                package = replace(package, optional_labels=extra,
                                  requirements=tuple(r for r in package.requirements if r[0] not in unsupplied),
                                  missing_labels=tuple(m for m in package.missing_labels if m not in extra))
                if precision_fix("OC1"):
                    context_only = frozenset(f"21 CFR {s.strip()}" for n in promotion.notes if n.startswith(OC1_NOTES)
                                             for s in n.split(": ", 1)[1].split(",") if s.strip())
                    package = companions_as_context(package, plan, context_only)
        normalization: dict[str, Any] | None = None
        # Failure observatory: the stages of the text, kept separately so that a
        # later step can never hide what an earlier one produced. None when no model call.
        raw_model_output: str | None = None
        repair_input: str | None = None
        repair_output: str | None = None
        timings["package_ms"] = round((time.perf_counter() - t_pkg) * 1000, 2)
        validation_initial: ValidationReport | None = None
        validation_final: ValidationReport | None = None
        repair: str | None = None
        generation_info: dict[str, Any] = {}
        partial = self.config.incomplete_behavior == "partial"
        incomplete = coverage is not None and coverage.state == STATE_INCOMPLETE
        unsupported = incomplete and partial and not coverage.satisfied_ids
        coverage_action = None
        # Stage 5 answerability gate: decided from the plan, the question analysis, coverage and the
        # selected evidence, before any prompt is built. Computed in both modes (so "off" runs record
        # what the gate would have decided); acted on only when answerability="gate".
        t_gate = time.perf_counter()
        gate_on = self.config.answerability == "gate"
        # Q1 (regulus.rag.validator_switches): the provision the question concerns is identified - named by the question
        # or resolved for it - so its relevant paragraphs may stand in for an answer that cannot be shown
        # ... and identified with confidence: named by the question itself, or resolved by the conversation, an exact
        # color-additive name or a named FD&C Act section - not by the picker's choice among candidates, which can be
        # wrong (a figure was checked against 71.18 for a 71.4 question, recorded follow-ups, 26 Oct)
        provision_identified = bool(gate_on and package is not None
                                    and (promotion is None or promotion.method in _CONFIDENT_RESOLUTION) and (
            provision_is_identified(plan, package) or (promotion is not None and bool(promotion.statute) and not plan.requirements)))
        focus_quote_allowed = provision_identified and precision_fix("Q1")
        # Q2: the same quotation for a provision the picker chose (a 21 CFR section, the question naming none or only
        # its part), under a lead that does not claim it is the provision the question concerns
        # under a lead that does not claim it is the provision the question concerns - and only when the reranker also
        # judges it relevant (Q2_MIN_RERANK): the picker's choice alone quoted Red No. 40 for a Blue No. 1 question and an
        # unrelated provision for must-refuse and false-premise questions (26 Oct, 25 recorded withheld answers)
        picked_quote = bool(gate_on and not focus_quote_allowed and precision_fix("Q2") and package is not None
                            and promotion is not None and promotion.sections
                            and (promotion.method == "picker"
                                 or (promotion.method == "part_paragraphs" and precision_fix("DV2")))
                            and promotion.rerank is not None and promotion.rerank >= Q2_MIN_RERANK
                            and provision_is_identified(plan, package))
        focus_lead = FOCUSED_QUOTE_LEAD_PICKED if picked_quote else FOCUSED_QUOTE_LEAD
        focus_quote_allowed = focus_quote_allowed or picked_quote
        gate_analysis = analysis if analysis is not None else analyze_question(plan)
        unknown_guidance = None
        if precision_fix("GT1") and getattr(self.retriever, "db", None) is not None:
            from regulus.rag.nonexistence import guidance_title_missing

            try:
                unknown_guidance = guidance_title_missing(self.retriever.db, question)
            except Exception:  # noqa: BLE001 - an aid: the question is answered as before
                unknown_guidance = None
        premise_fact = None
        if precision_fix("NX2") and getattr(self.retriever, "db", None) is not None:
            from regulus.rag.nonexistence import subpart_statement

            try:
                premise_fact = subpart_statement(self.retriever.db, question)
            except Exception:  # noqa: BLE001 - an aid: the question is answered as before
                premise_fact = None
        decision = gate.decide(question, plan, gate_analysis, coverage=coverage, sources=bundle.sources, no_context=no_context,
                               unknown_guidance=unknown_guidance, premise_fact=premise_fact,
                               resolved_statute=(promotion.statute if promotion is not None and promotion.method == "named_statute" else ()),
                               general_knowledge=(scope_decision.reason if scope_decision is not None and scope_decision.status == GENERAL_KNOWLEDGE else None),
                               product_choices=({k: [list(c) for c in v] for k, v in promotion.choices}
                                                if promotion is not None and promotion.method == "product_clarification" else None))
        conditions = decision.blocking_conditions
        block = gate_on and decision.blocked and not (
            ("no_evidence" in conditions and self.config.no_context_behavior != "refuse")
            or ("requested_provision_not_found" in conditions and self.config.incomplete_behavior != "partial"))
        fallback_type: str | None = None
        fallback_reason: str | None = None
        # where the answer itself ends: the notes after it (what the evidence does not establish, an unanswered part, a
        # gap notice, status notes) close the answer, and an addition about the answer's content (ISO1) goes before them
        answer_body_end: int | None = None
        prompt_sent: str | None = None  # the user message actually sent to the model, for evidence capture
        timings["answerability_ms"] = round((time.perf_counter() - t_gate) * 1000, 2)
        composed = composed_report = None
        # the composers decided from the database text (QM1, FP1, RS1) also answer a question the gate blocked only for
        # want of the value it asks for ("How often does 820.20 require management review?"): that is what they say
        grounded_only = block and set(conditions) <= {"requested_value_absent"}
        if (package is not None and package.answer_style == "apply" and (not block or grounded_only) and package.sources
                and not (grounding_decision is not None and grounding_decision.refused)):
            from regulus.rag.color_listing import compose as compose_listing

            composed_type = FB_COMPOSED_LISTING
            text = None
            if promotion is not None and promotion.method == "practice_silence" and not block:
                # P1: a practice no cosmetic provision mentions, answered from the Act text (regulus.rag.practice_silence)
                from regulus.rag.practice_silence import compose as compose_silence

                text, composed_type = compose_silence(package, question, promotion.notes), FB_COMPOSED_SILENCE
            if not text and (qmsr_former or qmsr_unknown):
                from regulus.rag.qmsr import compose as compose_qmsr

                unused = None
                if precision_fix("QM2"):
                    from regulus.rag.qmsr import record_terms

                    # only a term the database confirms no live part 820 text uses is said to be unused
                    unused = [t for t in record_terms(question) if self.part_mentions("820", re.escape(t)) is False]
                text, composed_type = (compose_qmsr(package, question, qmsr_former, qmsr_unknown, unused_terms=unused),
                                       FB_COMPOSED_QMSR)
            if not text and qmsr_cert:
                from regulus.rag.qmsr import compose_certification

                text, composed_type = compose_certification(package, question), FB_COMPOSED_QMSR
            if not text and provision_identified and (precision_fix("FP1") or precision_fix("RS1")):
                text, composed_type = self.compose_from_identified(package, question, plan, promotion)
            if not text and precision_fix("D1"):
                # D1: a question asking what terms mean, answered from the supplied paragraphs that define them
                from regulus.rag.definitions import compose as compose_definitions

                text, composed_type = compose_definitions(package, question), FB_COMPOSED_DEFINITIONS
            if not text and not block and precision_fix("WC1"):
                # WC1: what a Part 740 warning or label must say, quoted verbatim (regulus.rag.warning_statements)
                from regulus.rag.fallback import FB_COMPOSED_WARNING
                from regulus.rag.warning_statements import compose as compose_warning

                text, composed_type = compose_warning(package, question), FB_COMPOSED_WARNING
            if not text and not block and precision_fix("RC1"):
                # RC1: the MoCRA small-business exemption decided by rule from FD&C Act 612's text
                from regulus.rag.mocra_rules import compose as compose_mocra

                text, composed_type = compose_mocra(package, question), "composed_mocra_rule"
            if not text and not block and precision_fix("LS1"):
                # LS1: "what must X contain / include / show" answered with the regulation's whole list
                from regulus.rag.list_answers import compose as compose_list

                text, composed_type = compose_list(package, question), FB_COMPOSED_LIST
            if not text and not block and precision_fix("WS1"):
                # WS1: an overview question about the promoted section's own subject, answered with the whole section
                from regulus.rag.list_answers import compose_section

                text, composed_type = compose_section(package, question), FB_COMPOSED_LIST
            if not text and not block:
                composed_type = FB_COMPOSED_LISTING
                text = compose_listing(package, question)
            if not text and precision_fix("C82") and not block:
                from regulus.rag.color_listing import compose_certification

                text = compose_certification(package, question)
            if text:
                report = validate_answer(text, package, None)
                # FP1 / RS1 answer one point of the provision on purpose: the shape-completeness findings (a requested
                # provision, a list item or a requirement group not represented) do not apply; every grounding check does
                valid = report.ok or (composed_type in (FB_COMPOSED_PREMISE, FB_COMPOSED_RECORDS, FB_COMPOSED_QMSR,
                                                        FB_COMPOSED_DEFINITIONS) and not (
                    {p.code for p in report.problems if p.severity == "error"} - _FOCUSED_COMPLETENESS_CODES))
                if not valid and composed_type == FB_COMPOSED_LIST and (precision_fix("LS2") or precision_fix("WS1")):
                    # LS2: the whole list of one section, verbatim, is complete for that section; a completeness finding
                    # about another provision ("subject to Part 820" planned as a request for 820.1(a)(1)) does not apply
                    listed = re.match(r"Under 21 CFR (\S+?),", text)
                    valid = listed is not None and not [
                        p for p in report.problems if p.severity == "error" and not (
                            p.code in _FOCUSED_COMPLETENESS_CODES
                            and not re.match(rf"21 CFR {re.escape(listed.group(1))}(?![\d.])", p.detail or ""))]
                generation_info["listing_composer"] = {"composed": True, "valid": valid, "codes": list(report.codes()),
                                                       "kind": composed_type}
                if valid:
                    composed, composed_report = (text, composed_type), report
        if grounding_decision is not None and grounding_decision.refused:
            answer_text = grounding_decision.answer
            timings["generation_ms"] = 0.0
        elif premise is not None and premise.unsupported:
            # the question assumes a subpart, or a named thing in a part, that the database does not contain
            answer_text, fallback_type = premise.answer, FB_PREMISE_NOT_ESTABLISHED
            fallback_reason = f"{premise.kind}: {', '.join(premise.missing)}"
            coverage_action = "refuse_unsupported_premise"
            timings["generation_ms"] = 0.0
        elif block and composed is not None:
            # a value-absent block answered by a composer decided from the database text (QM1, FP1, RS1)
            answer_text, fallback_type = composed
            validation_initial = validation_final = composed_report
            fallback_reason = f"{decision.reason}; answered from the database text ({fallback_type})"
            coverage_action = "answer"
            timings["generation_ms"] = 0.0
        elif block:
            # the model is not called: nothing it could say would be supported by this evidence
            answer_text, fallback_type = blocked_answer(decision, bundle.sources, bundle.notes,
                                                        coverage.missing_labels if coverage is not None else (),
                                                        plan=plan, package=package)
            if fallback_type == FB_PROVISION_NOT_FOUND:
                answer_text = self.nonexistent_provision_answer(question, coverage) or answer_text
            fallback_reason = decision.reason
            coverage_action = _BLOCK_ACTIONS[fallback_type] if fallback_type in _BLOCK_ACTIONS else f"blocked_{decision.state.lower()}"
            timings["generation_ms"] = 0.0
        elif analysis is not None and analysis.scope_status == SCOPE_OUT:
            # Stage 3: a named non-U.S. framework; decided from the question, the model is not asked
            answer_text = out_of_scope_answer(analysis.scope_reason)
            coverage_action = "refuse_out_of_scope"
            timings["generation_ms"] = 0.0
        elif unsupported:
            answer_text = (self.nonexistent_provision_answer(question, coverage)
                           or incomplete_coverage_answer(coverage.missing_labels))
            coverage_action = "refuse_unsupported"
            timings["generation_ms"] = 0.0
        elif no_context and self.config.no_context_behavior == "refuse":
            answer_text = NO_CONTEXT_ANSWER
            if bundle.notes:
                answer_text += " " + " ".join(bundle.notes)
            timings["generation_ms"] = 0.0
        elif composed is not None:
            # a color-additive listing question: answered from the listing's own paragraphs (rag.color_listing), validated
            answer_text, fallback_type = composed
            validation_initial = validation_final = composed_report
            fallback_reason = {
                FB_COMPOSED_PREMISE: "a figure the question attributes to the identified provision is not in its text; "
                                     "the paragraphs stating figures of that kind are quoted",
                FB_COMPOSED_RECORDS: "the identified section's full text has no record-keeping language; what it requires "
                                     "is quoted",
                FB_COMPOSED_DEFINITIONS: "the question asks what terms mean; the supplied paragraphs that define them are "
                                         "quoted",
                FB_COMPOSED_QMSR: "a former Quality System regulation section (or a non-existent ISO 13485 clause): the "
                                  "current part 820 text is quoted and the ISO 13485 clause on the subject named",
            }.get(fallback_type, "color additive listing: per-product use decided from the listing's uses paragraph, "
                                 "21 CFR 70.3(s)/(v) and 70.5(a); every paragraph quoted")
            coverage_action = "answer"
            timings["generation_ms"] = 0.0
        else:
            t2 = time.perf_counter()
            messages = build_grounded_messages(package) if package is not None else build_messages(question, bundle.text)
            prompt_sent = messages[-1]["content"]
            timings["prompt_ms"] = round((time.perf_counter() - t2) * 1000, 2)
            emit("generation_started", {"max_new_tokens": package.max_new_tokens if package is not None else None})
            on_text = (lambda text: emit("delta", {"text": text})) if on_event is not None else None
            t2 = time.perf_counter()
            if package is not None and precision_fix("FR1") and package.answer_style == "apply" and package.sources:
                # FR1 (FaithfulRAG-style, facts first): the model first lists what the sources state about the question,
                # each with its source number, and then answers from that list and the evidence
                facts = self._generate([messages[0], {**messages[-1], "content": messages[-1]["content"] + FR1_PROMPT}],
                                       FR1_MAX_TOKENS, None)
                listed = (facts.response or "").strip()
                if listed and _CITED_RE.search(listed):
                    messages = list(messages[:-1]) + [{**messages[-1], "content": messages[-1]["content"]
                                                       + "\n\nFACTS THE SOURCES STATE ABOUT THE QUESTION (listed first):\n"
                                                       + listed + FR1_ANSWER}]
                    prompt_sent = messages[-1]["content"]
                    generation_info["facts_first"] = {"facts_tokens": facts.generated_tokens}
            gen = self._generate(messages, package.max_new_tokens if package is not None else None, on_text)
            if (package is not None and package.answer_style == "apply" and (package.satisfied_ids or package.resolved)
                    and package.sources and _BARE_REFUSAL_RE.match(gen.response or "")):
                # a bare decline although the provision the question is about was supplied (20 of 29 general questions
                # on Qwen3-4B-Instruct-2507 before promotion, 2 of 12 after): ask once more, saying so
                messages = ([{"role": "system", "content": APPLY_RETRY_SYSTEM_PROMPT}] + messages[1:-1]
                            + [{**messages[-1], "content": messages[-1]["content"] + REFUSAL_RETRY_NOTE}])
                prompt_sent = messages[-1]["content"]
                gen = self._generate(messages, package.max_new_tokens, on_text)
                generation_info["refusal_retry"] = True
            timings["generation_ms"] = round((time.perf_counter() - t2) * 1000, 2)
            emit("generation_finished", {"generated_tokens": gen.generated_tokens, "finished": gen.finished})
            # Model-internal breakdown, when the generator measured it (None-safe for fakes)
            for key, attr in (("model_prepare_ms", "prepare_seconds"), ("model_first_token_ms", "first_token_seconds"),
                              ("model_detokenize_ms", "detokenize_seconds")):
                value = getattr(gen, attr, None)
                if value is not None:
                    timings[key] = round(value * 1000, 2)
            if getattr(gen, "first_token_seconds", None) is not None:
                timings["model_decode_ms"] = round((gen.seconds - gen.first_token_seconds) * 1000, 2)
            raw_model_output = gen.response  # exactly as generated, before any transformation
            answer_text = gen.response.strip()
            # Stage 8: a sentence-repetition LOOP (3+ copies, citation markers ignored) is left intact so validation
            # rejects it as degenerate and the repair applies (verbatim provision, or withheld). Keeping the text
            # before the loop was measured and rejected: for S5-061 and BM-S-001 it was not a valid answer, and
            # lexical validation could not tell. An identical sentence said twice in a non-looping answer is said once.
            duplicates_removed = 0
            if not repeated_sentence(answer_text):
                answer_text, duplicates_removed = drop_duplicate_sentences(answer_text, _quote_sources(package))
            # requirement-id labels ("r1:") and empty list items copied from the prompt's layout: presentation only
            answer_text, plan_labels_removed = strip_plan_labels(answer_text)
            if analysis is not None:
                # Stage 3: presentation fixes before validation (markers onto their sentence, no echoed question)
                answer_text, normalization = normalize_answer(answer_text, question)
            # a marker attached to words that exactly ONE supplied source states verbatim, and its cited source
            # does not, is renumbered to that source (the model was measured to cite by paragraph number)
            repointed = []
            if package is not None:
                answer_text, repointed = repoint_citations(answer_text, [(s.index, s.text) for s in package.sources])
                if precision_fix("A4"):
                    answer_text, named_repointed = repoint_named_citations(answer_text, package)
                    if named_repointed:
                        generation_info["named_citations_repointed"] = named_repointed
            model_called = True
            input_tokens, output_tokens = gen.prompt_tokens, gen.generated_tokens
            generation_info = {**generation_info, "generated_tokens": gen.generated_tokens, "finished": gen.finished, "prompt_tokens": gen.prompt_tokens,
                               "stop_reason": getattr(gen, "stop_reason", None)}
            if getattr(gen, "first_token_seconds", None) is not None:
                generation_info["tokens_per_second"] = round(gen.tokens_per_second, 3)
                generation_info["decode_tokens_per_second"] = (round(gen.decode_tokens_per_second, 3)
                                                                if gen.decode_tokens_per_second is not None else None)
            if duplicates_removed:
                generation_info["duplicate_sentences_removed"] = duplicates_removed
            if plan_labels_removed:
                generation_info["plan_labels_removed"] = plan_labels_removed
            if repointed:
                generation_info["citations_repointed"] = [r.to_dict() for r in repointed]
            if package is not None and self.config.validation_behavior != "off":
                t3 = time.perf_counter()
                validation_initial = validate_answer(answer_text, package, gen)
                if self.config.validation_behavior == "repair" and not validation_initial.ok and precision_fix("A3"):
                    # unsupported filler sentences dropped whole, the rest validated again before any rewrite (A3)
                    excised = excise_unsupported_sentences(answer_text, validation_initial, package.question)
                    if excised is not None:
                        trimmed_report = validate_answer(excised[0], package, gen)
                        if trimmed_report.ok:
                            answer_text, validation_initial = excised[0], trimmed_report
                            generation_info["validator_precision"] = {"sentences_excised": excised[1]}
                validation_final = validation_initial
                error_codes = list(dict.fromkeys(p.code for p in validation_initial.problems if p.severity == "error"))
                if (self.config.validation_behavior == "repair" and not validation_initial.ok
                        and package.answer_style == "apply" and precision_fix("CX1") and len(error_codes) >= CX1_MAX_CODES):
                    # CX1: a draft with this many different errors is not handed back - the second generation is skipped
                    generation_info["correction_pass"] = {"attempted": False, "skipped": "CX1", "codes_before": error_codes}
                elif (self.config.validation_behavior == "repair" and not validation_initial.ok
                        and package.answer_style == "apply"):
                    corrected, info = self.correction_pass(messages, answer_text, package, validation_initial, question, analysis)
                    generation_info["correction_pass"] = info
                    if corrected is not None:
                        # delivered: the corrected draft, and every count below describes it. The first draft is kept as
                        # raw_model_output (never sent to the page) and by its problem codes; validation.initial is the
                        # corrected draft's report, so the accuracy evaluator never reads draft-1 sentences as text that
                        # post-processing removed from the delivered answer (code review)
                        answer_text, gen2, validation_initial = corrected
                        validation_final = validation_initial
                        info.update(first_draft_tokens=gen.generated_tokens, first_draft_finished=gen.finished)
                        generation_info.update(generated_tokens=gen2.generated_tokens, finished=gen2.finished,
                                               prompt_tokens=gen2.prompt_tokens, stop_reason=getattr(gen2, "stop_reason", None))
                        input_tokens, output_tokens = gen2.prompt_tokens, gen2.generated_tokens
                        gen = gen2
                if (precision_fix("RS2") and not precision_fix("BN2") and not validation_final.ok and package.answer_style == "apply"
                        and self.config.validation_behavior == "repair" and package.sources
                        and not (generation_info.get("correction_pass") or {}).get("accepted")):
                    # RS2 (rescue): the plain draft and its correction failed validation - a facts-first draft is tried
                    # once before the answer is quoted or withheld; it is delivered only if it validates
                    alt = self.facts_first_answer(messages, package, question, analysis)
                    generation_info["rescue"] = {"tried": True, "ok": bool(alt and alt[2].ok)}
                    if alt is not None and alt[2].ok:
                        answer_text, gen, validation_initial = alt
                        validation_final = validation_initial
                        generation_info.update(generated_tokens=gen.generated_tokens, finished=gen.finished)
                if (precision_fix("BN2") and not precision_fix("FR1") and package.answer_style == "apply"
                        and self.config.validation_behavior == "repair" and package.sources):
                    # BN2 (best of two with a verifier): a facts-first draft as well; the one that validates and scores
                    # better on answer_score is delivered, the plain draft on a tie - and a facts-first draft that
                    # validates replaces a plain one that would be withheld
                    alt = self.facts_first_answer(messages, package, question, analysis)
                    choice = "plain"
                    if alt is not None and alt[2].ok and (
                            not validation_final.ok or answer_score(alt[0], package) > answer_score(answer_text, package)):
                        answer_text, gen, validation_initial = alt
                        validation_final = validation_initial
                        generation_info.update(generated_tokens=gen.generated_tokens, finished=gen.finished)
                        choice = "facts_first"
                    generation_info["best_of_two"] = {"chosen": choice, "alt_ok": bool(alt and alt[2].ok)}
                if (self.config.validation_behavior == "repair" and not validation_final.ok
                        and not (generation_info.get("correction_pass") or {}).get("accepted")):
                    repair_input = answer_text
                    r = self.repair_with_fallback(answer_text, package, validation_initial, gen, question=question, plan=plan,
                                                  shape=shape, analysis=gate_analysis, sources=bundle.sources,
                                                  focus_quote=focus_quote_allowed, focus_lead=focus_lead)
                    answer_text, repair, repair_output, validation_final = r["text"], r["repair"], r["repair_output"], r["final"]
                    fallback_type, fallback_reason = r["fallback_type"], r["fallback_reason"]
                timings["validation_ms"] = round((time.perf_counter() - t3) * 1000, 2)
                if gate_on and fallback_type is None and _BARE_REFUSAL_RE.match(answer_text or ""):
                    # A general question the model declined although the CFR definitions of its terms were
                    # retrieved (lexical step 5c): quote them, cited, and say they are not applied. Kept only if
                    # it validates; otherwise the model's own decline stands.
                    keys = {c.result.key for c in bundle.sources if (c.result.metadata or {}).get("leg") == "defined_terms"}
                    indices = tuple(src.index for src in package.sources if src.key in keys)
                    if indices:
                        quoted = definitions_not_applied(package, indices)
                        report = validate_answer(quoted, package, gen)
                        if report.ok:
                            answer_text, validation_final = quoted, report
                            fallback_type = FB_DEFINITIONS_QUOTED
                            fallback_reason = ("the model declined; the CFR definitions of the question's terms are "
                                               "quoted verbatim and not applied")
                    if fallback_type is None and focus_quote_allowed:
                        # Q1: declined although the provision the question concerns was supplied: its relevant
                        # paragraphs, verbatim (grounded by construction, as above)
                        quoted = focused_quote(package, question, lead=focus_lead)
                        if quoted:
                            answer_text, validation_final, fallback_type = quoted, validate_answer(quoted, package, gen), FB_FOCUSED_QUOTE
                            fallback_reason = "the model declined; the governing provision is quoted instead"
                if (fallback_type is None and precision_fix("CV1") and validation_final is not None and validation_final.ok
                        and package.answer_style == "apply"):
                    verified, cv_info = self.verify_claims(messages, answer_text, package, question, analysis)
                    generation_info["claim_verification"] = cv_info
                    if verified is not None:
                        answer_text, validation_final = verified
                if fallback_type is None and precision_fix("LC1") and validation_final is not None and validation_final.ok:
                    # LC1: a color-listing answer keeps every paragraph of its listing (regulus.rag.color_listing)
                    from regulus.rag.color_listing import missing_listing_paragraphs

                    rest = missing_listing_paragraphs(package, answer_text)
                    if rest:
                        answer_text = answer_text.rstrip() + "\n\n" + rest
                        generation_info["listing_paragraphs_appended"] = rest.count("\n")
                if (fallback_type is None and precision_fix("PC1") and validation_final is not None and validation_final.ok
                        and package.answer_style == "apply" and not getattr(package, "earlier_question", None)):
                    # PC1: the uncited paragraphs of the answer's one requested section that bear on the question
                    rest = self.relevant_uncited(answer_text, package, question)
                    if rest:
                        answer_text = answer_text.rstrip() + "\n\n" + rest
                        generation_info["relevant_paragraphs_appended"] = rest.count("\n")
                if fallback_type is None and precision_fix("ND1"):
                    # ND1: the bare decline sentence inside an answer that states what its cited sources say
                    # contradicts it (9 of 154 answers on 26 Oct, each scored down for "a contradictory 'not enough
                    # information' line"); a scoped one ("... to answer whether ...") is a limit and stays
                    answer_text, stray = drop_stray_decline(answer_text)
                    if stray:
                        generation_info["stray_decline_removed"] = stray
            answer_body_end = len(answer_text.rstrip())
            if gate_on and decision.limitation and not (analysis is not None and analysis.scope_status == SCOPE_MIXED):
                # Stage 5 PARTIALLY_ANSWERABLE: say what the evidence does not establish (mixed scope in
                # "structured" mode already gets its own note below)
                answer_text = answer_text.rstrip() + "\n" + decision.limitation
            if incomplete and partial:
                # the extractive repair already states each missing provision as not found
                already = all(f"{label}: not found" in answer_text for label in coverage.missing_labels)
                if not already:
                    answer_text = (answer_text + " " if answer_text else "") + unresolved_provisions_statement(coverage.missing_labels)
                coverage_action = "answer_partial"
            elif coverage is not None:
                coverage_action = "answer"
            if analysis is not None and analysis.scope_status == SCOPE_MIXED and analysis.out_of_scope_part:
                answer_text = answer_text.rstrip() + "\n" + mixed_scope_note(analysis.out_of_scope_part)
        if answer_body_end is None:
            answer_body_end = len((answer_text or "").rstrip())
        # P1 (2026-10-20): paragraphs of a requested provision that the hard ceiling kept out of the prompt are named
        # in the delivered answer and recorded; the prompt and the model's text are unchanged.
        evidence_gap = None
        cut = ([d for d in selection.dropped if d.required and d.evidence.corpus in ("ecfr", "usc")]
               if (package is not None and selection is not None) else [])
        if premise is not None and premise.unsupported:
            cut = []   # that answer comes from the whole part in the database, not from the evidence the ceiling cut
        if fallback_type == FB_PRODUCT_CLARIFICATION:
            cut = []   # AMB1: a clarifying question answers nothing from the evidence, so nothing of it was left out
        if cut:
            requested = dict(package.requirements)
            left_out = Counter(next((requested[r] for r in d.evidence.satisfies if r in requested),
                                    (f"FD&C Act § {d.evidence.result.metadata.get('fdca_section')}" if d.evidence.corpus == "usc"
                                     else d.evidence.citation)) for d in cut)
            evidence_gap = {"reason": "hard_ceiling", "left_out": dict(left_out), "withheld": [d.evidence.citation for d in cut],
                            "notice_appended": bool(answer_text)}
            if answer_text:
                answer_text = answer_text.rstrip() + "\n" + withheld_evidence_statement(dict(left_out))
        if package is not None and answer_text and fallback_type != FB_UNVALIDATED and precision_fix("S1"):
            # S1: the dated, sourced status of a cited provision that its codified text does not carry (710/720
            # program closed, 740.18 stayed, no 606 rule), appended verbatim after validation
            from regulus.rag.status_notes import status_notes

            notes = status_notes(answer_text, package.sources)
            if notes:
                answer_text = answer_text.rstrip() + "\n\n" + "\n".join(n["text"] for n in notes)
                generation_info["status_notes"] = [n["id"] for n in notes]
        if package is not None and answer_text and fallback_type != FB_UNVALIDATED and precision_fix("ISO1"):
            # ISO1: the ISO 13485 clauses the cited Part 820 provisions name, stated with their citations
            from regulus.rag.iso_clauses import clause_line

            body, closing = answer_text[:answer_body_end].rstrip(), answer_text[answer_body_end:]
            line = clause_line(body, package.sources)
            if line:
                # after the answer, before its closing notes (30 Oct: the line followed "Not answered: ..." and the
                # PARTIALLY_ANSWERABLE limitation, which must end the answer)
                answer_text = body + "\n\n" + line + closing
                generation_info["iso_clauses_named"] = line.count("Clause ")
        t_post = time.perf_counter()
        if package is not None and model_called:
            addressed = validation_final.addressed if validation_final is not None else ()
            citations = build_answer_citations(answer_text, package, addressed)
        elif package is not None and fallback_type is not None and _CITE_RE.search(answer_text):
            citations = build_answer_citations(answer_text, package)  # Stage 5: a fallback that quotes its provision cites it
        elif package is not None:
            citations = []  # deterministic answers cite nothing; the evidence is in the metadata
        else:
            citations = self.build_citations(bundle, answer_text)
        timings["citations_ms"] = round((time.perf_counter() - t_post) * 1000, 2)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        t_meta = time.perf_counter()
        out = RagAnswer(
            question=asked,
            answer=answer_text,
            citations=citations,
            retrieved_results=list(response.results),
            context=bundle,
            evidence=registry,
            coverage=coverage,
            selection=selection,
            package=package,
            validation=validation_final,
            evidence_snapshot=self._capture_evidence(question, bundle, package, coverage, prompt_sent),
            no_context=no_context,
            model_called=model_called,
            timings_ms=timings,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "grounding": grounding_decision.to_dict() if grounding_decision is not None else None,
                "premise_check": premise.to_dict() if premise is not None else None,
                "advanced": (selection.diagnostics.get("advanced") if selection is not None else None),
                "query_type": response.query_type.value,
                # Stage 1 plan and Stage 2 evidence registry (None for retrievers
                # without run_plan). The registry is the per-requirement record
                # Stage 3 will consume; the context builder is unchanged.
                "plan": plan.to_dict(),
                # Stage 7: the question's own plan when a reference lookup re-planned it as a content search
                "original_plan": original_plan.to_dict() if plan is not original_plan else None,
                "reference_lookup": list(reference_targets) or None,
                "scope_policy": scope_decision.to_dict() if scope_decision is not None else None,
                "semantic_fallback": response.diagnostics.get("semantic_fallback"),
                # Stage 12 answer shape: None when answer_shape="off", and on the
                # legacy path, which has no package for a shape to organise
                "answer_shape": shape.to_dict() if shape is not None and package is not None else None,
                # answer-quality Stage 3 ("structured" only; None otherwise)
                "question_analysis": analysis.to_dict() if analysis is not None else None,
                # failure observatory: raw generation and the text repair received (repair output = the answer)
                "raw_model_output": raw_model_output,
                "repair_input": repair_input,
                "repair_output": repair_output,
                # Stage 5 answerability gate and safe fallback
                "answerability": {**decision.to_dict(), "acted_on": gate_on},
                "generation_called": model_called,
                "generation_blocked": block,
                "block_reason": decision.reason if block else None,
                "fallback_type": fallback_type,
                "fallback_reason": fallback_reason,
                "normalization": normalization,
                "quality_flags": (quality_flags(answer_text, package, analysis,
                                                validation_final.to_dict() if validation_final is not None else None,
                                                coverage_action=coverage_action, model_called=model_called, question=question)
                                  if analysis is not None else None),
                "evidence": registry.to_dict() if registry is not None else None,
                # Stage 3 coverage and selection (None on the legacy path)
                "coverage": ({**coverage.to_dict(), "evidence_complete": evidence_gap is None}
                             if coverage is not None else None),
                "evidence_gap": evidence_gap,
                "selection": selection.to_dict() if selection is not None else None,
                "coverage_action": coverage_action,
                # Stage 4 generation contract and validation (None on the legacy path)
                "generation": ({**package.to_dict(), **generation_info, "prompt_chars": len(build_grounded_messages(package)[1]["content"])} if package is not None else None),
                "validation": ({"initial": validation_initial.to_dict(), "final": validation_final.to_dict(), "repair": repair} if validation_initial is not None else None),
                "citation_support": citation_support(answer_text, package) if package is not None and model_called else None,
                "contextual_only_excluded": sum(1 for r in response.results if r.metadata.get("contextual_only")) if self.config.exclude_contextual_only else 0,
                "retrieval_counts": response.counts,
                "retrieval_timings_ms": response.timings_ms,
                "retrieval": dict(response.diagnostics),
                "section_promotion": (promotion.to_dict() if promotion is not None
                                      else ({"error": promotion_error} if promotion_error else
                                            ({"named_companions": named_companions_added} if named_companions_added else None))),
                "config": self.config.to_dict(),
                # follow-ups: what this turn was resolved to (the next question's anchor) and how it was read
                "spelling": [list(x) for x in spelling] if spelling else None,
                "conversation": {"anchor": turn_anchor(own_plan, promotion, followup).to_dict(),
                                 "followup": followup.to_dict() if followup is not None else None},
            },
        )
        # diagnostics payload built after total_ms (it re-renders the prompt and serialises the registry)
        timings["metadata_ms"] = round((time.perf_counter() - t_meta) * 1000, 2)
        return out

    def answer_fn(self, question: str) -> RagGenerationResult:
        """Adapter for regulus.evaluation.run_evaluation (answer_fn signature)."""
        out = self.answer(question)
        return RagGenerationResult(
            prompt=question,
            response=out.answer,
            prompt_tokens=out.input_tokens or 0,
            generated_tokens=out.output_tokens or 0,
            seconds=out.timings_ms["total_ms"] / 1000.0,
            diagnostics=out.diagnostics(),
            citations=[c.to_dict() for c in out.citations],
        )

    def describe(self) -> dict[str, Any]:
        describe = getattr(self.retriever, "describe", None)
        retriever = describe() if callable(describe) else {"retriever": type(self.retriever).__name__}
        out = {"rag": self.config.to_dict(), "retriever": retriever, "generator": self.generator.describe()}
        if self.grounding is not None:
            out["grounding"] = self.grounding.describe()
        return out
