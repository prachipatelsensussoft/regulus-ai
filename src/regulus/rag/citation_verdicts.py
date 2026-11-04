"""CC1 (4 Nov, report-only): one verdict for every citation in the delivered answer.

    valid                        the cited source is the provision the sentence names (or names none) and states it
    exists_not_supporting        the source was supplied, but the sentence's wording, figure, condition or duty is not
                                 the cited source's
    wrong_section_or_programme   the sentence names a supplied provision and cites another one, or cites a part outside
                                 the programme the question names
    cannot_verify                no supplied source holds the citation: a marker with no source, a named provision or
                                 paragraph that was never supplied

The checks are the validator's own: `validate_answer` is re-run on the delivered text (deterministic, no side effects)
and each [n] is classified from the problems it raised for that sentence, the cited source's identity, and the share of
the sentence's content words the source states. Nothing is corrected and no answer changes: PE1 and CG1 are the switches
that act. A sentence the pipeline appends as a notice ("Evidence limit: ...", "Not found in the available eCFR data: ...")
names provisions that were deliberately not supplied and is not a claim, so it is skipped.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from regulus.rag import claim_validation as CV
from regulus.rag import validation as V
from regulus.retrieval.evidence import TIER_EXACT, TIER_STRUCTURAL
from regulus.retrieval.references import parse_cfr_references

VERSION = 1
VALID = "valid"
NOT_SUPPORTING = "exists_not_supporting"
WRONG = "wrong_section_or_programme"
UNVERIFIABLE = "cannot_verify"
VERDICTS = (VALID, NOT_SUPPORTING, WRONG, UNVERIFIABLE)

WEAK_OVERLAP = 0.25   # below this share of the sentence's content words in the cited source: not supported (the
                      # validator's own LOW_OVERLAP warning is 0.40; the gap is reported as "overlap" for calibration)
MIN_WORDS = 4         # a sentence with fewer content words is not judged on overlap (the validator's own floor)

_NOT_SUPPORTING_CODES = frozenset({CV.PARAGRAPH_MISATTRIBUTION, CV.FIGURE_MISATTRIBUTED, CV.CONDITION_NOT_IN_CITED,
                                   CV.OBLIGATION_STRENGTHENED, CV.FALSE_DENIAL})
_NOTICE = re.compile(r"^\s*(?:Evidence limit:|Not found in the available eCFR data:|The answer above (?:covers|does not cover))",
                     re.I)


def _part(section: str | None) -> str | None:
    return section.split(".")[0] if section else None


def _general_part(part: str | None) -> bool:
    """Subchapter A (Parts 1-99): general provisions that every programme may cite (as XG1 treats them)."""
    return bool(part) and part.isdigit() and int(part) < 100


def _family(named_path: str, have_path: str) -> bool:
    """The named paragraph is the supplied one, or its parent or child (both designated)."""
    return bool(named_path and have_path) and (named_path == have_path or named_path.startswith(have_path + "(")
                                               or have_path.startswith(named_path + "("))


def _provision_text(src, sources) -> str:
    """The text of the provision a marker cites: a citation names a provision, not a database row, so every supplied row
    with the same label (a section stored one definition or paragraph per row, 111.3) and every supplied child of the
    cited paragraph counts as the cited text (smoke check 4 Nov, S-002: "terms ... including batch, lot ... [1]" cited
    111.3's lead-in row while the terms are its sibling rows)."""
    ident = V._source_identity(src)
    parts = [src.text or "", getattr(src, "heading", "") or ""]
    for s in sources:
        if s is src:
            continue
        other = V._source_identity(s)
        same_label = s.citation == src.citation
        child = (ident is not None and other is not None and other[1] == ident[1]
                 and (not ident[2] or other[2].startswith(ident[2] + "(")))
        if same_label or child:
            parts.append(s.text or "")
    return " ".join(parts)


def _overlap(sentence: str, src, sources=()) -> float | None:
    words = V._content_words(V.MARK_RE.sub("", sentence))
    if len(words) < MIN_WORDS:
        return None
    return len(words & V._content_words(_provision_text(src, sources))) / len(words)


def citation_verdicts(answer: str, package, question: str | None = None) -> dict[str, Any]:
    """{"version", "items": [...], "summary": {...}}: one item per (sentence, cited source) and per provision a sentence
    names without a marker."""
    text = (answer or "").strip()
    out: dict[str, Any] = {"version": VERSION, "items": [], "summary": {}}
    if not text or package is None:
        out["summary"] = {"citations": 0, **{v: 0 for v in VERDICTS}}
        return out
    report = V.validate_answer(text, package)
    by_index = {s.index: s for s in package.sources}
    identities = {s.index: V._source_identity(s) for s in package.sources}
    programme = None
    if question:
        from regulus.retrieval.programmes import programme_for

        programme = programme_for(question, extended=True)
    problems_by_sentence: dict[str, list] = {}
    for p in report.problems:
        if p.sentence:
            problems_by_sentence.setdefault(p.sentence, []).append(p)

    items = out["items"]
    for k, sentence in enumerate(V.split_sentences(V.attach_trailing_markers(text))):
        if _NOTICE.match(sentence):
            continue
        marks = list(dict.fromkeys(int(m) for m in V.MARK_RE.findall(sentence)))
        named_text = V._without_cited_quotes(sentence, [by_index[i].text for i in marks if i in by_index])
        named = [r for r in parse_cfr_references(named_text) if r.section]
        named_supplied = [r for r in named if any(i and V._related_identity((r.title, r.section, r.paragraph_path), i)
                                                  for i in identities.values())]
        found = problems_by_sentence.get(sentence, [])
        for n in marks:
            item: dict[str, Any] = {"sentence": k, "marker": n, "reasons": []}
            src = by_index.get(n)
            if src is None:
                item.update(verdict=UNVERIFIABLE, reasons=["no supplied source has this number"])
                items.append(item)
                continue
            ident = identities.get(n)
            item.update(source=src.citation, tier=src.tier)
            reasons, verdict = item["reasons"], VALID
            codes = {p.code for p in found if p.citation in (None, n)}
            # 1. wrong section: the validator found a mismatch in this sentence and this source is not what it names
            for p in found:
                if p.code in (V.CITATION_MISMATCH, V.DISCOVERY_CITED_FOR_REQUIREMENT):
                    rid = p.requirement_id
                    if rid and not (rid in src.requirement_ids and src.tier in (TIER_EXACT, TIER_STRUCTURAL)):
                        verdict = WRONG
                        reasons.append(f"{p.code}: not the EXACT/STRUCTURAL source of {rid}")
                elif p.code == V.PROVISION_CITATION_MISMATCH:
                    if ident is None or not any(V._related_identity((r.title, r.section, r.paragraph_path), ident)
                                                for r in named_supplied):
                        verdict = WRONG
                        reasons.append(f"{p.code}: {p.detail}")
            # 2. wrong programme: the question names one programme and the source is another programme's part
            part = _part(ident[1]) if ident else None
            if (verdict == VALID and programme is not None and part and part not in programme.parts
                    and not _general_part(part)):
                verdict = WRONG
                reasons.append(f"part {part} is outside the {programme.name} programme the question names")
            # 3. cannot verify: the sentence names a paragraph, the cited source is its section only, and no supplied
            # source is that paragraph
            if verdict == VALID and ident and not ident[2]:
                for r in named_supplied:
                    if (r.section == ident[1] and r.paragraph_path
                            and not any(i and i[1] == r.section and _family(r.paragraph_path, i[2])
                                        for i in identities.values())):
                        verdict = UNVERIFIABLE
                        reasons.append(f"{r.canonical}: the paragraph is not among the supplied sources")
                        break
            # 4. exists but does not support
            bad = sorted(codes & _NOT_SUPPORTING_CODES)
            overlap = _overlap(sentence, src, package.sources)
            item["overlap"] = None if overlap is None else round(overlap, 3)
            if verdict == VALID and bad:
                verdict = NOT_SUPPORTING
                reasons += bad
            elif verdict == VALID and overlap is not None and overlap < WEAK_OVERLAP:
                verdict = NOT_SUPPORTING
                reasons.append(f"only {overlap:.0%} of the sentence's content words are in the cited source")
            item["verdict"] = verdict
            items.append(item)
        if not marks:
            # a provision named by its label alone ("21 CFR 111.70(e) requires ..."), as verbatim fallbacks cite
            for r in named:
                item = {"sentence": k, "marker": None, "named": r.canonical, "reasons": []}
                fam = [i for i in identities.values() if i and i[1] == r.section
                       and (not r.paragraph_path or _family(r.paragraph_path, i[2]))]
                if r in named_supplied and fam:
                    item["verdict"] = VALID
                elif r in named_supplied:
                    item["verdict"] = UNVERIFIABLE
                    item["reasons"].append("the section was supplied, this paragraph was not")
                else:
                    item["verdict"] = UNVERIFIABLE
                    item["reasons"].append("the named provision is not among the supplied sources")
                items.append(item)
    counts = Counter(i["verdict"] for i in items)
    total = len(items)
    out["summary"] = {"citations": total, **{v: counts.get(v, 0) for v in VERDICTS},
                      "valid_share": round(counts.get(VALID, 0) / total, 4) if total else None,
                      "programme": getattr(programme, "name", None)}
    return out


def safe_citation_verdicts(answer: str, package, question: str | None = None) -> dict[str, Any]:
    """citation_verdicts(), with any failure recorded instead of raised: CC1 is report-only and never breaks an answer."""
    try:
        return citation_verdicts(answer, package, question)
    except Exception as exc:  # noqa: BLE001 - report-only
        return {"version": VERSION, "error": f"{type(exc).__name__}: {exc}"[:300]}
