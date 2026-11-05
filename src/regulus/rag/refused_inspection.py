"""RI1 (5 Nov 2026): what follows when a drug or device establishment refuses, delays, denies or limits an FDA inspection.

300Q M-006 ("What does Part 820 provide with respect to foreign manufacturers that refuse to permit an FDA inspection?",
key: refusal is grounds for refusal of admission; PARTIALLY FAIL) was answered from 21 CFR 820.1(c) alone, which says an
imported device "is subject to refusal of admission ... under section 801(a) ... if ... it appears to be adulterated" but
not why refusing an inspection makes it so. That link is FD&C Act § 501: "A drug or device shall be deemed to be
adulterated- ... (j) ... If it is a drug or device and ... the owner, operator, or agent ... delays, denies, or limits an
inspection, or refuses to permit entry or inspection." The answer quotes exactly these, verbatim: § 501's opening and
501(j) for a drug or a device, and for a device 820.1(c) as well. Food, dietary supplement, cosmetic and tobacco
establishments are governed by other provisions and are never answered here. Section 801(a) itself is one 5,684-character
paragraph in the Act store, too long to supply, so a drug answer states the adulteration and does not claim more.
"""
from __future__ import annotations

import re

_REFUSED = re.compile(
    r"\b(?:refus\w*|den(?:y|ies|ied|ial|ying)|delay\w*|limit\w*|obstruct\w*|block\w*|not\s+(?:allow|permit)\w*|"
    r"(?:won'?t|will\s+not|does\s+not|do\s+not|did\s+not)\s+(?:allow|permit|let))\b[^?.]{0,60}\binspect\w*"
    r"|\binspect\w*\b[^?.]{0,40}\b(?:is\s+|was\s+|gets?\s+)?(?:refus\w*|denied|delayed|limited|obstructed|blocked)\b", re.I)
_DEVICE = re.compile(r"\b(?:medical\s+)?devices?\b|\bpart\s*820\b|\bQMSR\b", re.I)
_DRUG = re.compile(r"\bdrugs?\b|\bpharmaceutical\w*|\bactive\s+pharmaceutical\s+ingredients?\b|\bAPIs?\b", re.I)
_OTHER = re.compile(r"\bfoods?\b|\bdietary\s+supplements?\b|\bcosmetics?\b|\btobacco\b|\bfarms?\b|\bseafood\b|\bjuice\b", re.I)
COMPOSED_TYPE = "composed_inspection"
DEVICE_SECTION = "820.1(c)"
_SECTION_501 = re.compile(r"^FD&C Act § 501(?:\s|$)")
_SECTION_501J = re.compile(r"^FD&C Act § 501\(j\)")
_J_BODY = re.compile(r"If it is a drug or device and [^.]*?refuses to permit entry or inspection\.")


def asks_refused_inspection(question: str) -> str | None:
    """"device", "drug" or None: whether the question asks about an establishment that refuses, delays, denies or limits
    an FDA inspection, and for which product."""
    q = question or ""
    if not _REFUSED.search(q) or _OTHER.search(q):
        return None
    if _DEVICE.search(q):
        return "device"
    if _DRUG.search(q):
        return "drug"
    return None


def compose(package, question: str, kind: str | None) -> str | None:
    """The RI1 answer from the supplied evidence, verbatim, or None when § 501's opening, 501(j) or (for a device)
    21 CFR 820.1(c) is not supplied."""
    from regulus.rag.generation import _verbatim, whole_sentences

    if package is None or kind not in ("device", "drug"):
        return None
    srcs = [s for s in package.sources if not s.is_docket]
    lead = next((s for s in srcs if _SECTION_501.match((s.citation or "").strip())
                 and (s.text or "").strip().startswith("A drug or device shall be deemed to be adulterated")), None)
    j = next((s for s in srcs if _SECTION_501J.match((s.citation or "").strip()) and _J_BODY.search(s.text or "")), None)
    reg = next((s for s in srcs if (s.citation or "").strip() == f"21 CFR {DEVICE_SECTION}"), None)
    if lead is None or j is None or (kind == "device" and (reg is None or not whole_sentences(reg.text or ""))):
        return None
    lines = ["FD&C Act § 501 makes a drug or device adulterated when its establishment delays, denies, or limits an "
             "inspection, or refuses to permit entry or inspection, verbatim:",
             f"{lead.citation} (FD&C Act, binding, verbatim): {(lead.text or '').strip()} [{lead.index}]",
             f"{j.citation} (FD&C Act, binding, verbatim): {_J_BODY.search(j.text).group(0)} [{j.index}]"]
    if kind == "device":
        lines += ["For an imported device, 21 CFR 820.1(c) provides, verbatim:",
                  f"{reg.citation} (eCFR, binding, verbatim): {_verbatim(whole_sentences(reg.text or ''), reg.index)}"]
    return "\n".join(lines)
