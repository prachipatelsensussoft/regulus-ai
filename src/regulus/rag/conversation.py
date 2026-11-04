"""Follow-up questions: a question asked later in a conversation is resolved against the provisions the conversation
is about.

Why (26 Oct 2026, cosmetics question bank, 1,385 follow-ups asked "in the same session"): 74% name no provision
("Can we use it in an eye shadow?", "An FDA district office manager can sign the suspension order, right?"), and the
verified key of 84% cites the section of the question that opened the conversation. Answered on its own, such a
follow-up reaches section resolution with no subject; answered in context it is a question about a known section.

How:
  * every answered turn records its ANCHOR: the provisions it was resolved to - the promoted section(s) or FD&C Act
    section(s), or the 21 CFR sections the question named - never their companions (those are re-derived for each
    question from its own words: "eye shadow" brings 70.5(a));
  * a later question is ANCHORED to the most recent anchor when it depends on the conversation - a pronoun or a
    pointer ("it", "this section", "paragraph (b)"), an elliptical or tag form ("What about ...", "..., right?"), or a
    subject word it shares with the earlier questions or the anchored sections' headings - and does not point
    elsewhere itself: a question naming its own provision, an FD&C Act section, or a color additive the anchored
    sections do not list is answered as it would be on its own;
  * an anchored question is promoted to the anchor (regulus.rag.section_promotion.SectionPromoter.anchored, method
    "conversation") instead of being ranked, and "paragraph (b)" / "this section" are written out as the anchored
    section ("21 CFR 74.2708(b)");
  * the model is shown the earlier question as context only (regulus.rag.generation.render_package).

Nothing here runs without a ConversationContext: a first question is answered byte-identically to a question asked
on its own. Every decision is made from text and the anchor, never from similarity or a model call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# how many earlier turns are read: the anchor comes from the most recent one that has one
MAX_TURNS = 4
# the earlier question shown to the model is cut to this many characters (a question, never an answer)
EARLIER_QUESTION_CHARS = 400

_SECTION = re.compile(r"^\d{1,4}\.\d{1,4}[a-z]?$")
_PART = re.compile(r"^\d{1,4}$")
_STATUTE = re.compile(r"^\d{3}[a-z]?$")
# words that point back into the conversation
# ("that", "their", "they" are left out: "FDA requires that ...", "do manufacturers list their ingredients" point nowhere)
_PRONOUN = re.compile(r"\b(?:it|its|itself|this|these|those|such|the\s+same|the\s+former|the\s+latter)\b(?!\s+(?:is|are)\s+not\b)",
                      re.I)
_ELLIPTICAL = re.compile(r"^\s*(?:and|but|so|also|what\s+about|how\s+about|what\s+if|and\s+if|then|or|plus)\b", re.I)
# FU2: a demonstrative with a word for what the conversation is about points back however long the question is
# ("Our formula adds silica abrasive. Does that affect use of this color?" was read as a new topic, 26 Oct)
_DEMONSTRATIVE = re.compile(r"\b(?:this|that|these|those|the\s+same)\s+(?:colou?r(?:\s+additive)?|colou?rant|pigment|dye|lake|"
                            r"batch|petition|listing|regulation|product|ingredient|exemption|order|facility|requirement|"
                            r"filing|form|statement|warning|caution|substance|material|use|limit|tolerance)s?\b", re.I)
_TAG = re.compile(r"(?:,\s*|\s)(?:right|correct|true|yes|no|isn't\s+it|don't\s+we|doesn't\s+it|is\s+that\s+so)\s*\?\s*$", re.I)
# a provision pointed at without its section: written out as the anchored section
_PARAGRAPH_POINTER = re.compile(r"\b(?:paragraph|subsection|clause)\s*((?:\((?:[a-z0-9]{1,4}|[ivxl]{5,8})\))+)", re.I)
_THIS_PROVISION = re.compile(r"\b(?:this|that|the\s+same)\s+(?:section|provision|regulation|rule)\b", re.I)
_WORD = re.compile(r"[a-z][a-z0-9&'-]+")
# not a subject: question words, function words, and words every question about the regulations shares
_STOP = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between both but
by can could did do does doing done down during each either else ever every few for from further had has have having
he her here hers him his how i if in into is it its itself just may me might more most must my no nor not now of off
on once only or other our ours out over own per same shall she should so some still such than that the their them
then there these they this those through to too under until up upon us very was we were what when where whether which
while who whom why will with within without would yes you your yours
right correct true okay ok please tell explain mean means meaning say says said apply applies applied applicable
need needs needed require requires required requirement requirements rule rules regulation regulations regulatory
section sections paragraph paragraphs part parts subpart provision provisions cfr fda fd&c act law federal agency
code title u.s us usc question answer case example thing things way something anything everything someone anyone
company firm business we're we've i'm it's doesn't don't isn't aren't can't won't didn't
cosmetic cosmetics product products use used using uses make makes made sell sells selling sold also still even
""".split())


def _stem(w: str) -> str:
    w = w.strip("'-")
    if _fu2("FS1"):
        # FS1 (27 Oct): "filing", "file" and "filed" are one word, as are "packaging" / "package" and "supplier's" /
        # "supplier" - the suffix rule below kept "filing" whole and "file" with its "e"
        w = re.sub(r"'s$", "", w)
        for suf in ("ings", "ing", "ies", "ied", "es", "ed", "s"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                w = w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
                break
        return w[:-1] if len(w) >= 4 and w.endswith("e") else w
    for suf in ("ings", "ing", "ies", "ied", "es", "ed", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def subject_words(text: str, stem=None) -> set[str]:
    """The question's subject words: lower-cased, stemmed (`stem`, else a crude suffix rule), without function words
    and the words every regulatory question shares ("cosmetic", "require", "FDA")."""
    stem = stem or _stem
    return {stem(w.strip("'-")) for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2} - _STOP


def snowball_stemmer(db, texts: Iterable[str]):
    """FS2 (27 Oct): the Snowball English stemmer of PostgreSQL full-text search ('english_stem', the stemmer the
    retrieval index uses), for the words of `texts`: "filing", "file" and "filed" are one subject word, as are
    "packaging" / "package" - the suffix rule kept them apart. A word the database does not stem keeps the suffix
    rule. -> a function word -> stem."""
    words = sorted({w.strip("'-") for t in texts for w in _WORD.findall((t or "").lower())} - {""})
    rows = db.query("SELECT w, (ts_lexize('english_stem', w))[1] AS s FROM unnest(%(ws)s::text[]) w", {"ws": words}) if words else []
    stems = {r["w"]: r["s"] for r in rows if r.get("s")}
    return lambda w: stems.get(w) or _stem(w)


@dataclass(frozen=True)
class Anchor:
    """The provisions a turn was resolved to: 21 CFR sections ("74.2708"), FD&C Act sections ("607"), or - for a
    question about a whole part ("What does Part 11 cover?") - 21 CFR parts ("11")."""
    cfr: tuple[str, ...] = ()
    statute: tuple[str, ...] = ()
    parts: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.cfr or self.statute or self.parts)

    def labels(self) -> list[str]:
        return ([f"21 CFR {s}" for s in self.cfr] + [f"FD&C Act § {s}" for s in self.statute]
                + [f"21 CFR part {p}" for p in self.parts])

    def to_dict(self) -> dict[str, list[str]]:
        return {"cfr": list(self.cfr), "statute": list(self.statute), "parts": list(self.parts)}

    @classmethod
    def from_dict(cls, d: Any) -> "Anchor":
        """A stored anchor; anything malformed is dropped, never trusted (it comes back from the chat store)."""
        if not isinstance(d, dict):
            return cls()
        cfr = tuple(dict.fromkeys(s for s in (d.get("cfr") or []) if isinstance(s, str) and _SECTION.match(s)))[:6]
        st = tuple(dict.fromkeys(s for s in (d.get("statute") or []) if isinstance(s, str) and _STATUTE.match(s)))[:3]
        parts = tuple(dict.fromkeys(s for s in (d.get("parts") or []) if isinstance(s, str) and _PART.match(s)))[:2]
        return cls(cfr=cfr, statute=st, parts=parts)


@dataclass(frozen=True)
class PriorTurn:
    question: str
    anchor: Anchor = field(default_factory=Anchor)


@dataclass(frozen=True)
class ConversationContext:
    """Earlier answered turns of the same conversation, oldest first (at most MAX_TURNS are kept)."""
    turns: tuple[PriorTurn, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "turns", tuple(self.turns)[-MAX_TURNS:])

    def __bool__(self) -> bool:
        return bool(self.turns)

    @property
    def anchor(self) -> Anchor:
        return next((t.anchor for t in reversed(self.turns) if t.anchor), Anchor())

    @property
    def earlier_question(self) -> str | None:
        """The question that opened the current topic: the earliest turn with the current anchor (the conversation's
        own words for "it"), else the most recent question."""
        if not self.turns:
            return None
        a = self.anchor
        same = [t for t in self.turns if a and t.anchor == a]
        return (same[0] if same else self.turns[-1]).question

    @classmethod
    def from_records(cls, records: Iterable[tuple[str, Any]]) -> "ConversationContext":
        """(question, stored anchor dict or None) pairs, oldest first."""
        return cls(tuple(PriorTurn(q.strip(), Anchor.from_dict(a)) for q, a in records if isinstance(q, str) and q.strip()))


@dataclass(frozen=True)
class FollowUp:
    """How a question was read against its conversation."""
    anchored: bool                 # resolved to the conversation's anchor
    reason: str                    # why (or why not)
    question: str                  # the text the pipeline answers: the user's, with pointers written out
    anchor: Anchor = field(default_factory=Anchor)
    earlier_question: str | None = None   # shown to the model as context (None: not shown)
    cues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"anchored": self.anchored, "reason": self.reason, "question": self.question,
                "anchor": self.anchor.to_dict(), "earlier_question": self.earlier_question, "cues": list(self.cues)}


def cues(question: str) -> tuple[str, ...]:
    """The question's signs of depending on what came before."""
    q = question or ""
    out = []
    m = _PRONOUN.search(q)
    # a pronoun points back only when the question has not named a subject of its own before it: "Can we use it in an
    # eye shadow?" does; "Our batch is certified. What must we do with it next?" does not
    if m and len(subject_words(q[:m.start()])) <= 1:
        out.append("pronoun")
    if _ELLIPTICAL.search(q):
        out.append("elliptical")
    if _TAG.search(q):
        out.append("tag_question")
    if _PARAGRAPH_POINTER.search(q) or _THIS_PROVISION.search(q):
        out.append("pointer")
    m = _DEMONSTRATIVE.search(q)
    if m and _fu2("FU2"):
        # "... a color additive that is exempt ... Can FDA take that exemption away?" introduced its own referent
        noun = _stem(m.group(0).split()[-1].lower())
        before = {_stem(w) for w in _WORD.findall(q[:m.start()].lower())}
        if not any(noun.startswith(w) or w.startswith(noun) for w in before if len(w) > 3):
            out.append("demonstrative")
    return tuple(out)


def _fu2(fix: str) -> bool:
    from regulus.rag.validator_switches import enabled

    return enabled(fix)


def write_out_pointers(question: str, section: str, statute: bool = False) -> str:
    """"paragraph (b)" -> "21 CFR 74.2708(b)", "this section" -> "21 CFR 74.2708": the provision the conversation is
    about, named - its primary section (the first anchored: the promoted section, before a sibling listing such as
    82.1708). `statute`: an FD&C Act section ("FD&C Act § 607(b)")."""
    name = f"FD&C Act § {section}" if statute else f"21 CFR {section}"
    q = _PARAGRAPH_POINTER.sub(lambda m: f"{name}{m.group(1)}", question)
    return _THIS_PROVISION.sub(name, q)


# a question with at most this many subject words of its own cannot stand alone: a cue is then enough to anchor it
SHORT_QUESTION_WORDS = 5
# shared subject: at least this share of the question's own subject words, or this many, appear earlier in the
# conversation. One shared word is not enough: a new scenario question shares "color" or "label" with most earlier
# ones (dev follow-ups vs main questions of another part after the same parent: any shared word anchored 71% / 28%,
# this rule 59% / 3%)
SHARED_FRACTION = 0.2
SHARED_WORDS = 3
SHARED_MAX_OWN_WORDS = 12   # FU2C: 2 of 22 false anchors of a new main question removed, no correct anchor lost (harness, 27 Oct)
# a question with more subject words than this is self-contained: its own ranking alone does not anchor it
SELF_CONTAINED_WORDS = 10
# a fragment ("Is there an exemption?", "Any deadline?", "Who must sign?") names at most this many subject words and no
# provision: in a conversation it is about what came before. A definition question ("What is a cosmetic?") is not one.
FRAGMENT_WORDS = 2
SHORT_OWN_WORDS = 3   # FU2B: at 4, 9 of 1,000 new main questions asked mid-conversation were anchored (harness, 27 Oct)
_DEFINITION_FORM = re.compile(r"^\s*(?:what\s+is\s+an?\s|what\s+are\s+(?!the\b)\w|what\s+does\s+[^?]{1,60}\s+mean\b|"
                              r"define\b|(?:the\s+)?(?:definition|meaning)\s+of\b)", re.I)


def resolve(question: str, context: ConversationContext | None, *, names_provision: bool, other_entity: bool,
            anchor_ranked: bool = False, heading_words: set[str] | None = None, stem=None) -> FollowUp | None:
    """How `question` is read in `context`. None without a context (the question is answered on its own).

    `names_provision`: the question names a provision of its own (a 21 CFR section or part, an FD&C Act section).
    `other_entity`: it names a color additive that none of the anchored sections lists.
    `anchor_ranked`: an anchored section is among the sections the question's own words rank (it points there itself).
    `heading_words`: subject words of the anchored sections' headings.

    Anchored when the question's own words rank the anchored section, when it shares a subject word with the
    conversation's questions or the anchored headings, or when it gives a cue and is too short to stand alone. A
    question that is wrongly anchored is answered from the wrong section, while one that is wrongly left alone is
    answered as it was before follow-ups existed - so every doubt goes to "on its own"."""
    if context is None or not context:
        return None
    anchor = context.anchor
    found = cues(question)
    earlier = (context.earlier_question or "")[:EARLIER_QUESTION_CHARS] or None
    if not anchor:
        return FollowUp(False, "no_anchor", question, anchor, earlier, found)
    if names_provision:
        return FollowUp(False, "names_own_provision", question, anchor, earlier, found)
    if other_entity:
        return FollowUp(False, "names_other_color", question, anchor, earlier, found)
    if _fu2("AMB1") and anchor.cfr and earlier:
        # AMB1: the reply to "which type of product is it?" (regulus.rag.product_clarification) - the earlier question is
        # answered from the chosen type's section(s), named, so it is resolved there and not ranked again
        from regulus.rag.product_clarification import LABELS, narrowed

        chosen = narrowed(question, anchor.cfr)
        if chosen is not None:
            family, secs = chosen
            text = f"{earlier.rstrip()} ({LABELS[family].lower()}: " + ", ".join(f"21 CFR {x}" for x in secs) + ")"
            return FollowUp(True, "product_type_chosen", text, Anchor(cfr=secs), earlier, found)
    own = subject_words(question, stem)
    common = own & (set().union(*(subject_words(t.question, stem) for t in context.turns)) | (heading_words or set()))
    shared = len(common) >= SHARED_WORDS or (bool(common) and len(common) >= SHARED_FRACTION * len(own))
    if shared and _fu2("FU2C") and len(own) > SHARED_MAX_OWN_WORDS and not found:
        shared = False   # FU2C: a long question with no pointer back states its own subject
    short = bool(found) and len(own) <= SHORT_QUESTION_WORDS
    fragment = len(own) <= FRAGMENT_WORDS and not _DEFINITION_FORM.search(question)
    if "demonstrative" in found and len(own) <= SELF_CONTAINED_WORDS:
        short = True   # FU2: "Does that affect use of this color?" names the conversation's subject
    if _fu2("FU2B"):
        # FU2B: a question of at most SHORT_OWN_WORDS subject words that names nothing of its own continues it
        fragment = fragment or (len(own) <= SHORT_OWN_WORDS and not _DEFINITION_FORM.search(question))
    anchor_ranked = anchor_ranked and len(own) <= SELF_CONTAINED_WORDS
    if not (anchor_ranked or shared or short or fragment):
        # nothing ties it to the conversation: a new topic, answered on its own and without the earlier question
        return FollowUp(False, "new_topic", question, Anchor(), None, found)
    text = question
    if "pointer" in found and anchor.cfr:
        text = write_out_pointers(question, anchor.cfr[0])
    elif "pointer" in found and anchor.statute:
        text = write_out_pointers(question, anchor.statute[0], statute=True)
    elif anchor.parts and not (anchor.cfr or anchor.statute):
        # a whole part: the question is asked about that part, as if it had named it
        text = question.rstrip() + " (21 CFR " + ", ".join(f"part {p}" for p in anchor.parts) + ")"
    why = ((("ranked",) if anchor_ranked else ()) + (("shared_subject",) if shared else ()) + (("short_with_cue",) if short else ())
           + (("fragment",) if fragment and not (anchor_ranked or shared or short) else ()))
    return FollowUp(True, "+".join(why + found), text, anchor, earlier, found)


# a follow-up that names provisions of its own keeps the conversation's anchor beside them, up to this many sections
MAX_ANCHOR_SECTIONS = 3


def turn_anchor(plan, promotion, followup: FollowUp | None) -> Anchor:
    """What a turn was resolved to, for the next question: the promoted section(s) or FD&C Act section(s), else the 21
    CFR sections the question named (`plan`: the question's own plan, before companions were added), else - for an
    anchored follow-up that resolved to nothing new - the anchor it inherited. A follow-up naming provisions of its own
    keeps the earlier anchor after them ("Does 70.5(a) change that?" stays about the listing too)."""
    if promotion is not None and getattr(promotion, "method", None) == "statute_reference":
        # ST1: the Act text accompanies a regulation the question named; the regulation is what the turn is about
        secs = tuple(dict.fromkeys(r.reference.section for r in plan.requirements
                                   if r.reference.title == "21" and r.reference.section))
        own = Anchor(cfr=secs[:4], statute=tuple(promotion.statute)[:2])
    elif promotion is not None and (promotion.sections or promotion.statute):
        # a promotion of paragraphs (DF1: "70.3(f)", "70.3(g)") anchors their section; a product clarification (AMB1)
        # anchors every section it offered, so the reply can choose any of them
        secs = tuple(dict.fromkeys(re.sub(r"\(.*$", "", s) for s in promotion.sections))
        keep = 6 if getattr(promotion, "method", None) == "product_clarification" else 4
        own = Anchor(cfr=secs[:keep], statute=tuple(promotion.statute)[:2])
    else:
        secs = tuple(dict.fromkeys(r.reference.section for r in plan.requirements
                                   if r.reference.title == "21" and r.reference.section))
        parts = tuple(dict.fromkeys(r.reference.part for r in plan.requirements
                                    if r.reference.title == "21" and not r.reference.section and r.reference.part))
        own = Anchor(cfr=secs[:4]) if secs else Anchor(parts=parts[:2])
    if followup is None or not followup.anchor:
        return own
    if not own:
        return followup.anchor if followup.anchored else own
    if followup.reason == "names_own_provision":
        cfr = tuple(dict.fromkeys(own.cfr + followup.anchor.cfr))[:max(MAX_ANCHOR_SECTIONS, len(own.cfr))]
        return Anchor(cfr=cfr, statute=tuple(dict.fromkeys(own.statute + followup.anchor.statute))[:2],
                      parts=() if cfr else tuple(dict.fromkeys(own.parts + followup.anchor.parts))[:2])
    return own


def evaluation_question(asked: str, conversation: dict | None) -> str:
    """The question a follow-up's answer is judged against (the live evaluator, regulus.accuracy): what the pipeline
    answered - the follow-up with its pointers written out and the provisions the conversation anchored it to - not
    its bare words. "Can we use it in an eye shadow?" names no subject; judged alone, an answer about D&C Yellow No. 8
    reads as off-topic. A question asked on its own is judged as asked."""
    fu = (conversation or {}).get("followup") if isinstance(conversation, dict) else None
    if not isinstance(fu, dict) or not fu.get("anchored"):
        return asked
    labels = Anchor.from_dict(fu.get("anchor")).labels()
    text = fu.get("question") or asked
    return f"{text} ({'; '.join(labels)})" if labels else text


# CQ1 (27 Oct 2026): conversational query rewriting ("condense question"), the way retrieval chat assistants resolve a
# follow-up: the model rewrites it into one standalone question from the earlier question and the provision that
# answered it, and the pipeline answers that question. A follow-up about the same subject comes back naming the
# provision ("... under 21 CFR 73.2125"), so it is resolved there; a new subject comes back unchanged. The word rules
# above decide whenever the rewrite is missing or unusable.
REWRITE_SYSTEM = ("You rewrite a follow-up question from a conversation about U.S. FDA regulations so that it can be "
                  "understood on its own. You never answer it.")
REWRITE_MAX_TOKENS = 96
REWRITE_MAX_CHARS = 500


def rewrite_messages(question: str, earlier: str, label: str, heading: str | None = None) -> list[dict]:
    about = f"{label} ({heading.rstrip('.')})" if heading else label
    user = (f"Earlier question: {earlier}\nIt was answered from {about}.\nFollow-up question: {question}\n\n"
            f"If the follow-up continues the earlier subject, rewrite it as one standalone question that names what it "
            f"refers to, and end it with \"under {label}\". If it asks about a different subject, repeat it exactly. "
            f"Reply with the question only.")
    return [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}]


_CITED_NUMBER = re.compile(r"(?<![\d.])\d{1,4}\.\d{1,4}[a-z]?(?![\d])|§\s*\d{3}")


def read_rewrite(reply: str, question: str, anchor: Anchor) -> tuple[str, bool] | None:
    """(standalone question, whether it continues the anchored subject) from the model's reply, or None when the
    reply is unusable: empty, several lines, not a question, too long, or citing a provision that neither the
    follow-up nor the anchor names (an invented citation never reaches retrieval)."""
    text = " ".join((reply or "").strip().strip('"').split())
    text = re.sub(r"^(?:standalone|rewritten)?\s*question\s*:\s*", "", text, flags=re.I)
    if not text or len(text) > REWRITE_MAX_CHARS or "\n" in (reply or "").strip() or not text.endswith("?"):
        return None
    allowed = {m.replace(" ", "") for m in _CITED_NUMBER.findall(question)} | set(anchor.cfr) | {f"§{s}" for s in anchor.statute}
    cited = {m.replace(" ", "") for m in _CITED_NUMBER.findall(text)}
    if cited - allowed:
        return None
    labels = set(anchor.cfr) | {f"§{s}" for s in anchor.statute}
    continues = bool(cited & labels) and not (cited & labels <= {m.replace(" ", "") for m in _CITED_NUMBER.findall(question)}
                                                 and text == " ".join(question.split()))
    return text, continues

