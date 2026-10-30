"""Definitions sections whose paragraph numbering never restarted at each definition (DEF1, 30 Oct 2026).

A definitions section lists its terms as undesignated paragraphs, each opening with the term in italics in the eCFR XML
(<P><I>Tobacco product,</I> as stated in ...</P>), and a definition may number its own items (1), (2)(i) ... The
original ingestion carried the designator stack across definitions: in 21 CFR 1100.3 "Component or part means ..."
was stored under the previous definition's "(2)(ii)" and its items as "(2)(ii)(1)", "(2)(ii)(2)", and "Tobacco
product"'s (1) as "(2)(ii)(2)(1)" - paths that do not exist. Answers cited them (300Q T-008, scored "nonexistent
paragraph labels"), and every consumer of the path hierarchy read unrelated definitions as children of the one before.
Measured 30 Oct: 596 definition leads in 42 sections (117.3, 1.227, 1.328, 1.500, 111.3, 820.3, 860.3, 1114.3, ...);
the evidence of 17 of the 300Q answers carried such paths.

The repair restarts the numbering at each definition, at the level the list of definitions sits at - "" in 1100.3,
"(c)" in 1.600 ("(c) For the purposes of this subpart: ...") - taken from the first definition's stored path, which the
ingestion got right. A definition's lead row has that path, its items (1), (2)(i) ... nest under it, and text continuing
an item keeps that item's path. path_ordinal is renumbered in document order so (section, path, path_ordinal) stays unique.

Source: a section is planned only from its part's eCFR XML whose SHA-256 equals the source_payload_sha256 its stored
rows record, and only when every definition lead of the XML is found among the stored rows, in order. Rows before the
first definition are never changed. Planning is read-only; the SQL is applied by a person, and a rollback is written
with it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence
from xml.etree import ElementTree as ET

from regulus.ingestion.dropped_text import inline_text, normalize

_LEAD_DESIGNATORS = re.compile(r"^\s*((?:\([A-Za-z0-9]{1,4}\))+)")
_ONE = re.compile(r"\(([A-Za-z0-9]{1,4})\)")
_ROMAN = re.compile(r"^(?:x{0,3})(?:ix|iv|v?i{0,3})$")
MATCH_CHARS = 60   # a stored row is a definition lead when its first characters are a lead's


# a lead defines its term ("X means", "X has the meaning", "X, as defined in"); an italic paragraph heading does not
# (201.327's test-procedure headings)
_DEFINES = re.compile(r"\b(?:means?|includes?|has\s+the\s+(?:same\s+)?meaning|refers?\s+to|is\s+defined|"
                      r"as\s+(?:stated|defined|used))\b", re.I)
_DEFINITIONS_HEADING = re.compile(r"\bdefinitions?\b|\bmeaning\s+of\s+terms\b|\bterms\b.*\bmean\b", re.I)
MIN_LEADS_WITHOUT_HEADING = 3


def definition_leads(xml_bytes: bytes) -> dict[str, list[str]]:
    """{section: the normalised text of each definition lead, in order} - an undesignated <P> whose first content is
    an <I> term and whose first words define it - for a section whose heading names definitions, or that has at least
    MIN_LEADS_WITHOUT_HEADING such leads. Other sections are left out."""
    root = ET.fromstring(xml_bytes)
    out: dict[str, list[str]] = {}
    for div in root.iter():
        if not (div.tag.startswith("DIV") and div.get("TYPE") == "SECTION"):
            continue
        leads = []
        for p in div:
            if p.tag != "P" or (p.text or "").strip() or not len(p) or p[0].tag != "I":
                continue
            text = inline_text(p)
            if text and not _LEAD_DESIGNATORS.match(text) and _DEFINES.search(text[:160]):
                leads.append(text)
        head = div.find("HEAD")
        named = head is not None and _DEFINITIONS_HEADING.search(inline_text(head) or "")
        if leads and (named or len(leads) >= MIN_LEADS_WITHOUT_HEADING):
            out[div.get("N") or ""] = leads
    return out


@dataclass(frozen=True)
class Row:
    id: str
    ordinal: int
    paragraph_path: str
    path_ordinal: int
    body_text: str


@dataclass
class SectionPlan:
    section_number: str
    status: str                       # "planned", "no_change" or "misaligned"
    reason: str | None = None
    updates: list[tuple[str, str, int, str, int]] = field(default_factory=list)   # id, old path, old po, new path, new po


def _kind(label: str, stack: list[tuple[str, str]]) -> str:
    if label.isdigit():
        return "num"
    if label.isupper():
        return "upper"
    if _ROMAN.match(label):
        lower = next((lab for k, lab in reversed(stack) if k == "lower"), None)
        # "(i)" after "(h)" is the next letter, not a roman one
        if lower is not None and len(label) == 1 and ord(label) == ord(lower) + 1 and not any(k == "roman" for k, _ in stack):
            return "lower"
        return "roman"
    return "lower"


_ROMANS = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii",
           "xviii", "xix", "xx", "xxi", "xxii", "xxiii", "xxiv", "xxv"]
_FIRST = {"num": "1", "roman": "i", "upper": "A", "lower": "a"}


def _successor(kind: str, label: str) -> str | None:
    if kind == "num":
        return str(int(label) + 1)
    if kind == "roman":
        return _ROMANS[_ROMANS.index(label) + 1] if label in _ROMANS[:-1] else None
    return chr(ord(label) + 1) if len(label) == 1 else None


def _place(stack: list[tuple[str, str]], labels: list[str]) -> list[tuple[str, str]]:
    """Each designator is a sibling at the deepest level of its kind whose next label it is; a first label ((1), (i),
    (A)) that is no level's next opens a level under the deepest; otherwise a sibling at the deepest level of its kind.
    The CFR's fifth and sixth levels repeat the second and third in italics ((1)(iii)(B)(1)), which the stored text
    no longer marks: 1.227 "Farm" (1)...(iii)...(B) then (1)-(3) under (B), then "(2) Secondary activities farm"."""
    for label in labels:
        kind = _kind(label, stack)
        same = [i for i, (k, _) in enumerate(stack) if k == kind]
        nxt = [i for i in same if _successor(kind, stack[i][1]) == label]
        if nxt:
            stack = stack[:nxt[-1]] + [(kind, label)]
        elif label == _FIRST[kind] or not same:
            stack = stack + [(kind, label)]
        else:
            stack = stack[:same[-1]] + [(kind, label)]
    return stack


def plan_section(section: str, leads: Sequence[str], rows: Sequence[Row]) -> SectionPlan:
    """The corrected path of every stored row of a definitions section (rows in ordinal order)."""
    plan = SectionPlan(section, "no_change")
    keys = [normalize(t)[:MATCH_CHARS] for t in leads]
    found, k = [], 0
    for i, r in enumerate(rows):
        if k < len(keys) and normalize(r.body_text)[:MATCH_CHARS] == keys[k]:
            found.append(i)
            k += 1
    if k != len(keys):
        plan.status, plan.reason = "misaligned", f"{len(keys) - k} of {len(keys)} definition leads not found in order"
        return plan
    lead_rows = set(found)
    # the level the definitions are listed at: the first definition's stored path, which the original ingestion got
    # right (the error begins at the second) - "" in 1100.3, "(c)" in 1.600 ("(c) For the purposes of this subpart:")
    base: list[tuple[str, str]] = []
    for label in _ONE.findall(rows[found[0]].paragraph_path or ""):
        base.append((_kind(label, base), label))   # the stored path as it is, level by level - never re-nested
    if len(base) > 1:
        # definitions listed deep inside a lettered structure (314.81's (b)(3)(iii)): where the list sits cannot be told
        # from the stored rows with certainty, so the section is left for review
        plan.status, plan.reason = "deep", f"definitions listed at {rows[found[0]].paragraph_path}"
        return plan
    new_paths: list[str] = []
    stack: list[tuple[str, str]] = []
    for i, r in enumerate(rows):
        if i < found[0]:
            new_paths.append(r.paragraph_path)   # before the first definition: as stored
            continue
        if i in lead_rows:
            stack = list(base)
        else:
            m = _LEAD_DESIGNATORS.match(r.body_text or "")
            if m:
                stack = _place(stack, _ONE.findall(m.group(1)))
        new_paths.append("".join(f"({lab})" for _, lab in stack))
    seen: dict[str, int] = {}
    for r, path in zip(rows, new_paths):
        po = seen.get(path, 0)
        seen[path] = po + 1
        if (path, po) != (r.paragraph_path, r.path_ordinal):
            plan.updates.append((r.id, r.paragraph_path, r.path_ordinal, path, po))
    if plan.updates:
        plan.status = "planned"
    return plan


def _sql(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _update(rid: str, sec: str, from_path: str, from_po: int, to_path: str, to_po: int) -> str:
    """One row moved, or the transaction fails: a row that is not as planned is never overwritten."""
    return (f"DO $$ BEGIN UPDATE cfr_records SET paragraph_path = {_sql(to_path)}, path_ordinal = {to_po} "
            f"WHERE id = {_sql(rid)} AND section_number = {_sql(sec)} AND paragraph_path = {_sql(from_path)} "
            f"AND path_ordinal = {from_po}; IF NOT FOUND THEN RAISE EXCEPTION 'DEF1: row {rid} is not as planned'; "
            f"END IF; END $$;")


def repair_sql(plans: Sequence[SectionPlan]) -> tuple[str, str]:
    """(repair, rollback): one transaction each, in two phases - every changed row first moves to a temporary path
    unique within its section ("(ZZZZ)(n)", which the path check accepts), then to its final (path, path_ordinal) -
    so the unique key holds at every statement. Every UPDATE names the row's id and the values it expects; a row that
    changed since planning fails the whole transaction."""
    moves = [(p.section_number, f"(ZZZZ)({n})", u) for p in plans for n, u in enumerate(p.updates)]
    fwd, back = ["BEGIN;"], ["BEGIN;"]
    for sec, t, (rid, old, old_po, new, new_po) in moves:
        fwd.append(_update(rid, sec, old, old_po, t, 0))
        back.append(_update(rid, sec, new, new_po, t, 0))
    for sec, t, (rid, old, old_po, new, new_po) in moves:
        fwd.append(_update(rid, sec, t, 0, new, new_po))
        back.append(_update(rid, sec, t, 0, old, old_po))
    fwd.append("COMMIT;")
    back.append("COMMIT;")
    n, s = len(moves), sum(1 for p in plans if p.updates)
    return (f"-- DEF1 definitions paragraph_path repair: {n} rows in {s} sections\n" + "\n".join(fwd) + "\n",
            f"-- DEF1 ROLLBACK of the definitions paragraph_path repair: {n} rows in {s} sections\n" + "\n".join(back) + "\n")
