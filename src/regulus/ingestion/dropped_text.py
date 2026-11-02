"""Restore section text the original eCFR ingestion dropped: <EXTRACT> block quotes and <FP> flush paragraphs.

The original ingestion stored one cfr_records row per <P> child of a section (split at inline paragraph designators)
and skipped the section's other children. Measured on six hash-verified part payloads (71, 73, 74, 81, 701, 740):
0 of 117 EXTRACT and 0 of 17 FP elements are stored, so a lead-in such as "shall bear the following warning:" is
stored and the warning itself is not.

Repair unit: an anchor row, the stored paragraph a dropped element follows. Every dropped EXTRACT or FP between the
anchor's <P> and the next <P> is restored as a row of the anchor's paragraph_path with the next free path_ordinal,
which is how the original ingestion already stores undesignated text (path_ordinal > 0). The section's ordinals are
renumbered to document order. Body text follows the stored rows' convention: the element's text, whitespace
normalised, with the element's block items (FP-1, LI, ...) joined by one space. content_sha256 follows the corpus's
rule (`content_hash`), which reproduces the stored hash of every row of the six parts.

Source: a part's XML is accepted only when its SHA-256 equals the source_payload_sha256 every stored row of the part
records. A section is planned only when its stored rows account for its source text exactly, in order. Planning is
read-only. Applying needs the SHA-256 of the exact plan and its source provenance (the authorisation), refuses a
read-only connection, re-checks every precondition under FOR UPDATE inside one transaction, and hands the manifest
to the caller to persist before the transaction commits; `rollback` undoes a manifest. A plan computed after the
repair is empty, so a rerun changes nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

RESTORED_TAGS = frozenset({"EXTRACT", "FP", "TABLE"})   # TABLE: a GPOTABLE, rendered row by row (22 Oct 2026)
# 2 Nov 2026: footnotes, notes and the flush-paragraph variants (FP-1 formula lines, FP-2 ...) are the regulation's text
# too (1020.30's aluminium-alloy footnote and line-voltage formula, 175.300's note) and were dropped the same way
RESTORED_EXTRA = frozenset({"FTNT", "NOTE", "NOTES"})


def restorable(tag: str) -> bool:
    return tag in RESTORED_TAGS or tag in RESTORED_EXTRA or tag.startswith("FP-")
# Tags that never open a new block: their text runs on inside the enclosing paragraph.
INLINE_TAGS = frozenset({"I", "E", "B", "SU", "FR", "AC", "PRTPAGE", "FTREF", "SUB", "SUP", "BR"})
# The corpus's content_sha256 is SHA-256 over body_text with typographic punctuation folded to ASCII, two
# compatibility characters mapped to their Greek letters and "§§" collapsed (docs/embedding_provenance.md, section 7).
_HASH_FOLD = str.maketrans({"—": "-", "–": "-", "−": "-", "‘": "'", "’": "'", "′": "'",
                            "“": '"', "”": '"', "″": '"', "µ": "μ", "Ω": "Ω"})


class RepairError(RuntimeError):
    """A precondition of the repair does not hold; nothing was changed."""


def normalize(text: str | None) -> str:
    return " ".join((text or "").split())


def _compact(text: str | None) -> str:
    return "".join((text or "").split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(body_text: str) -> str:
    """content_sha256 as the corpus computes it."""
    return sha256_text(body_text.translate(_HASH_FOLD).replace("§§", "§"))


def inline_text(el: ET.Element) -> str:
    """The text of a paragraph, as the original ingestion stored a <P>: all text, whitespace normalised."""
    return normalize("".join(el.itertext()))


def block_text(el: ET.Element) -> str:
    """The text of a block element: runs of inline content stay glued; each block item (FP-1, LI, P, ...) is one
    paragraph; paragraphs are joined by one space."""
    pieces: list[str] = []
    run = [el.text or ""]
    for child in el:
        if child.tag in INLINE_TAGS:
            run.append("".join(child.itertext()))
        else:
            pieces.append("".join(run))
            pieces.append(block_text(child))
            run = []
        run.append(child.tail or "")
    pieces.append("".join(run))
    return normalize(" ".join(normalize(p) for p in pieces if normalize(p)))


@dataclass(frozen=True)
class SourceElement:
    tag: str
    text: str
    # a block or table stored in parts: every part carries the unsplit text and its part number, so a block stored
    # whole before the split (a 3,579-character table of 23 Oct) still reads as stored
    whole: str = ""
    part: int = 0


@dataclass(frozen=True)
class StoredRow:
    id: str
    paragraph_path: str
    path_ordinal: int
    ordinal: int
    body_text: str


@dataclass(frozen=True)
class NewRow:
    section_number: str
    paragraph_path: str
    path_ordinal: int
    ordinal: int
    body_text: str
    content_sha256: str
    source_tag: str
    anchor_id: str


@dataclass
class SectionPlan:
    section_number: str
    status: str                                   # "planned", "no_change" or "misaligned"
    reason: str | None = None
    inserts: list[NewRow] = field(default_factory=list)
    ordinal_updates: list[tuple[str, int, int]] = field(default_factory=list)   # (row id, old ordinal, new ordinal)
    # (row id, old path_ordinal, new path_ordinal): a stored row of an anchor's path that follows the restored text
    path_ordinal_updates: list[tuple[str, int, int]] = field(default_factory=list)
    already_restored: int = 0
    untargeted_dropped: list[dict[str, Any]] = field(default_factory=list)
    preconditions: list[tuple[str, str, int, int, str]] = field(default_factory=list)  # id, path, path_ordinal, ordinal, sha


def _cell(e: ET.Element) -> str:
    return normalize("".join(e.itertext()))


def _html_columns(thead: ET.Element | None) -> list[str]:
    """Column labels of an HTML table head, rowspan/colspan resolved: a spanning cell labels every column under it
    ("Closing date" over "Food use" -> "Closing date, Food use")."""
    if thead is None:
        return []
    grid: dict[tuple[int, int], str] = {}
    for r, tr in enumerate(thead.iter("TR")):
        c = 0
        for th in tr:
            if th.tag not in ("TH", "TD"):
                continue
            while (r, c) in grid:
                c += 1
            rs, cs = int(th.get("rowspan") or 1), int(th.get("colspan") or 1)
            for dr in range(rs):
                for dc in range(cs):
                    grid[(r + dr, c + dc)] = _cell(th)
            c += cs
    width = 1 + max((c for _, c in grid), default=-1)
    rows = 1 + max((r for r, _ in grid), default=-1)
    labels = []
    for c in range(width):
        parts: list[str] = []
        for r in range(rows):
            v = grid.get((r, c), "")
            if v and v not in parts:
                parts.append(v)
        labels.append(", ".join(parts))
    return labels


def _table_parts(el: ET.Element) -> tuple[str, str, list[str]] | None:
    """(GPO title, HTML caption, one "heading: value; ..." line per row), or None when the element holds no table."""
    html = el if el.tag == "TABLE" else el.find(".//TABLE")
    if html is not None:
        heads = _html_columns(html.find("THEAD"))
        body = html.find("TBODY") if html.find("TBODY") is not None else html
        rows = [[_cell(e) for e in tr if e.tag in ("TD", "TH")] for tr in body.iter("TR")]
        title, caption = "", (_cell(html.find("CAPTION")) if html.find("CAPTION") is not None else "")
    else:
        table = el if el.tag == "GPOTABLE" else el.find(".//GPOTABLE")
        if table is None:
            return None
        heads = [_cell(h) for h in table.iter("CHED")]
        rows = [[_cell(e) for e in row.findall("ENT")] for row in table.iter("ROW")]
        title = _cell(table.find("TTITLE")) if table.find("TTITLE") is not None else ""
        caption = ""
    lines = []
    for cells in rows:
        pairs = [f"{heads[i]}: {c}" if i < len(heads) and heads[i] else c for i, c in enumerate(cells) if c]
        if pairs:
            lines.append("; ".join(pairs) + ("" if pairs[-1].endswith(".") else "."))
    return title, caption, lines


def table_text(el: ET.Element) -> str:
    """A table as text, one row after another: "header: value; header: value." - every non-empty cell kept with its
    column heading, so the reader (and the model) can tell "Established label name: Hydrofluorocarbon 152a" from the
    formula. Handles the eCFR renderer's HTML <TABLE> (THEAD/TBODY, rowspan/colspan) and the GPO <GPOTABLE>."""
    parts = _table_parts(el)
    if parts is None:
        return block_text(el)
    title, _, lines = parts
    return normalize(" ".join(([title] if title else []) + lines))


# A table longer than this is stored as several rows (30 Oct): the 23 Oct repair excluded tables "over 4000 chars" whole
# (the prompt cuts a result at 4,000 characters), which left 101.12's RACC tables (26,980), 1107.18's product-properties
# table (23,216) and 1002.1's reporting table (6,958) out of the corpus. Each part holds whole rows - every row already
# names its columns - and opens with the table's caption, so a part read alone says which table it is.
TABLE_PART_CHARS = 3500


def table_texts(el: ET.Element) -> list[str]:
    """The table as one text (table_text), or - over TABLE_PART_CHARS - as parts of whole rows, each opening with
    "<caption or title> (part k of n):"."""
    whole = table_text(el)
    parts = _table_parts(el)
    if len(whole) <= TABLE_PART_CHARS or parts is None:
        return [whole]
    title, caption, lines = parts
    label = caption or title or "Table"
    room = TABLE_PART_CHARS - len(label) - 24
    groups: list[list[str]] = [[]]
    for line in lines:
        if groups[-1] and sum(len(x) + 1 for x in groups[-1]) + len(line) > room:
            groups.append([])
        groups[-1].append(line)
    n = len(groups)
    return [normalize(f"{label} (part {k} of {n}): " + " ".join(g)) for k, g in enumerate(groups, 1)]


def _block_pieces(el: ET.Element) -> tuple[str, list[str]]:
    """(heading, paragraphs) of a block: its HD text and the text of each block item, inline runs glued."""
    heading, pieces, run = "", [], [el.text or ""]
    for child in el:
        if child.tag in INLINE_TAGS:
            run.append("".join(child.itertext()))
        else:
            pieces.append(normalize("".join(run)))
            text = block_text(child)
            if child.tag == "HD" and not heading:
                heading = text
            pieces.append(text)
            run = []
        run.append(child.tail or "")
    pieces.append(normalize("".join(run)))
    return heading, [p for p in pieces if p]


def _sentences(text: str, room: int) -> list[str]:
    """A paragraph longer than `room` cut after sentence ends ('. ', '; ') into pieces of at most `room` characters."""
    out, cur = [], ""
    for bit in [b for b in re.split(r"(?<=[.;])\s+", text) if b]:
        while len(bit) > room:               # no sentence end within room: a hard cut at a space
            cut = bit.rfind(" ", 0, room) if bit.rfind(" ", 0, room) > 0 else room
            if cur:
                out.append(cur)
                cur = ""
            out.append(bit[:cut])
            bit = bit[cut:].strip()
        if cur and len(cur) + 1 + len(bit) > room:
            out.append(cur)
            cur = bit
        else:
            cur = f"{cur} {bit}".strip()
    return out + ([cur] if cur else [])


def block_texts(el: ET.Element) -> list[str]:
    """A block as one text (block_text), or - over TABLE_PART_CHARS - as parts of whole paragraphs, each opening with
    "<heading> (part k of n):" (2 Nov 2026: 22 blocks over 4,000 characters - forms, analytical methods, 343.80's
    prescribing information - had been excluded whole)."""
    whole = block_text(el)
    if len(whole) <= TABLE_PART_CHARS:
        return [whole]
    heading, pieces = _block_pieces(el)
    label = heading or "Text"
    room = TABLE_PART_CHARS - len(label) - 24
    groups: list[list[str]] = [[]]
    for piece in (x for p in pieces for x in (_sentences(p, room) if len(p) > room else [p])):
        if groups[-1] and sum(len(x) + 1 for x in groups[-1]) + len(piece) > room:
            groups.append([])
        groups[-1].append(piece)
    n = len(groups)
    return [normalize(f"{label} (part {k} of {n}): " + " ".join(g)) for k, g in enumerate(groups, 1)]


def _parts(tag: str, texts: list[str], whole: str) -> list[SourceElement]:
    if len(texts) == 1:
        return [SourceElement(tag, texts[0])]
    return [SourceElement(tag, t, whole, k) for k, t in enumerate(texts, 1)]


def _elements(c: ET.Element) -> list[SourceElement]:
    if c.tag == "P":
        return [SourceElement("P", inline_text(c))]
    if c.tag in ("GPOTABLE", "TABLE") or (c.tag.startswith("DIV") and (c.find(".//GPOTABLE") is not None
                                                                       or c.find(".//TABLE") is not None)):
        return _parts("TABLE", table_texts(c), table_text(c))
    return _parts(c.tag, block_texts(c), block_text(c))


def sections_of(xml_bytes: bytes) -> dict[str, list[SourceElement]]:
    """{section number: its children in document order, HEAD excluded}. P keeps paragraph text; a table (a GPOTABLE,
    alone or in a DIV) its rows as text; other tags block text."""
    root = ET.fromstring(xml_bytes)
    out: dict[str, list[SourceElement]] = {}
    for div in root.iter():
        if not (div.tag.startswith("DIV") and div.get("TYPE") == "SECTION"):
            continue
        number = div.get("N") or ""
        out[number] = [e for c in div if c.tag not in ("HEAD", "CITA", "EDNOTE", "SECAUTH") for e in _elements(c)]
    return out


def verify_payload(xml_bytes: bytes, stored_payload_shas: Iterable[str]) -> str:
    """The payload's SHA-256, provided every stored row of the part records exactly that hash."""
    shas = set(stored_payload_shas)
    got = hashlib.sha256(xml_bytes).hexdigest()
    if shas != {got}:
        raise RepairError(f"source payload sha256 {got} does not match the stored source_payload_sha256 values {sorted(shas)}")
    return got


def align(elements: Sequence[SourceElement], rows: Sequence[StoredRow]) -> tuple[list[tuple[SourceElement, int | None]], int]:
    """Walk the source against the stored text. Returns the dropped EXTRACT/FP elements, each with the index of the
    stored row it follows, and the number of EXTRACT/FP elements already stored. Raises RepairError when the stored
    rows do not account for the section's paragraphs exactly, in order (the stored text is read as one stream, so a
    <P> split over several rows, or several <P> merged into one row, still align)."""
    # The stream is compared with every whitespace character removed: the original ingestion split a paragraph at an
    # inline designator ("... marketing plans—(1) Labeling.") into two rows, which reads back as "plans— (1)" - 865
    # sections, 1114.7 and 1040.10 among them, were refused for that space alone (23 Oct). Every other character must
    # still match exactly and in order.
    texts = [_compact(r.body_text) for r in rows]
    stream = "".join(texts)
    ends, pos_end = [], 0
    for t in texts:
        pos_end += len(t)
        ends.append(pos_end)
    boundary = {0: None} | {e: i for i, e in enumerate(ends)}
    pos, dropped, already = 0, [], 0

    def take(t: str) -> bool:
        nonlocal pos
        c = _compact(t)
        if c and stream.startswith(c, pos):
            pos += len(c)
            return True
        return False

    skip_rest = False
    for el in elements:
        if not el.text:
            continue
        if el.part > 1 and skip_rest:
            continue                         # its block was stored whole: the first part consumed it
        if el.part == 1:
            skip_rest = take(el.whole)
            if skip_rest:
                already += 1
                continue
        if el.tag == "P":
            if not take(el.text):
                raise RepairError(f"paragraph not found in the stored text at offset {pos}: {el.text[:80]!r}")
        elif restorable(el.tag):
            if take(el.text):
                already += 1
            elif el.tag in RESTORED_EXTRA and pos not in boundary:
                continue                     # as before 2 Nov: a note inside a stored row's span is left alone
            elif pos not in boundary:
                raise RepairError(f"dropped {el.tag} falls inside a stored row at offset {pos}: {el.text[:80]!r}")
            else:
                dropped.append((el, boundary[pos]))
        else:
            take(el.text)  # CITA, tables, notes: stored separately or not at all; consumed only if a row holds them
    if pos != len(stream):
        raise RepairError(f"stored text not accounted for by the source after offset {pos}: {stream[pos:pos + 80]!r}")
    return dropped, already


def plan_section(section: str, elements: Sequence[SourceElement], rows: Sequence[StoredRow],
                 targets: set[str] | None = None) -> SectionPlan:
    """The rows to insert and the ordinals to renumber for one section. `targets` (paragraph paths) limits the repair to
    dropped elements that follow those anchors; None repairs every dropped element of the section."""
    rows = sorted(rows, key=lambda r: r.ordinal)
    plan = SectionPlan(section, "no_change",
                       preconditions=[(r.id, r.paragraph_path, r.path_ordinal, r.ordinal, sha256_text(r.body_text)) for r in rows])
    if any(b.ordinal <= a.ordinal for a, b in zip(rows, rows[1:])):
        plan.status, plan.reason = "misaligned", "stored ordinals are not strictly increasing"
        return plan
    try:
        dropped, plan.already_restored = align(elements, rows)
    except RepairError as exc:
        plan.status, plan.reason = "misaligned", str(exc)
        return plan
    after: dict[int, list[SourceElement]] = {}
    before: list[SourceElement] = []   # text ahead of every stored row (369.20's first category, "ACETANILID.")
    for el, anchor in dropped:
        anchor_path = rows[anchor].paragraph_path if anchor is not None else rows[0].paragraph_path
        if targets is not None and anchor_path not in targets:
            plan.untargeted_dropped.append({"tag": el.tag, "anchor": anchor_path if anchor is not None else None,
                                            "text": el.text[:160], "reason": "not targeted"})
        elif anchor is None:
            before.append(el)
        else:
            after.setdefault(anchor, []).append(el)
    if not after and not before:
        return plan
    # path_ordinal follows document order within a path. When the anchor's path has later stored rows (2 Nov 2026: 175.300,
    # 201.327, 343.80, 369.20, 369.21, 511.1, 1020.30 were refused for it), those rows move up by the number of rows
    # restored before them, and the restored text takes the places freed after the anchor
    shift: dict[str, int] = {}
    base, ordinal = rows[0].ordinal, rows[0].ordinal
    if before:
        # in free places below the first row's path_ordinal when there are enough, else that path moves up
        first, k = rows[0], len(before)
        places = range(first.path_ordinal - k, first.path_ordinal) if first.path_ordinal >= k else range(first.path_ordinal, first.path_ordinal + k)
        if first.path_ordinal < k:
            shift[first.paragraph_path] = k
        for el, po in zip(before, places):
            plan.inserts.append(NewRow(section, first.paragraph_path, po, ordinal, el.text, content_hash(el.text), el.tag, first.id))
            ordinal += 1
    for i, r in enumerate(rows):
        if r.ordinal != ordinal:
            plan.ordinal_updates.append((r.id, r.ordinal, ordinal))
        if shift.get(r.paragraph_path):
            plan.path_ordinal_updates.append((r.id, r.path_ordinal, r.path_ordinal + shift[r.paragraph_path]))
        ordinal += 1
        po = r.path_ordinal + shift.get(r.paragraph_path, 0)
        for el in after.get(i, []):
            po += 1
            shift[r.paragraph_path] = shift.get(r.paragraph_path, 0) + 1
            plan.inserts.append(NewRow(section, r.paragraph_path, po, ordinal, el.text, content_hash(el.text), el.tag, r.id))
            ordinal += 1
    moved = {u[0]: u[2] for u in plan.path_ordinal_updates}
    final = [(r.paragraph_path, moved.get(r.id, r.path_ordinal)) for r in rows] + [(n.paragraph_path, n.path_ordinal) for n in plan.inserts]
    if len(set(final)) != len(final):
        plan.status, plan.reason, plan.inserts, plan.ordinal_updates, plan.path_ordinal_updates = (
            "misaligned", "restored rows would collide with a stored (paragraph_path, path_ordinal)", [], [], [])
        return plan
    assert ordinal - base == len(rows) + len(plan.inserts)
    plan.status = "planned"
    return plan


def plan_digest(plans: Sequence[SectionPlan], provenance: Mapping[str, Mapping[str, str]]) -> str:
    """SHA-256 of everything a plan changes, every precondition it relies on, and the provenance written with it
    ({part: {url, sha256, retrieved_utc}}): the authorisation token for apply."""
    body = {"sections": [{"section": p.section_number, "inserts": [asdict(r) for r in p.inserts],
                          "ordinal_updates": [list(u) for u in p.ordinal_updates], "preconditions": [list(c) for c in p.preconditions],
                          **({"path_ordinal_updates": [list(u) for u in p.path_ordinal_updates]} if p.path_ordinal_updates else {})}
                         for p in plans if p.status == "planned"],
            "provenance": {part: {k: v[k] for k in ("url", "sha256", "retrieved_utc")} for part, v in sorted(provenance.items())}}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# -- database ---------------------------------------------------------------------------------------------------------

ROWS_SQL = """
SELECT id::text AS id, section_number, paragraph_path, path_ordinal, ordinal, body_text, source_payload_sha256
FROM cfr_records WHERE section_number = ANY(%(sections)s) AND removed_on IS NULL ORDER BY section_number, ordinal
"""
PART_SHAS_SQL = "SELECT DISTINCT source_payload_sha256 FROM cfr_records WHERE part_number = %(part)s"

INSERT_SQL = """
INSERT INTO cfr_records (title, chapter, subchapter_code, part_number, subpart_id, section_number, paragraph_path,
    path_ordinal, ordinal, section_heading, parent_headings, body_text, effective_date, last_amended_date, fr_citation,
    ecfr_permalink, govinfo_pdf_url, source_endpoint, retrieved_at, snapshot_date, content_sha256, source_payload_sha256,
    is_reserved)
SELECT a.title, a.chapter, a.subchapter_code, a.part_number, a.subpart_id, a.section_number, %(paragraph_path)s,
    %(path_ordinal)s, %(ordinal)s, a.section_heading, a.parent_headings, %(body_text)s, a.effective_date,
    a.last_amended_date, a.fr_citation, a.ecfr_permalink, a.govinfo_pdf_url, a.source_endpoint, %(retrieved_at)s::timestamptz,
    a.snapshot_date, %(content_sha256)s, a.source_payload_sha256, false
FROM cfr_records a WHERE a.id = %(anchor_id)s::uuid AND a.section_number = %(section_number)s
RETURNING id::text AS id
"""

# Every foreign key that points at cfr_records(id): a rollback must not delete a row something now references.
REFERENCING_SQL = """
SELECT c.conrelid::regclass::text AS tbl, a.attname AS col
FROM pg_constraint c JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
WHERE c.contype = 'f' AND c.confrelid = 'cfr_records'::regclass
"""


def stored_rows(query, sections: Sequence[str]) -> dict[str, list[StoredRow]]:
    out: dict[str, list[StoredRow]] = {s: [] for s in sections}
    for r in query(ROWS_SQL, {"sections": list(sections)}):
        out[r["section_number"]].append(StoredRow(r["id"], r["paragraph_path"] or "", int(r["path_ordinal"]), int(r["ordinal"]),
                                                  r["body_text"]))
    return out


def plan_part(query, part: str, xml_bytes: bytes, targets: Mapping[str, set[str]] | None = None) -> list[SectionPlan]:
    """Plan one part from its verified payload. `query(sql, params)` is a read-only query function. `targets` maps a
    section to the anchor paragraph paths to repair; sections absent from it are not planned (None = every section)."""
    verify_payload(xml_bytes, [r["source_payload_sha256"] for r in query(PART_SHAS_SQL, {"part": part})])
    source = sections_of(xml_bytes)
    sections = sorted(source) if targets is None else sorted(s for s in targets if s in source)
    missing = [] if targets is None else sorted(s for s in targets if s not in source)
    if missing:
        raise RepairError(f"targeted sections absent from the part {part} payload: {missing}")
    rows = stored_rows(query, sections)
    return [plan_section(s, source[s], rows[s], None if targets is None else set(targets[s])) for s in sections]


def _writable_transaction_check(conn) -> None:
    if getattr(conn, "read_only", False):
        raise RepairError("the connection is read-only; the repair needs an explicitly writable connection")
    ro = conn.execute("SELECT current_setting('transaction_read_only') AS ro").fetchall()[0]["ro"]
    if ro != "off":
        raise RepairError("the transaction is read-only; refusing to override it")


def apply_plans(conn, plans: Sequence[SectionPlan], provenance: Mapping[str, Mapping[str, str]], authorized_sha256: str,
                write_manifest: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Apply planned sections in one transaction on a writable psycopg connection (dict rows, not inside a
    transaction). Refuses unless `authorized_sha256` equals the plan digest, the connection is writable, and every
    stored row of each section is exactly as planned. `write_manifest` is called inside the transaction, after the last
    write and before the commit, so a manifest that cannot be persisted aborts the whole change."""
    planned = [p for p in plans if p.status == "planned"]
    digest = plan_digest(plans, provenance)
    if authorized_sha256 != digest:
        raise RepairError(f"plan digest {digest} is not the authorised {authorized_sha256}")
    manifest: dict[str, Any] = {"plan_sha256": digest, "provenance": {k: dict(v) for k, v in provenance.items()},
                                "inserted": [], "ordinal_updates": [], "path_ordinal_updates": []}
    with conn.transaction():
        _writable_transaction_check(conn)
        for p in planned:
            current = conn.execute("SELECT id::text AS id, paragraph_path, path_ordinal, ordinal, body_text FROM cfr_records "
                                   "WHERE section_number = %(s)s AND removed_on IS NULL ORDER BY ordinal FOR UPDATE",
                                   {"s": p.section_number}).fetchall()
            now = [(r["id"], r["paragraph_path"] or "", int(r["path_ordinal"]), int(r["ordinal"]), sha256_text(r["body_text"]))
                   for r in current]
            if now != [tuple(x) for x in p.preconditions]:
                raise RepairError(f"{p.section_number}: stored rows changed since the plan was made")
            for row_id, old, new in sorted(p.ordinal_updates, key=lambda u: -u[1]):
                n = conn.execute("UPDATE cfr_records SET ordinal = %(new)s WHERE id = %(id)s::uuid AND ordinal = %(old)s",
                                 {"new": new, "id": row_id, "old": old}).rowcount
                if n != 1:
                    raise RepairError(f"{p.section_number}: ordinal update of {row_id} touched {n} rows")
                manifest["ordinal_updates"].append([row_id, old, new])
            # highest first: (section_number, paragraph_path, path_ordinal) is unique, and each row moves up
            for row_id, old, new in sorted(p.path_ordinal_updates, key=lambda u: -u[1]):
                n = conn.execute("UPDATE cfr_records SET path_ordinal = %(new)s WHERE id = %(id)s::uuid AND path_ordinal = %(old)s",
                                 {"new": new, "id": row_id, "old": old}).rowcount
                if n != 1:
                    raise RepairError(f"{p.section_number}: path_ordinal update of {row_id} touched {n} rows")
                manifest["path_ordinal_updates"].append([row_id, old, new])
            retrieved = provenance[p.section_number.split(".")[0]]["retrieved_utc"]
            for r in p.inserts:
                got = conn.execute(INSERT_SQL, {**asdict(r), "retrieved_at": retrieved}).fetchall()
                if len(got) != 1:
                    raise RepairError(f"{r.section_number}{r.paragraph_path}#{r.path_ordinal}: insert returned {len(got)} rows")
                manifest["inserted"].append({**asdict(r), "id": got[0]["id"]})
        write_manifest(manifest)
    return manifest


def rollback(conn, manifest: Mapping[str, Any]) -> dict[str, int]:
    """Undo an applied manifest in one transaction: refuse if anything now references an inserted row; delete the
    inserted rows (only if their text is unchanged); restore the renumbered ordinals. updated_at is not restored
    (the set_updated_at trigger stamps every update; nothing in this repository reads it)."""
    ids = [r["id"] for r in manifest["inserted"]]
    with conn.transaction():
        _writable_transaction_check(conn)
        for fk in conn.execute(REFERENCING_SQL).fetchall():
            n = conn.execute(f'SELECT count(*) AS n FROM {fk["tbl"]} WHERE "{fk["col"]}" = ANY(%(ids)s::uuid[])', {"ids": ids}).fetchall()[0]["n"]
            if n:
                raise RepairError(f"rollback: {n} rows of {fk['tbl']}.{fk['col']} reference restored rows")
        deleted = 0
        for r in manifest["inserted"]:
            n = conn.execute("DELETE FROM cfr_records WHERE id = %(id)s::uuid AND content_sha256 = %(sha)s",
                             {"id": r["id"], "sha": r["content_sha256"]}).rowcount
            if n != 1:
                raise RepairError(f"rollback: inserted row {r['id']} missing or changed")
            deleted += n
        restored = 0
        for row_id, old, new in sorted(manifest["ordinal_updates"], key=lambda u: u[2]):
            n = conn.execute("UPDATE cfr_records SET ordinal = %(old)s WHERE id = %(id)s::uuid AND ordinal = %(new)s",
                             {"old": old, "id": row_id, "new": new}).rowcount
            if n != 1:
                raise RepairError(f"rollback: ordinal of {row_id} is not {new}")
            restored += n
        # lowest first: the restored rows are gone, so each row moves back down into a free place
        for row_id, old, new in sorted(manifest.get("path_ordinal_updates", []), key=lambda u: u[1]):
            n = conn.execute("UPDATE cfr_records SET path_ordinal = %(old)s WHERE id = %(id)s::uuid AND path_ordinal = %(new)s",
                             {"old": old, "id": row_id, "new": new}).rowcount
            if n != 1:
                raise RepairError(f"rollback: path_ordinal of {row_id} is not {new}")
            restored += n
    return {"deleted": deleted, "ordinals_restored": restored}
