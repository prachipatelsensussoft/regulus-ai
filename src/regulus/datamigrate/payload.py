"""Migration files: data/migrations/cfr/NNNN_<name>.json, one per change, applied in version order.

    {"format": 1, "version": 3, "name": "...", "reference_applied": "2026-10-23", "description": "...",
     "provenance": {...},                                     # where the change came from (files, eCFR payloads)
     "fingerprint_before": {"rows": 66010, "sha256": "..."},  # regulus.datamigrate.runner.FINGERPRINT_SQL
     "fingerprint_after": {"rows": 66535, "sha256": "..."},
     "insert": [{<every column of INSERT_COLUMNS>}, ...],     # one row per line
     "update": [{"id": "...", "section": "...", "old": [path, path_ordinal, ordinal], "new": [...]}, ...]}

A file is written once by scripts/build_cfr_migrations.py and never edited: its checksum (SHA-256 of the bytes, line
endings normalised to LF so a Windows checkout with core.autocrlf gives the same value) is recorded in the ledger when
it is applied, and a file whose checksum no longer matches its ledger row stops the runner.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from regulus.config import PROJECT_ROOT

MIGRATIONS_DIR = PROJECT_ROOT / "data" / "migrations" / "cfr"
FORMAT = 1
_NAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.json$")

# the columns an inserted row carries; the others take their defaults (created_at/updated_at now(), search_vector
# generated, no embedding - as on the reference database, where no restored row has been embedded)
INSERT_COLUMNS = ("id", "title", "chapter", "subchapter_code", "part_number", "subpart_id", "section_number",
                  "paragraph_path", "path_ordinal", "ordinal", "section_heading", "parent_headings", "body_text",
                  "effective_date", "last_amended_date", "fr_citation", "ecfr_permalink", "govinfo_pdf_url",
                  "source_endpoint", "retrieved_at", "snapshot_date", "content_sha256", "source_payload_sha256",
                  "is_reserved", "incorporated_standard_id", "removed_on")


class MigrationFileError(RuntimeError):
    """A migration file is missing, malformed, out of sequence or does not chain to its predecessor."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path
    checksum: str
    description: str
    reference_applied: str
    provenance: Mapping[str, Any]
    fingerprint_before: tuple[int, str]
    fingerprint_after: tuple[int, str]
    insert: tuple[Mapping[str, Any], ...]
    update: tuple[Mapping[str, Any], ...]

    @property
    def label(self) -> str:
        return f"{self.version:04d}_{self.name}"

    def summary(self) -> str:
        paths = sum(1 for u in self.update if u["old"][0] != u["new"][0])
        path_ordinals = sum(1 for u in self.update if u["old"][1] != u["new"][1])
        ordinals = sum(1 for u in self.update if u["old"][2] != u["new"][2])
        return (f"{self.label}: insert {len(self.insert)} rows, update {len(self.update)} rows "
                f"({paths} paragraph paths, {path_ordinals} path ordinals, {ordinals} ordinals)")


def checksum(raw: bytes) -> str:
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def _fp(value: Any, where: str) -> tuple[int, str]:
    if not (isinstance(value, dict) and isinstance(value.get("rows"), int) and re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256")))):
        raise MigrationFileError(f"{where}: fingerprint must be {{rows, sha256}}")
    return value["rows"], value["sha256"]


def _triple(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 3 and isinstance(value[0], str)
            and isinstance(value[1], int) and isinstance(value[2], int))


def load_file(path: Path) -> Migration:
    m = _NAME.match(path.name)
    if not m:
        raise MigrationFileError(f"{path.name}: expected NNNN_<name>.json")
    raw = path.read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise MigrationFileError(f"{path.name}: not JSON ({exc})") from exc
    if doc.get("format") != FORMAT:
        raise MigrationFileError(f"{path.name}: format {doc.get('format')!r}, this runner reads format {FORMAT}")
    if doc.get("version") != int(m.group(1)) or doc.get("name") != m.group(2):
        raise MigrationFileError(f"{path.name}: version/name inside the file differ from the file name")
    inserts, updates = doc.get("insert", []), doc.get("update", [])
    ids: set[str] = set()
    for r in inserts:
        missing = [c for c in INSERT_COLUMNS if c not in r]
        if missing:
            raise MigrationFileError(f"{path.name}: inserted row {r.get('id')} lacks {missing}")
        ids.add(r["id"])
    for u in updates:
        if not (isinstance(u.get("id"), str) and _triple(u.get("old")) and _triple(u.get("new"))):
            raise MigrationFileError(f"{path.name}: malformed update {str(u)[:120]}")
        if u["id"] in ids:
            raise MigrationFileError(f"{path.name}: row {u['id']} is both inserted and updated, or updated twice")
        if u["old"] == u["new"]:
            raise MigrationFileError(f"{path.name}: update of {u['id']} changes nothing")
        ids.add(u["id"])
    if len(ids) != len(inserts) + len(updates):
        raise MigrationFileError(f"{path.name}: a row id appears more than once")
    return Migration(doc["version"], doc["name"], path, checksum(raw), doc.get("description", ""),
                     doc.get("reference_applied", ""), doc.get("provenance", {}),
                     _fp(doc.get("fingerprint_before"), path.name), _fp(doc.get("fingerprint_after"), path.name),
                     tuple(inserts), tuple(updates))


def load_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Every migration in version order, checked to be numbered 1, 2, 3 ... and to chain: each one's fingerprint
    before is its predecessor's fingerprint after."""
    if not directory.is_dir():
        raise MigrationFileError(f"no migrations directory {directory}")
    migrations = [load_file(p) for p in sorted(directory.glob("*.json"))]
    for i, m in enumerate(migrations, start=1):
        if m.version != i:
            raise MigrationFileError(f"{m.path.name}: expected version {i} (versions are 1, 2, 3 ... without gaps)")
        if i > 1 and m.fingerprint_before != migrations[i - 2].fingerprint_after:
            raise MigrationFileError(f"{m.path.name}: its fingerprint before is not {migrations[i - 2].label}'s after")
    return migrations


def write_file(path: Path, header: Mapping[str, Any], inserts: Sequence[Mapping[str, Any]],
               updates: Sequence[Mapping[str, Any]]) -> str:
    """Write a migration file - the header as indented JSON, then one inserted row / one update per line, so a diff
    or review reads row by row - and return its checksum. Refuses to overwrite: a published migration is never
    rewritten."""
    def rows(items: Iterable[Mapping[str, Any]]) -> str:
        lines = [json.dumps(dict(i), ensure_ascii=False, sort_keys=True) for i in items]
        return "[\n" + ",\n".join(lines) + "\n]" if lines else "[]"

    head = json.dumps({"format": FORMAT, **header}, ensure_ascii=False, indent=1)
    text = head[:-2] + ',\n "insert": ' + rows(inserts) + ',\n "update": ' + rows(updates) + "\n}\n"
    raw = text.encode("utf-8")
    json.loads(text)   # the hand-joined text is valid JSON
    with open(path, "xb") as fh:
        fh.write(raw)
    return checksum(raw)
