"""Heading migrations: corrections to subpart titles (cfr_subparts.title) and to the subpart headings copied onto
regulation rows (cfr_records.parent_headings).

The row migrations (data/migrations/cfr, regulus.datamigrate.runner) cannot carry these: their table fingerprint covers
where each row sits (path, path ordinal, ordinal), not its headings, so a heading-only change has the same fingerprint
before and after and the cfr runner could not tell whether it was applied. Headings therefore have their own track:

    data/migrations/headings/NNNN_<name>.json, applied in version order by `python -m regulus.setup` after the row
    migrations; ledger regulus_data.heading_migrations.

    {"format": 1, "version": 1, "name": "...", "reference_applied": "2026-11-04", "description": "...",
     "provenance": {...},
     "subpart_titles": [{"id": "<uuid>", "part": "117", "designation": "G", "old": "Chain Program",
                         "new": "Supply-Chain Program"}, ...],
     "parent_headings": [{"id": "<uuid>", "section": "117.405", "old": ["Part 117 - ...", "Subpart G - Chain Program"],
                          "new": ["Part 117 - ...", "Subpart G - Supply-Chain Program"]}, ...]}

Every value is checked, never assumed: a migration is applied only where every current value is its "old" value; one
whose every value is already its "new" value (a copy of a database it was applied to) is recorded as adopted without a
write; anything else stops with nothing changed. Rollback restores the "old" values the same way. A file is written once
and never edited; its checksum (LF-normalised, as regulus.datamigrate.payload) is recorded in the ledger.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import psycopg
from psycopg import sql

from regulus.config import PROJECT_ROOT
from regulus.datamigrate.payload import MigrationFileError, checksum
from regulus.datamigrate.runner import _LOCK_KEY, LEDGER_SCHEMA, TARGET_SCHEMA, MigrationError, client_string

HEADINGS_DIR = PROJECT_ROOT / "data" / "migrations" / "headings"
FORMAT = 1
_NAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.json$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

LEDGER_DDL = """
CREATE SCHEMA IF NOT EXISTS {schema};
CREATE TABLE IF NOT EXISTS {schema}.heading_migrations (
    version integer PRIMARY KEY,
    name text NOT NULL,
    checksum text NOT NULL,
    mode text NOT NULL CHECK (mode IN ('applied', 'adopted')),
    subpart_titles integer NOT NULL,
    parent_headings integer NOT NULL,
    client text NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT now()
)"""


@dataclass(frozen=True)
class HeadingMigration:
    version: int
    name: str
    path: Path
    checksum: str
    description: str
    subpart_titles: tuple[Mapping[str, Any], ...]
    parent_headings: tuple[Mapping[str, Any], ...]

    @property
    def label(self) -> str:
        return f"{self.version:04d}_{self.name}"

    def summary(self) -> str:
        return f"{self.label}: {len(self.subpart_titles)} subpart titles, {len(self.parent_headings)} rows' parent headings"


def load_file(path: Path) -> HeadingMigration:
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
    titles, heads = doc.get("subpart_titles", []), doc.get("parent_headings", [])
    for kind, items, ok in (("subpart title", titles, lambda v: isinstance(v, str) and v.strip() != ""),
                            ("parent headings", heads, lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v))):
        seen: set[str] = set()
        for it in items:
            if not (isinstance(it.get("id"), str) and _UUID.match(it["id"]) and ok(it.get("old")) and ok(it.get("new"))):
                raise MigrationFileError(f"{path.name}: malformed {kind} change {str(it)[:120]}")
            if it["old"] == it["new"]:
                raise MigrationFileError(f"{path.name}: {kind} change of {it['id']} changes nothing")
            if it["id"] in seen:
                raise MigrationFileError(f"{path.name}: {kind} of {it['id']} changed twice")
            seen.add(it["id"])
    if not titles and not heads:
        raise MigrationFileError(f"{path.name}: no changes")
    return HeadingMigration(doc["version"], doc["name"], path, checksum(raw), doc.get("description", ""),
                            tuple(titles), tuple(heads))


def load_heading_migrations(directory: Path = HEADINGS_DIR) -> list[HeadingMigration]:
    """Every heading migration in version order, numbered 1, 2, 3 ... without gaps; [] when the directory is absent."""
    if not directory.is_dir():
        return []
    migrations = [load_file(p) for p in sorted(directory.glob("*.json"))]
    for i, m in enumerate(migrations, start=1):
        if m.version != i:
            raise MigrationFileError(f"{m.path.name}: expected version {i} (versions are 1, 2, 3 ... without gaps)")
    return migrations


def headings_set(migrations: Sequence[HeadingMigration]) -> dict[str, list[list[str]]]:
    """{row id: [parent_headings, ...]} the heading migrations set: the cfr runner's adopt accepts them on a row a cfr
    migration inserted (a copy taken after the heading migrations ran), and the cfr builder undoes them."""
    out: dict[str, list[list[str]]] = {}
    for m in migrations:
        for c in m.parent_headings:
            out.setdefault(c["id"], []).append(list(c["new"]))
    return out


def write_file(path: Path, header: Mapping[str, Any], subpart_titles: Sequence[Mapping[str, Any]],
               parent_headings: Sequence[Mapping[str, Any]]) -> str:
    """Write a heading migration - the header indented, then one change per line - and return its checksum. Refuses to
    overwrite: a published migration is never rewritten."""
    def rows(items: Sequence[Mapping[str, Any]]) -> str:
        lines = [json.dumps(dict(i), ensure_ascii=False, sort_keys=True) for i in items]
        return "[\n" + ",\n".join(lines) + "\n]" if lines else "[]"

    head = json.dumps({"format": FORMAT, **header}, ensure_ascii=False, indent=1)
    text = head[:-2] + ',\n "subpart_titles": ' + rows(subpart_titles) + ',\n "parent_headings": ' + rows(parent_headings) + "\n}\n"
    json.loads(text)
    raw = text.encode("utf-8")
    with open(path, "xb") as fh:
        fh.write(raw)
    return checksum(raw)


# -- database ---------------------------------------------------------------------------------------------------------

def _ledger_exists(conn: psycopg.Connection, schema: str) -> bool:
    return conn.execute("SELECT to_regclass(%s) IS NOT NULL AS ok", (f'"{schema}".heading_migrations',)).fetchone()["ok"]


def _ledger(conn: psycopg.Connection, schema: str) -> dict[int, Mapping[str, Any]]:
    if not _ledger_exists(conn, schema):
        return {}
    rows = conn.execute(sql.SQL("SELECT version, name, checksum, mode FROM {}.heading_migrations ORDER BY version")
                        .format(sql.Identifier(schema))).fetchall()
    return {r["version"]: r for r in rows}


def _current(conn: psycopg.Connection, m: HeadingMigration, target: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    t = conn.execute(sql.SQL("SELECT id::text AS id, title FROM {}.cfr_subparts WHERE id = ANY(%s::uuid[])")
                     .format(sql.Identifier(target)), ([c["id"] for c in m.subpart_titles],)).fetchall()
    h = conn.execute(sql.SQL("SELECT id::text AS id, parent_headings FROM {}.cfr_records WHERE id = ANY(%s::uuid[])")
                     .format(sql.Identifier(target)), ([c["id"] for c in m.parent_headings],)).fetchall()
    return {r["id"]: r["title"] for r in t}, {r["id"]: list(r["parent_headings"] or []) for r in h}


def state_of(conn: psycopg.Connection, m: HeadingMigration, target: str = TARGET_SCHEMA) -> str:
    """'old' (every value is its old value: apply it), 'new' (every value is its new value: it is present), or
    'mixed' (anything else: missing rows or other values - refuse)."""
    titles, heads = _current(conn, m, target)
    values = [(titles.get(c["id"]), c) for c in m.subpart_titles] + [(heads.get(c["id"]), c) for c in m.parent_headings]
    if all(v == c["old"] for v, c in values):
        return "old"
    if all(v == c["new"] for v, c in values):
        return "new"
    return "mixed"


@dataclass(frozen=True)
class HeadingStatus:
    recorded: tuple[int, ...]
    pending: tuple[HeadingMigration, ...]   # not recorded; their values are the old ones
    present: tuple[HeadingMigration, ...]   # not recorded; their values are already the new ones (adopt)
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def heading_status(conn: psycopg.Connection, migrations: Sequence[HeadingMigration], *, schema: str = LEDGER_SCHEMA,
                   target: str = TARGET_SCHEMA) -> HeadingStatus:
    ledger = _ledger(conn, schema)
    problems, pending, present = [], [], []
    by_version = {m.version: m for m in migrations}
    for v, r in ledger.items():
        m = by_version.get(v)
        if m is None:
            problems.append(f"the ledger records heading migration {v} ({r['name']}), which this checkout does not have")
        elif r["checksum"] != m.checksum:
            problems.append(f"{m.label} was changed after it was recorded (checksum differs)")
    if sorted(ledger) != list(range(1, len(ledger) + 1)):
        problems.append(f"the heading ledger is not 1..n: {sorted(ledger)}")
    for m in migrations:
        if m.version in ledger:
            continue
        s = state_of(conn, m, target)
        if s == "old":
            pending.append(m)
        elif s == "new":
            present.append(m)
        else:
            problems.append(f"{m.label}: the database holds neither its old nor its new values; nothing will be changed")
    return HeadingStatus(tuple(sorted(ledger)), tuple(pending), tuple(present), tuple(problems))


def _begin(conn: psycopg.Connection, schema: str, target: str) -> None:
    if conn.execute("SELECT current_setting('transaction_read_only') AS ro").fetchone()["ro"] != "off":
        raise MigrationError("the connection is read-only; heading migrations need a role that can write cfr_records")
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
    for t in ("cfr_subparts", "cfr_records"):
        conn.execute(sql.SQL("LOCK TABLE {}.{} IN SHARE ROW EXCLUSIVE MODE").format(sql.Identifier(target), sql.Identifier(t)))
    if not _ledger_exists(conn, schema):
        for statement in LEDGER_DDL.split(";\n"):
            if statement.strip():
                conn.execute(sql.SQL(statement).format(schema=sql.Identifier(schema)))


def _write(conn: psycopg.Connection, m: HeadingMigration, target: str, *, forward: bool) -> None:
    src, dst = ("old", "new") if forward else ("new", "old")
    for c in m.subpart_titles:
        n = conn.execute(sql.SQL("UPDATE {}.cfr_subparts SET title = %s WHERE id = %s::uuid AND title = %s")
                         .format(sql.Identifier(target)), (c[dst], c["id"], c[src])).rowcount
        if n != 1:
            raise MigrationError(f"{m.label}: subpart {c['id']} is not as recorded; rolled back")
    for c in m.parent_headings:
        n = conn.execute(sql.SQL("UPDATE {}.cfr_records SET parent_headings = %s::text[] WHERE id = %s::uuid "
                                 "AND parent_headings = %s::text[]").format(sql.Identifier(target)),
                         (c[dst], c["id"], c[src])).rowcount
        if n != 1:
            raise MigrationError(f"{m.label}: row {c['id']} is not as recorded; rolled back")


def _record(conn: psycopg.Connection, m: HeadingMigration, schema: str, mode: str) -> None:
    conn.execute(sql.SQL("INSERT INTO {}.heading_migrations (version, name, checksum, mode, subpart_titles, parent_headings, "
                         "client) VALUES (%s, %s, %s, %s, %s, %s, %s)").format(sql.Identifier(schema)),
                 (m.version, m.name, m.checksum, mode, len(m.subpart_titles), len(m.parent_headings), client_string()))


def apply_headings(conn: psycopg.Connection, migrations: Sequence[HeadingMigration], *, schema: str = LEDGER_SCHEMA,
                   target: str = TARGET_SCHEMA) -> list[dict[str, Any]]:
    """Apply every unrecorded heading migration whose values are the old ones, and record as adopted every one whose
    values are already the new ones - each in its own transaction, re-checked under the lock. Refuses (nothing written)
    when the status shows a problem."""
    st = heading_status(conn, migrations, schema=schema, target=target)
    if not st.ok:
        raise MigrationError("; ".join(st.problems))
    done = []
    for m in sorted(st.pending + st.present, key=lambda x: x.version):
        with conn.transaction():
            _begin(conn, schema, target)
            if m.version in _ledger(conn, schema):
                raise MigrationError(f"{m.label}: recorded by another runner meanwhile; nothing changed")
            s = state_of(conn, m, target)
            if s == "old":
                _write(conn, m, target, forward=True)
                if state_of(conn, m, target) != "new":
                    raise MigrationError(f"{m.label}: values differ after the update; rolled back")
                mode = "applied"
            elif s == "new":
                mode = "adopted"
            else:
                raise MigrationError(f"{m.label}: the database changed meanwhile; nothing changed")
            _record(conn, m, schema, mode)
        done.append({"version": m.version, "name": m.name, "mode": mode,
                     "subpart_titles": len(m.subpart_titles), "parent_headings": len(m.parent_headings)})
    return done


def rollback_headings(conn: psycopg.Connection, migrations: Sequence[HeadingMigration], *, to_version: int,
                      schema: str = LEDGER_SCHEMA, target: str = TARGET_SCHEMA) -> list[dict[str, Any]]:
    """Restore the old values of every recorded heading migration newer than `to_version`, newest first."""
    st = heading_status(conn, migrations, schema=schema, target=target)
    if not st.ok:
        raise MigrationError("; ".join(st.problems))
    if not 0 <= to_version <= len(st.recorded):
        raise MigrationError(f"cannot roll headings back to {to_version}: versions 1-{len(st.recorded)} are recorded")
    done = []
    for v in range(len(st.recorded), to_version, -1):
        m = migrations[v - 1]
        with conn.transaction():
            _begin(conn, schema, target)
            if state_of(conn, m, target) != "new":
                raise MigrationError(f"rollback {m.label}: its values were changed since; nothing changed")
            _write(conn, m, target, forward=False)
            conn.execute(sql.SQL("DELETE FROM {}.heading_migrations WHERE version = %s").format(sql.Identifier(schema)),
                         (m.version,))
        done.append({"version": m.version, "name": m.name})
    return done
