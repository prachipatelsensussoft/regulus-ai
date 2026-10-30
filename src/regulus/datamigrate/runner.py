"""Apply, adopt and roll back cfr_records data migrations on one PostgreSQL database (psycopg only; no psql).

Safety, in the order the code enforces it:
  * the table's fingerprint (FINGERPRINT_SQL: every row's id, section, paragraph path, path ordinal, ordinal, content
    hash, body-text hash, reserved flag and removal date) must be the one the next migration was built from - a
    database that is not the 15 Sep 2026 baseline plus exactly the recorded migrations is "incompatible" and nothing
    is written;
  * each migration runs in one transaction that first takes an advisory lock (one runner at a time) and a SHARE ROW
    EXCLUSIVE lock on cfr_records (readers keep reading; no other writer), re-reads the ledger and the fingerprint,
    checks the old (path, path ordinal, ordinal) of every row it updates and that no row it inserts exists, writes,
    then verifies the fingerprint after and every inserted row column by column before it records itself in the
    ledger - any difference raises and nothing is committed;
  * the ledger (regulus_data.cfr_migrations, history in regulus_data.cfr_migration_events) lives outside Prisma's
    `public` schema; a version is recorded once (primary key), so a migration cannot run twice;
  * a migration file whose checksum differs from the one recorded when it was applied stops every command;
  * rollback undoes the newest migrations one transaction each, refusing while anything references a row it would
    delete, and verifies the fingerprint before each one.
Nothing is dropped or truncated. Rows a migration does not name are never touched.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from regulus.datamigrate.payload import INSERT_COLUMNS, Migration

LEDGER_SCHEMA = "regulus_data"
TARGET_SCHEMA = "public"
RUNNER_VERSION = "1"
_LOCK_KEY = 7_316_021_015   # pg_advisory_xact_lock key shared by every runner (any constant, fixed forever)

# one line per row, in id order (uuid order is byte order, the same on every server and locale); dates and booleans
# are formatted explicitly so DateStyle cannot change the text, and the body is hashed as UTF-8 whatever the server
# encoding. regulus.datamigrate.runner.fingerprint_rows computes the same value in Python.
FINGERPRINT_SQL = """
SELECT count(*) AS rows, encode(sha256(convert_to(coalesce(string_agg(concat_ws('|', id::text, section_number,
       paragraph_path, path_ordinal::text, ordinal::text, content_sha256, md5(convert_to(body_text, 'UTF8')),
       CASE WHEN is_reserved THEN 't' ELSE 'f' END, coalesce(to_char(removed_on, 'YYYY-MM-DD'), '')),
       E'\\n' ORDER BY id), ''), 'UTF8')), 'hex') AS sha256
FROM {table}
"""

LEDGER_DDL = """
CREATE SCHEMA IF NOT EXISTS {schema};
CREATE TABLE IF NOT EXISTS {schema}.cfr_migrations (
    version integer PRIMARY KEY CHECK (version > 0),
    name text NOT NULL,
    checksum text NOT NULL,
    mode text NOT NULL CHECK (mode IN ('applied', 'adopted')),
    fingerprint_before text NOT NULL,
    fingerprint_after text NOT NULL,
    rows_inserted integer NOT NULL,
    rows_updated integer NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now(),
    applied_by text NOT NULL DEFAULT current_user,
    client text NOT NULL
);
CREATE TABLE IF NOT EXISTS {schema}.cfr_migration_events (
    id bigserial PRIMARY KEY,
    version integer NOT NULL,
    name text NOT NULL,
    action text NOT NULL CHECK (action IN ('apply', 'adopt', 'rollback')),
    checksum text NOT NULL,
    at timestamptz NOT NULL DEFAULT now(),
    by text NOT NULL DEFAULT current_user,
    client text NOT NULL,
    detail jsonb NOT NULL DEFAULT '{{}}'::jsonb
);
"""

REFERENCING_SQL = """
SELECT c.conrelid::regclass::text AS tbl, a.attname AS col
FROM pg_constraint c JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
WHERE c.contype = 'f' AND c.confrelid = %s::regclass AND c.conrelid <> %s::regclass
"""


class MigrationError(RuntimeError):
    """The database is not in a state a migration may be applied to or rolled back from; nothing was written."""


@dataclass
class Status:
    state: str                       # up_to_date | pending | adoptable | incompatible | ledger_error
    fingerprint: tuple[int, str]
    applied: list[int] = field(default_factory=list)
    adopted: list[int] = field(default_factory=list)
    pending: list[Migration] = field(default_factory=list)
    adoptable_upto: int | None = None
    expected: tuple[int, str] | None = None
    message: str = ""


def connect(url: str, *, application_name: str = "regulus-setup") -> psycopg.Connection:
    """A writable connection, autocommit outside explicit transactions (each migration opens its own)."""
    return psycopg.connect(url, autocommit=True, row_factory=dict_row, application_name=application_name,
                           connect_timeout=15)


def client_string() -> str:
    return f"regulus.datamigrate {RUNNER_VERSION}; {platform.system()} {platform.release()}; python {platform.python_version()}"


def _table(target: str) -> sql.Composed:
    return sql.Identifier(target, "cfr_records")


def fingerprint(conn: psycopg.Connection, target: str = TARGET_SCHEMA) -> tuple[int, str]:
    row = conn.execute(sql.SQL(FINGERPRINT_SQL).format(table=_table(target))).fetchone()
    return int(row["rows"]), row["sha256"]


def fingerprint_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    """FINGERPRINT_SQL in Python, over dicts with id, section_number, paragraph_path, path_ordinal, ordinal,
    content_sha256, body_md5 (hex MD5 of the UTF-8 body), is_reserved and removed_on (None or 'YYYY-MM-DD')."""
    lines = sorted("|".join((r["id"], r["section_number"], r["paragraph_path"], str(r["path_ordinal"]), str(r["ordinal"]),
                             r["content_sha256"], r["body_md5"], "t" if r["is_reserved"] else "f", r["removed_on"] or ""))
                   for r in rows)
    return len(lines), hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def body_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _fmt(fp: tuple[int, str] | None) -> str:
    return "none" if fp is None else f"{fp[0]} rows / {fp[1][:16]}"


# -- ledger ------------------------------------------------------------------------------------------------------------

def _ledger_exists(conn: psycopg.Connection, schema: str) -> bool:
    return conn.execute("SELECT to_regclass(%s) IS NOT NULL AS ok", (f'"{schema}".cfr_migrations',)).fetchone()["ok"]


def _ledger(conn: psycopg.Connection, schema: str) -> list[dict[str, Any]]:
    if not _ledger_exists(conn, schema):
        return []
    return conn.execute(sql.SQL("SELECT version, name, checksum, mode FROM {}.cfr_migrations ORDER BY version")
                        .format(sql.Identifier(schema))).fetchall()


def _check_ledger(rows: Sequence[Mapping[str, Any]], migrations: Sequence[Migration]) -> str:
    """'' when the ledger is versions 1..k of these files with their checksums, else what is wrong."""
    for i, r in enumerate(rows, start=1):
        if r["version"] != i:
            return f"the ledger has version {r['version']} where {i} was expected"
        if i > len(migrations):
            return (f"the database has migration {i} ({r['name']}), which this checkout does not have: "
                    "update the repository before running setup")
        m = migrations[i - 1]
        if r["name"] != m.name or r["checksum"] != m.checksum:
            return (f"{m.label} was recorded from a file with checksum {r['checksum'][:16]} ({r['name']}); the file in "
                    f"this checkout has {m.checksum[:16]} - migration files must never change once applied")
    return ""


def status(conn: psycopg.Connection, migrations: Sequence[Migration], *, schema: str = LEDGER_SCHEMA,
           target: str = TARGET_SCHEMA) -> Status:
    """Read-only: what the database holds and what would be applied."""
    rows = _ledger(conn, schema)
    fp = fingerprint(conn, target)
    st = Status("incompatible", fp, applied=[r["version"] for r in rows],
                adopted=[r["version"] for r in rows if r["mode"] == "adopted"])
    problem = _check_ledger(rows, migrations)
    if problem:
        st.state, st.message = "ledger_error", problem
        return st
    k = len(rows)
    if not migrations:
        st.state, st.message = "up_to_date", "no migrations in this checkout"
        return st
    st.expected = migrations[k - 1].fingerprint_after if k else migrations[0].fingerprint_before
    if fp == st.expected:
        st.pending = list(migrations[k:])
        st.state = "pending" if st.pending else "up_to_date"
        return st
    if not k:
        for m in migrations:
            if fp == m.fingerprint_after:
                st.state, st.adoptable_upto = "adoptable", m.version
                st.message = (f"the table already holds migrations 1-{m.version} (fingerprint {_fmt(fp)}) but no ledger "
                              "records them - a copy of a database they were applied to; adopt records them without "
                              "changing any row")
                return st
        st.message = (f"cfr_records is not the baseline this repository's migrations start from: fingerprint {_fmt(fp)}, "
                      f"baseline {_fmt(st.expected)}, and it matches no migration's result either. Restore the 15 Sep "
                      "2026 baseline backup into a new database, or ask for a current dump; nothing was changed")
    else:
        st.message = (f"cfr_records changed since {migrations[k - 1].label} was recorded: fingerprint {_fmt(fp)}, "
                      f"expected {_fmt(st.expected)}. Something other than this runner wrote to it; nothing was changed")
    return st


# -- one transaction ---------------------------------------------------------------------------------------------------

def _begin(conn: psycopg.Connection, schema: str, target: str) -> None:
    """Inside a transaction: refuse a read-only session, take the runner lock and the table lock, create the ledger."""
    if conn.execute("SELECT current_setting('transaction_read_only') AS ro").fetchone()["ro"] != "off":
        raise MigrationError("the connection is read-only (default_transaction_read_only); setup needs a role that can "
                             "write cfr_records")
    conn.execute("SET LOCAL statement_timeout = 0")
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
    conn.execute(sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(_table(target)))
    if not _ledger_exists(conn, schema):
        for statement in LEDGER_DDL.split(";\n"):
            if statement.strip():
                conn.execute(sql.SQL(statement).format(schema=sql.Identifier(schema)))


def _expect(conn: psycopg.Connection, target: str, want: tuple[int, str], what: str) -> None:
    got = fingerprint(conn, target)
    if got != want:
        raise MigrationError(f"{what}: fingerprint {_fmt(got)}, expected {_fmt(want)}; rolled back, nothing changed")


def _ledger_top(conn: psycopg.Connection, schema: str) -> int:
    return conn.execute(sql.SQL("SELECT coalesce(max(version), 0) AS v, count(*) AS n FROM {}.cfr_migrations")
                        .format(sql.Identifier(schema))).fetchone()["v"]


def _event(conn: psycopg.Connection, schema: str, m: Migration, action: str, detail: Mapping[str, Any]) -> None:
    conn.execute(sql.SQL("INSERT INTO {}.cfr_migration_events (version, name, action, checksum, client, detail) "
                         "VALUES (%s, %s, %s, %s, %s, %s::jsonb)").format(sql.Identifier(schema)),
                 (m.version, m.name, action, m.checksum, client_string(), json.dumps(detail)))


def _check_rows(conn: psycopg.Connection, target: str, wanted: Sequence[tuple[str, str, int, int]], label: str) -> None:
    """Every (id, path, path_ordinal, ordinal) is the row's current value."""
    if not wanted:
        return
    bad = conn.execute(sql.SQL(
        "SELECT u.id::text AS id, c.paragraph_path, c.path_ordinal, c.ordinal, c.id IS NULL AS missing "
        "FROM json_to_recordset(%s::json) AS u(id uuid, p text, po int, o int) LEFT JOIN {} c ON c.id = u.id "
        "WHERE c.id IS NULL OR c.paragraph_path <> u.p OR c.path_ordinal <> u.po OR c.ordinal <> u.o LIMIT 5")
        .format(_table(target)), (json.dumps([{"id": i, "p": p, "po": po, "o": o} for i, p, po, o in wanted]),)).fetchall()
    if bad:
        raise MigrationError(f"{label}: {len(bad)}+ rows are not as recorded, e.g. {bad[0]}; rolled back, nothing changed")


def _insert_mismatches(conn: psycopg.Connection, target: str, rows: Sequence[Mapping[str, Any]]) -> int:
    """How many of `rows` are absent or differ from the stored row in any INSERT_COLUMNS column."""
    if not rows:
        return 0
    cols = [sql.Identifier(c) for c in INSERT_COLUMNS]
    same = sql.SQL("({}) IS NOT DISTINCT FROM ({})").format(
        sql.SQL(", ").join(sql.SQL("c.{}").format(c) for c in cols), sql.SQL(", ").join(sql.SQL("p.{}").format(c) for c in cols))
    n = conn.execute(sql.SQL("SELECT count(*) AS n FROM json_populate_recordset(NULL::{t}, %s::json) p JOIN {t} c ON c.id = p.id "
                             "WHERE {same}").format(t=_table(target), same=same), (json.dumps(list(rows)),)).fetchone()["n"]
    return len(rows) - n


def _base36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while True:
        n, d = divmod(n, 36)
        out = digits[d] + out
        if not n:
            return out


def _move(conn: psycopg.Connection, target: str, moves: Sequence[tuple[str, tuple, tuple]], label: str) -> None:
    """Set each row (id, from, to) to `to` = (path, path_ordinal, ordinal). Rows whose (path, path_ordinal) changes first
    move to a temporary path unique in the table ("(ZZZZ)(<n>)", which the path check accepts and no eCFR paragraph
    has), so the unique (section, path, path_ordinal) key holds after every statement whatever the order."""
    if not moves:
        return
    shifting = [i for i, a, b in moves if a[:2] != b[:2]]
    if len(shifting) > 36 ** 4:
        raise MigrationError(f"{label}: too many path changes for one transaction")
    if shifting:
        n = conn.execute(sql.SQL("UPDATE {} c SET paragraph_path = u.p, path_ordinal = 0 FROM json_to_recordset(%s::json) "
                                 "AS u(id uuid, p text) WHERE c.id = u.id").format(_table(target)),
                         (json.dumps([{"id": i, "p": f"(ZZZZ)({_base36(k)})"} for k, i in enumerate(shifting)]),)).rowcount
        if n != len(shifting):
            raise MigrationError(f"{label}: {n} of {len(shifting)} rows moved to a temporary path; rolled back")
    n = conn.execute(sql.SQL("UPDATE {} c SET paragraph_path = u.p, path_ordinal = u.po, ordinal = u.o "
                             "FROM json_to_recordset(%s::json) AS u(id uuid, p text, po int, o int) WHERE c.id = u.id")
                     .format(_table(target)),
                     (json.dumps([{"id": i, "p": b[0], "po": b[1], "o": b[2]} for i, a, b in moves]),)).rowcount
    if n != len(moves):
        raise MigrationError(f"{label}: {n} of {len(moves)} rows updated; rolled back")


def _apply_one(conn: psycopg.Connection, m: Migration, schema: str, target: str) -> dict[str, Any]:
    with conn.transaction():
        _begin(conn, schema, target)
        top = _ledger_top(conn, schema)
        if top != m.version - 1:
            raise MigrationError(f"{m.label}: the ledger's latest version is {top}, not {m.version - 1} "
                                 "(another runner, or already applied); nothing changed")
        _expect(conn, target, m.fingerprint_before, f"{m.label} before")
        _check_rows(conn, target, [(u["id"], *u["old"]) for u in m.update], f"{m.label} old values")
        ids = [r["id"] for r in m.insert]
        if ids:
            clash = conn.execute(sql.SQL("SELECT id::text AS id FROM {} WHERE id = ANY(%s::uuid[]) LIMIT 5").format(_table(target)),
                                 (ids,)).fetchall()
            if clash:
                raise MigrationError(f"{m.label}: rows it inserts already exist, e.g. {clash[0]['id']}; nothing changed")
        _move(conn, target, [(u["id"], tuple(u["old"]), tuple(u["new"])) for u in m.update], m.label)
        if m.insert:
            cols = sql.SQL(", ").join(sql.Identifier(c) for c in INSERT_COLUMNS)
            n = conn.execute(sql.SQL("INSERT INTO {t} ({c}) SELECT {c} FROM json_populate_recordset(NULL::{t}, %s::json)")
                             .format(t=_table(target), c=cols), (json.dumps(list(m.insert)),)).rowcount
            if n != len(m.insert):
                raise MigrationError(f"{m.label}: {n} of {len(m.insert)} rows inserted; rolled back")
        _expect(conn, target, m.fingerprint_after, f"{m.label} after")
        if _insert_mismatches(conn, target, m.insert):
            raise MigrationError(f"{m.label}: inserted rows differ from the file; rolled back")
        conn.execute(sql.SQL("INSERT INTO {}.cfr_migrations (version, name, checksum, mode, fingerprint_before, "
                             "fingerprint_after, rows_inserted, rows_updated, client) VALUES (%s, %s, %s, 'applied', %s, %s, %s, %s, %s)")
                     .format(sql.Identifier(schema)),
                     (m.version, m.name, m.checksum, m.fingerprint_before[1], m.fingerprint_after[1], len(m.insert),
                      len(m.update), client_string()))
        detail = {"inserted": len(m.insert), "updated": len(m.update), "fingerprint_after": m.fingerprint_after[1]}
        _event(conn, schema, m, "apply", detail)
    return {"version": m.version, "name": m.name, **detail}


def apply(conn: psycopg.Connection, migrations: Sequence[Migration], *, upto: int | None = None,
          schema: str = LEDGER_SCHEMA, target: str = TARGET_SCHEMA) -> list[dict[str, Any]]:
    """Apply the pending migrations up to `upto` (default all), each in its own transaction. Refuses unless the
    database is exactly the state the first pending migration starts from."""
    st = status(conn, migrations, schema=schema, target=target)
    if st.state not in ("pending", "up_to_date"):
        raise MigrationError(f"{st.state}: {st.message}")
    done = []
    for m in st.pending:
        if upto is not None and m.version > upto:
            break
        done.append(_apply_one(conn, m, schema, target))
    return done


def adopt(conn: psycopg.Connection, migrations: Sequence[Migration], *, schema: str = LEDGER_SCHEMA,
          target: str = TARGET_SCHEMA) -> list[dict[str, Any]]:
    """Record migrations 1..k as already present, without changing a row, for a database whose table is exactly the
    state after k (a copy of a database they were applied to) and that has no ledger."""
    st = status(conn, migrations, schema=schema, target=target)
    if st.state != "adoptable":
        raise MigrationError(f"{st.state}: nothing to adopt. {st.message}")
    k = st.adoptable_upto
    with conn.transaction():
        _begin(conn, schema, target)
        if _ledger_top(conn, schema):
            raise MigrationError("the ledger is no longer empty; nothing recorded")
        _expect(conn, target, migrations[k - 1].fingerprint_after, f"adopt 1-{k}")
        # inserted rows: every column but the place (path, path ordinal, ordinal), which later migrations may move
        for m in migrations[:k]:
            later = {u["id"]: u["new"] for n in migrations[m.version:k] for u in n.update}
            rows = [{**r, **dict(zip(("paragraph_path", "path_ordinal", "ordinal"), later[r["id"]]))} if r["id"] in later else r
                    for r in m.insert]
            if _insert_mismatches(conn, target, rows):
                raise MigrationError(f"adopt: rows {m.label} inserts differ from the file; nothing recorded")
        for m in migrations[:k]:
            conn.execute(sql.SQL("INSERT INTO {}.cfr_migrations (version, name, checksum, mode, fingerprint_before, "
                                 "fingerprint_after, rows_inserted, rows_updated, client) VALUES (%s, %s, %s, 'adopted', %s, %s, %s, %s, %s)")
                         .format(sql.Identifier(schema)),
                         (m.version, m.name, m.checksum, m.fingerprint_before[1], m.fingerprint_after[1], len(m.insert),
                          len(m.update), client_string()))
            _event(conn, schema, m, "adopt", {"fingerprint_after": m.fingerprint_after[1]})
    return [{"version": m.version, "name": m.name, "adopted": True} for m in migrations[:k]]


def _rollback_one(conn: psycopg.Connection, m: Migration, schema: str, target: str) -> dict[str, Any]:
    with conn.transaction():
        _begin(conn, schema, target)
        top = _ledger_top(conn, schema)
        if top != m.version:
            raise MigrationError(f"rollback {m.label}: the ledger's latest version is {top}; nothing changed")
        _expect(conn, target, m.fingerprint_after, f"rollback {m.label} before")
        if _insert_mismatches(conn, target, m.insert):
            raise MigrationError(f"rollback {m.label}: rows it inserted were changed since; nothing changed")
        ids = [r["id"] for r in m.insert]
        if ids:
            tbl = f'"{target}".cfr_records'
            for fk in conn.execute(REFERENCING_SQL, (tbl, tbl)).fetchall():
                n = conn.execute(sql.SQL("SELECT count(*) AS n FROM {} WHERE {} = ANY(%s::uuid[])")
                                 .format(sql.SQL(fk["tbl"]), sql.Identifier(fk["col"])), (ids,)).fetchone()["n"]
                if n:
                    raise MigrationError(f"rollback {m.label}: {n} rows of {fk['tbl']}.{fk['col']} reference rows it "
                                         "inserted; nothing changed")
            n = conn.execute(sql.SQL("DELETE FROM {} WHERE id = ANY(%s::uuid[])").format(_table(target)), (ids,)).rowcount
            if n != len(ids):
                raise MigrationError(f"rollback {m.label}: {n} of {len(ids)} rows deleted; rolled back")
        _move(conn, target, [(u["id"], tuple(u["new"]), tuple(u["old"])) for u in m.update], f"rollback {m.label}")
        _expect(conn, target, m.fingerprint_before, f"rollback {m.label} after")
        conn.execute(sql.SQL("DELETE FROM {}.cfr_migrations WHERE version = %s").format(sql.Identifier(schema)), (m.version,))
        detail = {"deleted": len(ids), "restored": len(m.update), "fingerprint_after": m.fingerprint_before[1]}
        _event(conn, schema, m, "rollback", detail)
    return {"version": m.version, "name": m.name, **detail}


def rollback(conn: psycopg.Connection, migrations: Sequence[Migration], *, to_version: int,
             schema: str = LEDGER_SCHEMA, target: str = TARGET_SCHEMA) -> list[dict[str, Any]]:
    """Undo every applied (or adopted) migration newer than `to_version`, newest first, one transaction each."""
    st = status(conn, migrations, schema=schema, target=target)
    if st.state not in ("pending", "up_to_date"):
        raise MigrationError(f"{st.state}: {st.message}")
    if not 0 <= to_version <= len(st.applied):
        raise MigrationError(f"cannot roll back to {to_version}: versions 1-{len(st.applied)} are applied")
    return [_rollback_one(conn, migrations[v - 1], schema, target) for v in range(len(st.applied), to_version, -1)]
