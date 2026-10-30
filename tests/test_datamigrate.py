"""regulus.datamigrate: migration files, the runner and the setup command.

The file tests run everywhere. The database tests (marker `integration`) need a PostgreSQL server where the role may
create schemas: REGULUS_TEST_DATABASE_URL (never REGULUS_DATABASE_URL - they must not touch a working database). Each
test creates its own schemas (regulus_mig_test_<random>), a small cfr_records table with the real table's columns,
key and checks, and three synthetic migrations, and drops only those schemas afterwards. Scenarios: the original
baseline, partly applied, fully applied and re-run, an incompatible baseline, a table changed after a migration was
recorded, a copy that already holds the changes (adopt), rollback, a referencing row blocking rollback, a migration file
changed after it was applied, and a failure after the writes leaving nothing behind.

    REGULUS_TEST_DATABASE_URL=postgresql://user:pw@localhost:5432/scratch pytest -m integration tests/test_datamigrate.py
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from regulus.datamigrate import payload as P
from regulus.datamigrate import runner as R

# -- files (no database) -----------------------------------------------------------------------------------------------

BASELINE_ROWS = 65341   # 15 Sep 2026 baseline: 65,264 live non-reserved rows + 77 reserved


def _row(rid: str, section: str, path: str, po: int, ordinal: int, body: str) -> dict:
    return {"id": rid, "title": "21", "chapter": "I", "subchapter_code": "G", "part_number": section.split(".")[0],
            "subpart_id": None, "section_number": section, "paragraph_path": path, "path_ordinal": po, "ordinal": ordinal,
            "section_heading": f"Section {section}", "parent_headings": ["Part"], "body_text": body,
            "effective_date": None, "last_amended_date": None, "fr_citation": "", "ecfr_permalink": "",
            "govinfo_pdf_url": "", "source_endpoint": "test", "retrieved_at": "2026-10-21T10:00:00+00:00",
            "snapshot_date": "2026-09-03", "content_sha256": R.hashlib.sha256(body.encode()).hexdigest(),
            "source_payload_sha256": "ab" * 32, "is_reserved": False, "incorporated_standard_id": None, "removed_on": None}


def _key(r: dict) -> dict:
    return {**{k: r[k] for k in ("id", "section_number", "paragraph_path", "path_ordinal", "ordinal", "content_sha256",
                                 "is_reserved", "removed_on")}, "body_md5": R.body_md5(r["body_text"])}


def _ids(n: int) -> list[str]:
    return sorted(str(uuid.uuid4()) for _ in range(n))


def synthetic(tmp: Path) -> tuple[list[dict], list[P.Migration]]:
    """A 6-row baseline and three migrations: insert + renumber; a path swap inside one section (needs the two-phase
    move under the unique key); insert + move + renumber."""
    a, b, c, d, e, f, g, h = _ids(8)
    base = [_row(a, "1.1", "(a)", 0, 0, "Alpha."), _row(b, "1.1", "(b)", 0, 1, "Bravo."), _row(c, "1.1", "(c)", 0, 2, "Charlie."),
            _row(d, "1.2", "", 0, 0, "Delta."), _row(e, "1.2", "(a)", 0, 1, "Echo."), _row(f, "1.2", "(b)", 0, 2, "Foxtrot.")]
    states = [{r["id"]: dict(r) for r in base}]
    steps = [
        ([_row(g, "1.1", "(b)", 1, 2, "Bravo, continued.")], [(c, ("(c)", 0, 2), ("(c)", 0, 3))]),
        ([], [(e, ("(a)", 0, 1), ("(b)", 0, 1)), (f, ("(b)", 0, 2), ("(a)", 0, 2))]),
        ([_row(h, "1.2", "(c)", 0, 3, "Golf.")], [(d, ("", 0, 0), ("(d)", 0, 0)), (a, ("(a)", 0, 0), ("(a)", 0, 9))]),
    ]
    out = []
    for k, (ins, upd) in enumerate(steps, start=1):
        cur = {i: dict(r) for i, r in states[-1].items()}
        for r in ins:
            cur[r["id"]] = dict(r)
        for rid, _, new in upd:
            cur[rid].update(zip(("paragraph_path", "path_ordinal", "ordinal"), new))
        header = {"version": k, "name": f"step_{k}", "reference_applied": "2026-10-2{k}", "description": f"step {k}",
                  "provenance": {},
                  "fingerprint_before": dict(zip(("rows", "sha256"), R.fingerprint_rows(_key(r) for r in states[-1].values()))),
                  "fingerprint_after": dict(zip(("rows", "sha256"), R.fingerprint_rows(_key(r) for r in cur.values())))}
        P.write_file(tmp / f"{k:04d}_step_{k}.json", header, ins,
                     [{"id": rid, "section": cur[rid]["section_number"], "old": list(o), "new": list(n)} for rid, o, n in upd])
        states.append(cur)
    return base, P.load_migrations(tmp)


def test_the_repository_migrations_load_chain_and_start_from_the_baseline():
    ms = P.load_migrations()
    assert [m.version for m in ms] == list(range(1, len(ms) + 1)) and len(ms) >= 7
    assert ms[0].fingerprint_before[0] == BASELINE_ROWS
    assert [len(m.insert) for m in ms[:7]] == [16, 128, 525, 0, 0, 0, 701]
    assert ms[6].fingerprint_after[0] == BASELINE_ROWS + 16 + 128 + 525 + 701


def test_checksum_ignores_crlf_line_endings(tmp_path):
    _, ms = synthetic(tmp_path)
    p = ms[0].path
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert P.load_file(p).checksum == ms[0].checksum


def test_a_published_migration_file_is_never_overwritten(tmp_path):
    _, ms = synthetic(tmp_path)
    with pytest.raises(FileExistsError):
        P.write_file(ms[0].path, {"version": 1, "name": "step_1"}, [], [])


def test_files_must_be_numbered_without_gaps_and_chain(tmp_path):
    _, ms = synthetic(tmp_path)
    ms[1].path.rename(tmp_path / "0004_step_2.json")
    with pytest.raises(P.MigrationFileError):
        P.load_migrations(tmp_path)


def test_a_broken_chain_is_refused(tmp_path):
    _, ms = synthetic(tmp_path)
    doc = json.loads(ms[2].path.read_text())
    doc["fingerprint_before"]["sha256"] = "0" * 64
    ms[2].path.write_text(json.dumps(doc))
    with pytest.raises(P.MigrationFileError, match="fingerprint before"):
        P.load_migrations(tmp_path)


def test_a_row_inserted_and_updated_in_one_migration_is_refused(tmp_path):
    _, ms = synthetic(tmp_path)
    doc = json.loads(ms[0].path.read_text())
    doc["update"].append({"id": doc["insert"][0]["id"], "old": ["(b)", 1, 2], "new": ["(b)", 1, 3]})
    ms[0].path.write_text(json.dumps(doc))
    with pytest.raises(P.MigrationFileError, match="more than once|both inserted"):
        P.load_file(ms[0].path)


def test_python_and_sql_fingerprints_are_defined_alike():
    rows = [{"id": "00000000-0000-0000-0000-000000000002", "section_number": "1.1", "paragraph_path": "(a)", "path_ordinal": 0,
             "ordinal": 1, "content_sha256": "c" * 64, "body_md5": R.body_md5("é"), "is_reserved": True, "removed_on": "2026-01-02"}]
    expect = "|".join(["00000000-0000-0000-0000-000000000002", "1.1", "(a)", "0", "1", "c" * 64,
                       R.hashlib.md5("é".encode()).hexdigest(), "t", "2026-01-02"])
    assert R.fingerprint_rows(rows) == (1, R.hashlib.sha256(expect.encode()).hexdigest())


# -- database ----------------------------------------------------------------------------------------------------------

TEST_URL = os.environ.get("REGULUS_TEST_DATABASE_URL", "").strip()
db = pytest.mark.skipif(not TEST_URL, reason="REGULUS_TEST_DATABASE_URL is not set")

TABLE_DDL = """
CREATE TABLE {s}.cfr_records (
    id uuid PRIMARY KEY, title text NOT NULL, chapter text NOT NULL, subchapter_code text NOT NULL, part_number text NOT NULL,
    subpart_id uuid, section_number text NOT NULL, paragraph_path text NOT NULL DEFAULT ''
        CHECK (paragraph_path = '' OR paragraph_path ~ '^(\\([A-Za-z0-9]{{1,4}}\\))+$'),
    path_ordinal integer NOT NULL DEFAULT 0 CHECK (path_ordinal >= 0), ordinal integer NOT NULL DEFAULT 0,
    section_heading text NOT NULL DEFAULT '', parent_headings text[] NOT NULL DEFAULT '{{}}', body_text text NOT NULL,
    effective_date date, last_amended_date date, fr_citation text NOT NULL DEFAULT '', ecfr_permalink text NOT NULL DEFAULT '',
    govinfo_pdf_url text NOT NULL DEFAULT '', source_endpoint text NOT NULL, retrieved_at timestamptz NOT NULL DEFAULT now(),
    snapshot_date date NOT NULL, content_sha256 text NOT NULL, source_payload_sha256 text NOT NULL DEFAULT '',
    is_reserved boolean NOT NULL DEFAULT false, incorporated_standard_id uuid, removed_on date,
    created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', body_text)) STORED,
    CONSTRAINT cfr_records_unique_paragraph UNIQUE (section_number, paragraph_path, path_ordinal));
CREATE TABLE {s}.answer_citations (id serial PRIMARY KEY, record_id uuid REFERENCES {s}.cfr_records(id))
"""


@pytest.fixture
def env(tmp_path):
    import psycopg
    from psycopg import sql
    suffix = uuid.uuid4().hex[:10]
    target, ledger = f"regulus_mig_test_{suffix}", f"regulus_mig_test_{suffix}_ledger"
    base, ms = synthetic(tmp_path)
    conn = R.connect(TEST_URL)
    conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(target)))
    for stmt in TABLE_DDL.split(";\n"):
        conn.execute(sql.SQL(stmt).format(s=sql.Identifier(target)))
    cols = sql.SQL(", ").join(sql.Identifier(c) for c in P.INSERT_COLUMNS)
    t = sql.Identifier(target, "cfr_records")
    conn.execute(sql.SQL("INSERT INTO {t} ({c}) SELECT {c} FROM json_populate_recordset(NULL::{t}, %s::json)").format(t=t, c=cols),
                 (json.dumps(base),))

    class Env:
        pass
    e = Env()
    e.conn, e.ms, e.target, e.ledger, e.tmp = conn, ms, target, ledger, tmp_path
    e.kw = {"schema": ledger, "target": target}
    e.fp = lambda: R.fingerprint(conn, target)
    e.ledger_rows = lambda: (conn.execute(sql.SQL("SELECT version, mode FROM {}.cfr_migrations ORDER BY 1").format(sql.Identifier(ledger)))
                             .fetchall() if conn.execute("SELECT to_regclass(%s) IS NOT NULL AS ok", (f'"{ledger}".cfr_migrations',)).fetchone()["ok"] else [])
    yield e
    for s in (ledger, target, ledger + "_b"):
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(s)))   # only the schemas this test made
    conn.close()


@pytest.mark.integration
@db
def test_original_baseline_applies_everything_and_a_rerun_changes_nothing(env):
    st = R.status(env.conn, env.ms, **env.kw)
    assert st.state == "pending" and [m.version for m in st.pending] == [1, 2, 3]
    done = R.apply(env.conn, env.ms, **env.kw)
    assert [d["version"] for d in done] == [1, 2, 3]
    assert env.fp() == env.ms[-1].fingerprint_after
    assert env.ledger_rows() == [{"version": v, "mode": "applied"} for v in (1, 2, 3)]
    assert R.apply(env.conn, env.ms, **env.kw) == []          # re-run: nothing pending, nothing written
    assert R.status(env.conn, env.ms, **env.kw).state == "up_to_date"
    assert env.ledger_rows() == [{"version": v, "mode": "applied"} for v in (1, 2, 3)]


@pytest.mark.integration
@db
def test_a_partly_applied_database_gets_only_the_rest(env):
    assert [d["version"] for d in R.apply(env.conn, env.ms, upto=1, **env.kw)] == [1]
    st = R.status(env.conn, env.ms, **env.kw)
    assert st.state == "pending" and [m.version for m in st.pending] == [2, 3] and env.fp() == env.ms[0].fingerprint_after
    assert [d["version"] for d in R.apply(env.conn, env.ms, **env.kw)] == [2, 3]
    assert env.fp() == env.ms[-1].fingerprint_after


@pytest.mark.integration
@db
def test_an_incompatible_baseline_is_refused_and_nothing_is_written(env):
    from psycopg import sql
    env.conn.execute(sql.SQL("UPDATE {} SET body_text = 'Changed.' WHERE paragraph_path = '(b)' AND section_number = '1.1'")
                     .format(sql.Identifier(env.target, "cfr_records")))
    before = env.fp()
    st = R.status(env.conn, env.ms, **env.kw)
    assert st.state == "incompatible" and "not the baseline" in st.message
    with pytest.raises(R.MigrationError):
        R.apply(env.conn, env.ms, **env.kw)
    assert env.fp() == before and env.ledger_rows() == []


@pytest.mark.integration
@db
def test_a_table_changed_after_a_recorded_migration_stops_the_runner(env):
    from psycopg import sql
    R.apply(env.conn, env.ms, upto=1, **env.kw)
    env.conn.execute(sql.SQL("UPDATE {} SET ordinal = 50 WHERE section_number = '1.2' AND paragraph_path = ''")
                     .format(sql.Identifier(env.target, "cfr_records")))
    st = R.status(env.conn, env.ms, **env.kw)
    assert st.state == "incompatible" and "changed since 0001_step_1" in st.message
    with pytest.raises(R.MigrationError):
        R.apply(env.conn, env.ms, **env.kw)
    assert env.ledger_rows() == [{"version": 1, "mode": "applied"}]


@pytest.mark.integration
@db
def test_a_copy_that_already_holds_the_changes_is_adopted_without_writing_rows(env):
    R.apply(env.conn, env.ms, upto=2, **env.kw)
    other = {"schema": env.ledger + "_b", "target": env.target}          # the same table seen by a fresh ledger
    st = R.status(env.conn, env.ms, **other)
    assert st.state == "adoptable" and st.adoptable_upto == 2
    with pytest.raises(R.MigrationError):
        R.apply(env.conn, env.ms, **other)                              # never re-applied on top of itself
    fp = env.fp()
    assert [d["version"] for d in R.adopt(env.conn, env.ms, **other)] == [1, 2]
    assert env.fp() == fp
    st = R.status(env.conn, env.ms, **other)
    assert st.state == "pending" and st.adopted == [1, 2] and [m.version for m in st.pending] == [3]
    assert [d["version"] for d in R.apply(env.conn, env.ms, **other)] == [3]


@pytest.mark.integration
@db
def test_rollback_returns_to_the_baseline_and_can_be_reapplied(env):
    base = env.fp()
    R.apply(env.conn, env.ms, **env.kw)
    assert [d["version"] for d in R.rollback(env.conn, env.ms, to_version=1, **env.kw)] == [3, 2]
    assert env.fp() == env.ms[0].fingerprint_after
    R.rollback(env.conn, env.ms, to_version=0, **env.kw)
    assert env.fp() == base and env.ledger_rows() == []
    R.apply(env.conn, env.ms, **env.kw)
    assert env.fp() == env.ms[-1].fingerprint_after


@pytest.mark.integration
@db
def test_rollback_refuses_while_a_row_it_would_delete_is_referenced(env):
    from psycopg import sql
    R.apply(env.conn, env.ms, **env.kw)
    env.conn.execute(sql.SQL("INSERT INTO {} (record_id) VALUES (%s)").format(sql.Identifier(env.target, "answer_citations")),
                     (env.ms[2].insert[0]["id"],))
    fp = env.fp()
    with pytest.raises(R.MigrationError, match="reference"):
        R.rollback(env.conn, env.ms, to_version=0, **env.kw)
    assert env.fp() == fp and len(env.ledger_rows()) == 3


@pytest.mark.integration
@db
def test_a_migration_file_changed_after_it_was_applied_stops_everything(env):
    R.apply(env.conn, env.ms, upto=1, **env.kw)
    doc = json.loads(env.ms[0].path.read_text())
    doc["description"] = "edited"
    env.ms[0].path.write_text(json.dumps(doc))
    ms = P.load_migrations(env.tmp)
    st = R.status(env.conn, ms, **env.kw)
    assert st.state == "ledger_error" and "must never change" in st.message
    with pytest.raises(R.MigrationError):
        R.apply(env.conn, ms, **env.kw)


@pytest.mark.integration
@db
def test_a_failure_after_the_writes_leaves_nothing_behind(env, tmp_path):
    doc = json.loads(env.ms[0].path.read_text())
    doc["fingerprint_after"]["sha256"] = "f" * 64                       # the verification after the writes fails
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / env.ms[0].path.name).write_text(json.dumps(doc))
    ms = P.load_migrations(bad_dir)
    fp = env.fp()
    with pytest.raises(R.MigrationError, match="after"):
        R.apply(env.conn, ms, **env.kw)
    assert env.fp() == fp
    assert env.ledger_rows() == []


@pytest.mark.integration
@db
def test_setup_cli_status_is_read_only_and_apply_needs_the_database_name(env, monkeypatch, capsys):
    from regulus.setup import cli
    monkeypatch.setattr(cli, "load_migrations", lambda: env.ms)
    monkeypatch.setattr(cli, "TARGET_SCHEMA", env.target)
    monkeypatch.setattr(R, "TARGET_SCHEMA", env.target)
    for fn in ("status", "apply", "adopt", "rollback"):
        orig = getattr(R, fn)
        monkeypatch.setattr(cli, fn, lambda *a, _o=orig, **k: _o(*a, **{**k, "schema": env.ledger, "target": env.target}))
    fp = env.fp()
    assert cli.main(["status", "--database-url", TEST_URL]) == 0
    assert env.fp() == fp and env.ledger_rows() == []
    assert cli.main(["apply", "--database-url", TEST_URL, "--confirm", "wrong_name", "--skip-semantic"]) == 1   # not confirmed
    assert env.fp() == fp and env.ledger_rows() == []
    dbname = env.conn.info.dbname
    assert cli.main(["apply", "--database-url", TEST_URL, "--confirm", dbname, "--skip-semantic"]) == 0
    assert env.fp() == env.ms[-1].fingerprint_after
    assert "state up_to_date" in capsys.readouterr().out
