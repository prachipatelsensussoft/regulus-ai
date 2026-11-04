"""Heading migrations (regulus.datamigrate.headings, scripts/build_heading_migrations.py): subpart titles and the subpart
heading on each row, a track of their own beside the fingerprinted row migrations. The database tests make and drop
their own schema (REGULUS_TEST_DATABASE_URL), as tests/test_datamigrate.py does."""
import importlib.util
import json
import os
import uuid
from pathlib import Path

import pytest

from regulus.datamigrate import headings as H
from regulus.datamigrate.payload import MigrationFileError

ROOT = Path(__file__).resolve().parents[1]
SUB = "11111111-1111-4111-8111-111111111111"
ROW1 = "22222222-2222-4222-8222-222222222222"
ROW2 = "33333333-3333-4333-8333-333333333333"
OLD_H = ["Part 117 - Current Good Manufacturing Practice", "Subpart G - Chain Program"]
NEW_H = ["Part 117 - Current Good Manufacturing Practice", "Subpart G - Supply-Chain Program"]


def _write(tmp: Path, version: int = 1, **over):
    titles = over.pop("titles", [{"id": SUB, "part": "117", "designation": "G", "old": "Chain Program", "new": "Supply-Chain Program"}])
    heads = over.pop("heads", [{"id": r, "section": "117.405", "old": OLD_H, "new": NEW_H} for r in (ROW1, ROW2)])
    header = {"version": version, "name": f"titles_{version}", "reference_applied": "2026-11-04", "description": "test",
              "provenance": {}, **over}
    H.write_file(tmp / f"{version:04d}_titles_{version}.json", header, titles, heads)


def test_a_heading_migration_round_trips_and_is_never_overwritten(tmp_path):
    _write(tmp_path)
    [m] = H.load_heading_migrations(tmp_path)
    assert (m.version, len(m.subpart_titles), len(m.parent_headings)) == (1, 1, 2)
    assert m.summary() == "0001_titles_1: 1 subpart titles, 2 rows' parent headings"
    with pytest.raises(FileExistsError):
        _write(tmp_path)


def test_malformed_or_empty_changes_are_refused(tmp_path):
    _write(tmp_path, titles=[{"id": SUB, "part": "117", "designation": "G", "old": "X", "new": "X"}], heads=[])
    with pytest.raises(MigrationFileError, match="changes nothing"):
        H.load_heading_migrations(tmp_path)
    for f in tmp_path.glob("*.json"):
        f.unlink()
    _write(tmp_path, titles=[], heads=[{"id": "not-a-uuid", "section": "1.1", "old": ["a"], "new": ["b"]}])
    with pytest.raises(MigrationFileError, match="malformed"):
        H.load_heading_migrations(tmp_path)


def test_versions_must_run_without_gaps(tmp_path):
    _write(tmp_path, version=2)
    with pytest.raises(MigrationFileError, match="expected version 1"):
        H.load_heading_migrations(tmp_path)


def test_no_directory_means_no_heading_migrations(tmp_path):
    assert H.load_heading_migrations(tmp_path / "absent") == []


def _builder():
    spec = importlib.util.spec_from_file_location("bhm", ROOT / "scripts" / "build_heading_migrations.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_builder_reads_the_ecfr_heading_after_the_first_em_dash_without_footnotes(tmp_path):
    (tmp_path / "part-117.xml").write_text(
        '<DIV6 N="G" TYPE="SUBPART"><HEAD>Subpart G&#x2014;Supply-Chain Program</HEAD></DIV6>'
        '<DIV6 N="G" TYPE="SUBPART" x="1"><HEAD>Subpart G&#x2014;Sequestrants\n<SU>1</SU>\n<FTREF/></HEAD></DIV6>', encoding="utf-8")
    (tmp_path / "part-610.xml").write_text(
        '<DIV6 N="C" TYPE="SUBPART"><HEAD>Subparts C&#x2014;D [Reserved]</HEAD></DIV6>', encoding="utf-8")
    titles, shas = _builder().official_titles(tmp_path)
    assert titles[("117", "G")] == "Sequestrants"            # the later heading of the same key wins; footnote removed
    assert ("610", "C") not in titles                        # a reserved range is not a title
    assert set(shas) == {"117", "610"}
    (tmp_path / "part-117.xml").write_text(
        '<DIV6 N="G" TYPE="SUBPART"><HEAD>Subpart G&#x2014;Supply-Chain Program</HEAD></DIV6>', encoding="utf-8")
    assert _builder().official_titles(tmp_path)[0][("117", "G")] == "Supply-Chain Program"   # not cut at the hyphen


def test_the_repository_heading_migrations_load():
    ms = H.load_heading_migrations()
    assert [m.version for m in ms] == list(range(1, len(ms) + 1))


# -- database ----------------------------------------------------------------------------------------------------------

TEST_URL = os.environ.get("REGULUS_TEST_DATABASE_URL", "").strip()
db = pytest.mark.skipif(not TEST_URL, reason="REGULUS_TEST_DATABASE_URL is not set")


@pytest.fixture
def env(tmp_path):
    from psycopg import sql
    from regulus.datamigrate import runner as R
    s = uuid.uuid4().hex[:10]
    target, ledger = f"regulus_head_test_{s}", f"regulus_head_test_{s}_ledger"
    conn = R.connect(TEST_URL)
    conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(target)))
    conn.execute(sql.SQL("CREATE TABLE {}.cfr_subparts (id uuid PRIMARY KEY, part_number text, designation text, title text)")
                 .format(sql.Identifier(target)))
    conn.execute(sql.SQL("CREATE TABLE {}.cfr_records (id uuid PRIMARY KEY, section_number text, parent_headings text[])")
                 .format(sql.Identifier(target)))
    conn.execute(sql.SQL("INSERT INTO {}.cfr_subparts VALUES (%s, '117', 'G', 'Chain Program')").format(sql.Identifier(target)), (SUB,))
    for r in (ROW1, ROW2):
        conn.execute(sql.SQL("INSERT INTO {}.cfr_records VALUES (%s, '117.405', %s)").format(sql.Identifier(target)), (r, OLD_H))
    _write(tmp_path)
    ms = H.load_heading_migrations(tmp_path)

    class Env:
        pass
    e = Env()
    e.conn, e.ms, e.kw = conn, ms, {"schema": ledger, "target": target}
    e.title = lambda: conn.execute(sql.SQL("SELECT title FROM {}.cfr_subparts").format(sql.Identifier(target))).fetchone()["title"]
    e.heads = lambda: [r["parent_headings"] for r in conn.execute(
        sql.SQL("SELECT parent_headings FROM {}.cfr_records ORDER BY id").format(sql.Identifier(target))).fetchall()]
    e.sql, e.target = sql, target
    yield e
    for x in (ledger, target):
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(x)))
    conn.close()


@pytest.mark.integration
@db
def test_apply_rerun_and_rollback(env):
    st = H.heading_status(env.conn, env.ms, **env.kw)
    assert [m.version for m in st.pending] == [1] and st.ok
    assert H.apply_headings(env.conn, env.ms, **env.kw)[0]["mode"] == "applied"
    assert env.title() == "Supply-Chain Program" and env.heads() == [NEW_H, NEW_H]
    assert H.apply_headings(env.conn, env.ms, **env.kw) == []                       # re-run: nothing to do
    assert H.rollback_headings(env.conn, env.ms, to_version=0, **env.kw) == [{"version": 1, "name": "titles_1"}]
    assert env.title() == "Chain Program" and env.heads() == [OLD_H, OLD_H]
    assert H.heading_status(env.conn, env.ms, **env.kw).recorded == ()


@pytest.mark.integration
@db
def test_a_copy_that_already_holds_the_new_values_is_adopted_and_a_mixed_one_refused(env):
    sql = env.sql
    env.conn.execute(sql.SQL("UPDATE {}.cfr_subparts SET title = 'Supply-Chain Program'").format(sql.Identifier(env.target)))
    env.conn.execute(sql.SQL("UPDATE {}.cfr_records SET parent_headings = %s").format(sql.Identifier(env.target)), (NEW_H,))
    assert H.apply_headings(env.conn, env.ms, **env.kw)[0]["mode"] == "adopted"
    H.rollback_headings(env.conn, env.ms, to_version=0, **env.kw)
    env.conn.execute(sql.SQL("UPDATE {}.cfr_records SET parent_headings = %s WHERE id = %s").format(sql.Identifier(env.target)),
                     (NEW_H, ROW1))                                                  # one row new, one old: mixed
    st = H.heading_status(env.conn, env.ms, **env.kw)
    assert not st.ok and "neither its old nor its new values" in st.problems[0]
    with pytest.raises(H.MigrationError):
        H.apply_headings(env.conn, env.ms, **env.kw)
    assert env.title() == "Chain Program"                                            # nothing written
