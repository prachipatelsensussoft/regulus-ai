"""python -m regulus.setup [status | apply | adopt | rollback --to N | semantic] [--database-url URL] [--confirm DBNAME]

    (no command) / apply   check the connection and the baseline, preview the pending migrations, ask, apply them
                           (--to N: only up to version N), then offer to rebuild the semantic corpus and index, verify
    status                 read-only: connection, ledger, fingerprint, pending migrations, semantic index state
    adopt                  a database copied from one the migrations were already applied to (same rows, no ledger):
                           record them without changing a row
    rollback --to N        undo the migrations newer than N, newest first
    semantic               rebuild the semantic corpus and index from the database (asks first)

The database is REGULUS_DATABASE_URL (environment or .env), or --database-url; its password is never printed. Every
command that writes asks for the database's name to be typed, or takes it as --confirm DBNAME (non-interactive use).
Exit status: 0 done / nothing to do, 1 refused, not confirmed or failed (nothing written by that step), 2 usage.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from regulus.config import PROJECT_ROOT
from regulus.datamigrate import (LEDGER_SCHEMA, MigrationError, MigrationFileError, adopt, apply, load_migrations,
                                 rollback, status)
from regulus.datamigrate.payload import INSERT_COLUMNS
from regulus.datamigrate.runner import TARGET_SCHEMA, connect

STAMP = "regulus_setup_stamp.json"   # in the semantic corpus's generated/ folder: the table fingerprint it was built from


def _say(text: str = "") -> None:
    print(text, flush=True)


def _url(arg: str | None) -> str:
    if arg:
        return arg
    from regulus.retrieval.db import DatabaseConfig
    return DatabaseConfig.from_env().url


def _confirm(dbname: str, given: str | None, action: str) -> bool:
    if given is not None:
        if given == dbname:
            return True
        _say(f"--confirm {given!r} is not this database's name ({dbname!r}); nothing done.")
        return False
    if not sys.stdin.isatty():
        _say(f"Not a terminal: pass --confirm {dbname} to {action}. Nothing done.")
        return False
    try:
        answer = input(f"Type the database name ({dbname}) to {action}, or press Enter to cancel: ").strip()
    except EOFError:
        answer = ""
    if answer != dbname:
        _say("Cancelled; nothing done.")
        return False
    return True


# -- checks ------------------------------------------------------------------------------------------------------------

def check_connection(conn) -> list[str]:
    """Problems that stop setup before any migration logic: missing table or columns, no write privilege."""
    info = conn.info
    version = conn.execute("SHOW server_version").fetchone()["server_version"]
    _say(f"  database {info.dbname} on {info.host}:{info.port} as {info.user} (PostgreSQL {version})")
    problems = []
    cols = {r["column_name"] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = 'cfr_records'",
        (TARGET_SCHEMA,)).fetchall()}
    if not cols:
        return [f"no table {TARGET_SCHEMA}.cfr_records: this is not a Regulus database (restore the baseline backup first)"]
    missing = [c for c in INSERT_COLUMNS if c not in cols]
    if missing:
        problems.append(f"cfr_records lacks columns {missing}: the schema is older than the baseline's")
    prisma = conn.execute("SELECT to_regclass('public._prisma_migrations') IS NOT NULL AS ok").fetchone()["ok"]
    if prisma:
        last = conn.execute("SELECT migration_name FROM public._prisma_migrations WHERE finished_at IS NOT NULL "
                            "ORDER BY migration_name DESC LIMIT 1").fetchone()
        _say(f"  schema: Prisma, latest migration {last['migration_name'] if last else 'none'} (not changed by setup)")
    priv = conn.execute("SELECT has_table_privilege(%s, 'INSERT, UPDATE, DELETE') AS ok",
                        (f'"{TARGET_SCHEMA}".cfr_records',)).fetchone()["ok"]
    if not priv:
        problems.append(f"{info.user} cannot write cfr_records")
    return problems


def describe(st, migrations) -> None:
    _say(f"  cfr_records: {st.fingerprint[0]} rows, fingerprint {st.fingerprint[1][:16]}")
    if st.applied:
        adopted = f" ({len(st.adopted)} adopted)" if st.adopted else ""
        _say(f"  ledger {LEDGER_SCHEMA}.cfr_migrations: versions 1-{len(st.applied)} recorded{adopted}")
    else:
        _say(f"  ledger {LEDGER_SCHEMA}.cfr_migrations: none recorded")
    _say(f"  this checkout: {len(migrations)} migrations")
    state = {"up_to_date": "UP TO DATE - nothing to apply", "pending": f"{len(st.pending)} pending",
             "adoptable": f"ALREADY CONTAINS migrations 1-{st.adoptable_upto} without a ledger (run: adopt)",
             "incompatible": "INCOMPATIBLE", "ledger_error": "LEDGER ERROR"}[st.state]
    _say(f"  state: {state}")
    if st.message and st.state in ("incompatible", "ledger_error", "adoptable"):
        _say(f"  {st.message}")


def preview(pending) -> None:
    ins = sum(len(m.insert) for m in pending)
    upd = sum(len(m.update) for m in pending)
    for m in pending:
        _say(f"    {m.summary()}  [{m.reference_applied}]")
        _say(f"      {m.description}")
    _say(f"  total: {ins} rows inserted, {upd} rows updated; no row deleted; each migration is one transaction")


# -- semantic index ----------------------------------------------------------------------------------------------------

def _semantic_paths():
    from regulus.retrieval.factory import SEMANTIC_CORPUS
    from regulus.retrieval.semantic.models import resolve_paths
    return SEMANTIC_CORPUS, resolve_paths(SEMANTIC_CORPUS).generated


def semantic_state(fp_sha: str) -> tuple[bool, str]:
    name, gen = _semantic_paths()
    if not (gen / "corpus.jsonl").exists() or not (gen / "vectors.npy").exists():
        return True, f"{name}: not built ({gen.relative_to(PROJECT_ROOT)} is missing)"
    try:
        stamp = json.loads((gen / STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True, f"{name}: built, but not by setup - which table state it reflects is unknown"
    if stamp.get("cfr_fingerprint") != fp_sha:
        return True, f"{name}: built from an earlier state of cfr_records ({str(stamp.get('cfr_fingerprint'))[:16]})"
    return False, f"{name}: current (built {stamp.get('built_at_utc')})"


def rebuild_semantic(url: str, fp_sha: str) -> bool:
    """Back up generated/, rebuild the corpus from the database, re-embed only new or changed items (reusing the
    backup's vectors), check the index, stamp it. Returns False on a failed step (the backup is kept either way)."""
    name, gen = _semantic_paths()
    env = {**os.environ, "REGULUS_DATABASE_URL": url,
           "PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT / "src"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    reuse: list[str] = []
    if (gen / "vectors.npy").exists():
        backup = gen.with_name(f"generated_backup_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
        shutil.copytree(gen, backup)
        _say(f"  previous index kept in {backup.relative_to(PROJECT_ROOT)}")
        reuse = ["--reuse-from", str(backup)]
    scripts = PROJECT_ROOT / "scripts"
    steps = [("corpus", [sys.executable, str(scripts / "build_semantic_corpus.py"), "--semantic-corpus", name]),
             ("index", [sys.executable, str(scripts / "build_semantic_index.py"), "--semantic-corpus", name, *reuse]),
             ("check", [sys.executable, str(scripts / "build_semantic_index.py"), "--semantic-corpus", name, "--check"])]
    for label, cmd in steps:
        _say(f"  {label}: {' '.join(Path(c).name if i < 2 else c for i, c in enumerate(cmd))}")
        if subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode != 0:
            _say(f"  semantic {label} step FAILED; the database is unaffected. Re-run: python -m regulus.setup semantic")
            return False
    (gen / STAMP).write_text(json.dumps({"cfr_fingerprint": fp_sha, "built_at_utc": datetime.now(timezone.utc).isoformat(
        timespec="seconds"), "corpus": name}, indent=1) + "\n", encoding="utf-8")
    return True


# -- commands ----------------------------------------------------------------------------------------------------------

def _semantic_step(conn, url: str, confirm: str | None, *, ask_always: bool = False) -> int:
    from regulus.datamigrate.runner import fingerprint
    fp = fingerprint(conn)
    needed, why = semantic_state(fp[1])
    _say(f"  {why}")
    if not needed and not ask_always:
        return 0
    _say("  Rebuilding reads cfr_records and guidance (read-only), embeds new or changed items with the pinned "
         "BAAI/bge-small-en-v1.5 (downloaded once if absent; minutes when reusing, much longer from scratch).")
    if not _confirm(conn.info.dbname, confirm, "rebuild the semantic corpus and index from it"):
        return 1
    return 0 if rebuild_semantic(url, fp[1]) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m regulus.setup", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="apply", choices=("apply", "status", "adopt", "rollback", "semantic"))
    ap.add_argument("--to", type=int, help="apply: the last version to apply; rollback: the version to return to (0 = baseline)")
    ap.add_argument("--database-url", help="default: REGULUS_DATABASE_URL from the environment or .env")
    ap.add_argument("--confirm", metavar="DBNAME", help="confirm a write non-interactively by naming the database")
    ap.add_argument("--skip-semantic", action="store_true", help="apply: do not offer the semantic index rebuild")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.command == "rollback" and a.to is None:
        ap.error("rollback needs --to N")

    _say("[1] Migrations in this checkout")
    try:
        migrations = load_migrations()
    except MigrationFileError as exc:
        _say(f"  {exc}")
        return 1
    _say(f"  {len(migrations)} files in {Path('data/migrations/cfr')}, checksums read")

    _say("[2] Connection")
    try:
        url = _url(a.database_url)
        conn = connect(url)
    except Exception as exc:   # noqa: BLE001 - report any connection failure plainly, without the URL
        _say(f"  cannot connect: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}")
        return 1
    try:
        conn.read_only = True   # everything up to the confirmed write is read-only, enforced by the server
        problems = check_connection(conn)
        for p in problems:
            _say(f"  PROBLEM: {p}")
        if problems:
            return 1

        _say("[3] Baseline and ledger")
        st = status(conn, migrations)
        describe(st, migrations)
        dbname = conn.info.dbname

        if a.command == "status":
            for m in st.pending:
                _say(f"    pending {m.summary()}")
            _say("[4] Semantic index")
            _say(f"  {semantic_state(st.fingerprint[1])[1]}")
            return 0 if st.state in ("up_to_date", "pending", "adoptable") else 1

        if a.command == "semantic":
            _say("[4] Semantic index")
            conn.read_only = False
            return _semantic_step(conn, url, a.confirm, ask_always=True)

        if a.command == "adopt":
            if st.state != "adoptable":
                _say("  Nothing to adopt." if st.state in ("pending", "up_to_date") else "  Refused; nothing changed.")
                return 0 if st.state in ("pending", "up_to_date") else 1
            _say(f"[4] Adopt: record migrations 1-{st.adoptable_upto} as already present (no row changes)")
            if not _confirm(dbname, a.confirm, "record them"):
                return 1
            conn.read_only = False
            for r in adopt(conn, migrations):
                _say(f"  adopted {r['version']:04d}_{r['name']}")
            return 0

        if a.command == "rollback":
            if st.state not in ("pending", "up_to_date"):
                _say("  Refused; nothing changed.")
                return 1
            undo = [m for m in migrations if a.to < m.version <= len(st.applied)]
            if not undo:
                _say(f"  Nothing to roll back to version {a.to}.")
                return 0
            _say(f"[4] Rollback to version {a.to}: undo, newest first")
            for m in reversed(undo):
                _say(f"    {m.summary()}")
            if not _confirm(dbname, a.confirm, f"undo {len(undo)} migrations"):
                return 1
            conn.read_only = False
            for r in rollback(conn, migrations, to_version=a.to):
                _say(f"  rolled back {r['version']:04d}_{r['name']}: {r['deleted']} rows deleted, {r['restored']} restored")
            _say("  The semantic index may now be ahead of the table: python -m regulus.setup semantic")
            return 0

        # apply
        if st.state in ("incompatible", "ledger_error"):
            _say("  Refused; nothing changed.")
            return 1
        if st.state == "adoptable":
            _say("  Run `python -m regulus.setup adopt` to record them; nothing changed.")
            return 1
        todo = [m for m in st.pending if a.to is None or m.version <= a.to]
        if todo:
            _say("[4] Preview")
            preview(todo)
            if not _confirm(dbname, a.confirm, f"apply {len(todo)} migrations"):
                return 1
            _say("[5] Apply")
            conn.read_only = False
            for r in apply(conn, migrations, upto=a.to):
                _say(f"  applied {r['version']:04d}_{r['name']}: {r['inserted']} inserted, {r['updated']} updated, verified")
        else:
            _say("[4] Nothing to apply")
        conn.read_only = False
        if not a.skip_semantic:
            _say("[6] Semantic index")
            if _semantic_step(conn, url, a.confirm):
                return 1
        _say("[7] Verify")
        conn.read_only = True
        final = status(conn, migrations)
        _say(f"  ledger: versions 1-{len(final.applied)}; state {final.state}; fingerprint {final.fingerprint[1][:16]}")
        return 0 if final.state == "up_to_date" or (a.to is not None and final.state == "pending") else 1
    except MigrationError as exc:
        _say(f"  REFUSED: {exc}")
        return 1
    finally:
        conn.close()
