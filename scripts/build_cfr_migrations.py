"""Build data/migrations/cfr/ from the repairs made on the reference database (21-30 Oct 2026). Read-only on the database.

    python scripts/build_cfr_migrations.py [--runs ~/regulus_eval_runs] [--out data/migrations/cfr] [--check]

Authoring tool, run once on the machine that holds the repair records (the manifests, SQL and rollback files each
repair wrote when it was applied; their SHA-256 is recorded in each migration's provenance). It is not needed to apply
the migrations: the files it writes carry every value.

How the values are obtained without the 15 Sep baseline: the live table is read, then every repair is undone in
memory, newest first, from its own record - a row it inserted must be present with exactly the inserted text, a value
it set must be the current value - giving the table before each repair and finally the baseline. Each migration is the
difference between consecutive states, with the fingerprint of both (regulus.datamigrate.runner.fingerprint_rows,
checked against FINGERPRINT_SQL on the live table). Any record that does not fit the table stops the build.
--check rebuilds in memory and compares with the files already written instead of writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psycopg import sql  # noqa: E402

from regulus.datamigrate.payload import INSERT_COLUMNS, MIGRATIONS_DIR, load_migrations, write_file  # noqa: E402
from regulus.datamigrate.runner import FINGERPRINT_SQL, body_md5, fingerprint_rows  # noqa: E402
from regulus.ingestion.dropped_text import content_hash  # noqa: E402
from regulus.retrieval import Database  # noqa: E402

KEY_SQL = """
SELECT id::text AS id, section_number, paragraph_path, path_ordinal, ordinal, content_sha256,
       md5(convert_to(body_text, 'UTF8')) AS body_md5, is_reserved, to_char(removed_on, 'YYYY-MM-DD') AS removed_on
FROM cfr_records
"""
FULL_SQL = "SELECT row_to_json(x) AS r FROM (SELECT {cols} FROM cfr_records WHERE id = ANY(%(ids)s::uuid[])) x"


class BuildError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- the repairs, as forward operations -------------------------------------------------------------------------------
# ("insert", row) | ("delete", row) | ("update", id, old, new) with old/new {path?, po?, ord?} (only the fields set)

def manifest_ops(path: Path, reverse: bool = False) -> list[tuple]:
    """An apply manifest (renumber then insert), or with reverse=True a rollback manifest (delete then restore)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    ids = [u[0] for u in d["ordinal_updates"]]
    if len(ids) != len(set(ids)):
        raise BuildError(f"{path.name}: a row is renumbered twice")
    rows = [{"id": r["id"], "section_number": r["section_number"], "paragraph_path": r["paragraph_path"],
             "path_ordinal": r["path_ordinal"], "ordinal": r["ordinal"], "content_sha256": r["content_sha256"],
             "body_md5": body_md5(r["body_text"]), "is_reserved": False, "removed_on": None} for r in d["inserted"]]
    for r, src in zip(rows, d["inserted"]):
        if content_hash(src["body_text"]) != src["content_sha256"]:
            raise BuildError(f"{path.name}: inserted row {r['id']}'s text does not hash to its content_sha256")
    if not reverse:
        return ([("update", i, {"ord": o}, {"ord": n}) for i, o, n in d["ordinal_updates"]]
                + [("insert", r) for r in rows])
    return [("delete", r) for r in rows] + [("update", i, {"ord": n}, {"ord": o}) for i, o, n in d["ordinal_updates"]]


_UPDATE = re.compile(r"UPDATE cfr_records SET (.*?) WHERE (.*?);")
_PAIR = re.compile(r"(\w+) = ('(?:[^']|'')*'|-?\d+)")
_FIELD = {"paragraph_path": "path", "path_ordinal": "po", "ordinal": "ord"}


def _values(text: str) -> dict:
    out = {}
    for col, val in _PAIR.findall(text):
        out[col] = val[1:-1].replace("''", "'") if val.startswith("'") else int(val)
    return out


def sql_net(path: Path) -> dict[str, tuple[dict, dict]]:
    """{id: (old, new)} of a repair SQL file: the values each row's first UPDATE requires, the values its last one
    sets; every UPDATE of a row after its first must require what the previous one set (the two-phase chains)."""
    state: dict[str, tuple[dict, dict]] = {}
    n = 0
    for sets, where in _UPDATE.findall(path.read_text(encoding="utf-8")):
        s, w = _values(sets), _values(where)
        rid = w.pop("id")
        w.pop("section_number", None)
        if not set(s) | set(w) <= set(_FIELD):
            raise BuildError(f"{path.name}: unexpected columns in {sets} / {where}")
        s = {_FIELD[k]: v for k, v in s.items()}
        w = {_FIELD[k]: v for k, v in w.items()}
        if rid in state:
            old, cur = state[rid]
            if any(cur.get(k) != v for k, v in w.items()):
                raise BuildError(f"{path.name}: {rid} requires {w} after being set to {cur}")
            state[rid] = (old, {**cur, **s})
        else:
            state[rid] = (dict(w), {**w, **s})
        n += 1
    if not n:
        raise BuildError(f"{path.name}: no UPDATE statements")
    return {i: (o, {k: v for k, v in c.items()}) for i, (o, c) in state.items()}


def sql_ops(repair: Path, rollback: Path) -> list[tuple]:
    fwd, back = sql_net(repair), sql_net(rollback)
    if set(fwd) != set(back):
        raise BuildError(f"{repair.name} and {rollback.name} touch different rows")
    for i, (old, new) in fwd.items():
        b_old, b_new = back[i]
        if {k: new[k] for k in b_old} != b_old or {k: old.get(k, new[k]) for k in b_new} != b_new:
            raise BuildError(f"{rollback.name} does not undo {repair.name} for {i}")
    return [("update", i, old, {k: new[k] for k in new}) for i, (old, new) in fwd.items() if
            any(old.get(k) != new[k] for k in new)]


# -- undo in memory ----------------------------------------------------------------------------------------------------

def undo(state: dict[str, dict], ops: list[tuple], label: str) -> None:
    for op in reversed(ops):
        if op[0] == "insert":
            r = op[1]
            cur = state.get(r["id"])
            if cur is None:
                raise BuildError(f"{label}: inserted row {r['id']} is not in the table")
            if (cur["content_sha256"], cur["body_md5"], cur["section_number"]) != (r["content_sha256"], r["body_md5"], r["section_number"]):
                raise BuildError(f"{label}: inserted row {r['id']} differs from its record")
            if (cur["paragraph_path"], cur["path_ordinal"], cur["ordinal"]) != (r["paragraph_path"], r["path_ordinal"], r["ordinal"]):
                raise BuildError(f"{label}: inserted row {r['id']} is not where it was inserted")
            del state[r["id"]]
        elif op[0] == "delete":
            r = op[1]
            if r["id"] in state:
                raise BuildError(f"{label}: deleted row {r['id']} is in the table")
            state[r["id"]] = dict(r)
        else:
            _, rid, old, new = op
            cur = state.get(rid)
            if cur is None:
                raise BuildError(f"{label}: updated row {rid} is not in the table")
            names = {"path": "paragraph_path", "po": "path_ordinal", "ord": "ordinal"}
            for k, v in new.items():
                if cur[names[k]] != v:
                    raise BuildError(f"{label}: row {rid} {names[k]} is {cur[names[k]]!r}, the repair set {v!r}")
            for k, v in old.items():
                cur[names[k]] = v


def triple(r: dict) -> list:
    return [r["paragraph_path"], r["path_ordinal"], r["ordinal"]]


# -- the seven repairs -------------------------------------------------------------------------------------------------

def repairs(runs: Path) -> list[dict]:
    art = ROOT / "artifacts"
    t21 = runs / "title21_repair_20261023"
    rp = runs / "programme_20261026" / "db_repair"
    pp = runs / "paragraph_path_repair_20261028"
    df = runs / "definition_path_repair_20261030"
    st = runs / "source_text_repair_20261030"
    out = [
        {"name": "restore_dropped_text_20261021", "reference_applied": "2026-10-21",
         "description": "Restore eCFR text the original ingestion dropped (EXTRACT block quotes, FP flush paragraphs) in "
                        "the cosmetics parts, plan bb95cdff: 16 rows inserted, the section's later rows renumbered "
                        "(scripts/repair_dropped_source_text.py, regulus.ingestion.dropped_text).",
         "files": [art / "retrieval_repair_step1_20261021" / "apply_manifest.json"],
         "ops": lambda: manifest_ops(art / "retrieval_repair_step1_20261021" / "apply_manifest.json")},
        {"name": "restore_dropped_text_20261022", "reference_applied": "2026-10-22",
         "description": "Restore dropped eCFR text, step 2 (tables and Part 740 warnings included), plan f840b9bb: 128 "
                        "rows inserted, later rows renumbered.",
         "files": [art / "retrieval_repair_step2_20261022" / "apply_manifest.json"],
         "ops": lambda: manifest_ops(art / "retrieval_repair_step2_20261022" / "apply_manifest.json")},
        {"name": "restore_title21_text_20261023", "reference_applied": "2026-10-23",
         "description": "Restore dropped eCFR text across Title 21, plan 47d18a53, as it stands after the same day's "
                        "partial rollbacks (Parts 606/1107/1114 sections and Part 201 taken back) and the Part 201 "
                        "re-apply, plan 25ef904c: 525 rows inserted net, later rows renumbered.",
         "files": [t21 / "apply_manifest.json", t21 / "partial_rollback_manifest.json",
                   t21 / "partial_rollback2_part201_manifest.json", t21 / "apply_manifest_part201_reapply.json"],
         "ops": lambda: (manifest_ops(t21 / "apply_manifest.json")
                         + manifest_ops(t21 / "partial_rollback_manifest.json", reverse=True)
                         + manifest_ops(t21 / "partial_rollback2_part201_manifest.json", reverse=True)
                         + manifest_ops(t21 / "apply_manifest_part201_reapply.json"))},
        {"name": "reserved_parent_paths_20261026", "reference_applied": "2026-10-26",
         "description": "Paragraphs stored under a reserved-range row's path ((a)(g) for (g) in 82.3; (b)(e), (a)(c), "
                        "(f)(j) in other sections) given their own path: 31 rows.",
         "files": [rp / "82.3_paths_fix.sql", rp / "82.3_paths_rollback.sql", rp / "reserved_parent_fix.sql",
                   rp / "reserved_parent_rollback.sql"],
         "ops": lambda: (sql_ops(rp / "82.3_paths_fix.sql", rp / "82.3_paths_rollback.sql")
                         + sql_ops(rp / "reserved_parent_fix.sql", rp / "reserved_parent_rollback.sql"))},
        {"name": "paragraph_path_corrections_20261028", "reference_applied": "2026-10-28",
         "description": "Paragraph paths corrected against the eCFR XML paragraph ids, text-verified (repair v3): 988 rows.",
         "files": [pp / "repair_v3.sql", pp / "rollback_v3.sql", pp / "verified_corrections_v3.tsv"],
         "ops": lambda: sql_ops(pp / "repair_v3.sql", pp / "rollback_v3.sql")},
        {"name": "definition_paths_20261030", "reference_applied": "2026-10-30",
         "description": "DEF1: definitions sections' paragraph numbering restarted at each definition "
                        "(regulus.ingestion.definition_paths, scripts/repair_definition_paths.py): 1,162 rows in 47 sections.",
         "files": [df / "repair.sql", df / "rollback.sql", df / "corrections.tsv"],
         "ops": lambda: sql_ops(df / "repair.sql", df / "rollback.sql")},
        {"name": "restore_dropped_text_20261030", "reference_applied": "2026-10-30",
         "description": "Restore dropped eCFR text in the remaining Title 21 parts, plan 52dfc74b (tables in captioned "
                        "parts): 701 rows inserted, later rows renumbered.",
         "files": [st / "apply_manifest.json", st / "plan.json"],
         "ops": lambda: manifest_ops(st / "apply_manifest.json")},
    ]
    return out


ECFR_URL = "https://www.ecfr.gov/api/versioner/v1/full/2026-09-03/title-21.xml?chapter=I&part={part}"


def provenance_of(rep: dict, touched_parts: set[str], part_sha: dict[str, str]) -> dict:
    """The repair's own record files (with their SHA-256), and the eCFR part payloads it rests on: from the manifest's
    provenance for a text restore, else (a path repair, recorded as SQL) the payload every stored row of the part
    records in source_payload_sha256 - the eCFR XML the paths were checked against (scripts/verify_migration_sources.py)."""
    prov: dict = {"source_files": [{"file": str(p).replace(str(Path.home()), "~"), "sha256": sha(p)} for p in rep["files"]]}
    payloads = {}
    for p in rep["files"]:
        if p.name.endswith("manifest.json") or p.name.startswith("apply_manifest"):
            for part, v in json.loads(p.read_text(encoding="utf-8")).get("provenance", {}).items():
                payloads[part] = v
    if not payloads:
        payloads = {part: {"url": ECFR_URL.format(part=part), "sha256": part_sha[part], "from": "rows' source_payload_sha256"}
                    for part in touched_parts}
    prov["ecfr_payloads"] = dict(sorted(payloads.items(), key=lambda kv: (len(kv[0]), kv[0])))
    return prov


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path.home() / "regulus_eval_runs")
    ap.add_argument("--out", type=Path, default=MIGRATIONS_DIR)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    db = Database.from_env()
    db.connect()
    live = {r["id"]: r for r in db.query(KEY_SQL)}
    fp_sql = db.query(sql.SQL(FINGERPRINT_SQL).format(table=sql.Identifier("public", "cfr_records")).as_string(db.connect()))[0]
    fp_live = fingerprint_rows(live.values())
    if fp_live != (fp_sql["rows"], fp_sql["sha256"]):
        raise BuildError(f"fingerprint in Python {fp_live} differs from SQL {fp_sql}")
    reps = repairs(a.runs)
    for rep in reps:
        rep["ops"] = rep["ops"]()
    # undo newest first, keeping each state
    states = [None] * (len(reps) + 1)
    state = {i: dict(r) for i, r in live.items()}
    states[len(reps)] = {i: dict(r) for i, r in state.items()}
    for k in range(len(reps), 0, -1):
        undo(state, reps[k - 1]["ops"], reps[k - 1]["name"])
        states[k - 1] = {i: dict(r) for i, r in state.items()}
    inserted_ids = set(states[-1]) - set(states[0])
    cols = ", ".join(INSERT_COLUMNS)
    full = {r["r"]["id"]: r["r"] for r in db.query(FULL_SQL.format(cols=cols), {"ids": sorted(inserted_ids)})}
    part_sha = {}
    for r in db.query("SELECT part_number, array_agg(DISTINCT source_payload_sha256) AS s FROM cfr_records GROUP BY 1"):
        if len(r["s"]) != 1:
            raise BuildError(f"part {r['part_number']}: rows record {len(r['s'])} different source payloads")
        part_sha[r["part_number"]] = r["s"][0]
    db.close()

    built = []
    for k, rep in enumerate(reps, start=1):
        before, after = states[k - 1], states[k]
        gone = set(before) - set(after)
        if gone:
            raise BuildError(f"{rep['name']}: deletes {len(gone)} baseline rows - not a supported migration")
        inserts = []
        for rid in sorted(set(after) - set(before)):
            row = dict(full[rid])
            if (row["content_sha256"], body_md5(row["body_text"])) != (after[rid]["content_sha256"], after[rid]["body_md5"]):
                raise BuildError(f"{rep['name']}: stored text of {rid} differs from the repair record")
            row.update(dict(zip(("paragraph_path", "path_ordinal", "ordinal"), triple(after[rid]))))
            inserts.append(row)
        updates = [{"id": rid, "section": after[rid]["section_number"], "old": triple(before[rid]), "new": triple(after[rid])}
                   for rid in sorted(set(before) & set(after)) if triple(before[rid]) != triple(after[rid])]
        header = {"version": k, "name": rep["name"], "reference_applied": rep["reference_applied"],
                  "description": rep["description"],
                  "provenance": provenance_of(rep, {r["section_number"].split(".")[0] for r in inserts}
                                              | {u["section"].split(".")[0] for u in updates}, part_sha),
                  "fingerprint_before": dict(zip(("rows", "sha256"), fingerprint_rows(before.values()))),
                  "fingerprint_after": dict(zip(("rows", "sha256"), fingerprint_rows(after.values())))}
        built.append((header, inserts, updates))
        print(f"{k:04d}_{rep['name']}: insert {len(inserts)}, update {len(updates)}; "
              f"before {header['fingerprint_before']['rows']} {header['fingerprint_before']['sha256'][:16]} -> "
              f"after {header['fingerprint_after']['rows']} {header['fingerprint_after']['sha256'][:16]}")

    if a.check:
        existing = load_migrations(a.out)
        bad = [h["name"] for (h, ins, upd), m in zip(built, existing)
               if (list(m.insert), list(m.update), m.fingerprint_after) != (ins, upd, (h["fingerprint_after"]["rows"], h["fingerprint_after"]["sha256"]))]
        print("check:", "OK" if not bad and len(existing) == len(built) else f"DIFFERENT {bad}")
        return 1 if bad or len(existing) != len(built) else 0
    a.out.mkdir(parents=True, exist_ok=True)
    for header, inserts, updates in built:
        path = a.out / f"{header['version']:04d}_{header['name']}.json"
        print(path.name, write_file(path, header, inserts, updates)[:16])
    load_migrations(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
