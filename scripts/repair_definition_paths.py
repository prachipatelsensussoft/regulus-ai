"""Plan the DEF1 repair of definitions sections' paragraph paths (regulus.ingestion.definition_paths). Read-only.

    python scripts/repair_definition_paths.py --sources DIR --out OUT_DIR [--accept-unverified]

DIR holds the eCFR part payloads (part-<N>.xml). For every section with definition leads (<P><I>term</I> ...), the
stored rows are read in ordinal order and the corrected paths planned. A part is used only when its payload's
SHA-256 equals the source_payload_sha256 its stored rows record (--accept-unverified plans the others too, flagged in
the summary, for review). Writes OUT_DIR/corrections.tsv (section, id, ordinal, old path/po, new path/po, text),
OUT_DIR/repair.sql, OUT_DIR/repair_dryrun.sql (the same, ending in ROLLBACK), OUT_DIR/rollback.sql and
OUT_DIR/summary.json (with the SHA-256 of repair.sql). Nothing is written to the database: a person runs the dry run,
then applies repair.sql (psql -v ON_ERROR_STOP=1 -f repair.sql) and keeps rollback.sql.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regulus.ingestion.definition_paths import Row, definition_leads, plan_section, repair_sql  # noqa: E402
from regulus.retrieval import Database  # noqa: E402

_ROWS = ("SELECT id::text AS id, ordinal, paragraph_path, path_ordinal, body_text, source_payload_sha256 "
         "FROM cfr_records WHERE removed_on IS NULL AND section_number = %(s)s ORDER BY ordinal")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--accept-unverified", action="store_true")
    a = ap.parse_args()
    db = Database.from_env()
    db.connect()
    plans, summary = [], {"parts": {}, "sections": {}}
    for xml_path in sorted(a.sources.glob("part-*.xml")):
        part = xml_path.stem.split("-", 1)[1]
        payload = xml_path.read_bytes()
        leads = definition_leads(payload)
        if not leads:
            continue
        got = hashlib.sha256(payload).hexdigest()
        for section, texts in leads.items():
            stored = db.query(_ROWS, {"s": section})
            if not stored:
                continue
            verified = {r["source_payload_sha256"] for r in stored} == {got}
            summary["parts"].setdefault(part, {"payload_sha256": got, "verified": verified})
            if not verified and not a.accept_unverified:
                summary["sections"][section] = {"status": "skipped", "reason": "payload SHA-256 differs from the stored rows'"}
                continue
            plan = plan_section(section, texts, [Row(r["id"], r["ordinal"], r["paragraph_path"] or "", r["path_ordinal"],
                                                     r["body_text"] or "") for r in stored])
            summary["sections"][section] = {"status": plan.status, "reason": plan.reason, "updates": len(plan.updates),
                                            "leads": len(texts), "verified": verified}
            if plan.updates:
                text = {r["id"]: (r["ordinal"], " ".join((r["body_text"] or "").split())[:80]) for r in stored}
                plan.rows_text = text   # for corrections.tsv
                plans.append(plan)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "corrections.tsv", "w", encoding="utf-8") as f:
        f.write("section\tid\tordinal\told_path\told_po\tnew_path\tnew_po\ttext\n")
        for p in plans:
            for rid, old, old_po, new, new_po in p.updates:
                ordinal, text = p.rows_text[rid]
                f.write(f"{p.section_number}\t{rid}\t{ordinal}\t{old}\t{old_po}\t{new}\t{new_po}\t{text}\n")
    fwd, back = repair_sql(plans)
    (a.out / "repair.sql").write_text(fwd, encoding="utf-8")
    (a.out / "rollback.sql").write_text(back, encoding="utf-8")
    # the same statements ending in ROLLBACK: run it first - it must finish without an error and change nothing
    dry = fwd.rstrip()
    (a.out / "repair_dryrun.sql").write_text(dry[: dry.rfind("COMMIT;")] + "ROLLBACK;\n", encoding="utf-8")
    summary["repair_sql_sha256"] = hashlib.sha256(fwd.encode("utf-8")).hexdigest()
    summary["totals"] = {"sections_planned": len(plans), "rows": sum(len(p.updates) for p in plans),
                         "misaligned": sorted(s for s, v in summary["sections"].items() if v["status"] == "misaligned"),
                         "deep_for_review": sorted(s for s, v in summary["sections"].items() if v["status"] == "deep"),
                         "skipped_unverified": sorted(s for s, v in summary["sections"].items() if v["status"] == "skipped")}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["totals"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
