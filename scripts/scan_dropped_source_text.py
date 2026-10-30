"""Read-only scan of the fetched eCFR part payloads: the dropped EXTRACT / FP / table text a repair would restore, and
the targets file for scripts/repair_dropped_source_text.py.

    python scripts/scan_dropped_source_text.py --sources DIR --out OUT_DIR [--parts 101,1114,...]

DIR holds part-<N>.xml payloads and the fetch logs (*.json, entries {"part", "verified"}) written with them. A part is
scanned only when it is logged as verified, and planned only when its payload's SHA-256 equals the one its stored rows
record (regulus.ingestion.dropped_text.plan_part). Excluded and listed, not restored: blank form templates (a run of
five or more underscores) and non-table blocks over MAX_CHARS (the prompt cuts a result at 4,000 characters; a large
table is planned in parts, regulus.ingestion.dropped_text.TABLE_PART_CHARS). An anchor holding an excluded block is
left out whole, since a plan restores an anchor's blocks together.

Writes OUT_DIR/targets.json and OUT_DIR/scan.json (per part, per section, misaligned sections with the reason, and the
excluded blocks). Successor of the 22-23 Oct scratch scan (tools_20261022/scan_t21.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regulus.ingestion import dropped_text as DT  # noqa: E402
from regulus.retrieval import Database  # noqa: E402

MAX_CHARS = 4000
_FORM = re.compile(r"_{5,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", default="", help="comma-separated parts (default: every verified part)")
    a = ap.parse_args()
    logs = {}
    for f in a.sources.glob("*.json"):
        try:
            entries = json.loads(f.read_text())
        except ValueError:
            continue
        for r in entries if isinstance(entries, list) else []:
            if isinstance(r, dict) and r.get("verified"):
                logs[str(r["part"])] = r
    wanted = {p.strip() for p in a.parts.split(",") if p.strip()}
    db = Database.from_env()
    db.connect()
    targets, excluded, misaligned, per_part, refused = [], [], {}, Counter(), {}
    rows_by_tag = Counter()
    for part in sorted(logs, key=lambda p: (len(p), p)):
        if wanted and part not in wanted:
            continue
        path = a.sources / f"part-{part}.xml"
        if not path.exists():
            continue
        try:
            plans = DT.plan_part(db.query, part, path.read_bytes(), None)
        except DT.RepairError as exc:
            refused[part] = str(exc)[:200]
            continue
        for pl in plans:
            if pl.status == "misaligned":
                misaligned[pl.section_number] = (pl.reason or "")[:200]
            for r in pl.inserts:
                item = {"section": pl.section_number, "paragraph_path": r.paragraph_path}
                if _FORM.search(r.body_text) or (r.source_tag != "TABLE" and len(r.body_text) > MAX_CHARS):
                    excluded.append({**item, "tag": r.source_tag, "chars": len(r.body_text), "text": r.body_text[:120],
                                     "why": "form template" if _FORM.search(r.body_text) else f"over {MAX_CHARS} chars"})
                else:
                    rows_by_tag[r.source_tag] += 1
                    if item not in targets:
                        targets.append(item)
                        per_part[part] += 1
    db.close()
    held = {(e["section"], e["paragraph_path"]) for e in excluded}
    targets = [t for t in targets if (t["section"], t["paragraph_path"]) not in held]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "targets.json").write_text(json.dumps(targets, indent=1))
    report = {"parts_scanned": sum(1 for p in logs if not wanted or p in wanted), "parts_refused": refused,
              "anchors": len(targets), "rows_by_tag": dict(rows_by_tag), "per_part": dict(per_part),
              "excluded": excluded, "misaligned_sections": misaligned}
    (a.out / "scan.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("parts_scanned", "anchors", "rows_by_tag")}
                     | {"parts_refused": len(refused), "excluded": len(excluded), "misaligned": len(misaligned)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
