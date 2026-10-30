"""Download the eCFR part payloads the cfr_records migrations rest on and check their SHA-256. Optional; read-only.

    python scripts/verify_migration_sources.py [--parts 73,74] [--keep DIR] [--local DIR [--local DIR2 ...]]

Applying the migrations needs none of these files: each migration carries every value it writes. This is for anyone
who wants to check the provenance - that the text restored and the paths corrected come from the eCFR as published for
2026-09-03, the snapshot the baseline was ingested from. Each migration's provenance.ecfr_payloads lists, per part, the
date-pinned eCFR versioner URL and the SHA-256 of the payload used. --local DIR checks files already on disk
(part-<N>.xml; the first folder holding a part is used) instead of downloading - the copies archived with the repairs:
artifacts/retrieval_repair_step{1,2}_*/source_xml/ in this repository for the cosmetics parts. --keep DIR saves the
downloads. Exit 0 when every payload matches. (30 Oct 2026: the eCFR API answered this script's requests with HTTP 406
from the reference machine; the archived copies all matched.)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regulus.datamigrate import load_migrations  # noqa: E402


def fetch(url: str, attempts: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "regulus-migration-source-check/1"})
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if i == attempts - 1:
                raise
            print(f"  retry after {exc}", flush=True)
            time.sleep(5 * (i + 1))
    raise AssertionError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="", help="comma-separated parts (default: every part any migration names)")
    ap.add_argument("--keep", type=Path, help="save downloaded payloads here as part-<N>.xml")
    ap.add_argument("--local", type=Path, action="append", help="check part-<N>.xml files in this folder instead of downloading")
    a = ap.parse_args()
    wanted = {p.strip() for p in a.parts.split(",") if p.strip()}
    sources: dict[str, dict] = {}
    for m in load_migrations():
        for part, v in m.provenance.get("ecfr_payloads", {}).items():
            prev = sources.setdefault(part, {"url": v["url"], "sha256": v["sha256"], "migrations": []})
            if prev["sha256"] != v["sha256"]:
                print(f"part {part}: migrations name two different payloads ({prev['sha256'][:16]}, {v['sha256'][:16]})")
                return 1
            prev["migrations"].append(m.version)
    if a.keep:
        a.keep.mkdir(parents=True, exist_ok=True)
    bad = 0
    parts = sorted((p for p in sources if not wanted or p in wanted), key=lambda p: (len(p), p))
    for n, part in enumerate(parts, start=1):
        s = sources[part]
        try:
            if a.local:
                found = [d / f"part-{part}.xml" for d in a.local if (d / f"part-{part}.xml").exists()]
                data = found[0].read_bytes() if found else None
            else:
                data = fetch(s["url"])
                time.sleep(1)   # be polite to the eCFR API
        except Exception as exc:   # noqa: BLE001 - report and go on to the next part
            print(f"[{n}/{len(parts)}] part {part}: FAILED to read ({type(exc).__name__}: {exc})")
            bad += 1
            continue
        if data is None:
            print(f"[{n}/{len(parts)}] part {part}: no local file")
            bad += 1
            continue
        got = hashlib.sha256(data).hexdigest()
        ok = got == s["sha256"]
        bad += not ok
        print(f"[{n}/{len(parts)}] part {part}: {'OK' if ok else 'MISMATCH'} {got[:16]} "
              f"(expected {s['sha256'][:16]}, {len(data):,} bytes; migrations {s['migrations']})", flush=True)
        if a.keep and not a.local:
            (a.keep / f"part-{part}.xml").write_bytes(data)
    print(f"{len(parts) - bad} of {len(parts)} payloads match")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
