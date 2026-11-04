"""Build a heading migration (data/migrations/headings) from the official eCFR XML: every subpart title the database
holds differently from the eCFR, and the subpart heading copied onto each of that subpart's rows. Read-only on the
database; writes one new migration file (never overwrites).

    python scripts/build_heading_migrations.py --xml-dir ~/regulus_eval_runs/title21_source_xml_20261022 \\
        --name subpart_titles_20261104 --reference-applied 2026-11-04 [--dry-run]

The eCFR heading is "Subpart G<em dash>Supply-Chain Program", footnote markers (<SU>1</SU><FTREF/>) inside it. The original
ingestion split it at the last dash and kept footnote numbers, storing "Chain Program" and "Sequestrants 1". Here the
title is the text after the FIRST em dash, with <SU>/<FTREF> removed. A row's parent heading is changed only where it is
exactly "Subpart <designation> - <stored title>" (the form the ingestion wrote); any other form stops the build.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regulus.datamigrate.headings import HEADINGS_DIR, load_heading_migrations, write_file  # noqa: E402
from regulus.retrieval import Database  # noqa: E402

_SUBPART = re.compile(r'<DIV6 N="([^"]+)" TYPE="SUBPART"[^>]*>\s*<HEAD>(.*?)</HEAD>', re.S)


def official_titles(xml_dir: Path) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """{(part, designation): title} and {part: sha256 of its XML file}."""
    out, shas = {}, {}
    for f in sorted(xml_dir.glob("part-*.xml")):
        part = re.fullmatch(r"part-(\w+)\.xml", f.name).group(1)
        raw = f.read_bytes()
        shas[part] = hashlib.sha256(raw).hexdigest()
        for m in _SUBPART.finditer(raw.decode("utf-8", errors="replace")):
            head = re.sub(r"<SU>.*?</SU>|<FTREF\s*/>", "", m.group(2), flags=re.S)
            head = " ".join(html.unescape(re.sub(r"<[^>]+>", "", head)).split())
            if head.lower().startswith("subparts "):
                continue   # a reserved range ("Subparts C-D [Reserved]"): the dash joins two letters, not a title
            title = head.split("\u2014", 1)[1].strip() if "\u2014" in head else ""
            if title:
                out[(part, m.group(1))] = title
    return out, shas


def norm(t: str | None) -> str:
    return " ".join((t or "").split()).rstrip(". ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml-dir", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--reference-applied", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    official, shas = official_titles(a.xml_dir.expanduser())
    db = Database.from_env()
    db.connect()
    try:
        subparts = db.query("SELECT id::text AS id, part_number AS part, designation AS d, title FROM cfr_subparts")
        titles, heads, parts = [], [], set()
        for s in sorted(subparts, key=lambda r: (len(r["part"]), r["part"], r["d"])):
            new = official.get((s["part"], s["d"]))
            if new is None or norm(new) == norm(s["title"]):
                continue
            titles.append({"id": s["id"], "part": s["part"], "designation": s["d"], "old": s["title"], "new": new})
            parts.add(s["part"])
            want_old, want_new = f"Subpart {s['d']} - {s['title']}", f"Subpart {s['d']} - {new}"
            for r in db.query("SELECT id::text AS id, section_number AS s, parent_headings AS h FROM cfr_records "
                              "WHERE subpart_id = %(i)s::uuid ORDER BY section_number, ordinal", {"i": s["id"]}):
                h = list(r["h"] or [])
                hits = [i for i, x in enumerate(h) if x.lower().startswith("subpart ")]
                if len(hits) != 1 or h[hits[0]] != want_old:
                    raise SystemExit(f"{r['s']} {r['id']}: subpart heading {h} is not exactly {want_old!r}; nothing written")
                h2 = h.copy()
                h2[hits[0]] = want_new
                heads.append({"id": r["id"], "section": r["s"], "old": h, "new": h2})
    finally:
        db.close()

    for t in titles:
        print(f"  {t['part']} {t['designation']}: {t['old']!r} -> {t['new']!r}")
    print(f"{len(titles)} subpart titles, {len(heads)} rows' parent headings")
    if a.dry_run or not titles:
        return 0
    version = len(load_heading_migrations()) + 1
    HEADINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = HEADINGS_DIR / f"{version:04d}_{a.name}.json"
    header = {"version": version, "name": a.name, "reference_applied": a.reference_applied,
              "description": (f"Subpart titles corrected to the eCFR heading text (the original ingestion split "
                              f"'Subpart G—Supply-Chain Program' at the last dash and kept footnote numbers): "
                              f"{len(titles)} subparts in parts {', '.join(sorted(parts, key=lambda p: (len(p), p)))}, and "
                              f"the subpart heading on each of their {len(heads)} rows."),
              "provenance": {"ecfr_xml": {p: {"file": f"{a.xml_dir}/part-{p}.xml".replace(str(Path.home()), "~"),
                                              "sha256": shas[p]} for p in sorted(parts, key=lambda p: (len(p), p))},
                             "builder": "scripts/build_heading_migrations.py"}}
    digest = write_file(path, header, titles, heads)
    print(f"wrote {path.relative_to(ROOT)} (checksum {digest[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
