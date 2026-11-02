"""Build data/reference/part_fdca_authority.json (switch AU1, regulus.retrieval.part_authority) from the eCFR XML.

    python scripts/build_part_authority.py --xml DIR [--store data/processed/statute_fdca/rows.jsonl] [--out ...] [--check]

DIR holds one eCFR versioner file per part (part-<N>.xml). Each part's <AUTH> line lists the U.S. Code sections it is
issued under ("21 U.S.C. 321, 331, 360hh-360ss, 371; 42 U.S.C. 262"); the 21 U.S.C. sections, ranges expanded over the
sections the statute store holds, are mapped to their FD&C Act numbers (387j -> 910). Only sections the store holds are
listed, so the map is rebuilt whenever the store grows (2 Nov 2026: 44 -> 88 sections). --check compares with --out.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ("eCFR versioner XML 2026-09-03 <AUTH> lines (21 U.S.C. sections), mapped to FD&C Act sections held in "
          "data/processed/statute_fdca")

_AUTH = re.compile(r"<AUTH>(.*?)</AUTH>", re.S)
_TITLE = re.compile(r"(\d+)\s+U\.\s*S\.\s*C\.?")
# a section's own "-1" suffix has one or two digits (343-1, 379aa-1); a range's end has three ("321-397", "360e-360j")
_SEC = re.compile(r"(\d+[a-z]*(?:-\d{1,2}(?![\da-z]))?)(?:\s*[-–]\s*(\d{3,}[a-z]*(?:-\d{1,2}(?![\da-z]))?))?")
_NOTE = re.compile(r"\bnote\b", re.I)
WIDE_RANGE = 20


def usc_key(sec: str) -> tuple:
    """Code order: 360 < 360a < 360b ... < 360z < 360aa ... < 360hh < 360ss < 360bbb; 343 < 343-1 < 343-2 < 343a."""
    m = re.fullmatch(r"(\d+)([a-z]*)(?:-(\d+))?", sec)
    if not m:
        return (10**9, 0, "", 0)
    return (int(m.group(1)), len(m.group(2)), m.group(2), int(m.group(3) or 0))


def fdca_key(sec: str) -> tuple:
    """FD&C Act order: 403 < 403A < 403B < 404; 515 < 515C."""
    m = re.fullmatch(r"(\d+)([A-Za-z]*)", sec)
    return (int(m.group(1)), m.group(2)) if m else (10**9, sec)


def title21_sections(auth: str, held: list[str]) -> list[str]:
    """The 21 U.S.C. sections an <AUTH> text lists, ranges expanded over the held sections, in order of appearance."""
    text = re.sub(r"<[^>]+>", " ", auth)
    out: list[str] = []
    marks = list(_TITLE.finditer(text))
    for i, m in enumerate(marks):
        if m.group(1) != "21":
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        chunk = text[m.end():end].split(";")[0]
        for item in chunk.split(","):
            item = re.sub(r"^and\s+", "", _NOTE.split(item)[0].strip())
            s = _SEC.match(item)
            if not s:
                continue
            lo, hi = s.group(1), s.group(2)
            if hi:
                # a range runs over the Code's own sections, not the "-1" sections inserted later (360e-3 is not in
                # "360e-360j"); one spanning the whole Act ("321-397", parts 5-25) names no section: its first only
                inside = [h for h in held if usc_key(lo) <= usc_key(h) <= usc_key(hi) and "-" not in h]
                out += inside if len(inside) <= WIDE_RANGE else [lo]
            else:
                out.append(lo)
    seen: set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, type=Path)
    ap.add_argument("--store", type=Path, default=ROOT / "data/processed/statute_fdca/rows.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/reference/part_fdca_authority.json")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    usc_to_fdca: dict[str, str] = {}
    for line in a.store.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            usc_to_fdca.setdefault(r["usc_section"], r["fdca_section"])
    held = sorted(usc_to_fdca, key=usc_key)
    parts: dict[str, list[str]] = {}
    for path in sorted(a.xml.glob("part-*.xml"), key=lambda p: usc_key(p.stem.split("-", 1)[1])):
        m = _AUTH.search(path.read_text(encoding="utf-8"))
        if not m:
            continue
        fdca = sorted({usc_to_fdca[s] for s in title21_sections(m.group(1), held) if s in usc_to_fdca}, key=fdca_key)
        if fdca:
            parts[path.stem.split("-", 1)[1]] = fdca
    result = {"source": SOURCE, "parts": parts}
    if a.check:
        old = json.loads(a.out.read_text(encoding="utf-8"))
        same = old == result
        changed = sorted(p for p in set(old["parts"]) | set(parts) if old["parts"].get(p) != parts.get(p))
        print("identical" if same else f"{len(changed)} parts differ: {changed[:20]}")
        return 0 if same else 1
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"{len(parts)} parts -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
