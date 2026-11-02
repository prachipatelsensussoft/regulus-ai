"""Fetch official U.S. Code sections (govinfo, most recent edition) as plain text for scripts/build_statute_store.py.

    python scripts/fetch_usc.py [--out data/raw/statute_fdca] "FDCA 910=387j" "FDCA 531=360hh" ...

One file per section (<out>/usc_title21/21usc<sec>.txt) starting with its SOURCE URL and its FD&C Act label, and one line
per request in <out>/usc_fetch_log.jsonl (URL, retrieval time, sha256 of the HTML). A section already fetched is skipped
unless --refetch. Moved into the package on 2 Nov 2026 from the 19 Oct verification tools, unchanged in its output; it
identifies itself honestly and writes UTF-8 explicitly (macOS and Windows).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = "https://www.govinfo.gov/link/uscode/21/{sec}?type=usc&year=mostrecent&link-type=html"
AGENT = "Regulus-statute-fetch/1.0 (regulatory research; govinfo U.S. Code)"


def to_text(raw: bytes) -> str:
    t = raw.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style).*?</\1>", "", t)
    t = re.sub(r"(?i)<br\s*/?>|</p>|</h\d>|</div>", "\n", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = "\n".join(" ".join(line.split()) for line in t.splitlines())
    return re.sub(r"\n{2,}", "\n", t).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/raw/statute_fdca"))
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("specs", nargs="+", help='"FDCA <section>=<21 U.S.C. section>"')
    a = ap.parse_args()
    outdir = Path(a.out) / "usc_title21"
    outdir.mkdir(parents=True, exist_ok=True)
    failed = 0
    with open(Path(a.out) / "usc_fetch_log.jsonl", "a", encoding="utf-8") as log:
        for spec in a.specs:
            label, sec = (x.strip() for x in spec.split("="))
            name = f"21usc{sec}.txt"
            if (outdir / name).exists() and not a.refetch:
                print("have", spec)
                continue
            req = urllib.request.Request(LINK.format(sec=sec), headers={"User-Agent": AGENT})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    raw, final = r.read(), r.geturl()
            except Exception as e:  # noqa: BLE001 - logged and reported
                log.write(json.dumps({"label": label, "usc": sec, "error": repr(e)}) + "\n")
                print("FAIL", spec, e)
                failed += 1
                continue
            text = to_text(raw)
            if f"§{sec}." not in text.replace(" ", "").replace("\u2013", "-"):   # "§343–1." is printed with an en dash
                log.write(json.dumps({"label": label, "usc": sec, "url": final, "error": "section heading not in page"}) + "\n")
                print("FAIL", spec, "section heading not in the page", final)
                failed += 1
                continue
            (outdir / name).write_text(f"SOURCE: {final}\nLABEL: {label} = 21 U.S.C. {sec}\n\n" + text, encoding="utf-8")
            log.write(json.dumps({"label": label, "usc": sec, "url": final,
                                  "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                  "html_sha256": hashlib.sha256(raw).hexdigest(), "file": name}) + "\n")
            print("ok", spec, final.rsplit("/", 1)[-1], len(text))
            time.sleep(a.pause)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
