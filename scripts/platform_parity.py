"""Mac / Windows parity check: the same regulatory questions through the llm-model API, recorded and compared.

    python scripts/platform_parity.py run --out parity_mac_run1.json            (the API must be running)
    python scripts/platform_parity.py compare A.json B.json                     (exit 0 only if every answer matches)
    python scripts/platform_parity.py from-raw <dir of saved /answer responses> --out X.json

`run` asks each question once through POST /llm-model/v1/answer, with the first key of LLM_MODEL_API_KEYS from
configs/llm_model.env (or REGULUS_PARITY_API_KEY), and writes one JSON record: per question the state, the live
evaluation status, the answer text and its SHA-256, the citations (label, URL, cited in the answer), the model identity
and the release (code revision, switches) the server reported. The key is never written or printed.

`compare` reports, per question: same state, same evaluation status, identical answer (else a similarity ratio and the
first differing sentence), identical citations. Run it on two Mac runs (determinism), and on a Mac run against a
Windows run made on the same commit and migrated database, for parity. A difference is a finding to read, not
automatically a defect: greedy decoding on MLX (bfloat16) and on PyTorch/CUDA (float16) can part where two tokens are within rounding.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import platform
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = [
    ("q1_820_10", "What does 21 CFR 820.10 require?", None),
    ("q2_otc_tamper", "What are the requirements for tamper-evident packaging of OTC drug products?", None),
    ("q3_mica", "Can we use mica in an eye shadow?", "cosmetics"),
    ("q4_ind_7day", "Within how many days must an IND sponsor report an unexpected fatal or life-threatening suspected "
                    "adverse reaction to FDA?", None),
    ("q5_drug_product", "How does 21 CFR 210.3 define 'drug product'?", None),
    ("q6_part11", "What controls does 21 CFR 11.10 require for closed systems?", None),
    ("q7_removed", "What does 21 CFR 820.250 require?", None),
    ("q8_fake", "What does 21 CFR 999.99 require?", None),
    ("q9_ind_7day_repeat", "Within how many days must an IND sponsor report an unexpected fatal or life-threatening "
                           "suspected adverse reaction to FDA?", None),
]


def api_key() -> str:
    if os.environ.get("REGULUS_PARITY_API_KEY"):
        return os.environ["REGULUS_PARITY_API_KEY"]
    profile = ROOT / "configs" / "llm_model.env"
    for line in profile.read_text(encoding="utf-8").splitlines():
        if line.startswith("LLM_MODEL_API_KEYS="):
            return line.split("=", 1)[1].split(",")[0].split(":", 1)[1].strip()
    sys.exit(f"no LLM_MODEL_API_KEYS in {profile}; set REGULUS_PARITY_API_KEY")


def normalise(qid: str, question: str, focus: str | None, http: int, payload: dict, seconds: float | None) -> dict:
    answer = payload.get("answer") or ""
    return {"id": qid, "question": question, "focus": focus, "http": http, "state": payload.get("state"),
            "evaluation": (payload.get("evaluation") or {}).get("status"), "seconds": seconds,
            "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(), "answer": answer,
            "citations": [{"label": c.get("label"), "url": c.get("url"), "cited": c.get("citedInAnswer")}
                          for c in payload.get("citations") or []],
            "model": payload.get("model"), "release": payload.get("release"), "snapshot": payload.get("snapshotDate"),
            "error": payload.get("code")}


def record(results: list[dict], source: str) -> dict:
    return {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": source,
            "machine": {"os": platform.platform(), "python": platform.python_version(), "processor": platform.processor()},
            "questions": len(results), "results": results}


def cmd_run(args) -> int:
    key, results = api_key(), []
    for qid, question, focus in QUESTIONS:
        body = json.dumps({"question": question, **({"focus": focus} if focus else {})}).encode()
        req = urllib.request.Request(f"{args.url.rstrip('/')}/llm-model/v1/answer", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                http, payload = r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            http, payload = e.code, json.loads(e.read() or b"{}")
        res = normalise(qid, question, focus, http, payload, round(time.perf_counter() - t0, 1))
        results.append(res)
        print(f"{qid:20s} HTTP {http} {res['state']} eval={res['evaluation']} {res['seconds']}s", flush=True)
    Path(args.out).write_text(json.dumps(record(results, args.url), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def cmd_from_raw(args) -> int:
    results = []
    for qid, question, focus in QUESTIONS:
        f = Path(args.dir) / f"{qid}.json"
        if not f.is_file():
            sys.exit(f"missing {f}")
        payload = json.loads(f.read_text(encoding="utf-8"))
        results.append(normalise(qid, question, focus, payload.get("_http", 200), payload, payload.get("_seconds")))
    Path(args.out).write_text(json.dumps(record(results, f"raw responses in {Path(args.dir).name}"), ensure_ascii=False,
                                         indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def first_difference(a: str, b: str) -> str:
    sa, sb = re.split(r"(?<=[.;:])\s+", a), re.split(r"(?<=[.;:])\s+", b)
    for x, y in zip(sa, sb):
        if x != y:
            return f"A: {x[:160]!r}\n      B: {y[:160]!r}"
    return "one answer continues where the other ends"


def cmd_compare(args) -> int:
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.a, args.b))
    ra, rb = {r["id"]: r for r in a["results"]}, {r["id"]: r for r in b["results"]}
    print(f"A: {args.a}  ({a['machine']['os']})\nB: {args.b}  ({b['machine']['os']})")
    switches = []
    for side, rec in (("A", ra), ("B", rb)):
        any_r = next(iter(rec.values()))
        release = any_r.get("release") or {}
        rev = release.get("revision") or {}
        rev = rev if isinstance(rev, dict) else {"git_commit": str(rev)}
        switches.append(release.get("switches"))
        print(f"{side} model {((any_r.get('model') or {}).get('id'))} weights "
              f"{((any_r.get('model') or {}).get('model_revision') or '')[:12]}  code {(rev.get('git_commit') or '')[:12]}"
              f"{' +uncommitted ' + (rev.get('tracked_diff_sha256') or '')[:12] if rev.get('git_dirty') else ''}"
              f"  db snapshot {any_r.get('snapshot')}")
    # a comparison is only fair on the same code, switches and database snapshot
    print(f"switches {'identical' if switches[0] == switches[1] else 'DIFFER: ' + str(sorted(set(switches[0] or []) ^ set(switches[1] or [])))}")
    identical = 0
    for qid, _, _ in QUESTIONS:
        x, y = ra.get(qid), rb.get(qid)
        if not x or not y:
            print(f"\n{qid}: missing on one side")
            continue
        same_answer = x["answer_sha256"] == y["answer_sha256"]
        same_cites = x["citations"] == y["citations"]
        same = same_answer and same_cites and x["state"] == y["state"] and x["evaluation"] == y["evaluation"]
        identical += same
        print(f"\n{qid}: {'IDENTICAL' if same else 'DIFFERENT'}")
        print(f"  state      {x['state']:20s} {y['state']}")
        print(f"  evaluation {str(x['evaluation']):20s} {y['evaluation']}")
        if not same_answer:
            ratio = difflib.SequenceMatcher(None, x["answer"], y["answer"]).ratio()
            print(f"  answer     differs (similarity {ratio:.3f})\n      {first_difference(x['answer'], y['answer'])}")
        if not same_cites:
            ca = [(c["label"], c["cited"]) for c in x["citations"]]
            cb = [(c["label"], c["cited"]) for c in y["citations"]]
            print(f"  citations  only A: {sorted(set(ca) - set(cb))}  only B: {sorted(set(cb) - set(ca))}")
    print(f"\n{identical}/{len(QUESTIONS)} identical")
    return 0 if identical == len(QUESTIONS) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8100")
    r.add_argument("--out", required=True)
    r.add_argument("--timeout", type=float, default=1200)
    f = sub.add_parser("from-raw")
    f.add_argument("dir")
    f.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = p.parse_args()
    return {"run": cmd_run, "from-raw": cmd_from_raw, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
