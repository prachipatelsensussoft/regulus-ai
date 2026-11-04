"""Per-question measurement record: what one benchmark answer was produced from, and how to verify it.

A measurement record is written next to each answer of an acceptance run (`measurement.jsonl`, one JSON
object per line, beside `answers.jsonl`). It exists so that an answer can be rescored later from what the
run actually had - the exact evidence, the raw model output, the validation reports and the repair - rather
than from evidence rebuilt out of today's database. It is an observer: it is built from a finished
`RagAnswer` after the answer is final, reads it only through a JSON copy, and never feeds anything back.
`answers.jsonl` keeps its shape; nothing here changes what was delivered or how it is judged.

Schema `regulus_measurement_record_v1` (see docs/measurement_record.md for the field table)
------------------------------------------------------------------------------------------
    schema, hash_algorithm, canonicalization     how to read and verify the record
    status            captured | pipeline_error | capture_error | cancelled (the request ended before its answer
                      was delivered: a client disconnect)
    complete          True only when every REQUIRED rescoring input is captured (see REQUIRED)
    gaps              [{field, state, reason}] for every field that is not captured and not merely
                      not applicable - required or not
    question_id, ordinal, captured_utc
    run               run identity supplied by the runner (model, decoding, code revision, rubric)
    question          {text, sha256, matches_pipeline_question}
    text              raw_model_output, validation_input, repair_output, delivered_answer: each a TEXT
                      block {state, text, sha256, chars, reason}
    transformations   what the pipeline itself recorded doing to the raw output before validation
    evidence          {provenance, reason, fingerprint, item_text_sha256, snapshot, retrieval_ranking,
                       selection, coverage, coverage_action}
    validation        {state, initial, final, repair, fallback_type, fallback_reason, answerability,
                       generation_blocked, block_reason, citation_support}
    citations         the delivered answer's citations
    evaluator_inputs  {model_called, scope_status, package_present, answer_plan, withheld}: the rest of
                      what the evaluator reads, for offline evaluation (absent from records built before it)
    inferred         text comparisons, labelled as inferred: NOT a pipeline record
    evaluation        {state: not_run, reason} or {state: captured, assessment, review}
    origin            (optional; present only when the caller gives one) where the answer was produced: for a UI
                      turn {source: "ui", endpoint, conversation_id, turn_id, request, delivery}
    record_sha256     see below

Field states
------------
    captured        the value is the pipeline's own, exactly as it was
    not_applicable  the stage did not run for this answer (no model call, so no raw output)
    not_exposed     the stage ran but the pipeline does not expose the value; absence says nothing about it
    not_recorded    (historical rows only) the artifact predates capture of this field
    capture_failed  the value existed but could not be captured

Hashes
------
Every hash is SHA-256, hex. A text hash is over the text's UTF-8 bytes. `record_sha256` is over the
CANONICAL JSON of the whole record except the `record_sha256` key itself:

    json.dumps(record_without_record_sha256, sort_keys=True, ensure_ascii=False,
               separators=(",", ":"), allow_nan=False).encode("utf-8")

Every other field is covered, `captured_utc` included: it is stored once, never regenerated on verification.
The evidence is covered twice over: per item by `item_text_sha256`, and as a whole by the snapshot's own
fingerprint (regulus.traceability.evidence), which `verify_record` recomputes from the item texts. This is
an integrity checksum: it detects a change to the record; it says nothing about who wrote it.

Historical runs
---------------
Runs written before this record existed have no `measurement.jsonl`. `load_run` returns them as legacy
views - `complete` False, and every field the old row never held marked `not_recorded`. Nothing is
reconstructed and an old row is never presented as a measurement record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from regulus.traceability.evidence import EXACT, MISSING, load as load_evidence

MEASUREMENT_SCHEMA = "regulus_measurement_record_v1"
HASH_ALGORITHM = "sha256"
CANONICALIZATION = 'json.dumps(sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False), utf-8'
MEASUREMENT_FILE = "measurement.jsonl"

CAPTURED = "captured"
NOT_APPLICABLE = "not_applicable"
NOT_EXPOSED = "not_exposed"
NOT_RECORDED = "not_recorded"
CAPTURE_FAILED = "capture_failed"

STATUS_CAPTURED = "captured"
STATUS_PIPELINE_ERROR = "pipeline_error"
STATUS_CAPTURE_ERROR = "capture_error"
STATUS_CANCELLED = "cancelled"

# The rescoring inputs a record must hold to be complete (when the stage applies to the answer).
REQUIRED = ("text.delivered_answer", "text.raw_model_output", "evidence", "validation")

NOT_RUN_BY_RUNNER = ("scripts/run_acceptance_test.py records answers only and makes no evaluator call; the "
                     "evaluator inputs are in this record (evidence.snapshot, validation, citations)")


# ------------------------------------------------------------------ hashing


def sha256_text(text: str | None) -> str | None:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str | None:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest() if value is not None else None


def record_hash(record: dict[str, Any]) -> str:
    return sha256_json({k: v for k, v in record.items() if k != "record_sha256"})


def _copy(value: Any) -> Any:
    """A JSON-native deep copy: the record never aliases pipeline objects, and what is hashed is what loads."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _seal(record: dict[str, Any]) -> dict[str, Any]:
    record["record_sha256"] = record_hash(record)
    return record


# ------------------------------------------------------------------ capture


def _text(state: str, text: str | None = None, reason: str | None = None) -> dict[str, Any]:
    if state != CAPTURED:
        text = None
    return {"state": state, "text": text, "sha256": sha256_text(text),
            "chars": len(text) if text is not None else None, "reason": reason}


def _gaps(record: dict[str, Any]) -> list[dict[str, Any]]:
    out = [{"field": f"text.{name}", "state": block["state"], "reason": block["reason"]}
           for name, block in record["text"].items() if block["state"] not in (CAPTURED, NOT_APPLICABLE)]
    if record["evidence"]["provenance"] != EXACT:
        out.append({"field": "evidence", "state": CAPTURE_FAILED, "reason": record["evidence"]["reason"]})
    if record["validation"]["state"] not in (CAPTURED, NOT_APPLICABLE):
        out.append({"field": "validation", "state": record["validation"]["state"],
                    "reason": record["validation"].get("reason")})
    return out


def build_record(*, question_id: str, ordinal: int, question: str, out: Any, run: dict[str, Any],
                 captured_utc: str, evaluation: dict[str, Any] | None = None,
                 review: dict[str, Any] | None = None, origin: dict[str, Any] | None = None,
                 not_run_reason: str | None = None) -> dict[str, Any]:
    """The measurement record of one finished `RagAnswer`. Raises on a capture defect; the runner calls
    `capture_safely`, which turns that into a visible capture_error record."""
    m = _copy(out.metadata)
    gen = m.get("generation") or {}
    val = m.get("validation")
    model_called = bool(out.model_called)

    raw = m.get("raw_model_output")
    if not model_called:
        raw_block = _text(NOT_APPLICABLE, reason="the model was not called; the answer is deterministic")
    elif raw is None:
        raw_block = _text(NOT_EXPOSED, reason="the model was called but metadata.raw_model_output is absent")
    else:
        raw_block = _text(CAPTURED, raw)

    if val is None:
        validation_input = _text(NOT_APPLICABLE, reason="validation did not run for this answer")
        repair_output = _text(NOT_APPLICABLE, reason="validation did not run for this answer")
    else:
        validation_input = (_text(CAPTURED, m["repair_input"]) if m.get("repair_input") is not None else
                            _text(NOT_EXPOSED, reason="the pipeline records the text it validated only when repair "
                                                      "runs (metadata.repair_input); no repair ran"))
        if val.get("repair") is None:
            repair_output = _text(NOT_APPLICABLE, reason="no repair ran")
        elif m.get("repair_output") is not None:
            repair_output = _text(CAPTURED, m["repair_output"])
        else:
            repair_output = _text(NOT_EXPOSED, reason="a repair ran but metadata.repair_output is absent")

    snapshot = out.evidence_snapshot
    if snapshot is not None:
        snap = load_evidence(_copy(snapshot.to_dict()))     # verified: the fingerprint matches the texts
        evidence = {"provenance": EXACT, "reason": None, "fingerprint": snap.fingerprint,
                    "item_text_sha256": [sha256_text(i.text) for i in snap.items], "snapshot": _copy(snapshot.to_dict())}
    else:
        evidence = {"provenance": MISSING, "reason": "the pipeline did not capture an evidence snapshot "
                                                    "(RagAnswer.evidence_snapshot is None)",
                    "fingerprint": None, "item_text_sha256": None, "snapshot": None}
    evidence.update({
        # retrieval rank and score of every result, before selection (no text: the text is in the snapshot)
        "retrieval_ranking": _copy(out.to_dict()["retrieved"]),
        "selection": m.get("selection"), "coverage": m.get("coverage"), "coverage_action": m.get("coverage_action"),
    })

    record: dict[str, Any] = {
        "schema": MEASUREMENT_SCHEMA, "hash_algorithm": HASH_ALGORITHM, "canonicalization": CANONICALIZATION,
        "status": STATUS_CAPTURED, "complete": False, "gaps": [],
        "question_id": question_id, "ordinal": ordinal, "captured_utc": captured_utc, "run": _copy(run),
        "question": {"text": question, "sha256": sha256_text(question),
                     "matches_pipeline_question": out.question == question},
        "text": {"raw_model_output": raw_block, "validation_input": validation_input,
                 "repair_output": repair_output, "delivered_answer": _text(CAPTURED, out.answer)},
        "transformations": {
            "label": "recorded by the pipeline",
            "duplicate_sentences_removed": gen.get("duplicate_sentences_removed", 0) if model_called else None,
            "citations_repointed": gen.get("citations_repointed") if model_called else None,
            "normalization": m.get("normalization"),
            "stop_reason": gen.get("stop_reason"), "finished": gen.get("finished"),
            "generated_tokens": gen.get("generated_tokens"), "max_new_tokens": gen.get("max_new_tokens"),
        },
        "evidence": evidence,
        "validation": ({"state": NOT_APPLICABLE, "reason": "validation did not run for this answer",
                        "initial": None, "final": None, "repair": None} if val is None else
                       {"state": CAPTURED, "reason": None, "initial": val.get("initial"), "final": val.get("final"),
                        "repair": val.get("repair")}),
        "citations": [_copy(c.to_dict()) for c in out.citations],
        # what the evaluator reads from the RagAnswer beyond the evidence, text and validation above, so an
        # offline evaluation (regulus.evaluation.offline_evaluation) can be given exactly what a live one is
        "evaluator_inputs": {
            "model_called": model_called,
            "scope_status": (m.get("question_analysis") or {}).get("scope_status"),
            "package_present": out.package is not None,
            "answer_plan": (_copy(out.package.answer_plan.to_dict())
                            if out.package is not None and out.package.answer_plan is not None else None),
            "withheld": [[c, t] for c, t in out.package.withheld] if out.package is not None else [],
        },
        "inferred": {
            "label": "inferred by text comparison - NOT a pipeline record of what was removed or appended",
            "delivered_equals_raw_output_stripped": (out.answer == raw.strip()) if raw is not None else None,
            "delivered_equals_validation_input": (out.answer == m["repair_input"])
                                                 if m.get("repair_input") is not None else None,
        },
        "evaluation": ({"state": CAPTURED, "assessment": _copy(evaluation), "review": _copy(review)}
                       if evaluation is not None else {"state": "not_run", "reason": not_run_reason or NOT_RUN_BY_RUNNER}),
    }
    if origin is not None:                       # absent from acceptance-run records, which keep their shape
        record["origin"] = _copy(origin)
    for key in ("trace", "citation_verdicts", "answer_verdict"):   # TR1, CC1, AV1 (report-only): present only when the switch is on
        if m.get(key) is not None:
            record[key] = m[key]
    record["validation"].update({k: m.get(k) for k in ("fallback_type", "fallback_reason", "answerability",
                                                       "generation_blocked", "block_reason", "citation_support")})
    record["gaps"] = _gaps(record)
    required = {g["field"] for g in record["gaps"]} & set(REQUIRED)
    record["complete"] = not required
    return _seal(record)


def failure_record(*, question_id: str, ordinal: int, question: str, run: dict[str, Any], captured_utc: str,
                   status: str, error: str, origin: dict[str, Any] | None = None) -> dict[str, Any]:
    """A record that says, in place of the measurement, why there is none. Never complete."""
    record = {
        "schema": MEASUREMENT_SCHEMA, "hash_algorithm": HASH_ALGORITHM, "canonicalization": CANONICALIZATION,
        "status": status, "complete": False,
        "gaps": [{"field": "*", "state": CAPTURE_FAILED if status == STATUS_CAPTURE_ERROR else NOT_APPLICABLE,
                  "reason": error}],
        "question_id": question_id, "ordinal": ordinal, "captured_utc": captured_utc, "run": _copy(run),
        "question": {"text": question, "sha256": sha256_text(question)}, "error": error,
    }
    if origin is not None:
        record["origin"] = _copy(origin)
    return _seal(record)


def capture_safely(**kwargs) -> dict[str, Any]:
    """`build_record` for the answering loop: a capture defect becomes a sealed capture_error record (the
    answer and its row are unaffected), so a run never loses or misreports a question to it."""
    try:
        record = build_record(**kwargs)
        canonical(record)                         # proven serialisable before it is handed back
        return record
    except Exception as exc:  # noqa: BLE001 - capture is an observer of the answer
        return failure_record(question_id=kwargs.get("question_id"), ordinal=kwargs.get("ordinal"),
                              question=kwargs.get("question"), run=kwargs.get("run") or {},
                              captured_utc=kwargs.get("captured_utc"), status=STATUS_CAPTURE_ERROR,
                              error=f"{type(exc).__name__}: {exc}"[:2000], origin=kwargs.get("origin"))


# ------------------------------------------------------------------ verification


def verify_record(record: dict[str, Any]) -> list[str]:
    """Every integrity problem in a record, recomputed from its own content ([] when it verifies)."""
    problems: list[str] = []
    if record.get("schema") != MEASUREMENT_SCHEMA:
        return [f"unknown measurement schema {record.get('schema')!r}"]
    if record.get("record_sha256") != record_hash(record):
        problems.append("record_sha256 does not match the record's content")
    for name, block in (record.get("text") or {}).items():
        if block.get("sha256") != sha256_text(block.get("text")):
            problems.append(f"text.{name}: sha256 does not match its text")
    evidence = record.get("evidence") or {}
    if evidence.get("snapshot") is not None:
        try:
            snap = load_evidence(evidence["snapshot"])
        except Exception as exc:  # noqa: BLE001 - reported, never repaired
            problems.append(f"evidence.snapshot: {exc}")
        else:
            if evidence.get("fingerprint") != snap.fingerprint:
                problems.append("evidence.fingerprint differs from the snapshot's")
            if evidence.get("item_text_sha256") != [sha256_text(i.text) for i in snap.items]:
                problems.append("evidence.item_text_sha256 does not match the snapshot's item texts")
    return problems


# ------------------------------------------------------------------ reading a run


def legacy_view(row: dict[str, Any], ordinal: int) -> dict[str, Any]:
    """An `answers.jsonl` row that has no measurement record, described as exactly what it holds."""
    has = lambda key: key in row                 # noqa: E731
    snapshot, evidence_state, reason = row.get("evidence_snapshot"), MISSING, "the row holds no evidence snapshot"
    if snapshot is not None:
        try:
            load_evidence(snapshot)
            evidence_state, reason = EXACT, None
        except Exception as exc:  # noqa: BLE001
            reason = f"stored snapshot does not verify: {exc}"

    def text(key):
        if not has(key):
            return {"state": NOT_RECORDED, "text": None}
        return {"state": CAPTURED if row[key] is not None else NOT_RECORDED, "text": row[key]}

    return {
        "source": "legacy_answers_row", "schema": None, "complete": False, "question_id": row.get("question_id"),
        "ordinal": ordinal, "execution_status": row.get("execution_status"),
        "text": {"raw_model_output": text("raw_model_output"), "delivered_answer": text("answer"),
                 "validation_input": {"state": NOT_RECORDED, "text": None},
                 "repair_output": {"state": NOT_RECORDED, "text": None}},
        "evidence": {"provenance": evidence_state, "reason": reason,
                     "labels": row.get("evidence") if has("evidence") else None},
        "validation": {"state": NOT_RECORDED if not has("validation_initial") else "status_only",
                       "initial_status": row.get("validation_initial"), "initial_codes": row.get("validation_initial_codes"),
                       "final_status": row.get("validation_final"), "repair": row.get("repair"),
                       "problems": NOT_RECORDED},
        "evaluation": {"state": NOT_RECORDED},
    }


def load_run(run_dir: Path | str) -> dict[str, Any]:
    """Every answer of a run in `answers.jsonl` order, with its measurement record when one exists and
    verifies against itself, otherwise its legacy view. Never raises on a malformed or missing line."""
    run_dir = Path(run_dir)
    problems: list[str] = []
    measured: dict[str, dict[str, Any]] = {}
    path = run_dir / MEASUREMENT_FILE
    if path.is_file():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError as exc:
                problems.append(f"{MEASUREMENT_FILE}:{n}: not JSON ({exc})")
                continue
            qid = rec.get("question_id")
            if qid in measured:
                problems.append(f"{MEASUREMENT_FILE}:{n}: duplicate record for {qid}; the first is kept")
                continue
            measured[qid] = rec
    rows = []
    answers = run_dir / "answers.jsonl"
    lines = answers.read_text(encoding="utf-8").splitlines() if answers.is_file() else []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            problems.append(f"answers.jsonl:{n}: not JSON ({exc})")
            continue
        rec = measured.get(row.get("question_id"))
        if rec is None:
            rows.append(legacy_view(row, ordinal=len(rows) + 1))
            continue
        issues = verify_record(rec)
        rows.append({"source": "measurement_record", "question_id": rec.get("question_id"),
                     "ordinal": rec.get("ordinal"), "verified": not issues, "problems": issues,
                     "complete": bool(rec.get("complete")) and not issues, "record": rec})
    orphans = sorted(set(measured) - {r["question_id"] for r in rows}, key=str)
    if orphans:
        problems.append(f"{MEASUREMENT_FILE}: records without an answers.jsonl row: {', '.join(map(str, orphans))}")
    return {"run_dir": str(run_dir), "rows": rows, "problems": problems}
