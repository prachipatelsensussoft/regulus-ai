"""Exact evidence capture and replay (regulus.traceability).

The property under test: for every generated answer, the accuracy evaluator
judges the answer against the exact evidence text that was placed in the
model's prompt - captured, serialised, reloaded and fingerprinted - and never
against evidence it retrieved, rebuilt or approximated itself.

Every test runs the real Stage 1-7 pipeline on the in-memory tables with a
generator that RECORDS the messages it is sent, so "what the model saw" is the
literal prompt, not a re-rendering of it.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from regulus.accuracy import evaluate
from regulus.accuracy import evaluator as EV
from regulus.inference.generate import GenerationResult
from regulus.rag import RagConfig, RagPipeline
from regulus.rag.context import ContextPolicy
from regulus.rag.generation import SOURCES_CLOSE, SOURCES_OPEN
from regulus.retrieval import LexicalRetriever
from regulus.traceability import evidence as TR
from tests.test_accuracy_evaluation import PURCHASING, TABLE
from tests.test_retrieval_plan import TableDatabase, guidance_row

ONE = "What does 21 CFR 820.10(b)(2) require?"
DOCKET_Q = "Summarize FDA guidance FDA-2016-D-2565."
MIXED_Q = "What guidance exists on 21 CFR 820.10(b)(2)?"
G1 = ("Manufacturers should evaluate suppliers on the basis of their ability to meet specified requirements, and "
      "should document the results of supplier evaluations.")
ALWAYS = RagConfig(optional_context="always")   # supporting guidance placed in the prompt too
G2 = ("Purchasing controls should be proportionate to the risk the purchased product presents to the finished "
      "device and its intended use.")


class Recording:
    """A scripted generator that keeps every message list it is sent."""

    def __init__(self, response):
        self.response = response
        self.calls: list[list[dict]] = []

    def generate(self, messages, max_new_tokens=None):
        self.calls.append([dict(m) for m in messages])
        return GenerationResult("p", self.response, 500, 40, 1.0, finished=True)

    def describe(self):
        return {"model_id": "recording"}


def answer(question, response, *, db=None, config=None):
    gen = Recording(response)
    db = db or TableDatabase(TABLE)
    out = RagPipeline(LexicalRetriever(db), gen, config or RagConfig()).answer(question)
    return out, gen, db


def prompt_block(gen) -> str:
    """The SOURCES block of the user message actually sent to the model."""
    assert len(gen.calls) == 1, "one model call per answer"
    return TR.sources_block(gen.calls[0][-1]["content"], SOURCES_OPEN, SOURCES_CLOSE)


_SOURCE_RE = re.compile(r"(?ms)^SOURCE (\d+)\n.*?^Citation: (.*?)\n(?:Heading: (.*?)\n)?Text:\n(.*?)(?=\n\nSOURCE \d+\n|\Z)")


def parsed_prompt(gen) -> list[dict]:
    """Each source as it literally appears in the prompt: number, citation, heading, text."""
    return [{"index": int(m.group(1)), "citation": m.group(2), "heading": m.group(3), "text": m.group(4)}
            for m in _SOURCE_RE.finditer(prompt_block(gen))]


def model_visible_fingerprint(gen, snapshot) -> str:
    """The fingerprint computed with every TEXT, citation, heading and number taken
    from the prompt as sent (identity and tier from the record they belong to)."""
    seen = parsed_prompt(gen)
    assert len(seen) == len(snapshot.items)
    items = [replace(item, index=p["index"], citation=p["citation"], heading=p["heading"], text=p["text"])
             for item, p in zip(snapshot.items, seen)]
    return TR.fingerprint(items)


def roundtrip(snapshot) -> TR.EvidenceSnapshot:
    return TR.load(json.loads(json.dumps(snapshot.to_dict())))


class SourcesSpy:
    """Records the sources the evaluator's judging code is handed."""

    def __init__(self, monkeypatch):
        self.seen: list[list[str]] = []
        real = EV.passages

        def spy(sources, **kw):
            self.seen.append([s.text for s in sources])
            return real(sources, **kw)

        monkeypatch.setattr(EV, "passages", spy)


# ======================================================= 1-3: capture


def test_1_ecfr_evidence_is_captured_exactly_as_placed_in_the_prompt():
    out, gen, _ = answer(ONE, f"21 CFR 820.10(b)(2): {PURCHASING} [1]")
    snap = out.evidence_snapshot
    assert snap is not None and snap.capture_point == TR.CAPTURE_PACKAGE and snap.model_called
    assert [i.text for i in snap.items] == [s.text for s in out.package.sources]
    assert [p["text"] for p in parsed_prompt(gen)] == [i.text for i in snap.items]
    assert snap.prompt_verified is True
    assert any(i.source_type == "ecfr" and i.section == "820.10" and i.paragraph == "(b)(2)" for i in snap.items)


def test_2_guidance_evidence_is_captured_like_ecfr():
    db = TableDatabase(TABLE, docket=[guidance_row(0, body_text=G1), guidance_row(1, body_text=G2)])
    out, gen, _ = answer(DOCKET_Q, f"The guidance says: {G1} [1]", db=db)
    snap = out.evidence_snapshot
    guidance = [i for i in snap.items if i.source_type == "guidance"]
    assert [i.text for i in guidance] == [G1, G2]
    assert all(i.corpus == "fda_guidance" and i.document == "QMS Guidance" for i in guidance)
    assert all(i.citation_fields.get("docket_number") == "FDA-2016-D-2565" for i in guidance)
    assert [p["text"] for p in parsed_prompt(gen)] == [G1, G2]


def test_3_multiple_sources_from_both_corpora_are_captured_in_prompt_order():
    # optional_context="always" places the supporting guidance in the prompt beside the
    # exact eCFR row (by default it is left out when coverage is complete - and then the
    # snapshot rightly holds no guidance, because the model saw none)
    db = TableDatabase(TABLE, guidance=[guidance_row(0, rank=0.3, body_text=G1)])
    out, gen, _ = answer(MIXED_Q, f"21 CFR 820.10(b)(2): {PURCHASING} [1]", db=db,
                         config=RagConfig(optional_context="always"))
    snap = out.evidence_snapshot
    assert {i.source_type for i in snap.items} == {"ecfr", "guidance"}
    assert [(p["index"], p["citation"], p["text"]) for p in parsed_prompt(gen)] == \
        [(i.index, i.citation, i.text) for i in snap.items]


# ======================================================= 4-6: order, truncation, serialisation


def test_4_evidence_order_is_the_prompt_order_end_to_end(monkeypatch):
    db = TableDatabase(TABLE, guidance=[guidance_row(0, rank=0.3, body_text=G1)])
    out, gen, _ = answer(MIXED_Q, f"21 CFR 820.10(b)(2): {PURCHASING} [1]", db=db, config=ALWAYS)
    prompt_order = [p["text"] for p in parsed_prompt(gen)]
    assert len(prompt_order) >= 3 and G1 in prompt_order             # A, B, C - eCFR and guidance
    stored = roundtrip(out.evidence_snapshot)
    assert [i.text for i in stored.items] == prompt_order and [i.order for i in stored.items] == list(range(len(prompt_order)))
    spy = SourcesSpy(monkeypatch)
    evaluate(MIXED_Q, out.answer, evidence=stored)
    assert spy.seen == [prompt_order], "the evaluator receives A, B, C - not a re-ranked order"


def test_5_truncated_evidence_is_stored_and_replayed_truncated(monkeypatch):
    long_text = PURCHASING + " " + " ".join(f"Supplementary sentence {n} about purchasing records." for n in range(40))
    table = [dict(r, body_text=long_text) if r["section_number"] == "820.10" and r["paragraph_path"] == "(b)(2)" else r
             for r in TABLE]
    config = RagConfig(context=ContextPolicy(required_max_chars_per_result=600))
    out, gen, _ = answer(ONE, f"21 CFR 820.10(b)(2): {PURCHASING} [1]", db=TableDatabase(table), config=config)
    item = next(i for i in out.evidence_snapshot.items if i.paragraph == "(b)(2)")
    assert len(long_text) > 1000 and item.truncated and len(item.text) <= 600
    assert item.text != long_text and item.text in prompt_block(gen)
    spy = SourcesSpy(monkeypatch)
    evaluate(ONE, out.answer, evidence=roundtrip(out.evidence_snapshot))
    assert item.text in spy.seen[0] and long_text not in spy.seen[0]


def test_6_serialisation_round_trip_is_lossless():
    db = TableDatabase(TABLE, guidance=[guidance_row(0, rank=0.3, body_text=G1)])
    out, _, _ = answer(MIXED_Q, f"21 CFR 820.10(b)(2): {PURCHASING} [1]", db=db)
    snap = out.evidence_snapshot
    back = TR.load(snap.to_json())
    assert back == snap and back.to_dict() == snap.to_dict()


# ======================================================= 7-9: fingerprint


def _items(*texts):
    return tuple(TR.EvidenceItem(order=n, index=n + 1, evidence_id=f"e{n}", source_type="ecfr", corpus="ecfr",
                                 citation=f"21 CFR 1.{n}", heading=None, text=t, truncated=False)
                 for n, t in enumerate(texts))


def test_7_fingerprint_is_stable():
    out_a, _, _ = answer(ONE, f"21 CFR 820.10(b)(2): {PURCHASING} [1]")
    out_b, _, _ = answer(ONE, f"21 CFR 820.10(b)(2): {PURCHASING} [1]")
    assert out_a.evidence_snapshot.fingerprint == out_b.evidence_snapshot.fingerprint
    assert TR.fingerprint(_items("A", "B")) == TR.fingerprint(_items("A", "B"))
    assert re.fullmatch(r"[0-9a-f]{64}", out_a.evidence_snapshot.fingerprint)


def test_8_fingerprint_changes_when_text_changes_or_items_are_added_or_removed():
    base = TR.fingerprint(_items("A text", "B text"))
    assert TR.fingerprint(_items("A text", "B text.")) != base          # one character
    assert TR.fingerprint(_items("A text", "B text", "C text")) != base  # added
    assert TR.fingerprint(_items("A text")) != base                      # removed


def test_9_fingerprint_changes_when_order_changes():
    a, b = _items("A text", "B text")
    swapped = (replace(b, order=0, index=1), replace(a, order=1, index=2))
    assert TR.fingerprint(swapped) != TR.fingerprint((a, b))


def test_9b_a_tampered_record_is_rejected_not_used():
    out, _, _ = answer(ONE, f"21 CFR 820.10(b)(2): {PURCHASING} [1]")
    record = out.evidence_snapshot.to_dict()
    record["items"][0]["text"] = record["items"][0]["text"].replace("shall", "may")
    with pytest.raises(TR.EvidenceIntegrityError):
        TR.load(record)
    a = evaluate(ONE, out.answer, evidence=record)
    assert a.accuracy is None and a.evidence["provenance"] == TR.MISSING and "integrity" in a.reason


# ======================================================= 10-11: the evaluator


def test_10_evaluator_receives_the_exact_stored_evidence(monkeypatch):
    out, gen, _ = answer(ONE, f"21 CFR 820.10(b)(2): {PURCHASING} [1]")
    spy = SourcesSpy(monkeypatch)
    a = evaluate(ONE, out.answer, package=out.package, evidence=out.evidence_snapshot, coverage=out.coverage,
                 validation=out.validation)
    assert spy.seen == [[p["text"] for p in parsed_prompt(gen)]]
    assert a.evidence["provenance"] == TR.EXACT and a.evidence["fingerprint"] == out.evidence_snapshot.fingerprint
    assert a.evidence["package_matches_snapshot"] is True and a.to_dict()["evidence_provenance"] == TR.EXACT


class _Forbidden:
    def __init__(self, name):
        self.name = name

    def __call__(self, *a, **k):
        raise AssertionError(f"the evaluator performed {self.name}")


class RecordingIndex:
    """The evaluator's optional identifier-existence index: booleans only."""

    def __init__(self):
        self.calls = []

    def section_status(self, section):
        self.calls.append(("section_status", section))
        return True, True

    def part_status(self, part):
        self.calls.append(("part_status", part))
        return True, True


class RecordingEmbedder:
    def __init__(self):
        self.texts: list[str] = []

    def embed_batch(self, texts, **kw):
        self.texts.extend(texts)
        return [[float(len(t) % 7), 1.0, float(len(t.split()) % 5)] for t in texts]

    def describe(self):
        return {"model_id": "recording"}


def test_11_evaluation_performs_no_retrieval_of_any_kind(monkeypatch):
    from regulus.accuracy.alignment import SemanticAligner
    from regulus.retrieval import hybrid, lexical, semantic

    db = TableDatabase(TABLE, guidance=[guidance_row(0, rank=0.3, body_text=G1)])
    out, _, _ = answer(MIXED_Q, f"21 CFR 820.10(b)(2): {PURCHASING} [1]", db=db)
    stored = roundtrip(out.evidence_snapshot)
    # every retrieval entry point, and the database, now fails loudly if touched
    monkeypatch.setattr(TableDatabase, "query", _Forbidden("a database query"))
    for module in (lexical, semantic, hybrid):
        for name in dir(module):
            cls = getattr(module, name)
            if isinstance(cls, type) and cls.__module__ == module.__name__:
                for method in ("search", "run_plan", "retrieve", "search_guidance", "search_ecfr"):
                    if hasattr(cls, method):
                        monkeypatch.setattr(cls, method, _Forbidden(f"{module.__name__}.{name}.{method}"))
    index, embedder = RecordingIndex(), RecordingEmbedder()
    a = evaluate(MIXED_Q, out.answer, evidence=stored, identifier_index=index, aligner=SemanticAligner(embedder))
    assert a.evidence["provenance"] == TR.EXACT and a.status is not None
    # the identifier index is asked only whether identifiers exist - never for text
    assert {name for name, _ in index.calls} <= {"section_status", "part_status"}
    # the embedder sees only the answer's claims and the captured passages: no corpus search.
    # (claims are the answer's sentences with citation markers moved, so compare words)
    words = lambda t: " ".join(re.findall(r"[a-z0-9]+", re.sub(r"\[\d+\]", " ", t.lower())))
    captured, said = words(" ".join(i.text for i in stored.items)), words(out.answer)
    assert embedder.texts and all(words(t) in captured or words(t) in said for t in embedder.texts)


# ======================================================= 12-13: provenance


def test_12_old_artifact_without_evidence_is_labelled_not_exact():
    out, _, _ = answer(ONE, f"21 CFR 820.10(b)(2): {PURCHASING} [1]")
    rebuilt = evaluate(ONE, out.answer, package=out.package, coverage=out.coverage)   # no snapshot: legacy
    assert rebuilt.evidence["provenance"] == TR.LEGACY_RECONSTRUCTED and rebuilt.evidence["fingerprint"] is None
    with pytest.raises(ValueError):
        evaluate(ONE, out.answer, package=out.package, evidence_provenance=TR.EXACT)   # cannot claim EXACT
    missing = evaluate(ONE, out.answer)
    assert missing.evidence["provenance"] == TR.MISSING and missing.accuracy is None
    assert missing.checked["evidence_status"] == "not_captured"


def test_13_evidence_not_captured_is_distinct_from_an_unsupported_claim():
    draft = "21 CFR 820.10(b)(2) requires an independent annual supplier audit by a notified body [1]."
    out, _, _ = answer(ONE, draft)
    # the validator withholds this invented draft, so the delivered answer holds no claim of the model's own; the
    # evaluator is given the draft, as it was when an unsupported claim could still be delivered
    judged = evaluate(ONE, draft, evidence=out.evidence_snapshot)
    claims = judged.checked["claims"]
    assert judged.evidence["provenance"] == TR.EXACT
    assert claims and all(c["support"] != "supported" for c in claims)          # unsupported, judged
    not_captured = evaluate(ONE, draft)
    assert not_captured.evidence["provenance"] == TR.MISSING                    # not judged at all
    assert "not captured" in not_captured.reason and not_captured.checked.get("claims") is None
    # and a captured package with nothing in it is "no evidence retrieved", still EXACT
    empty = replace(out.evidence_snapshot, items=(), fingerprint=TR.fingerprint(()))
    nothing = evaluate(ONE, draft, evidence=empty)
    assert nothing.evidence["provenance"] == TR.EXACT and "No authoritative regulatory evidence was retrieved" in nothing.reason


# ======================================================= 14-15: identity, production


def test_14_model_stored_and_evaluator_evidence_are_identical_at_text_level(monkeypatch):
    db = TableDatabase(TABLE, guidance=[guidance_row(0, rank=0.3, body_text=G1)])
    out, gen, _ = answer(MIXED_Q, f"21 CFR 820.10(b)(2): {PURCHASING} [1]", db=db, config=ALWAYS)
    sent = prompt_block(gen)
    stored = roundtrip(out.evidence_snapshot)
    spy = SourcesSpy(monkeypatch)
    a = evaluate(MIXED_Q, out.answer, evidence=stored)
    model_texts = [p["text"] for p in parsed_prompt(gen)]
    assert model_texts == [i.text for i in stored.items] == spy.seen[0]
    import hashlib
    assert hashlib.sha256(sent.encode()).hexdigest() == stored.prompt_sha256
    assert model_visible_fingerprint(gen, stored) == stored.fingerprint == a.evidence["fingerprint"]


def test_15_capture_does_not_change_the_answer_or_the_prompt(monkeypatch):
    cases = [(ONE, f"21 CFR 820.10(b)(2): {PURCHASING} [1]"), (ONE, ""), (MIXED_Q, f"{PURCHASING} [1]"),
             ("What does 21 CFR 820.99(z) require?", "It requires an annual audit [1].")]
    with_capture = [answer(q, r) for q, r in cases]
    monkeypatch.setattr(TR, "capture_safely", lambda *a, **k: None)            # capture switched off
    without = [answer(q, r) for q, r in cases]
    for (a, ga, _), (b, gb, _) in zip(with_capture, without):
        assert a.answer.encode() == b.answer.encode()
        assert ga.calls == gb.calls                                              # same prompts, same call count
        # wall-clock figures are the only thing allowed to differ (as in test_accuracy_observer)
        strip = lambda d: {**d, "timings_ms": None,
                           "metadata": {k: v for k, v in d["metadata"].items() if k != "retrieval_timings_ms"}}
        assert strip(a.to_dict()) == strip(b.to_dict())
        assert [c.to_dict() for c in a.citations] == [c.to_dict() for c in b.citations]
        assert a.evidence_snapshot is not None and b.evidence_snapshot is None


# ======================================================= 18: database changes after generation


def test_18_database_change_after_generation_does_not_reach_the_evaluator(monkeypatch):
    db = TableDatabase(TABLE, guidance=[guidance_row(0, rank=0.3, body_text=G1)])
    out, gen, _ = answer(MIXED_Q, f"21 CFR 820.10(b)(2): {PURCHASING} [1]", db=db, config=ALWAYS)
    before = [p["text"] for p in parsed_prompt(gen)]
    assert G1 in before
    stored = out.evidence_snapshot.to_json()
    for r in db.table:                                   # the corpus is amended after the answer was given
        r["body_text"] = "AMENDED: " + r["body_text"]
    db.canned["guidance"] = [dict(g, body_text="AMENDED guidance") for g in db.canned["guidance"]]
    spy = SourcesSpy(monkeypatch)
    a = evaluate(MIXED_Q, out.answer, package=out.package, evidence=TR.load(stored))
    assert spy.seen == [before] and not any("AMENDED" in t for t in spy.seen[0])
    assert a.evidence["fingerprint"] == json.loads(stored)["fingerprint"]


# ======================================================= the guidance regression (§15)


def test_guidance_regression_capture_serialise_replay(monkeypatch):
    """An answer supported by FDA guidance: the guidance passage is (1) in the
    model-visible evidence, (2) captured, (3) survives serialisation, (4) reaches
    the evaluator, (5) as the same text, (6) with no retrieval, (7) under the
    same fingerprint - and the evaluator's verdict rests on it."""
    db = TableDatabase(TABLE, docket=[guidance_row(0, body_text=G1), guidance_row(1, body_text=G2)])
    claim = ("Manufacturers should evaluate suppliers on the basis of their ability to meet specified requirements, "
             "and should document the results of supplier evaluations [1].")
    out, gen, _ = answer(DOCKET_Q, claim, db=db)
    assert G1 in prompt_block(gen)                                                      # 1
    captured = next(i for i in out.evidence_snapshot.items if i.text == G1)
    assert captured.source_type == "guidance"                                           # 2
    on_disk = json.dumps({"answer": out.answer, "evidence_snapshot": out.evidence_snapshot.to_dict()})
    record = json.loads(on_disk)
    reloaded = TR.load(record["evidence_snapshot"])
    assert any(i.text == G1 and i.source_type == "guidance" for i in reloaded.items)   # 3
    monkeypatch.setattr(TableDatabase, "query", _Forbidden("a database query"))         # 6
    spy = SourcesSpy(monkeypatch)
    a = evaluate(DOCKET_Q, record["answer"], evidence=reloaded)
    assert G1 in spy.seen[0]                                                            # 4, 5
    assert a.evidence["fingerprint"] == out.evidence_snapshot.fingerprint == \
        model_visible_fingerprint(gen, reloaded)                                        # 7
    assert a.evidence["provenance"] == TR.EXACT and a.evidence["guidance_items"] == 2
    supported = [c for c in a.checked["claims"] if c["support"] == "supported"]
    assert supported and supported[0]["evidence"][0]["citation"] == captured.citation


def test_a_blocked_answer_still_records_the_evidence_it_was_given():
    out, gen, _ = answer("What does 21 CFR 820.99(z) require?", "unused")
    snap = out.evidence_snapshot
    assert snap is not None and snap.model_called is (len(gen.calls) == 1)
    if not gen.calls:
        assert snap.prompt_sha256 is None and snap.prompt_verified is None
