"""TR1 (report-only): the retrieval and selection trace records, never decides - same answer with the switch on or off."""
from __future__ import annotations

import pytest

from regulus.rag import RagPipeline
from regulus.rag import section_promotion as SP
from regulus.rag import trace
from regulus.rag import validator_switches as VS
from regulus.rag.section_promotion import Promotion, SectionPromoter
from regulus.retrieval import LexicalRetriever
from tests.test_answer_validation import ScriptedGenerator
from tests.test_retrieval_plan import TableDatabase


@pytest.fixture
def tr1_on(monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"TR1"})


@pytest.fixture
def tr1_off(monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"TR1"})


def test_tr1_is_off_by_default_and_registered():
    assert "TR1" not in VS.DEFAULT.split(",")


def test_off_nothing_is_recorded_and_no_builder_runs(tr1_off):
    assert trace.begin() is None and trace.active() is None
    called = []
    trace.lazy("rank", "RO1", lambda: called.append(1) or {"x": 1})
    trace.event("rank", "RO1", x=1)
    assert called == []


def test_on_events_are_recorded_as_plain_json(tr1_on):
    tr = trace.begin()
    trace.event("candidates", "listed", candidates=[("111.70", 0.123456)], parts=("111",), seen={"b", "a"})
    trace.lazy("pick", "picker", lambda: {"raw": "1, 3", "picks": ["111.70", "111.75"]})
    trace.lazy("pick", "nothing", lambda: None)   # a builder may decline to record
    d = tr.to_dict()
    assert d["version"] == trace.VERSION
    assert d["events"] == [
        {"step": "candidates", "action": "listed", "candidates": [["111.70", 0.1235]], "parts": ["111"], "seen": ["a", "b"]},
        {"step": "pick", "action": "picker", "raw": "1, 3", "picks": ["111.70", "111.75"]}]


def test_a_failing_builder_never_reaches_the_caller(tr1_on):
    tr = trace.begin()
    trace.lazy("rank", "RO1", lambda: {"x": [][3]})   # IndexError inside the hook
    trace.lazy("rank", "RO1", lambda: {"x": undefined_name})  # noqa: F821 - NameError inside the hook
    assert tr.to_dict()["events"] == []


def test_a_runaway_hook_is_capped(tr1_on, monkeypatch):
    monkeypatch.setattr(trace, "MAX_EVENTS", 3)
    tr = trace.begin()
    for i in range(5):
        trace.event("x", "y", i=i)
    d = tr.to_dict()
    assert len(d["events"]) == 3 and d["events_dropped"] == 2


def test_begin_replaces_the_previous_questions_trace(tr1_on):
    first = trace.begin()
    trace.event("a", "b")
    second = trace.begin()
    trace.event("c", "d")
    assert [e["step"] for e in first.to_dict()["events"]] == ["a"]
    assert [e["step"] for e in second.to_dict()["events"]] == ["c"]


# --------------------------------------------------------------------------------------- promoter hooks, same output

def _cands(*secs):
    return [{"section": s, "part": s.split(".")[0], "heading": "", "subpart": "", "snippet": ""} for s in secs]


def _promoter(monkeypatch, scores):
    pr = SectionPromoter(db=None, mode="picker", index=object())
    monkeypatch.setattr(pr, "rerank_scores", lambda q, cands: {c["section"]: scores[c["section"]] for c in cands
                                                               if c["section"] in scores})
    monkeypatch.setattr(SP, "RERANKER", "stub")

    class _Ents:
        def match(self, q):
            return True

        def is_color_listing(self, s):
            return False

    import regulus.retrieval.entities as entities
    monkeypatch.setattr(entities, "entity_index", lambda index: _Ents())
    return pr


def test_ro1_order_is_the_same_with_tr1_and_recorded(monkeypatch):
    scores = {"111.70": (0.2, "(a)"), "111.75": (0.9, "(b)"), "111.80": (0.5, "")}
    pr = _promoter(monkeypatch, scores)
    q = "What specifications must be established?"
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"TR1"})
    trace.begin()
    off = [c["section"] for c in pr.rerank_order(q, _cands("111.70", "111.75", "111.80"))]
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"TR1"})
    tr = trace.begin()
    on = [c["section"] for c in pr.rerank_order(q, _cands("111.70", "111.75", "111.80"))]
    assert on == off
    ev = [e for e in tr.to_dict()["events"] if e["action"] == "RO1"]
    assert ev and ev[0]["before"] == ["111.70", "111.75", "111.80"] and ev[0]["after"] == on


def test_ms1_additions_are_the_same_with_tr1_and_the_pool_is_recorded(monkeypatch):
    scores = {"111.75": (0.95, "(a)"), "111.80": (0.75, "(c)"), "111.83": (0.4, "")}
    pr = _promoter(monkeypatch, scores)
    monkeypatch.setattr(pr, "_section_text", lambda s: "x" * 100)
    for code in ("MS2", "PL1", "MS3", "XC1"):
        monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {code})
    cands = _cands("111.70", "111.75", "111.80", "111.83")
    pick = Promotion(sections=("111.70",), companions=(), method="picker")
    q = "Which specifications apply?"
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"TR1"})
    trace.begin()
    off = pr.reranked_companions(q, cands, pick)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"TR1"})
    tr = trace.begin()
    on = pr.reranked_companions(q, cands, pick)
    assert on == off == ["111.75", "111.80"]
    ms1 = next(e for e in tr.to_dict()["events"] if e["action"] == "MS1")
    assert ms1["added"] == on and ms1["floor"] == SP.MS1_MIN_SCORE
    assert ms1["pool"] == [["111.75", 0.95], ["111.80", 0.75], ["111.83", 0.4]]   # 111.83: below the floor, recorded


def test_companions_are_attributed_to_the_step_that_added_them(monkeypatch):
    pr = SectionPromoter(db=None, mode="picker", index=object())
    monkeypatch.setattr(pr, "companions", lambda s, q, lex: ["111.3"])
    monkeypatch.setattr(pr, "sibling_companions", lambda s, q: ["111.75"])
    monkeypatch.setattr(pr, "heading_matches", lambda s, q: [])
    monkeypatch.setattr(pr, "definition_companions", lambda q: ["111.3", "111.10"])   # 111.3 already in: not counted
    monkeypatch.setattr(pr, "asked_definitions", lambda s, q: [])
    for code in ("HT1", "SP3", "CS1"):
        monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {code})
    for code in ("HC1", "HM1", "DQ1"):
        monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {code})

    def run():
        p = pr._cfr_promotion(["111.70"], "q", set(), (), "1", None, "picker", 0.0)
        return p.sections, p.companions

    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"TR1"})
    trace.begin()
    off = run()
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"TR1"})
    tr = trace.begin()
    assert run() == off == (("111.70",), ("111.3", "111.75", "111.10"))
    ev = next(e for e in tr.to_dict()["events"] if e["action"] == "picked_section")
    assert ev["added_by"] == {"companions": ["111.3"], "HC1": ["111.75"], "definition_companions": ["111.10"]}


# ------------------------------------------------------------------------------- the whole pipeline, same answer

MULTI = "What do 820.10(b)(2) and 820.10(b)(4) require?"
DRAFT = "21 CFR 820.10(b)(2) requires traceability [1]. 21 CFR 820.10(b)(4) requires advisory notices [2]."


def _answer():
    return RagPipeline(LexicalRetriever(TableDatabase()), ScriptedGenerator(DRAFT)).answer(MULTI)


def test_the_pipeline_answer_is_byte_identical_and_only_tr1_adds_the_trace(monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"TR1"})
    off = _answer()
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"TR1"})
    on = _answer()
    assert on.answer == off.answer
    assert [s.text for s in on.context.sources] == [s.text for s in off.context.sources]
    assert "trace" not in off.metadata and "trace" not in off.diagnostics()
    assert on.metadata["trace"]["version"] == trace.VERSION and "trace" in on.diagnostics()
    strip = lambda m: {k: v for k, v in m.items() if k not in ("trace", "timings_ms")}   # noqa: E731
    assert strip(on.metadata).keys() == strip(off.metadata).keys()
