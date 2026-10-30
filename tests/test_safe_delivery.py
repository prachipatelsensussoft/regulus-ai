"""The answering server's delivery boundary (scripts/serve_ui.py): the page receives the validated answer
and nothing of a draft.

The model generates privately; validation and repair (or withholding) run on the complete text; only then
is an answer sent. While the model generates, the stream carries content-free progress events, at most one
per PROGRESS_INTERVAL_S. A rejected draft, a sentence repair removed, or a draft cut short by an error or a
timeout never reaches the browser: not as a stream event, not in the final result's diagnostics, not in an
error message, not from conversation history.

Each rejected draft below carries a canary word. Every byte a browser can receive - each stream event, the
/api/ask body, the stored conversation and its replay - is searched for it. The real pipeline (production
configuration) and the real evaluator run behind a stand-in engine that returns MORE than serve_ui.Engine
does (raw draft, repair input and output included), so the boundary is tested, not the engine's key choice.
No model, no database.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer

import pytest

import regulus.rag.pipeline as P
from regulus.accuracy import evaluate_safely
from regulus.accuracy.alignment import LexicalAligner
from regulus.accuracy.review import question_review
from regulus.config import PROJECT_ROOT, load_settings
from regulus.conversations import ConversationStore
from regulus.rag import RagPipeline
from regulus.retrieval import LexicalRetriever
from regulus.system_identity import configured_rag_config
from tests.test_answer_shape import TABLE
from tests.test_retrieval_plan import TableDatabase
from tests.test_streaming import StreamingGenerator

# the delivered answer is the validated draft itself (tests/conftest.py)
pytestmark = pytest.mark.usefixtures("delivered_as_drafted")

_spec = importlib.util.spec_from_file_location("serve_ui_safe_delivery", PROJECT_ROOT / "scripts" / "serve_ui.py")
UI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(UI)

CANARY = "zqxcanary"
QUESTION = "What does 21 CFR 820.10(b)(2) require?"
SOUND = "21 CFR 820.10(b)(2) requires traceability procedures under part 821, if applicable [1]."
WITHHELD = "could not be validated"
DRAFTS = {   # what the model is scripted to say -> how the pipeline treats it (production configuration)
    "unsupported": SOUND + f" Quarterly {CANARY} turbine logbooks stay mandatory for 45 days [1].",     # withheld
    "fabricated_citation": f"Under 21 CFR 999.77 the manufacturer must keep {CANARY} certificates [1].",  # withheld
    "repaired": SOUND + f" If appropriate, the {CANARY} applies [1].",            # sentence removed, rest delivered
}


class PipelineEngine:
    """serve_ui.Engine.ask's contract over the real pipeline and evaluator. Returns a superset of what the
    real engine returns, draft texts included, so that only the server's boundary can keep them out."""

    def __init__(self, generator):
        self.generator = generator

    def ask(self, question, *, adapter="", grounding=False, top_k=8, max_new_tokens=None, on_event=None, accuracy=None):
        out = RagPipeline(LexicalRetriever(TableDatabase(TABLE)), self.generator,
                          configured_rag_config(load_settings())).answer(question, on_event=on_event)
        acc = evaluate_safely(out.question, out.answer, package=out.package, evidence=out.evidence_snapshot,
                              citations=out.citations, validation=out.validation, coverage=out.coverage,
                              model_called=out.model_called, fallback_type=out.metadata.get("fallback_type"),
                              scope_status=(out.metadata.get("question_analysis") or {}).get("scope_status"),
                              answerability=out.metadata.get("answerability"), repair=out.metadata.get("validation"),
                              aligner=LexicalAligner()).to_dict()
        acc["question_review"] = question_review(acc, question=out.question)
        m = out.metadata
        return {"question": out.question, "answer": out.answer, "model_called": out.model_called,
                "citations": [c.to_dict() for c in out.citations],
                **{k: m.get(k) for k in ("generation", "validation", "quality_flags", "normalization", "citation_support",
                                         "fallback_type", "fallback_reason", "raw_model_output", "repair_input",
                                         "repair_output")},
                "evidence_snapshot": out.evidence_snapshot.to_dict() if out.evidence_snapshot is not None else None,
                "accuracy": acc, "server_timings_ms": {}}


@pytest.fixture
def server(tmp_path):
    started = []

    def start(engine):
        handler = type("H", (UI.Handler,), {"engine": engine, "store": ConversationStore(tmp_path / f"{uuid.uuid4().hex}.sqlite3"),
                                            "log_message": lambda *a: None})
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}", handler

    yield start
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


def call(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def events_of(raw):
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def every_route(base, question=QUESTION):
    """Everything a browser can receive for one question, by route: the stream, /api/ask, and in a
    conversation the stream, the stored conversation and the replayed turn."""
    raw = {}
    raw["stream"] = call(base, "/api/ask/stream", {"question": question})[1]
    raw["ask"] = call(base, "/api/ask", {"question": question})[1]
    status, text = call(base, "/api/conversations", {})
    assert status == 201
    cid, tid = json.loads(text)["conversation"]["id"], uuid.uuid4().hex
    raw["chat_stream"] = call(base, "/api/ask/stream", {"question": question, "conversation_id": cid, "turn_id": tid})[1]
    raw["history"] = call(base, f"/api/conversations/{cid}")[1]
    raw["replay"] = call(base, "/api/ask/stream", {"question": question, "conversation_id": cid, "turn_id": tid})[1]
    return raw


# ------------------------------------------------------------ approved answers are delivered


def test_an_approved_answer_is_delivered_intact_on_every_route_and_only_in_the_final_result(server):
    base, _ = server(PipelineEngine(StreamingGenerator(SOUND)))
    raw = every_route(base)
    for route in ("stream", "chat_stream"):
        events = events_of(raw[route])
        assert events[-1]["type"] == "final" and events[-1]["result"]["answer"] == SOUND
        assert events[-1]["result"]["citations"][0]["label"] == "21 CFR 820.10(b)(2)"
        assert "delta" not in [e["type"] for e in events] and "progress" in [e["type"] for e in events]
        assert not any("traceability procedures" in json.dumps(e) for e in events[:-1])   # no text before "final"
    assert json.loads(raw["ask"])["answer"] == SOUND
    assert events_of(raw["replay"])[-1]["result"]["answer"] == SOUND
    assert json.loads(raw["history"])["conversation"]["messages"][-1]["content"] == SOUND


# ------------------------------------------------------------ rejected and repaired drafts never leak


@pytest.mark.parametrize("kind", sorted(DRAFTS))
def test_no_route_ever_carries_text_of_a_rejected_draft(server, kind):
    base, _ = server(PipelineEngine(StreamingGenerator(DRAFTS[kind])))
    raw = every_route(base)
    final = events_of(raw["stream"])[-1]["result"]
    if kind == "repaired":
        assert final["answer"] == SOUND and final["validation"]["repair"] == "invented_conditions_removed"
    else:
        assert WITHHELD in final["answer"]
    for route, text in raw.items():
        assert CANARY not in text.lower(), route
    # the diagnostics stay useful: the initial report keeps its codes and statuses, not the draft's words
    initial = final["validation"]["initial"]
    assert initial["codes"] and initial["status"] != "valid"
    assert all(set(p) == {"code", "severity", "requirement_id", "citation"} for p in initial["problems"])
    assert "raw_model_output" not in final and "repair_input" not in final and "repair_output" not in final


def test_the_final_report_of_a_delivered_answer_is_kept_whole(server):
    base, _ = server(PipelineEngine(StreamingGenerator(DRAFTS["repaired"])))
    final = events_of(call(base, "/api/ask/stream", {"question": QUESTION})[1])[-1]["result"]
    assert final["validation"]["final"]["status"] == "valid" and "problems" in final["validation"]["final"]
    assert final["accuracy"]["question_review"]["status"]          # the evaluation still reaches the page


# ------------------------------------------------------------ failures never fall back to the draft


class Failing(StreamingGenerator):
    """Streams its draft (canary included) to the pipeline, then raises."""

    def __init__(self, response, exc):
        super().__init__(response)
        self.exc = exc

    def generate(self, messages, max_new_tokens=None, on_text=None):
        if on_text is not None:
            for word in self.response.split(" "):
                on_text(word + " ")
        raise self.exc


@pytest.mark.parametrize("exc", [RuntimeError(f"model crashed after '{CANARY} turbine'"),
                                 TimeoutError(f"read timed out; partial text: {CANARY}")],
                         ids=["generation_failure", "timeout"])
def test_a_generation_failure_or_timeout_shows_an_error_and_no_draft(server, exc):
    base, _ = server(PipelineEngine(Failing(DRAFTS["unsupported"], exc)))
    events = events_of(call(base, "/api/ask/stream", {"question": QUESTION})[1])
    assert events[-1]["type"] == "error" and events[-1]["error"].startswith(type(exc).__name__)
    status, body = call(base, "/api/ask", {"question": QUESTION})
    assert status == 500 and json.loads(body)["error"].startswith(type(exc).__name__)
    assert CANARY not in json.dumps(events).lower() and CANARY not in body.lower()


def test_a_validator_exception_shows_an_error_and_no_draft(server, monkeypatch):
    def broken(answer, *a, **k):
        raise ValueError(f"cannot validate: {answer}")             # a message that quotes the draft

    monkeypatch.setattr(P, "validate_answer", broken)
    base, _ = server(PipelineEngine(StreamingGenerator(DRAFTS["unsupported"])))
    raw = every_route(base)
    assert events_of(raw["stream"])[-1]["type"] == "error"
    assert events_of(raw["stream"])[-1]["error"].startswith("ValueError")
    for route, text in raw.items():
        assert CANARY not in text.lower(), route
    stored = json.loads(raw["history"])["conversation"]["messages"][-1]
    assert stored["status"] == "error" and CANARY not in stored["content"].lower()


def test_page_error_names_only_the_type_once_a_draft_may_exist_except_a_categorised_remote_failure():
    from regulus.rag.remote_generator import InferenceError

    assert UI.page_error(RuntimeError(f"{CANARY}"), draft_possible=True).startswith("RuntimeError")
    assert CANARY not in UI.page_error(RuntimeError(f"{CANARY}"), draft_possible=True)
    assert UI.page_error(InferenceError("timeout"), draft_possible=True) == "InferenceError: remote inference failed (timeout)"
    assert UI.page_error(ValueError("bad top_k"), draft_possible=False) == "ValueError: bad top_k"


def test_an_error_before_generation_keeps_its_message(server):
    class Down:
        def ask(self, question, **kw):
            raise ConnectionError("database unavailable")         # no model text exists yet

    base, _ = server(Down())
    events = events_of(call(base, "/api/ask/stream", {"question": QUESTION})[1])
    assert events[-1] == {"type": "error", "error": "ConnectionError: database unavailable"}


# ------------------------------------------------------------ the stream itself


def test_only_allowed_events_reach_the_page(server):
    class Chatty:
        def ask(self, question, *, on_event=None, **kw):
            on_event("context", {"sources": [], "no_context": False})
            on_event("generation_started", {"max_new_tokens": 64})
            on_event("delta", {"text": f"{CANARY} "})
            on_event("draft_note", {"text": f"{CANARY} note"})     # an event the boundary does not know
            on_event("generation_finished", {"generated_tokens": 2, "finished": True})
            return {"question": question, "answer": "Final [1].", "server_timings_ms": {}}

    base, _ = server(Chatty())
    raw = call(base, "/api/ask/stream", {"question": QUESTION})[1]
    assert [e["type"] for e in events_of(raw)] == ["start", "context", "generation_started", "progress",
                                                   "generation_finished", "final"]
    assert CANARY not in raw


def test_progress_is_throttled_and_carries_no_text(server):
    words = 30
    g = StreamingGenerator(" ".join(["word"] * words), delay=0.05)      # about 1.5 s of steady output
    base, _ = server(PipelineEngine(g))
    t0 = time.perf_counter()
    events = events_of(call(base, "/api/ask/stream", {"question": QUESTION})[1])
    elapsed = time.perf_counter() - t0
    progress = [e for e in events if e["type"] == "progress"]
    assert g.emitted == words                                          # the pipeline saw every chunk ...
    assert 1 <= len(progress) <= elapsed / UI.PROGRESS_INTERVAL_S + 1   # ... the page one event per interval
    assert len(progress) < words / 3
    assert all(set(e) == {"type", "elapsed_ms"} for e in progress)
    stream = events[-1]["result"]["stream"]
    assert stream["progress_events"] == len(progress) and set(stream) == {"progress_events", "first_progress_ms", "final_ms"}


# ------------------------------------------------------------ history stored before the boundary existed


def test_a_legacy_history_record_is_served_without_its_draft_text(server):
    base, handler = server(PipelineEngine(StreamingGenerator(SOUND)))
    store = handler.store
    cid = store.create_conversation()["id"]
    tid = uuid.uuid4().hex
    store.add_user_message(cid, tid, QUESTION)
    legacy = {"answer": "The generated answer could not be validated.", "raw_model_output": f"{CANARY} draft",
              "validation": {"repair": "extractive_replacement",
                             "initial": {"status": "unsupported", "codes": ["unsupported_claim"],
                                         "problems": [{"code": "unsupported_claim", "severity": "error", "detail": "0%",
                                                       "requirement_id": None, "citation": 1,
                                                       "sentence": f"Quarterly {CANARY} logbooks [1]."}]},
                             "final": {"status": "valid", "codes": [], "problems": []}},
              "generation": {"citations_repointed": [{"from": [2], "to": 1, "text": f"{CANARY} words"}]}}
    store.record_answer(cid, tid, "ok", legacy["answer"], legacy)
    history = call(base, f"/api/conversations/{cid}")[1]
    replay = call(base, "/api/ask/stream", {"question": QUESTION, "conversation_id": cid, "turn_id": tid})[1]
    for text in (history, replay):
        assert CANARY not in text
    served = json.loads(history)["conversation"]["messages"][-1]["result"]
    assert served["validation"]["initial"]["problems"] == [{"code": "unsupported_claim", "severity": "error",
                                                            "requirement_id": None, "citation": 1}]
    assert served["generation"]["citations_repointed"] == [{"from": [2], "to": 1}]


# ------------------------------------------------------------ browser_safe itself


def test_browser_safe_leaves_its_input_unchanged_and_passes_non_results_through():
    result = {"answer": "A [1].", "raw_model_output": "draft",
              "validation": {"initial": {"problems": [{"code": "x", "severity": "error", "sentence": "draft"}]},
                             "final": {"problems": [{"code": "y", "sentence": "A [1]."}]}}}
    before = json.dumps(result, sort_keys=True)
    safe = UI.browser_safe(result)
    assert json.dumps(result, sort_keys=True) == before
    assert "draft" not in json.dumps(safe) and safe["validation"]["final"] == result["validation"]["final"]
    assert UI.browser_safe(None) is None and UI.browser_safe({"answer": "a"}) == {"answer": "a"}
