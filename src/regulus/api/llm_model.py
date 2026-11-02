"""llm-model: the Regulus answering service that regulus-backend calls (27 Oct 2026).

Routes (prefix /llm-model/v1):
    POST /answer          {"question", "conversation"?, "focus"?} -> the validated answer, its state and citations
    POST /answer/stream   the same, as newline-delimited JSON events: start, progress (no text), then final or error
    GET  /health          public: the process is up
    GET  /ready           public: the model and the database answer
    GET  /info            the model, the release (code revision, switches) and the database snapshot

It follows regulus-backend's conventions (Regulus-21-API-doc.md): `Authorization: Bearer <key>`, errors as
{"code", "message", "fieldErrors"}, 400/415 for a malformed body, 401 without a valid key, 413 over the body limit,
422 for a schema failure and for an out-of-scope question, 429 with Retry-After when a key is over its rate,
503 when the model is busy past the queue wait or a dependency is down, and X-Idempotency-Key on the answer routes.

The service is stateless: a follow-up carries its earlier turns, each with the `conversation.anchor` this service
returned for it, so any instance answers it the same way. Answers are the pipeline's validated text only: the model's
draft never leaves the process (as in scripts/serve_ui.py). The engine is the UI's (scripts/serve_ui.py Engine), so the
API answers exactly as the measured application does; scripts/serve_llm_model.py builds it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable

SERVICE = "llm-model"
API_VERSION = "v1"
PREFIX = f"/{SERVICE}/{API_VERSION}"
MAX_QUESTION_CHARS = 2000
MAX_TURNS = 10
FOCUS_VALUES = ("cosmetics", "general")
PROGRESS_INTERVAL_S = 0.5

# answer states, as regulus-backend names them
VERIFIED, PARTIAL, NO_SOURCE, NOT_COVERED = "VERIFIED", "PARTIAL", "NO_SOURCE", "NOT_COVERED"
NONEXISTENT_SECTION, SECTION_REMOVED, ISO_POINTER = "NONEXISTENT_SECTION", "SECTION_REMOVED", "ISO_POINTER"
GATE_FAILURE, OUT_OF_SCOPE = "GATE_FAILURE", "OUT_OF_SCOPE"
# the pipeline's fallback types (regulus.rag.fallback) -> state
_STATE_BY_FALLBACK = {
    "out_of_scope": OUT_OF_SCOPE,
    "provision_not_found": NONEXISTENT_SECTION, "wrong_provision": NONEXISTENT_SECTION,
    "no_evidence": NO_SOURCE,
    "unvalidated_answer_withheld": GATE_FAILURE,
    "value_not_established": NOT_COVERED, "premise_not_established": NOT_COVERED, "ambiguous": NOT_COVERED,
    "product_clarification": NOT_COVERED,   # asks which product type, as "ambiguous" asks which provision
    "unsupported_request": NOT_COVERED,
    "focused_provision_quote": PARTIAL, "definitions_quoted": PARTIAL,
}
_EVIDENCE_LIMIT = "Evidence limit:"


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, field_errors: dict | None = None,
                 headers: dict | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.field_errors = field_errors or {}
        self.headers = headers or {}

    def body(self) -> dict:
        return {"code": self.code, "message": self.message, "fieldErrors": self.field_errors}


@dataclass(frozen=True)
class ApiConfig:
    keys: tuple[tuple[str, str], ...]          # (key id, secret); the id is logged, the secret never
    rate_per_min: int = 30
    max_body_bytes: int = 64 * 1024
    queue_wait_s: float = 300.0
    idempotency_ttl_s: float = 24 * 3600.0
    idempotency_max: int = 2000

    @classmethod
    def from_env(cls, env: dict | None = None) -> "ApiConfig":
        """LLM_MODEL_API_KEYS: comma-separated `id:secret` pairs (or bare secrets, identified by a hash prefix);
        LLM_MODEL_RATE_PER_MIN, LLM_MODEL_MAX_BODY_BYTES, LLM_MODEL_QUEUE_WAIT_S. No key: the service refuses to start."""
        env = os.environ if env is None else env
        keys = []
        for item in (env.get("LLM_MODEL_API_KEYS") or "").split(","):
            item = item.strip()
            if not item:
                continue
            kid, sep, secret = item.partition(":")
            if not sep:
                kid, secret = "key-" + hashlib.sha256(item.encode()).hexdigest()[:8], item
            if len(secret) < 24:
                raise ValueError(f"API key {kid!r} is shorter than 24 characters")
            keys.append((kid.strip(), secret.strip()))
        if not keys:
            raise ValueError("LLM_MODEL_API_KEYS is not set: the service never answers without authentication")
        return cls(keys=tuple(keys), rate_per_min=int(env.get("LLM_MODEL_RATE_PER_MIN") or 30),
                   max_body_bytes=int(env.get("LLM_MODEL_MAX_BODY_BYTES") or 64 * 1024),
                   queue_wait_s=float(env.get("LLM_MODEL_QUEUE_WAIT_S") or 300))

    def key_id(self, authorization: str | None) -> str | None:
        """The id of the key the Authorization header carries, compared in constant time against every key."""
        m = re.match(r"Bearer\s+(\S+)\s*$", authorization or "")
        token = (m.group(1) if m else "").encode()
        found = None
        for kid, secret in self.keys:      # every key is compared: the time does not depend on which one matches
            if hmac.compare_digest(token, secret.encode()) and m:
                found = kid
        return found


# ---- requests -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerRequest:
    question: str
    turns: tuple[tuple[str, dict | None], ...] = ()
    focus: str | None = None


def parse_request(body: Any) -> AnswerRequest:
    """The request, or ApiError 422 naming every field that is wrong."""
    errors: dict[str, str] = {}
    if not isinstance(body, dict):
        raise ApiError(422, "validation_error", "the body must be a JSON object")
    for k in body:
        if k not in ("question", "conversation", "focus"):
            errors[k] = "unknown field"
    q = body.get("question")
    if not isinstance(q, str) or not q.strip():
        errors["question"] = "required: a non-empty string"
    elif len(q) > MAX_QUESTION_CHARS:
        errors["question"] = f"at most {MAX_QUESTION_CHARS} characters"
    focus = body.get("focus")
    if focus is not None and focus not in FOCUS_VALUES:
        errors["focus"] = f"one of {', '.join(FOCUS_VALUES)}, or omitted"
    turns: list[tuple[str, dict | None]] = []
    conv = body.get("conversation")
    if conv is not None:
        if not isinstance(conv, list) or len(conv) > MAX_TURNS:
            errors["conversation"] = f"a list of at most {MAX_TURNS} earlier turns, oldest first"
        else:
            for i, t in enumerate(conv):
                tq = t.get("question") if isinstance(t, dict) else None
                anchor = t.get("anchor") if isinstance(t, dict) else None
                if not isinstance(tq, str) or not tq.strip() or len(tq) > MAX_QUESTION_CHARS:
                    errors[f"conversation[{i}].question"] = f"required: a non-empty string of at most {MAX_QUESTION_CHARS} characters"
                elif anchor is not None and not _valid_anchor(anchor):
                    errors[f"conversation[{i}].anchor"] = "the conversation.anchor object a previous answer returned"
                else:
                    turns.append((tq.strip(), anchor))
    if errors:
        raise ApiError(422, "validation_error", "the request does not match the schema", errors)
    return AnswerRequest(question=q.strip(), turns=tuple(turns), focus=focus)


def _valid_anchor(anchor: Any) -> bool:
    if not isinstance(anchor, dict) or set(anchor) - {"cfr", "statute", "parts"}:
        return False
    return all(isinstance(v, list) and len(v) <= 8 and all(isinstance(x, str) and re.fullmatch(r"[0-9A-Za-z.()\-]{1,24}", x)
                                                           for x in v) for v in anchor.values())


def conversation_context(turns):
    """The earlier turns as the pipeline's ConversationContext, or None for a first question."""
    if not turns:
        return None
    from regulus.rag.conversation import ConversationContext

    return ConversationContext.from_records(turns) or None


# ---- responses ----------------------------------------------------------------------------------------------------


def answer_state(result: dict) -> str:
    fb = result.get("fallback_type")
    answer = result.get("answer") or ""
    if fb == "composed_qmsr":
        from regulus.rag.qmsr import NO_CLAUSE_SENTENCE, REMOVED_SENTENCE

        return SECTION_REMOVED if REMOVED_SENTENCE in answer else (NONEXISTENT_SECTION if NO_CLAUSE_SENTENCE in answer
                                                                   else ISO_POINTER)
    if fb in _STATE_BY_FALLBACK:
        return _STATE_BY_FALLBACK[fb]
    if fb is not None and not (fb.startswith("composed_") or fb == "verbatim_requested_provision"):
        return PARTIAL     # a fallback this service does not know is never reported as verified
    return PARTIAL if _EVIDENCE_LIMIT in answer else VERIFIED


def response_body(result: dict, request_id: str, release: dict) -> dict:
    review = ((result.get("accuracy") or {}).get("question_review") or {})
    followup = result.get("followup") or {}
    timings = result.get("server_timings_ms") or {}
    return {
        "id": request_id,
        "state": answer_state(result),
        "answer": result.get("answer") or "",
        "citations": [{"index": c.get("index"), "label": c.get("label"), "url": c.get("url"), "corpus": c.get("corpus"),
                       "citedInAnswer": bool(c.get("cited_in_answer"))} for c in result.get("citations") or ()],
        "conversation": {"anchor": followup.get("anchor") or {"cfr": [], "statute": [], "parts": []}},
        "evaluation": {"status": review.get("status"), "score": review.get("score")} if review else None,
        "model": {k: (result.get("model") or {}).get(k) for k in ("id", "label", "model_revision")},
        "release": release,
        "snapshotDate": result.get("snapshot_date"),
        "focus": result.get("focus"),
        "timingsMs": {"pipeline": timings.get("pipeline_answer_ms"), "total": result.get("total_ms")},
    }


# ---- limits -------------------------------------------------------------------------------------------------------


class RateLimiter:
    """At most `per_min` answer requests per key in any 60 s window."""

    def __init__(self, per_min: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_min, self.clock = per_min, clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, key_id: str) -> None:
        now = self.clock()
        with self._lock:
            hits = self._hits.setdefault(key_id, deque())
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self.per_min:
                retry = max(1, int(60 - (now - hits[0])) + 1)
                raise ApiError(429, "rate_limited", f"more than {self.per_min} answer requests a minute",
                               headers={"Retry-After": str(retry)})
            hits.append(now)


@dataclass
class _Entry:
    body_sha: str
    expires: float
    status: int | None = None
    payload: dict | None = None
    done: threading.Event = field(default_factory=threading.Event)


class IdempotencyCache:
    """X-Idempotency-Key: a repeat with the same key and body returns the first result instead of answering again; the
    same key with a different body is 409. Per API key, kept `ttl` seconds, at most `limit` entries."""

    def __init__(self, ttl: float, limit: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl, self.limit, self.clock = ttl, limit, clock
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._lock = threading.Lock()

    def begin(self, key_id: str, idem: str, body_sha: str) -> tuple[_Entry, bool]:
        """(entry, True) when this request must answer; (entry, False) when an earlier one did or is doing so."""
        now = self.clock()
        with self._lock:
            for k in [k for k, e in self._entries.items() if e.expires <= now]:
                del self._entries[k]
            e = self._entries.get((key_id, idem))
            if e is not None:
                if e.body_sha != body_sha:
                    raise ApiError(409, "idempotency_key_reused", "this X-Idempotency-Key was used with a different body")
                return e, False
            if len(self._entries) >= self.limit:
                oldest = min(self._entries, key=lambda k: self._entries[k].expires)
                del self._entries[oldest]
            e = _Entry(body_sha=body_sha, expires=now + self.ttl)
            self._entries[(key_id, idem)] = e
            return e, True

    def finish(self, key_id: str, idem: str, entry: _Entry, status: int, payload: dict) -> None:
        if status >= 500 or status == 429:     # a failure is not remembered: the client may retry with the same key
            with self._lock:
                self._entries.pop((key_id, idem), None)
        entry.status, entry.payload = status, payload
        entry.done.set()


# ---- the service --------------------------------------------------------------------------------------------------


class Service:
    """Answers through `engine` (scripts/serve_ui.py Engine or a test double), one question at a time."""

    def __init__(self, engine, config: ApiConfig, release: dict, *, log=print) -> None:
        self.engine, self.config, self.release, self.log = engine, config, release, log
        self.rate = RateLimiter(config.rate_per_min)
        self.idem = IdempotencyCache(config.idempotency_ttl_s, config.idempotency_max)
        self._model = threading.Lock()

    def answer(self, req: AnswerRequest, request_id: str, on_event=None) -> tuple[int, dict]:
        t0 = time.perf_counter()
        if not self._model.acquire(timeout=self.config.queue_wait_s):
            raise ApiError(503, "model_busy", "the model is answering other questions; retry later",
                           headers={"Retry-After": "30"})
        try:
            result = self.engine.ask(req.question, context=conversation_context(req.turns), focus=req.focus,
                                     **({"on_event": on_event} if on_event is not None else {}))
        finally:
            self._model.release()
        result = dict(result)
        result["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        body = response_body(result, request_id, self.release)
        if body["state"] == OUT_OF_SCOPE:
            return 422, {"code": "out_of_scope", "message": body["answer"], "fieldErrors": {}, **body}
        return 200, body

    def ready(self) -> tuple[bool, str]:
        try:
            self.engine.db.query("SELECT 1 AS ok")
        except Exception as exc:  # noqa: BLE001
            return False, f"database: {type(exc).__name__}"
        return True, "ready"

    def info(self) -> dict:
        ident = {}
        try:
            ident = self.engine.model_identity() or {}
        except Exception:  # noqa: BLE001 - informational
            pass
        return {"service": SERVICE, "apiVersion": API_VERSION, "release": self.release,
                "model": {k: ident.get(k) for k in ("model_id", "model_revision", "dtype", "backend")},
                "database": {"snapshotDate": getattr(self.engine, "snapshot", None)},
                "limits": {"ratePerMinute": self.config.rate_per_min, "maxBodyBytes": self.config.max_body_bytes,
                           "maxQuestionChars": MAX_QUESTION_CHARS, "maxTurns": MAX_TURNS}}


def release_info() -> dict:
    """The code revision and the switches this process answers with."""
    from regulus.config import PROJECT_ROOT
    from regulus.rag.validator_switches import VALIDATOR_PRECISION

    try:
        from regulus.traceability.ui_session import code_revision

        revision = code_revision(PROJECT_ROOT)
    except Exception:  # noqa: BLE001
        revision = None
    return {"revision": revision, "switches": sorted(VALIDATOR_PRECISION)}


class ClientGone(Exception):
    """The streaming client disconnected; raised out of the generation to stop it."""


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "llm-model/1"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):   # one line per request from _done, never the question
            pass

        # -- plumbing
        def _send_json(self, status: int, payload: dict, headers: dict | None = None) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Request-Id", self.request_id)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
            self._status = status

        def _error(self, err: ApiError) -> None:
            headers = dict(err.headers)
            if err.status == 401:
                headers["WWW-Authenticate"] = 'Bearer realm="llm-model"'
            # a body that was refused unread must not be parsed as the next request on this connection
            headers["Connection"] = "close"
            self.close_connection = True
            # ... and is read and dropped first (at most 1 MiB): Windows resets a connection closed with unread input,
            # and the client then never sees the error
            if not getattr(self, "_body_read", False):
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                if 0 < length <= 1024 * 1024:
                    self.rfile.read(length)
            self._send_json(err.status, err.body(), headers)

        def _begin(self) -> None:
            self.t0 = time.perf_counter()
            rid = (self.headers.get("X-Request-Id") or "").strip()
            self.request_id = rid if re.fullmatch(r"[A-Za-z0-9._\-]{8,64}", rid) else str(uuid.uuid4())
            self._status = None
            self.key = None
            self._body_read = False

        def _done(self) -> None:
            service.log(f"{self.command} {self.path.split('?')[0]} {self._status} "
                        f"{(time.perf_counter() - self.t0) * 1000:.0f}ms key={self.key or '-'} id={self.request_id}")

        def _auth(self) -> str:
            kid = service.config.key_id(self.headers.get("Authorization"))
            if kid is None:
                raise ApiError(401, "unauthorized", "a valid Authorization: Bearer key is required")
            self.key = kid
            return kid

        def _body(self) -> tuple[Any, bytes]:
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                raise ApiError(415, "unsupported_media_type", "Content-Type must be application/json")
            try:
                length = int(self.headers.get("Content-Length") or "")
            except ValueError:
                raise ApiError(411 if not self.headers.get("Content-Length") else 400, "length_required",
                               "a Content-Length header is required") from None
            if length > service.config.max_body_bytes:
                raise ApiError(413, "payload_too_large", f"the body is over {service.config.max_body_bytes} bytes")
            raw = self.rfile.read(length) if length > 0 else b""
            self._body_read = True
            try:
                return json.loads(raw.decode("utf-8") or "null"), raw
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ApiError(400, "invalid_json", "the body is not valid JSON") from None

        # -- routes
        def do_GET(self) -> None:
            self._begin()
            try:
                path = self.path.split("?")[0]
                if path == f"{PREFIX}/health":
                    self._send_json(200, {"status": "ok", "service": SERVICE})
                elif path == f"{PREFIX}/ready":
                    ok, why = service.ready()
                    self._send_json(200 if ok else 503, {"status": "ready" if ok else "not_ready", "detail": why})
                elif path == f"{PREFIX}/info":
                    self._auth()
                    self._send_json(200, service.info())
                else:
                    raise ApiError(404, "not_found", "no such route")
            except ApiError as err:
                self._error(err)
            finally:
                self._done()

        def do_POST(self) -> None:
            self._begin()
            try:
                path = self.path.split("?")[0]
                if path not in (f"{PREFIX}/answer", f"{PREFIX}/answer/stream"):
                    raise ApiError(404, "not_found", "no such route")
                kid = self._auth()
                body, raw = self._body()
                req = parse_request(body)
                idem = (self.headers.get("X-Idempotency-Key") or "").strip()
                if idem and not re.fullmatch(r"[A-Za-z0-9._:\-]{8,128}", idem):
                    raise ApiError(422, "validation_error", "invalid X-Idempotency-Key",
                                   {"X-Idempotency-Key": "8-128 characters: letters, digits, . _ : -"})
                entry = None
                if idem:
                    entry, mine = service.idem.begin(kid, idem, hashlib.sha256(path.encode() + b"\0" + raw).hexdigest())
                    if not mine:
                        if not entry.done.wait(timeout=service.config.queue_wait_s + 600):
                            raise ApiError(409, "idempotent_request_in_progress", "the first request with this key is still running")
                        if path.endswith("/stream"):
                            self._stream_replay(entry.status, entry.payload)
                        else:
                            self._send_json(entry.status, entry.payload, {"Idempotent-Replayed": "true"})
                        return
                service.rate.check(kid)
                if path.endswith("/stream"):
                    self._stream(req, kid, idem, entry)
                else:
                    try:
                        status, payload = service.answer(req, self.request_id)
                    except ApiError as err:
                        if entry is not None:
                            service.idem.finish(kid, idem, entry, err.status, err.body())
                        raise
                    except Exception as exc:  # noqa: BLE001 - never a stack trace or draft text to the caller
                        service.log(f"answer failed: {type(exc).__name__}")
                        err = ApiError(500, "internal_error", f"{type(exc).__name__} while answering")
                        if entry is not None:
                            service.idem.finish(kid, idem, entry, 500, err.body())
                        raise err from None
                    if entry is not None:
                        service.idem.finish(kid, idem, entry, status, payload)
                    self._send_json(status, payload)
            except ApiError as err:
                self._error(err)
            finally:
                self._done()

        # -- streaming: NDJSON, no draft text
        def _stream_headers(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("X-Request-Id", self.request_id)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self._status = 200

        def _write(self, event: dict) -> None:
            try:
                self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
                raise ClientGone() from exc

        def _stream_replay(self, status: int, payload: dict) -> None:
            self._stream_headers()
            self._write({"type": "start", "id": self.request_id})
            self._write({"type": "final", "status": status, "result": payload} if status < 400 or status == 422
                        else {"type": "error", "status": status, "error": payload})

        def _stream(self, req: AnswerRequest, kid: str, idem: str, entry) -> None:
            self._stream_headers()
            t0 = time.perf_counter()
            last = {"t": None}

            def on_event(kind: str, data: dict) -> None:
                if kind != "delta":
                    return
                now = time.perf_counter()
                if last["t"] is not None and now - last["t"] < PROGRESS_INTERVAL_S:
                    return
                last["t"] = now
                self._write({"type": "progress", "elapsedMs": round((now - t0) * 1000)})

            try:
                self._write({"type": "start", "id": self.request_id})
                status, payload = service.answer(req, self.request_id, on_event=on_event)
                if entry is not None:
                    service.idem.finish(kid, idem, entry, status, payload)
                self._write({"type": "final", "status": status, "result": payload})
            except ClientGone:
                if entry is not None:
                    service.idem.finish(kid, idem, entry, 503, {"code": "client_disconnected", "message": "", "fieldErrors": {}})
                service.log(f"stream {self.request_id}: client disconnected; generation stopped")
            except ApiError as err:
                if entry is not None:
                    service.idem.finish(kid, idem, entry, err.status, err.body())
                self._safe_write({"type": "error", "status": err.status, "error": err.body()})
            except Exception as exc:  # noqa: BLE001
                service.log(f"stream {self.request_id} failed: {type(exc).__name__}")
                if entry is not None:
                    service.idem.finish(kid, idem, entry, 500, {"code": "internal_error", "message": "", "fieldErrors": {}})
                self._safe_write({"type": "error", "status": 500, "error": {
                    "code": "internal_error", "message": f"{type(exc).__name__} while answering", "fieldErrors": {}}})

        def _safe_write(self, event: dict) -> None:
            try:
                self._write(event)
            except ClientGone:
                pass

    return Handler
