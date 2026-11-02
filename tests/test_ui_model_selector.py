"""The application's one model (scripts/serve_ui.py): Qwen3-1.7B quantized to 4 bits, run by the MLX worker.

The REAL Handler, Engine.ask and Engine start-up run; only the worker, the tokenizer and the memory check are
scripted. Pinned:
  * the page and /api/config offer exactly one model, and every answer and measurement session names it;
  * a request naming any other model, or an adapter, is refused - never answered by this model instead;
  * start-up runs the MLX worker and nothing else: no Qwen3-0.6B, no adapter, no remote server;
  * every start-up failure (not configured, a directory that is not Qwen3-1.7B 4-bit, a different tokenizer,
    the memory preflight, the worker itself) stops the server with what to fix, and nothing is loaded in its place.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

import regulus.model as MODEL
from regulus.config import PROJECT_ROOT, load_settings
from regulus.rag import mlx_generator as MG
from tests.test_ui_measurement import CLEAN_Q, FAKE_MLX, UI, comparable, engine, ui  # noqa: F401  (fixture)


def ask(ui, **body):
    """/api/ask writes its record after the response is sent: wait for it, as SimpleUI.ask does."""
    import time

    n = ui._count()
    status, text = ui.post("/api/ask", {"question": CLEAN_Q, **body})
    deadline = time.monotonic() + 10
    while status == 200 and ui.H.measure is not None and ui._count() <= n and time.monotonic() < deadline:
        time.sleep(0.01)
    return status, json.loads(text)


# ------------------------------------------------------------------ one model, named everywhere

def test_the_page_has_no_model_control_and_the_api_offers_one_model(ui):
    html = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert "<select" not in html and '$("model")' not in html and "regulus.model" not in html
    with urllib.request.urlopen(ui.base + "/api/config", timeout=10) as r:
        cfg = json.loads(r.read())
    assert cfg["models"] == [{"id": UI.MODEL_MLX, "label": UI.MLX_LABEL, "available": True}]
    assert cfg["active_model"] == UI.MODEL_MLX and cfg["inference_backend"] == "mlx"
    assert (cfg["model_id"], cfg["model_revision"], cfg["dtype"]) == ("Qwen/Qwen3-1.7B", "rev-1.7b", "mlx-4bit-g128")


def test_a_request_naming_no_model_or_1_7b_is_answered_and_names_1_7b(ui):
    s0, plain = ask(ui)
    s1, named = ask(ui, model=UI.MODEL_MLX)
    assert s0 == s1 == 200 and comparable(plain) == comparable(named)
    assert plain["model"] == {"id": UI.MODEL_MLX, "label": UI.MLX_LABEL, "backend": "mlx",
                              "model_id": "Qwen/Qwen3-1.7B", "model_revision": "rev-1.7b", "dtype": "mlx-4bit-g128"}


def test_the_session_record_names_1_7b_with_its_tokenizer_and_decoding():
    run = UI.measurement_run(load_settings(), engine())
    m = run["model"]
    assert (m["backend"], m["model_id"], m["model_revision"]) == ("mlx", "Qwen/Qwen3-1.7B", "rev-1.7b")
    assert m["quantization"]["bits"] == 4 and m["adapter"] is None and m["decoding"] == {"do_sample": False}
    assert m["tokenizer"]["tokenizer_json_sha256"] == "fake-tokenizer-sha"


# ------------------------------------------------------------------ nothing else answers

@pytest.mark.parametrize("name", ["qwen3-0.6b", "Qwen/Qwen3-0.6B", "gpt-9"])
def test_a_request_naming_another_model_is_refused_and_nothing_is_generated(ui, name):
    calls = []
    real = ui.H.engine._mlx.generate
    ui.H.engine._mlx.generate = lambda *a, **k: calls.append(1) or real(*a, **k)
    n = ui._count()
    status, body = ask(ui, model=name)
    assert status >= 400 and UI.MODEL_MLX in body["error"]
    status, text = ui.post("/api/ask/stream", {"question": CLEAN_Q, "model": name})
    assert status == 400 and UI.MODEL_MLX in json.loads(text)["error"]
    with pytest.raises(ValueError, match="only"):
        ui.H.engine.ask(CLEAN_Q, model=name)
    assert calls == [] and ui._count() == n


def test_an_adapter_is_refused_rather_than_applied_to_another_model(ui):
    status, body = ask(ui, adapter="stage13_v1")
    assert status >= 400
    assert UI.Engine.available_adapters(ui.H.engine) == [{"id": "", "label": "No adapter"}]


def test_no_in_process_or_remote_model_path_is_left_in_the_server():
    source = (PROJECT_ROOT / "scripts" / "serve_ui.py").read_text(encoding="utf-8")
    for gone in ("QwenGenerator", "load_model", "load_adapter_model", "attach_saved_adapter", "GuardedRemoteGenerator",
                 "ensure_model", "MODEL_DEFAULT", "--backend"):
        assert gone not in source, gone


# ------------------------------------------------------------------ start-up: the worker, or a stop

class Worker:
    """Stands in for MlxGenerator's constructor and describe()."""
    started: list["Worker"] = []

    def __init__(self, tokenizer, gen, *, model_id, model_path, python, revision=None, worker_script=MG.WORKER_SCRIPT):
        self.model_id, self.model_path, self.revision = model_id, Path(model_path), revision
        self.worker_script = Path(worker_script)
        Worker.started.append(self)

    def describe(self):
        return {**FAKE_MLX, "model_id": self.model_id, "model_revision": self.revision}


def mlx_dir(root: Path, *, tokenizer='{"vocab": 1}', **config) -> Path:
    snap = root / "models--Qwen--Qwen3-1.7B-MLX-4bit" / "snapshots" / "abc123"
    snap.mkdir(parents=True, exist_ok=True)
    cfg = {**UI.MLX_EXPECTED_CONFIG, "quantization": {"group_size": 128, "bits": 4}, **config}
    (snap / "config.json").write_text(json.dumps(cfg))
    (snap / "tokenizer.json").write_text(tokenizer)
    return snap


@pytest.fixture
def start(tmp_path, monkeypatch):
    """Engine start-up with the worker, tokenizer and memory check scripted. start(**model fields) -> Engine."""
    Worker.started = []
    for name in ("MODEL_MLX", "MLX_MODEL_ID", "MLX_LABEL", "MLX_EXPECTED_CONFIG"):
        monkeypatch.setattr(UI, name, getattr(UI, name))   # start-up fixes the module's one model; restored after each test
    monkeypatch.setattr(MODEL, "load_model", lambda *a, **k: pytest.fail("a transformers model was loaded"))
    monkeypatch.setattr(MODEL, "load_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(MG, "memory_preflight", lambda mb: {"checked": True})
    monkeypatch.setattr(MG, "MlxGenerator", Worker)
    python = tmp_path / "python"
    python.write_text("")
    mine = tmp_path / "loaded_tokenizer.json"
    mine.write_text('{"vocab": 1}')

    def run(**model):
        e = object.__new__(UI.Engine)
        s = load_settings(use_dotenv=False)
        fields = {"backend": "transformers", "adapter": None, "mlx_python": python, "mlx_revision": None, **model}
        fields.setdefault("mlx_model_path", mlx_dir(tmp_path / "mlx"))
        s.model = replace(s.model, **fields)
        e.settings = s
        e._tokenizer_file = lambda: mine
        UI.Engine._init_inference(e, offline=True)
        return e

    run.tmp = tmp_path
    run.tokenizer_sha = hashlib.sha256(mine.read_bytes()).hexdigest()
    return run


@pytest.mark.parametrize("backend", ["transformers", "mlx"])
def test_start_up_runs_the_default_4b_worker_and_nothing_else(start, backend):
    # Qwen3-4B-Instruct-2507 is the default since 23 Oct 2026 (the model the cosmetics benchmark is measured on)
    e = start(backend=backend)
    assert Worker.started == [e._mlx] and e._mlx.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert e._mlx.revision == "abc123"                         # the HF snapshot directory's commit
    ident = e.model_identity()
    assert (ident["backend"], ident["model_id"], ident["model_revision"]) == ("mlx", "Qwen/Qwen3-4B-Instruct-2507", "abc123")
    assert ident["tokenizer"]["tokenizer_json_sha256"] == start.tokenizer_sha


def test_the_default_runtime_starts_the_mlx_worker(start):
    e = start(backend="mlx")
    assert e._mlx.worker_script == MG.WORKER_SCRIPT


def test_the_torch_runtime_starts_the_same_model_in_the_torch_worker(start):
    # Windows / Linux: the same model directory and the same checks, another worker script
    e = start(backend="mlx", worker_runtime="torch")
    assert Worker.started == [e._mlx] and e._mlx.worker_script == MG.TORCH_WORKER_SCRIPT
    assert e._mlx.model_id == "Qwen/Qwen3-4B-Instruct-2507" and e._mlx.revision == "abc123"


def test_the_mlx_identity_has_no_runtime_key_and_the_torch_identity_names_it(start, monkeypatch):
    # records and the reply cache carry the identity: the MLX one must keep exactly its former keys
    e = start(backend="mlx")
    assert "runtime" not in e.model_identity() and list(e.model_identity())[:3] == ["backend", "model_id", "model_revision"]
    monkeypatch.setattr(Worker, "describe", lambda self: {**FAKE_MLX, "runtime": "torch", "device": "cuda (GPU)"})
    ident = e.model_identity()
    assert ident["runtime"] == "torch" and ident["device"] == "cuda (GPU)"


def test_a_failed_mlx_start_off_a_mac_names_the_torch_runtime(start, monkeypatch):
    monkeypatch.setattr(UI.sys, "platform", "win32")

    def no_worker(*a, **k):
        raise MG.MlxWorkerError("the MLX worker exited (exit code 1)")

    monkeypatch.setattr(MG, "MlxGenerator", no_worker)
    with pytest.raises(SystemExit, match="REGULUS_WORKER_RUNTIME=torch"):
        start(backend="mlx")
    monkeypatch.setattr(UI.sys, "platform", "darwin")
    with pytest.raises(SystemExit) as exc:
        start(backend="mlx")
    assert "REGULUS_WORKER_RUNTIME" not in str(exc.value)


def test_a_1_7b_directory_still_starts_the_1_7b_worker(start):
    from regulus.rag.mlx_models import SUPPORTED_MLX_MODELS

    cfg = SUPPORTED_MLX_MODELS["qwen3-1.7b"]["config"]
    e = start(mlx_model_path=mlx_dir(start.tmp / "q17", **cfg, max_position_embeddings=40960, rope_theta=1000000))
    assert Worker.started == [e._mlx] and e._mlx.model_id == "Qwen/Qwen3-1.7B"


def test_an_explicit_revision_is_recorded(start):
    assert start(mlx_revision="pinned-rev")._mlx.revision == "pinned-rev"


@pytest.mark.parametrize("case, fields, message", [
    ("remote server", {"backend": "openai_compatible"}, "openai_compatible"),
    ("adapter", {"adapter": Path("models/qwen3-0.6b-lora/stage13_v1/adapter")}, "REGULUS_ADAPTER"),
    ("not configured", {"mlx_python": None}, "REGULUS_MLX_PYTHON"),
    ("interpreter missing", {"mlx_python": Path("/nonexistent/python")}, "REGULUS_MLX_PYTHON"),
])
def test_a_configuration_that_would_answer_with_another_model_stops_start_up(start, case, fields, message):
    with pytest.raises(SystemExit, match=message) as stop:
        start(**fields)
    assert "never substitutes another" in str(stop.value) and Worker.started == []


@pytest.mark.parametrize("config, message", [
    ({"hidden_size": 1024, "intermediate_size": 3072}, "not Qwen3-1.7B or Qwen3-4B-Instruct-2507"),   # Qwen3-0.6B
    ({"max_position_embeddings": 40960, "rope_theta": 1000000}, "not Qwen3-1.7B or Qwen3-4B-Instruct-2507"),  # Qwen3-4B, not -2507
    ({"quantization": {"group_size": 64, "bits": 8}}, "quantization bits 8"),
])
def test_a_model_directory_that_is_not_a_supported_4bit_model_stops_start_up(start, config, message):
    with pytest.raises(SystemExit, match=message):
        start(mlx_model_path=mlx_dir(start.tmp / "other", **config))
    assert Worker.started == []


def test_a_different_tokenizer_stops_start_up(start):
    with pytest.raises(SystemExit, match="byte-identical"):
        start(mlx_model_path=mlx_dir(start.tmp / "other", tokenizer='{"vocab": 2}'))
    assert Worker.started == []


def test_a_refused_memory_preflight_stops_start_up_without_a_fallback(start, monkeypatch):
    def refuse(mb):
        raise MG.MemoryPreflightError("memory preflight: 900 MB available")

    monkeypatch.setattr(MG, "memory_preflight", refuse)
    with pytest.raises(SystemExit, match="MemoryPreflightError: memory preflight: 900 MB"):
        start()
    assert Worker.started == []


def test_a_worker_that_cannot_start_stops_start_up_without_a_fallback(start, monkeypatch):
    def broken(*a, **k):
        raise MG.MlxWorkerError("the MLX worker exited (exit code 1)")

    monkeypatch.setattr(MG, "MlxGenerator", broken)
    with pytest.raises(SystemExit, match="MlxWorkerError: the MLX worker exited"):
        start()


def test_qwen3_4b_instruct_2507_is_identified_and_the_hybrid_4b_is_not():
    """22 Oct: the 2507 release has the hybrid Qwen3-4B's widths; its context and rope base tell them apart."""
    import importlib.util, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("serve_ui_ident", root / "scripts" / "serve_ui.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["serve_ui_ident"] = mod; spec.loader.exec_module(mod)
    base = {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36, "intermediate_size": 9728}
    assert mod.identify_mlx_model({**base, "max_position_embeddings": 262144, "rope_theta": 5000000}) == "qwen3-4b-2507"
    assert mod.identify_mlx_model({**base, "max_position_embeddings": 40960, "rope_theta": 1000000}) is None
    assert mod.identify_mlx_model({"model_type": "qwen3", "hidden_size": 2048, "num_hidden_layers": 28,
                                   "intermediate_size": 6144}) == "qwen3-1.7b"
