"""scripts/torch_worker.py: the MLX worker's protocol and MLX 4-bit weights, run by PyTorch (Windows / Linux).

A tiny random Qwen3 is written in the mlx-community 4-bit layout (uint32 words, bfloat16 scales and biases, group 64),
and the worker - started as a real subprocess on the CPU and spoken to over its JSON-lines protocol - must generate
exactly the greedy tokens of a transformers Qwen3ForCausalLM holding the same dequantized weights. Also pinned: the
unpacking order matches MLX's (lowest bits first), the repetition penalty is mlx-lm's, and the chunk / stop / eos
rules of scripts/mlx_worker.py.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from regulus.config import PROJECT_ROOT  # noqa: E402

WORKER = PROJECT_ROOT / "scripts" / "torch_worker.py"
CONFIG = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "vocab_size": 256, "hidden_size": 128,
          "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
          "head_dim": 32, "max_position_embeddings": 512, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
          "tie_word_embeddings": True, "attention_bias": False, "hidden_act": "silu", "bos_token_id": 1,
          "eos_token_id": 2, "quantization": {"group_size": 64, "bits": 4}}


def load_worker_module():
    spec = importlib.util.spec_from_file_location("torch_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mlx_quantize(w: torch.Tensor, group: int = 64):
    """MLX's affine 4-bit layout: per group w ~ q * scale + bias, 8 values per uint32, lowest bits first."""
    rows, cols = w.shape
    g = w.float().reshape(rows, cols // group, group)
    lo, hi = g.min(-1).values, g.max(-1).values
    scale = ((hi - lo) / 15).clamp(min=1e-8).to(torch.bfloat16)
    bias = lo.to(torch.bfloat16)
    q = ((g - bias.float().unsqueeze(-1)) / scale.float().unsqueeze(-1)).round().clamp(0, 15).to(torch.int64)
    q = q.reshape(rows, cols // 8, 8)
    words = sum(q[..., j] << (4 * j) for j in range(8))
    return words.to(torch.int64).to(torch.uint32), scale, bias, q.reshape(rows, cols)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """(model directory, reference transformers model with the dequantized weights)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(**{k: v for k, v in CONFIG.items() if k != "quantization"})
    ref = Qwen3ForCausalLM(cfg).float().eval()
    tensors, state = {}, ref.state_dict()
    for name, value in state.items():
        if name == "lm_head.weight":
            continue                                    # tied to the embeddings
        if value.dim() == 2:
            prefix = name[: -len(".weight")]
            words, scale, bias, q = mlx_quantize(value)
            tensors[f"{prefix}.weight"], tensors[f"{prefix}.scales"], tensors[f"{prefix}.biases"] = words, scale, bias
            deq = (q.reshape(q.shape[0], -1, 64).float() * scale.float().unsqueeze(-1) + bias.float().unsqueeze(-1))
            state[name] = deq.reshape(value.shape)
        else:
            tensors[name] = value.to(torch.bfloat16)
            state[name] = value.to(torch.bfloat16).float()
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    ref.load_state_dict(state)
    d = tmp_path_factory.mktemp("mlx4bit")
    save_file(tensors, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps(CONFIG))
    return d, ref


class Worker:
    def __init__(self, model_dir: Path):
        env = dict(os.environ, REGULUS_TORCH_DEVICE="cpu", REGULUS_TORCH_DTYPE="float32", PYTHONDONTWRITEBYTECODE="1")
        self.proc = subprocess.Popen([sys.executable, "-B", str(WORKER), str(model_dir)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.ready = self.read()

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, f"the worker exited ({self.proc.poll()})"
        return json.loads(line)

    def send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def generate(self, prompt, max_new, *, stop_after_chunks=None, **extra):
        self.send({"op": "generate", "prompt_ids": prompt, "max_new_tokens": max_new, "eos_ids": [2],
                   "check_every": 4, **extra})
        ids, chunks = [], 0
        while True:
            msg = self.read()
            assert msg["op"] in ("chunk", "done"), msg
            ids += msg["token_ids"]
            if msg["op"] == "done":
                return ids, msg, chunks
            chunks += 1
            stop = stop_after_chunks is not None and chunks >= stop_after_chunks
            self.send({"op": "stop", "reason": "test"} if stop else {"op": "continue"})

    def close(self):
        self.send({"op": "shutdown"})
        assert self.read()["op"] == "bye"
        self.proc.wait(timeout=30)


@pytest.fixture(scope="module")
def worker(tiny):
    w = Worker(tiny[0])
    yield w
    w.close()


def reference_greedy(ref, prompt, n, penalty=None, context=64, eos=2):
    ids = list(prompt)
    out = []
    with torch.no_grad():
        for _ in range(n):
            logits = ref(torch.tensor([ids])).logits[:, -1, :].float()
            if penalty is not None:
                ctx = torch.tensor(ids[-context:])
                sel = logits[:, ctx]
                logits[:, ctx] = torch.where(sel < 0, sel * penalty, sel / penalty)
            tok = int(logits.argmax(-1)[0])
            out.append(tok)
            ids.append(tok)
            if tok == eos:
                break
    return out


def test_unpacking_follows_mlx_order():
    W = load_worker_module()
    w = torch.randn(4, 128)
    words, scale, bias, q = mlx_quantize(w)
    deq = W.dequantize(words.view(torch.uint8), scale, bias, 4, 64, torch.float32)
    expected = (q.reshape(4, -1, 64).float() * scale.float().unsqueeze(-1) + bias.float().unsqueeze(-1)).reshape(4, 128)
    assert torch.equal(deq, expected)


def test_the_repetition_penalty_is_mlx_lms():
    W = load_worker_module()
    logits = torch.tensor([[2.0, -2.0, 1.0, 0.5]])
    out = W.repetition_penalty(logits.clone(), [0, 1, 1], 2.0)   # a repeated id is penalised once
    assert out.tolist() == [[1.0, -4.0, 1.0, 0.5]]


def test_the_worker_announces_what_it_runs(worker):
    r = worker.ready
    assert r["op"] == "ready" and r["runtime"] == "torch" and r["device"].startswith("cpu") and r["dtype"] == "float32"
    assert r["weights"].startswith("mlx-4bit-g64") and set(r["versions"]) == {"torch", "transformers", "safetensors"}


def test_greedy_tokens_match_the_dequantized_reference(worker, tiny):
    prompt = [5, 17, 99, 42, 7, 3, 250, 128]
    ids, done, _ = worker.generate(prompt, 12)
    assert ids == reference_greedy(tiny[1], prompt, 12)
    assert done["prompt_tokens"] == len(prompt) and done["generated_tokens"] == len(ids)


def test_the_repetition_penalty_request_matches_the_reference(worker, tiny):
    prompt = [9, 9, 9, 10, 11, 9]
    ids, _, _ = worker.generate(prompt, 10, repetition_penalty=1.3, repetition_context_size=64)
    assert ids == reference_greedy(tiny[1], prompt, 10, penalty=1.3)


def test_the_caller_can_stop_at_a_chunk(worker):
    ids, done, chunks = worker.generate([5, 6, 7], 40, stop_after_chunks=1)
    assert chunks == 1 and done["stopped_by_caller"] == "test" and len(ids) == 4


def test_a_longer_prompt_matches_the_reference(worker, tiny):
    prompt = [int(i) % 250 + 3 for i in range(300)]
    ids, _, _ = worker.generate(prompt, 5)
    assert ids == reference_greedy(tiny[1], prompt, 5)


# ------------------------------------------------------------------ the MLX description is unchanged (reply-cache keys)

MLX_DESCRIBE_KEYS = ["model_id", "model_revision", "backend", "device", "dtype", "quantization", "mlx_model_path",
                     "runtime_versions", "chat_template", "generation"]


def _fake_mlx_generator(tmp_path):
    from regulus.rag import mlx_generator as MG
    from tests import test_mlx_backend as T

    d, script = T.model_dir(tmp_path, "ok")
    return MG.MlxGenerator(T.CharTokenizer(), T.Gen(), model_id="Qwen/Qwen3-4B-Instruct-2507", model_path=d,
                           python=sys.executable, worker_script=script)


def test_the_mlx_description_keeps_exactly_its_former_keys(tmp_path):
    """The reply cache keys on the whole describe(): a key added for the MLX worker would make every Mac cache miss."""
    g = _fake_mlx_generator(tmp_path)
    try:
        desc = g.describe()
        assert list(desc) == MLX_DESCRIBE_KEYS and "runtime" not in desc
        assert desc["device"] == "mlx-gpu" and desc["dtype"] == "mlx-4bit-g128"
    finally:
        g.close()


def test_the_torch_description_names_its_runtime_device_and_dtype(tmp_path):
    g = _fake_mlx_generator(tmp_path)
    try:
        g.ready = {**g.ready, "runtime": "torch", "device": "cpu (cpu)", "dtype": "float32"}
        desc = g.describe()
        assert desc["runtime"] == "torch" and desc["device"] == "cpu (cpu)"
        assert desc["dtype"] == "mlx-4bit-g128 computed in float32 (torch)"
    finally:
        g.close()


def test_an_adapter_is_refused(tiny):
    env = dict(os.environ, REGULUS_TORCH_DEVICE="cpu")
    out = subprocess.run([sys.executable, "-B", str(WORKER), str(tiny[0]), str(tiny[0])], capture_output=True, text=True,
                         env=env, timeout=120)
    assert json.loads(out.stdout.splitlines()[0])["op"] == "error"
