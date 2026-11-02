"""PyTorch generation worker: the MLX worker's protocol and model weights, for machines without MLX (Windows, Linux).

Runs under the interpreter named by model.mlx_python (default: this project's own Python, which has torch) when
model.worker_runtime is "torch" (REGULUS_WORKER_RUNTIME); nothing in the project imports this file.

It is scripts/mlx_worker.py with another runtime underneath, nothing more:
  * the same weights: the MLX 4-bit model directory itself (mlx-community build, model.mlx_model_path). The packed
    4-bit weights stay packed in memory and each layer's are dequantized when it runs, with MLX's affine formula
    (w = q * scale + bias per group), so no second copy of the model is downloaded or converted;
  * the same decoding as mlx-lm 0.32.0's generate_step: greedy argmax, the prompt prefilled in steps of 2048 tokens,
    the optional repetition penalty over the last `repetition_context_size` tokens of prompt + answer (negative
    logits multiplied, positive ones divided), and a stop on the first end-of-sequence id, which is kept;
  * the same protocol, so regulus.rag.mlx_generator renders the prompt, applies the token budget and the repetition
    stop, and decodes exactly as it does for MLX.
What differs is arithmetic: float16 on CUDA (or float32 on CPU) instead of MLX's bfloat16 Metal kernels, so a greedy
answer can part from the Mac's where two tokens are within rounding of each other. The worker says which device and
dtype it runs in its ready message, and the application records it with every answer.

Protocol (as scripts/mlx_worker.py): one JSON object per line on stdin/stdout.
  startup  -> {"op": "ready", "model_path", "load_seconds", "active_memory_mb", "peak_memory_mb", versions,
               "runtime": "torch", "device", "dtype", "weights"}
  request  <- {"op": "generate", "prompt_ids": [...], "max_new_tokens": N, "eos_ids": [...], "check_every": 16}
  chunk    -> {"op": "chunk", "token_ids": [...]}      (every check_every tokens; caller replies)
  reply    <- {"op": "continue"} | {"op": "stop", "reason": "..."}
  final    -> {"op": "done", "token_ids": [...remaining], "finished": bool, "stopped_by_caller": reason|None, ...}
  request  <- {"op": "shutdown"}
Errors are reported as {"op": "error", "message": "..."} and the worker keeps serving.

REGULUS_TORCH_DEVICE: auto (CUDA when available, else CPU) | cuda | cuda:N | cpu.
REGULUS_TORCH_DTYPE: auto (float16 on CUDA, float32 on CPU) | float16 | bfloat16 | float32.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

MB = 2**20
PREFILL_STEP = 2048          # mlx-lm generate_step's prefill_step_size
# largest weight block dequantized (to float32) at once: whole layers on CUDA (32M values, 128 MB), smaller on CPU where
# the temporaries live in RAM beside everything else
CHUNK_ELEMENTS = {"cuda": 32 * 2**20, "cpu": 8 * 2**20}
_PREFIX = {"cache": None, "ids": []}


def send(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def recv() -> dict:
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return json.loads(line)


# ---------------------------------------------------------------------------------------------- MLX 4-bit weights
def dequantize(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, bits: int, group: int,
               dtype: torch.dtype) -> torch.Tensor:
    """MLX affine dequantization. `packed` is MLX's uint32 words viewed as bytes: MLX fills a word lowest bits
    first, so on a little-endian machine byte k of a word holds that word's values 8k/bits .. 8(k+1)/bits - 1,
    each `8 // bits` of them lowest bits first - the same order, unpacked through a 1-byte temporary."""
    per_byte = 8 // bits
    shifts = torch.arange(0, 8, bits, device=packed.device, dtype=torch.uint8)
    q = (packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    rows = packed.shape[0]
    q = q.reshape(rows, -1, group).to(dtype)
    w = torch.addcmul(biases.to(dtype).unsqueeze(-1), q, scales.to(dtype).unsqueeze(-1))   # q * scale + bias, one pass
    return w.reshape(rows, packed.shape[1] * per_byte)


# Decoding multiplies one token's vector by every weight matrix, so its speed is the speed of reading the weights.
# Dequantizing to float16 first writes and reads them again several times (measured on a GTX 1650 SUPER: 4.7 ms for one
# 9728x2560 layer, about 1.5 tokens/s); this kernel unpacks, scales and accumulates in registers and reads only the packed
# bytes (0.15 ms, rel. error 4e-4 against float32). It needs Triton (triton-windows on Windows); without it, or with
# REGULUS_TORCH_KERNEL=torch, every call takes the dequantize-then-matmul path, which computes the same sum.
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _qgemv4(x_ptr, p_ptr, s_ptr, b_ptr, y_ptr, N, K, G: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # y[n] = sum_k (q[n,k] * s[n,k//G] + b[n,k//G]) * x[k]; two values per byte, the low nibble at the even k
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        rmask = rows < N
        j = tl.arange(0, BK // 2)
        acc = tl.zeros([BN], dtype=tl.float32)
        for k0 in range(0, K, BK):
            packed = tl.load(p_ptr + rows[:, None] * (K // 2) + (k0 // 2 + j)[None, :], mask=rmask[:, None], other=0)
            lo = (packed & 15).to(tl.float32)
            hi = (packed >> 4).to(tl.float32)
            grp = ((k0 + 2 * j) // G)[None, :]
            s = tl.load(s_ptr + rows[:, None] * (K // G) + grp, mask=rmask[:, None], other=0).to(tl.float32)
            b = tl.load(b_ptr + rows[:, None] * (K // G) + grp, mask=rmask[:, None], other=0).to(tl.float32)
            xe = tl.load(x_ptr + k0 + 2 * j).to(tl.float32)
            xo = tl.load(x_ptr + k0 + 2 * j + 1).to(tl.float32)
            acc += tl.sum((lo * s + b) * xe[None, :] + (hi * s + b) * xo[None, :], axis=1)
        tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=rmask)

    HAVE_TRITON = (os.environ.get("REGULUS_TORCH_KERNEL") or "auto").strip().lower() != "torch"
except Exception:  # noqa: BLE001 - no Triton: the dequantize path is used and the ready message says so
    HAVE_TRITON = False
GEMV_BN, GEMV_BK = 16, 256   # rows per program, values per step: the fastest tiles measured on the GTX 1650 SUPER


class QuantLinear(torch.nn.Module):
    """nn.Linear over MLX-packed weights: the fused kernel for one token on CUDA, else dequantized per call in blocks
    of output rows."""

    def __init__(self, packed, scales, biases, bits: int, group: int, bias=None):
        super().__init__()
        self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.register_buffer("biases", biases, persistent=False)
        self.register_buffer("bias", bias, persistent=False)
        self.bits, self.group = bits, group
        self.in_features = packed.shape[1] * (8 // bits)
        self.out_features = packed.shape[0]
        self.rows_per_block = max(1, CHUNK_ELEMENTS.get(packed.device.type, CHUNK_ELEMENTS["cpu"]) // self.in_features)
        self.fused = (HAVE_TRITON and packed.device.type == "cuda" and bits == 4 and GEMV_BK % group == 0
                      and self.in_features % GEMV_BK == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused and x.numel() == self.in_features:
            y = torch.empty(self.out_features, device=x.device, dtype=x.dtype)
            _qgemv4[(triton.cdiv(self.out_features, GEMV_BN),)](x.contiguous(), self.packed, self.scales, self.biases, y,
                                                                 self.out_features, self.in_features, G=self.group,
                                                                 BN=GEMV_BN, BK=GEMV_BK)
            y = y.reshape(*x.shape[:-1], self.out_features)
            return y if self.bias is None else y + self.bias.to(y.dtype)
        # the prompt (many rows at once) is multiplied in float32 whatever the activations' dtype: on a GPU without tensor
        # cores float16 GEMMs are the slow path (GTX 1650 SUPER, 2048x2560 @ 2560x9728: 195 ms in float16, 28 ms in
        # float32), and float32 is the more exact of the two
        xf = x.float()
        parts = []
        for a in range(0, self.out_features, self.rows_per_block):
            b = min(a + self.rows_per_block, self.out_features)
            w = dequantize(self.packed[a:b], self.scales[a:b], self.biases[a:b], self.bits, self.group, torch.float32)
            parts.append(F.linear(xf, w))
        y = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        y = y if self.bias is None else y + self.bias.float()
        return y.to(x.dtype)


class QuantEmbedding(torch.nn.Module):
    """nn.Embedding over MLX-packed rows: only the looked-up rows are dequantized."""

    def __init__(self, packed, scales, biases, bits: int, group: int, dtype: torch.dtype):
        super().__init__()
        self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.register_buffer("biases", biases, persistent=False)
        self.bits, self.group, self.dtype = bits, group, dtype

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1)
        w = dequantize(self.packed[flat], self.scales[flat], self.biases[flat], self.bits, self.group, self.dtype)
        return w.reshape(*ids.shape, -1)


def _read_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """Every tensor of the model's safetensors files, on the CPU."""
    out: dict[str, torch.Tensor] = {}
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    for f in files:
        with safe_open(str(f), framework="pt") as st:
            for key in st.keys():
                out[key] = st.get_tensor(key)
    return out


def _to(t: torch.Tensor, device, dtype=None) -> torch.Tensor:
    if t.dtype == torch.uint32:
        if sys.byteorder != "little":
            raise RuntimeError("MLX-packed weights are read as little-endian bytes")
        t = t.view(torch.uint8)          # (rows, words) -> (rows, 4 * words): the bit pattern unchanged
    return t.to(device=device, dtype=dtype) if dtype is not None else t.to(device)


def _quant_spec(quant: dict, prefix: str) -> tuple[int, int] | None:
    """(bits, group_size) for one module: mlx-lm writes per-module overrides into the same dict (False = unquantized)."""
    own = quant.get(prefix)
    if own is False:
        return None
    if isinstance(own, dict):
        return int(own["bits"]), int(own["group_size"])
    return int(quant["bits"]), int(quant["group_size"])


def resolve_device_dtype() -> tuple[torch.device, torch.dtype]:
    want = (os.environ.get("REGULUS_TORCH_DEVICE") or "auto").strip().lower()
    if want == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(want)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"REGULUS_TORCH_DEVICE={want} but this torch build has no CUDA "
                               f"(torch {torch.__version__}); install the CUDA build or use cpu")
    names = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    raw = (os.environ.get("REGULUS_TORCH_DTYPE") or "auto").strip().lower()
    if raw == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif raw in names:
        dtype = names[raw]
    else:
        raise RuntimeError(f"REGULUS_TORCH_DTYPE={raw!r}: use auto, float16, bfloat16 or float32")
    return device, dtype


def load(model_dir: str):
    """Qwen3ForCausalLM with the MLX directory's quantized weights, on the chosen device."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    path = Path(model_dir)
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    quant = cfg.get("quantization") or cfg.get("quantization_config")
    if cfg.get("model_type") != "qwen3":
        raise ValueError(f"model_type {cfg.get('model_type')!r}: this worker runs Qwen3 models only")
    if not quant:
        raise ValueError("config.json has no MLX quantization block: expected an MLX 4-bit model directory")
    if int(quant["bits"]) not in (2, 4, 8):
        raise ValueError(f"{quant['bits']}-bit weights are not supported (2, 4 or 8)")
    device, dtype = resolve_device_dtype()
    hf = {k: v for k, v in cfg.items() if k not in ("quantization", "quantization_config")}
    config = Qwen3Config(**hf)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    tensors = _read_tensors(path)

    def take(name: str) -> torch.Tensor:
        return tensors.pop(name)

    def quantized(prefix: str):
        spec = _quant_spec(quant, prefix)
        if spec is None:
            return None
        bits, group = spec
        packed = _to(take(f"{prefix}.weight"), device)
        return packed, _to(take(f"{prefix}.scales"), device), _to(take(f"{prefix}.biases"), device), bits, group

    def set_module(name: str, module: torch.nn.Module) -> None:
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, module)

    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and name != "lm_head":
            q = quantized(name)
            bias = tensors.pop(f"{name}.bias", None)
            bias = _to(bias, device, dtype) if bias is not None else None
            if q is None:
                lin = torch.nn.Linear(module.in_features, module.out_features, bias=bias is not None,
                                      device=device, dtype=dtype)
                lin.weight = torch.nn.Parameter(_to(take(f"{name}.weight"), device, dtype), requires_grad=False)
                if bias is not None:
                    lin.bias = torch.nn.Parameter(bias, requires_grad=False)
                set_module(name, lin)
            else:
                set_module(name, QuantLinear(*q, bias=bias))
    emb = quantized("model.embed_tokens")
    if emb is None:
        raise ValueError("unquantized embeddings are not supported by this worker")
    model.model.embed_tokens = QuantEmbedding(*emb, dtype=dtype)
    if config.tie_word_embeddings:
        model.lm_head = QuantLinear(*emb)                 # mlx-lm: embed_tokens.as_linear
    else:
        model.lm_head = QuantLinear(*quantized("lm_head"))
    # what is left are the norms (RMSNorm weights), loaded as they are
    rest = {k: _to(v, device, dtype) for k, v in tensors.items()}
    missing, unexpected = model.load_state_dict(rest, strict=False, assign=True)
    if unexpected:
        raise ValueError(f"tensors the model does not have: {unexpected[:5]}")
    still_meta = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    model.model.rotary_emb = Qwen3RotaryEmbedding(config=config).to(device)   # non-persistent buffers, rebuilt
    still_meta = [n for n in still_meta if not n.startswith("model.rotary_emb")]
    if still_meta:
        raise ValueError(f"weights missing from the model directory: {still_meta[:5]}")
    model.eval()
    return model, device, dtype


# ---------------------------------------------------------------------------------------------- generation
def memory(device: torch.device) -> tuple[float, float]:
    """(active, peak) MB: CUDA allocator figures, or this process's resident memory on CPU."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / MB, torch.cuda.max_memory_allocated(device) / MB
    import psutil

    rss = psutil.Process().memory_info().rss / MB
    return rss, rss


def repetition_penalty(logits: torch.Tensor, context: list[int], penalty: float) -> torch.Tensor:
    """mlx-lm's make_repetition_penalty: each token in the context once, sign-aware multiplicative penalty."""
    if not context:
        return logits
    idx = torch.tensor(context, device=logits.device, dtype=torch.long)
    sel = logits[:, idx]
    logits[:, idx] = torch.where(sel < 0, sel * penalty, sel / penalty)
    return logits


@torch.inference_mode()
def generate(model, device: torch.device, req: dict) -> None:
    from transformers import DynamicCache

    ids = [int(i) for i in req["prompt_ids"]]
    max_new = int(req["max_new_tokens"])
    eos = set(int(i) for i in req["eos_ids"])
    every = int(req.get("check_every", 16))
    penalty = req.get("repetition_penalty")
    penalty = None if penalty in (None, 1.0) else float(penalty)
    context_size = int(req.get("repetition_context_size", 64))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    first = None
    pending: list[int] = []
    n = 0
    finished = False
    stopped = None
    feed, reused = ids, 0
    if req.get("prefix_cache"):
        prev = _PREFIX["ids"]
        common = 0
        for a, b in zip(prev, ids):
            if a != b:
                break
            common += 1
        common = min(common, len(ids) - 1)
        if _PREFIX["cache"] is not None and common > 0:
            _PREFIX["cache"].crop(common)
        else:
            _PREFIX["cache"], common = DynamicCache(), 0
        cache, feed, reused = _PREFIX["cache"], ids[common:], common
    else:
        cache = DynamicCache()
    seen = list(ids)       # prompt + answer: the repetition penalty's context, as mlx-lm keeps it
    pos = reused

    def forward(chunk: list[int]) -> torch.Tensor:
        nonlocal pos
        x = torch.tensor([chunk], device=device, dtype=torch.long)
        positions = torch.arange(pos, pos + len(chunk), device=device).unsqueeze(0)
        out = model(input_ids=x, past_key_values=cache, use_cache=True, position_ids=positions,
                    cache_position=positions[0], logits_to_keep=1)
        pos += len(chunk)
        return out.logits[:, -1, :].float()

    # prefill all but the last prompt token in steps, then the last one gives the first token's logits
    while len(feed) > 1:
        step = feed[:min(PREFILL_STEP, len(feed) - 1)]
        forward(step)
        feed = feed[len(step):]
    logits = forward(feed)
    generated: list[int] = []
    while True:
        if not torch.isfinite(logits).all():
            raise FloatingPointError("the logits are not finite (float16 overflow?); set REGULUS_TORCH_DTYPE=float32")
        if penalty is not None:
            logits = repetition_penalty(logits, seen[-context_size:], penalty)
        tok = int(torch.argmax(logits, dim=-1)[0])
        generated.append(tok)
        seen.append(tok)
        if first is None:
            first = time.perf_counter() - start
        pending.append(tok)
        n += 1
        if tok in eos:
            finished = True
            break
        if n >= max_new:
            break
        if n % every == 0:
            send({"op": "chunk", "token_ids": pending})
            pending = []
            reply = recv()
            if reply.get("op") == "stop":
                stopped = reply.get("reason", "caller")
                break
        logits = forward([tok])
    if req.get("prefix_cache"):
        _PREFIX["ids"] = (ids + generated)[:cache.get_seq_length()]
    active, peak = memory(device)
    send({"op": "done", "token_ids": pending, "finished": finished, "stopped_by_caller": stopped, "prefix_reused": reused,
          "prompt_tokens": len(ids), "generated_tokens": n, "seconds": round(time.perf_counter() - start, 3),
          "first_token_seconds": None if first is None else round(first, 3),
          "peak_memory_mb": round(peak, 1), "active_memory_mb": round(active, 1)})
    if device.type == "cuda" and not req.get("prefix_cache"):
        torch.cuda.empty_cache()


def main(model_path: str, adapter_path: str | None = None) -> None:
    import importlib.metadata as md

    if adapter_path:
        send({"op": "error", "message": "LoRA adapters (REGULUS_MLX_ADAPTER_PATH) are MLX-only; unset it"})
        return
    t0 = time.perf_counter()
    torch.set_grad_enabled(False)
    model, device, dtype = load(model_path)
    active, peak = memory(device)
    quant = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8")).get("quantization") or {}
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    send({"op": "ready", "model_path": model_path, "adapter_path": None, "load_seconds": round(time.perf_counter() - t0, 3),
          "active_memory_mb": round(active, 1), "peak_memory_mb": round(peak, 1),
          "versions": {p: md.version(p) for p in ("torch", "transformers", "safetensors")},
          "runtime": "torch", "device": f"{device} ({name})", "dtype": str(dtype).replace("torch.", ""),
          "weights": f"mlx-{quant.get('bits')}bit-g{quant.get('group_size')}, dequantized per layer",
          "kernel": ("triton fused 4-bit GEMV per token, dequantize+matmul for the prompt"
                     if any(getattr(m, "fused", False) for m in model.modules()) else "dequantize+matmul")})
    while True:
        try:
            req = recv()
        except EOFError:
            return
        if req.get("op") == "shutdown":
            send({"op": "bye"})
            return
        if req.get("op") != "generate":
            send({"op": "error", "message": f"unknown op {req.get('op')!r}"})
            continue
        try:
            generate(model, device, req)
        except Exception as exc:  # explicit failure, never a silent fallback
            if device.type == "cuda":
                torch.cuda.empty_cache()
            send({"op": "error", "message": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
