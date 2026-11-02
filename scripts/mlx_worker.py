"""MLX generation worker for the opt-in `mlx` backend (regulus.rag.mlx_generator).

Runs under the interpreter named by model.mlx_python (an environment with mlx-lm installed, kept apart
from the project environment); nothing in the project imports this file.

The worker never tokenizes or applies a chat template: the caller sends token ids rendered with the
production tokenizer and template, and decodes the ids it gets back. Decoding is greedy (argmax).

Protocol: one JSON object per line on stdin/stdout.
  startup  -> {"op": "ready", "model_path", "load_seconds", "active_memory_mb", "peak_memory_mb", versions}
  request  <- {"op": "generate", "prompt_ids": [...], "max_new_tokens": N, "eos_ids": [...], "check_every": 16}
  chunk    -> {"op": "chunk", "token_ids": [...]}      (every check_every tokens; caller replies)
  reply    <- {"op": "continue"} | {"op": "stop", "reason": "..."}
  final    -> {"op": "done", "token_ids": [...remaining], "finished": bool, "stopped_by_caller": reason|None,
               "prompt_tokens", "generated_tokens", "prefill_seconds", "seconds", "first_token_seconds",
               "peak_memory_mb", "active_memory_mb"}
  request  <- {"op": "shutdown"}
Errors are reported as {"op": "error", "message": "..."} and the worker keeps serving.
"""
from __future__ import annotations

import json
import os
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

# PX1 prefix cache (28 Oct; default on since 2 Nov, REGULUS_MLX_PREFIX_CACHE=0 off; request field "prefix_cache"): the KV state of
# the previous request is kept, trimmed to the prefix the next prompt shares with it (system prompt, rules, format), and
# only the rest is processed - prefix reuse as llama.cpp and vLLM do it. Adopted only if answers are byte-identical.
_PREFIX = {"cache": None, "ids": []}

MB = 2**20
# MC1 (29 Oct): MLX keeps the buffers a request freed in its own cache for reuse and does not return them to the system.
# On the 8 GB Mac that cache pushed the harness's reranker and embedders into swap (stepA: 7.7 of 8 GB swap in use, section
# choice 34 s -> 75-95 s). The cache is capped and emptied after every request - allocation only: the computation, and so
# every token, is unchanged. REGULUS_MLX_CACHE_LIMIT_MB=-1 restores the unlimited cache.
CACHE_LIMIT_MB = int(os.environ.get("REGULUS_MLX_CACHE_LIMIT_MB", "256"))


def send(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def recv() -> dict:
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return json.loads(line)


def greedy(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def main(model_path: str, adapter_path: str | None = None) -> None:
    import importlib.metadata as md

    t0 = time.perf_counter()
    # a LoRA adapter (REGULUS_MLX_ADAPTER_PATH, evaluation only) is applied on top of the quantized weights
    model, _tokenizer = (load(model_path, adapter_path=adapter_path) if adapter_path else load(model_path))  # the tokenizer is never used for prompts
    mx.eval(model.parameters())
    if CACHE_LIMIT_MB >= 0:
        mx.set_cache_limit(CACHE_LIMIT_MB * MB)
    send({"op": "ready", "model_path": model_path, "adapter_path": adapter_path, "load_seconds": round(time.perf_counter() - t0, 3),
          "active_memory_mb": round(mx.get_active_memory() / MB, 1), "peak_memory_mb": round(mx.get_peak_memory() / MB, 1),
          "versions": {p: md.version(p) for p in ("mlx", "mlx-metal", "mlx-lm")}})
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
            generate(model, req)
        except Exception as exc:  # explicit failure, never a silent fallback
            send({"op": "error", "message": f"{type(exc).__name__}: {exc}"})


def generate(model, req: dict) -> None:
    ids = req["prompt_ids"]
    max_new = int(req["max_new_tokens"])
    eos = set(int(i) for i in req["eos_ids"])
    every = int(req.get("check_every", 16))
    mx.reset_peak_memory()
    start = time.perf_counter()
    first = None
    pending: list[int] = []
    n = 0
    finished = False
    stopped = None
    # optional repetition penalty (22 Oct 2026): greedy Qwen3-4B-Instruct-2507 repeats a sentence three times in about a
    # quarter of long answers; absent from the request = no penalty, exactly as before
    processors = None
    if req.get("repetition_penalty") not in (None, 1.0):
        from mlx_lm.sample_utils import make_logits_processors

        processors = make_logits_processors(repetition_penalty=float(req["repetition_penalty"]),
                                            repetition_context_size=int(req.get("repetition_context_size", 64)))
    feed, kw = ids, {}
    reused = 0
    if req.get("prefix_cache"):
        from mlx_lm.models import cache as C

        if _PREFIX["cache"] is None:
            _PREFIX["cache"], _PREFIX["ids"] = C.make_prompt_cache(model), []
        prev = _PREFIX["ids"]
        common = 0
        for a, b in zip(prev, ids):
            if a != b:
                break
            common += 1
        common = min(common, len(ids) - 1)   # at least one prompt token is processed
        if C.can_trim_prompt_cache(_PREFIX["cache"]) and common > 0:
            C.trim_prompt_cache(_PREFIX["cache"], len(prev) - common)
        else:
            _PREFIX["cache"], common = C.make_prompt_cache(model), 0
        feed, kw, reused = ids[common:], {"prompt_cache": _PREFIX["cache"]}, common
    generated: list[int] = []
    for token, _ in generate_step(mx.array(feed), model, max_tokens=max_new, sampler=greedy, logits_processors=processors, **kw):
        tok = int(token)
        generated.append(tok)
        if first is None:
            first = time.perf_counter() - start
        pending.append(tok)
        n += 1
        if tok in eos:
            finished = True
            break
        if n % every == 0 and n < max_new:
            send({"op": "chunk", "token_ids": pending})
            pending = []
            reply = recv()
            if reply.get("op") == "stop":
                stopped = reply.get("reason", "caller")
                break
    if req.get("prefix_cache"):
        # the cache now holds the prompt and every generated token but the last (never fed back)
        held = (ids + generated)[:_PREFIX["cache"][0].offset] if _PREFIX["cache"] else []
        _PREFIX["ids"] = held
    send({"op": "done", "token_ids": pending, "finished": finished, "stopped_by_caller": stopped, "prefix_reused": reused,
          "prompt_tokens": len(ids), "generated_tokens": n, "seconds": round(time.perf_counter() - start, 3),
          "first_token_seconds": None if first is None else round(first, 3),
          "peak_memory_mb": round(mx.get_peak_memory() / MB, 1), "active_memory_mb": round(mx.get_active_memory() / MB, 1),
          "cache_memory_mb": round(mx.get_cache_memory() / MB, 1)})
    if CACHE_LIMIT_MB >= 0:
        mx.clear_cache()   # after the reply is sent: the caller never waits for it


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
