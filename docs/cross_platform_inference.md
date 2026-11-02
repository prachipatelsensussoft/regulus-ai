# Cross-platform answering: the MLX worker and the PyTorch worker

The answering UI (`scripts/serve_ui.py`, port 8000) and the llm-model API (`scripts/serve_llm_model.py`, port 8100)
answer with one model: Qwen3-4B-Instruct-2507, the `mlx-community` 4-bit build at revision `50d4277…`. On an Apple
Silicon Mac its forward passes run in the MLX worker (`scripts/mlx_worker.py`). MLX does not exist on Windows or on
Linux, so there the same model directory runs in a PyTorch worker (`scripts/torch_worker.py`).

## What is the same on both

The application process does everything except the forward passes, unchanged on both platforms:
retrieval, planning, the prompt and chat template, the token budget, the sentence-repetition stop (every 16 tokens),
decoding, validation and repair, citations, the accuracy observer, the response format, chat history and measurement
records. The two workers speak one protocol (JSON lines on stdin/stdout, documented in both scripts), and
`regulus.rag.mlx_generator.MlxGenerator` drives either.

Inside the worker, the PyTorch worker reproduces mlx-lm 0.32.0 (the version recorded by the Mac runs):

| | MLX worker | PyTorch worker |
|---|---|---|
| Weights | the MLX 4-bit directory | **the same files**, read as they are: no conversion, no second download |
| Dequantization | `q * scale + bias` per group of 64 | the same formula, the same bfloat16 scales and biases |
| Decoding | greedy (argmax) | greedy (argmax) |
| Prompt prefill | steps of 2048 tokens | steps of 2048 tokens |
| Repetition penalty | mlx-lm `make_repetition_penalty`, last 64 tokens of prompt + answer | the same rule and window |
| Stop | first end-of-sequence id, kept | the same |
| Arithmetic | bfloat16 Metal kernels | CUDA: float16 activations and KV cache, float32 accumulation (a fused Triton kernel per generated token, float32 GEMMs for the prompt). CPU: float32 |

## What differs, and how it is recorded

Only the arithmetic. bfloat16 on Metal and float16/float32 on CUDA round differently, so where two candidate tokens
are within rounding of each other a greedy answer can take the other one and continue differently from there. The
Windows answers are therefore not guaranteed byte-identical to the Mac's, although they come from the same weights,
prompt and decoding rule. Nothing hides this:

- the runtime is chosen explicitly (`REGULUS_WORKER_RUNTIME=torch`); the default stays `mlx`, and an MLX start-up that
  fails on a machine that is not a Mac says to set it rather than switching on its own;
- the worker's ready message names the runtime, device, dtype and kernel, and `MlxGenerator.describe()` passes them on,
  so every response's model identity and every measurement record carry, for example, `"runtime": "torch"`,
  `"device": "cuda (NVIDIA GeForce GTX 1650 SUPER)"`, `"dtype": "mlx-4bit-g64 computed in float16 (torch)"`
  (`/api/config` shows the device and dtype). A Windows answer cannot pass for a Mac answer;
- the MLX description is left exactly as it was (no `runtime` key, the same device and dtype), because the reply cache
  keys on the whole `describe()`: existing Mac caches keep matching. `tests/test_torch_worker.py` pins both.

Checked on Windows (GTX 1650 SUPER): the CPU (float32) and GPU (float16) runs gave the same answer word for word to a
probe question, and the same CFR question asked twice gave byte-identical answers and citations. Not yet checked: the
same questions against the Mac's answers. `scripts/platform_parity.py` asks 9 fixed questions through the API and
compares two runs, made on the same commit and migrated database; `docs/mac_verification_checklist.md` says what to run
on each machine and how to read the result. Until then, do not treat Windows numbers as the Mac's.

## Performance (Windows, GTX 1650 SUPER 4 GB, i7-10700F, 14 GB RAM)

| | Measured |
|---|---|
| Model load | 8-10 s |
| GPU memory | 2.2 GB resident, about 2.8 GB at the peak of a 2048-token prefill |
| Prompt prefill | about 11-13 s per 2048 tokens |
| Generation | about 17 tokens/s on a short prompt, 7-9 tokens/s at RAG prompt lengths (2.4-3k tokens) |
| One answer through the UI | 82-165 s on the questions tried (the Mac documents 40-150 s) |
| CPU only (no CUDA) | works, but far too slow for interactive use |

Two things matter on a 4 GB card:

- **One answering server at a time.** The UI and the API each start their own worker; two copies of the model do not
  fit in 4 GB, and Windows then spills GPU memory into system RAM, which made a 2048-token prefill nine times slower in a test (119 s against 12.8 s).
  Run the UI or the API, not both (the 8 GB Mac has the same constraint).
- **Why the kernels are what they are.** Dequantizing the weights before each multiplication reached only about
  1.5 tokens/s: a GTX 16-series card has no tensor cores, and PyTorch's float16 GEMM on it ran at 0.5 TFLOPS against
  3.7 for float32. The fused Triton kernel reads the packed bytes once per token (0.15 ms for a 9728x2560 layer instead
  of 4.7 ms), and the prompt is multiplied in float32.

## Set-up

Windows: `README_LOCAL_SETUP.md` at the repository root. In short: `requirements.txt`, then
`requirements-windows-gpu.txt` (the CUDA build of torch 2.14.0 and `triton-windows`), the model downloads, and
`configs/ui.env` / `configs/llm_model.env` from the examples' Windows sections; start with `scripts/start_ui.ps1` or
`scripts/start_llm_model.ps1`.

Tests: `tests/test_torch_worker.py` writes a tiny random Qwen3 in the MLX 4-bit layout and checks that the worker,
driven over its real protocol, produces exactly the greedy tokens of a transformers model holding the same dequantized
weights (with and without the repetition penalty), plus the unpacking order, the penalty rule and the stop protocol.
