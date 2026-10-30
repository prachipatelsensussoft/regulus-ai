# Regulus AI

Client-facing AI system specialised in FDA and eCFR regulatory information.

**Completed stages**

1. Qwen3 base-model foundation: reproducible Python environment, configuration
   layer, model download, inference smoke test, unit tests.
2. Interactive terminal chat with in-memory multi-turn context.
3. Baseline evaluation: a fixed benchmark dataset, a pipeline-agnostic runner,
   saved results with reproducibility metadata, and a report tool.
4. Database inspection (`docs/database_architecture.md`) and embedding
   provenance verification (`docs/embedding_provenance.md`, result: UNVERIFIED).
5. Read-only lexical retrieval over the PostgreSQL corpus
   (`docs/lexical_retrieval.md`): exact CFR lookup, full-text, trigram,
   filters, citations, CLI, retrieval evaluation.
6. Minimal lexical RAG (`docs/lexical_rag.md`): retrieval → bounded context →
   Qwen → answer with authoritative citations; the `baseline_v1` benchmark run
   through RAG for comparison with the raw model.

7. Controlled OR fallback for the lexical retriever (Stage 7 sections of
   `docs/lexical_retrieval.md` and `docs/lexical_rag.md`): activates only when
   the primary lexical path finds fewer than 3 results; `--no-fallback`
   reproduces the Stage 5/6 behaviour everywhere.

8. Controlled semantic retrieval experiment
   (`docs/semantic_retrieval_experiment.md`): a pinned local embedding model
   (`BAAI/bge-small-en-v1.5`) over a 2,676-passage corpus extracted read-only
   from the database; opt-in via `--retriever semantic`; the database's
   existing 1024-d vectors are untouched and unused.

9. Controlled hybrid retrieval experiment
   (`docs/hybrid_retrieval_experiment.md`): three fusion strategies over the
   Stage 7 lexical and Stage 8 semantic legs, with lexical authority for
   identifiers and non-existence; opt-in via `--retriever hybrid`.

10. Expanded-corpus hybrid validation
    (`docs/expanded_hybrid_retrieval_experiment.md`): a 26,083-item semantic
    corpus version (`data/evaluation/semantic/semantic_expanded_v1/`, selected
    with `--semantic-corpus semantic_expanded_v1`) and the larger
    `retrieval_v2.jsonl` benchmark with fake-identifier negative controls.

11. Training dataset preparation (`docs/training_data_preparation.md`):
    9,445 instruction-tuning examples (question, context, grounded answer,
    citations) generated deterministically from the read-only database with
    templates, no model in the loop; nine categories including abstention /
    negative examples; benchmark questions excluded; group-level
    train/validation/test split; validation against the database
    (`scripts/prepare_training_data.py`, `validate_training_data.py`,
    `report_training_data.py`). Training itself is a later stage.

12. First LoRA fine-tune of Qwen3-0.6B (`docs/stage12_model_training.md`):
    the Stage 11 dataset trained with PEFT LoRA on CPU (r=16, 10.1M trainable
    parameters, 40 steps, validation loss 1.208 to 0.301), saved as a
    reloadable adapter in `models/qwen3-0.6b-lora/stage12_v1/` and evaluated
    through the unchanged Stage 7 RAG pipeline
    (`scripts/train_model.py`, `scripts/evaluate_stage12.py`). The adapter
    loads and generates, but scores below the un-finetuned model on
    `baseline_v1` (0.59 against 1.11) and fabricates content for fictitious
    CFR identifiers, so it is opt-in only via `--adapter` and is not used by
    any default path.

13. Grounding correction and a larger fine-tune
    (`docs/stage13_grounding_and_training.md`): a deterministic
    identifier-grounding layer (`src/regulus/grounding/`) that verifies a
    requested CFR section, part, paragraph or docket against the read-only
    database and refuses when it does not exist or when nothing retrieved
    carries it; a corrected training dataset (`configs/training_v2.yaml`,
    20.1 % refusal-required including 500 examples whose requested identifier
    genuinely exists); Stage 13 grounding metrics
    (`scripts/compare_grounding.py`); and a deadline-bounded LoRA run. The
    gate is opt-in via `--grounding` and eliminates unsupported answers
    without adding any false refusals.

14. Query understanding (`docs/stage1_query_understanding.md`, architecture
    Stage 1 of `docs/stage0_architecture_blueprint.md`): a single
    deterministic scanner (`src/regulus/retrieval/references.py`) that
    extracts every CFR reference in a question in text order, accepts bare
    forms such as `820.10(b)(1)`, expands ranges (`(b)(1) through (b)(4)`)
    and lists (`(b)(1), (b)(2), and (b)(4)`) into atomic requirements, and
    records them as a `QueryPlan` (`src/regulus/planning/`). No database or
    model is involved. `parse_cfr_reference()` and `classify()` keep their
    APIs; retrieval still consumes the first reference until Stage 2.
    `python scripts/plan.py "<question>"` prints the trace.

15. Requirement-level retrieval (`docs/stage2_requirement_retrieval.md`,
    architecture Stage 2): `LexicalRetriever.run_plan(plan)` retrieves every
    requirement of the `QueryPlan` on its own (exact rows plus the parent
    paragraph), classifies each retrieved row by identity into exact /
    structural / supporting / discovery evidence, and records it in an
    `EvidenceRegistry` (`src/regulus/retrieval/evidence.py`) with per-
    requirement diagnostics. Broad fallback runs only per unresolved
    requirement; exact and structural evidence is never cut by `top_k`.
    `RagPipeline` builds the plan once and uses `run_plan`; `run(question)`
    is unchanged for the hybrid retriever, scripts and baselines. Context
    budgeting, grounding and generation are unchanged (Stage 3).

16. Evidence coverage and protected context
    (`docs/stage3_coverage_and_context.md`, architecture Stage 3): a
    `CoverageChecker` (`src/regulus/rag/coverage.py`) gives every explicit
    requirement a deterministic status and the plan a state (complete,
    incomplete, or no explicit requirements) using only exact evidence; an
    `EvidenceSelector` (`src/regulus/rag/selection.py`) fills the context
    required evidence first, protected from `top_k`, `max_results` and the
    soft character budget, then structural, supporting and discovery
    evidence within it, with hard ceilings and explicit overflow reporting.
    Incomplete explicit requests answer only from the supported provisions
    and name the missing ones, or return a deterministic answer without the
    model when nothing is supported. Prompt, gate and generator unchanged.

17. Generation contract, answer validation and citation support
    (`docs/stage4_generation_and_validation.md`, architecture Stage 4): the
    model receives a `GenerationPackage` (`src/regulus/rag/generation.py`)
    that names the requested provisions and labels every source with its
    tier and the provisions it relates to; the token budget is set per
    question from the plan (default 64 + 64 per provision, cap 448, chosen
    from measurements; the old fixed 128 truncated 3 of 6 representative
    answers). Every generated answer is validated deterministically
    (`src/regulus/rag/validation.py`: requirement coverage, citation
    validity, evidence association, missing-provision assertions, figures
    absent from the sources, truncation, degeneration) and repaired from the
    verbatim exact text when it fails; citations trace `[n]` to the selected
    evidence and its requirement (`src/regulus/rag/citations.py`). No
    training, no database or UI change.

18. Stage 5 baseline evaluation (`docs/stage5_evaluation.md`, artifacts under
    `artifacts/stage5_stage6/<run>/stage5/`): a 51-question evaluation set
    (`data/evaluation/datasets/stage5_v1.jsonl`, protected like the other
    benchmarks) with explicit expectations per question, measured layer by
    layer. The evidence phases (retrieval, coverage, selection) need no model;
    the generation phase runs each question twice, once through the full
    pipeline and once with exact evidence only, so a failure can be attributed
    to retrieval, context, prompting, generation or validation. Raw model
    answers, validator results, repaired answers and the correctness of each
    are reported separately; "valid after repair" is never counted as model
    quality. Run it with `python scripts/run_stage5.py`.

19. Stage 6 training experiment (`docs/stage6_training_design.md`): design and
    data path only, no training run. Adds multi-provision, range and
    partial (proven-absent) training categories rendered with the Stage 4
    generation contract, group-level benchmark exclusion so no evaluation
    section is trained on, and database-backed re-verification that every
    identifier called absent really is absent. Build and check the dataset
    with `python scripts/prepare_stage6_dataset.py --plan <stage5_training_plan.json>`.

20. Model evaluation dashboard (`docs/model_dashboard_architecture.md`,
    `docs/model_dashboard_metrics.md`): a permanent, read-only view over every
    evaluation and training artifact the project has produced. It discovers
    runs generically, normalises them into one canonical schema, and resolves
    each one to CURRENT, HISTORICAL, STALE, INCOMPLETE or FAILED, so a
    historical number is never shown as the current model's performance. Every
    figure carries its model, benchmark content hash, run id, timestamp,
    configuration and source file; raw and repaired answers are never merged;
    training loss is kept in its own family and never presented as accuracy;
    and there is no composite score and no automatic winner. Run it with
    `python scripts/serve_dashboard.py` and open <http://127.0.0.1:8050>. It
    loads no model and opens no database connection, so it can run beside the
    answering UI. Regenerate the metric reference with
    `python scripts/generate_metric_docs.py`.

21. Adaptive answer length (`docs/answer_length_policy.md`): the generation
    contract no longer carries a preferred length. Each question gets a token
    **allowance** derived from the provisions asked about and the exact
    evidence the answer is grounded in, floored so a one-provision answer
    still has room for its conditions, and capped by a deliberately generous
    safety ceiling that sits above every allowance the policy can request.
    Brevity is asked for once, in the grounded system prompt, where the model
    can weigh it against completeness; no format line prescribes a sentence
    count, and a regression test asserts none ever reappears. Truncation is
    detected from the end-of-turn token, named by which limit it hit, filled
    in from the exact sources, and stated in the answer rather than hidden.
    Configured in one place under `answer_length:` in `configs/model.yaml`
    (`REGULUS_MAX_NEW_TOKENS`, `REGULUS_ANSWER_LENGTH_POLICY`).

22. Adaptive answer shape (`docs/adaptive_answer_shape.md`): the question is
    read into an explicit `AnswerShape` (synthesis, focused, enumeration,
    comparison or direct; scope; exhaustive or not) from the Stage 1 plan and
    its wording, deterministically. The selected evidence is organised into an
    `AnswerPlan` by eCFR hierarchy and role (lead-in, requirement,
    sub-requirement, consequence, context, related, with each source's
    qualifiers, applicability limits and cross-references), and the model is
    told to write per requirement group, per item, per comparison side or about
    the one provision asked for. Every source stays in the plan and keeps its
    citation. Validation adds shape checks: row-by-row dumps, scope expansion,
    missing items of an explicit enumeration (repaired verbatim), lost
    qualifiers or applicability limits, citations that do not support a
    synthesized statement, repetition. There is no length check, and the Stage
    11 budget never sees the shape. `RagConfig.answer_shape="off"` restores the
    previous behaviour for comparison; `scripts/evaluate_answer_shape.py` runs
    the live check on `data/evaluation/datasets/answer_shape_v1.jsonl`.

23. Live answer accuracy (`docs/LIVE_ANSWER_ACCURACY.md`, which describes v1
    and notes what changed; the evaluator emits
    `regulus_live_evidence_assessment_v2`): the test UI shows a per-answer
    accuracy assessment of the delivered answer against the authoritative
    evidence retrieved for its own question, over seven dimensions (answers the
    question, substantive correctness, evidence support, citation correctness,
    unsupported claims, completeness, scope) with a documented deduction
    formula, read as PASS / FAIL / NOT EVALUATED with a score out of 100. A
    statement counts as borne out only when it is verbatim in that evidence or
    a paraphrase whose regulatory operators (strength, polarity, duty-holder,
    quantities, dates, conditions) all agree with it - never by word overlap,
    because span-support and content-recall scoring were measured at 85.5% and
    97.8% false passes against the frozen substantive rubric
    (`docs/STAGE_A_VALIDATOR_LIMITATION.md`) - and it abstains rather than
    issuing a number it cannot support. The
    evaluator is an observer: `regulus.accuracy` runs after
    `RagPipeline.answer()` returns, nothing under `regulus.rag` imports it, and
    `--no-accuracy` (or `{"accuracy": false}` per request) leaves the answer,
    its citations and its evidence byte-for-byte unchanged -
    `scripts/accuracy_regression.py` and `tests/test_accuracy_observer.py`
    verify that. It is NOT the 300-question acceptance score and no system
    change may be justified by moving it.

Use of the existing (UNVERIFIED) embeddings, a full-database semantic index,
reranking, APIs, UI and deployment are later stages and are intentionally
absent. The lexical retriever and the un-finetuned base model remain the
default everywhere; the grounding gate and every fine-tuned adapter are
opt-in.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11 | 3.11.x is the tested version. 3.12/3.13 should work but are untested here. |
| Git | |
| ~2 GB disk for the model | `Qwen/Qwen3-0.6B` is a 1.5 GB download. |
| ~4 GB free RAM (CPU mode) | float32 weights are ~2.4 GB plus working memory. |
| NVIDIA GPU + CUDA (optional) | Detected automatically. Not required. |
| Internet on first run | Downloads the model from the Hugging Face Hub. |

No Hugging Face account or token is needed; Qwen3 is a public, Apache-2.0 model.

## 2. Setup

All commands are run from the project root.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `regulus` package from `src/` in editable
mode so scripts and tests can `import regulus` from anywhere.

### Database

Restore the 15 Sep 2026 baseline backup into PostgreSQL, set `REGULUS_DATABASE_URL` in `.env`, then bring the
eCFR records to the repaired data this repository expects (previews and asks before writing; Windows and macOS):

```bash
python -m regulus.setup status   # read-only
python -m regulus.setup          # apply the pending data migrations, rebuild the semantic index
```

See `docs/DATA_MIGRATIONS.md`.

### GPU machines

The `torch` wheel on PyPI is CPU-only on Windows. On a machine with an NVIDIA
GPU, install the CUDA build first and then the rest of the requirements:

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
```

The code detects CUDA at runtime; no configuration change is needed.

## 3. Verify the environment

```bash
python scripts/check_env.py
```

Prints the Python and PyTorch versions, whether CUDA is available, the GPU
name and CUDA version when present, and the resolved model configuration
(model id, cache directory, device, dtype).

## 4. Selected model: `Qwen/Qwen3-0.6B`

**Why this model for the first development stage**

- **Fits a normal development machine.** 0.6B parameters, 1.5 GB download,
  ~2.4 GB in RAM as float32. It runs on CPU with no GPU at all. The next size
  up (Qwen3-1.7B) needs ~7 GB RAM in float32 and is not practical on an 8 GB
  CPU-only machine.
- **Same family as the production candidates.** Qwen3-0.6B shares the
  architecture, tokenizer and chat template with Qwen3-4B/8B/14B/32B. Code
  written against it moves to a larger checkpoint by changing one config
  value.
- **Standard Transformers support.** Loads with `AutoTokenizer` /
  `AutoModelForCausalLM`; no custom code, no `trust_remote_code`.
- **Licence.** Apache-2.0, suitable for commercial, client-facing use.
- **Fine-tuning ready.** Plain safetensors checkpoint; compatible with
  LoRA / SFT tooling for later stages.

The development machine this was set up on is CPU-only (Intel i5-4590, 8 GB
RAM, no NVIDIA GPU), which is why a small model was chosen. This is a
development choice, not the final production model.

## 5. Model storage

Weights are downloaded with Hugging Face tooling into a **project-local cache**:

```
models/
└── models--Qwen--Qwen3-0.6B/     # standard Hugging Face hub cache layout
```

- `models/` is gitignored. Never commit weights.
- Re-running the download is a no-op when the files are present.
- To share one cache across projects, set `REGULUS_MODEL_CACHE_DIR` to your
  global Hugging Face cache (for example `~/.cache/huggingface/hub`).

Download explicitly (recommended, so the first inference run is fast):

```bash
python scripts/download_model.py
```

## 6. Configuration

All model settings live in [`configs/model.yaml`](configs/model.yaml):

```yaml
model:
  id: "Qwen/Qwen3-0.6B"
  cache_dir: "models"     # relative to project root
  device: "auto"          # auto | cpu | cuda | cuda:0
  dtype: "auto"           # auto | float32 | float16 | bfloat16
generation:
  max_new_tokens: 128
  ...
```

Every `model.*` value can be overridden by an environment variable, either
exported in the shell or placed in a `.env` file (copy `.env.example`; `.env`
is gitignored):

| Variable | Overrides |
|---|---|
| `REGULUS_MODEL_ID` | `model.id` |
| `REGULUS_MODEL_CACHE_DIR` | `model.cache_dir` |
| `REGULUS_DEVICE` | `model.device` |
| `REGULUS_DTYPE` | `model.dtype` |
| `HF_TOKEN` | Hugging Face token (only for gated/private models) |
| `REGULUS_DATABASE_URL` | PostgreSQL URL for the read-only retrieval layer (Stage 5); `.env` only |

`dtype: auto` resolves to float32 on CPU and bfloat16 on CUDA (float16 if the
GPU has no bf16 support). Secrets are never stored in source or YAML.

## 7. Run the inference smoke test

```bash
python scripts/run_inference.py
```

Options:

```
--prompt "..."          custom user message
--max-new-tokens N      cap the response length (faster on CPU)
--greedy                deterministic decoding
--thinking              enable Qwen3 thinking mode (much slower on CPU)
--offline               fail instead of downloading if not cached
-v                      show library logs
```

### Expected behaviour

1. Prints environment info (Python, torch, CUDA availability, GPU name).
2. Loads the tokenizer and model from `models/`, downloading on first run.
3. Prints device, dtype, parameter count and load time.
4. Generates one response to the prompt and prints it with token counts and
   tokens/second.
5. Ends with `OK: inference ran on CPU` (or `GPU`).

Errors such as a wrong model id, no network, a gated model, or out-of-memory
are caught and printed as a one-line `MODEL LOAD ERROR: ...` with exit code 1.

### CPU limitations

Measured on the reference CPU-only machine (Intel i5-4590, 4 cores, 8 GB RAM,
float32):

| Step | Measured |
|---|---|
| Model load (from local cache) | ~24 s |
| Generation speed | ~1.9 tokens/second |
| 80-token answer | ~15–45 s depending on length |

- A full 128-token answer can take over a minute. Use `--max-new-tokens` to
  keep smoke tests short.
- Thinking mode can emit hundreds of tokens before answering; keep it off on
  CPU unless you need it.
- Keep ~4 GB RAM free; close heavy applications before running.
- On Windows, Hugging Face prints a one-time warning that the cache cannot
  use symlinks. This is harmless (files are stored directly, no duplication).
  Enable Windows Developer Mode to silence it, or ignore it.
- Hugging Face also warns about "unauthenticated requests". This is
  informational; no token is required for Qwen3. Use `--offline` to skip Hub
  checks entirely once the model is cached.

## 7b. Interactive terminal chat

A multi-turn testing interface for the base model. The model is loaded once;
conversation history is kept in memory for the session and rendered through
the Qwen3 chat template, so follow-up questions see earlier turns. Nothing is
saved to disk.

```bash
python scripts/chat.py
python scripts/chat.py --max-new-tokens 200 --temperature 0.5 --top-p 0.9
python scripts/chat.py --greedy --system "You are a concise assistant."
python scripts/chat.py --stats          # token counts and speed per answer
```

Defaults come from the `generation` section of `configs/model.yaml`; flags
override them for that session. Answers stream token by token (use
`--no-stream` to print them whole).

| Command | Effect |
|---|---|
| `/help` | list commands |
| `/clear` | forget the conversation |
| `/history` | print the conversation so far |
| `/settings` | show model, device and generation settings |
| `/exit`, `/quit` | leave (Ctrl+D / Ctrl+Z+Enter also work) |

Ctrl+C while an answer is being generated stops that answer and discards the
message; the session continues. On CPU, a long history makes every turn slower
because the whole conversation is re-encoded; use `/clear` when switching
topics.

## 7c. Baseline Evaluation

**Why it exists.** Before adding retrieval or fine-tuning we need an honest,
repeatable measurement of what the untouched model can do. Every later stage
(Qwen + RAG, fine-tuned Qwen) is run against the *same* dataset with the
*same* runner so results are directly comparable. The questions must not be
changed between stages without a documented reason; if they change, bump the
dataset file name (`baseline_v2.jsonl`) rather than editing `v1` in place.

**Dataset.** `data/evaluation/datasets/baseline_v1.jsonl`, 27 hand-written
questions in five categories, one JSON object per line:

| Category key | Label | Purpose |
|---|---|---|
| `general_knowledge` | General knowledge | FDA, drugs, trials, devices |
| `regulatory_concepts` | Regulatory concepts | CFR, Federal Register, regulation vs guidance |
| `fda_ecfr` | FDA/eCFR-oriented | Does the base model know Title 21 at all? |
| `reasoning` | Reasoning | Plain-language explanations and comparisons |
| `hallucination` | Unknown/hallucination | Fictitious citations; the model should admit uncertainty |

Fields: `id`, `category`, `question`, `reference_answer`, `evaluation_notes`,
`expected_behavior` (`answer` or `acknowledge_uncertainty`). Where no single
correct answer exists the reference says so and the notes describe what a
good answer contains. The dataset is version-controlled; its SHA-256 is
recorded in every result file.

**Run it.**

```bash
python scripts/evaluate_baseline.py                  # full run, greedy decoding
python scripts/evaluate_baseline.py --limit 3        # quick check
python scripts/evaluate_baseline.py --ids baseline-023 baseline-027
python scripts/evaluate_baseline.py --max-new-tokens 100
python scripts/evaluate_baseline.py --sample --seed 7 --temperature 0.7
python scripts/evaluate_baseline.py --pipeline rag   # Stage 6: lexical retrieval + model
```

The model is loaded once. Each question is sent as a single-turn chat message
with no system prompt, no history and no retrieved context. Greedy decoding
is the default so two runs on the same machine and software give identical
answers; `--sample` switches to the sampling settings in `configs/model.yaml`
with the given `--seed`. `max_new_tokens` and the other generation values come
from the config file unless overridden. On the reference CPU machine a full
run of 27 questions at 128 tokens takes roughly 15–25 minutes.

**Results** are written to `data/evaluation/results/` as
`<pipeline>_<YYYYmmdd>_<HHMMSS>_<run_id>.json`, for example
`baseline_20260915_154512_a1b2c3d4.json`. Result files are gitignored by
default; commit a reference run deliberately with `git add -f`.

Each file records:

- `run`: run id, UTC timestamp, pipeline label, model id and Hub revision,
  device, dtype, decoding mode, seed, all generation parameters, dataset name
  / path / SHA-256 / size, number of questions, Python / torch / transformers
  versions, CUDA availability and GPU name, total duration. No file-system
  paths or secrets are stored.
- `summary`: total / completed / failed (with failed ids), average latency,
  average input and output tokens, average tokens per second, counts per
  category, and manual-scoring statistics once scores exist.
- `results`: one record per question with the question, reference answer,
  evaluation notes, expected behaviour, status, raw model answer, generation
  seconds, input and output token counts, tokens per second, plus
  `quality_score: null` and `manual_evaluation: null` for review.

A question that raises an exception is recorded as `failed` with a
path-scrubbed error message and the run continues.

**Manual evaluation.** No automatic quality score is assigned; string matching
against a small model's free-text answers would be misleading. To score a
run, edit the result file and set, per question:

```json
"quality_score": 2,
"manual_evaluation": {"reviewer": "name", "comment": "correct, mentions Part 314"}
```

Scale: `0` incorrect, `1` partially correct, `2` correct. For
`acknowledge_uncertainty` questions, `2` means the model said it could not
verify or that the item does not exist; `0` means it confidently invented
details. The report then shows the mean score overall and per category.

**Report.**

```bash
python scripts/report_baseline.py                          # newest result
python scripts/report_baseline.py data/evaluation/results/baseline_....json
python scripts/report_baseline.py --answers                # every Q, A, reference
python scripts/report_baseline.py --answers --category hallucination
python scripts/report_baseline.py --answers --ids baseline-023
```

**Reusing the benchmark.** The runner (`regulus.evaluation.run_evaluation`)
takes any `answer_fn(question) -> GenerationResult`. The baseline script
passes the raw model; a RAG stage will pass a function that retrieves context
first, and a fine-tuned stage will pass a different checkpoint. Dataset,
result schema, metrics and report stay the same, so the three result files
can be compared side by side.

## 7d. Lexical retrieval (read-only PostgreSQL)

Stage 5 adds `regulus.retrieval`: a `LexicalRetriever` that answers a query
with citation-ready passages from `cfr_records` and `fda_guidance_chunks`
using exact CFR lookup, PostgreSQL full-text search and trigram matching. The
database session is forced read-only by the server. No embeddings are used
(their provenance is UNVERIFIED, see `docs/embedding_provenance.md`) and no
model is involved. Full design, ranking formula, evaluation and limitations:
[`docs/lexical_retrieval.md`](docs/lexical_retrieval.md).

```bash
python scripts/search.py "21 CFR 820.10"
python scripts/search.py "quality system requirements" --top-k 5
python scripts/search.py "human factors" --corpus fda_guidance --sector device --status final
python scripts/evaluate_retrieval.py          # Recall@K / MRR on data/evaluation/datasets/retrieval_v1.jsonl
pytest -m integration                         # live-database tests (read-only)
```

Set `REGULUS_DATABASE_URL` in `.env` first (see `.env.example`).

## 7e. Lexical RAG (retrieval + Qwen)

Stage 6 adds `regulus.rag`: `RagPipeline` takes a question, runs the
`LexicalRetriever`, builds a bounded, source-delimited context (8 sources /
1,200 chars each / 3,600 chars total), asks Qwen with a short grounding
prompt, and returns the answer with citations taken from the retrieval
results, never from the model. When nothing relevant is retrieved it returns a
deterministic "not enough information" answer without calling the model.
Design, prompt, benchmark methodology and baseline-vs-RAG comparison:
[`docs/lexical_rag.md`](docs/lexical_rag.md).

```bash
python scripts/rag.py "What does 21 CFR 820.10 cover?"
python scripts/rag.py "design controls" --corpus ecfr --debug   # sources, context, generation details
python scripts/evaluate_baseline.py --pipeline rag             # same 27 questions through RAG
python scripts/compare_runs.py <baseline.json> <rag.json> --answers
pytest -m "slow and integration" tests/integration/test_rag_slow.py
```

## 8. Tests

Fast unit tests (no model weights are loaded):

```bash
pytest
```

They cover configuration loading and validation, environment-variable
overrides, model-id validation, device/dtype resolution, tokenizer behaviour,
chat history, and the evaluation infrastructure (dataset schema, runner
failure handling, metadata, aggregation, persistence, report rendering) using
a fake generation function. The tokenizer tests read the tokenizer from the local cache and are
skipped automatically if the model has not been downloaded yet.

Full model load + generation (slow, excluded by default):

```bash
pytest -m slow
```

Database integration tests (need `REGULUS_DATABASE_URL`; excluded by default;
they assert the server rejects writes and that write counters do not change):

```bash
pytest -m integration
```

## 9. Project layout

```
.
├── configs/
│   └── model.yaml            # model / generation configuration
├── data/
│   ├── raw/                  # (later stages)
│   ├── processed/            # (later stages)
│   └── evaluation/
│       ├── datasets/         # baseline_v1.jsonl, retrieval_v1.jsonl (tracked in git)
│       └── results/          # evaluation run outputs (gitignored)
├── docs/                     # database_architecture, embedding_provenance, lexical_retrieval, lexical_rag
├── logs/
├── models/                   # Hugging Face cache (gitignored)
├── scripts/
│   ├── chat.py               # interactive multi-turn terminal chat
│   ├── check_env.py          # environment + resolved config report
│   ├── download_model.py     # fetch weights into models/
│   ├── compare_runs.py       # side-by-side comparison of two benchmark runs
│   ├── evaluate_baseline.py  # run the benchmark (--pipeline baseline | rag)
│   ├── evaluate_retrieval.py # Recall@K / MRR for the lexical retriever
│   ├── rag.py                # lexical RAG CLI
│   ├── report_baseline.py    # summarise a saved result file
│   ├── run_inference.py      # end-to-end smoke test
│   └── search.py             # manual retrieval CLI (read-only)
├── src/regulus/
│   ├── config/               # settings loading (YAML + env overrides)
│   ├── model/                # device detection, tokenizer/model loading
│   ├── inference/            # generation + in-memory chat session
│   ├── evaluation/           # dataset schema, runner, results, report, retrieval metrics
│   ├── retrieval/            # read-only PostgreSQL access, lexical retriever, citations
│   └── rag/                  # context builder, prompt, generator wrapper, RagPipeline
├── tests/
│   └── integration/          # live-database tests (pytest -m integration)
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 10. Dependencies

Pinned in [`requirements.txt`](requirements.txt): `torch`, `transformers`,
`huggingface_hub`, `accelerate`, `safetensors`, `psycopg[binary]` (Stage 5,
PostgreSQL driver), `pyyaml`, `python-dotenv`, `pytest`. Nothing else is
installed.
