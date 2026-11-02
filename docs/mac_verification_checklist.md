# Mac verification checklist: the Windows support changes

Who: a developer with the usual Apple Silicon set-up (MLX worker, `configs/ui.env` and `configs/llm_model.env` filled in
as before). Goal: confirm that (1) the Mac path is unchanged, (2) the Mac answers the 9 parity questions consistently, and
(3) how far the Windows answers (PyTorch on a GTX 1650 SUPER) agree with the Mac's. Allow about an hour, mostly waiting
for answers.

## 0. Preconditions: the comparison is only fair if these match

| Must match | How to check |
|---|---|
| Code | both machines on the same commit: the one carrying the Windows support changes (`git log -1`) |
| Database content | all seven data migrations applied on both machines: `python -m regulus.setup status` (run with an admin connection, see `docs/DATA_MIGRATIONS.md`) reports `UP TO DATE`, ledger versions 1-7, `cfr_records` 66,711 rows, fingerprint **`f17a6ba9b8d725e6`**. `compare` also prints the snapshot date (2026-09-03) |
| Model | `mlx-community/Qwen3-4B-Instruct-2507-4bit` at revision `50d427756c6b…` (the profile's `REGULUS_MLX_MODEL_PATH`) |
| Pipeline switches | `compare` prints "switches identical" or the difference |
| Runtime | the Mac profiles must **not** set `REGULUS_WORKER_RUNTIME` (default `mlx`) |

There is no committed Windows baseline: the 30 Oct Windows run predates the main sync and migrations 6-7, so it is not
comparable. The Windows side records a fresh run on the same commit and migrated database (step 3) and shares the file.

## 1. The Mac path is unchanged

```bash
.venv/bin/python -m pytest                                    # expect the 11 known failures (release note 38624a4), nothing new
.venv/bin/python -m pytest tests/test_mlx_backend.py tests/test_ui_model_selector.py \
    tests/test_llm_model_api.py tests/test_conversation_server.py tests/test_torch_worker.py   # all pass
scripts/start_ui.sh                                           # starts as before
curl -s http://127.0.0.1:8000/api/config                      # "device": "mlx-gpu", "dtype": "mlx-4bit-g64" (as before)
```

The startup log must read exactly as before: `Starting Qwen3-4B-Instruct-2507 (MLX 4-bit) in the MLX worker ...` and
`Answering model: ... MLX worker {...}`. Stop the UI with Ctrl+C.

The MLX `describe()` and model identity are byte-for-byte what they were, so existing reply caches still match: if you
use `REGULUS_REPLY_CACHE`, a re-run of questions answered before must report cached replies, not fresh ones.
`tests/test_torch_worker.py::test_the_mlx_description_keeps_exactly_its_former_keys` pins this.

## 2. Mac consistency: two identical runs

```bash
scripts/start_llm_model.sh                                    # in its own terminal
curl -s http://127.0.0.1:8100/llm-model/v1/ready              # {"status": "ready", ...}
.venv/bin/python scripts/platform_parity.py run --out parity_mac_run1.json
.venv/bin/python scripts/platform_parity.py run --out parity_mac_run2.json
.venv/bin/python scripts/platform_parity.py compare parity_mac_run1.json parity_mac_run2.json     # expect 9/9 identical
```

Windows was deterministic: identical answers across repeats and between the UI and the API.

## 3. Mac against Windows

On the Windows machine, on the same commit and migrated database, with the API running (`scripts\start_llm_model.ps1`;
the UI must be stopped on a 4 GB GPU):

```powershell
python scripts\platform_parity.py run --out parity_windows.json
```

Exchange the files, then on either machine:

```bash
.venv/bin/python scripts/platform_parity.py compare parity_mac_run1.json parity_windows.json
```

How to read each question. The middle column is what Windows gave on 30 Oct, before the main sync and migrations 6-7:
a reference for what to look at, not the expected result (migration 6 renumbers definition paths, so q5's paragraph
label may legitimately differ now).

| Question | Windows, 30 Oct (pre-migration) | Path | Interpretation of a difference |
|---|---|---|---|
| q5 drug product (210.3) | PARTIAL, verbatim definition | no model call | **must be identical**; a difference is a code or data difference, not arithmetic |
| q7 820.250 removed | SECTION_REMOVED | no model call | must be identical |
| q8 999.99 fictitious | NONEXISTENT_SECTION | no model call | must be identical |
| q1 820.10 | VERIFIED, eval fail (qualifier attribution) | model | wording may differ; citations and state should match |
| q2 OTC tamper-evident (211.132) | PARTIAL, eval fail | model | **check the known Windows error**: it attributed "prominently placed / unaffected if breached" to the feature instead of the labeling statement (211.132(c)(1)(ii)-(iii)). Does the Mac make the same mistake? |
| q3 mica in eye shadow | NOT_COVERED, eval fail | model + term rule | Windows quoted 73.2496(b) verbatim ("including cosmetics applied to the area of the eye") but gave no direct yes; docs/llm-model-api.md shows VERIFIED. What does the Mac give on this code and database? |
| q4 / q9 IND 7-day (312.32(c)(2)) | PARTIAL, eval pass, identical | model | the deadline (7 calendar days) and citation 312.32(c)(2) must match |
| q6 Part 11 closed systems (11.10) | VERIFIED, eval pass | model | (a)-(k) cited on both |

For every DIFFERENT model-path question, read both answers against the cited regulation text and note which (if either)
is wrong. Similarity close to 1 with the same citations and state is the expected effect of bfloat16 (MLX) against
float16 (CUDA) rounding. A different state, different citations or a regulatory error is a finding to report.

## 4. Optional: the model-backed integration tests

```bash
.venv/bin/python -m pytest -m integration tests/integration/test_ind_deadline_generation.py \
    tests/integration/test_synthesis_completeness_live.py
```

On Windows `test_a_comparison_of_the_two_deadlines_assigns_each_to_its_own_category` and
`test_a_complete_real_synthesis_still_passes` failed. Whether they pass on the Mac tells us if those failures come from the
Windows arithmetic or are common to both platforms.

## 5. What to send back

`parity_mac_run1.json`, `parity_mac_run2.json`, the output of both `compare` commands, and the step 1 and step 4 test
summaries. The records hold questions, answers and citations only: no keys or credentials.
