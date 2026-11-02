# Regulus AI: local setup on Windows

How to run Regulus on a Windows PC. Mac developers keep using `README.md`, `scripts/start_ui.sh` and
`scripts/start_llm_model.sh`. Both platforms answer with the same model files, prompts, retrieval and response
format. Only the runtime under the model differs (MLX on a Mac, PyTorch here): see
[`docs/cross_platform_inference.md`](docs/cross_platform_inference.md).

## What runs where

| Component | URL / port | Start command |
|---|---|---|
| PostgreSQL 16 (the `regulus` database) | `localhost:5433` | Windows service `postgresql-x64-16` (starts at boot) |
| Answering web UI | http://127.0.0.1:8000 | `scripts\start_ui.ps1` |
| Answering API (llm-model) | http://127.0.0.1:8100/llm-model/v1/health | `scripts\start_llm_model.ps1` |
| Evaluation dashboard | http://127.0.0.1:8050 | `python scripts\serve_dashboard.py` |

There is no Node.js frontend. The web pages (`web/index.html`, `web/dashboard.html`) are served by the Python
servers above.

**On a 4 GB GPU, run the answering UI or the answering API, not both.** Each one loads its own copy of the model, and
two copies do not fit in 4 GB. The dashboard loads no model and can always run alongside either.

## Daily start (after a restart)

Open **PowerShell** in the project folder and run these in order.

```powershell
# 1. PostgreSQL: it starts by itself at boot. Check it is running:
Get-Service postgresql-x64-16
#    If Status is "Stopped", start it from an *Administrator* PowerShell:
#    Start-Service postgresql-x64-16

# 2. Go to the project (your clone's folder) and activate the virtual environment
cd <path-to-your-clone>\regulus-ai
.\.venv\Scripts\Activate.ps1
#    (if PowerShell refuses to run scripts: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned, once)

# 3. Answering web UI (keep this window open; wait for "Regulus test UI: http://127.0.0.1:8000")
powershell -ExecutionPolicy Bypass -File scripts\start_ui.ps1

#    OR, instead of the UI, the answering API (wait for ".../llm-model/v1/health")
#    powershell -ExecutionPolicy Bypass -File scripts\start_llm_model.ps1

# 4. Evaluation dashboard, in a second PowerShell window (activate the venv there too)
python scripts\serve_dashboard.py --port 8050
```

Then open http://127.0.0.1:8000 (UI) and http://127.0.0.1:8050 (dashboard).

**Database migrations:** nothing to run on a normal day; the application only reads the database. After a `git pull`
that adds files under `data/migrations/cfr/`, run the data migrations once, as described in
[Database data migrations](#database-data-migrations-python--m-regulussetup) below. The UI's chat history is a local
SQLite file (`data/chat/`) that creates itself.

### Stopping

- UI, API, dashboard: press **Ctrl+C** in their window. The model worker stops with the server.
- If a window was closed and a port is still taken, find the process and stop it:
  ```powershell
  Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object OwningProcess
  Stop-Process -Id <OwningProcess>
  ```
- Leave the virtual environment: `deactivate`
- PostgreSQL normally stays running. To stop it (Administrator PowerShell): `Stop-Service postgresql-x64-16`

## Configuration

| File | Purpose | In git? |
|---|---|---|
| `.env` | `REGULUS_DATABASE_URL` (the database connection) | no |
| `configs/ui.env` | UI model profile (copy of `configs/ui.env.example`, Windows section) | no |
| `configs/llm_model.env` | API profile: API key(s), port, model (copy of `configs/llm_model.env.example`) | no |
| `configs/model.yaml` | shared model/generation defaults | yes |

**Database connection** (`.env`):

```
REGULUS_DATABASE_URL=postgresql://regulus_ro:<password>@localhost:5433/regulus
```

Host `localhost`, port `5433`, database `regulus`, user **`regulus_ro`**: a least-privilege login that can only read the
regulatory content (the CFR, FDA guidance and ISO tables and the crosswalk view). It cannot write, cannot create tables,
cannot read the user, session, API-key or billing tables, and its sessions are read-only by default. Its long random
password is in your local `.env` only and is never committed.

Administration (restores, the dropped-text repair script, pgAdmin) uses the `postgres` superuser, which is not stored in
`.env`. In pgAdmin the server is registered as "Postgres Enterprise Manager".

Creating the role on another machine (as `postgres`, with a password of your own):

```sql
CREATE ROLE regulus_ro LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION CONNECTION LIMIT 20 PASSWORD '<long random password>';
ALTER ROLE regulus_ro SET default_transaction_read_only = on;
GRANT CONNECT ON DATABASE regulus TO regulus_ro;
GRANT USAGE ON SCHEMA public TO regulus_ro;
GRANT SELECT ON public.cfr_cross_references, public.cfr_definitions, public.cfr_iso_crosswalk, public.cfr_lexeme_stats,
  public.cfr_parts, public.cfr_records, public.cfr_sections, public.cfr_subchapters, public.cfr_subparts,
  public.fda_guidance, public.fda_guidance_anchors, public.fda_guidance_chunks, public.iso_clauses,
  public.legacy_qsr_terms, public.usable_crosswalk TO regulus_ro;
```

**Windows-specific model settings** (already in `configs/ui.env` and `configs/llm_model.env`):

| Variable | Value | Why |
|---|---|---|
| `REGULUS_MODEL_BACKEND` | `mlx` | the application's one-model worker path (same as the Mac) |
| `REGULUS_WORKER_RUNTIME` | `torch` | run that worker with PyTorch, since MLX exists only on Apple Silicon |
| `REGULUS_MLX_MODEL_PATH` | `models/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d4277…` | the same model files as the Mac |
| `REGULUS_MODEL_ID` / `REGULUS_MODEL_CACHE_DIR` | `Qwen/Qwen3-1.7B` / `models` | the tokenizer (byte-identical to the model's; checked at start) |
| `REGULUS_TORCH_DEVICE` | `auto` | the NVIDIA GPU when CUDA is available, else CPU (very slow) |
| `PYTHONUTF8` | `1` | Windows otherwise reads text files as cp1252 |

The API key in `configs/llm_model.env` (`LLM_MODEL_API_KEYS=<id>:<secret>`) is required as
`Authorization: Bearer <secret>` on `/answer` and `/info`. Generate a new secret with
`python -c "import secrets; print(secrets.token_urlsafe(32))"`.

## Database data migrations (`python -m regulus.setup`)

The eCFR records (`cfr_records`) are brought to the repaired data the code expects by ordered, checksummed data
migrations in `data/migrations/cfr/`. They insert and update rows only, never delete, and never touch the schema (the
backend's Prisma migrations own it). Full reference: [`docs/DATA_MIGRATIONS.md`](docs/DATA_MIGRATIONS.md).

**An admin database connection is required.** `regulus.setup` reads Prisma's `_prisma_migrations` table, keeps its
ledger in its own `regulus_data` schema, and writes `cfr_records`. The application's `regulus_ro` role can do none of
that by design: with `.env` alone, even `status` stops with `permission denied for table _prisma_migrations`. Use the
`postgres` superuser (or another role with those rights) **for these commands only**, and never store it in `.env`.
A variable set in the PowerShell window takes precedence over `.env`, and lasts only for that window:

```powershell
$env:REGULUS_DATABASE_URL = "postgresql://postgres:<postgres password>@localhost:5433/regulus"   # this window only
python -m regulus.setup status          # read-only: ledger, fingerprint, pending migrations
```

Then act on the state `status` reports:

| State | Do |
|---|---|
| `UP TO DATE` | nothing |
| `N pending` | back up (below), then `python -m regulus.setup` (shows a preview and asks you to type the database name) |
| `ALREADY CONTAINS migrations 1-k without a ledger` (for example the 30 Oct 2026 Mac dump: 1-5) | back up, then `python -m regulus.setup adopt` (records them, changes no row), then `python -m regulus.setup` |
| `INCOMPATIBLE` or `LEDGER ERROR` | stop: nothing is written. Read `docs/DATA_MIGRATIONS.md` and ask before doing anything |

Back up before any command that writes (`pg_dump` from PostgreSQL 16 or newer):

```powershell
New-Item -ItemType Directory -Force "$HOME\regulus_db_backups" | Out-Null
& "C:\Program Files\PostgreSQL\16\bin\pg_dump.exe" -h localhost -p 5433 -U postgres -d regulus -Fc `
    -f "$HOME\regulus_db_backups\regulus_before_migrations_$(Get-Date -Format yyyyMMdd_HHmmss).dump"
```

After applying, `setup` offers to rebuild the semantic index; the answering UI does not need it, and
`python -m regulus.setup --skip-semantic` skips the offer. Finish by running `status` again (it should report
`UP TO DATE`), then drop the admin connection from the window so everything else uses `regulus_ro` again:

```powershell
Remove-Item Env:REGULUS_DATABASE_URL
```

Expected result with the seven migrations of 30 Oct 2026: ledger versions 1-7, `cfr_records` 66,711 rows, fingerprint
`f17a6ba9b8d725e6`. Never edit or rename a file in `data/migrations/`, and never "fix" rows by hand.

## One-time setup (a new Windows machine)

Already done on this PC. Kept here for the next machine.

1. **Prerequisites:** Python 3.11 or newer (this PC uses 3.14), Git, PostgreSQL 16 with pgvector, and an NVIDIA
   driver that supports CUDA 12.6 (optional but strongly recommended).
2. **Git line endings:** clone with LF endings so files are byte-identical to the Mac's (several tests hash files):
   `git clone -c core.autocrlf=false <repo-url>`. `.gitattributes` also keeps the hashed fixtures, datasets and eCFR
   source payloads byte-exact whatever `core.autocrlf` says.
3. **Python environment** (from the project folder):
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r requirements-windows-gpu.txt     # NVIDIA GPU: CUDA torch + Triton
   pip install -e .
   ```
4. **Models** (into `models\`, about 6 GB, gitignored):
   ```powershell
   $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
   python -c "from huggingface_hub import snapshot_download as d; d('mlx-community/Qwen3-4B-Instruct-2507-4bit', revision='50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b', cache_dir='models')"
   python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-1.7B', cache_dir='models', allow_patterns=['tokenizer*','vocab.json','merges.txt','config.json','generation_config.json'])"
   python scripts\fetch_rerankers.py        # bge-m3 reranker used by the answering pipeline
   python scripts\download_model.py         # Qwen3-0.6B, for the command-line tools (scripts\rag.py, chat.py)
   ```
5. **Configuration:** create `.env` with `REGULUS_DATABASE_URL`, copy `configs\ui.env.example` to `configs\ui.env`
   and `configs\llm_model.env.example` to `configs\llm_model.env`, and apply their Windows sections (relative paths
   such as `models/...` are fine).
6. **Database:** restore the backup into PostgreSQL, create the `regulus_ro` role (SQL above), then run the data
   migrations with an admin connection ([Database data migrations](#database-data-migrations-python--m-regulussetup)).
7. **Check:** `python scripts\check_env.py` (should list `cuda_available: True`), then start the UI.

## Command-line tools and tests

```powershell
python scripts\search.py "21 CFR 820.10"                       # retrieval only (database)
python scripts\rag.py "What does 21 CFR 820.10 cover?"         # lexical RAG with Qwen3-0.6B
$env:PYTHONUTF8 = "1"; pytest                                  # unit tests (about 4.5 min)
pytest -m integration                                          # database tests (the database must be running)
```

For the integration tests that start the answering model, load the UI profile first, and stop the UI or API so the GPU
is free:

```powershell
. .\scripts\load_profile.ps1 configs\ui.env; pytest -m integration
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `connection refused` on port 5433 | PostgreSQL is stopped: `Start-Service postgresql-x64-16` (Administrator) |
| `password authentication failed` | check `REGULUS_DATABASE_URL` in `.env` |
| `permission denied for table _prisma_migrations` from `python -m regulus.setup` | it ran with `regulus_ro` from `.env`: set the admin connection for that window first (see *Database data migrations*) |
| `permission denied for table ...` (in the app) | the app was pointed at a table outside `regulus_ro`'s grants: add a `GRANT SELECT` for it (as `postgres`), never switch the app to the superuser |
| `... cannot start: ... MLX needs an Apple Silicon Mac` | `REGULUS_WORKER_RUNTIME=torch` is missing from the profile |
| `cannot start: ... tokenizer ... is not byte-identical` / not cached | run the tokenizer download in step 4 (the UI starts with `--offline`) |
| `memory preflight: ... MB available` | close large applications (the check needs `REGULUS_MLX_MIN_AVAILABLE_MB` of free RAM) |
| answers suddenly take many minutes | another program is using the GPU, or the UI and API are both running: stop one |
| `check_env.py` shows `cuda_available: False` | the CPU torch is installed: `pip install -r requirements-windows-gpu.txt` |
| `UnicodeDecodeError ... cp1252` | set `PYTHONUTF8=1` (the start scripts do it) |
| port already in use | see *Stopping* above |
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

## Expected speed on this PC (GTX 1650 SUPER 4 GB)

The model loads in about 10 s. A question takes about 1.5 to 3 minutes: about 11 s per 2,000 prompt tokens, then 7 to
9 tokens per second. The Mac documents 40 to 150 s. Answers come from the same weights and rules as on the Mac,
but GPU rounding differs, so an answer can occasionally differ in wording. Every answer records
`"runtime": "torch"` and the device, so Windows and Mac results are never mixed up.
