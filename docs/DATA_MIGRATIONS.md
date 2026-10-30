# Database data migrations (`python -m regulus.setup`)

The eCFR records table (`cfr_records`) on the reference development Mac was repaired several times between 21 and
30 Oct 2026. Some text the original ingestion dropped was restored, and some paragraph paths it numbered wrongly were
corrected. A database restored from the 15 Sep 2026 baseline backup has none of these repairs.

This repository carries every repair as an ordered, checksummed migration. One command brings any developer database
to the same rows, row for row, on Windows or macOS. It needs only Python and psycopg; `psql` is not used.

The schema is not touched. The backend owns it through Prisma, in schema `public`. These migrations change rows only.

## The migrations

| # | File (`data/migrations/cfr/`) | Applied on the reference DB | Change |
|---|---|---|---|
| 1 | `0001_restore_dropped_text_20261021.json` | 21 Oct | Insert 16 rows of dropped text (cosmetics parts); renumber 135 ordinals |
| 2 | `0002_restore_dropped_text_20261022.json` | 22 Oct | Insert 128 rows (tables, Part 740 warnings); renumber 459 ordinals |
| 3 | `0003_restore_title21_text_20261023.json` | 23 Oct | Insert 525 rows across Title 21; renumber 2,192 ordinals. This is the net of the apply, two partial rollbacks and the Part 201 re-apply |
| 4 | `0004_reserved_parent_paths_20261026.json` | 26 Oct | Correct 31 paragraph paths that were nested under a reserved-range row |
| 5 | `0005_paragraph_path_corrections_20261028.json` | 28 Oct | Correct 988 paragraph paths against the eCFR paragraph ids |
| 6 | `0006_definition_paths_20261030.json` | 30 Oct | DEF1: restart the numbering at each definition (1,162 rows in 47 sections) |
| 7 | `0007_restore_dropped_text_20261030.json` | 30 Oct | Insert 701 rows (remaining parts; large tables in captioned parts); renumber 2,879 ordinals |

In total, migrations 1–7 insert 1,370 rows and update 7,846. They delete nothing.

Every file holds:
- every inserted row, with all its columns;
- the old and new `(paragraph_path, path_ordinal, ordinal)` of every row it updates;
- a fingerprint of the whole table before and after the migration;
- its provenance: the repair records it was built from, with their SHA-256, and the eCFR part payloads behind it.

No other file or download is needed to apply them.

## Before you start

1. **Python environment**, as in the README (`pip install -r requirements.txt` and `pip install -e .`).
   psycopg's binary wheel covers Windows and macOS.
2. **A database restored from one of the two backups below**, on PostgreSQL 12 or later (18 recommended), with
   UTF8 encoding and the pgvector extension:
   - the 15 Sep 2026 baseline (`regulus db backup.sql`): `setup` applies all seven migrations;
   - or the 30 Oct 2026 Mac dump (`regulus_live_2026-10-30.dump`; Windows steps in its `RESTORE_ON_WINDOWS.md`).
     This dump already contains migrations 1–5 without a ledger: run `adopt` once, then `apply` adds 6 and 7.

   Restoring a backup is a one-time PostgreSQL step (pgAdmin's Restore, `pg_restore` or `psql`); `setup` never
   creates, drops or restores a database.
3. **`REGULUS_DATABASE_URL` in `.env`** (see `.env.example`), pointing at that database. The role needs INSERT,
   UPDATE and DELETE on `cfr_records` and CREATE on the database for the ledger schema (`postgres` has both).
   Alternatively, pass `--database-url` on every command.

## Commands

Run every command from the project root with the virtual environment's Python.

On Windows: `.\.venv\Scripts\python -m regulus.setup ...`

On macOS: `.venv/bin/python -m regulus.setup ...`

```text
python -m regulus.setup status            read-only: connection, ledger, fingerprint, pending migrations, semantic index
python -m regulus.setup                   = apply: check, preview, ask, apply, offer the semantic rebuild, verify
python -m regulus.setup apply --to 3      apply only up to version 3
python -m regulus.setup adopt             record migrations a copied database already contains (no row changes)
python -m regulus.setup rollback --to 0   undo migrations newer than 0 (= back to the 15 Sep baseline), newest first
python -m regulus.setup semantic          rebuild the semantic corpus and index from the database
```

Every command that writes first shows what it will do, then asks you to type the database's name. For a script or
CI, pass `--confirm <dbname>` instead. Without one or the other, nothing is written.

### What a normal first run looks like

```text
[1] Migrations in this checkout
  7 files in data/migrations/cfr, checksums read
[2] Connection
  database regulus on localhost:5432 as postgres (PostgreSQL 18.6)
  schema: Prisma, latest migration 0041_acceptance_benchmark (not changed by setup)
[3] Baseline and ledger
  cfr_records: 65341 rows, fingerprint 50597016a537e3a6
  ledger regulus_data.cfr_migrations: none recorded
  state: 7 pending
[4] Preview
    0001_restore_dropped_text_20261021: insert 16 rows, update 135 rows (...)  [2026-10-21]
    ...
  total: 1370 rows inserted, 7846 rows updated; no row deleted; each migration is one transaction
Type the database name (regulus) to apply 7 migrations, or press Enter to cancel: regulus
[5] Apply
  applied 0001_restore_dropped_text_20261021: 16 inserted, 135 updated, verified
  ...
[6] Semantic index
[7] Verify
  ledger: versions 1-7; state up_to_date; fingerprint f17a6ba9b8d725e6
```

The seven migrations take about 25 seconds. A rerun prints `Nothing to apply` and writes nothing.

### States `status` can report

| State | Meaning | What to do |
|---|---|---|
| `N pending` | The table is exactly the baseline plus the recorded migrations | Run `python -m regulus.setup` |
| `UP TO DATE` | Every migration is applied | Nothing |
| `ALREADY CONTAINS migrations 1-k without a ledger` | A copy of a database the migrations were applied to (the 30 Oct dump: 1-5; the reference Mac: 1-7) | Run `adopt`, then `apply` |
| `INCOMPATIBLE` | The table is neither the baseline nor any migration's result, or it changed after a migration was recorded | Nothing is written. Restore the baseline into a new database; do not "fix" rows by hand |
| `LEDGER ERROR` | A migration file differs from the one recorded when it was applied, or the database has a migration this checkout lacks | Update your checkout. Never edit an applied migration file |

## Safety: what the runner guarantees

- **Nothing runs on its own.** The application never calls the runner at start-up. Only the commands above write,
  and only after confirmation.
- **No reset, no drop, no truncate, no overwrite of other data.** A migration touches only the rows it names. It
  inserts rows that must not exist yet, and updates rows whose current values must equal its recorded old values.
- **One transaction per migration.** The transaction takes a runner-wide advisory lock and a SHARE ROW EXCLUSIVE
  lock on `cfr_records`, so readers (the running app) keep reading and no other writer can interleave. It then:
  1. re-checks the ledger and the whole-table fingerprint, and every old value;
  2. writes the changes;
  3. checks the fingerprint after the change and every inserted row, column by column;
  4. only then records the migration in the ledger.

  Any difference raises an error, and the transaction is rolled back with nothing changed.
- **No migration runs twice.** The ledger's primary key is the version. The runner applies only versions after the
  last recorded one, and only when the table's fingerprint is the expected one.
- **Separate ledger.** The ledger is kept apart from Prisma's tables: `regulus_data.cfr_migrations` holds one row
  per applied or adopted migration (checksum, fingerprints, time, role, client), and
  `regulus_data.cfr_migration_events` holds the append-only history of apply, adopt and rollback. Prisma manages
  `public` only and never sees the `regulus_data` schema.
- **Rollback.** `rollback --to N` undoes migrations newest first, one transaction each. It deletes the rows a
  migration inserted, only if they are unchanged and nothing references them (for example a saved answer citation),
  and restores the old values. Then it verifies the fingerprint.
- **Fingerprint.** SHA-256 over each row's id, section, paragraph path, path ordinal, ordinal, content hash,
  body-text MD5, reserved flag and removal date, in id order. It is independent of server locale, collation,
  DateStyle and encoding, so Windows and macOS servers compute the same value.

## Semantic corpus and index

The semantic index (`data/evaluation/semantic/semantic_expanded_v1/generated/`, gitignored) is built from the
database, so it is stale once rows change.

After applying migrations, `setup` checks the stamp it writes (`regulus_setup_stamp.json`, which holds the table
fingerprint the index was built from) and offers a rebuild:
1. It copies the current `generated/` to `generated_backup_<UTC time>/`; nothing is deleted.
2. It runs `scripts/build_semantic_corpus.py`.
3. It runs `scripts/build_semantic_index.py --reuse-from <backup>`, which embeds only new or changed items with the
   pinned `BAAI/bge-small-en-v1.5`. The model is downloaded once from Hugging Face at a pinned revision if it is
   not in the model cache.
4. It runs the index `--check`.

Without a previous index, the first build embeds about 29,000 items and takes much longer on CPU.

The paragraph and section vector caches (`data/processed/paragraph_dense/`, `section_dense/`) are keyed by the text
they embed, so they re-embed changed paragraphs by themselves the next time they load. To prebuild them:
`python -m regulus.retrieval.paragraph_dense`.

## Provenance (optional check)

Each migration's `provenance.ecfr_payloads` lists, per part, the date-pinned eCFR versioner URL (2026-09-03, the
snapshot the baseline was ingested from) and the SHA-256 of the payload the repair used. To check them:

```text
python scripts/verify_migration_sources.py                       # downloads each part from ecfr.gov
python scripts/verify_migration_sources.py --local <folder> ...  # or checks part-<N>.xml copies on disk
```

On 30 Oct 2026, all 137 payloads matched the archived copies: Title 21 parts in
`~/regulus_eval_runs/title21_source_xml_20261022/` on the reference Mac, and cosmetics parts in
`artifacts/retrieval_repair_step{1,2}_*/source_xml/` in git. The eCFR API answered that machine's automated
requests with HTTP 406 at the time.

## For maintainers: how the files were built and verified

`scripts/build_cfr_migrations.py` built the files on the reference Mac, read-only, from the repairs' own records:
- the apply and rollback manifests;
- the repair and rollback SQL with their SHA-256;
- the table as it stands.

It reads the live table and undoes each repair in memory, newest first. A row a repair inserted must be present with
exactly the inserted text, and every value a repair set must be the current value; otherwise the build stops. The
build yields the table before each repair, and finally the baseline. `--check` rebuilds in memory and compares with
the committed files.

Independent evidence that the result is exact (30 Oct 2026):

| Check | Result |
|---|---|
| State after 2 vs the 22 Oct 18:15 snapshot dump (`cfr_records_before_title21_repair_20261022.dump`) | fingerprint equal (65,485 rows) |
| State after 1 vs the 22 Oct 15:12 snapshot dump | fingerprint equal (65,357 rows) |
| State after 5 vs the 30 Oct 11:17 full dump (restored into a scratch database) | fingerprint equal (66,010 rows), paths included |
| Runner rollback of that scratch database to 2, 1, 0: recorded corpus identities | 65,408/75fb6482, 65,280/4f0457d7, **65,264/9657a55c (15 Sep baseline)**, all equal |
| Runner apply 1–7 from that baseline vs the live database | every column of all 66,711 rows equal (embeddings and search vectors included; only created_at/updated_at differ) |
| Every inserted row's text vs its content_sha256 and the repair record | equal |

The 15 Sep backup file itself was no longer on the reference Mac, so a database restored from it was not compared
directly. `python -m regulus.setup status` on such a database reports `7 pending` exactly when its table matches
the reconstructed baseline (65,341 rows, fingerprint `50597016a537e3a6`). If it reports `INCOMPATIBLE`, stop and
report it.

**Adding a migration.**
1. Make the repair on a scratch copy, recording what it changes.
2. Extend `repairs()` in the builder with the record files.
3. Run the builder; it writes only the new file and refuses to overwrite existing ones.
4. Run `--check` and the tests, then apply the new file with `setup`.

Never edit or renumber a published file: the ledger stores each applied file's checksum, and any change stops every
command.

## Tests

`tests/test_datamigrate.py`:

```text
pytest tests/test_datamigrate.py                                   # file format, checksums, chaining (no database)
REGULUS_TEST_DATABASE_URL=postgresql://postgres:...@localhost:5432/<scratch db> \
  pytest -m integration tests/test_datamigrate.py                  # 10 database scenarios
```

The database tests create and drop only their own `regulus_mig_test_*` schemas. Never point
`REGULUS_TEST_DATABASE_URL` at a database you work in. The scenarios cover:
- the original baseline;
- partly applied;
- fully applied and re-run;
- an incompatible baseline;
- a table changed after a migration was recorded;
- adopt;
- rollback, and rollback blocked by a referencing row;
- a changed migration file;
- a failure after the writes that leaves nothing behind;
- the CLI's confirmation.
