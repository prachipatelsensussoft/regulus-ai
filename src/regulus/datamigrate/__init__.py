"""Ordered, version-controlled data migrations of the eCFR records table (cfr_records).

The schema belongs to the backend (Prisma, schema `public`). This package changes rows only: text the original
ingestion dropped, restored, and paragraph paths it numbered wrongly, corrected - the repairs made on the reference
database from 21 Oct 2026 on, so that every developer database restored from the same 15 Sep 2026 baseline reaches
the same rows. Each migration is one JSON file under data/migrations/cfr/ (regulus.datamigrate.payload) holding every
row it inserts and the exact old and new values of every row it updates, and the fingerprint of the whole table
before and after it.

The runner (regulus.datamigrate.runner) applies a migration only when the table's fingerprint is the one it was built
from, re-checks every old value inside the transaction, verifies the fingerprint after, and records it in a ledger kept
in its own schema (`regulus_data`), never in Prisma's. Nothing runs at application start-up: `python -m regulus.setup`
previews and asks before it writes.
"""

from regulus.datamigrate.payload import Migration, MigrationFileError, load_migrations
from regulus.datamigrate.runner import LEDGER_SCHEMA, MigrationError, Status, apply, adopt, fingerprint, rollback, status

__all__ = ["LEDGER_SCHEMA", "Migration", "MigrationError", "MigrationFileError", "Status", "adopt", "apply",
           "fingerprint", "load_migrations", "rollback", "status"]
