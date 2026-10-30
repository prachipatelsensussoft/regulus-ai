Data migrations of `cfr_records`, applied in version order by `python -m regulus.setup` (see docs/DATA_MIGRATIONS.md).

Written once by `scripts/build_cfr_migrations.py`; never edit, rename or renumber a file. Each file's checksum is
recorded in the database ledger (`regulus_data.cfr_migrations`) when it is applied, and a changed file stops the runner.
Line endings are kept exactly (`.gitattributes`: `data/migrations/** -text`).
