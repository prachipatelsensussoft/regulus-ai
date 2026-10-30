"""Developer database setup: `python -m regulus.setup` (regulus.setup.cli).

Brings a developer's database, restored from the 15 Sep 2026 baseline backup, to the data this repository expects -
the cfr_records migrations of regulus.datamigrate - and rebuilds the semantic index from it. Previews and asks before
writing; never runs on its own (the application does not call it at start-up).
"""
