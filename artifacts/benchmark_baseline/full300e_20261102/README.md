# Locked baseline: 300-question benchmark, run full300e (2 Nov 2026)

The regression baseline for every accuracy change. Nothing here is edited after the lock; a new baseline goes in a new folder.

- `Regulus_AI_Acceptance_Benchmark_REV02_300Q_RESULTS_2026-11-02.xlsx`: the results workbook exactly as received (sha256 `8f48fa55a542f5fb9c7c4bfc1ee33a14a2e4e1a90c4f972d0112eedace72b43e`).
- `baseline.json`: each question's verdict, rubric scores, answer path, sections supplied and answer hash; the run
  identity; the gate figures; and the protected list (the 184 PASS and PARTIALLY PASS rows).
- Question Bank (the official key, File 2): `Regulus_AI_Acceptance_Benchmark_REV02_300Q(Question Bank).xlsx`, sha256 `42414c73306e08b61a5468276f383170d82bca8ce1fa77bb431f6e362d864e22`.
  Its questions, expected citations and Key Answer Elements equal the results workbook's on all 300 rows.

Result: 161 PASS, 23 PARTIALLY PASS, 89 PARTIALLY FAIL, 27 FAIL;
PASS + PARTIALLY PASS 184 of 300 (61.3%).

Check a new run against it (exit status 1 on any protected regression or unrun question):

    python scripts/benchmark_regression.py --new <scoring/verdicts.json or results .xlsx>

Limits: the code that produced this run (4b816e6) is on the Mac only, so this repository cannot reproduce it until that
commit is pushed. The answer key is the draft key (column K blank). The scorer was one blind Sonnet scorer.
