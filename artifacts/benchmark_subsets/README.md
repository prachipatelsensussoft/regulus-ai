# Frozen benchmark subsets (300-question REV02 benchmark)

Fixed for the whole RAG improvement programme of 4 Nov 2026. A later run never redefines them.

| File | Questions | full300e PASS+PP | Use |
|---|---|---|---|
| `evidence_complete_159.json` | 159 | 114 (71.7%): 110 PASS, 4 PP, 37 PF, 8 FAIL | Protects evidence-complete answers. Every key section already reached the model, so a retrieval change can only lose ground here. |
| `complement_141.json` | 141 | 70: 51 PASS, 19 PP, 52 PF, 19 FAIL | Where retrieval gains have to be shown: `key_section_missing` (94) and `no_section_key` (47, the key names a part, an Act section or nothing). |

- **Rule:** a row is evidence-complete when every CFR section number in its "Expected citation (verified key)" appears in its "Sections supplied", in the locked full300e results workbook (`../benchmark_baseline/full300e_20261102/`).
- **Origin:** this is the `all_expected_supplied` field of the 3 Nov root-cause review. The recomputation gives the same 159 IDs: ids sha256 `c2056be34239142f…`.
- **Recompute and check:** `python scripts/build_benchmark_subsets.py --check`. `tests/test_benchmark_subsets.py` runs the check.
- **Regression check on one subset:**

      python scripts/benchmark_regression.py --new <verdicts> --subset artifacts/benchmark_subsets/evidence_complete_159.json
