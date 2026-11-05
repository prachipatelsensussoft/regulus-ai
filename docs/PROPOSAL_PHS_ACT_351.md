# Proposal: PHS Act § 351 (42 U.S.C. 262) as a second statute

5 November 2026. Status: **proposal, not built.** Recommendation: **defer** (see the end).

## What is missing

The statute store (`src/regulus/retrieval/statute.py`, built by `scripts/build_statute_store.py`) holds one Act: the
FD&C Act, 21 U.S.C. chapter 9, with 4,839 paragraphs from the official govinfo text. The biologics regulations
(21 CFR Parts 600–680) are issued under **PHS Act § 351**, which covers:
- biologics licences, 351(a);
- the definition of a biological product, 351(i);
- biosimilars and interchangeability, 351(k);
- recalls, 351(d).

The store does not contain § 351. Today the code works around this in two places:
- `programmes.py`: the biologics programme offers no Act sections.
- `section_promotion._OTHER_ACT`: a question naming the PHS Act gets no FD&C statute matching.

So "What does section 351(k) require for a biosimilar?" is answered from 21 CFR only, or declined.

## Measured demand

| Question set | Rows whose key cites PHS Act § 351 / 42 U.S.C. 262 |
|---|---|
| REV02 300Q (client keys) | 0 (the 5 biologics rows B-001–B-005 cite 600.10/600.11, 600.80, 601.2, 601.12, 606.x) |
| Cosmetics bank + follow-ups | 0 |

The benchmark gain is therefore **0 rows**. The value is for real users asking statute-level biologics questions
(licensure, biosimilars, interchangeability, exclusivity), which no current question set measures.

## What building it takes

1. **Data.** Fetch 42 U.S.C. 262 from govinfo (`USCODE-2024-title42`, chapter 6A, subchapter II, part F, sec262), with
   its URL and sha256 in the manifest, as for title 21. § 351 is long: 351(k) and 351(l) (the patent exchange) alone
   run to many thousands of words.
2. **Store schema.** Each row gains an `act` field (`fdca` / `phsa`), and lookups are keyed by (act, section). The
   label becomes "PHS Act § 351(k)(4) [42 U.S.C. 262(k)(4)]".
3. **Reference parsing.** Recognise "PHS Act § 351", "section 351 of the Public Health Service Act" and
   "42 U.S.C. 262". `_OTHER_ACT` must stop excluding the PHS Act once it is in the store.
   - Collision to handle: a bare "section 351" or "21 U.S.C. 351" today means FD&C **§ 501** (21 U.S.C. 351), not PHS
     § 351.
4. **Code that assumes one Act.** Today's FD&C references by file: `section_promotion.py` (about 50, including
   `_SIGN_REF`, `_USC_REF`, statute promotion and its budget), `pipeline.py` (21), `paragraph_dense.py` (9),
   `conversation.py` (8), `presentation.py` (7), and `validation.py`, `generation.py` and the accuracy evaluator
   (5 or fewer each). The validator's source identity must include the Act, so that "§ 351(k)" never matches FD&C 501.
5. **Routing.** `biologics.act_for()` returns PHS § 351 for licensure, biosimilar and interchangeability questions.
6. **Measurement.** No benchmark row exists, so a 15–20 question PHS set (licensure, 351(i) definition, 351(k)
   biosimilar data, interchangeability, exclusivity, recall) has to be written and keyed from the statute text, and
   scored blind with both scorers. In addition, every 300Q row that uses the FD&C statute must replay byte-identical
   (0 PASS lost).

Estimate: about 1–1.5 days including tests and measurement. The risk sits in step 4: the same code serves every FD&C
answer that passes today.

## Recommendation

Defer. The work is a sizeable change to a path that many PASS rows depend on, for a gain the benchmark cannot show
(0 of 300). Build it when either:
- the client adds biologics statute questions to the benchmark, or
- users ask PHS Act questions in practice.

Then start with step 6's question set, so the change can be measured.
