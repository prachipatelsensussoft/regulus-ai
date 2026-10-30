"""Validator precision switches (23 Oct 2026, holdout failure analysis, fix A).

Correct drafts were withheld by validator findings that are themselves false (9 of 63 non-PASS holdout answers).
REGULUS_VALIDATOR_PRECISION lists the fixes switched on, e.g. "A1,A2,A3"; the default is every fix that passed the gates,
"A1,A2,A3,A4,G1,S1,E1,C82,P1" since 26 Oct; "off" keeps the validator exactly as it was before them. Each
fix ships only after the exact replay of the recorded answers (holdout PASS rows unchanged or re-scored PASS, the named
misleading drafts still withheld) and a blind answer A/B.
  A1  a CFR citation whose number the evidence carries is not a figure ("§74.705(a)(1)" read as "74"). FD&C Act
      section numbers are left as figures: on the blind A/B (23 Oct) A1 over both released one misleading draft whose
      only finding was the false figure 605 (FDCA605-Q1, an invented reporting route); CFR-only keeps it withheld and
      still gains 82.1602 (exact replay: 6 answers changed, each already blind-scored or re-scored).
  A2  a scoped absence statement ("these provisions set no limit on the amount") is checked against the cited
      text's limiting terms instead of word overlap; unscoped denials stay errors
  A3  whole sentences that carry no citation and fail support (or repeat a supported sentence) are dropped and the
      rest re-validated, before the draft is handed back to the model; never on a truncated draft or one whose verdict
      says the evidence does not govern the question
  A4  a citation marker re-pointed to the one supplied source that IS the provision the sentence names first
      ("21 CFR 71.30(b) requires ... [1]" with [1] = 71.30(a) and [2] = 71.30(b)), when its words are that source's
Blind A/B (23 Oct, 170 recorded answers replayed, two scorers + adjudicator): A2+A3+A4 gained 3 PASS (70.10, 73.2120,
FDCA201(i)), lost none, released no misleading draft; A1 is held (above).
Answer post-processing fixes share the switch (failure analysis fix G):
  G1  duplicate removal keeps a repeated sentence that is part of a passage quoted verbatim from a supplied source
      (740.17: the full caution quoted twice lost three of its five sentences the second time)
  E1  a duty its cited source only requests or recommends ("is required to file" for 720.1's "is requested to file")
      is an error, and the draft goes back to the model with the source's own word (regulus.rag.claim_validation)
  S1  a cited provision's current status that its text does not carry (710/720 program closed, 740.18 stayed, no 606
      rule) is appended verbatim from data/reference/status_notes_v1.json (regulus.rag.status_notes)
  C82 a Part 82 certification listing that is one conformity sentence is answered from its text and the subpart's
      general section (regulus.rag.color_listing.compose_certification); for an FD&C (subpart B) certification
      question 82.5 joins as a companion and the answer says its 82.5(b)(1) limits govern; a provision named inside a
      verbatim quotation of the sentence's own cited source is not read as mis-cited (82.50 quoting "§ 82.5")
  P1  a cosmetics question about a manufacturing practice no general cosmetic provision and no cosmetics section of
      the Act mentions (calibration, pest control, cleaning SOPs, equipment certification - a search of the database
      text decides) is answered from FD&C Act 606 and 601 (and 610 for records) (regulus.rag.practice_silence).
      26 Oct: fires on 5 of the 1,211 section-benchmark questions (UC-004/005/008/010/012); blind two-lens scoring
      PASS 0 -> 5 on both lenses, exact replay of 228 recorded answers byte-identical
Fallback fixes (26 Oct 2026; in the default since 27 Oct):
  Q1  an answer that cannot be shown (withheld after validation, or a model decline although the provision was
      supplied) on a question whose provision is identified - named by it or resolved for it - gives that provision's
      paragraphs sharing most subject words with the question, verbatim and cited, instead of "not shown"
      (regulus.rag.fallback.focused_quote)
  FP1 a confirmation question ("..., correct?", "Does X require a 24-month ...?") that attributes to the identified
      provision a figure none of the supplied text states is answered "No", with the provision's paragraphs stating
      figures of that kind, verbatim (regulus.rag.premise_figures)
  RS1 a records question about identified 21 CFR sections whose full text has no record-keeping language is answered
      "sets no recordkeeping requirement", with what the section requires, verbatim (regulus.rag.records_silence)
  D1  a question asking what terms mean is answered with the supplied paragraphs that define every one of them,
      verbatim and cited (regulus.rag.definitions)
  ST1 the FD&C Act text a requested regulation itself cites ("section 403(i) of the act") joins the evidence after the
      regulation's own, at most two Act sections, within the hard ceiling (regulus.rag.section_promotion.statute_references)
  QM1 a section of the former Quality System regulation (820.30, 820.100 ...) is said not to be in the current eCFR, the
      live part 820 text that carries its subject is quoted, and the ISO 13485:2016 clause on that subject is named by
      number and title with its basis; a non-existent ISO 13485 clause is said not to exist (regulus.rag.qmsr)
Promotion fixes (26 Oct 2026; C2, Q2, PR2 and PO2 in the default since 27 Oct, SP2 and SA1 off):
  SP2 the picker's second regulation joins the first as a companion when it is in the same part, outside the cosmetics
      programme (regulus.rag.section_promotion.second_in_part): 300-question keys need more than one section for 21
      of 35 recorded picks; a same-part second pick was a key section 11 of 14 times
  AU1 (28 Oct, off until measured) a general question is offered only the FD&C Act sections its top regulations' parts
      are issued under (their eCFR Authority line, data/reference/part_fdca_authority.json): SA1 offered 502 for a PMTA
      question (Part 1114 is issued under 910) and 761 for a structure/function claim
  AP1 (28 Oct, off until measured) the FD&C Act paragraphs supplied are chosen by the cross-encoder over the section's
      most similar paragraphs, not by word overlap: 910 is 13,500 characters and 2,500 fit beside a regulation
  MS2 (28 Oct, off until measured, needs MS1) MS1's pool also takes the promoted section's structure - its subpart's
      nearest sections and the sections of its part it cites - scored by the same cross-encoder, same floor and cap
      (regulus.rag.section_promotion.structural_neighbours): 66 of 148 answers lacking a key section had it there
  DQ1 (28 Oct, off until measured) the definition paragraph of a term the question asks about, from the promoted
      section's subpart (else its part), via cfr_definitions (regulus.rag.section_promotion.asked_definitions): V-001
      'importer' 1.500, F-002 'qualified facility' 117.3; one paragraph, never a whole definitions section
  BA1 (28 Oct, off until measured) the evidence budget is charged for what the prompt renders: no URL line (the grounded
      prompt has none; every block was charged ~95 characters for it)
  CB1 (28 Oct, off until measured) the required-evidence hard ceiling is 15,000 characters for the GPU model (was 9,000,
      set for the 0.6B model on CPU); chosen companions were dropped as hard_ceiling_chars (F-020 117.405) - use with HX1
  AU2 (28 Oct, off until measured) an FD&C Act section promoted alone brings the best-scored regulation candidate from a
      part issued under it (its Authority line): L-010 Act 403(w) + 101.4, P-002 Act 513(i) + 807.100
  RR2 (28 Oct, off until measured) RR1 redesigned: the first section keeps all its paragraphs first; only the sections
      after it share the rest round-robin (RR1 halved S-018's 111.210 list)
  XP1 (28 Oct, off until measured) the provision in another part that the promoted provision cites (820.1(d) -> 10.30),
      at most 1, whole within MS1_MAX_CHARS else the cited paragraph (M-007)
  NAV2 (28 Oct, off until measured, with NAV1) NAV1's table of contents also lists the 15 sections of the part outside
      the subpart closest to the question by paragraph embedding (S-005's key 111.610 is in the records subpart)
  NAV1 (28 Oct, off until measured) table-of-contents navigation: the promoted section's subpart headings are shown to
      the model, which names the other sections the question needs (at most 3, within 6,000 chars); 15 of 17 missing key
      sections in the first 33 dev questions were in the promoted section's subpart, scored low by the cross-encoder
  LS3 (28 Oct, off until measured) LS2's whole-list answer also for "What X requirements apply ... under Part N?"
      when the promoted section's lead-in introduces its list (S-006 111.15 (a)-(k)); offline: fires on 6 of 300 (4 PF,
      2 PASS at risk: S-023, S-024)
  RR1 (28 Oct, off until measured) round-robin budget across the requested sections: each one's best paragraph first,
      then each one's second (diversity-aware allocation, as MMR); M-029's 820.10 got only its lead-in behind 5 sections
  AG1 (28 Oct, off until measured) anchored guidance: a question asking beyond the regulation (framework, guidance,
      recommend, expect, Barr ...) also gets the passages of the guidance anchored to its requested sections, ranked by an
      OR query of its words (D-023: the OOS guidance is anchored to 211.192; the all-words guidance search found nothing)
  MS3 (28 Oct, off until measured, with MS1) small-to-big at paragraph level: a section MS1 would add but cannot fit
      whole brings the top-level paragraph family of its best-scored paragraph (L-008: 101.12's tables never fit whole)
  PC1 (28 Oct, off until measured) model cascade: when the cross-encoder's best regulation candidate scores at least
      REGULUS_PC1_MIN_SCORE and REGULUS_PC1_MARGIN ahead of the next, its two best are the picks and the language-model
      picker (14 s a question) is not called; otherwise the picker decides as before
  AR2 (28 Oct, off until measured) active retrieval: a draft that says the provisions do not specify something gets one
      more pass with the section of the promoted part that holds it (lexical rank in the part, cross-encoder at
      REGULUS_AR2_MIN_SCORE); the second answer is kept only if it validates, on the same answer path, with fewer such
      claims. 300Q: 47 model answers made one, 23% PASS+PP vs 43%; in 29 of the 37 that failed a key section was absent
  SP2O (28 Oct, off until measured) of the picker's two same-part picks, the one the cross-encoder rates at least
      REGULUS_SP2O_MARGIN higher leads the promotion: SP2 live, Z-001 answered from the first pick 900.4 with the key
      section 900.11 second
  SP2F (28 Oct, off until measured) SP2's second pick stands only when its cross-encoder score reaches
      REGULUS_SP2F_MIN_SCORE: SP2 live 28 Oct, the 3 losses in 29 changed answers were second picks outside the key
  SN1 (28 Oct, off until measured) a question naming "Part N Subpart X" is answered from that subpart's candidates
      (regulus.rag.section_promotion.named_subpart): F-003 promoted 117.4 for 117.10, F-016 117.5 for 117.201, and SP2's
      second pick 117.135 cost F-004 (Subpart B named)
  PT1 (28 Oct, off until measured) a product the yes/no question names is not an unstated premise when a supplied source
      permits its category outside an exception (regulus.rag.claim_validation.covered_product_words): "Is mica permitted in
      eye shadow?" was declined because no source contains "shadow" though 73.2496(b) allows eye-area use
  FQ1 (28 Oct, off until measured) the focused verbatim quote takes up to 8 paragraphs / 4,000 characters of the
      requested provision (was 3 / 1,600): 300Q 27 Oct, 27 focused quotes scored 0 PASS+PP, 22 as "omits (b)-(g)"
  MS1 (28 Oct, off until measured) rerank-threshold-budget selection: the other candidates of the promoted section's
      part whose cross-encoder score reaches REGULUS_MS1_MIN_SCORE join it as companions, best first, at most
      REGULUS_MS1_MAX and REGULUS_MS1_MAX_CHARS (regulus.rag.section_promotion.reranked_companions); 300Q 27 Oct: 68
      wrong answers had a key section among the candidates and not in the prompt
  C2  a condition stated in the supplied parent or child of a cited paragraph ("FDA may withdraw a qualified facility
      exemption if:" in 117.251(a) for a claim citing (a)(1)) counts as stated in the cited source; a sibling does not
      (regulus.rag.claim_validation). 26 Oct: 397 recorded answers, 6 changed, blind scored FAIL->PASS 2 (U-003, M-004),
      FAIL->PARTIALLY FAIL 3, unchanged 1, none worse
  SA1 a question outside the cosmetics programme is offered, after the regulations, the 3 FD&C Act sections it is most
      about (lexical + paragraph-embedding order; 301 never), as a cosmetics question is offered chapter VI
      (regulus.rag.section_promotion.general_act_offered): 25 of 225 300-question dev rows are keyed to the Act; the key
      Act section is offered for 18 of the 19 general ones
  Q2  Q1's quotation for a section the picker chose, when the reranker scores it at least 0.6, under a lead that does
      not claim it is the provision the question concerns (regulus.rag.pipeline.Q2_MIN_RERANK). 26 Oct: 25 recorded
      withheld answers; without the score floor 3 PASS declines (must-refuse, false premise, out of scope) were lost
  PR2 the yes/no premise check ignores words used in 40 or more of Title 21's sections ("instead", "outer", "appear"):
      ordinary regulatory English is not a premise (regulus.rag.premise_vocabulary). 26 Oct: 397 recorded answers, 9
      changed; blind scored FAIL->PASS 4, FAIL->PARTIALLY PASS 1, unchanged 4, none worse
  PO2 a question naming only a 21 CFR part outside the cosmetics programme ("under Part 211") is resolved to its
      section(s) inside that part, as "Under Part 82, ..." is (regulus.rag.section_promotion.part_only_any): 74 of
      225 300-question dev rows; a key section ranks first inside the part for 57
  ND1 (27 Oct) the bare decline sentence ("The available regulatory sources do not provide enough information to
      answer this.") is removed from an answer that states something cited without it (regulus.rag.pipeline.
      drop_stray_decline): 9 of 154 answers of the 26 Oct full run were scored down for that contradictory line
Fixes of 27 Oct 2026 (first 50 cosmetics main questions, the 26 Oct follow-up verdicts):
  DF1 a question asking how FDA's definitions classify what it describes ("Are these 'color additives' in FDA's
      sense?") is resolved to the definition paragraphs of 70.3 / 700.3 / 82.3 it names, with the paragraph limiting
      a definition and the short definitions a matched one-word term uses (SectionPromoter.definitions_asked): the 5
      bank questions it fires on get exactly their key paragraphs; 0 of 300 acceptance questions
  EY1 an unnamed color whose listing the question says does not provide for eye-area use: 70.5(a), not a guessed
      listing
  MF1 "$75" is the source's "$75.00": a whole amount equals the same amount with zero decimals (figure check)
  E2  E1 for a sentence with no citation marker that names the section ("720.1 still requires a filing"), or as the
      answer's opening sentence says it of the supplied provisions
  CM1 a sentence naming a provision that its cited source itself names ("the fee for listing a diluent under
      § 80.35" citing 70.19(j)) is not a provision-citation mismatch
  LV1 "list"/"listing" asks for an enumeration only as an instruction or followed by what to enumerate: in Parts
      70-82 "petitioning to list a colorant" is approval, and the question was read as "every paragraph of 70.19"
      (64 of 1,962 bank, follow-up and 300Q questions change, all from enumeration)
  FU2 a demonstrative with a subject word ("this color", "that exemption") points back to the conversation unless the
      question introduced it itself; FU2B a follow-up of at most 3 subject words naming nothing continues the
      conversation (resolution harness, 1,000 dev follow-ups: anchored with the key's section 512 -> 523, false
      anchoring of a new main question 22 -> 22). FU3 (the follow-up's own top-ranked section as a companion) was
      measured and dropped: 55 added, 5 in the key. FU2C: shared subject words alone do not anchor a question of more
      than 12 subject words with no pointer back (false anchors 22 -> 20, no correct anchor lost)
  FU2D a color named in a conversation anchored to a general section (82.51 lakes, 81.30, 82.50) does not make the
      follow-up a new question unless it asks whether that color may be used (harness: +8 correct anchors, 0 false;
      all FU2 rules together: anchored with the key's section 512 -> 531, false anchors 22 -> 20 of 1,000)
  CS1 a color additive regulation (Parts 70-82, which carry out section 721) is not given MoCRA's 603-614 as a
      co-promoted Act section (5 of 16 Part 70 questions got 607/608/609, filling the budget)
  FS2 (off) follow-up subject words stemmed by PostgreSQL's Snowball English stemmer (the retrieval index's) instead of
      the suffix rule: right in itself, but on the harness +6 correct anchors for +6 false ones at today's thresholds
  RO1 the picker reads the regulation candidates reordered by reciprocal-rank fusion of lexical rank and the
      cross-encoder's score (reorder-then-pick); the veto reuses the same scores
  V2  a reranker veto in a near tie (within 0.05) takes the picker's own other choice instead of the reranker's top
  UR1 the listing composer reads the uses paragraph under each heading the listings use ("Uses and restrictions.",
      "Use and restrictions.", "Use and restriction."): carmine 73.2087 and annatto 73.2030 were answered by the model
      and annatto's answer denied the eye-area use its listing grants
  LC1 an opening listing question ("May we use X, and what applies?") the model answered gets the listing's paragraphs
      no citation of the answer points to, verbatim and cited (73.2125: the 0.1 percent limit said, (a), (c), (d) not)
  EX1 a source's "certification ... is not necessary" supports an answer's "exempt from certification"
  SP1 spelling correction against the regulations' vocabulary, SymSpell-style (regulus.retrieval.spelling): "carmin"
      -> "carmine", "anatto" -> "annatto". 277 bank questions with seeded typos: key section first 124 -> 131 (clean 134),
      in the top 5 189 -> 200 (202), color matched 78 -> 90 (95); 0 of 1,962 correctly spelled questions changed
  QX1 (off until measured) LLM query expansion: the model restates the question in the regulation's terms and the
      sections that ranks first join the picker's candidates (71.18 "petition for exemption" for "ask FDA to exempt")
  LM1 (off until measured) "lost in the middle": with 6+ sources the prompt names, before the evidence, the (at most 3)
      the cross-encoder scores closest to the question (70.19-Q2 read the new-petition fee (c) for an amendment, (i))
  CV1 (off until measured) Chain-of-Verification with an entailment model: each cited sentence of a validated answer is
      judged against its cited sources by MiniCheck-RoBERTa-Large (regulus.rag.entailment; Tang et al., EMNLP 2024);
      sentences below REGULUS_CV1_MIN_SUPPORT go back to the model once, the revision kept if it validates and its
      weakest sentence improves, else the unsupported sentences are removed when cited statements remain
  FR1 (off until measured) facts first (FaithfulRAG-style): the model lists what the sources state about the question,
      with source numbers and conditions, then answers from that list (one more generation)
  FC1 a Part 70/71 section whose heading names a subject a 70.19 fee paragraph prices (a two-word phrase used in at
      most 3 sections of the two parts) gets that paragraph as a companion: of all 40 sections, exactly 70.51 -> 70.19(l),
      (m) and 70.55 -> 70.19(n), the fees their keys name. RO1 measured 27 Oct on the 50: +3 picks, -3 (the reranker
      favours color listings for "exempt from certification" questions) - off; QX1 recovered nothing - off
  CT1 (off until measured) conversation topic memory: with no focus chosen and no deployment scope, a follow-up in a
      conversation anchored only to cosmetics provisions (Parts 70-82, 700-799, FD&C Act 601-614) is answered with the
      cosmetics focus
  SY1 (off until measured) "ask FDA to / for ..." with no "petition" adds "petition" to the search terms (domain
      synonym expansion): 71.18 was not among the candidates of "Can we ask FDA to exempt it from certification?"
  27 Oct measurements: SY1 (71.18 not a candidate -> first; 0 other changes) and CT1 (802 bank follow-ups read alone
      in a general deployment: key section in the top 5, 531 -> 643) are on. LM1 + FR1 together on 22 questions:
      +3 / -2 blind, answers 30% slower - off
  BN2 (off until measured) best of two with a verifier: a facts-first draft beside the plain one; the validated draft
      citing more of the requested paragraphs with fewer denials is delivered (plain on a tie), and a validated
      facts-first draft replaces a plain one that would be withheld
  FX1 (off until measured) dynamic answer example: one verified dev-split key of a similar question about another
      provision, section numbers removed, shown as an example of a complete answer's content (regulus.rag.answer_examples);
      never from the question's own provisions or the question itself, never a sealed question. The examples file is
      the client's keys and stays outside the repository (REGULUS_ANSWER_EXAMPLES)
  RO1 ON since 27 Oct 12:30: with color listings given no reranker credit when no color is named, 118 dev picker
      questions key first 44 (lexical) -> 59, MRR 0.489 -> 0.588; live 5 of 9 problem picks right (baseline 3).
      BN2 on 22: +1 / -1 by its own choice, answers 75% slower - off; FC1 confirmed live (70.51, 70.55 -> PASS)
  WC1 (off until measured) a question asking what a Part 740 warning or label must say is answered with the statement's
      paragraph named and the section's paragraphs verbatim (regulus.rag.warning_statements); "do we need it?" stays
      with the model
  TK1 an answer applying provisions gets at least 640 tokens (70.55-Q1 cut at 256, 740.2-Q1 at 416, both withheld); the
      allowance is not in the prompt, so with greedy decoding only answers that hit the old limit change
  E1S a prescribed label statement's own words ("Caution—... a physician should be consulted") are not the strength of
      the duty to bear it (740.12-Q1 withheld as "'must' where the source says 'should'")
  27 Oct stratified 50 (unseen, bank mix): 64% PASS+PP; listings 13/14, Part 82 8/8, labeling 6/15, FD&C Act 2/7
  DF2 a question classifying something under the FD&C Act's own definitions ("Is it a cosmetic under the FD&C Act
      definition?") is resolved to the defining paragraph, read from the Act text of 201, 604 and 608 ('The term "X"
      means'); MoCRA questions use 604 / 608. Fires on 4 of 1,962 bank, follow-up and 300Q questions, each its key
      paragraph (201(i), 604(2), 608(c)(1), 608(c)(2)); 201(i)-Q1 had gone to 604
  NL1 a listing that names the cosmetics it may color (74.2052, 74.2053: "the following cosmetics ...: Eyeliner, ...") is
      composed too: a named product is permitted, any other not (the model permitted D&C Black No. 2 in a hair gel)
  ST2 (off until measured) the S1 status note of a supplied provision is also shown to the model before it answers
  VB1 after a reranker veto the picker's second pick stands only if it is from the other corpus (bug: FDCA606-Q2's veto
      720.1 -> Act 606 kept Act 612 as a regulation, "21 CFR 612" was promoted and the answer declined)
  SY2 domain thesaurus for the search terms: "become public / keep confidential" -> public disclosure, confidentiality;
      "hide / disguise / cover up" -> conceal, deception. Fires on 7 of 1,962 questions; key section rank 71.22-Q1 14 -> 1,
      71.15-Q1 / Q2 / Q2-F4 2 -> 1, one anchored follow-up 1 -> 2
  PC1 (off until measured) after a validated answer from one requested 21 CFR section, its uncited paragraphs that the
      cross-encoder scores >= 0.5 for the question (at most 3) are appended verbatim and cited
  QD1 (off until measured) question decomposition (Question Decomposition for RAG, ACL SRW 2025): a multi-part question
      is split by the model into at most 3 sub-questions; each one's best regulation section by the cross-encoder
      (>= 0.5) joins the picked section as a companion, at most 2
  WC1 ON since 27 Oct 14:00: diagnostic 30 (unseen weak-family dev questions), both "what must the warning say"
      questions composed and blind-scored PASS (740.11-Q1, 740.17-Q1); the three withheld drafts there had real errors
      (invented duties, a mis-cited sub-paragraph), so the validator stays as it is
  ISO1 (measured gain, held back from the 27 Oct release) the ISO 13485 clauses that the cited Part 820 paragraphs (or their section's lead-in) name,
      and the answer does not, are added on one line with those citations (regulus.rag.iso_clauses); never guessed
  AF1 a section number a source states with its paragraph ("section 601(c) of the act") counts as supplied, so an
      answer's "§ 601" is not an invented figure (sealed 700.27-Q1 withheld over it)
  RS2 (off until measured) rescue: when the plain draft and its correction fail validation, a facts-first draft is
      tried once before the answer is quoted or withheld, and delivered only if it validates (a fallback chain; it
      never touches an answer that validated)
  LS1 (on since 27 Oct release) "what must X include / contain / show / record" about one requested 21 CFR section is
      answered with the lead-in paragraph that introduces the matching list and every paragraph under it, verbatim
      (regulus.rag.list_answers)
  SY3 MoCRA thesaurus: "small brand / under $1 million / annual sales" -> 612 (small businesses), "also an OTC drug /
      drug and cosmetic" -> 613 (products also subject to chapter V). 23 bank questions keyed to 605-613 that fire:
      Act rank better 14, same 8, worse 1 (2 -> 3); the 10 "small brand" listing questions keep their CFR rank
  RC1 (off until measured) rules as code for the MoCRA small-business exemption: a small-business question with FD&C
      Act 612 supplied gets the verdict from its facts (a product of 612(b) excludes; 612(a) covers only 364b and 364c)
      and 612's text verbatim (regulus.rag.mocra_rules)
  DV1 (off until measured) a question naming only a small 21 CFR part (<= 12 live sections; Part 820, the QMSR, has 6)
      or, naming none, ISO 13485 / the QMSR, is answered from that part's most relevant paragraphs: cross-encoder rank
      fused with shared two-word phrases, top 3 (regulus.rag.section_promotion.part_paragraphs). Dev device questions it
      handles: key paragraph among its picks 13 of 19 (the baseline read the whole part and missed the paragraph)
  QM2 (on since 27 Oct release) the former regulation's record types named in words - device master record, device history
      record, design history file, quality system record - are its former sections for QM1 (regulus.rag.qmsr.former_by_name)
  E2B (off until measured) an uncited duty sentence E2 would call strengthened stands when another supplied source states
      that duty itself with at least 40% of its words - E1's escape (regulus.rag.claim_validation.duty_stated)
  LS2 (on since 27 Oct release) "What does 21 CFR X require?" about a section whose own lead-in introduces its list (820.10
      "... must:" then (a)-(e)) is answered with LS1's whole list, verbatim (regulus.rag.list_answers)
  AR1 (off until measured) a figure the answer computes in plain sight ("2,000 pounds x $0.45 = $900") is supported when
      both operands are supplied figures and the arithmetic holds (regulus.rag.validation.verified_arithmetic)
  C3 (off until measured) a paragraph of the same section that names the cited paragraph ("unless an exemption pursuant
      to paragraph (b)") counts as cited for the condition checks, as C2 counts a parent or child
  XR1 (on since 27 Oct release) a requested section is fetched whole up to 400 rows (was 100), so the selector's relevance
      order - not the row limit - decides which of a long section's paragraphs fit (720.4(d)/(e); 27 sections exceed 100)
  HC1 (on since 27 Oct release) a promoted section whose heading names one topic of the question brings the same-subpart
      sibling whose heading names another (211.137 "Expiration dating" + 211.134 "Drug product inspection"), at most two
      (regulus.rag.section_promotion.sibling_companions)
  HT1 (on since 27 Oct release) a section repeated by heading and number in another part (213.192 / 211.192) is read as the
      larger, general part's unless the question names the specialised part's subject (213: medical gases)
  HM1 (on since 27 Oct release) a section of the promoted section's part whose heading's lead phrase the question contains
      word for word (two topic words or more) joins it: "use of approved components ... rejected components" -> 211.86,
      211.89 (regulus.rag.section_promotion.heading_matches)
  WS1 (measured gain, held back from the 27 Oct release) an overview question whose subject is the promoted section's own heading ("What are the
      reserve sample requirements under Part 211?" -> 211.170) is answered with the whole section verbatim when it is at
      most 3,600 characters (regulus.rag.list_answers.compose_section)
  SP3 (on since 27 Oct release) in Part 111 a promoted section brings the rest of its subpart when the subpart is at most 8
      sections and 4,500 characters (Subpart O complaints: 111.553 + 111.560 + 111.570), except for a list question
      (regulus.rag.section_promotion.small_subpart)
  HX1 (off until measured) up to 48 required paragraphs (was 24) under the unchanged character ceiling (111.260's list)
  AL1 (29 Oct, off until measured) the answer allowance counts up to 18 required units and may reach 3,072 tokens (was 10
      / 2,048): 300Q full300c, 29 answers were cut at 1,696 tokens after CB1/HX1 widened the evidence (none before), S-024
      lost its PASS mid-sentence; greedy decoding leaves every answer that finished unchanged (REGULUS_AL1_MAX_UNITS,
      REGULUS_AL1_MAX_TOKENS; regulus.rag.pipeline.answer_budget)
  DV2 (29 Oct, off until measured) DV1 narrowed: a question about a guidance document itself is not answered from the
      part's paragraphs (NC-022); "the scope of" / "which definitions apply" gets the whole Scope / Definitions section
      (M-001, M-008); DV1's promotion carries its cross-encoder score so Q2 may quote it for an answer that cannot be shown
      (M-026, M-042 withheld) - 300Q full300c, 4 DV1 PASS lost; a question about which other parts govern a subject and how
      they interact with the named part (M-042: Parts 807 and 830 beside Part 820) is not confined to its paragraphs
  CX1 (29 Oct, off until measured) a draft with 3 or more different error codes skips the correction pass (speed): 300Q
      full300c, 31 of 32 such passes were rejected and the one accepted (F-001) stayed FAIL - 0.8 h of 11.8 h. A truncated
      draft keeps its pass (K-005's PASS came from one) (REGULUS_CX1_MAX_CODES)
  OC1 (29 Oct, off until measured) sections added as context - by the reranker (MS1/MS2), absence retrieval, navigation,
      cross-part citations - stay in the evidence as SUPPORTING sources and leave the requested provisions and the
      answer plan, so the model is no longer told to restate them paragraph by paragraph (300Q full300c: they were
      requested units in all 122 answers with added sections; S-024's 40 paragraphs were cut at the token limit). The
      picker's second pick (SP2) stays requested. Fires on 111 of the 300Q (52 PASS; 22 of the 29 truncated answers)
      (regulus.rag.pipeline.companions_as_context)
  FC2 (29 Oct, off until measured) a figure (amount, multi-digit number, date, period) stated under a cited paragraph
      must be in that paragraph, its parent or its child; found only in another supplied paragraph it is an error
      (figure_misattributed), so the draft is corrected or quoted (stepA: F-016's deadlines slid one label past a
      lead-in, F-015's "7 working days" moved from 117.180(a)(6) to (a)(5) - wording-overlap checks passed both)
      (regulus.rag.claim_validation)
  FD1 (29 Oct, off until measured) a denial a supplied EXACT paragraph contradicts is an error (false_denial): "does not
      apply to X" beside a clause applying the part to X, or "does not provide/address/... X" beside a paragraph holding
      nearly all of X's key words; a denial about a named provision is checked against it only; relative clauses,
      anaphora ("such manufacturers"), "no provision requires" and paragraphs that themselves exclude are left alone.
      Offline on 318 recorded answers it fires on M-001 only (820.1(a)(4) applies the part to HCT/Ps)
  CL1 (29 Oct, off until measured) an answer of two or more lines none of which says anything once its citation labels
      and source markers are removed is degenerate (M-042: a correction pass returned fifteen bare citation lines and was
      accepted, since nothing was asserted). Over 618 recorded answers it fires on M-042 only
  FJ1 (29 Oct, off until measured) a question naming a non-U.S. jurisdiction or regulator beside FDA (Health Canada, EU,
      MHRA ...; not a notified body or a mark) is answered for FDA only, with a note that the other jurisdiction's
      requirement is neither confirmed nor described (SD-013). Fires on SD-013 only of the benchmark questions
  GT1 (29 Oct, off until measured) a guidance document asked about by a title quoted right after "guidance", with at
      least 3 distinctive words, that no title in fda_guidance contains (one word may differ in a 6+ word title), is
      said not to be in FDA's guidance listing as held - never summarised (NC-022, NC-015). A looser match called NC-022's
      invented "QMSR Certification Body Accreditation Program" found (3 of 4 words in a real food-safety title)
      (regulus.rag.nonexistence.guidance_title_missing)
  HQ1 (29 Oct, off until measured; step D) sections whose index-time questions (doc2query: plain-language questions the
      model wrote for each section, scripts/build_section_questions.py) match the user's question at cosine >= 0.75 are
      appended to the candidate list, at most 3, inside the named part(s) (regulus.retrieval.section_questions); off
      without data/processed/section_questions/questions.jsonl
  NX2 (29 Oct, off until measured) a subpart the question names that its part does not have, or has only as reserved
      (cfr_subparts), is said to be absent or reserved, with the part's current subparts - and, for Part 820, where the
      QMSR put the requirements; a question about a document built on them is told compliance cannot be confirmed
      (PB-008 "Subparts C through O still apply", NC-013 "Part 1250 Subpart Q"; also NC-003, NC-018 sealed)
  QD2 (29 Oct, off until measured; step E2) a comparison or interaction ("differ from", "versus", "interact with",
      "relate to") is a multi-part question, so QD1's decomposition runs for it (F-022: 117 Subpart G vs 1 Subpart L;
      M-042: Parts 807/830 with 820), and its sub-questions search every part - not only the part the question names -
      at the stricter 0.7 floor (REGULUS_QD2_MIN_SCORE). Fires on 14 of the 300Q. Sections QD1 adds are optional, like
      SP2's and MS1's (an unsupplied one is never announced as "not found")
  PL1 (29 Oct, off until measured; step E3) a section longer than 2,500 characters added by the reranker (MS1/MS2) comes
      as the top-level paragraph its best-scored paragraph sits in, with everything under it - small-to-big passage
      retrieval - and up to 3 may be added: E1/E1b, a whole 117.475 filled the paragraph limit and pushed F-020's key
      117.405 out (REGULUS_PL1_MAX, REGULUS_PL1_WHOLE_CHARS)
  XC1 (29 Oct, off until measured; step E4) the best regulation candidate from a part other than the promoted section's
      joins as a companion when the cross-encoder scores it >= 0.85 - at most one, within MS1's character allowance, a long
      section as its best paragraph family. MS1 pools the promoted part only, so these were never scored for addition
      (dev 225: 8 key sections, e.g. U-001 801.1, G-006 211.137, X-003 4.4). Only the question's own candidates are
      considered, unlike XP1's citations (REGULUS_XC1_MIN_SCORE)
  DR1 (30 Oct, off until measured) a question naming exactly one regulatory programme by its own words (tobacco, food
      labeling, device submissions, radiological health, combination products) and no 21 CFR part gets candidates from that
      programme's parts only - as a cosmetics question has since 22 Oct - and is offered the FD&C Act sections the programme
      is issued under (regulus.retrieval.programmes). full300d: those five domains were 40 of 56 below PASS/PP; tobacco
      candidates were mostly other programmes' sections and Act 904-911 was never offered
  SY4 (30 Oct, off until measured) an acronym the regulations define in their own text ("reference amount commonly consumed
      (RACC)") is spelled out in the section search's words - the picker and the model still read the question as asked
      (regulus.retrieval.acronyms, data/reference/acronyms_v1.json from scripts/build_acronyms.py; senses questions do not
      use, such as PDP = product development protocol, are excluded). L-008: 101.12 was never a candidate for "RACC"
  AP2 (30 Oct, off until measured) the subsection of a picked FD&C Act section that the question (or else the picker) writes
      - "513(g)" - is supplied first. P-008: the picker named 513(g) and the statute step supplied (b), (c), (f) by word
      ranking, leaving (g) among 54 blocks out
  QD3 (30 Oct, off until measured) a QD1 sub-question searched across every part adds a section of a part other than the
      promoted section's only at QD2's stricter cross-encoder floor (0.7, not 0.5). P-012 (UDI, Part 830) got 822.3
      (postmarket surveillance) beside the key 801.20
  AMB1 (30 Oct, off until measured) a question naming no product type whose first-ranked candidates are the same provision
      written for several types (101.7 / 201.62 / 701.13 / 801.62 "Declaration of net quantity of contents") is answered
      with a clarifying question listing them, cited, without calling the model; a short reply naming one type ("food",
      "It is for a medical device.") is answered from that type's section(s) (regulus.rag.product_clarification,
      regulus.rag.conversation). full300d: L-004 and L-005 were answered from the device rules for food keys
  BR1 (29 Oct, off until measured) requests a research tool must not carry out as asked - certify or guarantee
      compliance, sign or attest, keep a missing record from an investigator, draft a court filing, reproduce a copyrighted
      standard, another agency's rules, an unscoped retention question - get a fixed answer saying what it cannot do, why,
      and what it can do instead; the model is not called (regulus.rag.boundary). Fires on 9 of the 300Q (SD-002/003/004/
      006/007/008/009/012, PB-007) and on none of 2,574 other benchmark questions
  NX1 (29 Oct, off until measured) a requested provision the database shows does not exist is said not to exist, with any
      premise term its section never uses ("101.9(k) ... has no (k)(5) ... does not mention 'QR code'"), instead of "the
      available eCFR data does not contain"; no claim where a data gap, a reserved section or a former QSR section could be
      the cause (regulus.rag.nonexistence)
  QC1 (on since 27 Oct release) "does Part 820 require ISO 13485 certification / does a certificate exempt us from inspection"
      is answered from 820.10(a) once the database confirms no live part 820 text mentions certification
      (regulus.rag.qmsr.compose_certification)
  CQ1 conversational query rewriting (regulus.rag.conversation.rewrite_messages): the model rewrites a follow-up into
      a standalone question naming the anchored provision, or repeats it for a new subject; the word rules decide when
      the rewrite is unusable
Blind A/B (23 Oct evening; exact replay of 228 recorded answers, the 6 E1 corrections run live on Qwen3-4B; 20 changed
questions, plain + skeptic scorers, adjudicator on 7 splits): PASS 2 -> 7 (740.17 G1, 82.1602 A1, 710.3 S1, 82.1328 and
82.1330 C82), none lost, misleading 4 -> 2.
"""
from __future__ import annotations

import os

# every fix measured as a gain is on (27 Oct 2026: the UI ran the 26 Oct default A1..P1 while tests ran all of these);
# SP2, SA1, ST1, FS2 stay off; RO1, V2, QX1, CQ1 join after their measurement.
# Release 27 Oct evening: LS1, LS2, QM2, QC1, XR1, HC1, HT1, HM1, SP3 joined after blind-scored live checks
# (devices +9, drug CGMP 7->14/15, supplements 2->5/15, cosmetics +4) and a replay gate over every recorded answer
# (230 cosmetics, 30 devices: 0 PASS lost). Measured and left off: HX1 (S-018 lost), RS2, E2B, C3, AR1; DV1 pending.
DEFAULT_BEFORE_27OCT_RELEASE = ("A1,A2,A3,A4,G1,S1,E1,C82,P1,Q1,FP1,RS1,QM1,D1,PO2,C2,Q2,PR2,ND1,"
                                "DF1,EY1,MF1,E2,CM1,LV1,CS1,EX1,UR1,LC1,FU2,FU2B,FU2C,FU2D,SP1,FC1,SY1,CT1,RO1,TK1,E1S,DF2,NL1,"
                                "VB1,SY2,WC1,AF1,SY3")
# WS1 and ISO1 measured as gains but held back from this release: they change the delivered-answer path that 25 tests
# pin (WS1 answers fixture questions before the draft is validated; ISO1 appends after validation) - reconcile first
DEFAULT = DEFAULT_BEFORE_27OCT_RELEASE + ",LS1,LS2,QM2,QC1,XR1,HC1,HT1,HM1,SP3"
VALIDATOR_PRECISION = frozenset(x.strip().upper() for x in os.environ.get("REGULUS_VALIDATOR_PRECISION", DEFAULT).split(",")
                                 if x.strip() and x.strip().lower() not in ("off", "none"))


def enabled(fix: str) -> bool:
    return fix in VALIDATOR_PRECISION
