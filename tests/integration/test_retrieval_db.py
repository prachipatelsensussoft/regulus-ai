"""Integration tests against the local regulus PostgreSQL database.

Run:  pytest -m integration
Skipped when REGULUS_DATABASE_URL is not configured or the server is down.
Read-only: the session-level write counters are asserted unchanged.
"""

import psycopg
import pytest

from regulus.retrieval import Database, DatabaseConfigError, Filters, LexicalRetriever, QueryType
from regulus.retrieval.models import (METHOD_DEFINITION, METHOD_EXACT_CFR, METHOD_EXACT_DOCKET, METHOD_FULL_TEXT,
                                     METHOD_TRIGRAM)

pytestmark = pytest.mark.integration

WRITE_COUNTERS = "SELECT coalesce(sum(n_tup_ins + n_tup_upd + n_tup_del), 0) AS writes FROM pg_stat_user_tables"


@pytest.fixture(scope="module")
def db():
    try:
        database = Database.from_env()
        database.connect()
    except DatabaseConfigError as exc:
        pytest.skip(str(exc))
    except psycopg.OperationalError as exc:
        pytest.skip(f"database unavailable: {type(exc).__name__}")
    before = database.query(WRITE_COUNTERS)[0]["writes"]
    yield database
    after = database.query(WRITE_COUNTERS)[0]["writes"]
    database.close()
    assert after == before, "write counters changed during the integration tests"


@pytest.fixture(scope="module")
def retriever(db):
    return LexicalRetriever(db)


def test_connection_and_read_only_flags(db):
    s = db.status()
    assert s["server_version"].startswith("18.")
    assert s["database"] == "regulus"
    assert s["transaction_read_only"] == "on" and s["default_transaction_read_only"] == "on"
    assert "***" in s["url"] or "@" not in s["url"]


@pytest.mark.parametrize("stmt", ["INSERT INTO cfr_records (id) VALUES (gen_random_uuid())", "UPDATE cfr_records SET updated_at = now() WHERE false", "DELETE FROM fda_guidance_chunks WHERE false", "CREATE TEMP TABLE t (x int)"])
def test_writes_are_rejected_by_the_server(db, stmt):
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        db.query(stmt)


def test_exact_cfr_section_lookup(retriever):
    resp = retriever.run("21 CFR 820.10", corpus="ecfr", top_k=20)
    assert resp.query_type is QueryType.EXACT_CFR
    exact = [r for r in resp.results if r.retrieval_method == METHOD_EXACT_CFR]
    assert len(exact) == 13   # 12 until migration 0003 restored 820.10(c)(2)'s device table (23 Oct)
    assert exact[0].citation.label == "21 CFR 820.10" and exact[0].relevance_score == 100.0
    assert exact[1].citation.label == "21 CFR 820.10(a)"
    assert exact[0].source_url.startswith("https://www.ecfr.gov/current/title-21/")
    assert exact[0].citation.heading == "Requirements for a quality management system."
    assert exact[0].metadata["part_title"]


def test_exact_cfr_paragraph_lookup_with_descendants(retriever):
    res = retriever.search("Explain 21 CFR 820.10(b)", corpus="ecfr", top_k=10)
    labels = [r.citation.label for r in res if r.retrieval_method == METHOD_EXACT_CFR]
    assert labels[0] == "21 CFR 820.10(b)"
    assert labels[1:] == ["21 CFR 820.10(b)(1)", "21 CFR 820.10(b)(2)", "21 CFR 820.10(b)(3)", "21 CFR 820.10(b)(4)"]
    assert res[0].metadata["match_kind"] == "exact" and res[1].metadata["match_kind"] == "descendant"


def test_reserved_section_returns_no_exact_hit(retriever):
    res = retriever.search("21 CFR 820.30(b)(1)", corpus="ecfr")
    assert not any(r.retrieval_method == METHOD_EXACT_CFR for r in res)


def test_full_text_ecfr(retriever):
    res = retriever.search("tamper-evident packaging requirements", corpus="ecfr", top_k=5)
    assert res and res[0].retrieval_method == METHOD_FULL_TEXT
    assert any(r.citation.section == "211.132" for r in res)
    assert all(r.relevance_score > 0 for r in res)


def test_full_text_guidance_and_citation(retriever):
    res = retriever.search("process validation general principles", corpus="fda_guidance", top_k=5)
    assert res and all(r.corpus == "fda_guidance" for r in res)
    hit = next(r for r in res if r.citation.docket_number == "FDA-2008-D-0559")
    assert hit.citation.status == "final" and hit.citation.issued_on == "2011-01-25" and hit.citation.url.startswith("https://")
    assert hit.citation.label.startswith("FDA Guidance: Process Validation")
    assert hit.metadata["retrievable"] is True


def test_exact_docket_lookup(retriever):
    res = retriever.search("FDA-2016-D-2565", corpus="fda_guidance")
    assert res[0].retrieval_method == METHOD_EXACT_DOCKET and res[0].citation.docket_number == "FDA-2016-D-2565"


def test_trigram_fallback_on_identifier_and_misspelling(retriever):
    res = retriever.search("820.3", corpus="ecfr", methods=[METHOD_TRIGRAM], top_k=5)
    assert res and res[0].citation.section == "820.3" and res[0].retrieval_method == METHOD_TRIGRAM
    res = retriever.search("informd consent", corpus="ecfr", methods=[METHOD_TRIGRAM], top_k=5)
    assert any("consent" in (r.heading or "").lower() for r in res)


def test_metadata_filters(retriever):
    res = retriever.search("labeling", corpus="ecfr", filters=Filters(part="801"), top_k=10)
    assert res and all(r.metadata["part"] == "801" for r in res)
    res = retriever.search("labeling", corpus="ecfr", filters=Filters(section="801.109"), top_k=10)
    assert res and all(r.citation.section == "801.109" for r in res)
    res = retriever.search("quality system", corpus="fda_guidance", filters=Filters(sector="device", status="final"), top_k=10)
    assert res and all(r.metadata["sector"] == "device" and r.metadata["status"] == "final" for r in res)
    res = retriever.search("quality system", corpus="fda_guidance", filters=Filters(center="CDER"), top_k=5)
    assert res and all(r.metadata["center"] == "CDER" for r in res)
    res_all = retriever.search("labeling", corpus="ecfr", filters=Filters(subchapter="H"), top_k=10)
    assert res_all and all(r.metadata["subchapter"] == "H" for r in res_all)


def test_cross_corpus_dedup_and_determinism(retriever):
    a = retriever.search("recall policy", top_k=10)
    b = retriever.search("recall policy", top_k=10)
    assert [r.key for r in a] == [r.key for r in b]
    assert len({r.key for r in a}) == len(a)
    assert {r.corpus for r in a} == {"ecfr", "fda_guidance"}


LONG_QUESTION = "Explain to someone with no medical background why a new drug must be approved before it can be sold."


def test_stage7_fallback_recovers_long_question_and_stage5_mode_returns_nothing(db):
    stage5 = LexicalRetriever(db, enable_or_fallback=False).run(LONG_QUESTION, top_k=8)
    assert stage5.results == [] and stage5.diagnostics["fallback_used"] is False and stage5.diagnostics["retrieval_mode"] == "none"
    stage7 = LexicalRetriever(db).run(LONG_QUESTION, top_k=8)
    d = stage7.diagnostics
    assert d["primary_result_count"] == 0 and d["fallback_used"] is True and d["retrieval_mode"] == "fallback"
    assert 0 < len(stage7.results) <= 8 and d["fallback_candidate_count"] <= 16
    assert all(r.retrieval_method == "full_text_or" for r in stage7.results)
    assert all(r.relevance_score < 50 for r in stage7.results)
    again = LexicalRetriever(db).run(LONG_QUESTION, top_k=8)
    assert [r.key for r in again.results] == [r.key for r in stage7.results]


def test_stage7_does_not_change_exact_and_sufficient_primary_queries(db):
    for q in ("21 CFR 820.10", "tamper-evident packaging requirements", "FDA-2016-D-2565"):
        a = LexicalRetriever(db, enable_or_fallback=False).run(q, top_k=10)
        b = LexicalRetriever(db).run(q, top_k=10)
        assert [r.key for r in a.results] == [r.key for r in b.results], q
        assert b.diagnostics["fallback_used"] is False, q


def test_stage7_negative_control_stays_free_of_exact_hits(db):
    resp = LexicalRetriever(db).run("21 CFR 820.30(b)(1)", top_k=10)
    assert not any(r.retrieval_method == METHOD_EXACT_CFR for r in resp.results)
    assert resp.diagnostics["fallback_terms"] in ([], ["820.30"])


def test_results_serialise_without_internal_ids(retriever):
    payload = retriever.run("21 CFR 801.109(a)", top_k=3).to_dict()
    text = str(payload)
    assert "guidance_id" not in text and "password" not in text
    assert payload["results"][0]["citation"]["label"] == "21 CFR 801.109(a)"


# --------------------------------------------- C1: part-level stems ranked by relevance
# The live ts_rank_cd signal, against the real corpus. S-002 of the acceptance
# benchmark is the recorded defect: 84 section stems of Part 111 all scored
# 100.0, so ASCII key order put the governing section 21 CFR 111.3 46th and the
# bounded context dropped it.

S002 = ("Where are the definitions that apply to 21 CFR Part 111 located, "
        "and does Part 111 incorporate definitions from elsewhere?")


def _part_stems(db, question):
    """The part's section STEMS (one per section, as ECFR_EXACT_PART returns them) and the
    whole registry. C2 adds paragraph rows of the ranked sections as further exact
    evidence; they are excluded here so the C1 assertions keep measuring the ranking."""
    from regulus.planning import plan_query
    from regulus.rag.selection import is_required

    registry = LexicalRetriever(db).run_plan(plan_query(question), corpus="ecfr", top_k=10).registry
    stems = [e for e in registry.evidence
             if is_required(e) and "part_expanded_from" not in e.result.metadata]
    return stems, registry


def test_s002_ranks_the_governing_section_first_not_46th(db):
    stems, _ = _part_stems(db, S002)
    labels = [e.citation for e in stems]
    assert len(stems) == 84                      # the whole part is still retrieved
    assert labels[0] == "21 CFR 111.3"           # was 46th, behind 111.103/111.105/...
    assert stems[0].result.metadata["ts_rank_cd"] > 0.0
    assert "definitions" in stems[0].result.metadata["part_rank_query"]
    # the stems no longer share one score, which is what left ASCII order deciding
    assert len({e.score for e in stems}) == len(stems)
    assert all(99.9 < e.score <= 100.0 for e in stems)


def test_s002_governing_section_survives_the_selection_ceiling(db):
    from regulus.rag.selection import select_evidence

    _, registry = _part_stems(db, S002)
    selection = select_evidence(registry)
    kept = [s.evidence.citation for s in selection.selected if s.protected]
    assert "21 CFR 111.3" in kept and kept[0] == "21 CFR 111.3"
    assert len(kept) < 84  # the ceiling still binds; it now cuts the irrelevant tail


def test_part_level_relevance_ranking_is_deterministic_on_the_live_corpus(db):
    runs = [[(e.citation, e.score) for e in _part_stems(db, S002)[0]] for _ in range(3)]
    assert all(r == runs[0] for r in runs[1:])


# Four acceptance-benchmark part-level questions, from four different Parts, whose
# governing section the ASCII ordering pushed outside the selection ceiling.
@pytest.mark.parametrize("question_id, question, governing", [
    ("S-002", S002, "111.3"),
    ("F-001", "Who is subject to 21 CFR Part 117 and what are the principal exemptions?", "117.5"),
    ("L-009", "What are the small-package labeling exemptions and alternatives in Part 101?", "101.9"),
    ("D-003", "What are the personnel qualification and training requirements under Part 211?", "211.25"),
])
def test_relevance_ranking_brings_the_governing_section_inside_the_ceiling(db, question_id, question, governing,
                                                                          cpu_era_budget):
    """Measured across four Parts: the benchmark's expected section was retrieved but sorted
    past the ceiling by ASCII order, and relevance ranking brings it inside."""
    from regulus.rag.selection import select_evidence

    stems, registry = _part_stems(db, question)
    # one stem per section; the stem's own label may carry a paragraph path ("211.25(a)")
    sections = [e.result.metadata["section"] for e in stems]
    assert governing in sections, question_id
    selection = select_evidence(registry)
    assert len(selection.selected) < len(stems), question_id  # the ceiling binds: order decides
    kept = [s.evidence.result.metadata["section"] for s in selection.selected]
    assert governing in kept, question_id
    # ASCII section order would have placed it behind more stems than the context can hold
    assert sorted(sections).index(governing) >= len(selection.selected), question_id


def test_part_lookup_without_a_question_keeps_the_pre_c1_order(db):
    """run() (hybrid retriever, scripts, eval baselines) passes no question: the
    statement must behave exactly as the plain `ORDER BY r.section_number` did."""
    resp = LexicalRetriever(db).run("What is in 21 CFR part 111?", corpus="ecfr", top_k=10)
    exact = [r for r in resp.results if r.retrieval_method == METHOD_EXACT_CFR]
    assert exact and all(r.relevance_score == 100.0 for r in exact)
    assert "part_relevance_rank" not in exact[0].metadata
    assert [r.key for r in exact] == sorted(r.key for r in exact)  # the pre-C1 tie-break


# ------------------------------- C2: a ranked part section supplies its own paragraphs
# Live corpus. 21 CFR 111.3's first row is the lead-in "The definitions and
# interpretations of terms in section 201 of the ... Act ... apply", and the 28
# definitions it introduces are the 28 rows below it. C1 ranks 111.3 first; C2
# must let it reach the prompt with those rows.


def test_s002_governing_section_supplies_its_definitions_not_just_the_lead_in(db):
    stems, registry = _part_stems(db, S002)
    rows = [e for e in registry.evidence if e.result.metadata.get("section") == "111.3"]
    assert len(rows) > 1, "111.3 reached the registry as its lead-in only"
    assert [e.result.metadata["ordinal"] for e in rows] == sorted(e.result.metadata["ordinal"] for e in rows)
    assert rows[0].result.metadata["part_relevance_rank"] == 0          # C1 ranking intact
    assert all(e.result.metadata["part_expanded_from"] == "111.3" for e in rows[1:])
    assert any("means" in e.text for e in rows[1:]), "no definition text was supplied"
    assert all(e.tier == "exact" and "r1" in e.satisfies for e in rows)


def test_s002_definitions_survive_selection_in_document_order(db):
    from regulus.rag.selection import select_evidence

    _, registry = _part_stems(db, S002)
    selection = select_evidence(registry)
    kept = [s for s in selection.selected if s.evidence.result.metadata.get("section") == "111.3"]
    assert len(kept) > 1 and kept[0].index == 1   # the ranked section leads the context
    assert [s.evidence.result.metadata["ordinal"] for s in kept] == list(range(len(kept)))
    assert selection.total_chars <= selection.policy.hard_max_total_chars
    assert len(selection.selected) <= selection.policy.hard_max_results


def test_part_expansion_only_deepens_sections_the_question_matches(db):
    """Bounded on both axes: at most part_expand_sections matching sections, and the
    expansion never touches a section C1 gave a zero relevance rank."""
    r = LexicalRetriever(db)
    stems, registry = _part_stems(db, S002)
    expanded = {e.result.metadata["part_expanded_from"] for e in registry.evidence
                if "part_expanded_from" in e.result.metadata}
    assert len(expanded) <= r.part_expand_sections
    ranked = {e.result.metadata["section"] for e in stems if e.result.metadata["ts_rank_cd"] > 0}
    assert expanded <= ranked, "a section with no relevance signal was expanded"
    per_section = {}
    for e in registry.evidence:
        src = e.result.metadata.get("part_expanded_from")
        if src:
            per_section[src] = per_section.get(src, 0) + 1
    assert all(n <= r.part_expand_rows for n in per_section.values())


def test_part_expansion_cannot_pull_in_the_whole_part(db):
    """84 stems and 29 rows in 111.3 alone: the registry must stay far below the part."""
    from regulus.retrieval.lexical import LexicalRetriever as LR

    _, registry = _part_stems(db, S002)
    whole_part = db.query(
        "SELECT count(*) AS n FROM cfr_records WHERE part_number='111' AND removed_on IS NULL")[0]["n"]
    required = [e for e in registry.evidence if e.tier == "exact"]
    r = LR(db)
    assert len(required) <= 84 + r.part_expand_sections * r.part_expand_rows < whole_part


def test_part_expansion_is_deterministic_on_the_live_corpus(db):
    runs = [[(e.citation, e.score, e.result.metadata["ordinal"])
             for e in _part_stems(db, S002)[1].evidence] for _ in range(3)]
    assert all(r == runs[0] for r in runs[1:])


def test_section_level_lookup_is_unchanged_by_c2(db):
    """A section-level request is retrieved exactly as before: same rows, same scores,
    and no part-expansion metadata anywhere."""
    resp = LexicalRetriever(db).run("21 CFR 820.10", corpus="ecfr", top_k=20)
    exact = [r for r in resp.results if r.retrieval_method == METHOD_EXACT_CFR]
    assert len(exact) == 13   # with 820.10(c)(2)'s device table (migration 0003)
    assert exact[0].citation.label == "21 CFR 820.10" and exact[0].relevance_score == 100.0
    assert exact[1].citation.label == "21 CFR 820.10(a)"
    assert not any("part_expanded_from" in r.metadata for r in resp.results)


# ------------------------------------- C3: cross-section ranges against the live corpus
# "21 CFR 50.50 through 50.56" parses to two endpoints; which sections lie between them
# is a fact about the corpus. Part 50 holds all seven; Part 820 holds no standalone
# 820.30 (it sits in a consolidated 820.20-820.30 block), which must not be invented.

RANGE_Q = "What do 21 CFR 50.50 through 50.56 require?"


def _expanded(db, question):
    from regulus.planning import plan_query
    from regulus.planning.ranges import expand_section_ranges

    return expand_section_ranges(plan_query(question), LexicalRetriever(db).sections_in_part)


def test_sections_in_part_returns_document_order_not_ascii_order(db):
    listing = LexicalRetriever(db).sections_in_part("50")
    numbers = [n for n, _ in listing]
    assert numbers == ["50.1", "50.3", "50.20", "50.22", "50.23", "50.24", "50.25", "50.27",
                       "50.50", "50.51", "50.52", "50.53", "50.54", "50.55", "50.56"]
    assert numbers != sorted(numbers), "ASCII order would put 50.3 after 50.20"
    orders = [o for _, o in listing]
    assert orders == sorted(orders) and len(set(orders)) == len(orders)


def test_live_range_expands_to_every_existing_intermediate_section(db):
    plan, notes = _expanded(db, RANGE_Q)
    assert [r.canonical for r in plan.requirements] == [f"21 CFR 50.{n}" for n in (50, 51, 52, 53, 54, 55, 56)]
    assert any("expanded to 7 sections" in n for n in notes)


def test_live_range_does_not_invent_an_endpoint_the_corpus_lacks(db):
    """Part 820 has no standalone 820.30; only the real 820.35 is added."""
    plan, _ = _expanded(db, "What do 21 CFR 820.30 through 820.40 require?")
    assert [r.canonical for r in plan.requirements] == ["21 CFR 820.30", "21 CFR 820.35", "21 CFR 820.40"]
    known = {n for n, _ in LexicalRetriever(db).sections_in_part("820")}
    assert "820.30" not in known and "820.35" in known


def test_live_range_is_retrieved_through_the_ordinary_exact_section_path(db):
    plan, _ = _expanded(db, RANGE_Q)
    registry = LexicalRetriever(db).run_plan(plan, corpus="ecfr", top_k=8).registry
    assert registry.unresolved_requirements() == ()   # all seven exist and were retrieved
    for req in plan.requirements:
        assert registry.exact_for(req.id), req.canonical
    assert not any("part_expanded_from" in e.result.metadata for e in registry.evidence)


def test_live_range_selection_still_respects_both_ceilings(db, cpu_era_budget):
    from regulus.rag.selection import select_evidence

    plan, _ = _expanded(db, RANGE_Q)
    registry = LexicalRetriever(db).run_plan(plan, corpus="ecfr", top_k=8).registry
    selection = select_evidence(registry)
    assert len(selection.selected) <= selection.policy.hard_max_results == 24
    assert selection.total_chars <= selection.policy.hard_max_total_chars == 9000
    # the range does not fit; the overflow is recorded rather than silently dropped
    assert len(selection.diagnostics["required_dropped"]) > 0


def test_live_range_expansion_is_deterministic(db):
    runs = [[r.canonical for r in _expanded(db, RANGE_Q)[0].requirements] for _ in range(3)]
    assert all(r == runs[0] for r in runs[1:])


def test_a_live_non_range_question_is_not_touched_by_c3(db):
    from regulus.planning import plan_query
    from regulus.planning.ranges import expand_section_ranges

    sections_in_part = LexicalRetriever(db).sections_in_part
    for q in ("What does 21 CFR 820.10 require?", "What is in 21 CFR part 111?",
              "21 CFR 820.10(b)(1) through (b)(4)"):
        before = plan_query(q)
        after, notes = expand_section_ranges(before, sections_in_part)
        assert after is before and notes == (), q


# ------------------------------------------------------------- definition retrieval on the live corpus

def _defines(term: str):
    """A paragraph that OPENS with the term (after its paragraph label, if any) followed by "means"."""
    import re

    return re.compile(r"^\s*(?:\([0-9A-Za-z]{1,6}\)\s*)*" + r"\s+".join(map(re.escape, term.split())) + r"\s+means\b", re.I)


@pytest.mark.parametrize("term", ["FDA", "drug product"])
def test_definition_lookup_returns_only_the_cfrs_own_definitions(db, term):
    """ECFR_DEFINITION's PostgreSQL regular expression, on the real corpus: one-word and multi-word terms,
    with and without a paragraph label, and nothing that merely mentions the term."""
    rows = LexicalRetriever(db).definition_ecfr(term, Filters(), 3)
    assert 1 <= len(rows) <= 3
    assert all(r.retrieval_method == METHOD_DEFINITION for r in rows)
    assert all(_defines(term).match(r.text) for r in rows), [r.text[:80] for r in rows]


def test_definition_lookup_of_an_undefined_term_returns_nothing(db):
    assert LexicalRetriever(db).definition_ecfr("zzqx nonexistent", Filters(), 3) == []


def test_a_definitional_question_retrieves_the_definition_and_skips_the_keyword_search(db):
    from regulus.planning import plan_query

    resp = LexicalRetriever(db).run_plan(plan_query("What does FDA stand for?"), corpus="ecfr", top_k=8).response
    methods = {r.retrieval_method for r in resp.results}
    assert METHOD_DEFINITION in methods and METHOD_FULL_TEXT not in methods
    assert any("keyword search was not run" in n for n in resp.notes)
