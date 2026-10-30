"""Stage 2: plan-driven, requirement-level retrieval and the EvidenceRegistry.

A small in-memory eCFR table stands in for PostgreSQL. The fake understands
the exact-lookup statements semantically (section, paragraph, descendants,
part) and returns canned rows for the text statements, so tests exercise
behaviour, not SQL.
"""

from datetime import date

import pytest

from regulus.planning import plan_query
from regulus.rag import RagConfig, RagPipeline
from regulus.retrieval import Evidence, EvidenceRegistry, Filters, LexicalRetriever, PlanRetrieval, RetrievalResult
from regulus.retrieval import sql as sql_module
from regulus.retrieval.evidence import (
    REL_CHILD,
    REL_EXACT,
    REL_NONE,
    REL_PARENT,
    REL_SIBLING,
    TIER_DISCOVERY,
    TIER_EXACT,
    TIER_STRUCTURAL,
    TIER_SUPPORTING,
    relate,
)
from regulus.retrieval.models import METHOD_EXACT_CFR, METHOD_EXACT_DOCKET, METHOD_FULL_TEXT, METHOD_FULL_TEXT_OR, METHOD_TRIGRAM
from tests.test_rag import FakeGenerator

# ------------------------------------------------------------------ fixtures


def row(section, para, ordinal, part=None, text=None, **over):
    part = part or section.split(".")[0]
    r = {
        "title": "21", "chapter": "I", "subchapter_code": "H", "part_number": part, "section_number": section,
        "paragraph_path": para, "path_ordinal": 0, "ordinal": ordinal, "section_heading": f"Section {section} heading",
        "parent_headings": ["Medical Devices"], "body_text": text or f"{section}{para} body text",
        "ecfr_permalink": f"https://www.ecfr.gov/current/title-21/section-{section}", "govinfo_pdf_url": None,
        "snapshot_date": date(2026, 9, 3), "effective_date": None, "last_amended_date": None, "is_reserved": False,
        "part_title": f"Part {part}",
    }
    r.update(over)
    return r


def guidance_row(ordinal=0, docket="FDA-2016-D-2565", **over):
    r = {
        "body_text": "guidance chunk text", "title_text": "QMS Guidance", "ordinal": ordinal, "start_offset": 0,
        "title": "QMS Guidance", "docket_number": docket, "status": "final", "center": "CDRH", "sector": "device",
        "issued_on": date(2024, 11, 21), "source_url": "https://www.fda.gov/media/1.pdf", "anchor_part": "820",
        "anchor_section": None, "doc_class": "guidance", "retrievable": True,
    }
    r.update(over)
    return r


# 820.10: section row, (a), (b), (b)(1)..(b)(4), (c); 820.30: section row and (a)
TABLE = [
    row("820.10", "", 0), row("820.10", "(a)", 1), row("820.10", "(b)", 2),
    row("820.10", "(b)(1)", 3), row("820.10", "(b)(2)", 4), row("820.10", "(b)(3)", 5), row("820.10", "(b)(4)", 6),
    row("820.10", "(c)", 7),
    row("820.30", "", 0), row("820.30", "(a)", 1),
]


class TableDatabase:
    """Answers the exact statements from TABLE and the text statements from canned rows."""

    def __init__(self, table=TABLE, *, full_text=(), guidance=(), trigram=(), or_ecfr=(), or_guidance=(), docket=(),
                 coverage=(), anchored=()):
        self.table = [dict(r) for r in table]
        self.canned = {"full_text": list(full_text), "guidance": list(guidance), "trigram": list(trigram),
                       "or_ecfr": list(or_ecfr), "or_guidance": list(or_guidance), "docket": list(docket),
                       "coverage": list(coverage), "anchored": list(anchored)}
        self.calls: list[tuple[str, dict]] = []

    def query(self, statement, params=None):
        name = next(n for n, s in sql_module.ALL_STATEMENTS.items() if s == statement)
        params = dict(params or {})
        self.calls.append((name, params))
        if name == "ECFR_EXACT_SECTION":
            para, prefix = params["paragraph"], params["paragraph_prefix"]
            rows = [r for r in self.table if r["section_number"] == params["section_number"]
                    and (para == "" or r["paragraph_path"] == para or r["paragraph_path"].startswith(prefix[:-1]))]
            for r in rows:
                r["match_kind"] = "exact" if para == "" or r["paragraph_path"] == para else "descendant"
            return sorted(rows, key=lambda r: r["ordinal"])[: params["limit"]]
        if name == "ECFR_PART_SECTIONS":
            # sections of the part, in document order; sort_order is the table position
            seen, out = set(), []
            for r in self.table:
                if r["part_number"] == params["part_number"] and r["section_number"] not in seen:
                    seen.add(r["section_number"])
                    out.append({"number": r["section_number"], "sort_order": len(out)})
            return out
        if name == "ECFR_EXACT_ROW":
            return [dict(r, match_kind="exact") for r in self.table
                    if r["section_number"] == params["section_number"] and r["paragraph_path"] == params["paragraph"]][:1]
        if name == "ECFR_EXACT_PART":
            # the statement orders by the question's relevance, then section number, and
            # always returns a `rank` column; `part_rank` on a table row stands in for
            # ts_rank_cd so a test can say which stems the question actually matches
            rows = [dict(r, match_kind="exact", rank=float(r.get("part_rank", 0.0)))
                    for r in self.table if r["part_number"] == params["part_number"] and r["ordinal"] == 0]
            if not params.get("query"):
                for r in rows:
                    r["rank"] = 0.0  # no query text -> ts_rank_cd is 0 for every row
            return sorted(rows, key=lambda r: (-r["rank"], r["section_number"]))[: params["limit"]]
        if name == "ECFR_FULL_TEXT":
            return [dict(r) for r in (self.canned["or_ecfr"] if " or " in params["query"] or params["query"] in ("820.10", "820.30", "820.99", "820.999") else self.canned["full_text"])]
        if name == "GUIDANCE_FULL_TEXT":
            return [dict(r) for r in (self.canned["or_guidance"] if " or " in params["query"] or params["query"].replace(".", "").isdigit() else self.canned["guidance"])]
        if name == "ECFR_TRIGRAM":
            return [dict(r) for r in self.canned["trigram"]]
        if name == "GUIDANCE_EXACT_DOCKET":
            return [dict(r) for r in self.canned["docket"]]
        if name == "ECFR_SECTION_COVERAGE":
            return [dict(r) for r in self.canned["coverage"]][: params["limit"]]
        if name == "GUIDANCE_ANCHORED":   # AG1: guidance anchored to the requested sections (none unless a test supplies them)
            return [dict(r) for r in self.canned["anchored"]][: params["limit"]]
        raise AssertionError(name)

    def names(self):
        return [n for n, _ in self.calls]


def run(question, db=None, **kw) -> PlanRetrieval:
    return LexicalRetriever(db or TableDatabase()).run_plan(plan_query(question), **kw)


def labels(items):
    return [e.citation if isinstance(e, Evidence) else e.citation.label for e in items]


# ----------------------------------------------------------------- relation


def test_relation_rules_by_identity():
    ref = plan_query("820.10(b)(2)").requirements[0].reference
    mk = lambda sec, para: LexicalRetriever._ecfr_result(row(sec, para, 0), METHOD_FULL_TEXT, 1.0, {})
    assert relate(ref, mk("820.10", "(b)(2)")) == REL_EXACT
    assert relate(ref, mk("820.10", "(b)")) == REL_PARENT
    assert relate(ref, mk("820.10", "")) == REL_PARENT
    assert relate(ref, mk("820.10", "(b)(2)(i)")) == REL_CHILD
    assert relate(ref, mk("820.10", "(b)(3)")) == REL_SIBLING
    assert relate(ref, mk("820.10", "(b)(20)")) == REL_SIBLING  # "(b)(2" is not a prefix of "(b)(20)"
    assert relate(ref, mk("820.30", "(b)(2)")) == REL_NONE
    guidance = LexicalRetriever._guidance_result(guidance_row(), METHOD_FULL_TEXT, 1.0, {})
    assert relate(ref, guidance) == REL_NONE
    section_ref = plan_query("21 CFR 820.10").requirements[0].reference
    assert relate(section_ref, mk("820.10", "(b)(2)")) == REL_EXACT  # a section is its rows
    part_ref = plan_query("21 CFR part 820").requirements[0].reference
    assert relate(part_ref, mk("820.30", "")) == REL_EXACT and relate(part_ref, mk("801.1", "")) == REL_NONE


# ------------------------------------------------------------ single / multi


def test_single_provision_gets_its_own_exact_evidence():
    out = run("What does 820.10(b)(2) require?")
    reg = out.registry
    assert isinstance(reg, EvidenceRegistry) and [r.id for r in reg.plan.requirements] == ["r1"]
    assert labels(reg.exact_for("r1")) == ["21 CFR 820.10(b)(2)"]
    assert reg.exact_for("r1")[0].tier == TIER_EXACT and reg.exact_for("r1")[0].retrieval_method == METHOD_EXACT_CFR
    assert reg.resolved_requirements() == ("r1",) and reg.unresolved_requirements() == ()
    assert labels(reg.structural_for("r1")) == ["21 CFR 820.10(b)"]  # parent fetched as context, not as satisfaction
    assert out.response.results[0].citation.label == "21 CFR 820.10(b)(2)"


def test_multiple_provisions_are_retrieved_independently():
    db = TableDatabase()
    out = run("What do 820.10(b)(2) and 820.10(b)(4) require?", db)
    reg = out.registry
    exact_calls = [p for n, p in db.calls if n == "ECFR_EXACT_SECTION"]
    assert [p["paragraph"] for p in exact_calls] == ["(b)(2)", "(b)(4)"]
    assert labels(reg.exact_for("r1")) == ["21 CFR 820.10(b)(2)"]
    assert labels(reg.exact_for("r2")) == ["21 CFR 820.10(b)(4)"]
    assert reg.resolved_requirements() == ("r1", "r2")
    # the shared parent is one piece of structural evidence for both
    parent = [e for e in reg.evidence if e.citation == "21 CFR 820.10(b)"]
    assert len(parent) == 1 and parent[0].structural_for == ("r1", "r2") and parent[0].tier == TIER_STRUCTURAL
    assert [n for n in db.names() if n == "ECFR_EXACT_ROW"] == ["ECFR_EXACT_ROW"]  # parent fetched once


def test_range_yields_four_independent_associations():
    out = run("List the requirements in 21 CFR 820.10(b)(1) through (b)(4).")
    reg = out.registry
    assert [r.canonical for r in reg.plan.requirements] == [f"21 CFR 820.10(b)({i})" for i in range(1, 5)]
    for rid, i in zip(("r1", "r2", "r3", "r4"), range(1, 5)):
        assert labels(reg.exact_for(rid)) == [f"21 CFR 820.10(b)({i})"]
    assert reg.unresolved_requirements() == ()
    assert labels(out.response.results[:4]) == [f"21 CFR 820.10(b)({i})" for i in range(1, 5)]


def test_list_yields_three_requirements_and_no_sibling_satisfaction():
    reg = run("820.10(b)(1), (b)(2), and (b)(4)").registry
    assert [r.canonical for r in reg.plan.requirements] == ["21 CFR 820.10(b)(1)", "21 CFR 820.10(b)(2)", "21 CFR 820.10(b)(4)"]
    assert all(len(reg.exact_for(rid)) == 1 for rid in ("r1", "r2", "r3"))
    assert "21 CFR 820.10(b)(3)" not in labels(reg.evidence)  # the unrequested sibling is not fetched by exact lookups


def test_mixed_sections():
    reg = run("Compare 21 CFR 820.10(b)(1) with 21 CFR 820.30(a).").registry
    assert labels(reg.exact_for("r1")) == ["21 CFR 820.10(b)(1)"]
    assert labels(reg.exact_for("r2")) == ["21 CFR 820.30(a)"]
    assert labels(reg.structural_for("r2")) == ["21 CFR 820.30"]


def test_section_level_request_is_satisfied_by_its_rows_and_part_by_sections():
    reg = run("What does 21 CFR 820.10 say?").registry
    assert len(reg.exact_for("r1")) == 8 and reg.unresolved_requirements() == ()
    assert reg.structural_for("r1") == ()
    reg = run("What is in 21 CFR part 820?").registry
    assert labels(reg.exact_for("r1")) == ["21 CFR 820.10", "21 CFR 820.30"]


# -------------------------------------------------------------- duplicates


def test_duplicate_reference_reuses_one_exact_query():
    db = TableDatabase()
    plan = plan_query("820.10(b)(2) and 820.10(b)(2)")
    assert len(plan.requirements) == 1  # Stage 1 de-duplicates
    LexicalRetriever(db).run_plan(plan)
    assert db.names().count("ECFR_EXACT_SECTION") == 1


def test_equivalent_duplicate_requirements_share_evidence_without_a_second_query(monkeypatch):
    # simulate a plan that still carries two requirements for one provision
    from dataclasses import replace

    plan = plan_query("820.10(b)(2) and 820.10(b)(4)")
    twin = replace(plan.requirements[1], reference=plan.requirements[0].reference, raw="820.10(b)(2)")
    plan = replace(plan, requirements=(plan.requirements[0], twin))
    db = TableDatabase()
    reg = LexicalRetriever(db).run_plan(plan).registry
    assert db.names().count("ECFR_EXACT_SECTION") == 1
    assert labels(reg.exact_for("r1")) == labels(reg.exact_for("r2")) == ["21 CFR 820.10(b)(2)"]
    assert reg.exact_for("r1")[0].satisfies == ("r1", "r2")
    assert [r.id for r in reg.plan.requirements] == ["r1", "r2"]  # order and identity preserved


# ------------------------------------------------------------ parent/child


def test_child_request_parent_row_is_structural_not_exact():
    reg = run("820.10(b)(2)").registry
    parent = next(e for e in reg.evidence if e.citation == "21 CFR 820.10(b)")
    assert parent.tier == TIER_STRUCTURAL and parent.satisfies == () and parent.structural_for == ("r1",)
    assert parent.relations == {"r1": REL_PARENT}


def test_parent_request_children_are_structural_not_exact():
    reg = run("What does 21 CFR 820.10(b) require?").registry
    assert labels(reg.exact_for("r1")) == ["21 CFR 820.10(b)"]
    structural = reg.structural_for("r1")  # the section row (parent) first, then the children
    assert labels(structural) == ["21 CFR 820.10"] + [f"21 CFR 820.10(b)({i})" for i in range(1, 5)]
    assert all(e.tier == TIER_STRUCTURAL and e.satisfies == () for e in structural)
    assert [e.relations["r1"] for e in structural] == [REL_PARENT] + [REL_CHILD] * 4
    assert reg.resolved_requirements() == ("r1",)


def test_parent_request_without_its_own_row_stays_unresolved():
    # a paragraph whose own row is missing but whose children exist
    table = [r for r in TABLE if not (r["section_number"] == "820.10" and r["paragraph_path"] == "(b)")]
    reg = run("What does 21 CFR 820.10(b) require?", TableDatabase(table)).registry
    assert reg.exact_for("r1") == () and reg.unresolved_requirements() == ("r1",)
    assert len(reg.structural_for("r1")) == 5  # section row plus the four children remain structural evidence


def test_wrong_sibling_never_satisfies():
    table = [r for r in TABLE if r["paragraph_path"] != "(b)(2)"]  # (b)(2) missing, (b)(3) present
    full_text = [row("820.10", "(b)(3)", 5, rank=0.9)]
    reg = run("820.10(b)(2)", TableDatabase(table, full_text=full_text)).registry
    assert reg.exact_for("r1") == () and reg.unresolved_requirements() == ("r1",)
    sib = next(e for e in reg.evidence if e.citation == "21 CFR 820.10(b)(3)")
    assert sib.tier == TIER_SUPPORTING and sib.relations == {"r1": REL_SIBLING}


# ------------------------------------------------------------- unresolved


def test_missing_exact_evidence_triggers_targeted_fallback_for_that_requirement_only():
    db = TableDatabase(or_ecfr=[row("801.1", "(a)", 1, rank=0.5)], trigram=[row("820.30", "", 0, sim=0.55)])
    out = run("What do 820.10(b)(2) and 820.99 require?", db)
    reg = out.registry
    assert reg.resolved_requirements() == ("r1",) and reg.unresolved_requirements() == ("r2",)
    st = reg.requirement_state["r2"]
    assert not st.resolved and st.exact_attempted and st.fallback_run and st.fallback_terms == ("820.99",)
    assert "No live record for 21 CFR 820.99" in st.note
    # fallback queries carry the unresolved identifier, never the whole question
    or_calls = [p for n, p in db.calls if n in ("ECFR_FULL_TEXT", "GUIDANCE_FULL_TEXT") and p["query"] == "820.99"]
    assert len(or_calls) == 2
    assert not any(" or " in p.get("query", "") for _, p in db.calls)
    assert not any(p.get("query") == "820.10" for n, p in db.calls if n == "ECFR_TRIGRAM")
    # what the fallback found is discovery for r2 and nothing for r1
    disc = reg.discovered_for("r2")
    assert set(labels(disc)) == {"21 CFR 820.30", "21 CFR 801.1(a)"}
    assert all(e.tier == TIER_DISCOVERY and e.satisfies == () for e in disc)
    assert all(e.retrieval_method in (METHOD_FULL_TEXT_OR, METHOD_TRIGRAM) for e in disc)
    assert reg.discovered_for("r1") == ()
    assert labels(reg.exact_for("r1")) == ["21 CFR 820.10(b)(2)"]
    assert st.discovery_hits == 2
    assert out.response.diagnostics["fallback_used"] and out.response.diagnostics["fallback_terms"] == ["820.99"]


def test_resolved_requirements_do_not_trigger_any_broad_fallback():
    db = TableDatabase()
    out = run("What does 820.10(b)(2) require?", db)
    assert "ECFR_TRIGRAM" not in db.names()
    assert not any(" or " in p.get("query", "") for _, p in db.calls)
    assert out.response.diagnostics["fallback_used"] is False
    assert out.response.diagnostics["primary_result_count"] >= 1


def test_discovery_is_never_promoted_to_exact_by_score():
    # a trigram hit on the requested section scores in the discovery band but IS the section row -> structural by identity
    db = TableDatabase(trigram=[row("820.10", "", 0, sim=0.9), row("172.820", "", 0, sim=0.8)])
    reg = run("820.10(b)(2) and 820.999", db).registry
    for e in reg.evidence:
        if e.retrieval_method in (METHOD_FULL_TEXT_OR, METHOD_TRIGRAM):
            assert e.tier != TIER_EXACT
    assert reg.unresolved_requirements() == ("r2",)


def test_nonexistent_section_alone_completes_deterministically():
    db = TableDatabase()
    out = run("What are the requirements of 21 CFR 314.999?", db)
    reg = out.registry
    assert reg.unresolved_requirements() == ("r1",) and reg.exact_for("r1") == ()
    assert out.response.notes and "314.999" in out.response.notes[0]
    assert out.response.diagnostics["retrieval_mode"] in ("none", "fallback")
    assert reg.requirement_state["r1"].fallback_run


def test_outside_title_21_is_not_attempted_and_is_diagnosed():
    reg = run("42 CFR 493.1").registry
    st = reg.requirement_state["r1"]
    assert not st.exact_attempted and not st.resolved and "outside Title 21" in st.note and not st.fallback_run


def test_missing_paragraph_with_existing_parent_skips_discovery_and_says_so():
    db = TableDatabase(trigram=[row("822.10", "", 0, sim=0.7)])
    reg = run("What does 820.10(b)(5) require?", db).registry
    st = reg.requirement_state["r1"]
    assert not st.resolved and st.exact_attempted and not st.fallback_run and st.discovery_hits == 0
    assert "parent 21 CFR 820.10(b) exists" in st.note
    assert labels(reg.structural_for("r1")) == ["21 CFR 820.10(b)"]
    assert "ECFR_TRIGRAM" not in db.names()


def test_one_missing_requirement_does_not_affect_the_others():
    reg = run("820.10(b)(1), (b)(2), (b)(9) and (b)(4)").registry
    assert reg.resolved_requirements() == ("r1", "r2", "r4") and reg.unresolved_requirements() == ("r3",)
    assert labels(reg.exact_for("r4")) == ["21 CFR 820.10(b)(4)"]


# -------------------------------------------------------- protection / top_k


def test_exact_evidence_survives_a_small_top_k():
    high = [row("801.1", "(a)", 1, rank=2.0), row("801.2", "", 0, rank=1.9), row("801.3", "", 0, rank=1.8)]
    out = run("820.10(b)(1) through (b)(4)", TableDatabase(full_text=high), top_k=2)
    got = labels(out.response.results)
    assert got[:4] == [f"21 CFR 820.10(b)({i})" for i in range(1, 5)]
    assert "21 CFR 820.10(b)" in got                     # structural parent is protected too
    assert not any(g.startswith("21 CFR 801") for g in got)  # top_k already spent on protected evidence
    reg = out.registry
    assert len(reg.by_tier(TIER_SUPPORTING)) == 3         # the registry still holds everything


def test_top_k_bounds_only_the_supporting_remainder():
    high = [row(f"801.{i}", "", 0, rank=2.0 - i / 10) for i in range(1, 7)]
    out = run("820.10(b)(2)", TableDatabase(full_text=high), top_k=4)
    got = labels(out.response.results)
    assert got[:2] == ["21 CFR 820.10(b)(2)", "21 CFR 820.10(b)"]
    assert len(got) == 4 and got[2:] == ["21 CFR 801.1", "21 CFR 801.2"]


def test_exact_evidence_outranks_higher_scoring_supporting_evidence():
    high = [row("801.1", "(a)", 1, rank=5.0)]
    out = run("820.10(b)(2)", TableDatabase(full_text=high))
    e = out.registry.evidence
    assert e[0].tier == TIER_EXACT and e[0].citation == "21 CFR 820.10(b)(2)"
    assert e[1].tier == TIER_STRUCTURAL
    assert e[2].tier == TIER_SUPPORTING and e[2].citation == "21 CFR 801.1(a)"


def test_full_text_hit_that_is_the_requested_row_is_exact_by_identity():
    # identity, not method, decides: the same row found by full-text and exact merges into one exact evidence
    db = TableDatabase(full_text=[row("820.10", "(b)(2)", 4, rank=0.7)])
    reg = run("820.10(b)(2)", db).registry
    hits = [e for e in reg.evidence if e.citation == "21 CFR 820.10(b)(2)"]
    assert len(hits) == 1 and hits[0].tier == TIER_EXACT and set(hits[0].result.methods) == {METHOD_EXACT_CFR, METHOD_FULL_TEXT}


# ------------------------------------------------------- guidance / general


def test_docket_question_keeps_exact_docket_retrieval():
    db = TableDatabase(docket=[guidance_row(0), guidance_row(1)])
    out = run("Summarise FDA-2016-D-2565", db)
    assert "GUIDANCE_EXACT_DOCKET" in db.names()
    top = out.registry.evidence[0]
    assert top.retrieval_method == METHOD_EXACT_DOCKET and top.tier == TIER_EXACT and top.satisfies == ()
    assert out.response.diagnostics["plan"]["docket_hits"] == 2 and out.response.notes == []


def test_missing_docket_is_noted():
    out = run("Summarise FDA-2023-D-99999", TableDatabase())
    assert out.response.notes and "FDA-2023-D-99999" in out.response.notes[0]


def test_cfr_requirement_alongside_guidance_language_is_still_retrieved():
    db = TableDatabase(guidance=[guidance_row(0, rank=0.3)])
    reg = run("What guidance exists on 21 CFR 820.10(b)(2)?", db).registry
    assert labels(reg.exact_for("r1")) == ["21 CFR 820.10(b)(2)"]
    assert any(e.corpus == "fda_guidance" and e.tier == TIER_SUPPORTING for e in reg.evidence)


def test_general_question_uses_the_legacy_text_path_and_question_fallback():
    db = TableDatabase(or_ecfr=[row("820.30", "(a)", 1, rank=0.4)])
    out = run("What are the design control requirements for medical devices?", db)
    assert out.registry.plan.requirements == () and out.registry.unresolved_requirements() == ()
    assert "ECFR_FULL_TEXT" in db.names() and "GUIDANCE_FULL_TEXT" in db.names()
    assert out.response.diagnostics["plan"]["question_fallback_used"] is True
    assert any(" or " in p.get("query", "") for _, p in db.calls)
    assert all(e.tier in (TIER_SUPPORTING, TIER_DISCOVERY) for e in out.registry.evidence)


def test_general_question_with_enough_primary_hits_skips_fallback():
    db = TableDatabase(full_text=[row(f"801.{i}", "", 0, rank=0.5) for i in range(1, 5)])
    out = run("labeling requirements for prescription drugs", db)
    assert out.response.diagnostics["fallback_used"] is False and out.response.diagnostics["retrieval_mode"] == "primary"


def test_corpus_restriction_to_guidance_leaves_cfr_requirements_unresolved_and_diagnosed():
    reg = run("820.10(b)(2)", corpus="fda_guidance").registry
    st = reg.requirement_state["r1"]
    assert not st.exact_attempted and "not selected" in st.note and reg.unresolved_requirements() == ("r1",)


# -------------------------------------------------------------- diagnostics


def test_diagnostics_answer_why_a_requirement_failed():
    out = run("820.10(b)(2) and 820.10(b)(9)")
    d = out.response.diagnostics["plan"]
    assert d["kind"] == "multi_provision" and d["requirement_count"] == 2 and d["requirement_ids"] == ["r1", "r2"]
    assert d["exact_attempts"] == 2 and d["resolved"] == ["r1"] and d["unresolved"] == ["r2"]
    assert d["tier_counts"][TIER_EXACT] == 1 and d["tier_counts"][TIER_STRUCTURAL] >= 1
    r2 = d["requirements"]["r2"]
    # (b)(9) is missing but its parent (b) exists: diagnosed, no broad search
    assert r2["exact_hits"] == 0 and r2["structural_hits"] == 1 and not r2["fallback_run"]
    assert "No live record" in r2["note"] and "parent 21 CFR 820.10(b) exists" in r2["note"]
    assert set(d["method_counts"]) <= {METHOD_EXACT_CFR, METHOD_FULL_TEXT, METHOD_FULL_TEXT_OR, METHOD_TRIGRAM, METHOD_EXACT_DOCKET}
    for k in ("primary_result_count", "fallback_used", "fallback_terms", "fallback_candidate_count", "final_result_count", "retrieval_mode"):
        assert k in out.response.diagnostics
    import json
    json.dumps(out.registry.to_dict())
    json.dumps(out.to_dict())


def test_registry_lookups_and_evidence_ids_are_stable():
    reg = run("820.10(b)(2) and 820.10(b)(4)").registry
    assert [e.evidence_id for e in reg.evidence] == [f"e{i}" for i in range(1, len(reg.evidence) + 1)]
    ids = lambda items: sorted(e.evidence_id for e in items)
    assert ids(reg.evidence_for("r1")) == sorted(ids(reg.exact_for("r1")) + ids(reg.structural_for("r1")))
    assert reg.all_evidence() == reg.evidence and reg.protected[0].tier == TIER_EXACT
    again = run("820.10(b)(2) and 820.10(b)(4)").registry
    assert [e.key for e in again.evidence] == [e.key for e in reg.evidence]


# ------------------------------------------------------- pipeline wiring


def test_pipeline_builds_one_plan_and_passes_it_to_run_plan():
    class Spy(LexicalRetriever):
        def __init__(self, db):
            super().__init__(db)
            self.plans = []

        def run_plan(self, plan, **kw):
            self.plans.append(plan)
            return super().run_plan(plan, **kw)

        def run(self, *a, **kw):  # the legacy path must not be used by the pipeline
            raise AssertionError("run() called")

    spy = Spy(TableDatabase())
    out = RagPipeline(spy, FakeGenerator("Answer [1]."), RagConfig(top_k=8)).answer("What do 820.10(b)(2) and 820.10(b)(4) require?")
    assert len(spy.plans) == 1 and spy.plans[0].question == "What do 820.10(b)(2) and 820.10(b)(4) require?"
    assert isinstance(out.evidence, EvidenceRegistry)
    assert out.metadata["plan"]["kind"] == "multi_provision"
    assert out.metadata["evidence"]["resolved"] == ["r1", "r2"]
    assert [s.result.citation.label for s in out.context.sources[:2]] == ["21 CFR 820.10(b)(2)", "21 CFR 820.10(b)(4)"]


def test_pipeline_legacy_retriever_without_run_plan_still_works():
    from tests.test_rag import E, FakeRetriever

    out = RagPipeline(FakeRetriever([E()]), FakeGenerator()).answer("What does 21 CFR 820.10(b) say?")
    assert out.evidence is None and out.metadata["evidence"] is None and out.model_called


def test_legacy_run_is_unchanged_in_shape():
    db = TableDatabase()
    resp = LexicalRetriever(db).run("What does 820.10(b)(2) require?", corpus="ecfr")
    assert resp.results[0].citation.label == "21 CFR 820.10(b)(2)" and "ECFR_EXACT_ROW" not in db.names()


# ------------------------------------------- C1: part-level stems ranked by relevance
#
# A part-level request makes every section stem of the part exact evidence with
# the same identifier score, so before C1 the only thing ordering them was the
# ASCII sort of their key. On S-002 ("Where are the definitions that apply to
# 21 CFR Part 111 located...") that put 21 CFR 111.3 - the part's definitions
# section - 46th of 84 stems, behind 111.103/111.105/111.110/..., and the
# bounded context dropped it. These tests cover the production transformation;
# the live ts_rank_cd signal itself is covered in tests/integration.

# Part 111 as the defect saw it: the governing section sorts late in ASCII order
# ("111.3" > "111.103"), and only it answers a question about definitions.
PART_111 = [
    row("111.3", "", 0, text="The definitions in section 201 of the act apply, and the following apply as well."),
    row("111.103", "", 0, text="written procedures for quality control operations"),
    row("111.105", "", 0, text="what quality control personnel must do"),
    row("111.110", "", 0, text="quality control operations for laboratory operations"),
    row("111.117", "", 0, text="quality control operations for equipment and utensils"),
]
# the relevance the live statement would compute: a heading/body hit on
# "definitions" for 111.3, nothing for the quality-control sections
PART_111_RANKED = [dict(r, part_rank=0.56 if r["section_number"] == "111.3" else 0.0) for r in PART_111]

S002 = "Where are the definitions that apply to 21 CFR Part 111 located, and does Part 111 incorporate definitions from elsewhere?"


def part_order(question, table=PART_111_RANKED, **kw):
    reg = run(question, TableDatabase(table), **kw).registry
    return labels(reg.exact_for("r1"))


def test_part_stems_are_ordered_by_question_relevance_not_by_key():
    """Test A: the S-002 case. 111.3 must lead, ahead of the ASCII-first stems."""
    assert part_order(S002)[0] == "21 CFR 111.3"
    # and the ranking is carried by the score, which is what selection reads
    reg = run(S002, TableDatabase(PART_111_RANKED)).registry
    stems = reg.exact_for("r1")
    assert [e.citation for e in stems] == ["21 CFR 111.3", "21 CFR 111.103", "21 CFR 111.105",
                                          "21 CFR 111.110", "21 CFR 111.117"]
    assert stems[0].score > stems[1].score > stems[2].score  # strictly ordered, no ties left
    assert stems[0].result.metadata["part_relevance_rank"] == 0
    assert stems[0].result.metadata["ts_rank_cd"] > 0.0
    assert all(e.result.metadata["ts_rank_cd"] == 0.0 for e in stems[1:])
    # every stem is still exact evidence for the part: ranking orders, it never filters
    assert len(stems) == len(PART_111) and reg.unresolved_requirements() == ()


def test_part_stem_ranking_asks_the_database_for_the_questions_topical_terms():
    """The relevance signal is the Stage 7 term selection, so the generic regulatory
    words that would otherwise drive the ranking are not in the query at all."""
    db = TableDatabase(PART_111_RANKED)
    run(S002, db)
    params = next(p for n, p in db.calls if n == "ECFR_EXACT_PART")
    terms = params["query"].split(" or ")
    assert "definitions" in terms and "incorporate" in terms
    for generic in ("requirements", "labeling", "records", "procedures", "applicable", "section", "part"):
        assert generic not in terms, generic
    # OR, not AND: a part-level question rarely has every term in one section stem
    assert " or " in params["query"]


def test_part_stem_scores_stay_inside_the_exact_band():
    """The rank is spent as the same 0.001 step exact_score spends on document order, so
    exact evidence still outranks descendants, structural context and every text match."""
    stems = run(S002, TableDatabase(PART_111_RANKED)).registry.exact_for("r1")
    assert all(99.9 < e.score <= 100.0 for e in stems)
    # exact_row_limit bounds the rank, so the band cannot grow without bound
    from regulus.retrieval.ranking import EXACT_DESCENDANT_SCORE, STRUCTURAL_SCORE, part_stem_score
    assert part_stem_score(LexicalRetriever(TableDatabase()).exact_row_limit - 1) > EXACT_DESCENDANT_SCORE > STRUCTURAL_SCORE


def test_a_part_question_with_no_topical_term_keeps_the_pre_c1_scores():
    """No term survives selection -> no relevance signal -> the pre-C1 score and order.
    Reached through exact_cfr directly, the way run() reaches it."""
    db = TableDatabase(PART_111_RANKED)
    ref = plan_query("21 CFR part 111").requirements[0].reference
    rows = LexicalRetriever(db).exact_cfr(ref, Filters(), 100)
    params = next(p for n, p in db.calls if n == "ECFR_EXACT_PART")
    assert params["query"] == ""
    assert all(r.relevance_score == 100.0 for r in rows)
    assert "part_relevance_rank" not in rows[0].metadata
    assert [r.citation.label for r in rows][0] == "21 CFR 111.103"  # ASCII order, as before C1


def test_part_stem_ranking_is_deterministic():
    """Test B: same inputs, same order and same scores, every time."""
    runs = [[(e.citation, e.score) for e in run(S002, TableDatabase(PART_111_RANKED)).registry.exact_for("r1")]
            for _ in range(5)]
    assert all(r == runs[0] for r in runs[1:])
    # equally irrelevant stems are not left to chance either: they keep section order
    flat = [dict(r, part_rank=0.0) for r in PART_111]
    runs = [part_order(S002, flat) for _ in range(5)]
    assert all(r == runs[0] for r in runs[1:])
    assert runs[0] == ["21 CFR 111.103", "21 CFR 111.105", "21 CFR 111.110", "21 CFR 111.117", "21 CFR 111.3"]


@pytest.mark.usefixtures("cpu_era_budget")   # fixtures sized to overflow the CPU-era evidence budget (tests/conftest.py)
def test_relevance_wins_when_it_conflicts_with_alphabetical_order():
    """Test D: the relevant stem sorts LAST in ASCII order and still leads - and with a
    budget that fits two sources it is kept while the ASCII-first stems are dropped."""
    from regulus.rag.context import ContextPolicy
    from regulus.rag.selection import select_evidence

    reg = run(S002, TableDatabase(PART_111_RANKED)).registry
    ascii_first = sorted(e.citation for e in reg.exact_for("r1"))[0]
    assert ascii_first == "21 CFR 111.103"  # what the old ordering selected first
    assert labels(reg.exact_for("r1"))[0] == "21 CFR 111.3"

    tight = ContextPolicy(max_results=2, hard_max_results=2)
    kept = [s.evidence.citation for s in select_evidence(reg, tight).selected]
    assert kept[0] == "21 CFR 111.3"
    assert "21 CFR 111.105" not in kept and "21 CFR 111.110" not in kept


def test_relevance_ranking_does_not_touch_other_retrieval_paths():
    """Test C: section-, paragraph- and question-level retrieval are unchanged."""
    db = TableDatabase()
    reg = run("What does 21 CFR 820.10(b) require?", db).registry
    # the section statement is never asked for a relevance ranking
    assert all("query" not in p for n, p in db.calls if n.startswith("ECFR_EXACT_S") or n == "ECFR_EXACT_ROW")
    # exact rows keep the documented identifier scores, descendants keep document order
    assert labels(reg.exact_for("r1")) == ["21 CFR 820.10(b)"]
    assert reg.exact_for("r1")[0].score == 100.0 - 0.001 * 2
    assert labels(reg.structural_for("r1")) == [
        "21 CFR 820.10", "21 CFR 820.10(b)(1)", "21 CFR 820.10(b)(2)",
        "21 CFR 820.10(b)(3)", "21 CFR 820.10(b)(4)",
    ]
    # supporting evidence still orders by score alone
    sup = [row("211.1", "", 0, rank=0.2), row("211.2", "", 0, rank=0.9)]
    reg = run("What does 21 CFR 820.10(b) require?", TableDatabase(full_text=sup)).registry
    supporting = [e for e in reg.evidence if e.tier == TIER_SUPPORTING]
    assert labels(supporting) == ["21 CFR 211.2", "21 CFR 211.1"]


def test_legacy_query_level_part_lookup_is_unchanged():
    """Test C, second path: run() (hybrid retriever, scripts, eval baselines) asks for no
    ranking, so a part lookup there returns the statement's plain section order."""
    db = TableDatabase(PART_111_RANKED)
    resp = LexicalRetriever(db).run("What is in 21 CFR part 111?", corpus="ecfr", top_k=10)
    params = next(p for n, p in db.calls if n == "ECFR_EXACT_PART")
    assert params["query"] == ""
    exact = [r for r in resp.results if r.retrieval_method == METHOD_EXACT_CFR]
    assert all(r.relevance_score == 100.0 for r in exact)
    assert [r.citation.label for r in exact] == sorted(r.citation.label for r in exact)


# ------------------------------------- C2: a ranked part section supplies its own paragraphs
#
# ECFR_EXACT_PART returns the FIRST row of each section (`ordinal = 0`). For many
# sections that row is a lead-in: 21 CFR 111.3's is "The definitions and
# interpretations of terms in section 201 of the act apply ...", and the 28
# definitions it introduces are the rows below it. C1 ranks 111.3 first; C2 lets
# it supply those rows, through the same ECFR_EXACT_SECTION statement a
# section-level request uses. Only sections the question MATCHES are expanded,
# at most part_expand_sections of them and part_expand_rows rows each.

# 111.3 is a lead-in plus three definitions; 111.105 is a lead-in plus (a),(b),(c)
PART_111_DEEP = PART_111_RANKED + [
    row("111.3", "", 1, text="Actual yield means the quantity that is actually produced.", path_ordinal=2),
    row("111.3", "", 2, text="Batch means a specific quantity of a dietary supplement.", path_ordinal=3),
    row("111.3", "", 3, text="Component means any substance intended for use.", path_ordinal=4),
    row("111.105", "(a)", 1, text="(a) Approving or rejecting all processes and specifications;"),
    row("111.105", "(b)", 2, text="(b) Reviewing and approving the documentation for suppliers;"),
    row("111.105", "(c)", 3, text="(c) Reviewing and approving in-process specifications;"),
]


def test_a_ranked_part_section_supplies_its_substantive_rows():
    """Test A: the S-002 case. 111.3 reaches the prompt with its definitions, not just
    the lead-in that introduces them."""
    db = TableDatabase(PART_111_DEEP)
    reg = LexicalRetriever(db, part_expand_rows=3).run_plan(plan_query(S002)).registry
    rows_1113 = [e for e in reg.exact_for("r1") if e.result.metadata["section"] == "111.3"]
    assert len(rows_1113) == 4  # the lead-in plus its three definitions
    assert [e.result.metadata["ordinal"] for e in rows_1113] == [0, 1, 2, 3]
    assert "Actual yield means" in rows_1113[1].text
    # the stem is still the stem: C1's ranking metadata is untouched
    assert rows_1113[0].result.metadata["part_relevance_rank"] == 0
    assert "part_expanded_from" not in rows_1113[0].result.metadata
    assert all(e.result.metadata["part_expanded_from"] == "111.3" for e in rows_1113[1:])
    # the expanded rows are exact evidence for the part requirement, like every row of it
    assert all(e.tier == TIER_EXACT and e.satisfies == ("r1",) for e in rows_1113)


def test_expanded_rows_follow_their_own_section_in_document_order():
    """Test B: a multi-paragraph section keeps (a),(b),(c) order, and the sections keep
    C1's relevance order around them."""
    # a question that matches both 111.3 and 111.105
    table = [dict(r, part_rank={"111.3": 0.9, "111.105": 0.5}.get(r["section_number"], 0.0))
             if r["ordinal"] == 0 else dict(r) for r in PART_111_DEEP]
    reg = LexicalRetriever(TableDatabase(table), part_expand_rows=3).run_plan(plan_query(S002)).registry
    seen = [(e.result.metadata["section"], e.result.metadata["ordinal"]) for e in reg.exact_for("r1")]
    assert seen[:4] == [("111.3", 0), ("111.3", 1), ("111.3", 2), ("111.3", 3)]
    assert seen[4:8] == [("111.105", 0), ("111.105", 1), ("111.105", 2), ("111.105", 3)]
    paras = [e.result.metadata["paragraph_path"] for e in reg.exact_for("r1")[4:8]]
    assert paras == [None, "(a)", "(b)", "(c)"]
    assert [e.citation for e in reg.exact_for("r1")[5:8]] == ["21 CFR 111.105(a)", "21 CFR 111.105(b)", "21 CFR 111.105(c)"]


def test_only_matching_sections_are_expanded_and_only_up_to_the_bounds():
    """Test E, first half: C2 is bounded on both axes and never expands a section the
    question does not match, so a part question cannot pull in the whole part."""
    db = TableDatabase(PART_111_DEEP)
    run(S002, db)
    expanded = [p["section_number"] for n, p in db.calls if n == "ECFR_EXACT_SECTION"]
    assert expanded == ["111.3"]  # 111.105 has ts_rank 0 for this question: not expanded
    # the row cap is passed to the statement, +1 for the stem it skips
    r = LexicalRetriever(TableDatabase(PART_111_DEEP), part_expand_rows=2)
    assert [e.result.metadata["ordinal"] for e in r.run_plan(plan_query(S002)).registry.exact_for("r1")
            if e.result.metadata["section"] == "111.3"] == [0, 1, 2]
    # and the section cap bounds how many sections are deepened
    table = [dict(r, part_rank=0.5) if r["ordinal"] == 0 else dict(r) for r in PART_111_DEEP]
    db = TableDatabase(table)
    LexicalRetriever(db, part_expand_sections=2).run_plan(plan_query(S002))
    assert len([n for n, _ in db.calls if n == "ECFR_EXACT_SECTION"]) == 2


def test_part_expansion_is_off_without_a_relevance_signal():
    """No C1 ranking signal -> nothing to deepen -> exactly the pre-C1 part behaviour."""
    db = TableDatabase(PART_111_DEEP)
    ref = plan_query("21 CFR part 111").requirements[0].reference
    rows = LexicalRetriever(db).exact_cfr(ref, Filters(), 100)
    assert not [n for n, _ in db.calls if n == "ECFR_EXACT_SECTION"]
    assert all(r.relevance_score == 100.0 for r in rows)
    assert len(rows) == 5  # the five stems, nothing deeper


@pytest.mark.usefixtures("cpu_era_budget")   # fixtures sized to overflow the CPU-era evidence budget (tests/conftest.py)
def test_part_expansion_respects_the_existing_ceilings():
    """Test E: the expanded rows go through select_evidence like any required evidence -
    they are bounded by the hard ceilings and recorded when dropped, never smuggled in."""
    from regulus.rag.context import ContextPolicy
    from regulus.rag.selection import DROP_HARD_CEILING_COUNT, select_evidence

    reg = LexicalRetriever(TableDatabase(PART_111_DEEP), part_expand_rows=3).run_plan(plan_query(S002)).registry
    assert len(reg.exact_for("r1")) == 8  # 5 stems + 3 expanded rows of 111.3

    full = select_evidence(reg)
    assert len(full.selected) == 8 and not full.diagnostics["required_dropped"]

    tight = select_evidence(reg, ContextPolicy(max_results=3, hard_max_results=3))
    assert len(tight.selected) == 3
    assert len(tight.diagnostics["required_dropped"]) == 5
    assert all(d.reason == DROP_HARD_CEILING_COUNT for d in tight.dropped if d.required)
    # depth goes to the ranked section first, so the ceiling cuts the irrelevant tail
    assert [s.evidence.result.metadata["section"] for s in tight.selected] == ["111.3"] * 3

    chars = select_evidence(reg, ContextPolicy(max_results=8, max_total_chars=300, hard_max_total_chars=300))
    assert sum(s.chars for s in chars.selected) <= 300
    assert chars.diagnostics["required_dropped"]


def test_part_expansion_default_bounds_are_the_measured_ones():
    """The defaults are tuned against the existing ceilings (see lexical.py); a change to
    them changes what every part-level question sends to the model."""
    r = LexicalRetriever(TableDatabase())
    assert (r.part_expand_sections, r.part_expand_rows) == (3, 2)
    reg = run(S002, TableDatabase(PART_111_DEEP)).registry
    rows = [e for e in reg.exact_for("r1") if e.result.metadata["section"] == "111.3"]
    assert [e.result.metadata["ordinal"] for e in rows] == [0, 1, 2]  # lead-in + 2 rows


def test_part_expansion_is_deterministic():
    """Test F: identical input, identical evidence order, scores and selection."""
    from regulus.rag.selection import select_evidence

    def once():
        reg = run(S002, TableDatabase(PART_111_DEEP)).registry
        sel = select_evidence(reg)
        return ([(e.citation, e.score, e.result.metadata["ordinal"]) for e in reg.evidence],
                [s.evidence.key for s in sel.selected])

    runs = [once() for _ in range(5)]
    assert all(r == runs[0] for r in runs[1:])


def test_section_level_retrieval_is_unchanged_by_c2():
    """Test C: a section-level request still uses ECFR_EXACT_SECTION the same way, with
    its own paragraph scores, and asks for no part expansion."""
    db = TableDatabase()
    reg = run("What does 21 CFR 820.10 say?", db).registry
    calls = [p for n, p in db.calls if n == "ECFR_EXACT_SECTION"]
    from regulus.rag.validator_switches import enabled
    from regulus.retrieval.lexical import XR1_EXACT_ROW_LIMIT
    # XR1 (27 Oct release): a section is fetched whole up to 400 rows; without it the 100-row limit
    assert len(calls) == 1 and calls[0]["paragraph"] == "" and calls[0]["limit"] == (XR1_EXACT_ROW_LIMIT if enabled("XR1") else 100)
    assert len(reg.exact_for("r1")) == 8
    assert labels(reg.exact_for("r1"))[:3] == ["21 CFR 820.10", "21 CFR 820.10(a)", "21 CFR 820.10(b)"]
    assert [e.score for e in reg.exact_for("r1")][:3] == [100.0, 100.0 - 0.001, 100.0 - 0.002]
    assert not any("part_expanded_from" in e.result.metadata for e in reg.evidence)


def test_non_part_question_issues_no_expansion_query():
    """Test D: a paragraph-level question's retrieval is untouched by C2."""
    db = TableDatabase()
    reg = run("What does 21 CFR 820.10(b)(2) require?", db).registry
    assert [n for n, _ in db.calls if n.startswith("ECFR_EXACT")] == ["ECFR_EXACT_SECTION", "ECFR_EXACT_ROW"]
    assert labels(reg.exact_for("r1")) == ["21 CFR 820.10(b)(2)"]
    assert not any("part_expanded_from" in e.result.metadata for e in reg.evidence)
