"""C3: cross-section range expansion (Stage 1, corpus-aware).

`scan_references` parses "21 CFR 50.50 through 50.56" into its two endpoints and
notes that it cannot do more: which sections lie between them is a fact about the
corpus, not about the identifier, and guessing would invent provisions. These
tests cover the step that finishes the job from a section listing
(regulus.planning.ranges) and the retrieval it feeds.

The listing is passed in, so everything here runs without a database; the live
corpus is covered in tests/integration/test_retrieval_db.py.
"""

import pytest

from regulus.planning import plan_query
from regulus.planning.plan import KIND_SINGLE
from regulus.planning.ranges import expand_section_ranges, sections_between
from regulus.retrieval.references import (
    MAX_RANGE_EXPANSION,
    ORIGIN_RANGE,
    scan_references,
    section_range_endpoints,
    section_sort_key,
)

# Part 50 as this corpus holds it: sort_order is document order, which ASCII order
# gets wrong (50.3 sorts after 50.20 lexically).
PART_50 = [("50.1", 0), ("50.3", 1), ("50.20", 2), ("50.22", 3), ("50.23", 4), ("50.24", 5),
           ("50.25", 6), ("50.27", 7), ("50.50", 8), ("50.51", 9), ("50.52", 10), ("50.53", 11),
           ("50.54", 12), ("50.55", 13), ("50.56", 14)]
# Part 820: 820.30 is not a standalone section here, it sits inside a consolidated block
PART_820 = [("820.1", 0), ("820.3", 1), ("820.5", 2), ("820.7", 3), ("820.10", 4),
            ("820.20-820.30", 5), ("820.35", 6), ("820.40", 7), ("820.45", 8)]
CORPUS = {"50": PART_50, "820": PART_820}


def listing(part):
    return CORPUS.get(part, [])


def expand(question, sections=listing):
    plan, notes = expand_section_ranges(plan_query(question), sections)
    return [r.canonical for r in plan.requirements], notes, plan


# ------------------------------------------------------------------ A: basic range


def test_basic_range_requests_every_existing_intermediate_section():
    """Test A: 50.50 through 50.56 is seven sections, not two."""
    labels, notes, plan = expand("What do 21 CFR 50.50 through 50.56 require?")
    assert labels == [f"21 CFR 50.{n}" for n in (50, 51, 52, 53, 54, 55, 56)]
    assert [r.origin for r in plan.requirements] == ["explicit"] + [ORIGIN_RANGE] * 5 + ["explicit"]
    assert any("expanded to 7 sections" in n for n in notes)
    # the endpoints keep their own provenance and the inserted ones point at the range text
    assert plan.requirements[1].raw.startswith("21 CFR 50.50")
    assert [r.id for r in plan.requirements] == [f"r{i}" for i in range(1, 8)]
    assert [r.order for r in plan.requirements] == list(range(1, 8))


def test_the_scanner_itself_is_unchanged_and_still_refuses_to_guess():
    """C3 adds a corpus-aware step; it does not make the parser invent sections."""
    scan = scan_references("21 CFR 50.50 through 50.56")
    assert [r.label for r in scan.references] == ["21 CFR 50.50", "21 CFR 50.56"]
    assert scan.ranges and not scan.ranges[0].expanded
    assert any("not expanded" in n for n in scan.notes)


# ------------------------------------------------------------------ B: two-section range


def test_two_section_range_does_not_reach_past_its_end():
    """Test B: 50.50 through 50.51 is exactly those two."""
    labels, notes, _ = expand("21 CFR 50.50 through 50.51")
    assert labels == ["21 CFR 50.50", "21 CFR 50.51"]
    assert all("50.52" not in n for n in notes)
    labels, _, _ = expand("21 CFR 50.50 through 50.52")
    assert labels == ["21 CFR 50.50", "21 CFR 50.51", "21 CFR 50.52"]


# ------------------------------------------------------------------ C: reversed range


def test_reversed_range_keeps_its_endpoints_as_it_already_did_for_paragraphs():
    """Test C: the existing rule is "the end must follow the start, otherwise keep both
    endpoints and expand nothing" (test_reversed_range_keeps_endpoints). C3 declines the
    same way rather than inventing a descending semantic."""
    a, b = plan_query("21 CFR 50.56").requirements[0], plan_query("21 CFR 50.50").requirements[0]
    assert section_range_endpoints(a.reference, b.reference) is None
    labels, notes, plan = expand("21 CFR 50.56 through 50.50")
    assert labels == ["21 CFR 50.56", "21 CFR 50.50"]
    assert not notes and plan.notes == plan_query("21 CFR 50.56 through 50.50").notes


# ------------------------------------------------------------------ D: missing sections


def test_a_section_the_corpus_does_not_have_is_never_fabricated():
    """Test D: only the sections that exist between the endpoints are requested."""
    gapped = [("50.50", 8), ("50.52", 10), ("50.56", 14)]  # 50.51, 50.53, 50.54, 50.55 absent
    labels, notes, _ = expand("21 CFR 50.50 through 50.56", lambda part: gapped)
    assert labels == ["21 CFR 50.50", "21 CFR 50.52", "21 CFR 50.56"]
    assert any("50.52" in n for n in notes)


def test_a_range_with_nothing_in_between_is_left_alone_and_said_so():
    labels, notes, _ = expand("21 CFR 50.50 through 50.56", lambda part: [("50.50", 8), ("50.56", 14)])
    assert labels == ["21 CFR 50.50", "21 CFR 50.56"]
    assert any("no section between the endpoints" in n for n in notes)


def test_an_endpoint_the_corpus_does_not_have_still_bounds_the_range():
    """820.30 is not a standalone section in this corpus (it sits in a consolidated
    820.20-820.30 block), and the range must still work off its parsed number."""
    labels, notes, _ = expand("21 CFR 820.30 through 820.40")
    assert labels == ["21 CFR 820.30", "21 CFR 820.35", "21 CFR 820.40"]
    assert any("820.35" in n for n in notes)


# ------------------------------------------------------------------ E: numeric ordering


def test_bounds_and_order_are_numeric_not_lexical():
    """Test E: ASCII order puts 50.3 after 50.20 and 7.100 before 7.20; document order
    does not. The bound test and the output order must both use the numeric key."""
    assert section_sort_key("50.3") < section_sort_key("50.20") < section_sort_key("50.50")
    assert sorted(["50.50", "50.3", "50.20", "50.56"], key=section_sort_key) == ["50.3", "50.20", "50.50", "50.56"]
    # a letter suffix sorts after the bare number; a consolidated block keys off its start
    assert section_sort_key("184.1141") < section_sort_key("184.1141a") < section_sort_key("184.1141b")
    assert section_sort_key("820.20-820.30") == section_sort_key("820.20")
    assert section_sort_key("not a section") is None

    # 50.3 through 50.50 must pick up 50.20..50.27, which lexical bounds would drop
    labels, _, _ = expand("21 CFR 50.3 through 50.50")
    assert labels == ["21 CFR 50.3", "21 CFR 50.20", "21 CFR 50.22", "21 CFR 50.23",
                      "21 CFR 50.24", "21 CFR 50.25", "21 CFR 50.27", "21 CFR 50.50"]

    odd = [("7.3", 0), ("7.20", 1), ("7.100", 2)]  # lexical order would be 7.100, 7.20, 7.3
    assert sections_between(odd, section_sort_key("7.3"), section_sort_key("7.100")) == ["7.20"]


def test_output_order_follows_the_corpus_sort_order():
    """The listing is ordered by the corpus's own sort_order, whatever order it arrives in."""
    shuffled = list(reversed(PART_50))
    assert sections_between(shuffled, section_sort_key("50.50"), section_sort_key("50.56")) == \
        ["50.51", "50.52", "50.53", "50.54", "50.55"]


def test_oversized_range_keeps_its_endpoints_like_a_paragraph_range():
    big = [("1.1", 0)] + [(f"1.{i}", i) for i in range(2, MAX_RANGE_EXPANSION + 10)]
    labels, notes, _ = expand(f"21 CFR 1.1 through 1.{MAX_RANGE_EXPANSION + 9}", lambda part: big)
    assert labels == ["21 CFR 1.1", f"21 CFR 1.{MAX_RANGE_EXPANSION + 9}"]
    assert any(f"above the limit of {MAX_RANGE_EXPANSION}" in n for n in notes)


# ------------------------------------------------------------------ F/G: nothing else moves


@pytest.mark.parametrize("question", [
    "What does 21 CFR 50.50 require?",                      # single section
    "What does 21 CFR 820.10(b)(2) require?",               # paragraph
    "List the requirements in 21 CFR 820.10(b)(1) through (b)(4).",  # paragraph RANGE
    "What is in 21 CFR part 111?",                          # part level (C1/C2)
    "Compare 21 CFR 111.10 and 21 CFR 117.35.",             # two sections, no range
    "What are the labeling requirements for dietary supplements?",   # no reference
    "FDA-2016-D-2565",                                      # docket
])
def test_a_question_without_a_cross_section_range_is_untouched(question):
    """Tests F and G: the plan is returned as-is - same object, same requirements, same
    kind, same notes - for every question that does not name a cross-section range."""
    before = plan_query(question)
    after, notes = expand_section_ranges(before, listing)
    assert after is before and notes == ()


def test_a_paragraph_range_is_left_to_the_parser():
    """The parser already enumerates paragraph ranges; C3 must not touch them."""
    before = plan_query("21 CFR 820.10(b)(1) through (b)(4)")
    assert [r.canonical for r in before.requirements] == [f"21 CFR 820.10(b)({i})" for i in (1, 2, 3, 4)]
    after, notes = expand_section_ranges(before, listing)
    assert after is before and notes == ()


def test_a_range_with_paragraph_endpoints_is_declined():
    """"50.50(a) through 50.56(c)" asks for paragraphs across sections; C3 does not
    invent a semantic for it and leaves the parser's endpoints standing."""
    labels, notes, _ = expand("21 CFR 50.50(a) through 50.56(c)")
    assert labels == ["21 CFR 50.50(a)", "21 CFR 50.56(c)"]
    assert notes == ()


def test_a_range_across_parts_is_declined():
    labels, notes, _ = expand("21 CFR 50.50 through 56.10")
    assert labels == ["21 CFR 50.50", "21 CFR 56.10"]
    assert notes == ()


def test_the_plan_kind_is_recomputed_by_the_existing_rule():
    _, _, plan = expand("What do 21 CFR 50.50 through 50.56 require?")
    assert plan.kind == plan_query("What do 21 CFR 50.50 and 21 CFR 50.56 require?").kind
    assert plan_query("What does 21 CFR 50.50 require?").kind == KIND_SINGLE


# ------------------------------------------------------------------ I: determinism


def test_expansion_is_deterministic():
    """Test I: same question, same sections, same order, same ids, every time."""
    runs = [[(r.id, r.canonical, r.origin, r.order)
             for r in expand("What do 21 CFR 50.50 through 50.56 require?")[2].requirements]
            for _ in range(5)]
    assert all(r == runs[0] for r in runs[1:])
    assert len({tuple(expand("21 CFR 50.3 through 50.50")[1]) for _ in range(5)}) == 1


def test_no_duplicate_requirement_is_created():
    """A section the question already named explicitly is not added twice."""
    labels, _, plan = expand("What do 21 CFR 50.50, 21 CFR 50.52 and 21 CFR 50.50 through 50.53 require?")
    assert len(labels) == len(set(labels))
    assert len({r.key for r in plan.requirements}) == len(plan.requirements)


# ------------------------------------------------------------------ H: ceilings hold
#
# These drive the real retrieval and selection path with the expanded plan, so the
# existing 24-source / 9,000-character ceilings are the ones under test.

from tests.test_retrieval_plan import TableDatabase, row  # noqa: E402

# fixtures sized to overflow the CPU-era evidence budget (tests/conftest.py)
pytestmark = pytest.mark.usefixtures("cpu_era_budget")

# five sections of part 50, each with a long section row and two paragraphs
RANGE_TABLE = [r for n in (50, 51, 52, 53, 54) for r in (
    row(f"50.{n}", "", 0, part="50", text=f"Section 50.{n} body. " + "x" * 900),
    row(f"50.{n}", "(a)", 1, part="50", text=f"(a) 50.{n} first. " + "y" * 900),
    row(f"50.{n}", "(b)", 2, part="50", text=f"(b) 50.{n} second. " + "z" * 900),
)]


def range_registry(db=None, question="What do 21 CFR 50.50 through 50.54 require?"):
    from regulus.retrieval import LexicalRetriever

    db = db or TableDatabase(RANGE_TABLE)
    retriever = LexicalRetriever(db)
    plan, _ = expand_section_ranges(plan_query(question), retriever.sections_in_part)
    return plan, retriever.run_plan(plan, corpus="ecfr", top_k=8).registry


def test_the_expanded_range_goes_through_the_registry_and_is_bounded_by_the_ceilings():
    """Test H: every requested section is retrieved, and selection still enforces both
    hard ceilings, recording what it had to drop."""
    from regulus.rag.context import ContextPolicy
    from regulus.rag.selection import select_evidence

    plan, registry = range_registry()
    assert [r.canonical for r in plan.requirements] == [f"21 CFR 50.{n}" for n in (50, 51, 52, 53, 54)]
    assert registry.unresolved_requirements() == ()          # all five resolved in the registry
    assert len(registry.evidence) == 15                      # 5 sections x 3 rows

    policy = ContextPolicy()
    selection = select_evidence(registry, policy)
    assert len(selection.selected) <= policy.hard_max_results == 24
    assert selection.total_chars <= policy.hard_max_total_chars == 9000
    # 15 rows of ~950 chars cannot fit in 9,000: the overflow is recorded, not hidden
    assert selection.diagnostics["required_dropped"]
    assert selection.overflow
    assert len(selection.selected) + len(selection.diagnostics["required_dropped"]) == 15


def test_a_long_range_cannot_exceed_the_source_ceiling():
    many = [r for n in range(1, 40) for r in (row(f"50.{n}", "", 0, part="50", text=f"50.{n} body"),)]
    from regulus.rag.context import ContextPolicy
    from regulus.rag.selection import select_evidence

    plan, registry = range_registry(TableDatabase(many), "21 CFR 50.1 through 50.39")
    assert len(plan.requirements) == 39
    selection = select_evidence(registry, ContextPolicy())
    assert len(selection.selected) <= 24 and selection.total_chars <= 9000
    assert len(selection.diagnostics["required_dropped"]) == len(registry.evidence) - len(selection.selected)


def test_retrieval_of_an_expanded_range_is_deterministic():
    """Test I, through retrieval: identical evidence identity, order, scores and tiers."""
    from regulus.rag.selection import select_evidence

    def once():
        _, registry = range_registry()
        selection = select_evidence(registry)
        return ([(e.evidence_id, e.citation, e.score, e.tier, e.satisfies) for e in registry.evidence],
                [s.evidence.key for s in selection.selected])

    runs = [once() for _ in range(5)]
    assert all(r == runs[0] for r in runs[1:])


def test_range_members_are_ordinary_exact_evidence_for_their_own_requirement():
    """Tier and requirement association are the existing ones; C3 introduces no new
    evidence semantics."""
    plan, registry = range_registry()
    for req in plan.requirements:
        evidence = registry.exact_for(req.id)
        assert evidence, req.canonical
        assert all(e.tier == "exact" and req.id in e.satisfies for e in evidence)
    assert not any("part_expanded_from" in e.result.metadata for e in registry.evidence)
