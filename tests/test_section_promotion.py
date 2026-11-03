"""Section promotion, the answer-first format, the statute corpus and the refusal retry (22 Oct 2026)."""
from dataclasses import replace
from types import SimpleNamespace

from regulus.planning import plan_query
from regulus.rag import RagConfig, RagPipeline
from regulus.rag.answer_plan import APPLY_FORMAT
from regulus.rag.generation import render_package
from regulus.rag.pipeline import REFUSAL_RETRY_NOTE, _without_part_requirements
import pytest

from regulus.rag import section_promotion as SP
from regulus.inference.types import GenerationResult
from regulus.rag.section_promotion import Promotion, SectionPromoter, parse_picks, picker_prompt
from regulus.rag.validation import UNSUPPORTED_FIGURE, validate_answer
from regulus.retrieval import LexicalRetriever
from regulus.retrieval.section_index import SectionIndex, cosmetic_question
from tests.test_answer_validation import TWO, ScriptedGenerator, gen
from tests.test_generation_contract import package_for
from tests.test_retrieval_plan import TableDatabase


# ------------------------------------------------------------ picker reply parsing


def test_picker_reply_bracket_numbers_and_section_numbers():
    secs = ["71.1", "70.25", "73.2550"]
    assert parse_picks("[2], [1]", 20, secs) == [2, 1]
    # a reply naming a section is the candidate carrying that number, not candidates 70 and 25
    assert parse_picks("70.25", 20, secs) == [2]
    assert parse_picks("NONE", 20, secs) == []
    assert parse_picks("[99]", 3, secs) == []


def test_picker_prompt_lists_cfr_and_statute_candidates():
    p = picker_prompt("Q?", [
        {"section": "74.2203", "heading": "FD&C Green No. 3.", "part": "74", "part_title": "Listing", "subpart": "Subpart C - Cosmetics",
         "snippet": "may be safely used"},
        {"statute": True, "section": "605", "usc": "364a", "heading": "Adverse events", "snippet": "shall submit"}])
    assert "[1] 21 CFR 74.2203 FD&C Green No. 3." in p and "[2] FD&C Act § 605 [21 U.S.C. 364a] Adverse events" in p
    assert p.rstrip().endswith("Reply NONE if no candidate answers it.")


# ------------------------------------------------------------ siblings and part-only plans


def _cand(section, part, heading, subpart):
    return {"section": section, "part": part, "heading": heading, "subpart": subpart}


def test_color_listing_siblings_keep_cosmetics_and_part_82_not_drug_listing():
    pr = SectionPromoter(db=None, mode="lexical", index=object())
    cands = [_cand("82.1254", "82", "D&C Orange No. 4.", "Subpart D - Externally Applied Drugs and Cosmetics"),
             _cand("74.1254", "74", "D&C Orange No. 4.", "Subpart B - Drugs"),
             _cand("74.2254", "74", "D&C Orange No. 4.", "Subpart C - Cosmetics"),
             _cand("70.5", "70", "General restrictions on use of color additives.", "")]
    assert pr.siblings(cands[0], cands, ["toothpast", "orang"]) == ["82.1254", "74.2254"]
    # a drug question keeps the drug listing instead
    assert "74.1254" in pr.siblings(cands[1], cands, ["tablet", "orang"])


def test_part_only_question_loses_the_part_requirement_once_sections_are_resolved():
    p = plan_query("Under Part 82, what specifications must a batch of FD&C Green No. 3 meet? (21 CFR 82.203, 82.50)")
    out = _without_part_requirements(p)
    assert [(r.id, r.canonical) for r in out.requirements] == [("r1", "21 CFR 82.203"), ("r2", "21 CFR 82.50")]


def test_promotion_suffix_names_sections_then_companions():
    pr = Promotion(sections=("74.2203",), companions=("70.5(a)",), method="picker")
    assert pr.suffix == " (21 CFR 74.2203, 70.5(a))"
    assert Promotion(sections=(), companions=(), method="named_statute", statute=("605",)).suffix == ""


# ------------------------------------------------------------ section ranking


def _rows():
    def r(sec, part, ordinal, heading, body, head, subpart=""):
        return {"section_number": sec, "part_number": part, "ordinal": ordinal, "subpart": subpart,
                "section_heading": heading, "body": body, "head": head}
    return [r("71.2", "71", 0, "Notice of filing of petition.", ["petit", "file", "notic", "publish", "feder", "regist"], ["notic", "file", "petit"]),
            r("71.1", "71", 0, "Petitions.", ["petit", "color", "addit", "submit"], ["petit"]),
            r("74.2203", "74", 0, "FD&C Green No. 3.", ["green", "3", "cosmet", "safe", "use"], ["fd", "c", "green", "3"], "Subpart C - Cosmetics"),
            r("74.1203", "74", 0, "FD&C Green No. 3.", ["green", "3", "drug", "safe", "use"], ["fd", "c", "green", "3"], "Subpart B - Drugs"),
            r("101.1", "101", 0, "Principal display panel.", ["label", "panel", "food"], ["princip", "display", "panel"])]


def test_section_index_prefers_the_cosmetics_listing_for_a_cosmetics_question():
    ix = SectionIndex.from_rows(_rows())
    ranked = [r.section for r in ix.rank(["lipstick", "green", "3", "fd", "c"])]
    assert ranked[0] == "74.2203" and ranked.index("74.2203") < ranked.index("74.1203")
    assert cosmetic_question(["lipstick"]) and not cosmetic_question(["tablet"])


def test_section_index_ranks_the_specific_heading_first():
    ix = SectionIndex.from_rows(_rows())
    assert ix.rank(["notic", "file", "petit", "public"])[0].section == "71.2"


# ------------------------------------------------------------ answer-first format and validation


def test_apply_style_replaces_the_answer_format_line():
    pkg = package_for(TWO)
    assert APPLY_FORMAT not in render_package(pkg)
    assert render_package(replace(pkg, answer_style="apply")).rstrip().endswith(APPLY_FORMAT)


def test_apply_style_accepts_figures_the_question_states():
    pkg = replace(package_for("For our 12 fl oz bottle, what do 820.10(b)(2) and 820.10(b)(4) require?"), answer_style="apply")
    text = ("For the 12 fl oz bottle, 21 CFR 820.10(b)(2) requires traceability procedures [1]. "
            "21 CFR 820.10(b)(4) requires advisory notices [2].")
    assert UNSUPPORTED_FIGURE not in validate_answer(text, pkg, gen("x")).codes()
    assert UNSUPPORTED_FIGURE in validate_answer(text, replace(pkg, answer_style="restate"), gen("x")).codes()


def test_statute_resolution_is_rendered_as_the_requested_provision():
    pkg = replace(package_for("What does the law say?"), resolved=("FD&C Act § 605 [21 U.S.C. 364a] Adverse events",))
    assert "FD&C Act § 605 [21 U.S.C. 364a] Adverse events (the statute section this question is about" in render_package(pkg)


# ------------------------------------------------------------ refusal retry


class TwoAnswers(ScriptedGenerator):
    def __init__(self, first, second):
        super().__init__(first); self.answers = [first, second]

    def generate(self, messages, max_new_tokens=None, on_text=None):
        self.response = self.answers[min(len(self.calls), 1)]
        return super().generate(messages, max_new_tokens)


def test_a_bare_decline_with_the_requested_provision_supplied_is_asked_once_more():
    g = TwoAnswers("The available regulatory sources do not provide enough information to answer this.",
                   "21 CFR 820.10(b)(2) requires traceability procedures [1]. 21 CFR 820.10(b)(4) requires advisory notices [2].")
    out = RagPipeline(LexicalRetriever(TableDatabase()), g, RagConfig(answer_style="apply")).answer(TWO)
    assert len(g.calls) == 2 and g.calls[1]["messages"][-1]["content"].endswith(REFUSAL_RETRY_NOTE)
    assert out.metadata["generation"]["refusal_retry"] is True and "traceability" in out.answer


def test_no_retry_in_restate_style():
    g = TwoAnswers("The available regulatory sources do not provide enough information to answer this.", "unused")
    RagPipeline(LexicalRetriever(TableDatabase()), g, RagConfig()).answer(TWO)
    assert len(g.calls) == 1


# ------------------------------------------------------------ correction pass


def test_a_rejected_draft_is_corrected_once_and_used_only_if_it_validates():
    g = TwoAnswers("21 CFR 820.10(b)(2) requires traceability procedures [9]. 21 CFR 820.10(b)(4) requires advisory notices [2].",
                   "21 CFR 820.10(b)(2) requires traceability procedures [1]. 21 CFR 820.10(b)(4) requires advisory notices [2].")
    out = RagPipeline(LexicalRetriever(TableDatabase()), g, RagConfig(answer_style="apply")).answer(TWO)
    second = g.calls[1]["messages"]
    assert second[0]["content"].startswith("You write answers") and "9. The EVIDENCE contains the provision" in second[0]["content"]
    assert "PROBLEMS FOUND IN THE DRAFT:" in second[-1]["content"] and "[9]" in second[-1]["content"]
    info = out.metadata["generation"]["correction_pass"]
    assert info["accepted"] is True and out.answer.startswith("21 CFR 820.10(b)(2) requires traceability procedures [1]")


def test_rx1_the_correction_keeps_the_drafts_prompt_as_its_prefix(monkeypatch):
    from regulus.rag import validator_switches as VS
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"RX1"})
    g = TwoAnswers("21 CFR 820.10(b)(2) requires traceability procedures [9]. 21 CFR 820.10(b)(4) requires advisory notices [2].",
                   "21 CFR 820.10(b)(2) requires traceability procedures [1]. 21 CFR 820.10(b)(4) requires advisory notices [2].")
    out = RagPipeline(LexicalRetriever(TableDatabase()), g, RagConfig(answer_style="apply")).answer(TWO)
    first, second = g.calls[0]["messages"], g.calls[1]["messages"]
    assert second[0] == first[0] and second[1:-1] == first[1:-1]               # the same system prompt and history
    assert second[-1]["content"].startswith(first[-1]["content"] + "\n\nYOUR DRAFT ANSWER:")   # the prompt is the prefix
    assert "Rule 9 does not apply to this correction: The EVIDENCE contains the provision" in second[-1]["content"]
    assert out.metadata["generation"]["correction_pass"]["accepted"] is True


def test_a_correction_that_still_fails_leaves_the_normal_repair_path():
    bad = "21 CFR 820.10(b)(2) requires traceability procedures [9]."
    g = TwoAnswers(bad, bad)
    out = RagPipeline(LexicalRetriever(TableDatabase()), g, RagConfig(answer_style="apply")).answer(TWO)
    assert out.metadata["generation"]["correction_pass"]["accepted"] is False and "[9]" not in out.answer


def test_cx1_a_draft_with_many_different_errors_is_not_handed_back(monkeypatch):
    import regulus.rag.pipeline as PL
    from regulus.rag import validator_switches as VS
    bad = "21 CFR 820.10(b)(2) requires traceability procedures [9]."
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"CX1"})
    monkeypatch.setattr(PL, "CX1_MAX_CODES", 1)   # this fixture's draft has one error code
    g = TwoAnswers(bad, "unused")
    out = RagPipeline(LexicalRetriever(TableDatabase()), g, RagConfig(answer_style="apply")).answer(TWO)
    info = out.metadata["generation"]["correction_pass"]
    assert len(g.calls) == 1 and info["skipped"] == "CX1" and info["attempted"] is False and "[9]" not in out.answer
    monkeypatch.setattr(PL, "CX1_MAX_CODES", 9)   # fewer codes than the floor: handed back as before
    g = TwoAnswers(bad, bad)
    RagPipeline(LexicalRetriever(TableDatabase()), g, RagConfig(answer_style="apply")).answer(TWO)
    assert len(g.calls) == 2


# ------------------------------------------------------------ code review 22 Oct: apply-style validation stays strict


def test_a_false_number_taken_from_the_question_is_still_unsupported():
    from regulus.rag.validation import _question_figures
    assert _question_figures("may it contain lead up to 10 percent?", ["not more than 0.6 percent"]) == set()
    assert "12" in _question_figures("our 12 fl oz bottle", ["the declaration of net quantity"])


def test_question_words_do_not_count_as_support_for_what_a_sentence_adds():
    from regulus.rag.claim_validation import check_claims
    pkg = replace(package_for(TWO), answer_style="apply")
    q = "Our company received a consumer complaint of a mild rash from our shampoo; what must we do?"
    restates = "Your company received a consumer complaint of a mild rash from your shampoo [1]."
    invents = "Your company must notify every distributor and recall the shampoo batch [1]."
    codes = lambda s: {f.code for f in check_claims([s], pkg, question=q)}
    assert "unsupported_claim" not in codes(restates)
    assert "unsupported_claim" in codes(invents)


def test_promotion_never_runs_where_a_refusal_is_decided_from_the_question(monkeypatch):
    # without PO2: with it, a part outside the cosmetics programme may be promoted into, and "What does 21 CFR Part 11
    # require?" is refused as a whole-part question by SectionPromoter.prepare instead
    from regulus.rag import validator_switches
    from regulus.rag.pipeline import may_promote
    monkeypatch.setattr(validator_switches, "VALIDATOR_PRECISION", validator_switches.VALIDATOR_PRECISION - {"PO2"})
    for q in ("Under the EU Cosmetics Regulation, what must a lipstick label list?",
              "What does paragraph (b) require for lipstick colors?",
              "Under 16 CFR Part 500, what must a label state?",
              "What does 21 CFR Part 11 require?",
              "Which FDA guidance covers cosmetic labeling?"):
        assert not may_promote(q, plan_query(q)), q
    for q in ("Is FD&C Green No. 3 permitted in eye shadow?", "Under Part 82, what specifications must a batch meet?",
              "What does FD&C Act § 607 require of a cosmetics facility?"):
        assert may_promote(q, plan_query(q)), q


# ------------------------------------------------------------ code review round 2


def test_reasoning_block_check_accepts_the_1_7b_empty_think_and_refuses_an_open_one():
    from regulus.rag.mlx_models import opens_reasoning
    assert not opens_reasoning("")                              # Qwen3-4B-Instruct-2507's own template
    assert not opens_reasoning("<think>\n\n</think>\n\n")       # Qwen3-1.7B, thinking off
    assert opens_reasoning("<think>\n")                         # a thinking model


def test_ambiguous_questions_may_resolve_only_a_named_statute_section():
    from regulus.rag.pipeline import may_promote
    assert may_promote("What does paragraph (b) of the FD&C Act require for lipstick colors?",
                       plan_query("What does paragraph (b) of the FD&C Act require for lipstick colors?")) == "named_statute"
    assert may_promote("What does 40 CFR require for pesticide residues on cosmetics?",
                       plan_query("What does 40 CFR require for pesticide residues on cosmetics?")) is None
    q = "Which section of 21 CFR lists iron oxides for use in cosmetics?"
    assert may_promote(q, plan_query(q)) == "any"


def test_a_question_without_regulatory_vocabulary_is_never_promoted():
    class Vocab:
        def domain_terms(self, q):
            return ()

    class Policy:
        vocabulary = Vocab()

        def decide(self, *a, **k):
            from regulus.retrieval.domain import GENERAL_KNOWLEDGE, ScopeDecision
            return ScopeDecision(GENERAL_KNOWLEDGE, "uses no word of the regulatory vocabulary", ())

    calls = []

    class Promoter:
        mode = "picker"
        generator = None

        def resolve(self, *a, **k):
            calls.append(a)

    pipe = RagPipeline(LexicalRetriever(TableDatabase()), ScriptedGenerator("x"), RagConfig(section_promotion="picker"),
                       scope_policy=Policy())
    pipe._promoter = Promoter()
    pipe.section_promoter = lambda: pipe._promoter
    pipe.answer("What is the capital of France?")
    assert calls == []


def test_mlx_request_carries_the_repetition_penalty_only_when_set():
    import types
    from regulus.rag.mlx_generator import MlxGenerator
    g = object.__new__(MlxGenerator)
    g.eos_ids = [1]
    g.gen = types.SimpleNamespace(repetition_penalty=1.1)
    assert g.generate_request([5, 6], 8)["repetition_penalty"] == 1.1
    g.gen = types.SimpleNamespace(repetition_penalty=1.0)
    assert "repetition_penalty" not in g.generate_request([5, 6], 8)


class _PickRecorder:
    def __init__(self, thinking):
        self.thinking, self.calls = thinking, []

    def generate(self, messages, max_new_tokens=None, on_text=None, **kw):
        self.calls.append({"max_new_tokens": max_new_tokens, **kw})
        return GenerationResult(prompt="", response="", prompt_tokens=0, generated_tokens=0, seconds=0.0, finished=True)


@pytest.mark.parametrize("thinking,expected", [(True, {"max_new_tokens": SP.PICKER_MAX_TOKENS, "think_budget": 0}),
                                               (False, {"max_new_tokens": SP.PICKER_MAX_TOKENS})])
def test_the_picker_closes_a_thinking_models_reasoning_block(monkeypatch, thinking, expected):
    gen = _PickRecorder(thinking)
    pr = SectionPromoter(db=None, generator=gen, mode="picker", index=object())
    monkeypatch.setattr(SectionPromoter, "statute", property(lambda self: None))
    monkeypatch.setattr(SP.SectionIndex, "question_lexemes", staticmethod(lambda db, q: {"lash", "adhes"}))
    monkeypatch.setattr(pr, "named_statute", lambda q, lex: [])
    monkeypatch.setattr(pr, "candidates", lambda q, parts=None: ({"lash", "adhes"}, [
        {"section": "700.3", "part": "700", "heading": "Definitions.", "subpart": "", "snippet": "x"}]))
    assert pr.resolve("Which rules apply to our lash adhesive?") is None   # an empty reply picks nothing
    assert gen.calls == [expected]


# per-section relevance order (22 Oct 2026): a promoted section with a companion lost 701.3(m) and (r) to document order
def _two_section_required(question):
    from regulus.rag.selection import is_required
    from tests.test_retrieval_plan import row
    table = [row("701.3", "(a)", 0, text="(a) The label shall bear a declaration of the name of each ingredient in descending order."),
             row("701.3", "(b)", 1, text="(b) The declaration shall appear on an information panel in letters not less than 1/16 inch."),
             row("701.3", "(r)", 2, text="(r) A cosmetic sold solely by direct mail order may bear the declaration in a catalog or brochure."),
             row("701.30", "", 3, text="The Commissioner establishes the following names for cosmetic ingredient labeling.")]
    reg = LexicalRetriever(TableDatabase(table)).run_plan(plan_query(question)).registry
    return [e for e in reg.evidence if is_required(e)]


def test_each_section_is_ordered_by_relevance_and_sections_keep_their_order(monkeypatch):
    from regulus.rag import validator_switches as VS
    from regulus.rag.selection import _required_order
    required = _two_section_required("We sell by direct mail order. Where does the declaration go under 21 CFR 701.3 and 701.30?")
    assert [e.citation for e in required][:3] == ["21 CFR 701.3(a)", "21 CFR 701.3(b)", "21 CFR 701.3(r)"]
    question = "We sell by direct mail order. Where does the declaration go?"
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"CF1"})
    order = [e.citation for e in _required_order(required, question)]
    assert order[0] == "21 CFR 701.3(r)"                       # the paragraph the question is about, first
    assert order[-1] == "21 CFR 701.30"                        # the companion section still follows the promoted one
    assert sorted(order) == sorted(e.citation for e in required)
    # CF1 (default since 2 Nov): the companion's best paragraph competes right after the promoted section's best
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"CF1"})
    order = [e.citation for e in _required_order(required, question)]
    assert order[:2] == ["21 CFR 701.3(r)", "21 CFR 701.30"]
    assert sorted(order) == sorted(e.citation for e in required)


def _statute_row(sec, path, text, heading="", top_caption=""):
    return {"fdca_section": sec, "usc_section": {"604": "364", "605": "364a"}[sec], "paragraph_path": path, "ordinal": 0,
            "heading": "Definitions" if sec == "604" else "Adverse events", "paragraph_heading": heading,
            "body_text": text, "top_caption": top_caption, "source_url": "https://uscode.house.gov"}


def test_a_promoted_subchapter_section_brings_the_604_definitions_it_uses(monkeypatch):
    from regulus.retrieval.statute import StatuteIndex
    rows = [_statute_row("604", "", "In this subchapter:"),
            _statute_row("604", "(1)", 'The term "adverse event" means any health-related event that is adverse.', "Adverse event"),
            _statute_row("604", "(3)(A)", '(A) In general.—The term "facility" includes any establishment.', top_caption="Facility"),
            _statute_row("604", "(5)", 'The term "serious adverse event" means an adverse event that—', "Serious adverse event"),
            _statute_row("604", "(5)(A)", "(A) results in death;"),
            _statute_row("605", "(a)", "(a) The responsible person shall submit each serious adverse event report.")]
    st = StatuteIndex(rows, [set() for _ in rows])
    monkeypatch.setattr(SectionPromoter, "statute", property(lambda self: st))
    pr = SectionPromoter(db=None, mode="picker", index=object())
    pro = pr._statute_promotion([("605", "")], set(), "picker", 0.0)
    blocks = [[rows[i]["paragraph_path"] for i in g] for g in pro.statute_blocks]
    assert blocks[0] == ["(a)"]                                    # the section's own text first
    assert ["(5)", "(5)(A)"] in blocks and ["(1)"] in blocks     # the terms 605(a) uses, a definition kept whole
    assert ["(3)(A)"] not in blocks                               # "facility" is not used by the kept text
    assert pro.statute == ("605",)                                # 604 is supporting text, not the resolved section
    # only the sections 604 governs get its definitions (601 adulteration, 201 definitions and 604 itself do not)
    assert SP.SUBCHAPTER_DEFINED == frozenset(str(n) for n in range(605, 615))



# ------------------------------------------------------------ DV1 paragraph-level retrieval (27 Oct)


class _PartRows:
    def __init__(self, rows):
        self.rows = rows

    def query(self, sql, params):
        return [dict(s=s, p=p, h=h, t=t) for s, p, h, t in self.rows]


class _FlatReranker:
    def __init__(self, scores):
        self.scores = scores

    def score(self, question, texts):
        return [max([v for k, v in self.scores.items() if k in t] or [0.0]) for t in texts]


def _part_promoter(monkeypatch, rows, scores):
    import regulus.retrieval.rerank as RR
    monkeypatch.setattr(SP, "RERANKER", "fake")
    monkeypatch.setattr(RR, "reranker", lambda name: _FlatReranker(scores))
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db = _PartRows(rows)
    return pr


def test_every_row_of_a_definitions_paragraph_is_scored(monkeypatch):
    rows = [("820.3", "(b)", "Definitions.", "All definitions in section 201 of the act shall apply."),
            ("820.3", "(b)", "Definitions.", "Safety and Performance shall have the meaning of safety and effectiveness."),
            ("820.1", "(a)", "Scope.", "Finished devices will be safe and effective.")]
    pr = _part_promoter(monkeypatch, rows, {"Safety and P": 0.9, "Finished dev": 0.1})
    assert pr.part_paragraphs("What meaning does Part 820 give to 'safety and performance'?", "820")[0] == "820.3"   # short: whole


def test_the_paragraph_sharing_the_most_phrases_is_kept_under_the_score_floor(monkeypatch):
    rows = [("820.7", "(b)", "Incorporation by reference.", "ISO 13485 is incorporated by reference."),
            ("820.10", "(c)", "Requirements.", "Design and development. Manufacturers of class II devices must comply."),
            ("820.45", "(a)", "Labeling.", "Labeling must be examined.")]
    pr = _part_promoter(monkeypatch, rows, {"ISO 13485 is": 0.6, "Design and d": 0.05})
    got = pr.part_paragraphs("Which devices must apply the design and development requirements?", "820")
    assert got == ["820.10", "820.7"]   # phrase rank 1 and cross-encoder rank 2 fuse ahead; short sections whole


def test_a_paragraph_of_a_long_section_brings_its_top_level_family():
    items = [("820.1(a)", "x" * 1900), ("820.1(a)(1)", "x" * 300), ("820.1(a)(4)", "x" * 300), ("820.1(b)", "x" * 400),
             ("820.7", "y" * 200), ("820.7(a)", "y" * 300), ("820.7(b)", "y" * 300)]
    assert SectionPromoter.paragraph_families(["820.1(a)(4)", "820.7(b)"], items) == \
        ["820.1(a)", "820.1(a)(1)", "820.1(a)(4)", "820.7"]


# ------------------------------------------------------------ DV2: DV1 narrowed (29 Oct)

_SCOPE_ROWS = [("820.1", "(a)", "Scope.", "Applicability. The requirements in this part govern finished devices. " + "x" * 2000),
               ("820.1", "(b)", "Scope.", "The quality management system regulation in this part supplements regulations."),
               ("820.1", "(c)", "Scope.", "Foreign manufacturers that refuse to permit inspection. " + "y" * 600),
               ("820.1", "(d)", "Scope.", "Exemptions or variances may be requested by petition."),
               ("820.45", "(a)", "Labeling.", "Labeling must be examined.")]


def test_dv2_a_scope_question_gets_the_whole_scope_section(monkeypatch):
    from regulus.rag import validator_switches as VS
    pr = _part_promoter(monkeypatch, _SCOPE_ROWS, {"Applicability": 0.9, "Foreign": 0.5})
    q = "What is the scope of 21 CFR Part 820 and which manufacturers are subject to it?"
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"DV2"})
    assert "820.1" not in pr.part_paragraphs(q, "820")          # DV1: paragraph families of the long section
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"DV2"})
    assert pr.part_paragraphs(q, "820")[0] == "820.1"
    assert pr.__dict__["last_part_score"] == pytest.approx(0.9)


def test_dv2_a_point_question_keeps_its_paragraph_family(monkeypatch):
    from regulus.rag import validator_switches as VS
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"DV2"})
    pr = _part_promoter(monkeypatch, _SCOPE_ROWS, {"Foreign": 0.9})
    got = pr.part_paragraphs("What does Part 820 provide for foreign manufacturers that refuse to permit inspection?", "820")
    assert got[0] == "820.1(c)" and "820.1" not in got


def test_dv2_a_guidance_document_question_is_not_answered_from_part_paragraphs():
    from regulus.rag.pipeline import _GUIDANCE_DOCUMENT
    assert _GUIDANCE_DOCUMENT.search("Summarize FDA's final guidance 'QMSR Certification Body Accreditation Program,' issued March 2026.")
    assert not _GUIDANCE_DOCUMENT.search("A firm uses a cloud-hosted LIMS. Which Part 11 provisions apply to the vendor's "
                                         "controls, and how does the FDA guidance on scope and application affect them?")
    assert not _GUIDANCE_DOCUMENT.search("What is the scope of 21 CFR Part 820?")


def test_hc1_coordinated_topics_split_across_sibling_headings():
    from regulus.rag.section_promotion import _coordinated, _topic_stems
    pairs = _coordinated("What are the drug product inspection and expiration dating requirements?")
    assert any("inspec" in left and {"expira", "dating"} <= right for left, right in pairs)
    assert _coordinated("What are the general laboratory control requirements under Part 211?") == []
    assert _topic_stems("Drug product salvaging.") & _topic_stems("returned and salvaged drug products")


def test_ht1_generic_headings_are_never_twins():
    from regulus.rag.section_promotion import _GENERIC_HEADING
    assert _GENERIC_HEADING.match("Scope.") and _GENERIC_HEADING.match("Definitions.")
    assert not _GENERIC_HEADING.match("Production record review.")


class _NoDB:
    def query(self, *a, **k):
        raise AssertionError("a question naming the section's own part needs no twin lookup")


def test_ht1_keeps_a_section_of_the_part_the_question_names():
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db = _NoDB()
    q = "Where are the definitions that apply to 21 CFR Part 111 located?"   # S-002: was moved to 112.3
    assert pr.general_twin("111.3", q) == "111.3"


class _TwinRows:
    def __init__(self, heading):
        self.heading = heading

    def query(self, *a, **k):
        return [{"s": "111.3", "p": "111", "title": "Part 111 - Dietary Supplements", "h": self.heading, "n": 60},
                {"s": "112.3", "p": "112", "title": "Part 112 - Produce", "h": self.heading, "n": 100}]


def test_ht1_a_heading_about_its_own_part_is_never_a_twin():
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db = _TwinRows("What definitions apply to this part?")
    assert pr.general_twin("111.3", "Where are the dietary supplement definitions?") == "111.3"
    pr.db = _TwinRows("Production record review.")
    assert pr.general_twin("111.3", "What must production record review include?") == "112.3"   # a real twin still moves


def test_hm1_a_heading_phrase_must_occur_word_for_word():
    from regulus.rag.section_promotion import heading_phrase_in
    q = "What are the requirements for use of approved components and for rejected components?"
    assert heading_phrase_in("Use of approved components, drug product containers, and closures.", q)
    assert heading_phrase_in("Rejected components, drug product containers, and closures.", q)
    assert not heading_phrase_in("Testing and approval or rejection of components, drug product containers, and closures.", q)
    assert not heading_phrase_in("Control of records.", "Which record control section of Part 820 states that the UDI must be recorded?")


def _ms1_promoter(texts, scores):
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.__dict__["_text_cache"] = dict(texts)
    pr.__dict__["_order_scores"] = ("q", tuple(texts), {s: (v, "") for s, v in scores.items()})
    return pr


def _cands(*secs):
    return [{"section": s, "part": s.split(".")[0]} for s in secs]


def test_ms1_adds_the_best_scored_sections_of_the_part_above_the_floor(monkeypatch):
    from regulus.rag import validator_switches as VS
    # MS1 on its own: XC1 (default since 2 Nov) adds the best other-part candidate on top - its own tests cover that
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"XC1"})
    monkeypatch.setattr(SP, "MS1_MIN_SCORE", 0.5)
    monkeypatch.setattr(SP, "MS1_MAX", 2)
    monkeypatch.setattr(SP, "MS1_MAX_CHARS", 6000)
    texts = {"111.80": "a", "111.83": "b" * 900, "111.465": "c" * 900, "111.10": "d", "211.84": "e"}
    pr = _ms1_promoter(texts, {"111.80": 0.9, "111.83": 0.8, "111.465": 0.6, "111.10": 0.2, "211.84": 0.95})
    got = pr.reranked_companions("q", _cands(*texts), Promotion(sections=("111.80",), companions=(), method="picker"))
    assert got == ["111.83", "111.465"]   # 211.84 is another part, 111.10 under the floor


def test_ms1_keeps_the_character_cap_and_skips_what_is_already_promoted(monkeypatch):
    monkeypatch.setattr(SP, "MS1_MIN_SCORE", 0.5)
    monkeypatch.setattr(SP, "MS1_MAX", 2)
    monkeypatch.setattr(SP, "MS1_MAX_CHARS", 1000)
    texts = {"117.3": "a", "117.5": "b" * 2000, "117.201": "c" * 500, "117.10": "d" * 400}
    pr = _ms1_promoter(texts, {"117.5": 0.9, "117.201": 0.8, "117.10": 0.7})
    got = pr.reranked_companions("q", _cands(*texts), Promotion(sections=("117.3",), companions=("117.10",), method="picker"))
    assert got == ["117.201"]   # 117.5 is over the cap, 117.10 already a companion


def test_ms1_never_touches_the_cosmetics_programme():
    pr = _ms1_promoter({"70.3": "a", "70.5": "b"}, {"70.5": 0.99})
    assert pr.reranked_companions("q", _cands("70.3", "70.5"), Promotion(sections=("70.3",), companions=(), method="picker")) == []


class _SubpartRows:
    def query(self, sql, params=None):
        assert params == {"p": "117", "h": "Subpart B %"}
        return [{"s": "117.10"}, {"s": "117.20"}]


def test_sn1_a_named_subpart_keeps_its_own_candidates():
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db = _SubpartRows()
    cands = _cands("117.4", "117.10", "117.135", "117.20")
    got = pr.named_subpart("What CGMP requirements apply to personnel under Part 117 Subpart B?", cands)
    assert [c["section"] for c in got] == ["117.10", "117.20"]   # F-003: 117.4 (Subpart A) was promoted


def test_sn1_leaves_questions_without_a_named_subpart_alone():
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db = None   # never queried
    cands = _cands("117.4", "117.10")
    assert pr.named_subpart("What does the subpart of Part 117 on personnel require?", cands) == cands


class _HeadingSubpartRows:
    def query(self, sql, params=None):
        assert "~*" in sql and params["p"] == "111" and params["h"].endswith("- personnel\\M")
        return [{"s": "111.12"}, {"s": "111.14"}]


def test_sn1_a_subpart_named_by_its_heading():
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db = _HeadingSubpartRows()
    cands = _cands("111.605", "111.140", "111.14", "111.12")
    q = "What records must be made and kept under the Part 111 personnel subpart?"   # S-005: SP2 added 111.140
    assert [c["section"] for c in pr.named_subpart(q, cands)] == ["111.14", "111.12", "111.605", "111.140"]   # boost


class _DefinitionRows:
    def __init__(self):
        self.calls = 0

    def query(self, sql, params=None):
        self.calls += 1
        if "subpart_id" in sql:
            return [{"t": "importer", "s": "1.500", "p": "(2)"}, {"t": "food", "s": "1.500", "p": "(5)"},
                    {"t": "importer", "s": "1.502", "p": ""}]
        return []


def test_dq1_supplies_the_asked_terms_definition_paragraph_from_the_subpart():
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db = _DefinitionRows()
    q = "Who must comply with the Foreign Supplier Verification Programs regulation and who is the 'importer'?"
    assert pr.asked_definitions("1.502", q) == ["1.500(2)"]   # V-001; a pathless (whole-section) row is never added


def test_dq1_ignores_questions_that_ask_no_meaning():
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db = _DefinitionRows()
    assert pr.asked_definitions("1.502", "What records must an importer keep?") == [] and pr.db.calls == 0


class _StructureRows:
    def query(self, sql, params=None):
        if "subpart_id" in sql:
            return [{"s": "111.70", "o": 1}, {"s": "111.73", "o": 2}, {"s": "111.75", "o": 3}, {"s": "111.95", "o": 9}]
        return [{"target_citation": "21 CFR 111.123(b)"}, {"target_citation": "21 CFR 117.3"}]   # cited sections


def test_ms2_structural_neighbours_are_cited_sections_of_the_part_then_nearest_siblings():
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db = _StructureRows()
    assert pr.structural_neighbours("111.73", limit=3) == ["111.123", "111.75", "111.70"]   # nearest first; 117.3 is another part


def test_ms2_scores_the_structure_with_the_candidates(monkeypatch):
    monkeypatch.setattr(SP, "MS1_MIN_SCORE", 0.5)
    monkeypatch.setattr(SP, "MS1_MAX", 2)
    monkeypatch.setattr(SP, "MS1_MAX_CHARS", 6000)
    from regulus.rag import validator_switches as VS
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"MS2"})
    pr = _ms1_promoter({"111.87": "a", "111.10": "b"}, {"111.10": 0.1})
    pr.db = _StructureRows()
    monkeypatch.setattr(pr, "structural_neighbours", lambda s: ["111.75", "111.123"])
    monkeypatch.setattr(pr, "rerank_scores", lambda q, cands: {"111.75": (0.9, ""), "111.123": (0.3, "")})
    pr.__dict__["_text_cache"].update({"111.75": "x" * 500, "111.123": "y"})
    got = pr.reranked_companions("q", _cands("111.87", "111.10"), Promotion(sections=("111.87",), companions=(), method="picker"))
    assert got == ["111.75"]   # S-012: the structural neighbour above the floor; 111.123 and the candidate 111.10 below it


def test_pc1_the_cascade_picks_only_when_the_reranker_is_confident(monkeypatch):
    monkeypatch.setattr(SP, "PC1_MIN_SCORE", 0.95)
    monkeypatch.setattr(SP, "PC1_MARGIN", 0.10)
    cands = _cands("111.80", "111.83", "111.465")
    pr = _ms1_promoter({"111.80": "", "111.83": "", "111.465": ""}, {"111.80": 0.98, "111.83": 0.70, "111.465": 0.5})
    pr.__dict__["_order_scores"] = ("q", ("111.80", "111.83", "111.465"), {"111.80": (0.98, ""), "111.83": (0.70, ""), "111.465": (0.5, "")})
    raw = pr.cascade_pick("q", cands)
    assert raw == "cascade 98-70: 21 CFR 111.80, 21 CFR 111.83"
    assert parse_picks(raw, 3, ["111.80", "111.83", "111.465"]) == [1, 2]   # the marker never becomes a pick
    pr.__dict__["_order_scores"] = ("q", ("111.80", "111.83", "111.465"), {"111.80": (0.98, ""), "111.83": (0.93, ""), "111.465": (0.5, "")})
    assert pr.cascade_pick("q", cands) is None   # too close: the language-model picker decides


class _BigSectionRows:
    def query(self, sql, params=None):
        return [{"paragraph_path": "(a)", "body_text": "a" * 5000}, {"paragraph_path": "(b)", "body_text": "b" * 700},
                {"paragraph_path": "(b)(1)", "body_text": "c" * 300}, {"paragraph_path": "(c)", "body_text": "d" * 5000}]


def test_ms3_a_section_too_large_to_add_whole_brings_its_best_paragraph_family(monkeypatch):
    from regulus.rag import validator_switches as VS
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"MS3"})
    monkeypatch.setattr(SP, "MS1_MIN_SCORE", 0.7)
    monkeypatch.setattr(SP, "MS1_MAX", 2)
    monkeypatch.setattr(SP, "MS1_MAX_CHARS", 6000)
    pr = _ms1_promoter({"101.9": "x", "101.12": "y" * 11000}, {})
    pr.__dict__["_order_scores"] = ("q", ("101.9", "101.12"), {"101.12": (0.9, "(b)(1)")})
    pr.db = _BigSectionRows()
    got = pr.reranked_companions("q", _cands("101.9", "101.12"), Promotion(sections=("101.9",), companions=(), method="picker"))
    assert got == ["101.12(b)"]   # L-008: 101.12 is 11,000 characters; its (b) family is 1,000


class _SubpartToc:
    def query(self, sql, params=None):
        return [{"s": "111.70", "h": "What specifications must you establish?", "o": 1, "c": 3000},
                {"s": "111.73", "h": "What is your responsibility to determine whether established specifications are met?", "o": 2, "c": 400},
                {"s": "111.87", "h": "Who conducts a material review and makes a disposition decision?", "o": 5, "c": 300},
                {"s": "111.90", "h": "What requirements apply to treatments, in-process adjustments, and reprocessing?", "o": 6, "c": 900}]


class _Nav:
    def __init__(self, reply):
        self.reply, self.seen = reply, None

    def generate(self, messages, max_new_tokens=None, **kw):
        self.seen = messages
        return GenerationResult(prompt="", response=self.reply, prompt_tokens=0, generated_tokens=3, seconds=0.1, finished=True)


def test_nav1_adds_the_subpart_sections_the_model_names_from_the_table_of_contents():
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db, pr.generator = _SubpartToc(), _Nav("21 CFR 111.73, 21 CFR 111.90, 21 CFR 112.3")
    got = pr.navigate_subpart("A finished batch fails an identity specification. What does Part 111 require?",
                              Promotion(sections=("111.87",), companions=(), method="picker"))
    assert got == ["111.73", "111.90"]   # S-012; 112.3 is not in the table of contents
    toc = pr.generator.seen[-1]["content"]
    assert "ALREADY CHOSEN: 21 CFR 111.87" in toc and "21 CFR 111.87 Who" not in toc.split("OTHER SECTIONS:")[1]


def test_nav1_none_adds_nothing():
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db, pr.generator = _SubpartToc(), _Nav("none")
    assert pr.navigate_subpart("q", Promotion(sections=("111.87",), companions=(), method="picker")) == []


class _XrefRows:
    def query(self, sql, params=None):
        if "cfr_cross_references" in sql:
            assert params == {"s": "820.1", "p": "(d)", "pp": "(d)%"}
            return [{"target_citation": "21 CFR 820.3"}, {"target_citation": "21 CFR 10.30"}]
        return [{"paragraph_path": "", "body_text": "x" * 3000}]


def test_xp1_adds_the_other_part_provision_the_promoted_paragraph_cites():
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db = _XrefRows()
    got = pr.cross_part_citations(Promotion(sections=("820.1(d)",), companions=(), method="picker"))
    assert got == ["10.30"]   # M-007: 820.3 is the same part; 10.30 (citizen petitions) is cited by 820.1(d)(1)


def test_nav2_lists_closest_sections_of_the_part_outside_the_subpart(monkeypatch):
    from regulus.rag import validator_switches as VS
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"NAV2"})
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db, pr.generator = _SubpartToc(), _Nav("21 CFR 111.610")
    monkeypatch.setattr(pr, "part_neighbours", lambda q, s, ex: [{"s": "111.610", "h": "What records must be kept?", "c": 800}])
    got = pr.navigate_subpart("What records must be kept under the Part 111 personnel subpart?",
                              Promotion(sections=("111.87",), companions=(), method="picker"))
    assert got == ["111.610"] and "21 CFR 111.610 What records must be kept?" in pr.generator.seen[-1]["content"]   # S-005


def test_nav1_v2_reads_list_numbers_and_rejects_invented_sections():
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db, pr.generator = _SubpartToc(), _Nav("2, 4")   # [1] 111.70 [2] 111.73 [3] 111.90 ... after 111.87 is chosen
    got = pr.navigate_subpart("q", Promotion(sections=("111.87",), companions=(), method="picker"))
    assert got == ["111.73"] or got == ["111.73", "111.90"] or got[0] == "111.73"
    pr.generator = _Nav("211.150, 211.160")   # D-018: numbers outside the subpart are never taken
    assert pr.navigate_subpart("q", Promotion(sections=("111.87",), companions=(), method="picker")) == []


def test_nav1_v2_shows_each_sections_opening_words():
    pr = SectionPromoter.__new__(SectionPromoter)

    class Rows(_SubpartToc):
        def query(self, sql, params=None):
            return [dict(r, t="You must examine labeling materials for identity and conformity.") for r in super().query(sql, params)]

    pr.db, pr.generator = Rows(), _Nav("none")
    pr.navigate_subpart("q", Promotion(sections=("111.87",), companions=(), method="picker"))
    assert "[1] 21 CFR 111.70 What specifications must you establish? - You must examine labeling" in pr.generator.seen[-1]["content"]


# ------------------------------------------------------------ OC1 added sections as context (29 Oct)


def test_oc1_an_added_section_is_context_not_a_requested_provision():
    from regulus.rag.pipeline import companions_as_context
    from regulus.retrieval.evidence import TIER_SUPPORTING
    pkg = package_for(TWO)
    plan = plan_query(TWO)
    companion = "21 CFR 820.10(b)(4)"
    rid = next(r for r, label in pkg.requirements if label == companion)
    out = companions_as_context(pkg, plan, frozenset({companion}))
    assert [label for _, label in out.requirements] == ["21 CFR 820.10(b)(2)"] and rid not in out.satisfied_ids
    moved = [s for s in out.sources if s.citation == companion]
    assert moved and all(s.tier == TIER_SUPPORTING and not s.requirement_ids for s in moved)   # still in the evidence
    prompt = render_package(out)
    assert "REQUESTED PROVISIONS:" in prompt and f"{rid}: {companion}" not in prompt and companion in prompt
    text = "21 CFR 820.10(b)(2) requires traceability procedures [1]."
    report = validate_answer(text, out, gen(text))
    assert "requirement_not_addressed" not in report.codes()   # the context section is not demanded


def test_oc1_the_remaining_provision_is_still_identified():
    # 30 Oct full300d: F-002, P-009, K-003 ... lost the focused quote - the moved companion's id stayed in the plan
    from regulus.rag.fallback import provision_is_identified
    from regulus.rag.pipeline import companions_as_context
    pkg, plan = package_for(TWO), plan_query(TWO)
    assert provision_is_identified(plan, pkg)
    out = companions_as_context(pkg, plan, frozenset({"21 CFR 820.10(b)(4)"}))
    assert out.context_ids and provision_is_identified(plan, out)
    unsatisfied = replace(out, satisfied_ids=())
    assert not provision_is_identified(plan, unsatisfied)   # the requested provision itself must still have its evidence


def test_oc1_changes_nothing_when_every_requested_provision_is_added():
    from regulus.rag.pipeline import companions_as_context
    pkg = package_for(TWO)
    labels = frozenset(label for _, label in pkg.requirements)
    assert companions_as_context(pkg, plan_query(TWO), labels) is pkg
    assert companions_as_context(pkg, plan_query(TWO), frozenset()) is pkg


def test_dv2_a_question_about_other_parts_is_not_confined_to_the_named_part():
    from regulus.rag.pipeline import _ASKS_OTHER_PARTS
    assert _ASKS_OTHER_PARTS.search("Which parts of Title 21 govern device establishment registration, listing, and unique "
                                    "device identification, and how do they interact with Part 820?")
    assert not _ASKS_OTHER_PARTS.search("Which parts of our submission become public, and when?")
    assert not _ASKS_OTHER_PARTS.search("How must advisory notices be handled under Part 820?")


# ------------------------------------------------------------ step E: QD2 trigger, PL1 paragraph-level additions (29 Oct)


def test_qd2_comparisons_and_interactions_are_multi_part(monkeypatch):
    from regulus.rag import validator_switches as VS
    q = "How does the Part 117 supply-chain program differ from the FSVP requirements for importers?"
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"QD2"})
    assert not SP.multi_part(q)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"QD2"})
    assert SP.multi_part(q)
    assert SP.multi_part("Which parts govern device listing, and how do they interact with Part 820?")
    assert not SP.multi_part("What records must a qualified facility keep?")


class _Sections:
    """_SECTION_SQL rows: section -> [(paragraph_path, text)]."""

    def __init__(self, sections):
        self.sections = sections

    def query(self, sql, params):
        return [dict(paragraph_path=p, body_text=t) for p, t in self.sections.get(params["s"], [])]


def _companion_promoter(monkeypatch, sections, scores):
    pr = SectionPromoter.__new__(SectionPromoter)
    pr.db = _Sections(sections)
    monkeypatch.setattr(pr, "rerank_scores", lambda q, cands: scores, raising=False)
    monkeypatch.setattr(pr, "_section_text", lambda s: " ".join(t for _, t in sections.get(s, [])), raising=False)
    return pr


def test_pl1_a_long_added_section_comes_as_its_best_paragraph_family(monkeypatch):
    from regulus.rag import validator_switches as VS
    sections = {"117.475": [("(a)", "x" * 2000), ("(a)(1)", "x" * 500), ("(b)", "y" * 3000)],
                "117.405": [("(a)", "z" * 1500)]}
    scores = {"117.475": (0.95, "(a)(1)"), "117.405": (0.90, "(a)")}
    cands = [{"section": s, "part": "117"} for s in sections]
    promotion = Promotion(sections=("117.410",), companions=(), method="picker")
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", (VS.VALIDATOR_PRECISION | {"PL1"}) - {"MS2"})
    pr = _companion_promoter(monkeypatch, sections, scores)
    assert pr.reranked_companions("q", cands, promotion) == ["117.475(a)", "117.405"]   # 2,500 + 1,500 chars: both fit
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"PL1"})
    assert pr.reranked_companions("q", cands, promotion) == ["117.475"]   # before: whole 117.475 crowded out 117.405 (F-020)


def _qd2_promoter(monkeypatch, subs, found):
    """`found(sub, scope) -> (section, score)`; records the scopes searched."""
    seen = []

    class Gen:
        def generate(self, messages, max_new_tokens=None):
            return GenerationResult(prompt="", response="\n".join(subs), prompt_tokens=0, generated_tokens=0, seconds=0.0)

    pr = SectionPromoter.__new__(SectionPromoter)
    pr.generator = Gen()
    monkeypatch.setattr(SP, "RERANKER", "fake")

    def candidates(q, parts=None):
        seen.append(parts)
        sec, _ = found(q, parts)
        return None, [{"section": sec, "part": sec.split(".")[0]}]

    monkeypatch.setattr(pr, "candidates", candidates, raising=False)
    monkeypatch.setattr(pr, "rerank_scores", lambda q, c: {c[0]["section"]: (found(q, seen[-1])[1], "")}, raising=False)
    return pr, seen


def test_qd2_a_sub_question_the_named_part_cannot_answer_searches_every_part(monkeypatch):
    from regulus.rag import validator_switches as VS
    subs = ["What is the Part 117 supply-chain program?", "What must an importer do under FSVP?"]
    found = lambda q, scope: (("117.405", 0.9) if "117" in q else ("117.410", 0.3) if scope == ("117",) else ("1.506", 0.8))
    q = "How does the Part 117 supply-chain program differ from the FSVP requirements for importers?"
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"QD2"})
    pr, seen = _qd2_promoter(monkeypatch, subs, found)
    assert pr.decomposed_sections(q, ("117",)) == ["117.405", "1.506"] and seen == [("117",), ("117",), None]
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"QD2"})
    pr, seen = _qd2_promoter(monkeypatch, subs, found)
    assert pr.decomposed_sections(q, ("117",)) == ["117.405"] and None not in seen


def test_qd2_a_comparison_inside_one_part_stays_in_it(monkeypatch):
    from regulus.rag import validator_switches as VS
    subs = ["What are the responsibilities of quality control personnel?", "What are production personnel responsible for?"]
    found = lambda q, scope: (("111.105", 0.8) if "quality" in q else ("111.110", 0.7)) if scope == ("111",) else ("211.22", 0.95)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"QD2"})
    pr, seen = _qd2_promoter(monkeypatch, subs, found)
    q = "Under Part 111, what is the difference between the responsibilities of quality control personnel and production?"
    assert pr.decomposed_sections(q, ("111",)) == ["111.105", "111.110"] and None not in seen   # never 211.22


def test_qd2_trigger_wording():
    from regulus.rag import validator_switches as VS
    on = VS.VALIDATOR_PRECISION | {"QD2"}
    import regulus.rag.validator_switches as V
    old, V.VALIDATOR_PRECISION = V.VALIDATOR_PRECISION, on
    try:
        for q in ["What are the differences between Part 117 and Part 507?", "How do a 510(k) and a De Novo request differ?",
                  "How is a recall different than a market withdrawal?", "How does Part 211 relate to Part 210?"]:
            assert SP.multi_part(q), q
        for q in ["What must be done when the distributor is different from the manufacturer?",
                  "What requirements relate to supplier verification under Part 117?"]:
            assert not SP.multi_part(q), q
    finally:
        V.VALIDATOR_PRECISION = old


def test_the_key_recall_harness_asks_decomposition_instead_of_taking_it_for_the_answer():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import key_recall_eval as KR
    from regulus.rag.section_promotion import NAV1_SYSTEM, PICKER_SYSTEM, QD1_SYSTEM

    class Live:
        def generate(self, messages, max_new_tokens=None, **kw):
            return GenerationResult(prompt="", response="Sub one?\nSub two?", prompt_tokens=0, generated_tokens=0, seconds=0.0)

    gen = KR.ReplayGenerator(PICKER_SYSTEM, Live(), (NAV1_SYSTEM, QD1_SYSTEM))
    out = gen.generate([{"role": "system", "content": QD1_SYSTEM}, {"role": "user", "content": "q"}])
    assert out.response == "Sub one?\nSub two?" and gen.prompt_sha is None   # the answer fingerprint is untouched


def test_hq1_additions_keep_places_inside_the_statute_cut():
    ranked = list(range(22))
    out = SP.with_reserved(ranked, ["hq1", "hq2"], SP.CFR_CANDIDATES_WITH_STATUTE)
    assert out[:SP.CFR_CANDIDATES_WITH_STATUTE][-2:] == ["hq1", "hq2"] and len(out) == SP.CFR_CANDIDATES_WITH_STATUTE
    assert SP.with_reserved([1, 2], ["hq"], 20) == [1, 2, "hq"] and SP.with_reserved([1, 2], [], 20) == [1, 2]


def test_a_section_added_as_a_paragraph_is_not_added_again_whole(monkeypatch):
    from regulus.rag import validator_switches as VS
    sections = {"117.475": [("(a)", "x" * 1000)], "117.405": [("(a)", "z" * 1000)]}
    scores = {"117.475": (0.95, "(a)"), "117.405": (0.90, "(a)")}
    cands = [{"section": s, "part": "117"} for s in sections]
    promotion = Promotion(sections=("117.410",), companions=("117.475(a)",), method="picker")
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"MS2", "PL1"})
    pr = _companion_promoter(monkeypatch, sections, scores)
    assert pr.reranked_companions("q", cands, promotion) == ["117.405"]   # 117.475 is cited already, as its (a)


def test_xc1_one_clearly_relevant_candidate_of_another_part_joins(monkeypatch):
    from regulus.rag import validator_switches as VS
    # V-004's shape: a Part 117 promotion, the FSVP section (Part 1) among the candidates
    sections = {"117.430": [("(a)", "x" * 500)], "1.506": [("(a)", "y" * 800)], "1.505": [("(a)", "z" * 800)],
                "507.175": [("(a)", "w" * 800)]}
    scores = {"117.430": (0.2, "(a)"), "1.506": (0.99, "(a)"), "1.505": (0.90, "(a)"), "507.175": (0.6, "(a)")}
    cands = [{"section": s, "part": s.split(".")[0]} for s in sections]
    promotion = Promotion(sections=("117.425",), companions=(), method="picker")
    pr = _companion_promoter(monkeypatch, sections, scores)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", (VS.VALIDATOR_PRECISION | {"XC1"}) - {"MS2", "PL1"})
    assert pr.reranked_companions("q", cands, promotion) == ["1.506"]       # the best other-part one, at most one
    monkeypatch.setattr(SP, "XC1_MIN_SCORE", 0.995)
    assert pr.reranked_companions("q", cands, promotion) == []              # none clearly relevant enough
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"XC1"})
    monkeypatch.setattr(SP, "XC1_MIN_SCORE", 0.85)
    assert pr.reranked_companions("q", cands, promotion) == []              # off: other parts are never pooled


def test_xg1_the_cross_part_companion_stays_in_the_promoted_programme(monkeypatch):
    from regulus.rag import validator_switches as VS
    # D-008's shape (full300e): drug 211.68 promoted, dietary-supplement 111.30 the best other-part candidate at 0.97
    sections = {"111.30": [("(a)", "x" * 800)], "213.68": [("(a)", "y" * 800)]}
    scores = {"111.30": (0.97, "(a)"), "213.68": (0.90, "(a)")}
    cands = [{"section": s, "part": s.split(".")[0]} for s in sections]
    promotion = Promotion(sections=("211.68",), companions=(), method="picker")
    pr = _companion_promoter(monkeypatch, sections, scores)
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", (VS.VALIDATOR_PRECISION | {"XC1"}) - {"MS2", "PL1", "XG1"})
    assert pr.reranked_companions("q", cands, promotion) == ["111.30"]      # XC1 alone: another programme's rule
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"XG1"})
    assert pr.reranked_companions("q", cands, promotion) == ["213.68"]      # XG1: the drug candidate instead


def test_xg1_programme_families():
    assert SP.subchapter("211.68") == SP.subchapter("314.50") == "CD"      # Subchapters C and D: human drugs
    assert SP.subchapter("225.1") == SP.subchapter("558.6") == "E"          # medicated feed CGMP is animal food
    assert SP.same_programme("102.5(d)", "101") and SP.same_programme("1003.10", "1004")
    assert not SP.same_programme("111.30", "211") and not SP.same_programme("201.5", "801")
    assert not SP.same_programme("101.7(t)", "201") and not SP.same_programme("507.125", "117")
    # Subchapter A applies across programmes, either way round (K-002: 812.47 beside 50.24; S-011: 1.511 beside 111.70)
    assert SP.same_programme("812.47", "50") and SP.same_programme("1.511(a)", "111") and SP.same_programme("4.4", "820")
    assert SP.same_programme("x", "211")                                    # unparseable: not filtered


# ------------------------------------------------------------ TE1: targeted revision of the sentences that failed


class Answers(ScriptedGenerator):
    def __init__(self, *answers):
        super().__init__(answers[0]); self.answers = list(answers)

    def generate(self, messages, max_new_tokens=None, on_text=None):
        self.response = self.answers[min(len(self.calls), len(self.answers) - 1)]
        return super().generate(messages, max_new_tokens)


TE1_DRAFT = ("21 CFR 820.10(b)(2) requires traceability procedures within 30 days [1]. "
             "21 CFR 820.10(b)(4) requires advisory notices [2].")


def _te1(monkeypatch):
    from regulus.rag import validator_switches as VS
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"TE1"})


def test_te1_only_the_failing_sentence_is_revised_and_the_rest_kept(monkeypatch):
    _te1(monkeypatch)
    g = Answers(TE1_DRAFT, "1: 21 CFR 820.10(b)(2) requires traceability procedures [1].", "unused full rewrite")
    out = RagPipeline(LexicalRetriever(TableDatabase()), g, RagConfig(answer_style="apply")).answer(TWO)
    ask = g.calls[1]["messages"]
    assert ask[0] == g.calls[0]["messages"][0] and ask[-1]["content"].startswith(g.calls[0]["messages"][-1]["content"])
    assert "THESE SENTENCES OF YOUR DRAFT ARE NOT SUPPORTED AS WRITTEN:\n1. 21 CFR 820.10(b)(2)" in ask[-1]["content"]
    assert len(g.calls) == 2                                                  # no full rewrite
    info = out.metadata["generation"]["correction_pass"]
    assert info["accepted"] is True and info["targeted"]["accepted"] is True
    assert "30 days" not in out.answer and "21 CFR 820.10(b)(4) requires advisory notices [2]." in out.answer


def test_te1_a_revision_that_does_not_validate_falls_back_to_the_full_rewrite(monkeypatch):
    _te1(monkeypatch)
    good = "21 CFR 820.10(b)(2) requires traceability procedures [1]. 21 CFR 820.10(b)(4) requires advisory notices [2]."
    g = Answers(TE1_DRAFT, "1: 21 CFR 820.10(b)(2) requires traceability procedures within 45 days [1].", good)
    out = RagPipeline(LexicalRetriever(TableDatabase()), g, RagConfig(answer_style="apply")).answer(TWO)
    info = out.metadata["generation"]["correction_pass"]
    assert len(g.calls) == 3 and "PROBLEMS FOUND IN THE DRAFT:" in g.calls[2]["messages"][-1]["content"]
    assert info["targeted"]["accepted"] is False and info["accepted"] is True and out.answer.startswith(good[:40])


def test_te1_splice_rules():
    from regulus.rag.pipeline import apply_targeted

    draft = "A must be done [1]. B must be done within 9 days [2]. C must be done [3]."
    assert apply_targeted(draft, ["B must be done within 9 days [2]."], "1: DELETE") == "A must be done [1]. C must be done [3]."
    # a replacement without its citation keeps the original markers
    assert apply_targeted(draft, ["B must be done within 9 days [2]."], "1: B must be done.") == \
        "A must be done [1]. B must be done [2]. C must be done [3]."
    # an unanswered number leaves the sentence; a sentence not in the draft gives None
    assert apply_targeted(draft, ["B must be done within 9 days [2]."], "nothing useful") == draft
    assert apply_targeted(draft, ["Z is required [4]."], "1: DELETE") is None


def test_te1_only_when_every_error_is_one_sentences():
    from regulus.rag.pipeline import targeted_problems
    from regulus.rag.validation import ValidationProblem, ValidationReport

    def report(*problems):
        return SimpleNamespace(problems=list(problems))
    fig = ValidationProblem("unsupported_figure", "error", "the figure 9 appears in no source", sentence="B within 9 days [2].")
    missing = ValidationProblem("requirement_not_addressed", "error", "21 CFR 1.1 is not addressed", requirement_id="r1")
    assert targeted_problems(report(fig))[0][0] == "B within 9 days [2]."
    assert targeted_problems(report(fig, missing)) is None                     # new content is needed: a full rewrite


# ------------------------------------------------------------ CP1: each compared item judged on its own


def test_cp1_compared_items_in_the_questions_words():
    from regulus.rag.section_promotion import compared_items
    assert compared_items("What is the difference between a Substantial Equivalence report, an Exemption from SE "
                          "request, and a PMTA?") == ["a Substantial Equivalence report", "an Exemption from SE request", "a PMTA"]
    assert compared_items("How does the Part 117 supply-chain program differ from the FSVP requirements for importers?") \
        == ["the Part 117 supply-chain program", "the FSVP requirements for importers"]
    assert compared_items("What definitions distinguish an open system from a closed system under Part 11?") \
        == ["an open system", "a closed system"]
    assert compared_items("What records must be kept under Part 111?") == []


def test_cp1_an_item_query_spells_out_its_acronym():
    from regulus.rag.section_promotion import item_query
    assert item_query("a PMTA") == "What is a PMTA (premarket tobacco product application)?"
    assert item_query("a recall") == "What is a recall?"


# ------------------------------------------------------------ DX1: a definition that points elsewhere is followed once


class _DefDb:
    ROWS = {"4.2": [{"paragraph_path": "", "body_text": "Combination product has the meaning set forth in § 3.2(e) of this chapter."},
                    {"paragraph_path": "", "body_text": "Device has the meaning set forth in § 3.2(f) of this chapter."}],
            "207.1": [{"paragraph_path": "", "body_text": "Relabeler means a person who changes the content of the labeling."}],
            "251.2": [{"paragraph_path": "", "body_text": "Relabeler has the meaning set forth in § 207.1 of this chapter."}]}

    def query(self, sql, params):
        return self.ROWS.get(params["s"], [])


def test_dx1_a_definition_redirect_is_followed_for_a_definition_question():
    pr = SectionPromoter(_DefDb())
    assert pr.definition_redirects(["4.1", "4.2"], "How is a combination product defined?") == ["3.2(e)"]


def test_dx1_not_for_a_question_that_asks_no_definition():
    pr = SectionPromoter(_DefDb())
    assert pr.definition_redirects(["4.2"], "What does Part 4 require for a device constituent part?") == []


def test_dx1_a_whole_section_is_never_added():
    # 207.1's definitions carry no paragraph letter: the redirect is not followed rather than supplying all of 207.1
    assert SectionPromoter(_DefDb()).definition_redirects(["251.2"], "What is a relabeler under Part 251?") == []


class _CutOff(Answers):
    """The second call (the targeted reply) is cut off at its allowance after one complete line."""
    def generate(self, messages, max_new_tokens=None, on_text=None):
        out = super().generate(messages, max_new_tokens)
        if len(self.calls) == 2:
            from dataclasses import replace as _r
            return _r(out, finished=False)
        return out


def test_te1_a_reply_cut_off_after_a_complete_line_uses_that_line(monkeypatch):
    _te1(monkeypatch)
    g = _CutOff(TE1_DRAFT, "1: 21 CFR 820.10(b)(2) requires traceability procedures [1].\n2: and then the model kept wri",
                "unused full rewrite")
    out = RagPipeline(LexicalRetriever(TableDatabase()), g, RagConfig(answer_style="apply")).answer(TWO)
    info = out.metadata["generation"]["correction_pass"]
    assert len(g.calls) == 2 and info["targeted"]["cut_off"] is True and info["targeted"]["accepted"] is True
    assert "30 days" not in out.answer and "kept wri" not in out.answer


def test_te1_a_reply_cut_off_inside_its_only_line_is_not_spliced(monkeypatch):
    _te1(monkeypatch)
    good = "21 CFR 820.10(b)(2) requires traceability procedures [1]. 21 CFR 820.10(b)(4) requires advisory notices [2]."
    g = _CutOff(TE1_DRAFT, "1: 21 CFR 820.10(b)(2) requires trace", good)
    out = RagPipeline(LexicalRetriever(TableDatabase()), g, RagConfig(answer_style="apply")).answer(TWO)
    info = out.metadata["generation"]["correction_pass"]
    assert len(g.calls) == 3 and info["targeted"]["spliced"] is False and out.answer.startswith(good[:40])
