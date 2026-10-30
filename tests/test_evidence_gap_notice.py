import pytest
"""P1 (2026-10-20): when the hard ceiling keeps paragraphs of a requested provision out of the prompt, the delivered
answer names what was left out and the record says the evidence was not complete. Found on the benchmark: for
F-002 46 required rows of Part 117 and for R-002 13 rows of Part 803 were cut, while coverage said COMPLETE and the
answer said nothing. The notice is appended after validation (like the unresolved-provision statement); the prompt
and the model's text are unchanged, and an answer with no cut evidence is unchanged."""

from regulus.accuracy.claims import MODEL, classify
from regulus.evaluation.grounding_eval import split_sentences
from regulus.rag import ContextPolicy, RagConfig, RagPipeline
from regulus.rag.prompt import withheld_evidence_statement
from regulus.retrieval import LexicalRetriever
from tests.test_rag import FakeGenerator
from tests.test_retrieval_plan import TABLE, TableDatabase

# fixtures sized to overflow the CPU-era evidence budget (tests/conftest.py)
pytestmark = pytest.mark.usefixtures("cpu_era_budget")

QUESTION = "What do 820.10(b)(1) through (b)(4) require?"
CUT = ["21 CFR 820.10(b)(3)", "21 CFR 820.10(b)(4)"]


def answer(table, policy=None, response="The requirement is stated in the source [1]."):
    gen = FakeGenerator(response)
    config = RagConfig(context=policy) if policy is not None else RagConfig()
    return RagPipeline(LexicalRetriever(TableDatabase(table)), gen, config).answer(QUESTION), gen


def ceiling():
    return ContextPolicy(max_total_chars=3600, hard_max_total_chars=3600, required_max_chars_per_result=4000)


def test_the_statement_names_the_requested_provision_and_how_much_was_left_out_and_nothing_it_says():
    one = withheld_evidence_statement({"21 CFR 803.3": 1})
    assert one.startswith("Evidence limit: not all of the retrieved text of 21 CFR 803.3 fit") and "(1 paragraph left out)" in one
    two = withheld_evidence_statement({"21 CFR part 117": 46, "21 CFR part 210": 9})
    assert "21 CFR part 117 and 21 CFR part 210" in two and "(55 paragraphs left out)" in two
    assert two.endswith("The answer above does not cover that text; consult the full provisions.")


def test_the_statement_is_recognised_as_the_pipeline_s_own_words_not_a_claim():
    for sentence in split_sentences(withheld_evidence_statement({"21 CFR 803.3": 2, "21 CFR part 117": 46})):
        assert classify(sentence).origin != MODEL, sentence


def test_an_answer_whose_required_evidence_was_cut_says_so_and_is_not_recorded_complete():
    long_table = [dict(r, body_text="L" * 1500) for r in TABLE]
    out, gen = answer(long_table, ceiling())
    gap = out.metadata["evidence_gap"]
    assert gap["reason"] == "hard_ceiling" and gap["withheld"] == CUT and gap["notice_appended"]
    assert gap["left_out"] == {"21 CFR 820.10(b)(3)": 1, "21 CFR 820.10(b)(4)": 1}
    assert out.metadata["coverage"]["evidence_complete"] is False
    assert out.answer.endswith(withheld_evidence_statement(gap["left_out"]))
    prompt = gen.messages[-1][-1]["content"]
    assert "Evidence limit" not in prompt  # the model's input is unchanged


def test_an_answer_with_nothing_cut_is_unchanged():
    long_table = [dict(r, body_text="L" * 1500) for r in TABLE]
    with_cut, _ = answer(long_table, ceiling())
    plain, _ = answer(TABLE)
    assert plain.metadata["evidence_gap"] is None and plain.metadata["coverage"]["evidence_complete"] is True
    assert "Evidence limit" not in plain.answer
    # the cut answer is its own text plus one notice line, appended once
    body, notice = with_cut.answer.rsplit("\n", 1)
    assert notice == withheld_evidence_statement(with_cut.metadata["evidence_gap"]["left_out"]) and "Evidence limit" not in body
