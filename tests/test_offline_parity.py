"""Offline evaluation versus the LIVE evaluation path, input by input.

The live reference is the real `serve_ui.Engine.assess` (the UI's evaluator call, including whatever
`repair=` wiring the working tree has), driven here with an injected aligner and identifier index instead
of the embedding model and the database. Each result's `parity` block says, per input, whether offline is
identical, differs without effect, or is limited; these tests hold it to that:

  - where `live_equivalent` is true, the offline assessment IS the live one (verdicts, claims, dimensions,
    score, review status), differing at most in the recorded flags checked.aligner / identifier_index_used;
  - where aligner choice, the identifier index or the repair record would change the outcome, the row is
    flagged limited, and the tests show the difference is real.

Answers are SCRIPTED over the in-memory table; the "semantic" embedder is a deterministic stand-in. No
model, no database, no GPU.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from regulus.accuracy.alignment import LexicalAligner, SemanticAligner, passages
from regulus.accuracy.review import NOT_EVALUATED, PASS, question_review
from regulus.config import load_settings
from regulus.evaluation import offline_evaluation as O
from regulus.rag import RagPipeline
from regulus.retrieval import LexicalRetriever
from regulus.system_identity import configured_rag_config
from regulus.traceability import measurement as M
from tests.test_accuracy_observer import CASES, Scripted, answer_one
from tests.test_accuracy_window_support import BROAD, S1, S2, S3, S4, T211
from tests.test_measurement_record import A, B, C, INVENTED, RUN, build, focused
from tests.test_retrieval_plan import TableDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serve_ui import Engine  # noqa: E402

# scripted model drafts: the model-drafted path (tests/conftest.py)
pytestmark = pytest.mark.usefixtures("model_drafted_path")

PARAPHRASE = "There must be enough qualified personnel to perform and supervise the manufacture of each drug product. [3]"
NAMES_ABSENT = f"{A} [1] See also 21 CFR 211.999. [1]"


class HashEmbedder:
    """Deterministic stand-in for the pinned encoder: a hashed bag of words, unit length."""

    def embed_batch(self, texts):
        out = []
        for t in texts:
            v = np.zeros(64, dtype=np.float32)
            for w in t.lower().split():
                v[int(hashlib.sha256(w.encode()).hexdigest(), 16) % 64] += 1.0
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.stack(out)

    def describe(self):
        return {"model_id": "test/hash-embedder"}


class FavourEmbedder(HashEmbedder):
    """Ranks one evidence sentence first for every claim: passages (and two-sentence windows) point one way,
    the favoured passage and every claim the other."""

    def __init__(self, favoured: str, passage_texts):
        self.favoured, self.passage_texts = favoured, list(passage_texts)

    def embed_batch(self, texts):
        e1, e2 = np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)
        return np.stack([e1 if t == self.favoured else e2 if any(p in t for p in self.passage_texts) else e1
                         for t in texts])


class DictIndex:
    """The live identifier index, from a table instead of the database; records every lookup."""

    def __init__(self, status=None):
        self.status, self.lookups = dict(status or {}), []

    def section_status(self, section):
        self.lookups.append(section)
        return self.status.get(section, (True, True))

    def part_status(self, part):
        self.lookups.append(f"part {part}")
        return self.status.get(f"part {part}", (True, True))


def live(out, *, aligner, index):
    """The real UI evaluator call (serve_ui.Engine.assess), with this aligner and index."""
    engine = object.__new__(Engine)
    engine._aligner, engine._identifier_index = aligner, index
    d = engine.assess(out).to_dict()
    d.pop("elapsed_ms")
    return json.loads(json.dumps(d, default=str))


def offline(out, *, aligner=None):
    rec = json.loads(json.dumps(build(out)))
    row = {"question_id": "Q-1", "question": out.question, "answer": out.answer}
    result, _ = O.evaluate_record(rec, row, run_id=RUN["run_id"], ordinal=1, evaluator={}, aligner=aligner)
    assert result["outcome"] == O.EVALUATED, result["reason"]
    return result


def without_flags(d, *flags):
    d = json.loads(json.dumps(d))
    for flag in flags:
        d["checked"].pop(flag, None)
    return d


def production(response, question=BROAD, table=T211):
    return RagPipeline(LexicalRetriever(TableDatabase(table)), Scripted(response),
                       configured_rag_config(load_settings())).answer(question)


MAKERS = [*[(lambda q=q, r=r: answer_one(q, r)) for q, r in CASES],
          lambda: focused(f"{A} [1] {B} [2] {C} [3]"), lambda: focused(INVENTED), lambda: focused(f"{C} [3]"),
          lambda: focused(NAMES_ABSENT), lambda: focused(PARAPHRASE)]


# ------------------------------------------------------------------ the claim holds where it is made


@pytest.mark.parametrize("semantic", [False, True], ids=["offline-lexical", "offline-semantic"])
def test_where_parity_is_claimed_the_offline_assessment_is_the_live_one(semantic):
    claimed = 0
    for make in MAKERS:
        out = make()
        result = offline(out, aligner=SemanticAligner(HashEmbedder()) if semantic else None)
        index = DictIndex()
        reference = live(out, aligner=SemanticAligner(HashEmbedder()), index=index)   # the UI default: semantic
        parity = result["parity"]
        assert parity["live_equivalent"] == all(parity[k]["status"] in (O.IDENTICAL, O.NO_EFFECT)
                                                for k in ("aligner", "identifier_index", "repair"))
        assert (parity["identifier_index"]["status"] == O.NO_EFFECT) == (index.lookups == [])
        if not parity["live_equivalent"]:
            continue
        claimed += 1
        flags = ("identifier_index_used",) + (() if semantic else ("aligner",))
        assert without_flags(result["assessment"], *flags) == without_flags(reference, *flags)
        assert result["review_status"] == question_review(reference, question=out.question)["status"]
        assert result["score"] == reference["accuracy"]
        assert result["assessment"]["checked"]["claims"] == reference["checked"]["claims"]
        assert result["assessment"]["dimensions"] == reference["dimensions"]
    assert claimed >= (5 if semantic else 1)                 # never vacuous


# ------------------------------------------------------------------ the aligner


def test_lexical_and_semantic_alignment_pick_different_passages_for_the_same_claim():
    out = focused(PARAPHRASE)
    ps = passages(out.evidence_snapshot.items)
    claim = PARAPHRASE.rsplit(" [", 1)[0]
    lexical = LexicalAligner().align(claim, ps)
    assert len(lexical) >= 2 and lexical[0].passage.source_index == 3          # 211.25(c) by word overlap
    favoured = lexical[1].passage                                               # also clears the floor
    semantic = SemanticAligner(FavourEmbedder(favoured.text, [p.text for p in ps])).align(claim, ps)
    assert semantic[0].passage == favoured and semantic[0].passage.source_index != 3


def test_an_aligner_that_changes_a_claim_match_is_flagged_and_the_difference_is_real():
    out = focused(PARAPHRASE)
    ps = passages(out.evidence_snapshot.items)
    favoured = LexicalAligner().align(PARAPHRASE.rsplit(" [", 1)[0], ps)[1].passage
    semantic = lambda: SemanticAligner(FavourEmbedder(favoured.text, [p.text for p in ps]))   # noqa: E731
    lex, sem = offline(out), offline(out, aligner=semantic())
    assert lex["parity"]["aligner"]["status"] == O.LIMITED and lex["parity"]["aligner"]["claims_aligned"] >= 1
    assert sem["parity"]["aligner"]["status"] == O.IDENTICAL
    assert lex["assessment"]["checked"]["claims"] != sem["assessment"]["checked"]["claims"]
    assert (lex["review_label"], sem["review_label"]) == ("FAIL", "NOT EVALUATED")   # it decides the status here
    reference = live(out, aligner=semantic(), index=DictIndex())                      # same aligner: the live result
    assert without_flags(sem["assessment"], "identifier_index_used") == without_flags(reference, "identifier_index_used")


def test_an_answer_no_claim_of_which_aligns_is_unaffected_by_the_aligner():
    out = answer_one(*CASES[6])                                   # out of scope: no evidence to align against
    result = offline(out)
    assert result["parity"]["aligner"] == {"status": O.NO_EFFECT, "offline": "lexical", "claims_aligned": 0}
    assert result["parity"]["live_equivalent"]
    reference = live(out, aligner=SemanticAligner(HashEmbedder()), index=DictIndex())
    assert without_flags(result["assessment"], "aligner", "identifier_index_used") == \
        without_flags(reference, "aligner", "identifier_index_used")


def test_semantic_offline_alignment_is_never_loaded_unless_asked(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("the embedding model was loaded")

    monkeypatch.setattr(O, "semantic_aligner", forbidden)
    identity = O.evaluator_identity(Path("."))
    assert (identity["aligner"], identity["embedder"]) == ("lexical", None)


def test_the_semantic_option_records_its_embedder(tmp_path, monkeypatch):
    from tests.test_offline_evaluation import make_run, read
    from tests.test_measurement_record import runner

    monkeypatch.setattr(O, "semantic_aligner", lambda: SemanticAligner(HashEmbedder()))
    run_dir = make_run(tmp_path)
    assert runner.evaluate_offline(run_dir, str(tmp_path / "sem.jsonl"), "semantic") == 0
    lines = read(tmp_path / "sem.jsonl")
    assert {l["result"]["evaluator"]["aligner"] for l in lines} == {"semantic"}
    assert lines[0]["result"]["evaluator"]["embedder"] == {"model_id": "test/hash-embedder"}
    assert all(l["result"]["parity"]["aligner"]["status"] == O.IDENTICAL for l in lines)


# ------------------------------------------------------------------ the identifier index


def test_the_identifier_index_changes_the_live_result_and_offline_says_so():
    out = focused(NAMES_ABSENT)
    result = offline(out)
    assert result["parity"]["identifier_index"] == {"status": O.LIMITED, "lookups": [{"kind": "section", "id": "211.999"}]}
    assert not result["parity"]["live_equivalent"]
    exists = live(out, aligner=LexicalAligner(), index=DictIndex({"211.999": (True, True)}))
    absent = live(out, aligner=LexicalAligner(), index=DictIndex({"211.999": (False, False)}))
    # today's database decides it: the same saved answer is judged differently by what the index says
    assert absent["checked"]["identifiers_verified_absent"] == 1 and exists["checked"]["identifiers_verified_absent"] == 0
    assert absent != exists
    assert result["assessment"]["checked"]["identifiers_verified_absent"] == 0   # offline: unverifiable, not absent
    assert result["assessment"]["checked"]["identifier_index_used"] is False


def test_a_refusal_of_a_requested_section_would_consult_the_index_for_reservation():
    out = answer_one(*CASES[3])                                   # 820.99(z): declined, nothing retrieved
    result = offline(out)
    assert result["parity"]["identifier_index"]["lookups"] == [{"kind": "section", "id": "820.99"}]
    reserved = live(out, aligner=LexicalAligner(), index=DictIndex({"820.99": (True, False)}))
    missing = live(out, aligner=LexicalAligner(), index=DictIndex({"820.99": (False, False)}))
    assert reserved["checked"]["abstention"] != missing["checked"]["abstention"]


# ------------------------------------------------------------------ the repair record


def test_a_repaired_answer_that_lost_required_content_is_judged_as_live_judges_it():
    sentences = [S1, S2[:-1] + ", unless the batch was destroyed.", S3, S4]
    out = production(" ".join(f"{s} [1]" for s in sentences))
    assert out.metadata["validation"]["repair"] == "invented_conditions_removed"
    result = offline(out)
    reference = live(out, aligner=LexicalAligner(), index=DictIndex())
    completeness = next(d for d in result["assessment"]["dimensions"] if d["dimension"] == "completeness")
    assert completeness["state"] == "not_established" and "21 CFR 211.192" in completeness["detail"]
    assert result["review_status"] == question_review(reference, question=out.question)["status"] == NOT_EVALUATED
    assert without_flags(result["assessment"], "identifier_index_used") == without_flags(reference, "identifier_index_used")
    assert result["parity"]["repair"]["status"] == O.IDENTICAL


def test_a_complete_repaired_answer_is_not_penalised_offline_or_live():
    out = production(f"{S1} [1] {S2} [1] {S3} [1] {S4} [1] Records are exempt unless the batch was destroyed. [1]")
    assert out.metadata["validation"]["repair"] == "invented_conditions_removed"
    result = offline(out)
    reference = live(out, aligner=LexicalAligner(), index=DictIndex())
    assert result["review_status"] == PASS and result["score"] == 100
    assert without_flags(result["assessment"], "identifier_index_used") == without_flags(reference, "identifier_index_used")
    assert result["parity"]["repair"]["status"] == O.IDENTICAL


@pytest.mark.parametrize("make", [lambda: answer_one(*CASES[4]), lambda: answer_one(*CASES[2]), lambda: focused(INVENTED)],
                         ids=["extractive_replacement", "appended_missing_provisions", "invented_conditions_removed"])
def test_every_repaired_answer_gets_the_live_repair_record(make):
    out = make()
    assert out.metadata["validation"]["repair"] is not None
    result = offline(out)
    reference = live(out, aligner=LexicalAligner(), index=DictIndex())
    assert without_flags(result["assessment"], "identifier_index_used") == without_flags(reference, "identifier_index_used")


@pytest.mark.parametrize("initial", [None, {"status": "valid"}, "valid"], ids=["missing", "fields-missing", "legacy-status-string"])
def test_a_captured_validation_without_a_sound_initial_report_is_not_evaluable(initial):
    out = focused(INVENTED)                                       # live: completeness not established
    rec = json.loads(json.dumps(build(out)))
    rec["validation"]["initial"] = initial                        # e.g. an archived row's bare status string
    rec["record_sha256"] = M.record_hash(rec)
    result, _ = O.evaluate_record(rec, {"question_id": "Q-1", "question": out.question, "answer": out.answer},
                                  run_id=RUN["run_id"], ordinal=1, evaluator={})
    assert result["outcome"] == O.NOT_EVALUABLE and "initial" in result["reason"]
    assert result["review_status"] is None and result["assessment"] is None   # no PASS can be shown


def test_validation_that_did_not_run_passes_no_repair_record_as_live():
    out = answer_one(*CASES[6])                                   # out of scope: no validation
    assert out.metadata.get("validation") is None
    result = offline(out)
    reference = live(out, aligner=SemanticAligner(HashEmbedder()), index=DictIndex())
    assert without_flags(result["assessment"], "aligner", "identifier_index_used") == \
        without_flags(reference, "aligner", "identifier_index_used")


# ------------------------------------------------------------------ repeatability


def test_evaluating_the_same_record_twice_gives_the_same_result_and_parity():
    out = focused(NAMES_ABSENT)
    rec = json.loads(json.dumps(build(out)))
    row = {"question_id": "Q-1", "question": out.question, "answer": out.answer}
    first, _ = O.evaluate_record(rec, row, run_id=RUN["run_id"], ordinal=1, evaluator={})
    second, _ = O.evaluate_record(rec, row, run_id=RUN["run_id"], ordinal=1, evaluator={})
    assert first == second and M.sha256_json(first) == M.sha256_json(second)
