"""SB1: co-promoted FD&C Act text shares the regulation evidence's effective ceiling (CB1), not the 9,000 default."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from regulus.rag import RagPipeline
from regulus.rag import validator_switches as VS
from regulus.rag.context import ContextPolicy
from regulus.rag.section_promotion import Promotion
from regulus.retrieval import LexicalRetriever
from regulus.retrieval.statute import StatuteIndex
from tests.test_answer_validation import ScriptedGenerator
from tests.test_retrieval_plan import TableDatabase

ACT_TEXT = "(a) The responsible person shall submit to the Secretary each serious adverse event report received. " * 3


@pytest.fixture
def pipe(monkeypatch):
    rows = [{"fdca_section": "605", "usc_section": "364a", "paragraph_path": "(a)", "ordinal": 0, "heading": "Adverse events",
             "paragraph_heading": "", "body_text": ACT_TEXT, "top_caption": "", "source_url": "https://uscode.house.gov"}]
    st = StatuteIndex(rows, [set()])
    import regulus.retrieval.statute as statute
    monkeypatch.setattr(statute, "statute_index", lambda db: st)
    return RagPipeline(LexicalRetriever(TableDatabase()), ScriptedGenerator("x"))


def _selection(total_chars, ceiling=15000):
    # what select_evidence returns: the regulation's evidence, selected under the effective (CB1) policy
    return SimpleNamespace(selected=(object(),), dropped=(), total_chars=total_chars, diagnostics={},
                           policy=ContextPolicy(hard_max_total_chars=ceiling))


def _co_promotion():
    return Promotion(sections=("1.1",), companions=(), method="picker", statute=("605",), statute_rows=(0,),
                     statute_blocks=((0,),))


def test_sb1_is_off_by_default():
    assert "SB1" not in VS.DEFAULT.split(",")


def test_off_a_regulation_over_9000_characters_leaves_no_room_for_the_act_and_the_drop_vanishes(pipe, monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"SB1"})
    sel = _selection(10_000)
    out = pipe.statute_selection(_co_promotion(), sel)
    assert out is sel and len(out.selected) == 1 and not out.dropped   # today: the Act block is neither shown nor named


def test_on_the_act_block_fits_under_the_effective_ceiling(pipe, monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"SB1"})
    out = pipe.statute_selection(_co_promotion(), _selection(10_000))
    assert len(out.selected) == 2 and not out.dropped
    assert out.selected[1].reason == "statute_promotion" and out.total_chars <= 15_000


def test_on_the_effective_ceiling_still_binds(pipe, monkeypatch):
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"SB1"})
    out = pipe.statute_selection(_co_promotion(), _selection(14_900))
    # still too big: left out, and now recorded as dropped required evidence (the answer's evidence-limit notice names it)
    assert len(out.selected) == 1 and [(d.reason, d.required) for d in out.dropped] == [("hard_ceiling_chars", True)]


def test_a_statute_only_promotion_is_unchanged(pipe, monkeypatch):
    statute_only = Promotion(sections=(), companions=(), method="named_statute", statute=("605",), statute_rows=(0,),
                             statute_blocks=((0,),))
    results = []
    for on in (False, True):
        monkeypatch.setattr(VS, "VALIDATOR_PRECISION", (VS.VALIDATOR_PRECISION | {"SB1"}) if on else
                            (VS.VALIDATOR_PRECISION - {"SB1"}))
        out = pipe.statute_selection(statute_only, _selection(10_000))
        results.append(([s.reason for s in out.selected], [d.reason for d in out.dropped], out.total_chars))
    assert results[0] == results[1] and results[0][0] == ["statute_promotion"]
