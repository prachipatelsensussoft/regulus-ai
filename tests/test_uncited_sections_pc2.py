"""PC2 (2 Nov 2026): a supplied section the answer never cites contributes its most relevant paragraph, verbatim."""
from types import SimpleNamespace as N

import pytest

from regulus.rag.pipeline import PC2_LEAD, RagPipeline


def _src(i, citation, text):
    return N(index=i, citation=citation, text=text, corpus="ecfr")


PKG = N(sources=[
    _src(1, "21 CFR 117.251(a)", "(a) FDA may withdraw a qualified facility exemption in the event of an active "
                                 "investigation of a foodborne illness outbreak that is directly linked to the facility."),
    _src(2, "21 CFR 117.257(a)", "(a) If FDA withdraws the exemption, the facility may appeal the order by submitting a "
                                 "written appeal within 15 calendar days of receipt of the order."),
    _src(3, "21 CFR 117.264(a)", "(a) A facility may request reinstatement of an exemption that FDA has withdrawn by "
                                 "submitting a written request presenting data and information demonstrating compliance."),
    _src(4, "21 CFR 117.270(a)", "(a) This paragraph concerns something unrelated to the question that was asked by the user."),
])
SCORES = {2: 0.9, 3: 0.7, 4: 0.1}


@pytest.fixture
def scorer(monkeypatch):
    import regulus.retrieval.rerank as RR

    class Fake:
        def score(self, question, texts):
            by_text = {s.text: SCORES.get(s.index, 0.0) for s in PKG.sources}
            return [by_text[t] for t in texts]
    monkeypatch.setattr(RR, "reranker", lambda name: Fake())


def test_uncited_supplied_sections_contribute_their_best_paragraph(scorer):
    out = RagPipeline.uncited_sections(None, "FDA may withdraw the exemption during an outbreak investigation [1].", PKG,
                                       "When may FDA withdraw a qualified facility exemption and what then?")
    assert out.startswith(PC2_LEAD)
    assert "21 CFR 117.257(a)" in out and "21 CFR 117.264(a)" in out          # appeal and reinstatement (F-018)
    assert "21 CFR 117.270(a)" not in out and "21 CFR 117.251(a)" not in out  # below the floor / already cited


def test_an_answer_that_cites_nothing_is_not_completed(scorer):
    assert RagPipeline.uncited_sections(None, "No citation here.", PKG, "Q?") is None


def test_at_most_two_sections_are_added(scorer, monkeypatch):
    import regulus.rag.pipeline as PL
    monkeypatch.setattr(PL, "PC2_MAX", 1)
    out = RagPipeline.uncited_sections(None, "Withdrawal applies [1].", PKG, "Q?")
    assert "117.257(a)" in out and "117.264(a)" not in out                   # the best one only
