"""CF1 (2 Nov 2026): a companion section keeps its best paragraph when a long first section would take every slot."""
import pytest

from regulus.planning import plan_query
from regulus.rag import select_evidence
from regulus.rag import validator_switches
from regulus.retrieval import LexicalRetriever
from tests.test_retrieval_plan import TableDatabase, row

pytestmark = pytest.mark.usefixtures("cpu_era_budget")

QUESTION = "What do 21 CFR 117.5, 117.3 and 117.201 require of a qualified facility?"


def _long(lead, size):
    filler = "The exemption from the requirements of subparts C and G applies to the facility as described here. "
    return (lead + " " + filler * (size // len(filler) + 1))[:size]


def _table():
    # shaped like F-002: 117.5's exemptions fill the ceiling; 117.3 defines the term; 117.201 states the modified duties
    return ([row("117.5", f"({c})", k, text=_long(f"({c}) Exemption {c} for a qualified facility.", 1900))
             for k, c in enumerate("abcdefgh")]
            + [row("117.3", "", 20, text=("Qualified facility means a very small business or a facility with average annual "
                                          "sales under $500,000, as further described in this definition. " * 8)[:700]),
               row("117.201", "(a)", 30, text=("(a) A qualified facility must submit an attestation to FDA that it is a "
                                               "qualified facility, as further described in this paragraph. " * 8)[:700])])


def _switch(monkeypatch, *on):
    monkeypatch.setattr(validator_switches, "VALIDATOR_PRECISION",
                        validator_switches.VALIDATOR_PRECISION - {"CF1"} | set(on))


def _sections(selection):
    return {s.evidence.citation.split("(")[0].replace("21 CFR ", "") for s in selection.selected}


def _select():
    registry = LexicalRetriever(TableDatabase(_table())).run_plan(plan_query(QUESTION)).registry
    return select_evidence(registry, question=QUESTION)


def test_off_the_first_section_takes_every_slot(monkeypatch):
    _switch(monkeypatch)
    sel = _select()
    assert sel.overflow, "the fixture must not fit, or it tests nothing"
    assert _sections(sel) == {"117.5"}


def test_on_each_companion_keeps_its_best_paragraph(monkeypatch):
    _switch(monkeypatch, "CF1")
    sel = _select()
    assert {"117.5", "117.3", "117.201"} <= _sections(sel)
    ordinals = [s.evidence.result.metadata["ordinal"] for s in sel.selected]
    assert ordinals == sorted(ordinals)                                   # still presented in document order
