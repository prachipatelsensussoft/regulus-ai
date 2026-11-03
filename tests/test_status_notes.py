"""QS1 (3 Nov 2026): a status note gated by its switch and by the question it answers (regulus.rag.status_notes)."""


def test_qs1_the_qmsr_effective_date_note_goes_only_with_a_question_asking_for_it(monkeypatch):
    from types import SimpleNamespace as N
    from regulus.rag import validator_switches as VS
    from regulus.rag.status_notes import status_notes, supplied_status_notes
    src = [N(index=1, citation="21 CFR 820.10(a)", heading=None, corpus="ecfr")]
    asked = "What changed when FDA amended Part 820 to the Quality Management System Regulation, and what is the compliance date?"
    ids = lambda q: [n["id"] for n in status_notes("The QMS must comply with ISO 13485 [1].", src, q)]   # noqa: E731
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - {"QS1"})
    assert "qmsr-effective" not in ids(asked)                                   # off: nothing changes
    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION | {"QS1"})
    assert ids(asked) == ["qmsr-effective"]                                    # M-030 (300Q)
    assert "qmsr-effective" not in ids("What must a manufacturer document for complaint handling?")
    assert "qmsr-effective" not in ids(None)
    # never in the prompt (ST2 passes no question): the note is a post-answer status line only
    assert "qmsr-effective" not in [n["id"] for n in supplied_status_notes(src)]
