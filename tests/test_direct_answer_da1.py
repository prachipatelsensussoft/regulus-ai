"""DA1 (2 Nov 2026): the apply format's first part is the direct answer only; the requirements stay with their paragraphs."""
from regulus.rag import validator_switches
from regulus.rag.answer_plan import APPLY_FORMAT, DIRECT_ANSWER_ONLY, apply_format


def _switch(monkeypatch, on):
    monkeypatch.setattr(validator_switches, "VALIDATOR_PRECISION",
                        validator_switches.VALIDATOR_PRECISION - {"DA1"} | ({"DA1"} if on else set()))


def test_off_the_format_is_unchanged(monkeypatch):
    _switch(monkeypatch, False)
    assert apply_format() == APPLY_FORMAT


def test_on_only_the_first_part_is_limited(monkeypatch):
    _switch(monkeypatch, True)
    text = apply_format()
    assert text.replace(DIRECT_ANSWER_ONLY, "") == APPLY_FORMAT          # every other instruction is kept, in order
    assert text.index(DIRECT_ANSWER_ONLY) < text.index("Then go through the requested provisions paragraph by paragraph")


def test_the_identity_names_the_wording_in_use(monkeypatch):
    from regulus.system_identity import _apply_format

    _switch(monkeypatch, False)
    off = _apply_format()
    _switch(monkeypatch, True)
    assert _apply_format() != off and DIRECT_ANSWER_ONLY in _apply_format()
