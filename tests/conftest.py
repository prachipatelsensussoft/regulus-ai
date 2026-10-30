import pytest

from regulus.config import Settings, load_settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings(use_dotenv=False)


# Composers that answer a question from the regulation's text before any draft is written (WS1: an overview question
# about a section's own heading is answered with the whole section). Tests of the evaluator, the measurement record and
# the validator feed the pipeline a scripted model draft; with such a composer on, the draft is never used and they stop
# testing what they are for. They declare the model-drafted path with this fixture (30 Oct, promoting WS1).
_DRAFT_BYPASSING_COMPOSERS = frozenset({"WS1"})


@pytest.fixture
def model_drafted_path(monkeypatch):
    from regulus.rag import validator_switches as VS

    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - _DRAFT_BYPASSING_COMPOSERS)


# Lines added to an answer after it validated, about its content (ISO1: the ISO 13485 clauses the cited Part 820
# provisions name). Tests that the delivered answer is exactly the validated (or repaired) draft - delivery routes,
# streaming equivalence, repair in place - declare that path with this fixture (30 Oct, promoting ISO1).
_POST_ANSWER_ADDITIONS = frozenset({"ISO1"})


@pytest.fixture
def delivered_as_drafted(monkeypatch):
    from regulus.rag import validator_switches as VS

    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - _POST_ANSWER_ADDITIONS)


# The evidence budget as it was built for the 0.6B model on the CPU: a 9,000-character / 24-source hard ceiling with every
# block charged for a URL line the prompt never renders. CB1 (15,000), HX1 (48) and BA1 (no URL charge) raised it for the
# GPU model. Tests of what happens when required evidence overflows the ceiling use fixtures sized for the old budget;
# with the new one the same fixtures fit and the overflow paths go untested, so they declare that budget (30 Oct).
_GPU_BUDGET = frozenset({"CB1", "HX1", "BA1"})


@pytest.fixture
def cpu_era_budget(monkeypatch):
    from regulus.rag import validator_switches as VS

    monkeypatch.setattr(VS, "VALIDATOR_PRECISION", VS.VALIDATOR_PRECISION - _GPU_BUDGET)
