"""AO1 (2 Nov 2026): a list the answer restates keeps its source's "and" / "or"."""
from types import SimpleNamespace as N

from regulus.rag.conjunctions import fix_conjunctions

SOURCES = [N(citation="21 CFR 50.24(a)(2)", text="(2) Obtaining informed consent is not feasible because:"),
           N(citation="21 CFR 50.24(a)(2)(i)", text="(i) The subjects will not be able to give their informed consent as a "
                                                     "result of their medical condition;"),
           N(citation="21 CFR 50.24(a)(2)(ii)", text="(ii) The intervention under investigation must be administered before "
                                                      "consent from the subjects' legally authorized representatives is feasible; and"),
           N(citation="21 CFR 50.24(a)(2)(iii)", text="(iii) There is no reasonable way to identify prospectively the individuals "
                                                       "likely to become eligible for participation in the clinical investigation.")]
K002 = ("(2) obtaining informed consent is not feasible because (i) the subject’s medical condition prevents them from "
        "giving informed consent, (ii) the intervention must be administered before consent from a legally authorized "
        "representative is feasible, or (iii) there is no reasonable way to identify individuals likely to become eligible "
        "for participation;")


def test_k002_three_required_conditions_are_joined_by_and_again():
    out, n = fix_conjunctions(K002, SOURCES)
    assert n == 1 and out == K002.replace("is feasible, or (iii)", "is feasible, and (iii)")


def test_a_list_already_joined_as_its_source_is_unchanged():
    right = K002.replace("is feasible, or (iii)", "is feasible, and (iii)")
    assert fix_conjunctions(right, SOURCES) == (right, 0)


def test_a_list_that_does_not_match_the_source_item_by_item_is_left_alone():
    own = ("Sponsors may (i) consult the community, (ii) disclose results publicly, or (iii) request an independent "
           "data monitoring committee review;")
    assert fix_conjunctions(own, SOURCES) == (own, 0)


def test_the_lists_own_parent_label_is_not_counted_as_an_item():
    # "(2) ... because (i) ..." - the (2) is the answer's numbering, not a fourth condition
    out, n = fix_conjunctions("Under 21 CFR 50.24(a): " + K002, SOURCES)
    assert n == 1
