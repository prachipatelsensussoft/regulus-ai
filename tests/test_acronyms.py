"""SY4 (30 Oct): the acronyms 21 CFR defines, spelled out for the section search."""
from regulus.retrieval.acronyms import expansions, table


def test_the_corpus_table_holds_the_regulations_own_expansions():
    t = table()
    assert t["RACC"] == "reference amount commonly consumed"
    assert t["GUDID"] == "global unique device identification database"
    assert "PDP" not in t and "SE" not in t   # a sense questions do not use; fewer than three letters


def test_a_question_writing_an_acronym_gets_its_expansion():
    q = "A product is packaged in a container that holds between 200 and 300 percent of the RACC. How must the Nutrition Facts be presented?"
    assert expansions(q) == ["reference amount commonly consumed"]
    assert expansions("What must be submitted to GUDID and what is a UDI?") == [
        "global unique device identification database", "unique device identifier"]


def test_lower_case_spelled_out_and_unknown_acronyms_add_nothing():
    assert expansions("Is the ace of spades a device?") == []
    assert expansions("What is a unique device identifier (UDI)?") == []
    assert expansions("What does the XYZQ program require?") == []


def test_programme_names_the_cfr_never_defines_are_aliased():
    # P-003: "510(k)" never reached 807.81 "When a premarket notification submission is required"
    assert expansions("When must a 510(k) be submitted for a change to a legally marketed device?") == ["premarket notification"]
    assert expansions("What are the PMA supplement requirements?") == ["premarket approval"]
    assert expansions("What hazard analysis must an importer conduct under FSVP?") == ["foreign supplier verification programs"]
    assert expansions("What does a premarket notification 510(k) contain?") == []   # already spelled out
    assert expansions("Is a pma required?") == []                                    # capitals only
