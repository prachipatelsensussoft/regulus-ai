"""SY4 (30 Oct): the acronyms 21 CFR defines, spelled out for the section search."""
from regulus.retrieval.acronyms import expansions, table


def test_the_corpus_table_holds_the_regulations_own_expansions():
    t = table()
    assert t["RACC"] == "reference amount commonly consumed"
    assert t["GUDID"] == "global unique device identification database"
    assert "PDP" not in t and "SE" not in t   # a sense questions do not use; fewer than three letters
    assert "FDA" not in t and "CGMP" not in t  # used in hundreds of sections: not specific


def test_a_question_writing_an_acronym_gets_its_expansion():
    q = "A product is packaged in a container that holds between 200 and 300 percent of the RACC. How must the Nutrition Facts be presented?"
    assert expansions(q) == ["reference amount commonly consumed"]
    assert expansions("What must be submitted to GUDID and what is a UDI?") == [
        "global unique device identification database", "unique device identifier"]


def test_lower_case_spelled_out_and_unknown_acronyms_add_nothing():
    assert expansions("Is the ace of spades a device?") == []
    assert expansions("What is a unique device identifier (UDI)?") == []
    assert expansions("What does the XYZQ program require?") == []


def test_programme_names_the_cfr_never_defines_are_aliased_when_specific():
    # P-010: "IDE" brought 812.43 and 812.46 into the candidates
    assert expansions("What are the sponsor responsibilities under an IDE?") == ["investigational device exemption"]
    assert expansions("What hazard analysis must an importer conduct under FSVP?") == ["foreign supplier verification programs"]
    assert expansions("What is a PMTA?") == ["premarket tobacco product application"]
    assert expansions("Is a pmta required?") == []                                   # capitals only


def test_common_expansions_are_not_added():
    # SY4 calibration (30 Oct): "food and drug administration" (906 sections), "current good manufacturing practice" (507)
    # and "premarket notification" (1,158 - every device classification rule) dilute the search; F-003 lost 117.10
    assert expansions("What are the Part 117 CGMP requirements for warehousing?") == []
    assert expansions("Does FDA require it?") == []
    assert expansions("When must a 510(k) be submitted?") == []
    assert expansions("What must a PMA contain?") == []
