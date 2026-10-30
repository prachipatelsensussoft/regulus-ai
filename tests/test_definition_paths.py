"""DEF1 (30 Oct): definitions sections restart their paragraph numbering at each definition."""
from regulus.ingestion.definition_paths import Row, definition_leads, plan_section, repair_sql

XML = b"""<ECFR><DIV8 N="1100.3" TYPE="SECTION"><HEAD>Definitions.</HEAD>
<P>For the purposes of this part:</P>
<P><I>Accessory</I> means any product that meets either of the following:</P>
<P>(1) Is not intended to affect a tobacco product; or</P>
<P>(2) Is intended to affect it but</P>
<P>(i) Solely controls moisture; or</P>
<P>(ii) Solely provides an external heat source.</P>
<P><I>Component</I> or <I>part</I> means any software intended or reasonably expected:</P>
<P>(1) To alter the tobacco product; or</P>
<P>(2) To be used with a tobacco product.</P>
<P><I>Tobacco product,</I> as stated in section 201(rr):</P>
<P>(1) Means any product made or derived from tobacco; and</P>
</DIV8><DIV8 N="1100.2" TYPE="SECTION"><P>No definitions here.</P></DIV8></ECFR>"""

STORED = [  # as the original ingestion stored 1100.3 (the stack carried across definitions)
    ("", "For the purposes of this part:"), ("", "Accessory means any product that meets either of the following:"),
    ("(1)", "(1) Is not intended to affect a tobacco product; or"), ("(2)", "(2) Is intended to affect it but"),
    ("(2)(i)", "(i) Solely controls moisture; or"), ("(2)(ii)", "(ii) Solely provides an external heat source."),
    ("(2)(ii)", "Component or part means any software intended or reasonably expected:"),
    ("(2)(ii)(1)", "(1) To alter the tobacco product; or"), ("(2)(ii)(2)", "(2) To be used with a tobacco product."),
    ("(2)(ii)(2)", "Tobacco product, as stated in section 201(rr):"),
    ("(2)(ii)(2)(1)", "(1) Means any product made or derived from tobacco; and"),
]


def rows():
    seen, out = {}, []
    for i, (path, text) in enumerate(STORED):
        po = seen.get(path, 0)
        seen[path] = po + 1
        out.append(Row(f"id{i}", i, path, po, text))
    return out


def test_leads_are_the_italic_undesignated_paragraphs():
    leads = definition_leads(XML)
    assert list(leads) == ["1100.3"] and [t.split()[0] for t in leads["1100.3"]] == ["Accessory", "Component", "Tobacco"]


def test_numbering_restarts_at_each_definition():
    plan = plan_section("1100.3", definition_leads(XML)["1100.3"], rows())
    final = {rid: (new, po) for rid, _, _, new, po in plan.updates}
    assert plan.status == "planned"
    assert final["id6"] == ("", 2)            # "Component or part" lead: undesignated text of the section
    assert final["id7"] == ("(1)", 1) and final["id8"] == ("(2)", 1)
    assert final["id9"] == ("", 3) and final["id10"] == ("(1)", 2)   # "Tobacco product (1)", not (2)(ii)(2)(1)
    assert "id2" not in final and "id4" not in final                  # Accessory's own items were right


def test_a_section_whose_leads_are_not_all_found_is_left_alone():
    broken = [r for r in rows() if not r.body_text.startswith("Component")]
    assert plan_section("1100.3", definition_leads(XML)["1100.3"], broken).status == "misaligned"


def test_a_letter_after_h_is_a_letter_not_a_roman_numeral():
    xml = b"""<ECFR><DIV8 N="9.3" TYPE="SECTION"><HEAD>Definitions.</HEAD><P><I>Term</I> means:</P><P>(h) eight</P><P>(i) nine</P></DIV8></ECFR>"""
    stored = [Row("a", 0, "", 0, "Term means:"), Row("b", 1, "(x)", 0, "(h) eight"), Row("c", 2, "(x)(i)", 0, "(i) nine")]
    final = {rid: new for rid, _, _, new, _ in plan_section("9.3", definition_leads(xml)["9.3"], stored).updates}
    assert final == {"b": "(h)", "c": "(i)"}


def test_the_sql_moves_through_temporary_paths_and_refuses_changed_rows():
    plan = plan_section("1100.3", definition_leads(XML)["1100.3"], rows())
    fwd, back = repair_sql([plan])
    assert fwd.count("BEGIN;") == 1 and fwd.rstrip().endswith("COMMIT;") and "(ZZZZ)(0)" in fwd
    assert "IF NOT FOUND THEN RAISE EXCEPTION" in fwd and "IF NOT FOUND THEN RAISE EXCEPTION" in back
    assert fwd.count("UPDATE cfr_records") == 2 * len(plan.updates) == back.count("UPDATE cfr_records")


def test_definitions_listed_inside_a_lettered_paragraph_restart_at_that_paragraph():
    # 1.600: "(c) For the purposes of this subpart:" then the definitions, which live in (c)
    xml = b"""<ECFR><DIV8 N="1.600" TYPE="SECTION"><P>(c) For the purposes of this subpart:</P>
    <P><I>Assessment</I> means:</P><P>(i) With respect to a body, an evaluation;</P><P>(ii) With respect to another, a review.</P>
    <P><I>Audit</I> means the examination of an entity.</P>
    <P><I>Consultative audit</I> means an audit:</P><P>(i) To determine compliance;</P></DIV8></ECFR>"""
    stored = [Row("r0", 0, "(c)", 0, "(c) For the purposes of this subpart:"), Row("r1", 1, "(c)", 1, "Assessment means:"),
              Row("r2", 2, "(c)(i)", 0, "(i) With respect to a body, an evaluation;"),
              Row("r3", 3, "(c)(ii)", 0, "(ii) With respect to another, a review."),
              Row("r4", 4, "(c)(ii)", 1, "Audit means the examination of an entity."),
              Row("r5", 5, "(c)(ii)", 2, "Consultative audit means an audit:"),
              Row("r6", 6, "(c)(ii)(i)", 0, "(i) To determine compliance;")]
    final = {rid: (new, po) for rid, _, _, new, po in plan_section("1.600", definition_leads(xml)["1.600"], stored).updates}
    assert final == {"r4": ("(c)", 2), "r5": ("(c)", 3), "r6": ("(c)(i)", 1)}   # (c) kept; Assessment's rows untouched


def test_the_italic_fifth_level_nests_and_the_next_top_item_does_not():
    # 1.227 "Farm": (1) ... (iii) ... (B) then italic (1)-(3) under (B), then "(2) Secondary activities farm"
    xml = b"""<ECFR><DIV8 N="1.227" TYPE="SECTION"><HEAD>What definitions apply to this subpart?</HEAD><P><I>Farm</I> means:</P><P>(1) Primary production farm.</P>
    <P>(iii) Manufacture food, provided that:</P><P>(A) All food is consumed;</P><P>(B) Any processing is limited to:</P>
    <P>(1) Drying;</P><P>(2) Treating;</P><P>(3) Packaging;</P><P>(2) Secondary activities farm.</P>
    <P><I>Food</I> has the meaning given:</P><P>(1) Except for this subpart, it does not include:</P></DIV8></ECFR>"""
    texts = ["Farm means:", "(1) Primary production farm.", "(iii) Manufacture food, provided that:", "(A) All food is consumed;",
             "(B) Any processing is limited to:", "(1) Drying;", "(2) Treating;", "(3) Packaging;", "(2) Secondary activities farm.",
             "Food has the meaning given:", "(1) Except for this subpart, it does not include:"]
    stored = [Row(f"r{i}", i, "(9)", i, t) for i, t in enumerate(texts)]   # all wrong on purpose
    final = {rid: new for rid, _, _, new, _ in plan_section("1.227", definition_leads(xml)["1.227"], [Row("r0", 0, "", 0, texts[0])] + stored[1:]).updates}
    assert final["r5"] == "(1)(iii)(B)(1)" and final["r7"] == "(1)(iii)(B)(3)"
    assert final["r8"] == "(2)" and final["r9"] == "" and final["r10"] == "(1)"


def test_italic_headings_that_define_nothing_are_not_definitions():
    # 201.327: italic test-procedure headings, not definitions
    xml = b"""<ECFR><DIV8 N="201.327" TYPE="SECTION"><HEAD>Over-the-counter sunscreen drug products.</HEAD>
    <P><I>UV Spectrometry.</I> Use plates.</P><P><I>Sample holder.</I> A holder.</P><P><I>Light source.</I> A lamp.</P></DIV8>
    <DIV8 N="1.3" TYPE="SECTION"><HEAD>Definitions.</HEAD><P><I>Label</I> includes all written matter.</P></DIV8></ECFR>"""
    assert list(definition_leads(xml)) == ["1.3"]   # a definitions heading: one lead is enough


def test_definitions_deep_in_a_lettered_structure_are_left_for_review():
    xml = b"""<ECFR><DIV8 N="314.81" TYPE="SECTION"><HEAD>Other postmarketing reports.</HEAD>
    <P><I>Drug shortage</I> means a period.</P><P><I>Life supporting</I> means essential.</P><P><I>Meaningful</I> means a disruption.</P></DIV8></ECFR>"""
    stored = [Row("a", 0, "(b)(3)(iii)", 0, "Drug shortage means a period."), Row("b", 1, "(b)(3)(iii)", 1, "Life supporting means essential."),
              Row("c", 2, "(b)(3)(iii)", 2, "Meaningful means a disruption.")]
    assert plan_section("314.81", definition_leads(xml)["314.81"], stored).status == "deep"


def test_a_deep_base_path_is_read_literally_and_left_for_review():
    # 314.81: definitions in (b)(3)(iii)(f) - re-nesting the stored path collapsed it to "(f)" and moved (b)(3)(iv) under it
    xml = b"""<ECFR><DIV8 N="314.81" TYPE="SECTION"><HEAD>Other postmarketing reports.</HEAD>
    <P><I>Drug shortage</I> means a period.</P><P><I>Life supporting</I> means essential.</P><P><I>Meaningful</I> means a change.</P></DIV8></ECFR>"""
    stored = [Row("a", 0, "(b)(3)(iii)(f)", 0, "Drug shortage means a period."), Row("b", 1, "(b)(3)(iii)(f)", 1, "Life supporting means essential."),
              Row("c", 2, "(b)(3)(iii)(f)", 2, "Meaningful means a change."), Row("d", 3, "(b)(3)(iv)", 0, "(iv) Withdrawal.")]
    plan = plan_section("314.81", definition_leads(xml)["314.81"], stored)
    assert plan.status == "deep" and not plan.updates
