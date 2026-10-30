"""Restoring eCFR text the original ingestion dropped (regulus.ingestion.dropped_text, scripts/repair_dropped_source_text.py).
No database: stored rows are built here the way the original ingestion stored them (one row per <P>, nothing for
EXTRACT/FP). The part 740 payload is the real one (eCFR versioner, 2026-09-03), whose SHA-256 every stored row of part
740 records."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from regulus.ingestion import dropped_text as DT

PART_740 = (Path(__file__).parent / "fixtures" / "ecfr_part740_20260903.xml").read_bytes()
PART_740_SHA = "1b33d44258aaa462d5c328358b610ee8b37361b608827d07074ae0f5bd6838fd"
WARNING_A1 = ("Warning—Avoid spraying in eyes. Contents under pressure. Do not puncture or incinerate. "
              "Do not store at temperature above 120 °F. Keep out of reach of children.")
PROVENANCE = {"740": {"url": "https://www.ecfr.gov/api/versioner/v1/full/2026-09-03/title-21.xml?chapter=I&part=740",
                      "sha256": PART_740_SHA, "retrieved_utc": "2026-10-21T09:12:55Z"}}


def stored_as_ingested(section: str, paths: list[str]) -> list[DT.StoredRow]:
    """The defective state: one row per <P> of the real payload, in order, with the given paragraph paths."""
    paras = [e.text for e in DT.sections_of(PART_740)[section] if e.tag == "P"]
    assert len(paras) == len(paths)
    return [DT.StoredRow(f"id-{i}", p, 0, i, t) for i, (p, t) in enumerate(zip(paths, paras))]


PATHS_740_11 = ["(a)(1)", "(a)(2)", "(a)(3)", "(a)(4)", "(b)(1)", "(b)(2)", "(b)(2)(i)", "(b)(2)(ii)", "(b)(2)(iii)",
                "(b)(2)(iv)", "(c)"]


def applied(rows: list[DT.StoredRow], plan: DT.SectionPlan) -> list[DT.StoredRow]:
    """The rows as they would be after the plan: renumbered, plus the inserted rows."""
    new_ord = {i: n for i, _, n in plan.ordinal_updates}
    out = [replace(r, ordinal=new_ord.get(r.id, r.ordinal)) for r in rows]
    out += [DT.StoredRow(f"new-{k}", n.paragraph_path, n.path_ordinal, n.ordinal, n.body_text) for k, n in enumerate(plan.inserts)]
    return sorted(out, key=lambda r: r.ordinal)


def plan_740_11(targets=frozenset({"(a)(1)", "(b)(1)"}), rows=None):
    rows = rows or stored_as_ingested("740.11", PATHS_740_11)
    return rows, DT.plan_section("740.11", DT.sections_of(PART_740)["740.11"], rows, set(targets))


# -- source and provenance ----------------------------------------------------------------------------------------------

def test_the_fixture_is_the_payload_the_database_records():
    assert DT.verify_payload(PART_740, [PART_740_SHA, PART_740_SHA]) == PART_740_SHA


@pytest.mark.parametrize("shas", [["0" * 64], [PART_740_SHA, "0" * 64], []])
def test_a_payload_that_is_not_the_recorded_one_is_refused(shas):
    with pytest.raises(DT.RepairError):
        DT.verify_payload(PART_740, shas)


def test_block_text_joins_list_items_with_one_space_and_keeps_inline_markup_glued():
    import xml.etree.ElementTree as ET
    el = ET.fromstring("<EXTRACT><FP-1>Acid Black 58</FP-1><FP-1>Acid Black 107</FP-1>"
                       "<SCOL2><LI>weight wt.</LI><LI>square sq.</LI></SCOL2><FP-1>Ca<SU>5</SU>OH(PO<SU>4</SU>)<SU>3</SU></FP-1></EXTRACT>")
    assert DT.block_text(el) == "Acid Black 58 Acid Black 107 weight wt. square sq. Ca5OH(PO4)3"


def test_block_text_keeps_inline_runs_glued_beside_a_block_sibling():
    import xml.etree.ElementTree as ET
    el = ET.fromstring('<EXTRACT>Ca<SU>5</SU>OH (<E T="03">see</E>s) <P>x</P>tail <I>y</I>z</EXTRACT>')
    assert DT.block_text(el) == "Ca5OH (sees) x tail yz"


def test_content_hash_follows_the_corpus_fold():
    folded = WARNING_A1.replace("—", "-")
    assert DT.content_hash(WARNING_A1) == hashlib.sha256(folded.encode()).hexdigest()
    assert DT.content_hash("“a” ‘b’ – c − d §§ 1 µ Ω") == hashlib.sha256('"a" \'b\' - c - d § 1 μ Ω'.encode()).hexdigest()
    assert DT.content_hash("plain ascii") == hashlib.sha256(b"plain ascii").hexdigest()


# -- planning -----------------------------------------------------------------------------------------------------------

def test_the_740_11_warnings_are_planned_after_their_lead_ins_in_document_order():
    rows, plan = plan_740_11()
    assert plan.status == "planned"
    assert [(r.paragraph_path, r.path_ordinal, r.ordinal, r.source_tag, r.anchor_id) for r in plan.inserts] == [
        ("(a)(1)", 1, 1, "EXTRACT", "id-0"), ("(b)(1)", 1, 6, "EXTRACT", "id-4")]
    assert plan.inserts[0].body_text == WARNING_A1
    assert plan.inserts[0].content_sha256 == DT.content_hash(WARNING_A1)
    after = applied(rows, plan)
    assert [r.ordinal for r in after] == list(range(13))                       # contiguous, no duplicates
    assert [r.body_text for r in after][:2] == [rows[0].body_text, WARNING_A1]  # the warning follows its lead-in
    assert len({(r.paragraph_path, r.path_ordinal) for r in after}) == len(after)


def test_a_flush_paragraph_after_the_extract_belongs_to_the_same_anchor():
    rows = stored_as_ingested("740.12", ["(a)", "(b)", "(c)"])
    plan = DT.plan_section("740.12", DT.sections_of(PART_740)["740.12"], rows, {"(b)"})
    assert [(r.paragraph_path, r.path_ordinal, r.ordinal, r.source_tag) for r in plan.inserts] == [
        ("(b)", 1, 2, "EXTRACT"), ("(b)", 2, 3, "FP")]
    assert plan.inserts[0].body_text.startswith("Caution—For external use only.")
    assert plan.ordinal_updates == [("id-2", 2, 4)]


def test_the_plan_is_empty_once_applied_so_a_rerun_changes_nothing():
    rows, first = plan_740_11()
    elements = DT.sections_of(PART_740)["740.11"]
    second = DT.plan_section("740.11", elements, applied(rows, first), {"(a)(1)", "(b)(1)"})
    assert second.status == "no_change" and not second.inserts and not second.ordinal_updates
    assert second.already_restored == 2


def test_dropped_elements_after_other_anchors_are_reported_not_planned():
    _, plan = plan_740_11({"(a)(1)"})
    assert [r.paragraph_path for r in plan.inserts] == ["(a)(1)"]
    assert [(d["anchor"], d["tag"], d["reason"]) for d in plan.untargeted_dropped] == [("(b)(1)", "EXTRACT", "not targeted")]


def test_a_restored_row_that_would_number_before_a_later_row_of_its_path_is_refused():
    rows = stored_as_ingested("740.11", PATHS_740_11)
    rows = [replace(r, paragraph_path="(a)(1)", path_ordinal=1) if r.id == "id-1" else r for r in rows]  # (a)(1)#1 later
    _, plan = plan_740_11({"(a)(1)"}, rows)
    assert plan.status == "misaligned" and "invert path_ordinal order" in plan.reason and not plan.inserts


def test_a_paragraph_stored_split_over_two_rows_still_aligns():
    rows = stored_as_ingested("740.11", PATHS_740_11)
    head, tail = rows[0].body_text.split(" packaged in ", 1)
    split = [replace(rows[0], body_text=head), DT.StoredRow("id-0b", "(a)(1)", 1, 1, "packaged in " + tail)]
    rows = split + [replace(r, ordinal=r.ordinal + 1) for r in rows[1:]]
    _, plan = plan_740_11({"(a)(1)"}, rows)
    assert plan.status == "planned"
    assert [(r.paragraph_path, r.path_ordinal, r.ordinal, r.anchor_id) for r in plan.inserts] == [("(a)(1)", 2, 2, "id-0b")]


def test_stored_text_the_source_does_not_contain_refuses_the_section():
    rows = stored_as_ingested("740.11", PATHS_740_11)
    rows[3] = replace(rows[3], body_text=rows[3].body_text + " (edited)")
    _, plan = plan_740_11({"(a)(1)"}, rows)
    assert plan.status == "misaligned" and not plan.inserts and not plan.ordinal_updates


def test_a_dropped_element_inside_a_stored_row_is_refused():
    rows = stored_as_ingested("740.11", PATHS_740_11)
    merged = replace(rows[0], body_text=rows[0].body_text + " " + rows[1].body_text)  # (a)(1)+(a)(2) in one row
    rows = [merged] + [replace(r, ordinal=r.ordinal - 1) for r in rows[2:]]
    _, plan = plan_740_11({"(a)(1)"}, rows)
    assert plan.status == "misaligned" and "inside a stored row" in plan.reason


def test_the_digest_binds_the_provenance_written_with_the_rows():
    _, plan = plan_740_11()
    other = {"740": {**PROVENANCE["740"], "retrieved_utc": "1999-01-01T00:00:00Z"}}
    assert DT.plan_digest([plan], PROVENANCE) != DT.plan_digest([plan], other)


# -- applying -----------------------------------------------------------------------------------------------------------

class FakeConn:
    """Records statements; answers the precondition SELECT from `rows`; `rowcount` and `read_only` are settable."""

    def __init__(self, rows, rowcount=1, read_only=False, tx_read_only="off"):
        self.rows, self.statements, self.events = rows, [], []
        self.rowcount, self.read_only, self.tx_read_only = rowcount, read_only, tx_read_only

    @contextlib.contextmanager
    def transaction(self):
        self.events.append("begin")
        try:
            yield
        except BaseException:
            self.events.append("rollback")
            raise
        self.events.append("commit")

    def execute(self, sql, params=None):
        text = " ".join(sql.split())
        self.statements.append((text[:40], params))
        conn = self

        class Result:
            rowcount = conn.rowcount

            def fetchall(self):
                if "transaction_read_only" in text:
                    return [{"ro": conn.tx_read_only}]
                if text.startswith("SELECT c.conrelid"):
                    return [{"tbl": "bookmarks", "col": "record_id"}]
                if text.startswith("SELECT count(*)"):
                    return [{"n": 0}]
                if text.startswith("SELECT"):
                    return [{"id": r.id, "paragraph_path": r.paragraph_path, "path_ordinal": r.path_ordinal, "ordinal": r.ordinal,
                             "body_text": r.body_text} for r in conn.rows]
                return [{"id": f"new-{len(conn.statements)}"}]
        return Result()

    def close(self):
        pass


def _apply(conn, plan, authorized=None, writer=None):
    written = []
    out = DT.apply_plans(conn, [plan], PROVENANCE, authorized or DT.plan_digest([plan], PROVENANCE),
                         writer or (lambda m: (written.append(list(conn.events)), conn.events.append("manifest"))))
    return out, written


def test_apply_refuses_a_plan_that_was_not_authorised():
    rows, plan = plan_740_11()
    conn = FakeConn(rows)
    with pytest.raises(DT.RepairError):
        _apply(conn, plan, authorized="0" * 64)
    assert conn.statements == []


@pytest.mark.parametrize("kwargs", [{"read_only": True}, {"tx_read_only": "on"}])
def test_apply_refuses_a_read_only_connection_instead_of_overriding_it(kwargs):
    rows, plan = plan_740_11()
    conn = FakeConn(rows, **kwargs)
    with pytest.raises(DT.RepairError, match="read-only"):
        _apply(conn, plan)
    assert not any(s.startswith(("UPDATE", "INSERT")) for s, _ in conn.statements)
    assert conn.events[-1] == "rollback"


def test_apply_refuses_when_the_stored_rows_changed_after_planning():
    rows, plan = plan_740_11()
    conn = FakeConn([replace(r, body_text="x") if r.id == "id-5" else r for r in rows])
    with pytest.raises(DT.RepairError, match="changed since the plan"):
        _apply(conn, plan)
    assert not any(s.startswith(("UPDATE", "INSERT")) for s, _ in conn.statements) and conn.events[-1] == "rollback"


def test_an_update_that_touches_no_row_aborts_the_transaction():
    rows, plan = plan_740_11()
    conn = FakeConn(rows, rowcount=0)
    with pytest.raises(DT.RepairError, match="touched 0 rows"):
        _apply(conn, plan)
    assert conn.events == ["begin", "rollback"]


def test_a_manifest_that_cannot_be_written_aborts_the_transaction():
    rows, plan = plan_740_11()
    conn = FakeConn(rows)

    def fail(_):
        raise OSError("disk full")
    with pytest.raises(OSError):
        _apply(conn, plan, writer=fail)
    assert conn.events == ["begin", "rollback"]


def test_apply_renumbers_then_inserts_writes_the_manifest_before_commit_and_rolls_back():
    rows, plan = plan_740_11()
    conn = FakeConn(rows)
    manifest, written = _apply(conn, plan)
    assert written == [["begin"]] and conn.events == ["begin", "manifest", "commit"]
    kinds = [s.split()[0] for s, _ in conn.statements]
    assert kinds.count("UPDATE") == len(plan.ordinal_updates) and kinds.count("INSERT") == 2
    updates = [p["old"] for s, p in conn.statements if s.startswith("UPDATE")]
    assert updates == sorted(updates, reverse=True)
    assert all(p["retrieved_at"] == "2026-10-21T09:12:55Z" for s, p in conn.statements if s.startswith("INSERT"))
    assert [m["paragraph_path"] for m in manifest["inserted"]] == ["(a)(1)", "(b)(1)"]
    back = FakeConn(rows)
    assert DT.rollback(back, manifest) == {"deleted": 2, "ordinals_restored": len(plan.ordinal_updates)}
    assert any(s.startswith("SELECT count(*)") for s, _ in back.statements)   # references checked before deleting


def test_rollback_refuses_when_another_table_references_a_restored_row():
    rows, plan = plan_740_11()
    manifest, _ = _apply(FakeConn(rows), plan)
    back = FakeConn(rows)
    original = back.execute

    def execute(sql, params=None):
        res = original(sql, params)
        if sql.startswith("SELECT count(*)"):
            res.fetchall = lambda: [{"n": 1}]
        return res
    back.execute = execute
    with pytest.raises(DT.RepairError, match="reference restored rows"):
        DT.rollback(back, manifest)
    assert not any(s.startswith("DELETE") for s, _ in back.statements)


# -- command line -------------------------------------------------------------------------------------------------------

def _cli():
    spec = importlib.util.spec_from_file_location("repair_cli", Path(__file__).parents[1] / "scripts" / "repair_dropped_source_text.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture_sources(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "part-740.xml").write_bytes(PART_740)
    (src / "log.json").write_text(json.dumps([{"part": "740", "verified": True, **PROVENANCE["740"]}]))
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps([{"section": "740.11", "paragraph_path": "(a)(1)"}]))
    rows = stored_as_ingested("740.11", PATHS_740_11)

    def query(sql, params=None):
        if "DISTINCT source_payload_sha256" in sql:
            return [{"source_payload_sha256": PART_740_SHA}]
        return [{"id": r.id, "section_number": "740.11", "paragraph_path": r.paragraph_path, "path_ordinal": r.path_ordinal,
                 "ordinal": r.ordinal, "body_text": r.body_text, "source_payload_sha256": PART_740_SHA} for r in rows]

    class Q:
        def __enter__(self):
            return query

        def __exit__(self, *exc):
            pass
    return src, targets, rows, Q


def test_cli_plan_then_apply_with_the_authorised_digest(tmp_path):
    cli = _cli()
    src, targets, rows, Q = _fixture_sources(tmp_path)
    with Q() as q:
        _, _, content = cli.make_plan(q, src, targets)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(content))
    conn = FakeConn(rows)
    manifest = cli.cmd_apply(plan_file, content["plan_sha256"], tmp_path / "manifest.json", Q, lambda: conn)
    assert len(manifest["inserted"]) == 1 and conn.events == ["begin", "commit"]
    assert json.loads((tmp_path / "manifest.json").read_text())["plan_sha256"] == content["plan_sha256"]


def test_cli_apply_refuses_before_any_database_work_when_the_manifest_cannot_be_created(tmp_path):
    cli = _cli()
    src, targets, rows, Q = _fixture_sources(tmp_path)
    with Q() as q:
        _, _, content = cli.make_plan(q, src, targets)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(content))
    touched = []
    with pytest.raises(FileNotFoundError):
        cli.cmd_apply(plan_file, content["plan_sha256"], tmp_path / "missing" / "m.json", Q, lambda: touched.append(1))
    existing = tmp_path / "m.json"
    existing.write_text("{}")
    with pytest.raises(FileExistsError):
        cli.cmd_apply(plan_file, content["plan_sha256"], existing, Q, lambda: touched.append(1))
    assert touched == [] and existing.read_text() == "{}"


def test_cli_apply_refuses_a_digest_that_is_not_the_recomputed_plan(tmp_path):
    cli = _cli()
    src, targets, rows, Q = _fixture_sources(tmp_path)
    with Q() as q:
        _, _, content = cli.make_plan(q, src, targets)
    content["plan_sha256"] = "f" * 64                      # an edited plan file, authorised with its own digest
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(content))
    touched = []
    with pytest.raises(DT.RepairError, match="not one and the same plan"):
        cli.cmd_apply(plan_file, "f" * 64, tmp_path / "manifest.json", Q, lambda: touched.append(1))
    assert touched == [] and not (tmp_path / "manifest.json").exists()
    assert (tmp_path / "manifest.not_applied.json").exists()


# Tables (22 Oct 2026): the eCFR renderer emits HTML <TABLE> inside <DIV>s; 701.30's label names and 81.1's lake lists
# were dropped with them.
TABLE_SECTION = b"""<ROOT><DIV8 N="701.30" TYPE="SECTION"><HEAD>Sec. 701.30 Names.</HEAD>
<P>The Commissioner establishes the following names:</P>
<DIV width="100%"><DIV class="gpotbl_div"><TABLE><THEAD>
<TR><TH rowspan="2">Chemical name</TH><TH colspan="2">Label</TH></TR>
<TR><TH>Formula</TH><TH>Established name</TH></TR></THEAD>
<TBODY><TR><TD>Dichlorodifluoromethane</TD><TD>CCl<sub>2</sub> F<sub>2</sub></TD><TD>Chlorofluorocarbon 12.</TD></TR>
<TR><TD>Propane</TD><TD /><TD>Propane</TD></TR></TBODY></TABLE></DIV></DIV>
<CITA TYPE="N">[42 FR 24255, May 13, 1977]</CITA></DIV8></ROOT>"""


def test_an_html_table_is_one_restorable_element_with_every_cell_under_its_column_heading():
    els = DT.sections_of(TABLE_SECTION)["701.30"]
    assert [e.tag for e in els] == ["P", "TABLE"]   # the citation is not section text
    assert els[1].text == ("Chemical name: Dichlorodifluoromethane; Label, Formula: CCl2 F2; Label, Established name: "
                           "Chlorofluorocarbon 12. Chemical name: Propane; Label, Established name: Propane.")


def test_a_dropped_table_is_planned_after_the_paragraph_it_follows():
    els = DT.sections_of(TABLE_SECTION)["701.30"]
    rows = [DT.StoredRow("id-0", "", 0, 0, els[0].text)]
    plan = DT.plan_section("701.30", els, rows)
    assert plan.status == "planned"
    assert [(r.paragraph_path, r.path_ordinal, r.ordinal, r.source_tag) for r in plan.inserts] == [("", 1, 1, "TABLE")]
    assert DT.plan_section("701.30", els, applied(rows, plan)).inserts == []


# -- 30 Oct: split-at-designator rows align; large tables are stored in parts ------------------------------------------

SPLIT_SECTION = b"""<ROOT><DIV8 N="1114.7" TYPE="SECTION"><HEAD>Required content and format.</HEAD>
<P>(f) Labeling and description of marketing plans&#x2014;(1) Labeling. The application must contain specimens.</P>
<EXTRACT><P>Specimen text.</P></EXTRACT>
<P>(2) Marketing plans. The application must describe them.</P></DIV8></ROOT>"""


def test_a_paragraph_the_ingestion_split_at_an_inline_designator_aligns():
    # 1114.7 (23 Oct): stored as "...marketing plans—" + "(1) Labeling. ..." - the joined stream reads "plans— (1)"
    els = DT.sections_of(SPLIT_SECTION)["1114.7"]
    head, tail = els[0].text.split("(1) Labeling", 1)
    rows = [DT.StoredRow("a", "(f)", 0, 0, head), DT.StoredRow("b", "(f)(1)", 0, 1, "(1) Labeling" + tail),
            DT.StoredRow("c", "(f)(2)", 0, 2, els[2].text)]
    plan = DT.plan_section("1114.7", els, rows)
    assert plan.status == "planned" and [(n.paragraph_path, n.source_tag, n.anchor_id) for n in plan.inserts] == [("(f)(1)", "EXTRACT", "b")]


def _big_table(rows: int) -> bytes:
    trs = "".join(f"<TR><TD>Category {i} of the general food supply</TD><TD>{i} g</TD><TD>{i} oz</TD></TR>" for i in range(rows))
    return (b"""<ROOT><DIV8 N="101.12" TYPE="SECTION"><HEAD>Reference amounts.</HEAD><P>(b) The following tables:</P>
<DIV><DIV class="gpotbl_div"><TABLE><CAPTION><P>Table 2&#x2014;Reference Amounts Customarily Consumed: General Food Supply</P></CAPTION>
<THEAD><TR><TH>Product category</TH><TH>Reference amount</TH><TH>Label statement</TH></TR></THEAD><TBODY>"""
            + trs.encode() + b"</TBODY></TABLE></DIV></DIV></DIV8></ROOT>")


def test_a_large_table_is_stored_in_parts_that_name_their_table():
    els = DT.sections_of(_big_table(120))["101.12"]
    parts = [e for e in els if e.tag == "TABLE"]
    assert len(parts) > 1 and all(len(e.text) <= DT.TABLE_PART_CHARS for e in parts)
    assert all(e.text.startswith(f"Table 2—Reference Amounts Customarily Consumed: General Food Supply (part {k} of {len(parts)}): ")
               for k, e in enumerate(parts, 1))
    rows = [r for e in parts for r in e.text.split("): ", 1)[1].split(". Product category")]
    assert len(rows) == 120                                   # no row split across parts
    small = [e for e in DT.sections_of(_big_table(3))["101.12"] if e.tag == "TABLE"]
    assert len(small) == 1 and not small[0].text.startswith("Table 2")   # a small table: as before, byte for byte


def test_the_parts_are_planned_in_order_and_a_rerun_after_apply_changes_nothing():
    els = DT.sections_of(_big_table(120))["101.12"]
    rows = [DT.StoredRow("id-0", "(b)", 0, 0, els[0].text)]
    plan = DT.plan_section("101.12", els, rows)
    parts = [e for e in els if e.tag == "TABLE"]
    assert plan.status == "planned" and [n.body_text for n in plan.inserts] == [e.text for e in parts]
    assert [n.path_ordinal for n in plan.inserts] == list(range(1, len(parts) + 1))
    assert DT.plan_section("101.12", els, applied(rows, plan)).inserts == []
