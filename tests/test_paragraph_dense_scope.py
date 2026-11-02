"""REGULUS_PARAGRAPH_DENSE_SCOPE=title21: the cosmetics/Act index is unchanged and the other parts are appended, each
part cached on its own (regulus.retrieval.paragraph_dense, 26 Oct 2026)."""
import numpy as np

import regulus.retrieval.paragraph_dense as PD


class _Db:
    def __init__(self, other):
        self.other = other

    def query(self, sql, params):
        if sql == PD._ROWS_SQL:
            return [{"section_number": "700.3", "part_number": 700, "paragraph_path": "(a)", "ordinal": 1,
                     "section_heading": "Definitions.", "body_text": "The term cosmetic means ..."}]
        if sql == PD._OTHER_PARTS_SQL:
            return [{"part_number": str(n)} for n in sorted({r["part_number"] for r in self.other})]
        assert sql == PD._OTHER_ROWS_SQL
        self.batches = getattr(self, "batches", 0) + 1
        return [r for r in self.other if str(r["part_number"]) in params["parts"]]


class _Embedder:
    def __init__(self):
        self.batches = []

    def describe(self):
        return {"revision": "r1"}

    def embed_batch(self, texts):
        self.batches.append(list(texts))
        return np.array([[float(len(t)), 1.0] for t in texts])

    def embed_query(self, q):
        return np.array([0.0, 1.0])


def _row(part, section, path, body):
    return {"section_number": section, "part_number": part, "paragraph_path": path, "ordinal": 1,
            "section_heading": f"Heading {section}.", "body_text": body}


OTHER = [_row(211, "211.166", "(a)", "There shall be a written testing program."),
         _row(1114, "1114.7", "(a)", "A PMTA shall contain ...")]


def test_cosmetics_scope_reads_no_other_part(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "CACHE_DIR", tmp_path)
    pd = PD.ParagraphDense.build(_Db(OTHER), None, _Embedder(), scope="cosmetics")
    assert list(pd.by_section) == ["700.3"]


def test_title21_appends_other_parts_after_an_unchanged_core(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "CACHE_DIR", tmp_path)
    core = PD.ParagraphDense.build(_Db(OTHER), None, _Embedder(), scope="cosmetics")
    emb = _Embedder()
    wide = PD.ParagraphDense.build(_Db(OTHER), None, emb, scope="title21")
    assert emb.batches == [[t] for t in wide.texts[1:]]          # core read from its cache; one batch per part
    assert np.array_equal(wide.vectors[:1], core.vectors)
    assert list(wide.by_section) == ["700.3", "211.166", "1114.7"]
    assert [p.name.split("_")[1] for p in sorted((tmp_path / "parts").glob("*.npy"))] == ["1114", "211"]


def test_a_changed_part_reembeds_only_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "CACHE_DIR", tmp_path)
    PD.ParagraphDense.build(_Db(OTHER), None, _Embedder(), scope="title21")
    emb = _Embedder()
    changed = [OTHER[0], _row(1114, "1114.7", "(a)", "A PMTA must contain ...")]
    PD.ParagraphDense.build(_Db(changed), None, emb, scope="title21")
    assert len(emb.batches) == 1 and "1114.7" in emb.batches[0][0]


def test_a_failing_extension_leaves_the_core_index(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "CACHE_DIR", tmp_path)

    class _Broken(_Db):
        def query(self, sql, params):
            if sql == PD._OTHER_ROWS_SQL:
                raise RuntimeError("statement timeout")
            return super().query(sql, params)

    pd = PD.ParagraphDense.build(_Broken(OTHER), None, _Embedder(), scope="title21")
    assert list(pd.by_section) == ["700.3"] and len(pd.texts) == len(pd.vectors) == 1


def test_other_parts_are_read_in_batches_with_the_same_result(tmp_path, monkeypatch):
    # 2 Nov: one query for every other part ran into the 15 s statement timeout under load and the index fell back
    monkeypatch.setattr(PD, "CACHE_DIR", tmp_path)
    whole = PD.ParagraphDense.build(_Db(OTHER), None, _Embedder(), scope="title21")
    monkeypatch.setattr(PD, "OTHER_PARTS_PER_QUERY", 1)
    db = _Db(OTHER)
    batched = PD.ParagraphDense.build(db, None, _Embedder(), scope="title21")
    assert db.batches == 2 and batched.texts == whole.texts and np.array_equal(batched.vectors, whole.vectors)
