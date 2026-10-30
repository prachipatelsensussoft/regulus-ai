"""All SQL used by the lexical retriever. Every statement is a SELECT with
named server-side parameters (%(name)s). User input is never interpolated.

Index usage (see docs/database_architecture.md):
  cfr_records_search_idx           GIN  (search_vector)              -> ECFR_FULL_TEXT
  cfr_records_section_idx          btree(section_number, ordinal)    -> ECFR_EXACT_SECTION, ECFR_EXACT_ROW
  cfr_records_part_idx             btree(part_number, section, ord)  -> ECFR_EXACT_PART, part filter
  cfr_sections_number_trgm_idx     GIN  (number gin_trgm_ops)        -> ECFR_TRIGRAM
  cfr_sections_heading_trgm_idx    GIN  (heading gin_trgm_ops)       -> ECFR_TRIGRAM
  cfr_records_search_idx           GIN  (search_vector), one lexeme  -> ECFR_SECTION_COVERAGE
  fda_guidance_chunks_search_idx   GIN  (search_vector)              -> GUIDANCE_FULL_TEXT
  fda_guidance_docket_idx          btree(docket_number)              -> GUIDANCE_EXACT_DOCKET
  fda_guidance_sector_idx / _retrieval_idx                            -> guidance filters
"""

_ECFR_COLUMNS = """
    r.title, r.chapter, r.subchapter_code, r.part_number, r.section_number,
    r.paragraph_path, r.path_ordinal, r.ordinal, r.section_heading, r.parent_headings,
    r.body_text, r.ecfr_permalink, r.govinfo_pdf_url, r.snapshot_date,
    r.effective_date, r.last_amended_date, r.is_reserved,
    p.title AS part_title
"""

_ECFR_FILTERS = """
    AND r.removed_on IS NULL
    AND (%(part)s::text IS NULL OR r.part_number = %(part)s)
    AND (%(section)s::text IS NULL OR r.section_number = %(section)s)
    AND (%(subchapter)s::text IS NULL OR r.subchapter_code = %(subchapter)s)
"""

# Exact lookup, section or paragraph level. paragraph = '' returns the whole
# section; otherwise the paragraph itself plus its descendants ("(b)" -> (b), (b)(1), ...).
ECFR_EXACT_SECTION = f"""
SELECT {_ECFR_COLUMNS},
       CASE WHEN %(paragraph)s = '' OR r.paragraph_path = %(paragraph)s THEN 'exact' ELSE 'descendant' END AS match_kind
FROM cfr_records r
LEFT JOIN cfr_parts p ON p.number = r.part_number
WHERE r.section_number = %(section_number)s
  AND (%(paragraph)s = '' OR r.paragraph_path = %(paragraph)s OR r.paragraph_path LIKE %(paragraph_prefix)s)
  {_ECFR_FILTERS}
ORDER BY r.ordinal
LIMIT %(limit)s
"""

# Stage 2: exactly one row, the requested paragraph itself (no descendants).
# Used to fetch the parent paragraph of a requested provision as structural
# evidence without dragging in the parent's other children.
ECFR_EXACT_ROW = f"""
SELECT {_ECFR_COLUMNS}, 'exact' AS match_kind
FROM cfr_records r
LEFT JOIN cfr_parts p ON p.number = r.part_number
WHERE r.section_number = %(section_number)s
  AND COALESCE(r.paragraph_path, '') = %(paragraph)s
  {_ECFR_FILTERS}
ORDER BY r.ordinal
LIMIT %(limit)s
"""

# Part-level lookup: the first live row of each section in the part, most
# relevant to the question first.
#
# Every stem of a part is equally "exact" - the request named the part, not a
# section - so an identifier score cannot order them and ASCII section order
# decided which ones a bounded context kept (21 CFR 111.3, the definitions
# section, sorted behind 111.103/111.105/... and was dropped). `rank` is the
# question's lexical relevance from the same search_vector and the same
# ts_rank_cd normalisation ECFR_FULL_TEXT uses, so heading matches (weight A)
# outrank body matches (weight B).
#
# There is deliberately no `search_vector @@ q` filter here: this is an exact
# identifier lookup, not a search. Every live stem of the part is still
# returned; relevance only orders them. websearch_to_tsquery('english', '')
# is an empty tsquery and ts_rank_cd then returns 0 for every row, so an empty
# %(query)s reproduces the previous plain `ORDER BY r.section_number` exactly
# (verified against Parts 111, 117, 211, 803, 820 and 1271).
ECFR_EXACT_PART = f"""
SELECT {_ECFR_COLUMNS}, 'exact' AS match_kind,
       ts_rank_cd(r.search_vector, q, 1) AS rank
FROM cfr_records r
LEFT JOIN cfr_parts p ON p.number = r.part_number,
     websearch_to_tsquery('english', %(query)s) q
WHERE r.part_number = %(part_number)s
  AND r.ordinal = 0
  {_ECFR_FILTERS}
ORDER BY rank DESC, r.section_number
LIMIT %(limit)s
"""

# Every section of one part, in the corpus's own document order. Used by Stage 1
# to fill in a cross-section range ("50.50 through 50.56"): the question names
# only the endpoints, and which sections sit between them is a fact about the
# corpus, not about the identifier. `sort_order` is the authoritative order
# (unique and non-null on cfr_sections, and it agrees with numeric section order
# in every part of this corpus, which ASCII order does not: 50.3 < 50.20).
# Reserved sections are returned: they exist, and the existing selection and
# consolidation rules already decide what a "[Reserved]" row is worth.
ECFR_PART_SECTIONS = """
SELECT s.number, s.sort_order
FROM cfr_sections s
WHERE s.part_number = %(part_number)s
ORDER BY s.sort_order
"""

# Full-text search over the generated search_vector (heading weight A, body B).
# websearch_to_tsquery accepts free text, quoted phrases and -exclusions.
# ts_rank_cd normalisation 1 divides by 1 + log(document length) so short eCFR
# paragraphs and long guidance chunks are comparable (measured, see docs).
ECFR_FULL_TEXT = f"""
SELECT {_ECFR_COLUMNS}, ts_rank_cd(r.search_vector, q, 1) AS rank
FROM cfr_records r
LEFT JOIN cfr_parts p ON p.number = r.part_number,
     websearch_to_tsquery('english', %(query)s) q
WHERE r.search_vector @@ q
  AND NOT r.is_reserved
  {_ECFR_FILTERS}
ORDER BY rank DESC, r.section_number, r.ordinal
LIMIT %(limit)s
"""

# Trigram search on section numbers and headings (both GIN-indexed on
# cfr_sections). The % operator applies pg_trgm.similarity_threshold (0.3).
# Returns the first live row of each matching section so the result is a
# citable passage, never the whole 218-char-average body scanned by trigram.
ECFR_TRIGRAM = f"""
SELECT {_ECFR_COLUMNS},
       GREATEST(similarity(s.number, %(query)s), similarity(s.heading, %(query)s)) AS sim
FROM cfr_sections s
JOIN LATERAL (
    SELECT * FROM cfr_records rr
    WHERE rr.section_number = s.number AND rr.removed_on IS NULL
    ORDER BY rr.ordinal LIMIT 1
) r ON TRUE
LEFT JOIN cfr_parts p ON p.number = r.part_number
WHERE (s.number %% %(query)s OR s.heading %% %(query)s)
  AND NOT s.is_reserved
  {_ECFR_FILTERS}
ORDER BY sim DESC, s.number
LIMIT %(limit)s
"""

# Section-coverage search for a question that names no provision (RC1). Neither text
# statement above suits it: AND (ECFR_FULL_TEXT) requires every question word in ONE short
# paragraph, so eCFR rarely matches at all, and OR over a few long words ranks thousands of
# one-word matches by density. Here every content lexeme of the question counts, weighted by
# its section-level inverse document frequency (computed from the same matches, so common
# words such as "record" weigh little but are not dropped). A section scores the weight it
# covers across ALL its live paragraphs plus the weight its single best paragraph covers, both
# as fractions of the question's total weight: a requirement spread over (a)-(e) is credited
# as a whole, and a paragraph that concentrates the question still separates sections that
# tie on the union. The best `sections` sections are kept, each with at most
# `rows_per_section` of its paragraphs (most question weight first, then document order).
# `quote_literal(l)::tsquery` matches the stemmed lexeme as-is (no second stemming pass).
ECFR_SECTION_COVERAGE = f"""
SELECT {_ECFR_COLUMNS}, k.section_score, k.row_score
FROM (
WITH lex AS (
    SELECT DISTINCT unnest(tsvector_to_array(to_tsvector('english', %(query)s))) AS l
),
hits AS (
    SELECT r.section_number, r.ordinal, lex.l
    FROM lex
    JOIN cfr_records r ON r.search_vector @@ quote_literal(lex.l)::tsquery
    WHERE NOT r.is_reserved
      {_ECFR_FILTERS}
),
df AS (SELECT l, count(DISTINCT section_number) AS n FROM hits GROUP BY l),
total AS (SELECT count(*)::float8 AS n FROM cfr_sections WHERE NOT is_reserved),
w AS (SELECT df.l, ln(GREATEST(total.n / df.n, 1.0)) AS idf FROM df, total),
mass AS (SELECT sum(idf) AS m FROM w),
row_cov AS (
    SELECT h.section_number, h.ordinal, sum(w.idf) AS cov
    FROM hits h JOIN w USING (l)
    GROUP BY h.section_number, h.ordinal
),
sec AS (
    SELECT s.section_number,
           (s.cov + b.best) / NULLIF((SELECT m FROM mass), 0) AS section_score
    FROM (SELECT u.section_number, sum(w.idf) AS cov
          FROM (SELECT DISTINCT section_number, l FROM hits) u JOIN w USING (l)
          GROUP BY u.section_number) s
    JOIN (SELECT section_number, max(cov) AS best FROM row_cov GROUP BY section_number) b USING (section_number)
    ORDER BY section_score DESC NULLS LAST, s.section_number
    LIMIT %(sections)s
),
ranked AS (
    SELECT rc.section_number, rc.ordinal, sec.section_score,
           rc.cov / NULLIF((SELECT m FROM mass), 0) AS row_score,
           ROW_NUMBER() OVER (PARTITION BY rc.section_number ORDER BY rc.cov DESC, rc.ordinal) AS rn
    FROM row_cov rc JOIN sec USING (section_number)
)
SELECT * FROM ranked
) k
JOIN cfr_records r ON r.section_number = k.section_number AND r.ordinal = k.ordinal
LEFT JOIN cfr_parts p ON p.number = r.part_number
WHERE k.rn <= %(rows_per_section)s
  AND r.removed_on IS NULL
  AND k.section_score IS NOT NULL
ORDER BY k.section_score + k.row_score DESC, r.section_number, r.ordinal
LIMIT %(limit)s
"""

_GUIDANCE_COLUMNS = """
    c.body_text, c.title_text, c.ordinal, c.start_offset,
    g.title, g.docket_number, g.status, g.center, g.sector, g.issued_on,
    g.source_url, g.anchor_part, g.anchor_section, g.doc_class, g.retrievable
"""

_GUIDANCE_FILTERS = """
    AND (%(retrievable)s::boolean IS NULL OR g.retrievable = %(retrievable)s)
    AND (%(sector)s::text IS NULL OR g.sector = %(sector)s)
    AND (%(center)s::text IS NULL OR g.center = %(center)s)
    AND (%(status)s::text IS NULL OR g.status = %(status)s)
    AND (%(anchor_part)s::text IS NULL OR g.anchor_part = %(anchor_part)s)
    AND (%(anchor_section)s::text IS NULL OR g.anchor_section = %(anchor_section)s)
"""

# Full-text over chunks joined to their document. The best-ranked candidates
# are taken first (limit * 10), then at most per_document chunks of the same
# guidance survive so one long document cannot fill the whole result list.
GUIDANCE_FULL_TEXT = f"""
SELECT * FROM (
    SELECT candidates.*,
           ROW_NUMBER() OVER (PARTITION BY guidance_id ORDER BY rank DESC, ordinal) AS rn
    FROM (
        SELECT {_GUIDANCE_COLUMNS}, c.guidance_id,
               ts_rank_cd(c.search_vector, q, 1) AS rank
        FROM fda_guidance_chunks c
        JOIN fda_guidance g ON g.id = c.guidance_id,
             websearch_to_tsquery('english', %(query)s) q
        WHERE c.search_vector @@ q
          {_GUIDANCE_FILTERS}
        ORDER BY rank DESC, g.source_url, c.ordinal
        LIMIT %(candidate_limit)s
    ) candidates
) ranked
WHERE rn <= %(per_document)s
ORDER BY rank DESC, source_url, ordinal
LIMIT %(limit)s
"""

# AG1 anchored guidance: the chunks of the guidance documents anchored to the requested sections
# (fda_guidance_anchors), ranked by an OR query of the question's words, at most per_document per document.
GUIDANCE_ANCHORED = f"""
SELECT * FROM (
    SELECT candidates.*,
           ROW_NUMBER() OVER (PARTITION BY guidance_id ORDER BY rank DESC, ordinal) AS rn
    FROM (
        SELECT DISTINCT ON (c.id) {_GUIDANCE_COLUMNS}, c.guidance_id,
               ts_rank_cd(c.search_vector, websearch_to_tsquery('english', %(query)s), 1) AS rank
        FROM fda_guidance_chunks c
        JOIN fda_guidance g ON g.id = c.guidance_id
        JOIN fda_guidance_anchors a ON a.guidance_id = g.id
        WHERE a.section = ANY(%(sections)s)
          AND c.search_vector @@ websearch_to_tsquery('english', %(query)s)
          {_GUIDANCE_FILTERS}
        ORDER BY c.id
    ) candidates
) ranked
WHERE rn <= %(per_document)s
ORDER BY rank DESC, source_url, ordinal
LIMIT %(limit)s
"""

# Exact docket lookup: the opening chunks of the document(s) with that docket.
GUIDANCE_EXACT_DOCKET = f"""
SELECT {_GUIDANCE_COLUMNS}
FROM fda_guidance_chunks c
JOIN fda_guidance g ON g.id = c.guidance_id
WHERE g.docket_number = %(docket)s
  AND c.ordinal < %(per_document)s
  {_GUIDANCE_FILTERS}
ORDER BY g.issued_on DESC NULLS LAST, g.source_url, c.ordinal
LIMIT %(limit)s
"""

# The CFR's own definition of one term: a paragraph that OPENS with the term
# (after its paragraph label, if any) followed by "means" - the definitional
# convention of every Title 21 definitions section ("(x) FDA means the Food and
# Drug Administration."). The full-text condition only narrows the scan through
# the GIN index; the regular expression decides. Shortest paragraphs first.
ECFR_DEFINITION = f"""
SELECT {_ECFR_COLUMNS}
FROM cfr_records r
LEFT JOIN cfr_parts p ON p.number = r.part_number
WHERE r.search_vector @@ websearch_to_tsquery('english', %(query)s)
  AND r.body_text ~* %(pattern)s
  AND NOT r.is_reserved
  {_ECFR_FILTERS}
ORDER BY length(r.body_text), r.section_number, r.ordinal
LIMIT %(limit)s
"""

# The paragraphs that define one term of a definitional general question (regulus.retrieval.query.
# defined_term_pattern): the phrase prefilter uses the GIN search index, the regex keeps only paragraphs that
# OPEN by defining the term. All defining paragraphs are returned; the retriever picks the best section.
ECFR_DEFINED_TERM = f"""
SELECT {_ECFR_COLUMNS}
FROM cfr_records r
LEFT JOIN cfr_parts p ON p.number = r.part_number
WHERE r.search_vector @@ phraseto_tsquery('english', %(query)s)
  AND r.body_text ~* %(pattern)s
  AND NOT r.is_reserved
  {_ECFR_FILTERS}
ORDER BY r.section_number, r.ordinal
LIMIT %(limit)s
"""

# All text of one section, for the tie-break between sections that define the same terms.
ECFR_SECTION_TEXT = """
SELECT string_agg(r.body_text, ' ' ORDER BY r.ordinal) AS body_text
FROM cfr_records r
WHERE r.section_number = %(section_number)s
  AND r.removed_on IS NULL
"""

# How many live section headings match a regular expression: how specific a heading (or one of its words) is.
ECFR_HEADING_COUNT = """
SELECT count(*) AS n
FROM cfr_sections s
WHERE NOT s.is_reserved
  AND lower(s.heading) ~ %(query)s
"""

# P3 cross-reference expansion: the provisions that the seed rows (section, paragraph) explicitly cite, through the
# resolved links in cfr_cross_references, in seed order. A reference to another paragraph of the seed's own section
# is structure, not a cross-reference, and is not followed. One hop only.
ECFR_CROSS_REFERENCED = f"""
SELECT {_ECFR_COLUMNS}, x.raw_text AS xref_text, s.n AS seed_rank,
       f.section_number AS xref_from_section, COALESCE(f.paragraph_path, '') AS xref_from_paragraph
FROM unnest(%(section_number)s::text[], %(paragraph)s::text[]) WITH ORDINALITY AS s(section_number, paragraph_path, n)
JOIN cfr_records f ON f.section_number = s.section_number AND COALESCE(f.paragraph_path, '') = s.paragraph_path
                  AND f.removed_on IS NULL
JOIN cfr_cross_references x ON x.from_record_id = f.id AND x.resolvable
JOIN cfr_records r ON r.id = x.to_record_id
LEFT JOIN cfr_parts p ON p.number = r.part_number
WHERE r.section_number <> f.section_number
  AND NOT r.is_reserved
  {_ECFR_FILTERS}
ORDER BY s.n, r.section_number, r.ordinal
LIMIT %(limit)s
"""

ALL_STATEMENTS = {
    "ECFR_CROSS_REFERENCED": ECFR_CROSS_REFERENCED,
    "ECFR_HEADING_COUNT": ECFR_HEADING_COUNT,
    "ECFR_DEFINED_TERM": ECFR_DEFINED_TERM,
    "ECFR_SECTION_TEXT": ECFR_SECTION_TEXT,
    "ECFR_DEFINITION": ECFR_DEFINITION,
    "ECFR_EXACT_SECTION": ECFR_EXACT_SECTION,
    "ECFR_EXACT_ROW": ECFR_EXACT_ROW,
    "ECFR_EXACT_PART": ECFR_EXACT_PART,
    "ECFR_PART_SECTIONS": ECFR_PART_SECTIONS,
    "ECFR_FULL_TEXT": ECFR_FULL_TEXT,
    "ECFR_TRIGRAM": ECFR_TRIGRAM,
    "ECFR_SECTION_COVERAGE": ECFR_SECTION_COVERAGE,
    "GUIDANCE_FULL_TEXT": GUIDANCE_FULL_TEXT,
    "GUIDANCE_EXACT_DOCKET": GUIDANCE_EXACT_DOCKET,
    "GUIDANCE_ANCHORED": GUIDANCE_ANCHORED,   # AG1 (28 Oct) - registered 30 Oct, as every statement is
}
