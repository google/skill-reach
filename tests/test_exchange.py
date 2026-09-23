# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify query set export, external format import, and schema validation."""

from __future__ import annotations

import csv
import io
import json

import pytest

from reach.exchange import BOM as STRIPPED_BOM
from reach.exchange import Exchange, FieldMap, export_query_set, import_query_set
from reach.models import QueryKind
from reach.queries import Origin, QuerySet, query_set_digest

FORMATS = list(Exchange)

BOM = "\N{ZERO WIDTH NO-BREAK SPACE}"


def rows_of(exported: str) -> list[dict[str, str]]:
    """Parse exported CSV string into dictionary rows."""
    return list(csv.DictReader(io.StringIO(exported)))


@pytest.mark.parametrize("fmt", FORMATS)
def test_our_own_format_round_trips_without_losing_a_query(
    exchange_set: QuerySet,
    fmt: Exchange,
) -> None:
    """Verify exporting and importing QuerySet preserves all queries unchanged."""
    returned = import_query_set(
        export_query_set(exchange_set, fmt),
        fmt,
        catalog_id=exchange_set.catalog_id,
    )
    assert returned.queries == exchange_set.queries


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_round_trip_does_not_fork_the_evidence(exchange_set: QuerySet, fmt: Exchange) -> None:
    """Verify ground truth digest remains invariant across export and import."""
    returned = import_query_set(
        export_query_set(exchange_set, fmt),
        fmt,
        catalog_id=exchange_set.catalog_id,
    )
    assert query_set_digest(returned) == query_set_digest(exchange_set)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_set_a_spreadsheet_saved_still_carries_its_own_ids(
    exchange_set: QuerySet,
    fmt: Exchange,
) -> None:
    """Verify import strips leading byte order mark without losing query IDs."""
    returned = import_query_set(
        BOM + export_query_set(exchange_set, fmt),
        fmt,
        catalog_id=exchange_set.catalog_id,
    )
    assert [q.id for q in returned.queries] == [q.id for q in exchange_set.queries]


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_spreadsheet_saving_a_set_does_not_fork_the_evidence(
    exchange_set: QuerySet,
    fmt: Exchange,
) -> None:
    """Verify BOM-prefixed imports produce identical ground truth digests."""
    returned = import_query_set(
        BOM + export_query_set(exchange_set, fmt),
        fmt,
        catalog_id=exchange_set.catalog_id,
    )
    assert query_set_digest(returned) == query_set_digest(exchange_set)


def test_the_mark_the_reader_strips_is_the_one_a_spreadsheet_writes() -> None:
    """Verify BOM constant matches stripped BOM and UTF-8 encoded bytes."""
    assert BOM == STRIPPED_BOM
    assert BOM.encode("utf-8") == b"\xef\xbb\xbf"


@pytest.mark.parametrize("fmt", FORMATS)
def test_an_unlabeled_query_comes_back_unlabeled(exchange_set: QuerySet, fmt: Exchange) -> None:
    """Verify queries with kind=None preserve None across round trip."""
    returned = import_query_set(
        export_query_set(exchange_set, fmt),
        fmt,
        catalog_id=exchange_set.catalog_id,
    )
    assert returned.queries[1].kind is None


@pytest.mark.parametrize("fmt", FORMATS)
def test_an_abstention_survives_the_trip(exchange_set: QuerySet, fmt: Exchange) -> None:
    """Verify out-of-scope abstention queries preserve kind and expected_skill=None."""
    returned = import_query_set(
        export_query_set(exchange_set, fmt),
        fmt,
        catalog_id=exchange_set.catalog_id,
    )
    abstain = next(q for q in returned.queries if q.id == "x-abstain")
    assert abstain.kind is QueryKind.OUT_OF_SCOPE
    assert abstain.expected_skill is None


@pytest.mark.parametrize("fmt", FORMATS)
def test_punctuation_in_a_query_survives_the_trip(
    exchange_set: QuerySet,
    fmt: Exchange,
) -> None:
    """Verify multi-line text and quotes survive export/import cycle."""
    returned = import_query_set(
        export_query_set(exchange_set, fmt),
        fmt,
        catalog_id=exchange_set.catalog_id,
    )
    punctuated = next(q for q in returned.queries if q.id == "x-punctuated")
    assert punctuated.text == 'Delete "cold" objects, then archive\nwhatever is left.'


def test_an_export_names_our_own_fields(exchange_set: QuerySet) -> None:
    """Verify CSV export includes standard Reach field headers."""
    header = rows_of(export_query_set(exchange_set, Exchange.CSV))[0].keys()
    assert set(header) == {
        "id",
        "text",
        "kind",
        "expected_skill",
        "acceptable_skills",
        "notes",
    }


def test_an_export_holds_one_row_per_query(exchange_set: QuerySet) -> None:
    """Verify JSONL export outputs exactly one line per query."""
    lines = export_query_set(exchange_set, Exchange.JSONL).splitlines()
    assert len(lines) == len(exchange_set.queries)


def test_an_absent_kind_exports_as_absent_rather_than_as_a_word(
    exchange_set: QuerySet,
) -> None:
    """Verify absent kind exports as empty string in CSV."""
    rows = {r["id"]: r for r in rows_of(export_query_set(exchange_set, Exchange.CSV))}
    assert rows["x-unlabeled"]["kind"] == ""


FOREIGN = "prompt|answer\nrotate our keys|kms-rotation\n"

FOREIGN_MAP = FieldMap(
    text="prompt",
    expected_skill="answer",
)


def foreign_csv() -> str:
    """Return synthetic foreign CSV table representation."""
    return FOREIGN.replace("|", ",")


def test_a_mapping_renames_columns_without_a_converter() -> None:
    """Verify FieldMap remaps external column names to standard Reach fields."""
    imported = import_query_set(
        foreign_csv(),
        Exchange.CSV,
        catalog_id="c",
        mapping=FOREIGN_MAP,
    )
    assert imported.queries[0].text == "rotate our keys"
    assert imported.queries[0].expected_skill == "kms-rotation"


def test_a_mapping_imports_delimited_acceptable_skills_and_notes() -> None:
    """Verify custom mappings preserve neutral skills and per-query notes."""
    document = (
        "prompt,answer,neutral,rationale\n"
        "rotate our keys,kms-rotation,skill-finder|kms-router,reviewed manually\n"
    )
    imported = import_query_set(
        document,
        Exchange.CSV,
        catalog_id="c",
        mapping=FieldMap(
            text="prompt",
            expected_skill="answer",
            acceptable_skills="neutral",
            notes="rationale",
            separator="|",
        ),
    )
    query = imported.queries[0]
    assert query.acceptable_skills == ("skill-finder", "kms-router")
    assert query.notes == "reviewed manually"


def test_jsonl_import_accepts_an_acceptable_skills_array() -> None:
    """Verify canonical JSONL arrays import without string coercion."""
    document = json.dumps(
        {
            "text": "rotate our keys",
            "expected_skill": "kms-rotation",
            "acceptable_skills": ["skill-finder", "kms-router"],
        },
    )
    imported = import_query_set(document, Exchange.JSONL, catalog_id="c")
    assert imported.queries[0].acceptable_skills == ("skill-finder", "kms-router")


@pytest.mark.parametrize(
    "bad_skill",
    [123, {"nested": "object"}],
    ids=["integer", "object"],
)
def test_jsonl_import_refuses_non_string_acceptable_skills(bad_skill: object) -> None:
    """Verify JSONL acceptable_skills arrays reject non-string entries."""
    document = json.dumps(
        {
            "text": "rotate our keys",
            "expected_skill": "kms-rotation",
            "acceptable_skills": ["skill-finder", bad_skill],
        },
    )
    with pytest.raises(
        ValueError,
        match="acceptable_skills array elements must be strings",
    ):
        import_query_set(document, Exchange.JSONL, catalog_id="c")


def test_an_empty_separator_is_refused() -> None:
    """Verify FieldMap rejects a separator that cannot split imported cells."""
    with pytest.raises(ValueError, match="separator must not be empty"):
        FieldMap(separator="")


def test_a_row_with_no_id_is_numbered() -> None:
    """Verify imported rows without ID columns receive auto-generated sequence IDs."""
    imported = import_query_set(
        foreign_csv(),
        Exchange.CSV,
        catalog_id="c",
        mapping=FOREIGN_MAP,
    )
    assert imported.queries[0].id == "q-1"


def test_the_generated_id_prefix_belongs_to_the_caller() -> None:
    """Verify FieldMap id_prefix customizes auto-generated sequence ID format."""
    imported = import_query_set(
        foreign_csv(),
        Exchange.CSV,
        catalog_id="c",
        mapping=FOREIGN_MAP.model_copy(update={"id_prefix": "kms"}),
    )
    assert imported.queries[0].id == "kms-1"


def test_an_import_records_that_it_was_imported() -> None:
    """Verify import metadata populates provenance origin and source file name."""
    imported = import_query_set(
        foreign_csv(),
        Exchange.CSV,
        catalog_id="c",
        mapping=FOREIGN_MAP,
        source="theirs.csv",
    )
    assert imported.provenance is not None
    assert imported.provenance.origin is Origin.IMPORTED
    assert imported.provenance.source == "theirs.csv"


def test_an_import_carries_the_catalog_its_labels_are_valid_in() -> None:
    """Verify imported QuerySet attaches designated catalog_id."""
    imported = import_query_set(
        foreign_csv(),
        Exchange.CSV,
        catalog_id="neighborhood:kms",
        mapping=FOREIGN_MAP,
    )
    assert imported.catalog_id == "neighborhood:kms"


def test_blank_lines_in_a_jsonl_document_are_skipped() -> None:
    """Verify blank lines in JSONL documents are ignored during import."""
    line = json.dumps({"text": "rotate our keys", "expected_skill": "kms"})
    imported = import_query_set(f"\n{line}\n\n", Exchange.JSONL, catalog_id="c")
    assert len(imported.queries) == 1


def test_a_mapping_naming_a_column_that_is_not_there_is_refused() -> None:
    """Verify FieldMap referencing missing column raises ValueError."""
    with pytest.raises(ValueError, match="question"):
        import_query_set(
            foreign_csv(),
            Exchange.CSV,
            catalog_id="c",
            mapping=FieldMap(text="question"),
        )


def test_a_refusal_lists_the_columns_the_file_does_have() -> None:
    """Verify missing column error message lists available column names."""
    with pytest.raises(ValueError, match="prompt"):
        import_query_set(
            foreign_csv(),
            Exchange.CSV,
            catalog_id="c",
            mapping=FieldMap(text="question"),
        )


@pytest.mark.parametrize(
    ("cell", "reason"),
    [("", "blank text"), ("   ", "whitespace-only text")],
)
def test_a_row_with_no_query_in_it_is_refused(cell: str, reason: str) -> None:
    """Verify import raises ValueError when query text is blank or whitespace."""
    document = f"text,expected_skill\n{cell},kms\n"
    with pytest.raises(ValueError, match="row 1"):
        import_query_set(document, Exchange.CSV, catalog_id="c")


def test_an_unknown_kind_is_refused_against_the_row_it_is_on() -> None:
    """Verify import rejects unrecognized query kind values with row number context."""
    document = "text,kind,expected_skill\nrotate keys,explicit,kms\n"
    with pytest.raises(ValueError, match="row 1"):
        import_query_set(document, Exchange.CSV, catalog_id="c")


def test_an_abstention_that_names_an_expected_skill_is_refused() -> None:
    """Verify import rejects out_of_scope query that specifies expected_skill."""
    document = "text,kind,expected_skill\nrotate keys,out_of_scope,kms\n"
    with pytest.raises(ValueError, match="row 1"):
        import_query_set(document, Exchange.CSV, catalog_id="c")


SPOILED = (
    "id,text,kind,expected_skill\n"
    "q1,rotate our keys,implicit,kms\n"
    "q2,,implicit,kms\n"
    "q3,tier cold objects,explicit,gcs\n"
    "q4,what is the capital of France,out_of_scope,gcs\n"
)


@pytest.mark.parametrize("row", ["row 2", "row 3", "row 4"])
def test_every_unusable_row_is_named_and_not_just_the_first(row: str) -> None:
    """Verify multi-row import errors report all failing row numbers."""
    with pytest.raises(ValueError, match=row):
        import_query_set(SPOILED, Exchange.CSV, catalog_id="c")


def test_a_usable_row_is_not_named_among_the_refusals() -> None:
    """Verify valid rows are omitted from unusable row error reports."""
    with pytest.raises(ValueError, match="unusable rows") as refusal:
        import_query_set(SPOILED, Exchange.CSV, catalog_id="c")
    assert "row 1" not in str(refusal.value)


def test_a_refusal_counts_the_rows_it_is_about() -> None:
    """Verify import error summary counts total number of unusable rows."""
    with pytest.raises(ValueError, match="3 unusable rows"):
        import_query_set(SPOILED, Exchange.CSV, catalog_id="c")


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        ("text\n   \n", "text is blank"),
        ("text,kind,expected_skill\nrotate keys,explicit,kms\n", "kind"),
        (
            "text,kind,expected_skill\nrotate keys,out_of_scope,kms\n",
            "disagrees with",
        ),
    ],
    ids=["blank-text", "unknown-kind", "abstention-with-a-skill"],
)
def test_a_refusal_says_what_is_wrong_in_the_reader_s_terms(document: str, reason: str) -> None:
    """Verify import failure messages explain errors clearly without raw validator internals."""
    with pytest.raises(ValueError, match="row 1") as refusal:
        import_query_set(document, Exchange.CSV, catalog_id="c")
    reported = str(refusal.value)
    assert reason in reported
    assert "validation error" not in reported
    assert "errors.pydantic.dev" not in reported


def test_duplicate_ids_are_refused_at_the_boundary() -> None:
    """Verify import rejects documents with duplicate query IDs."""
    document = "id,text,expected_skill\nq,a,kms\nq,b,kms\n"
    with pytest.raises(ValueError, match="duplicate query ids"):
        import_query_set(document, Exchange.CSV, catalog_id="c", source="theirs.csv")


def test_a_document_with_no_rows_is_refused() -> None:
    """Verify import rejects completely empty input string."""
    with pytest.raises(ValueError, match="no rows"):
        import_query_set("", Exchange.CSV, catalog_id="c")


def test_a_csv_with_a_header_and_nothing_under_it_is_refused() -> None:
    """Verify import rejects CSV input containing only headers."""
    with pytest.raises(ValueError, match="no rows"):
        import_query_set("text,expected_skill\n", Exchange.CSV, catalog_id="c")


def test_a_jsonl_line_that_is_not_an_object_is_refused() -> None:
    """Verify import rejects JSONL records that are not JSON objects."""
    with pytest.raises(ValueError, match="row 2"):
        import_query_set(
            '{"text": "a", "expected_skill": "s"}\n["not", "an", "object"]\n',
            Exchange.JSONL,
            catalog_id="c",
        )


def test_a_jsonl_line_that_is_not_json_is_refused() -> None:
    """Verify import rejects malformed JSON lines with row number context."""
    with pytest.raises(ValueError, match="row 1"):
        import_query_set("{not json}\n", Exchange.JSONL, catalog_id="c")


def test_import_generates_sequential_ids_when_missing() -> None:
    """Verify import assigns sequential IDs using custom prefix when IDs omitted."""
    document = "text,expected_skill\nFirst query,s1\nSecond query,s2\n"
    mapping = FieldMap(id_prefix="custom")
    imported = import_query_set(
        document,
        Exchange.CSV,
        catalog_id="c",
        mapping=mapping,
    )
    assert [q.id for q in imported.queries] == ["custom-1", "custom-2"]


def test_jsonl_import_ignores_trailing_empty_lines() -> None:
    """Verify JSONL import gracefully ignores trailing empty whitespace lines."""
    document = (
        '{"id": "q1", "text": "test 1", "expected_skill": "s1"}\n'
        '{"id": "q2", "text": "test 2", "expected_skill": "s2"}\n\n   \n'
    )
    imported = import_query_set(document, Exchange.JSONL, catalog_id="c")
    assert len(imported.queries) == 2
