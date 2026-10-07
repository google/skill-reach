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

"""Unit tests for model JSON completion extraction and sanitization."""

from __future__ import annotations

from pathlib import Path

import pytest

from reach._io import read_model, write_model
from reach._json import (
    extract_json_payload,
    parse_model_json,
    sanitize_json_string,
)


def test_parse_model_json_plain_object() -> None:
    """Verify clean JSON object parses successfully."""
    raw = '{"name": "test", "count": 42}'
    data = parse_model_json(raw)
    assert data == {"name": "test", "count": 42}


def test_parse_model_json_plain_array() -> None:
    """Verify clean JSON array parses successfully."""
    raw = '[{"name": "a"}, {"name": "b"}]'
    data = parse_model_json(raw)
    assert data == [{"name": "a"}, {"name": "b"}]


def test_parse_model_json_with_trailing_comma_in_object() -> None:
    """Verify trailing commas in JSON objects are sanitized."""
    raw = '{"name": "test", "count": 42, }'
    data = parse_model_json(raw)
    assert data == {"name": "test", "count": 42}


def test_parse_model_json_with_trailing_comma_in_array() -> None:
    """Verify trailing commas in JSON arrays are sanitized."""
    raw = '["a", "b", ]'
    data = parse_model_json(raw)
    assert data == ["a", "b"]


def test_parse_model_json_nested_trailing_commas() -> None:
    """Verify nested trailing commas in objects and arrays are sanitized."""
    raw = """
    {
      "queries": [
        {
          "text": "How do I deploy?",
          "citation": "deployment guide",
          "reason": "covers deployment",
        },
      ],
    }
    """
    data = parse_model_json(raw)
    assert len(data["queries"]) == 1
    assert data["queries"][0]["text"] == "How do I deploy?"


def test_parse_model_json_does_not_corrupt_commas_in_strings() -> None:
    """Verify commas followed by braces or brackets inside string literals are preserved."""
    raw = '{"query": "SELECT a, } FROM b", "filter": "val, ]"}'
    data = parse_model_json(raw)
    assert data["query"] == "SELECT a, } FROM b"
    assert data["filter"] == "val, ]"


def test_parse_model_json_fenced_json() -> None:
    """Verify markdown fenced json code blocks are extracted and parsed."""
    raw = 'Here is the result:\n```json\n{"queries": [{"text": "q1"}]}\n```\nDone.'
    data = parse_model_json(raw)
    assert data["queries"][0]["text"] == "q1"


def test_parse_model_json_fenced_generic_with_surrounding_braces() -> None:
    """Verify code fences without json tag parse even when preamble has braces."""
    raw = (
        "Context: {config_var}\n"
        "```\n"
        '{"status": "ok", "items": [1, 2, ], }\n'
        "```\n"
        "Note: check {other_var}."
    )
    data = parse_model_json(raw)
    assert data == {"status": "ok", "items": [1, 2]}


def test_parse_model_json_fenced_with_inner_code_blocks() -> None:
    """Verify outer code fences do not truncate on inner markdown code blocks."""
    raw = (
        "```json\n"
        "{\n"
        '  "queries": [\n'
        "    {\n"
        '      "text": "How to run CLI?",\n'
        '      "citation": "Run this snippet:\\n```bash\\nreach eval\\n```\\nDone."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```"
    )
    data = parse_model_json(raw)
    citation = data["queries"][0]["citation"]
    assert "```bash\nreach eval\n```" in citation


def test_parse_model_json_top_level_array_with_fences() -> None:
    """Verify top-level JSON array inside fences is not sliced into an invalid object."""
    raw = '```json\n[\n  {"id": 1},\n  {"id": 2, },\n]\n```'
    data = parse_model_json(raw)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["id"] == 1
    assert data[1]["id"] == 2


def test_parse_model_json_invalid_raises_value_error() -> None:
    """Verify non-JSON response raises ValueError with descriptive message."""
    with pytest.raises(ValueError, match="generator reply was not JSON"):
        parse_model_json("I apologize, but I cannot assist with this request.")


def test_sanitize_json_string_handles_escaped_quotes() -> None:
    """Verify state machine handles escaped quotes in string literals properly."""
    raw = '{"text": "He said \\"Hello, }\\" before leaving", }'
    cleaned = sanitize_json_string(raw)
    assert cleaned == '{"text": "He said \\"Hello, }\\" before leaving" }'
    data = parse_model_json(raw)
    assert data["text"] == 'He said "Hello, }" before leaving'


def test_extract_json_payload() -> None:
    """Verify extract_json_payload unwraps outer code fence and trims text."""
    raw = 'Preamble\n```json\n{"key": "value"}\n```\nPostamble'
    assert extract_json_payload(raw) == '{"key": "value"}'


def test_parse_model_json_with_citation_tag_preamble() -> None:
    """Verify preamble citation tag like [1] does not confuse bracket extraction."""
    raw = 'Here are the drafted queries [1]: {"queries": [{"text": "deploy", "citation": "guide"}]}'
    data = parse_model_json(raw)
    assert len(data["queries"]) == 1
    assert data["queries"][0]["text"] == "deploy"


def test_parse_model_json_with_subsequent_code_fence() -> None:
    """Verify closing fence matches opening fence without capturing subsequent blocks."""
    raw = (
        "Here is the query set:\n"
        "```json\n"
        '{"queries": [{"text": "run app"}]}\n'
        "```\n"
        "You can evaluate it with:\n"
        "```bash\n"
        "reach eval --queries queries.json\n"
        "```\n"
        "Done!"
    )
    data = parse_model_json(raw)
    assert data["queries"][0]["text"] == "run app"


def test_parse_model_json_raw_trailing_comma_prior_to_fence_extraction() -> None:
    """Verify raw JSON with trailing comma succeeds without fence extraction."""
    raw = '{"queries": [{"text": "q1", }], }'
    data = parse_model_json(raw)
    assert data["queries"][0]["text"] == "q1"


def test_read_model_and_write_model_round_trip(tmp_path: Path) -> None:
    """Verify write_model and read_model serialize and restore Pydantic models faithfully."""
    from pydantic import BaseModel

    class SampleModel(BaseModel):
        name: str
        value: int

    sample = SampleModel(name="test", value=42)
    target = tmp_path / "subdir" / "model.json"
    written = write_model(sample, target)
    assert written == target.resolve()
    assert target.exists()

    loaded = read_model(SampleModel, target)
    assert loaded == sample
