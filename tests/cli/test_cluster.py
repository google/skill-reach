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

"""Verify `reach cluster` command line interface, formatting, and community partitioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reach.cli import main

if TYPE_CHECKING:
    from conftest import SyntheticCorpusBuilder


@pytest.fixture
def cluster_corpus(corpus_builder: type[SyntheticCorpusBuilder], tmp_path: Path) -> Path:
    """Provide a corpus of skills with clear community structure."""
    return (
        corpus_builder()
        .add("cloud-run", "Deploy docker containers and serverless microservices.")
        .add("cloud-functions", "Deploy serverless event handlers and functions.")
        .add("postgres-db", "Manage postgres sql database tables schemas queries.")
        .add("mysql-db", "Manage mysql sql database tables schemas queries.")
        .build_disk(tmp_path)
    )


def test_cluster_text_output(cluster_corpus: Path, capsys) -> None:
    """Verify reach cluster text output renders community summary table."""
    exit_code = main(["cluster", str(cluster_corpus)])
    assert exit_code == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Skill Partition:" in combined
    assert "Subagent Candidate Catalogs" in combined
    assert "cluster-1" in combined


def test_cluster_json_output(cluster_corpus: Path, capsys) -> None:
    """Verify reach cluster --format json outputs valid ClusterPartition JSON."""
    exit_code = main(["cluster", str(cluster_corpus), "--format", "json"])
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "clusters" in data
    assert "modularity" in data
    assert data["total_skills"] == 4
    assert len(data["clusters"]) >= 1


def test_cluster_csv_output(cluster_corpus: Path, capsys) -> None:
    """Verify reach cluster --format csv outputs CSV document with headers."""
    exit_code = main(["cluster", str(cluster_corpus), "--format", "csv"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "cluster_id,cohesion,skill" in out
    assert "cloud-run" in out


def test_cluster_target_size_and_max_clusters(cluster_corpus: Path) -> None:
    """Verify reach cluster with target-size and max-clusters parameters."""
    exit_code = main(
        [
            "cluster",
            str(cluster_corpus),
            "--target-size",
            "2",
            "--max-clusters",
            "2",
            "--format",
            "json",
        ]
    )
    assert exit_code == 0


def test_cluster_empty_corpus_fails(tmp_path: Path, capsys) -> None:
    """Verify reach cluster on empty corpus exits with error code 2."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    exit_code = main(["cluster", str(empty_dir)])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "no skills found" in err


def test_cluster_respects_config(cluster_corpus: Path, tmp_path: Path) -> None:
    """Verify reach cluster loads and respects explicit --config file."""
    config_file = tmp_path / "reach.toml"
    config_file.write_text('[runtime]\nagent = "fake"\n')
    exit_code = main(
        [
            "cluster",
            str(cluster_corpus),
            "--config",
            str(config_file),
            "--format",
            "json",
        ]
    )
    assert exit_code == 0
