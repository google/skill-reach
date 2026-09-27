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

"""Validate modularity skill clustering without external ML dependencies."""

from __future__ import annotations

from pathlib import Path

import pytest

from reach.models import Skill


def _make_skill(name: str, description: str) -> Skill:
    """Create a minimal Skill instance for clustering tests."""
    return Skill(name=name, description=description, path=Path(f"/mock/{name}"))


def test_cluster_does_not_import_heavy_ml_libraries() -> None:
    """Ensure reach.cluster does not import numpy, scipy, or sklearn."""
    import reach.cluster

    heavy_libs = {"numpy", "scipy", "sklearn"}
    cluster_module_dict = reach.cluster.__dict__
    for lib in heavy_libs:
        assert lib not in cluster_module_dict, f"reach.cluster should not import {lib}"


def test_cluster_skills_three_communities() -> None:
    """Verify 3 disjoint semantic communities partition cleanly with positive modularity."""
    from reach.cluster import cluster_skills

    skills = [
        _make_skill("cloud-run", "deploy cloud run containers and microservices on gcp"),
        _make_skill("cloud-functions", "deploy cloud functions serverless event handlers on gcp"),
        _make_skill("postgres-db", "manage postgres relational database tables schemas and sql"),
        _make_skill("mysql-db", "manage mysql relational database tables schemas and sql"),
        _make_skill("react-ui", "build react frontend ui components web client jsx"),
        _make_skill("vue-ui", "build vue frontend ui components web client vuejs"),
    ]

    partition = cluster_skills(skills)
    assert partition.total_skills == 6
    assert partition.modularity > 0.0
    assert len(partition.clusters) == 3

    # Check each community contains the expected pair
    clusters_as_sets = [set(c.skills) for c in partition.clusters]
    assert {"cloud-run", "cloud-functions"} in clusters_as_sets
    assert {"postgres-db", "mysql-db"} in clusters_as_sets
    assert {"react-ui", "vue-ui"} in clusters_as_sets

    # Verify cohesion is positive for multi-skill clusters
    for c in partition.clusters:
        assert c.cohesion > 0.0
        assert c.id.startswith("cluster-")


def test_cluster_skills_empty_corpus() -> None:
    """Verify empty corpus returns empty partition with zero modularity."""
    from reach.cluster import cluster_skills

    partition = cluster_skills([])
    assert partition.total_skills == 0
    assert partition.modularity == 0.0
    assert partition.clusters == ()


def test_cluster_skills_single_skill() -> None:
    """Verify single skill returns single cluster with zero modularity."""
    from reach.cluster import cluster_skills

    skill = _make_skill("solo", "unique solo skill doing something alone")
    partition = cluster_skills([skill])
    assert partition.total_skills == 1
    assert partition.modularity == 0.0
    assert len(partition.clusters) == 1
    assert partition.clusters[0].skills == ("solo",)


def test_cluster_skills_completely_disconnected() -> None:
    """Verify skills with zero lexical overlap remain in singleton clusters."""
    from reach.cluster import cluster_skills

    skills = [
        _make_skill("alpha", "aardvark alpine acrobat"),
        _make_skill("beta", "banana butterfly badminton"),
        _make_skill("gamma", "giraffe glacier geometry"),
    ]
    partition = cluster_skills(skills)
    assert partition.total_skills == 3
    # When all weights are zero, no merges should occur
    assert len(partition.clusters) == 3
    assert partition.modularity == 0.0


def test_cluster_skills_determinism() -> None:
    """Verify repeated clustering on same skills yields identical results."""
    from reach.cluster import cluster_skills

    skills = [
        _make_skill("s1", "python fastapi backend service"),
        _make_skill("s2", "python flask rest api service"),
        _make_skill("s3", "docker container build image"),
        _make_skill("s4", "docker compose multi container environment"),
    ]
    p1 = cluster_skills(skills)
    p2 = cluster_skills(skills)

    assert p1.modularity == p2.modularity
    assert len(p1.clusters) == len(p2.clusters)
    for c1, c2 in zip(p1.clusters, p2.clusters, strict=True):
        assert c1.id == c2.id
        assert c1.skills == c2.skills
        assert pytest.approx(c1.cohesion) == c2.cohesion


def test_cluster_skills_resolution_parameter() -> None:
    """Verify higher resolution promotes smaller clusters, lower resolution promotes merging."""
    from reach.cluster import cluster_skills

    skills = [
        _make_skill("s1", "python fastapi backend"),
        _make_skill("s2", "python backend rest"),
        _make_skill("s3", "docker container deploy"),
        _make_skill("s4", "docker kubernetes deploy"),
    ]
    low_res = cluster_skills(skills, resolution=0.1)
    high_res = cluster_skills(skills, resolution=5.0)

    assert len(low_res.clusters) <= len(high_res.clusters)


def test_cluster_skills_deduplicates_skills_with_identical_names() -> None:
    """Verify cluster_skills deduplicates skills with identical names without singletons."""
    from reach.cluster import cluster_skills

    skills = [
        _make_skill("cloud-run", "deploy cloud run containers and microservices on gcp"),
        _make_skill("cloud-run", "duplicate copy of cloud run from another plugin path"),
        _make_skill("cloud-functions", "deploy cloud functions serverless event handlers on gcp"),
        _make_skill("postgres-db", "manage postgres relational database tables schemas and sql"),
        _make_skill("mysql-db", "manage mysql relational database tables schemas and sql"),
    ]
    partition = cluster_skills(skills)
    assert partition.total_skills == 4
    all_members = [s for c in partition.clusters for s in c.skills]
    assert len(all_members) == len(set(all_members)) == 4


def test_cluster_partition_cluster_map() -> None:
    """Verify cluster_map property returns a direct mapping of skill name to cluster id."""
    from reach.cluster import cluster_skills

    skills = [
        _make_skill("cloud-run", "deploy containers on gcp"),
        _make_skill("cloud-functions", "serverless functions on gcp"),
        _make_skill("postgres-db", "manage postgres relational database"),
    ]
    partition = cluster_skills(skills)
    c_map = partition.cluster_map

    assert isinstance(c_map, dict)
    assert set(c_map.keys()) == {"cloud-run", "cloud-functions", "postgres-db"}
    for skill_name, cluster_id in c_map.items():
        matching = [c for c in partition.clusters if c.id == cluster_id]
        assert len(matching) == 1
        assert skill_name in matching[0].skills
