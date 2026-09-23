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

"""Provide shared test fixtures, synthetic factories, and mock servers for the test suite."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import metadata
from pathlib import Path
from typing import Any, ClassVar, Self, cast
from urllib.parse import parse_qs, urlparse

import bm25s  # type: ignore[import-untyped]
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from reach import retrieval
from reach.artifact import Artifact
from reach.catalog import build_catalogs, load_skills
from reach.config import RunConfig
from reach.metrics import classification_report, confusion, labeled_pairs
from reach.models import (
    NO_SKILL,
    Catalog,
    CatalogMode,
    InvocationPattern,
    ProbeResult,
    Query,
    QueryKind,
    Skill,
)
from reach.queries import Origin, QuerySet, QuerySetProvenance, save_query_set
from reach.retrieval import K1, B, skill_text, tokenize
from reach.run import Composition, append_result, compose, write_sidecar
from reach.runtime.fake import FakeGenerator, FakeRuntime, register_fake_agent
from reach.views import build_console

# ==============================================================================
# 1. Environment & Global Test Setup
# ==============================================================================

register_fake_agent()

os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture
def clean_browser_env() -> dict[str, str]:
    """Provide an environment mapping stripped of CI and browser bypass flags."""
    return {k: v for k, v in os.environ.items() if k not in ("CI", "REACH_NO_BROWSER")}


class _DummyModel2Vec:
    """Provide deterministic embeddings for tests without external model downloads."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode input texts into deterministic unit-normalized vectors."""
        embeddings: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [float(b) / 255.0 for b in digest[:8]]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings


cast(Any, retrieval)._load_model2vec_model = lambda _name: _DummyModel2Vec()


# ==============================================================================
# 2. Pytest Hooks & Session Lifecycle
# ==============================================================================


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom CLI options for integration tests."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests marked with @pytest.mark.integration",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Conditionally skip integration tests unless explicitly requested."""
    if config.getoption("--integration"):
        return

    markexpr = config.getoption("markexpr", "")
    if "integration" in markexpr:
        return

    args = config.args or []
    if any("integration" in arg for arg in args):
        return

    skip_integration = pytest.mark.skip(
        reason="Integration test: use --integration, -m integration, or target file to run",
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def _clean_repo_reach_dir(session: pytest.Session) -> None:
    """Clean any .reach directory from the repository root for the controller process."""
    if not hasattr(session.config, "workerinput"):
        reach = Path(getattr(session.config, "rootpath", Path.cwd())) / ".reach"
        if reach.exists():
            shutil.rmtree(reach, ignore_errors=True)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Clean any existing .reach directory before test execution begins."""
    _clean_repo_reach_dir(session)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Ensure no .reach directory leaks into the repository root after tests complete."""
    _clean_repo_reach_dir(session)


# ==============================================================================
# 3. Skill & Corpus Fixtures
# ==============================================================================


def format_skill_markdown(
    name: str,
    description: str,
    body: str | None = None,
    category: str | None = None,
) -> str:
    """Format standard SKILL.md markdown text with YAML frontmatter."""
    meta_line = f"metadata:\n  category: {category}\n" if category else ""
    body_content = f"\n{body}\n" if body is not None else f"\n# {name}\n"
    return f"---\nname: {name}\n{meta_line}description: >-\n  {description}\n---\n{body_content}"


@pytest.fixture
def skill_repo(tmp_path: Path) -> Path:
    """Build a synthetic skill repository spanning multiple categories."""
    root = tmp_path / "skills"
    specs = [
        ("gcs-lifecycle-rules", "Storage", "Configures object lifecycle rules."),
        ("gcs-retention-policy", "Storage", "Configures retention and bucket lock."),
        ("gke-basics", "Containers", "Explains GKE cluster fundamentals."),
    ]
    for name, category, description in specs:
        directory = root / category.lower() / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            format_skill_markdown(name=name, category=category, description=description),
            encoding="utf-8",
        )
    return root


@pytest.fixture
def make_skill_root(tmp_path: Path) -> Callable[[str, Mapping[str, str]], Path]:
    """Return a factory function that creates a skill directory root with specified skills."""

    def build(where: str, skills: Mapping[str, str]) -> Path:
        root = tmp_path / where
        for name, description in skills.items():
            directory = root / name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "SKILL.md").write_text(
                format_skill_markdown(name=name, category="Storage", description=description),
                encoding="utf-8",
            )
        root.mkdir(parents=True, exist_ok=True)
        return root

    return build


@pytest.fixture
def make_skill() -> Callable[..., Skill]:
    """Return a factory function creating an in-memory Skill instance."""

    def build(
        name: str,
        description: str,
        *,
        model_invocable: bool = True,
        path: Path | None = None,
    ) -> Skill:
        return Skill(
            name=name,
            description=description,
            path=path if path is not None else Path(name),
            model_invocable=model_invocable,
        )

    return build


@pytest.fixture
def write_skill(tmp_path: Path) -> Callable[..., Path]:
    """Write a minimal SKILL.md file and return its enclosing directory."""

    def _create(
        name: str,
        description: str = "Test description",
        body: str = "# Body",
        root: Path | None = None,
        raw_yaml: str | None = None,
        dir_name: str | None = None,
        path: Path | None = None,
    ) -> Path:
        skill_dir = path or (root or tmp_path) / (dir_name or name)
        skill_dir.mkdir(parents=True, exist_ok=True)
        manifest = skill_dir / "SKILL.md"
        if raw_yaml is not None:
            manifest.write_text(raw_yaml, encoding="utf-8")
        else:
            manifest.write_text(
                format_skill_markdown(name=name, description=description, body=body),
                encoding="utf-8",
            )
        return skill_dir

    return _create


@pytest.fixture
def write_skill_model(write_skill: Callable[..., Path]) -> Callable[..., Skill]:
    """Write a minimal SKILL.md file to disk and return an initialized Skill model."""

    def _create(
        name: str,
        description: str = "Test description",
        body: str = "# Body",
        root: Path | None = None,
        raw_yaml: str | None = None,
        dir_name: str | None = None,
        path: Path | None = None,
        *,
        model_invocable: bool = True,
    ) -> Skill:
        skill_dir = write_skill(
            name=name,
            description=description,
            body=body,
            root=root,
            raw_yaml=raw_yaml,
            dir_name=dir_name,
            path=path,
        )
        return Skill(
            name=name,
            description=description,
            path=skill_dir,
            model_invocable=model_invocable,
        )

    return _create


@dataclass(slots=True)
class _QueuedSkill:
    """Hold skill attributes for synthetic corpus builder rendering."""

    name: str
    description: str
    body: str
    model_invocable: bool
    path: Path | None
    text: str | None


class SyntheticCorpusBuilder:
    """Construct in-memory or on-disk synthetic skill corpora and catalogs."""

    def __init__(self) -> None:
        """Initialize empty skill queue."""
        self._queued: list[_QueuedSkill] = []

    def add(
        self,
        name: str,
        description: str,
        body: str = "",
        *,
        model_invocable: bool = True,
        path: Path | None = None,
        text: str | None = None,
    ) -> Self:
        """Enqueue a synthetic skill definition."""
        self._queued.append(
            _QueuedSkill(name, description, body, model_invocable, path, text),
        )
        return self

    def build_skills(self, root: Path | None = None) -> list[Skill]:
        """Return list of queued Skill models."""
        base = Path() if root is None else root
        return [
            Skill(
                name=q.name,
                description=q.description,
                path=q.path if q.path is not None else base / q.name,
                model_invocable=q.model_invocable,
            )
            for q in self._queued
        ]

    def build_disk(self, root: Path) -> Path:
        """Write all queued skills as SKILL.md files on disk under root directory."""
        for q in self._queued:
            directory = root / q.name if q.path is None else q.path
            directory.mkdir(parents=True, exist_ok=True)
            body = f"# {q.name}\n\n{q.body}".rstrip() if q.body else f"# {q.name}"
            text = (
                format_skill_markdown(name=q.name, description=q.description, body=body)
                if q.text is None
                else q.text
            )
            (directory / "SKILL.md").write_text(text, encoding="utf-8")
        return root

    def build_catalog(self, mode: CatalogMode = CatalogMode.ALL) -> Catalog:
        """Build catalog from all queued skills."""
        return build_catalogs(self.build_skills(), mode)[0]


@pytest.fixture
def corpus_builder() -> type[SyntheticCorpusBuilder]:
    """Return the SyntheticCorpusBuilder class as a test fixture."""
    return SyntheticCorpusBuilder


@pytest.fixture
def corpus(skill_repo: Path) -> list[Skill]:
    """Load skill models from synthetic skill repository."""
    return load_skills(skill_repo)


@pytest.fixture
def resident_names() -> tuple[str, ...]:
    """Provide a canonical resident skills name list for runtime parser tests."""
    return ("cloud-deploy", "pizza-calculator", "database-migrate")


@pytest.fixture
def synthetic_skills_repo(tmp_path: Path) -> Path:
    """Build a self-contained synthetic skill catalog for end-to-end CLI testing."""
    builder = SyntheticCorpusBuilder()
    builder.add(
        "cloud-run-basics",
        "Deploy and scale containerized web applications and microservices on Cloud Run.",
        body=(
            "Use Cloud Run to run stateless HTTP containers.\n"
            "Configure CPU, memory, concurrency limits, and environment variables.\n"
            "Integrate with Cloud Build for automatic continuous deployment."
        ),
    )
    builder.add(
        "cloud-sql-basics",
        "Manage relational databases using Cloud SQL including Postgres and MySQL.",
        body=(
            "Provision managed database instances, configure automated backups,\n"
            "and establish secure private IP connectivity for relational workloads."
        ),
    )
    builder.add(
        "gke-basics",
        "Deploy, manage, and scale containerized workloads on Google Kubernetes Engine.",
        body=(
            "Manage Kubernetes clusters, configure node pools, deployments, and pods.\n"
            "Monitor container resource utilization and cluster autoscaling."
        ),
    )
    builder.add(
        "gke-networking",
        "Configure GKE cluster networking, Gateway API, Ingress, and service routing.",
        body=(
            "Set up Gateway resources, HTTPRoute rules, load balancer attachments,\n"
            "and Private Service Connect for multi-cluster networking."
        ),
    )
    builder.add(
        "cloud-storage-basics",
        "Store and retrieve unstructured files and objects in Cloud Storage buckets.",
        body=(
            "Create buckets, manage object lifecycle rules, configure retention locks,\n"
            "and generate signed URLs for secure temporary file downloads."
        ),
    )
    builder.add(
        "cloud-storage-fuse",
        "Mount Cloud Storage buckets as local file systems using Cloud Storage FUSE.",
        body=(
            "Mount GCS buckets to local directory mount points on Linux and GKE nodes\n"
            "for POSIX-like file access to object storage."
        ),
    )
    root = tmp_path / "synthetic_skills"
    return builder.build_disk(root)


# ==============================================================================
# 4. Query & Dataset Fixtures
# ==============================================================================

_DEFAULT_QUERIES: tuple[Query, ...] = (
    Query(
        id="q-lifecycle",
        text="Tier old objects to Coldline after 30 days.",
        kind=QueryKind.IMPLICIT,
        expected_skill="gcs-lifecycle-rules",
    ),
    Query(
        id="q-retention",
        text="Keep audit logs for seven years for compliance.",
        kind=QueryKind.NEIGHBOR_NEGATIVE,
        expected_skill="gcs-retention-policy",
    ),
)


@pytest.fixture
def queries() -> list[Query]:
    """Return labeled queries covering implicit and neighbor negative cases."""
    return list(_DEFAULT_QUERIES)


@pytest.fixture
def query_file(
    tmp_path: Path,
    queries: list[Query],
    write_queries: Callable[..., Path],
) -> Path:
    """Write synthetic query set to temporary file and return path."""
    return write_queries(
        root=tmp_path,
        catalog_id="neighborhood:gcs-lifecycle-rules",
        queries=queries,
        filename="queries.json",
    )


_DEFAULT_EXCHANGE_QUERIES: tuple[Query, ...] = (
    Query(
        id="x-lifecycle",
        text="Tier old objects to Coldline after 30 days.",
        kind=QueryKind.IMPLICIT,
        expected_skill="gcs-lifecycle-rules",
        acceptable_skills=("finding-google-skills", "gcs-router"),
        notes="A neutral router may run before the target skill.",
    ),
    Query(
        id="x-unlabeled",
        text="Our cluster keeps evicting pods.",
        expected_skill="gke-basics",
    ),
    Query(
        id="x-abstain",
        text="What is the capital of France?",
        kind=QueryKind.OUT_OF_SCOPE,
    ),
    Query(
        id="x-punctuated",
        text='Delete "cold" objects, then archive\nwhatever is left.',
        kind=QueryKind.CONTEXTUAL,
        expected_skill="gcs-lifecycle-rules",
    ),
)


@pytest.fixture
def exchange_queries() -> tuple[Query, ...]:
    """Return representative query set covering all supported QueryKinds and edge cases."""
    return _DEFAULT_EXCHANGE_QUERIES


@pytest.fixture
def exchange_set(exchange_queries: tuple[Query, ...]) -> QuerySet:
    """Return QuerySet populated with exchange test queries."""
    return QuerySet(
        catalog_id="all",
        notes="written for the round trip",
        queries=exchange_queries,
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )


@pytest.fixture
def write_queries(tmp_path: Path) -> Callable[..., Path]:
    """Write a synthetic query dataset to disk and return its path."""

    def _create(
        target: str = "test-skill",
        count: int = 4,
        filename: str = "queries.json",
        root: Path | None = None,
        catalog_id: str = "test-cat",
        queries: Sequence[Query | dict[str, Any]] | None = None,
        notes: str = "",
        tool_version: str = "",
    ) -> Path:
        q_dir = root or tmp_path
        q_dir.mkdir(parents=True, exist_ok=True)
        path = q_dir / filename

        if queries is not None and queries and isinstance(queries[0], dict):
            payload = json.dumps(
                {
                    "catalog_id": catalog_id,
                    "queries": queries,
                    "provenance": {"origin": "authored"},
                },
            )
            path.write_text(payload, encoding="utf-8")
        else:
            q_list = (
                tuple(queries)  # type: ignore[arg-type]
                if queries is not None
                else tuple(
                    Query(
                        id=f"q-{i}",
                        text=f"Sample query {i} for {target}",
                        expected_skill=target,
                    )
                    for i in range(count)
                )
            )
            qs = QuerySet(
                catalog_id=catalog_id,
                notes=notes,
                queries=q_list,
                provenance=QuerySetProvenance(
                    origin=Origin.AUTHORED,
                    tool_version=tool_version,
                ),
            )
            save_query_set(qs, path)

        return path

    return _create


@pytest.fixture
def synthetic_query_file(tmp_path: Path, write_queries: Callable[..., Path]) -> Path:
    """Write a synthetic query set targeting skills in synthetic_skills_repo."""
    queries = (
        Query(
            id="q-run-1",
            text="How do I deploy a containerized service to Cloud Run?",
            expected_skill="cloud-run-basics",
        ),
        Query(
            id="q-run-2",
            text="Can I set concurrency limits on my Cloud Run service?",
            expected_skill="cloud-run-basics",
        ),
    )
    return write_queries(
        root=tmp_path,
        catalog_id="neighborhood:cloud-run-basics",
        notes="Self-contained synthetic query set",
        queries=queries,
        tool_version=metadata.version("skill-reach"),
        filename="synthetic_queries.json",
    )


@pytest.fixture
def synthetic_citations_file(synthetic_query_file: Path) -> Path:
    """Write a companion citations trail file for synthetic_query_file."""
    from reach.generate import Citation, CitationTrail, citations_path

    cpath = citations_path(synthetic_query_file)
    trail = CitationTrail(
        root=(
            Citation(
                skill="cloud-run-basics",
                text="How do I deploy a containerized service to Cloud Run?",
                citation="Use Cloud Run to run stateless HTTP containers.",
            ),
            Citation(
                skill="cloud-run-basics",
                text="Can I set concurrency limits on my Cloud Run service?",
                citation="Configure CPU, memory, concurrency limits, and environment variables.",
            ),
        ),
    )
    cpath.write_text(trail.model_dump_json(indent=2), encoding="utf-8")
    return cpath


# ==============================================================================
# 5. Runtime & Stream Simulation Fixtures
# ==============================================================================


def stream_lines(
    *,
    catalog: list[str] | None = None,
    invoked: str | None = None,
    cost: float | None = 0.21,
    duration_ms: int | None = 1911,
    tools: list[str] | None = None,
    model: str | None = "claude-opus-5",
) -> list[str]:
    """Format stream-json transcript lines matching runtime output format."""
    init: dict = {
        "type": "system",
        "subtype": "init",
        "skills": catalog or [],
        "tools": ["Skill"] if tools is None else tools,
    }
    if model is not None:
        init["model"] = model
    events: list[dict] = [init]
    if invoked is not None:
        events.append(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Skill",
                            "input": {"skill": invoked, "args": "..."},
                        },
                    ],
                },
            },
        )
    else:
        events.append(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "No skill needed."}]},
            },
        )
    events.append(
        {
            "type": "result",
            "subtype": "success",
            "total_cost_usd": cost,
            "duration_ms": duration_ms,
        },
    )
    return [json.dumps(e) for e in events]


@pytest.fixture
def make_stream() -> Callable[..., list[str]]:
    """Return factory function generating mock stream lines."""
    return stream_lines


@pytest.fixture
def real_stream() -> list[str]:
    """Return recorded real stream lines from test fixture file."""
    path = Path(__file__).parent / "fixtures" / "real_stream.jsonl"
    return path.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def make_runtime() -> type[FakeRuntime]:
    """Return FakeRuntime class for test runtime injection."""
    return FakeRuntime


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    """Provide a fresh FakeRuntime test fixture."""
    return FakeRuntime()


@pytest.fixture
def fake_generator() -> FakeGenerator:
    """Provide a fresh FakeGenerator test fixture."""
    return FakeGenerator()


@pytest.fixture
def answering_runtime(queries: list[Query]) -> FakeRuntime:
    """Return FakeRuntime pre-configured to answer expected skill for each query."""
    return FakeRuntime({q.text: q.expected_skill for q in queries})


@pytest.fixture
def mock_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Callable[..., subprocess.CompletedProcess[str]]]:
    """Mock subprocess.run with canned stdout, stderr, return codes, or handlers."""

    def _mock(
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        side_effect: Exception | None = None,
        handler: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        lines: Sequence[str] | None = None,
    ) -> Callable[..., subprocess.CompletedProcess[str]]:
        out = "\n".join(lines) if lines is not None else stdout

        def _run(cmd: Any, *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if side_effect is not None:
                raise side_effect
            if handler is not None:
                return handler(cmd, *args, **kwargs)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=returncode,
                stdout=out,
                stderr=stderr,
            )

        monkeypatch.setattr(subprocess, "run", _run)
        return _run

    return _mock


# ==============================================================================
# 6. Evaluation, Artifact & Metric Verification Fixtures
# ==============================================================================


@pytest.fixture
def make_result() -> Callable[..., ProbeResult]:
    """Return factory function creating ProbeResult models with default catalog parameters."""

    def _make(
        query_id: str,
        invoked: str | None,
        attempt: int = 1,
        error: str | None = None,
        fingerprint: str = "",
        condition: str = "",
        queries: str = "",
        reasoning: tuple[str, ...] = (),
        runtime: str = "fake",
    ) -> ProbeResult:
        return ProbeResult(
            query_id=query_id,
            catalog_id="fixture",
            catalog_mode=CatalogMode.ALL,
            catalog_size=4,
            model="sonnet",
            runtime=runtime,
            attempt=attempt,
            invoked_skills=(invoked,) if invoked is not None else (),
            error=error,
            config_fingerprint=fingerprint,
            condition_digest=condition,
            queries_digest=queries,
            reasoning=reasoning,
        )

    return _make


@pytest.fixture
def make_paired_results() -> Callable[
    ...,
    tuple[list[ProbeResult], list[ProbeResult], list[Query]],
]:
    """Provide a factory creating paired baseline and scaled ProbeResult sequences."""

    def _make_paired(
        both_pass: int = 0,
        both_fail: int = 0,
        ctx_loss: int = 0,
        shd_loss: int = 0,
        attempts: int = 1,
    ) -> tuple[list[ProbeResult], list[ProbeResult], list[Query]]:
        baseline_results: list[ProbeResult] = []
        scaled_results: list[ProbeResult] = []
        queries: list[Query] = []

        categories = (
            (
                both_pass,
                "pass",
                "Both pass query",
                ("skill-a",),
                InvocationPattern.ORACLE_ONLY,
                ("skill-a",),
                InvocationPattern.ORACLE_ONLY,
            ),
            (
                both_fail,
                "fail",
                "Both fail query",
                (),
                InvocationPattern.ABANDONED,
                (),
                InvocationPattern.ABANDONED,
            ),
            (
                ctx_loss,
                "ctx",
                "Context loss query",
                ("skill-a",),
                InvocationPattern.ORACLE_ONLY,
                (),
                InvocationPattern.ABANDONED,
            ),
            (
                shd_loss,
                "shd",
                "Shadowing loss query",
                ("skill-a",),
                InvocationPattern.ORACLE_ONLY,
                ("distractor",),
                InvocationPattern.DISTRACTOR_HIJACK,
            ),
        )

        for count, prefix, label, base_inv, base_pat, scaled_inv, scaled_pat in categories:
            for i in range(count):
                qid = f"q-{prefix}-{i}"
                queries.append(
                    Query(
                        id=qid,
                        text=f"{label} {i}",
                        kind=QueryKind.IMPLICIT,
                        expected_skill="skill-a",
                    ),
                )
                for a in range(1, attempts + 1):
                    baseline_results.append(
                        ProbeResult(
                            query_id=qid,
                            catalog_id="baseline",
                            catalog_mode=CatalogMode.SINGLETON,
                            catalog_size=1,
                            model="mock-model",
                            runtime="fake",
                            attempt=a,
                            invoked_skills=base_inv,
                            invocation_pattern=base_pat,
                        ),
                    )
                    scaled_results.append(
                        ProbeResult(
                            query_id=qid,
                            catalog_id="scaled",
                            catalog_mode=CatalogMode.SWEEP,
                            catalog_size=10,
                            model="mock-model",
                            runtime="fake",
                            attempt=a,
                            invoked_skills=scaled_inv,
                            invocation_pattern=scaled_pat,
                        ),
                    )

        return baseline_results, scaled_results, queries

    return _make_paired


@pytest.fixture
def whole_catalog(corpus: list[Skill]) -> Catalog:
    """Assemble all corpus skills into a single catalog."""
    return build_catalogs(corpus, CatalogMode.ALL)[0]


@pytest.fixture
def whole_catalog_queries(queries: list[Query]) -> QuerySet:
    """Return QuerySet labeled for whole catalog scope."""
    return QuerySet(
        catalog_id="all",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )


#: Synthetic prediction fixtures for whole catalog test verification.
WHOLE_CATALOG_PREDICTIONS = {
    "q-lifecycle": (
        "gcs-lifecycle-rules",
        "gcs-lifecycle-rules",
        "gcs-lifecycle-rules",
    ),
    "q-retention": ("gcs-retention-policy", "gke-basics", None),
}


@pytest.fixture
def whole_catalog_results(make_result) -> list[ProbeResult]:
    """Return synthetic ProbeResult list across whole catalog test queries."""
    return [
        make_result(query_id, invoked, attempt=i)
        for query_id, picks in WHOLE_CATALOG_PREDICTIONS.items()
        for i, invoked in enumerate(picks, start=1)
    ]


@pytest.fixture
def artifact(
    whole_catalog_results: list[ProbeResult],
    whole_catalog_queries: QuerySet,
    whole_catalog: Catalog,
    corpus: list[Skill],
    make_config,
) -> Artifact:
    """Return a fully populated Artifact built from whole catalog test fixtures."""
    return Artifact.assemble(
        Composition(
            config=make_config(catalog={"mode": CatalogMode.ALL}, plan={"attempts": 3}),
            query_set=whole_catalog_queries,
            catalog=whole_catalog,
            skills=tuple(corpus),
        ),
        whole_catalog_results,
    )


#: Ground truth query specifications for standard metric verification tests.
WORKED_QUERIES = (
    Query(
        id="wq-cost",
        text="Where is our spend going?",
        kind=QueryKind.IMPLICIT,
        expected_skill="waf-cost",
    ),
    Query(
        id="wq-sec",
        text="Harden our perimeter.",
        kind=QueryKind.NEIGHBOR_NEGATIVE,
        expected_skill="waf-security",
    ),
    Query(
        id="wq-rel",
        text="Survive a zonal outage.",
        kind=QueryKind.IMPLICIT,
        expected_skill="waf-reliability",
    ),
    Query(
        id="wq-oos",
        text="What is the capital of France?",
        kind=QueryKind.OUT_OF_SCOPE,
    ),
    Query(
        id="wq-sus",
        text="Cut the carbon footprint of these workloads.",
        kind=QueryKind.IMPLICIT,
        expected_skill="waf-sustainability",
    ),
)

#: Prediction mappings for worked metric verification tests.
WORKED_PREDICTIONS = {
    "wq-cost": ("waf-cost", "waf-cost"),
    "wq-sec": ("waf-cost", "waf-security"),
    "wq-rel": (None, "waf-reliability"),
    "wq-oos": (None, "waf-security"),
    "wq-sus": ("waf-security", "waf-security"),
}


@pytest.fixture
def worked_queries() -> tuple[Query, ...]:
    """Return standard worked query set fixture."""
    return WORKED_QUERIES


@pytest.fixture
def worked_results(make_result) -> list[ProbeResult]:
    """Return standard worked probe results fixture."""
    return [
        make_result(query_id, invoked, attempt=i)
        for query_id, picks in WORKED_PREDICTIONS.items()
        for i, invoked in enumerate(picks, start=1)
    ]


@pytest.fixture
def record_arm(
    tmp_path: Path,
) -> Callable[[str, RunConfig, Mapping[str, Sequence[str | None]]], Path]:
    """Return factory function recording probe results and sidecar files to disk."""

    def _record(
        name: str,
        config: RunConfig,
        predictions: Mapping[str, Sequence[str | None]],
    ) -> Path:
        composed = compose(config)
        catalog = composed.catalog
        provenance = composed.provenance
        out = tmp_path / f"{name}.jsonl"
        out.touch()
        for query_id, picks in predictions.items():
            for attempt, invoked in enumerate(picks, start=1):
                append_result(
                    out,
                    ProbeResult(
                        query_id=query_id,
                        catalog_id=catalog.id,
                        catalog_mode=catalog.mode,
                        catalog_size=catalog.size,
                        model=config.runtime.resolved_options().get("model", ""),
                        runtime=config.runtime.agent,
                        attempt=attempt,
                        invoked_skills=(invoked,) if invoked is not None else (),
                        **provenance.model_dump(),
                    ),
                )
        write_sidecar(config, out)
        return out

    return _record


@pytest.fixture(scope="session")
def matches_sklearn() -> Callable[..., None]:
    """Provide a helper verifying Reach metrics against scikit-learn implementations."""

    def _assert(results: Any, queries: Any, labels: Any = None) -> None:
        y_true, y_pred = labeled_pairs(results, queries)
        report = classification_report(results, queries, labels=labels)
        universe = list(labels) if labels else sorted(set(y_true) | set(y_pred))
        kwargs: dict[str, Any] = {
            "labels": universe,
            "average": "macro",
            "zero_division": 0,
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            accuracy = accuracy_score(y_true, y_pred)
            macro = (
                precision_score(y_true, y_pred, **kwargs),
                recall_score(y_true, y_pred, **kwargs),
                f1_score(y_true, y_pred, **kwargs),
            )
            precisions, recalls, f1s, supports = cast(
                "tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[int]]",
                precision_recall_fscore_support(
                    y_true,
                    y_pred,
                    labels=universe,
                    average=None,
                    zero_division=0,
                ),
            )
            matrix = confusion_matrix(y_true, y_pred, labels=universe)

        assert report.top1_accuracy == pytest.approx(accuracy)
        assert report.macro_precision == pytest.approx(macro[0])
        assert report.macro_recall == pytest.approx(macro[1])
        assert report.macro_f1 == pytest.approx(macro[2])

        for entry, p, r, f, s in zip(
            report.per_class,
            precisions,
            recalls,
            f1s,
            supports,
            strict=True,
        ):
            assert entry.precision == pytest.approx(p), entry.label
            assert entry.recall == pytest.approx(r), entry.label
            assert entry.f1 == pytest.approx(f), entry.label
            assert entry.support == s, entry.label

        ours = confusion(results, queries)
        for i, truth in enumerate(universe):
            for j, predicted in enumerate(universe):
                key = (truth, None if predicted == NO_SKILL else predicted)
                assert ours[key] == matrix[i][j], f"{truth} -> {predicted}"

    return _assert


@pytest.fixture(scope="session")
def bm25s_reference() -> Callable[
    [Sequence[Skill]],
    Callable[[Sequence[str]], list[float]],
]:
    """Provide a factory creating bm25s query scorers for a given skill corpus."""

    def _for_corpus(skills: Sequence[Skill]) -> Callable[[Sequence[str]], list[float]]:
        engine = bm25s.BM25(k1=K1, b=B, method="lucene")
        engine.index([tokenize(skill_text(s)) for s in skills])
        return lambda query: list(engine.get_scores(list(query)))

    return _for_corpus


# ==============================================================================
# 7. Configuration & Workspace Fixtures
# ==============================================================================


@pytest.fixture
def make_config(skill_repo: Path, query_file: Path, tmp_path: Path) -> Callable[..., RunConfig]:
    """Return factory function generating test RunConfig instances."""

    def _make(**overrides: Any) -> RunConfig:
        study = {
            "skills": skill_repo,
            "queries": query_file,
            "workdir": tmp_path / "work",
            "partial": True,
            **overrides.pop("study", {}),
        }
        runtime = {"agent": "fake", **overrides.pop("runtime", {})}
        catalog = {"size": 3, "rivals": 2, **overrides.pop("catalog", {})}
        plan = {"attempts": 1, "backoff_s": 0.0, **overrides.pop("plan", {})}
        return RunConfig.model_validate(
            {
                "study": study,
                "runtime": runtime,
                "catalog": catalog,
                "plan": plan,
                **overrides,
            },
        )

    return _make


@pytest.fixture
def write_reach_toml(tmp_path: Path) -> Callable[..., Path]:
    """Write a custom reach.toml configuration file in tmp_path or a target directory."""

    def _write(
        content: str,
        filename: str = "reach.toml",
        *,
        directory: Path | None = None,
    ) -> Path:
        target_dir = tmp_path if directory is None else directory
        path = target_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return path

    return _write


FAKE_REGISTRY_TOML = """\
[general]
default_agent = "mock-agent"

[agents.mock-agent]
default_model = "mock-model"
models = ["mock-model", "mock-model-variant", "mock-opus"]

[agents.mock-antigravity]
default_model = "mock-flash"
models = ["mock-flash", "mock-pro"]

[agents.mock-empty]
# Agent with no default model and no models list

[models.mock-model]
chars_per_token = 3.0
context_window = 100_000

[models.mock-flash]
chars_per_token = 4.0
context_window = 500_000
"""


@pytest.fixture
def fake_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an isolated, synthetic agent and model registry for invariant testing."""
    config_file = tmp_path / "reach.toml"
    config_file.write_text(FAKE_REGISTRY_TOML, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return config_file


@dataclass(frozen=True)
class IntegrationWorkspace:
    """Represent an isolated, hermetic skill-reach workspace with trivial base data."""

    root: Path
    skills_dir: Path
    queries_file: Path
    config_file: Path

    def run_reach(
        self,
        args: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Execute the reach CLI in a subprocess within the workspace directory."""
        reach_bin = Path(sys.executable).parent / "reach"
        if reach_bin.exists():
            cmd = [str(reach_bin), *args]
        elif shutil.which("reach"):
            cmd = ["reach", *args]
        else:
            cmd = [
                sys.executable,
                "-c",
                "from reach.cli import main; import sys; sys.exit(main(sys.argv[1:]))",
                *args,
            ]

        return subprocess.run(
            cmd,
            cwd=self.root,
            capture_output=True,
            text=True,
            env=dict(env) if env is not None else None,
            check=check,
        )


@pytest.fixture
def integration_workspace(tmp_path: Path) -> IntegrationWorkspace:
    """Construct an idempotent, isolated workspace containing minimal valid skills and queries."""
    root = tmp_path / "workspace"
    skills_dir = root / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    base_skills = (
        (
            "file-copier",
            "Copy and synchronize files and directories across local paths.",
            "# File Copier\nInstructions for copying files and directory structures.",
        ),
        (
            "file-compressor",
            "Compress and archive files into zip and tar formats.",
            "# File Compressor\nInstructions for compressing and archiving files.",
        ),
        (
            "file-deleter",
            "Securely remove and shred files and directories from storage.",
            "# File Deleter\nInstructions for deleting and shredding filesystem items.",
        ),
    )
    for name, desc, body in base_skills:
        skill_dir = skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            format_skill_markdown(
                name=name,
                description=desc,
                category="filesystem",
                body=body,
            ),
            encoding="utf-8",
        )

    # 2. Companion queries.json dataset
    queries_file = root / "queries.json"
    dataset = QuerySet(
        catalog_id="all",
        notes="Hermetic integration test queries",
        provenance=QuerySetProvenance(
            origin=Origin.AUTHORED,
            tool_version=metadata.version("skill-reach"),
        ),
        queries=(
            Query(
                id="q-copy",
                text="Please use file-copier to duplicate this directory",
                expected_skill="file-copier",
            ),
            Query(
                id="q-compress",
                text="Use file compressor to archive these documents",
                expected_skill="file-compressor",
            ),
        ),
    )
    save_query_set(dataset, queries_file)

    # 3. Hermetic reach.toml
    config_file = root / "reach.toml"
    config_file.write_text(
        "[general]\n"
        'default_agent = "keyword"\n\n'
        "[discovery]\n"
        'precedence = ["skills"]\n\n'
        "[check]\n"
        "strict = true\n"
        "budget = 10\n"
        "min_accuracy = 0.50\n"
        "min_recall = 0.50\n",
        encoding="utf-8",
    )

    return IntegrationWorkspace(
        root=root,
        skills_dir=skills_dir,
        queries_file=queries_file,
        config_file=config_file,
    )


@pytest.fixture
def empty_integration_workspace(tmp_path: Path) -> IntegrationWorkspace:
    """Construct an empty, isolated workspace without skills, queries, or config."""
    root = tmp_path / "empty_workspace"
    root.mkdir(parents=True, exist_ok=True)
    return IntegrationWorkspace(
        root=root,
        skills_dir=root / "skills",
        queries_file=root / "queries.json",
        config_file=root / "reach.toml",
    )


# ==============================================================================
# 8. Console & Rendering Fixtures
# ==============================================================================

#: Regular expression matching terminal escape sequences.
ESCAPES = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

#: Regular expression matching un-sandboxed external URLs or remote assets.
EXTERNAL_REFERENCE_PATTERN = re.compile(r"https?://|<link[ >]|<script[^>]+src=")


@pytest.fixture(scope="session")
def external_reference_re() -> re.Pattern[str]:
    """Provide regex pattern matching external network references and scripts."""
    return EXTERNAL_REFERENCE_PATTERN


@pytest.fixture
def make_console() -> Callable[..., tuple[object, io.StringIO]]:
    """Return factory function generating Console instances writing to StringIO buffers."""

    def _make(
        *,
        terminal: bool = True,
        quiet: bool = False,
        width: int = 100,
    ) -> tuple[object, io.StringIO]:
        buffer = io.StringIO()
        console = build_console(
            file=buffer,
            width=width,
            force_terminal=terminal,
            quiet=quiet,
        )
        return console, buffer

    return _make


@pytest.fixture(autouse=True)
def _standard_terminal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate test runs from ambient terminal types and color flags."""
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)


@pytest.fixture
def wide(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set terminal width environment variable to 200 columns for test consistency."""
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture
def rendered() -> Callable[[io.StringIO], str]:
    """Return helper function stripping ANSI escape sequences from buffer text."""

    def _rendered(buffer: io.StringIO) -> str:
        return ESCAPES.sub("", buffer.getvalue())

    return _rendered


# ==============================================================================
# 9. Remote Registry Mock Server Fixtures
# ==============================================================================


class MockRegistryHandler(BaseHTTPRequestHandler):
    """Handle mock Google Cloud Agent Registry REST API HTTP requests."""

    captured_headers: ClassVar[dict[str, str]] = {}
    captured_params: ClassVar[dict[str, list[str]]] = {}
    request_counts: ClassVar[dict[str, int]] = {}

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress standard HTTP server request logging."""

    def do_GET(self) -> None:
        """Process incoming GET requests for skills, pagination, retries, and errors."""
        MockRegistryHandler.captured_headers = dict(self.headers)
        path_key = self.path.split("?")[0]
        MockRegistryHandler.request_counts[path_key] = (
            MockRegistryHandler.request_counts.get(path_key, 0) + 1
        )

        auth = self.headers.get("Authorization", "")
        if auth == "Bearer invalid-token":
            self._send_json(401, {"error": {"message": "Invalid credentials", "code": 401}})
            return

        if "/forbidden" in self.path:
            self._send_json(403, {"error": {"message": "Permission denied", "code": 403}})
            return

        if "/disabled" in self.path:
            self._send_json(
                403,
                {
                    "error": {
                        "message": "Service disabled",
                        "code": 403,
                        "details": [{"reason": "SERVICE_DISABLED"}],
                    },
                },
            )
            return

        if "/notfound" in self.path:
            self._send_json(404, {"error": {"message": "Resource not found", "code": 404}})
            return

        if "/retry-test" in self.path and MockRegistryHandler.request_counts.get(path_key, 0) == 1:
            self._send_json(429, {"error": {"message": "Resource exhausted", "code": 429}})
            return

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        MockRegistryHandler.captured_params = params
        next_page = params.get("pageToken", [None])[0]

        data: dict[str, object]
        if next_page == "page-2":
            data = {
                "skills": [
                    {
                        "name": "projects/test-proj/locations/global/skills/cloud-sql",
                        "displayName": "cloud-sql",
                        "description": "Manage database instances with Cloud SQL.",
                        "state": "STATE_ACTIVE",
                    },
                ],
            }
        elif "/paginated" in self.path:
            data = {
                "skills": [
                    {
                        "name": "projects/test-proj/locations/global/skills/cloud-run",
                        "displayName": "cloud-run",
                        "description": "Deploy containerized services with Cloud Run.",
                        "state": "STATE_ACTIVE",
                    },
                ],
                "nextPageToken": "page-2",
            }
        else:
            data = {
                "skills": [
                    {
                        "name": "projects/test-proj/locations/global/skills/cloud-storage",
                        "displayName": "cloud-storage",
                        "description": "Store files and objects in Cloud Storage.",
                        "state": "STATE_ACTIVE",
                        "skillId": "urn:skill:cloud.google.com:storage:basics",
                    },
                ],
            }

        self._send_json(200, data)

    def _send_json(self, status: int, payload: object) -> None:
        """Serialize and send JSON response."""
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


@pytest.fixture
def local_registry_server() -> Generator[str]:
    """Provide a hermetic local HTTP server serving canned Agent Registry REST responses."""
    MockRegistryHandler.captured_headers.clear()
    MockRegistryHandler.captured_params.clear()
    MockRegistryHandler.request_counts.clear()

    server = HTTPServer(("127.0.0.1", 0), MockRegistryHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def registry_handler() -> type[MockRegistryHandler]:
    """Return the MockRegistryHandler class for inspecting captured requests."""
    return MockRegistryHandler
