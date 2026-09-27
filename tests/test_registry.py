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

"""Verify Google Cloud Agent Registry client, authentication, caching, and error mapping."""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path
from typing import override
from unittest.mock import MagicMock, patch

import pytest

from reach.registry import (
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RegistryCacheManager,
    RegistryClient,
    RegistryError,
    RegistryManifest,
    RegistrySkillData,
    ServiceDisabledError,
    _matches_publisher,
    _safe_resolve_subpath,
    find_adc_path,
    get_access_token,
    is_adc_available,
)


def test_registry_skill_data_properties() -> None:
    """Verify RegistrySkillData parses resource names, display names, and revision slugs."""
    data = RegistrySkillData(
        name="projects/test-proj/locations/global/skills/gke-inference",
        displayName="gke-inference",
        description="Deploy inference on GKE",
        defaultRevision="projects/test-proj/locations/global/skills/gke-inference/revisions/rev-12345",
        skillId="urn:skill:cloud.google.com:container:gke-inference",
        publisher="projects/test-proj/locations/global/publishers/cloud.google.com",
    )
    assert data.identifier == "gke-inference"
    assert data.revision_slug == "rev-12345"


def test_registry_skill_data_fallback_identifier() -> None:
    """Verify identifier falls back to trailing segment of resource name if displayName is empty."""
    data = RegistrySkillData(
        name="projects/test-proj/locations/global/skills/cloud.google.com-gemini-api",
        description="Gemini API skill",
    )
    assert data.identifier == "cloud.google.com-gemini-api"
    assert data.revision_slug == "default"


def test_registry_skill_data_allows_snake_case_kwargs() -> None:
    """Verify RegistrySkillData can be instantiated with Python snake_case keyword arguments."""
    data = RegistrySkillData(
        name="projects/test-proj/locations/global/skills/gke-inference",
        display_name="gke-inference",
        description="Deploy inference on GKE",
        default_revision="projects/test-proj/locations/global/skills/gke-inference/revisions/rev-12345",
        skill_id="urn:skill:cloud.google.com:container:gke-inference",
        target_state="TARGET_STATE_ACTIVE",
    )
    assert data.identifier == "gke-inference"
    assert data.revision_slug == "rev-12345"
    assert data.target_state == "TARGET_STATE_ACTIVE"


def test_client_list_skills_pagination() -> None:
    """Verify RegistryClient handles multi-page responses using nextPageToken."""
    client = RegistryClient(token="mock-token")  # noqa: S106

    page1 = {
        "skills": [
            {
                "name": "projects/p/locations/global/skills/s1",
                "displayName": "skill-1",
                "description": "First skill",
                "publisher": "projects/p/locations/global/publishers/cloud.google.com",
            }
        ],
        "nextPageToken": "token-page-2",
    }
    page2 = {
        "skills": [
            {
                "name": "projects/p/locations/global/skills/s2",
                "displayName": "skill-2",
                "description": "Second skill",
                "publisher": (
                    "projects/p/locations/global/publishers/discoveryengine.googleapis.com"
                ),
            }
        ],
    }

    def mock_urlopen(req: object) -> MagicMock:
        url = getattr(req, "full_url", "")
        body = json.dumps(page2 if "token-page-2" in url else page1).encode("utf-8")
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        skills = client.list_skills(project="p", location="global")
        assert len(skills) == 2
        assert skills[0].identifier == "skill-1"
        assert skills[1].identifier == "skill-2"


def test_client_publisher_filter() -> None:
    """Verify list_skills filters results by publisher."""
    client = RegistryClient(token="mock-token")  # noqa: S106
    payload = {
        "skills": [
            {
                "name": "projects/p/locations/global/skills/s1",
                "displayName": "skill-1",
                "description": "Cloud skill",
                "publisher": "projects/p/locations/global/publishers/cloud.google.com",
            },
            {
                "name": "projects/p/locations/global/skills/s2",
                "displayName": "skill-2",
                "description": "Discovery skill",
                "publisher": (
                    "projects/p/locations/global/publishers/discoveryengine.googleapis.com"
                ),
            },
        ],
    }

    def mock_urlopen(_req: object) -> MagicMock:
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode("utf-8")
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        cloud_skills = client.list_skills(
            project="p", location="global", publisher="cloud.google.com"
        )
        assert len(cloud_skills) == 1
        assert cloud_skills[0].identifier == "skill-1"


@pytest.mark.parametrize(
    ("status_code", "error_payload", "expected_exc"),
    [
        (401, '{"error": {"message": "Invalid token"}}', AuthenticationError),
        (
            403,
            '{"error": {"message": "API disabled", "details": [{"reason": "SERVICE_DISABLED"}]}}',
            ServiceDisabledError,
        ),
        (403, '{"error": {"message": "Permission denied"}}', PermissionDeniedError),
        (404, '{"error": {"message": "Project not found"}}', NotFoundError),
    ],
)
def test_client_error_mapping(
    status_code: int, error_payload: str, expected_exc: type[RegistryError]
) -> None:
    """Verify HTTP status codes map to domain exceptions with actionable context."""
    client = RegistryClient(token="mock-token")  # noqa: S106

    def mock_urlopen(_req: object) -> None:
        fp = io.BytesIO(error_payload.encode("utf-8"))
        raise urllib.error.HTTPError(
            url="https://agentregistry.googleapis.com",
            code=status_code,
            msg="Error",
            hdrs=Message(),
            fp=fp,
        )

    with (
        patch("urllib.request.urlopen", side_effect=mock_urlopen),
        pytest.raises(expected_exc),
    ):
        client.list_skills(project="p")


def test_client_rate_limit_retry() -> None:
    """Verify HTTP 429 triggers retry with backoff."""
    client = RegistryClient(token="mock-token", max_retries=2, backoff_factor=0.01)  # noqa: S106
    attempts = 0

    def mock_urlopen(_req: object) -> MagicMock:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            fp = io.BytesIO(b'{"error": {"message": "Rate limited"}}')
            raise urllib.error.HTTPError(
                url="https://agentregistry.googleapis.com",
                code=429,
                msg="Too Many Requests",
                hdrs=Message(),
                fp=fp,
            )
        resp = MagicMock()
        resp.read.return_value = b'{"skills": []}'
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        skills = client.list_skills(project="p")
        assert attempts == 2
        assert skills == []


def test_cache_manager_manifest_ttl(tmp_path: Path) -> None:
    """Verify cache manager respects TTL on cached manifest."""
    cache = RegistryCacheManager(cache_root=tmp_path)

    # 1. Fresh manifest
    fresh_manifest = RegistryManifest(
        project="proj-a",
        location="global",
        fetched_at=datetime.now(UTC),
        skills=(
            RegistrySkillData(
                name="projects/proj-a/locations/global/skills/s1",
                displayName="s1",
                description="desc 1",
            ),
        ),
    )
    cache.save_manifest(fresh_manifest)

    loaded = cache.get_cached_manifest("proj-a", "global", max_age_seconds=60)
    assert loaded is not None
    assert len(loaded.skills) == 1

    # 2. Expired manifest
    old_manifest = RegistryManifest(
        project="proj-a",
        location="global",
        fetched_at=datetime.now(UTC) - timedelta(seconds=120),
        skills=(
            RegistrySkillData(
                name="projects/proj-a/locations/global/skills/s1",
                displayName="s1",
                description="desc 1",
            ),
        ),
    )
    cache.save_manifest(old_manifest)

    expired = cache.get_cached_manifest("proj-a", "global", max_age_seconds=60)
    assert expired is None

    # Stale fallback (max_age_seconds < 0)
    stale = cache.get_cached_manifest("proj-a", "global", max_age_seconds=-1)
    assert stale is not None


@pytest.mark.parametrize(
    ("actual", "requested", "expected"),
    [
        (None, None, True),
        ("google", None, True),
        (None, "google", False),
        ("google", "google", True),
        ("projects/p/locations/l/publishers/google", "google", True),
        ("google", "projects/p/locations/l/publishers/google", True),
        ("publishers/google/", "google", True),
        ("google", "publishers/google", True),
        ("my-google", "google", False),
        ("google", "my-google", False),
        ("google", "other", False),
    ],
)
def test_matches_publisher_cases(actual: str | None, requested: str | None, expected: bool) -> None:
    """Verify publisher matching supports bidirectional and normalized identifiers."""
    assert _matches_publisher(actual, requested) is expected


def test_get_cached_manifest_filters_unscoped_fallback_by_publisher(tmp_path: Path) -> None:
    """Verify unscoped fallback manifest is filtered by publisher without leaking others."""
    cache = RegistryCacheManager(cache_root=tmp_path)
    unscoped_manifest = RegistryManifest(
        project="proj-a",
        location="global",
        publisher=None,
        fetched_at=datetime.now(UTC),
        skills=(
            RegistrySkillData(
                name="projects/proj-a/locations/global/skills/s1",
                displayName="s1",
                description="desc 1",
                publisher="google",
            ),
            RegistrySkillData(
                name="projects/proj-a/locations/global/skills/s2",
                displayName="s2",
                description="desc 2",
                publisher="acme-corp",
            ),
        ),
    )
    cache.save_manifest(unscoped_manifest)

    # Scoped query should hit unscoped fallback but ONLY return matching skills
    loaded = cache.get_cached_manifest("proj-a", "global", publisher="google", max_age_seconds=60)
    assert loaded is not None
    assert loaded.publisher == "google"
    assert len(loaded.skills) == 1
    assert loaded.skills[0].display_name == "s1"
    assert loaded.skills[0].publisher == "google"


def test_cache_manager_hydration_and_resolve(tmp_path: Path) -> None:
    """Verify cache manager creates physical SKILL.md and returns valid Skill models."""
    cache = RegistryCacheManager(cache_root=tmp_path)
    client = MagicMock()
    skills_data = (
        RegistrySkillData(
            name="projects/proj-x/locations/global/skills/my-skill",
            displayName="my-skill",
            description="My awesome skill",
            defaultRevision="projects/proj-x/locations/global/skills/my-skill/revisions/rev-1",
            skillId="urn:skill:cloud.google.com:test:my-skill",
            publisher="projects/proj-x/locations/global/publishers/cloud.google.com",
            state="STATE_ACTIVE",
        ),
    )
    client.fetch_manifest.return_value = RegistryManifest(
        project="proj-x",
        location="global",
        fetched_at=datetime.now(UTC),
        skills=skills_data,
    )

    skills = cache.resolve_skills("proj-x", "global", client=client)
    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "my-skill"
    assert skill.description == "My awesome skill"
    assert skill.manifest_source == "agent-registry://proj-x/global"
    assert skill.model_invocable is True
    assert (skill.path / "SKILL.md").is_file()

    # Verify SKILL.md contents
    content = (skill.path / "SKILL.md").read_text()
    assert "name: my-skill" in content
    assert "My awesome skill" in content
    assert "urn:skill:cloud.google.com:test:my-skill" in content


def test_cache_manager_clean(tmp_path: Path) -> None:
    """Verify clean calculates sizes and purges cache directories."""
    cache = RegistryCacheManager(cache_root=tmp_path)
    proj_dir = cache.location_dir("proj-y", "global")
    proj_dir.mkdir(parents=True, exist_ok=True)
    test_file = proj_dir / "sample.txt"
    test_file.write_text("hello world")

    # Dry run
    bytes_reclaimed, paths = cache.clean(dry_run=True)
    assert bytes_reclaimed > 0
    assert len(paths) == 1
    assert test_file.exists()

    # Actual clean
    bytes_cleaned, paths_cleaned = cache.clean(dry_run=False)
    assert bytes_cleaned > 0
    assert paths_cleaned
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "malicious_project",
    [
        "/etc",
        "/var/target",
        "../../escape",
        "..",
        ".",
        "nested/path",
        "nested\\winpath",
    ],
)
def test_cache_manager_clean_rejects_path_traversal(
    malicious_project: str,
    tmp_path: Path,
) -> None:
    """Verify clean raises ValueError and refuses to delete paths that attempt directory escape."""
    cache = RegistryCacheManager(cache_root=tmp_path / "cache")
    sensitive_outside_dir = tmp_path / "outside"
    sensitive_outside_dir.mkdir()
    canary = sensitive_outside_dir / "canary.txt"
    canary.write_text("protected")

    with pytest.raises(ValueError, match="escapes cache directory"):
        cache.clean(project=malicious_project)

    assert canary.is_file()
    assert canary.read_text() == "protected"


@pytest.mark.parametrize("bad_id", ["..", ".", "  "])
def test_cache_manager_location_dir_rejects_empty_or_dot(bad_id: str, tmp_path: Path) -> None:
    """Verify location_dir raises ValueError when project or location resolves to dot."""
    cache = RegistryCacheManager(cache_root=tmp_path)
    with pytest.raises(ValueError, match="Invalid project identifier"):
        cache.location_dir(bad_id, "global")
    with pytest.raises(ValueError, match="Invalid location identifier"):
        cache.location_dir("proj", bad_id)


def test_cache_manager_skill_dir_sanitizes_path_traversal(tmp_path: Path) -> None:
    """Verify skill_dir sanitizes remote skill names and ensures directory is within location."""
    cache = RegistryCacheManager(cache_root=tmp_path)
    loc_dir = cache.location_dir("proj", "global")
    skill_path = cache.skill_dir("proj", "global", "../../malicious_name", "../../rev")
    assert skill_path.is_relative_to(loc_dir)
    assert not any(part == ".." for part in skill_path.parts)


@pytest.mark.parametrize(
    ("segments", "disallow_separators", "fallbacks", "expected_rel"),
    [
        (
            [("my-proj", "project"), ("us-central1", "location")],
            False,
            None,
            "my-proj/us-central1",
        ),
        (
            [("skill@1", "skill"), ("..", "revision")],
            False,
            {"revision": "default"},
            "skill_1/default",
        ),
    ],
)
def test_safe_resolve_subpath_valid(
    tmp_path: Path,
    segments: list[tuple[str, str]],
    disallow_separators: bool,
    fallbacks: dict[str, str] | None,
    expected_rel: str,
) -> None:
    """Verify _safe_resolve_subpath resolves valid components strictly within root."""
    resolved = _safe_resolve_subpath(
        tmp_path,
        *segments,
        disallow_separators=disallow_separators,
        fallbacks=fallbacks,
    )
    assert resolved == (tmp_path / expected_rel).resolve()
    assert resolved.is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize(
    ("segments", "disallow_separators", "error_match"),
    [
        ([("/etc/passwd", "project")], True, "escapes cache directory"),
        ([("nested/dir", "project")], True, "escapes cache directory"),
        ([("..", "project")], False, "Invalid project identifier"),
        ([("   ", "location")], False, "Invalid location identifier"),
    ],
)
def test_safe_resolve_subpath_invalid(
    tmp_path: Path,
    segments: list[tuple[str, str]],
    disallow_separators: bool,
    error_match: str,
) -> None:
    """Verify _safe_resolve_subpath raises ValueError on traversal or invalid segments."""
    with pytest.raises(ValueError, match=error_match):
        _safe_resolve_subpath(
            tmp_path,
            *segments,
            disallow_separators=disallow_separators,
        )


def test_client_get_skill() -> None:
    """Verify get_skill fetches single skill resource."""
    client = RegistryClient(token="mock-token")  # noqa: S106
    payload = {
        "name": "projects/p/locations/global/skills/s1",
        "displayName": "skill-1",
        "description": "Specific skill",
    }

    def mock_urlopen(_req: object) -> MagicMock:
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode("utf-8")
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        skill = client.get_skill("p", "global", "s1")
        assert skill.identifier == "skill-1"
        assert skill.description == "Specific skill"


def test_client_fetch_manifest() -> None:
    """Verify fetch_manifest compiles complete RegistryManifest snapshot."""
    client = RegistryClient(token="mock-token")  # noqa: S106
    payload = {
        "skills": [
            {
                "name": "projects/p/locations/global/skills/s1",
                "displayName": "skill-1",
                "description": "Skill 1",
            }
        ]
    }

    def mock_urlopen(_req: object) -> MagicMock:
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode("utf-8")
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        manifest = client.fetch_manifest("p", "global")
        assert manifest.project == "p"
        assert manifest.location == "global"
        assert len(manifest.skills) == 1
        assert manifest.skills[0].identifier == "skill-1"


def test_corrupted_cache_recovery(tmp_path: Path) -> None:
    """Verify corrupted .manifest.json falls back to re-fetching without throwing."""
    cache = RegistryCacheManager(cache_root=tmp_path)
    man_path = cache.manifest_path("proj-corrupt", "global")
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text("{ broken json malformed", encoding="utf-8")

    client = MagicMock()
    client.fetch_manifest.return_value = RegistryManifest(
        project="proj-corrupt",
        location="global",
        fetched_at=datetime.now(UTC),
        skills=(
            RegistrySkillData(
                name="projects/proj-corrupt/locations/global/skills/s1",
                displayName="s1",
                description="Recovered skill",
            ),
        ),
    )

    skills = cache.resolve_skills("proj-corrupt", "global", client=client)
    assert len(skills) == 1
    assert skills[0].name == "s1"
    assert client.fetch_manifest.called


def test_empty_registry_handling(tmp_path: Path) -> None:
    """Verify empty remote registry returns empty list without error."""
    cache = RegistryCacheManager(cache_root=tmp_path)
    client = MagicMock()
    client.fetch_manifest.return_value = RegistryManifest(
        project="proj-empty",
        location="global",
        fetched_at=datetime.now(UTC),
        skills=(),
    )

    skills = cache.resolve_skills("proj-empty", "global", client=client)
    assert skills == []


def test_resolve_skills_custom_ttl(tmp_path: Path) -> None:
    """Verify custom max_age_seconds is forwarded and respected."""
    cache = RegistryCacheManager(cache_root=tmp_path)
    # Save a manifest 10 seconds old
    old_time = datetime.now(UTC) - timedelta(seconds=10)
    manifest = RegistryManifest(
        project="proj-ttl",
        location="global",
        fetched_at=old_time,
        skills=(
            RegistrySkillData(
                name="projects/proj-ttl/locations/global/skills/s1",
                displayName="s1",
                description="Skill 1",
            ),
        ),
    )
    cache.save_manifest(manifest)

    client = MagicMock()
    # With TTL 5s, the 10s manifest is expired -> should fetch
    cache.resolve_skills("proj-ttl", "global", client=client, cache_ttl_seconds=5)
    assert client.fetch_manifest.called


def test_atomic_write_text_creates_unique_temp_file(tmp_path: Path) -> None:
    """Verify atomic_write_text uses process- and thread-unique temporary file suffixes."""
    import re

    from reach._io import atomic_write_text

    target = tmp_path / "subdir" / "manifest.json"
    written = atomic_write_text(target, '{"test": true}\n')
    assert written == target
    assert target.read_text(encoding="utf-8") == '{"test": true}\n'

    # Verify temp path suffix format when mocked
    recorded_temp_paths: list[Path] = []
    orig_replace = Path.replace

    def mock_replace(self: Path, dest: Path) -> Path:
        recorded_temp_paths.append(self)
        return orig_replace(self, dest)

    with patch.object(Path, "replace", mock_replace):
        atomic_write_text(target, '{"updated": true}\n')

    assert len(recorded_temp_paths) == 1
    temp_name = recorded_temp_paths[0].name
    # Must match .tmp.<pid>_<uuid_hex>
    assert re.search(r"\.tmp\.\d+_[0-9a-f]{32}$", temp_name) is not None


def test_registry_cache_concurrent_save_manifest_and_hydrate(tmp_path: Path) -> None:
    """Verify concurrent threads saving manifests and hydrating skills do not collide."""
    import concurrent.futures

    cache = RegistryCacheManager(cache_root=tmp_path)
    manifest = RegistryManifest(
        project="concurrent-proj",
        location="global",
        fetched_at=datetime.now(UTC),
        skills=(
            RegistrySkillData(
                name="projects/concurrent-proj/locations/global/skills/shared-skill",
                displayName="shared-skill",
                description="A shared skill across threads",
                defaultRevision=(
                    "projects/concurrent-proj/locations/global/skills/shared-skill/revisions/rev-abc"
                ),
            ),
        ),
    )
    skill_data = manifest.skills[0]

    def worker(_idx: int) -> None:
        # Concurrent saves
        cache.save_manifest(manifest)
        # Concurrent hydration
        hydrated_dir = cache.hydrate_skill_file("concurrent-proj", "global", skill_data)
        assert hydrated_dir.is_dir()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    loaded = cache.get_cached_manifest("concurrent-proj", "global")
    assert loaded is not None
    assert len(loaded.skills) == 1
    assert loaded.skills[0].identifier == "shared-skill"


def test_registry_cache_publisher_scoping_prevents_cache_poisoning(tmp_path: Path) -> None:
    """Verify publisher-filtered fetches do not poison subsequent unfiltered resolution calls."""
    all_skills = (
        RegistrySkillData(
            name="projects/p/skills/a",
            displayName="alpha",
            publisher="pub-a",
            description="Alpha skill.",
        ),
        RegistrySkillData(
            name="projects/p/skills/b",
            displayName="bravo",
            publisher="pub-b",
            description="Bravo skill.",
        ),
    )

    class FakeClient(RegistryClient):
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        @override
        def fetch_manifest(
            self, project: str, location: str = "global", publisher: str | None = None
        ) -> RegistryManifest:
            self.calls.append(publisher)
            selected = tuple(s for s in all_skills if not publisher or s.publisher == publisher)
            return RegistryManifest(
                project=project,
                location=location,
                publisher=publisher,
                fetched_at=datetime.now(UTC),
                skills=selected,
            )

    cache = RegistryCacheManager(cache_root=tmp_path)
    client = FakeClient()

    # Scoped call caches only pub-a
    scoped_skills = cache.resolve_skills(project="p", publisher="pub-a", client=client)
    assert [s.name for s in scoped_skills] == ["alpha"]
    assert client.calls == ["pub-a"]

    # Unscoped call must NOT return only the cached pub-a subset
    unscoped_skills = cache.resolve_skills(project="p", client=client)
    assert sorted(s.name for s in unscoped_skills) == ["alpha", "bravo"]
    assert client.calls == ["pub-a", None]


def test_registry_cache_global_manifest_satisfies_subsequent_filtered_requests(
    tmp_path: Path,
) -> None:
    """Verify fresh unfiltered manifest satisfies publisher-filtered queries without re-fetching."""
    all_skills = (
        RegistrySkillData(
            name="projects/p/skills/a",
            displayName="alpha",
            publisher="pub-a",
            description="Alpha skill.",
        ),
        RegistrySkillData(
            name="projects/p/skills/b",
            displayName="bravo",
            publisher="pub-b",
            description="Bravo skill.",
        ),
    )

    class FakeClient(RegistryClient):
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        @override
        def fetch_manifest(
            self, project: str, location: str = "global", publisher: str | None = None
        ) -> RegistryManifest:
            self.calls.append(publisher)
            selected = tuple(s for s in all_skills if not publisher or s.publisher == publisher)
            return RegistryManifest(
                project=project,
                location=location,
                publisher=publisher,
                fetched_at=datetime.now(UTC),
                skills=selected,
            )

    cache = RegistryCacheManager(cache_root=tmp_path)
    client = FakeClient()

    # Fetch global/unscoped first
    unscoped_skills = cache.resolve_skills(project="p", client=client)
    assert len(unscoped_skills) == 2
    assert client.calls == [None]

    # Filtered call should use cached global manifest (0 extra network calls)
    scoped_skills = cache.resolve_skills(project="p", publisher="pub-a", client=client)
    assert [s.name for s in scoped_skills] == ["alpha"]
    assert client.calls == [None]


def test_registry_cache_stale_fallback_respects_publisher_filter(tmp_path: Path) -> None:
    """Verify stale fallback with max_age=-1 selects only compatible cached manifests."""
    cache = RegistryCacheManager(cache_root=tmp_path)

    # Save a cached manifest specifically for pub-a
    manifest_a = RegistryManifest(
        project="p",
        location="global",
        publisher="pub-a",
        fetched_at=datetime.now(UTC) - timedelta(days=10),
        skills=(
            RegistrySkillData(
                name="projects/p/skills/a",
                displayName="alpha",
                publisher="pub-a",
                description="Alpha skill.",
            ),
        ),
    )
    cache.save_manifest(manifest_a)

    class FailingClient(RegistryClient):
        @override
        def fetch_manifest(
            self,
            project: str,
            location: str = "global",
            publisher: str | None = None,
        ) -> RegistryManifest:
            del project, location, publisher
            msg = "Network unavailable"
            raise RegistryError(msg)

    failing_client = FailingClient()

    # Querying for pub-b must NOT fallback to stale pub-a manifest (must raise RegistryError)
    with pytest.raises(RegistryError, match="Network unavailable"):
        cache.resolve_skills(project="p", publisher="pub-b", client=failing_client)

    # Querying for pub-a should successfully fall back to stale pub-a manifest
    resolved_a = cache.resolve_skills(project="p", publisher="pub-a", client=failing_client)
    assert [s.name for s in resolved_a] == ["alpha"]

    # Querying for publisher="pub" (substring of pub-a) must NOT match pub-a
    with pytest.raises(RegistryError, match="Network unavailable"):
        cache.resolve_skills(project="p", publisher="pub", client=failing_client)


def test_registry_cache_publisher_slug_sanitization(tmp_path: Path) -> None:
    """Verify publisher names with slashes are sanitized into flat filename slugs."""
    cache = RegistryCacheManager(cache_root=tmp_path)
    path = cache.manifest_path("my-proj", "global", publisher="publishers/google")
    assert path.name == ".manifest.publishers_google.json"
    assert path.parent == cache.location_dir("my-proj", "global")


def test_find_adc_path_respects_cloudsdk_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify find_adc_path respects CLOUDSDK_CONFIG directory setting."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    cfg_dir = tmp_path / "gcloud_custom"
    cfg_dir.mkdir(parents=True)
    adc = cfg_dir / "application_default_credentials.json"
    adc.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(cfg_dir))

    found = find_adc_path()
    assert found == adc
    assert is_adc_available()


def test_find_adc_path_finds_windows_appdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify find_adc_path locates ADC file in Windows APPDATA environment directory."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake_home")
    appdata = tmp_path / "AppData" / "Roaming"
    gcloud = appdata / "gcloud"
    gcloud.mkdir(parents=True)
    adc = gcloud / "application_default_credentials.json"
    adc.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))

    found = find_adc_path()
    assert found == adc
    assert is_adc_available()


def test_find_adc_path_none_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify find_adc_path returns None when no ADC files exist."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake_home")

    assert find_adc_path() is None
    assert not is_adc_available()


def test_get_access_token_uses_cloud_platform_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_access_token requests the cloud-platform scope from google-auth."""
    import sys
    import types
    from typing import Any

    mock_creds = MagicMock()
    mock_creds.token = "ya29.mock"  # noqa: S105
    mock_default = MagicMock(return_value=(mock_creds, "mock-proj"))

    fake_google: Any = types.ModuleType("google")
    fake_auth: Any = types.ModuleType("google.auth")
    fake_auth.default = mock_default
    fake_transport: Any = types.ModuleType("google.auth.transport")
    fake_requests: Any = types.ModuleType("google.auth.transport.requests")
    fake_requests.Request = MagicMock()
    fake_google.auth = fake_auth
    fake_auth.transport = fake_transport
    fake_transport.requests = fake_requests

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.auth", fake_auth)
    monkeypatch.setitem(sys.modules, "google.auth.transport", fake_transport)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", fake_requests)

    token = get_access_token()
    assert token == "ya29.mock"  # noqa: S105
    mock_default.assert_called_once_with(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def test_get_access_token_fails_if_gcloud_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_access_token raises AuthenticationError when gcloud CLI is not in PATH."""
    monkeypatch.setattr("shutil.which", lambda _cmd: None)

    # Ensure google.auth throws so it falls back to gcloud
    with (
        patch("google.auth.default", side_effect=Exception("No google-auth")),
        pytest.raises(AuthenticationError, match="gcloud CLI not found in PATH"),
    ):
        get_access_token()


def test_get_access_token_parses_multiline_stdout_with_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify get_access_token extracts the ya29 token line amidst CLI warnings."""
    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/gcloud")

    multiline_stdout = (
        "WARNING: Your active project does not match your quota project.\n"
        "WARNING: A new gcloud release is available.\n"
        "\n"
        "ya29.c.b0Aaekm12345_mocked_token\n"
    )

    fake_proc = MagicMock(stdout=multiline_stdout, returncode=0)
    with (
        patch("google.auth.default", side_effect=Exception("No google-auth")),
        patch("subprocess.run", return_value=fake_proc),
    ):
        token = get_access_token()
        assert token == "ya29.c.b0Aaekm12345_mocked_token"  # noqa: S105


def test_get_access_token_accepts_non_ya29_token_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify get_access_token accepts valid bearer token formats without hardcoded ya29 prefix."""
    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/gcloud")
    multiline_stdout = (
        "WARNING: A new gcloud release is available.\n"
        "\n"
        "eyJhbGciOiJSUzI1NiIsImtpZCI6IjEyMzQ1In0.payload.signature\n"
    )
    fake_proc = MagicMock(stdout=multiline_stdout, returncode=0)
    with (
        patch("google.auth.default", side_effect=Exception("No google-auth")),
        patch("subprocess.run", return_value=fake_proc),
    ):
        token = get_access_token()
        assert token == "eyJhbGciOiJSUzI1NiIsImtpZCI6IjEyMzQ1In0.payload.signature"  # noqa: S105


@pytest.mark.parametrize(
    "bad_stdout",
    [
        "",
        "   \n  \n",
        "ERROR: (gcloud.auth.application-default.print-access-token) Credentials expired.",
        "WARNING: Just warnings without a token\nAnother warning line",
        "ya29.token has whitespace in the middle",
    ],
)
def test_get_access_token_rejects_malformed_token_output(
    bad_stdout: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify get_access_token raises AuthenticationError if output lacks a ya29 token."""
    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/gcloud")
    fake_proc = MagicMock(stdout=bad_stdout, returncode=0)

    with (
        patch("google.auth.default", side_effect=Exception("No google-auth")),
        patch("subprocess.run", return_value=fake_proc),
        pytest.raises(AuthenticationError, match="Unexpected or invalid token format"),
    ):
        get_access_token()


def test_registry_client_caches_token_across_requests() -> None:
    """Verify RegistryClient reuses cached token across calls to prevent process churn."""
    client = RegistryClient(base_url="http://127.0.0.1:8080")

    with patch("reach.registry.get_access_token", return_value="ya29.cached_token") as mock_get:
        tok1 = client._get_token()
        tok2 = client._get_token()
        tok3 = client._get_token()

        assert tok1 == "ya29.cached_token"
        assert tok2 == "ya29.cached_token"
        assert tok3 == "ya29.cached_token"
        assert mock_get.call_count == 1


def test_registry_client_token_cache_refreshes_after_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify RegistryClient refreshes token after token_ttl_seconds expires."""
    client = RegistryClient(base_url="http://127.0.0.1:8080", token_ttl_seconds=60.0)

    current_time = 1000.0
    monkeypatch.setattr("time.time", lambda: current_time)

    with patch("reach.registry.get_access_token", side_effect=["ya29.token_one", "ya29.token_two"]):
        assert client._get_token() == "ya29.token_one"
        assert client._get_token() == "ya29.token_one"

        # Advance time past TTL
        current_time += 65.0

        assert client._get_token() == "ya29.token_two"


def test_registry_client_clears_token_cache_on_401() -> None:
    """Verify RegistryClient invalidates cached token when receiving HTTP 401 Unauthorized."""
    client = RegistryClient(base_url="http://127.0.0.1:8080")
    expired_token: str | None = "ya29.expired_token"  # noqa: S105
    client._cached_token = expired_token
    client._cached_token_expiry = 9999999999.0

    with pytest.raises(AuthenticationError):
        client._handle_http_error(401, "Unauthorized", "projects/p/skills")

    assert client._cached_token is None
    assert client._cached_token_expiry == 0.0


@pytest.mark.parametrize(
    "valid_url",
    [
        "https://agentregistry.googleapis.com/v1alpha",
        "https://custom-region.googleapis.com/v1",
        "https://agentregistry.google.com/v1alpha",
        "http://127.0.0.1:8080",
        "http://localhost:8000/v1",
        "http://[::1]:9000",
    ],
)
def test_registry_client_accepts_valid_base_urls(valid_url: str) -> None:
    """Verify RegistryClient accepts HTTPS Google API domains and loopback URLs."""
    client = RegistryClient(base_url=valid_url, token="ya29.mock")  # noqa: S106
    assert client.base_url == valid_url.rstrip("/")


@pytest.mark.parametrize(
    ("invalid_url", "expected_err"),
    [
        ("http://agentregistry.googleapis.com", "must use HTTPS for non-local endpoints"),
        ("http://insecure-remote.com/api", "must use HTTPS for non-local endpoints"),
        ("https://evil-exfiltration.com/api", "must be a Google API domain or loopback"),
        ("ftp://agentregistry.googleapis.com", "must use HTTP or HTTPS scheme"),
        ("", "Invalid Agent Registry base_url"),
        ("invalid-url-no-scheme", "Invalid Agent Registry base_url"),
    ],
)
def test_registry_client_rejects_insecure_or_untrusted_base_urls(
    invalid_url: str,
    expected_err: str,
) -> None:
    """Verify RegistryClient rejects non-HTTPS URLs, untrusted domains, or invalid schemes."""
    with pytest.raises(ValueError, match=expected_err):
        RegistryClient(base_url=invalid_url, token="ya29.mock")  # noqa: S106
