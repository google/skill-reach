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

"""Provide client access, authentication, and two-tier caching for Google Cloud Agent Registry."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reach._io import atomic_write_text
from reach.config import resolve_path
from reach.models import Skill
from reach.runtime._fs import ensure_private_directory

__all__ = [
    "AuthenticationError",
    "NotFoundError",
    "PermissionDeniedError",
    "RegistryClient",
    "RegistryError",
    "RegistryManifest",
    "RegistrySkillData",
    "ServiceDisabledError",
]


#: Minimum character length for valid bearer access tokens.
_MIN_TOKEN_LENGTH: Final = 16


class RegistryError(RuntimeError):
    """Base exception for Google Cloud Agent Registry operations."""


class AuthenticationError(RegistryError):
    """Raised when authentication credentials cannot be obtained or are invalid."""


class PermissionDeniedError(RegistryError):
    """Raised when caller lacks required IAM permissions (403 Forbidden)."""


class ServiceDisabledError(RegistryError):
    """Raised when the Agent Registry API is not enabled on the target project."""


class NotFoundError(RegistryError):
    """Raised when target project, location, or skill is not found (404)."""


class RegistrySkillData(BaseModel):
    """Represent raw skill metadata returned by the Agent Registry REST API."""

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    name: str
    display_name: str = Field(alias="displayName", default="")
    description: str = ""
    type: str = "SIMPLE"
    state: str = "STATE_ACTIVE"
    target_state: str = Field(alias="targetState", default="TARGET_STATE_ACTIVE")
    default_revision: str | None = Field(alias="defaultRevision", default=None)
    skill_id: str | None = Field(alias="skillId", default=None)
    publisher: str | None = None
    create_time: str | None = Field(alias="createTime", default=None)
    update_time: str | None = Field(alias="updateTime", default=None)

    @property
    def identifier(self) -> str:
        """Return the display name if available, otherwise trailing segment of resource name."""
        if self.display_name:
            return self.display_name
        return self.name.rsplit("/", 1)[-1]

    @property
    def revision_slug(self) -> str:
        """Extract the revision ID from defaultRevision or return 'default'."""
        if self.default_revision and "/" in self.default_revision:
            return self.default_revision.rsplit("/", 1)[-1]
        return "default"


class RegistryManifest(BaseModel):
    """Persisted snapshot of skills metadata in a project and location."""

    model_config = ConfigDict(frozen=True)

    project: str
    location: str
    publisher: str | None = None
    fetched_at: datetime
    skills: tuple[RegistrySkillData, ...] = ()


def find_adc_path() -> Path | None:
    """Locate local Google Cloud Application Default Credentials file if present."""
    if custom := os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        custom_path = Path(custom).expanduser()
        if custom_path.is_file():
            return custom_path
    if cloudsdk_config := os.environ.get("CLOUDSDK_CONFIG"):
        sdk_path = Path(cloudsdk_config).expanduser() / "application_default_credentials.json"
        if sdk_path.is_file():
            return sdk_path
    candidates = [
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json",
    ]
    if appdata := os.environ.get("APPDATA"):
        candidates.append(Path(appdata) / "gcloud" / "application_default_credentials.json")
    candidates.append(
        Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def is_adc_available() -> bool:
    """Return True if Application Default Credentials are configured locally without network I/O."""
    return find_adc_path() is not None


def get_access_token() -> str:
    """Resolve an OAuth2 access token for Google Cloud APIs via ADC.

    Attempts to resolve Application Default Credentials using google-auth first,
    falling back to `gcloud auth application-default print-access-token`.

    Returns:
        A valid OAuth2 bearer access token string.

    Raises:
        AuthenticationError: If credentials cannot be acquired or refreshed.
    """
    # 1. Try google-auth library if installed
    with contextlib.suppress(Exception):
        import google.auth
        import google.auth.transport.requests

        creds, _project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        request = google.auth.transport.requests.Request()
        creds.refresh(request)
        if creds.token:
            return str(creds.token)

    # 2. Fall back to gcloud auth application-default print-access-token
    gcloud_bin = shutil.which("gcloud")
    if not gcloud_bin:
        msg = (
            "gcloud CLI not found in PATH. Please install Google Cloud SDK "
            "or run 'gcloud auth application-default login'."
        )
        raise AuthenticationError(msg)

    try:
        proc = subprocess.run(  # noqa: S603
            [gcloud_bin, "auth", "application-default", "print-access-token"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as err:
        msg = (
            "Unable to obtain Google Cloud access token. Please run "
            "'gcloud auth application-default login' to authenticate."
        )
        raise AuthenticationError(msg) from err

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    token = ""
    for line in reversed(lines):
        if (
            not any(c.isspace() for c in line)
            and not line.upper().startswith(("WARNING", "INFO", "ERROR", "NOTE"))
            and len(line) >= _MIN_TOKEN_LENGTH
        ):
            token = line
            break

    if not token or any(c.isspace() for c in token):
        msg = (
            "Unexpected or invalid token format from gcloud. Please run "
            "'gcloud auth application-default login' to refresh credentials."
        )
        raise AuthenticationError(msg)

    return token


def _normalize_publisher(publisher: str | None) -> str:
    """Extract terminal publisher identifier from a resource name or slug."""
    if not publisher:
        return ""
    normalized = publisher.strip().rstrip("/")
    if "/" in normalized:
        return normalized.rsplit("/", 1)[-1]
    return normalized


def _matches_publisher(actual_publisher: str | None, requested_publisher: str | None) -> bool:
    """Check if actual publisher resource name or identifier matches the requested publisher."""
    if not requested_publisher:
        return True
    if not actual_publisher:
        return False
    actual_clean = actual_publisher.strip().rstrip("/")
    requested_clean = requested_publisher.strip().rstrip("/")
    if actual_clean == requested_clean:
        return True
    actual_norm = _normalize_publisher(actual_clean)
    requested_norm = _normalize_publisher(requested_clean)
    return bool(actual_norm and requested_norm and actual_norm == requested_norm)


class RegistryClient:
    """HTTP client for querying the Google Cloud Agent Registry REST API."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://agentregistry.googleapis.com/v1alpha",
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        token_ttl_seconds: float = 3000.0,
        allow_custom_host: bool = False,
    ) -> None:
        """Initialize RegistryClient with optional bearer token and retry settings."""
        self.token = token
        self.base_url = self._validate_base_url(base_url, allow_custom_host=allow_custom_host)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.token_ttl_seconds = token_ttl_seconds
        self._cached_token: str | None = None
        self._cached_token_expiry: float = 0.0

    @staticmethod
    def _validate_base_url(base_url: str, *, allow_custom_host: bool = False) -> str:
        """Validate that base_url uses HTTPS and points to an approved Google domain or loopback."""
        clean_url = base_url.strip().rstrip("/")
        parsed = urllib.parse.urlsplit(clean_url)
        if not parsed.scheme or not parsed.hostname:
            msg = f"Invalid Agent Registry base_url: {base_url!r}"
            raise ValueError(msg)

        if parsed.scheme not in ("http", "https"):
            msg = f"Agent Registry base_url must use HTTP or HTTPS scheme, got: {parsed.scheme!r}"
            raise ValueError(msg)

        is_loopback = parsed.hostname in ("127.0.0.1", "localhost", "::1")
        if parsed.scheme != "https" and not is_loopback:
            msg = f"Agent Registry base_url must use HTTPS for non-local endpoints: {base_url!r}"
            raise ValueError(msg)

        is_google_api = (
            parsed.hostname == "googleapis.com"
            or parsed.hostname.endswith(".googleapis.com")
            or parsed.hostname.endswith(".google.com")
        )
        if not is_loopback and not is_google_api and not allow_custom_host:
            msg = (
                f"Agent Registry base_url host must be a Google API domain or loopback, "
                f"got: {parsed.hostname!r}"
            )
            raise ValueError(msg)

        return clean_url

    def _get_token(self) -> str:
        """Resolve active authentication token, fetching dynamically or from cache."""
        if self.token:
            return self.token

        now = time.time()
        if self._cached_token and now < self._cached_token_expiry:
            return self._cached_token

        token = get_access_token()
        self._cached_token = token
        self._cached_token_expiry = now + self.token_ttl_seconds
        return token

    def _request(self, endpoint: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        """Execute an authenticated GET request with rate-limit retry and error mapping."""
        query_string = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"{self.base_url}/{endpoint.lstrip('/')}{query_string}"
        token = self._get_token()

        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(  # noqa: S310
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "User-Agent": "skill-reach",
                },
            )
            try:
                with urllib.request.urlopen(req) as resp:  # noqa: S310
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw)  # type: ignore[no-any-return]
            except urllib.error.HTTPError as http_err:
                status = http_err.code
                error_body = ""
                with contextlib.suppress(OSError):
                    error_body = http_err.read().decode("utf-8")
                http_err.close()

                if status == HTTPStatus.TOO_MANY_REQUESTS and attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2**attempt)
                    time.sleep(sleep_time)
                    continue

                self._handle_http_error(status, error_body, endpoint)
            except urllib.error.URLError as url_err:
                msg = f"Failed to connect to Agent Registry endpoint ({url}): {url_err.reason}"
                raise RegistryError(msg) from url_err

        msg = f"Agent Registry request exceeded maximum retries for {url}"
        raise RegistryError(msg)

    def _handle_http_error(self, status: int, body: str, endpoint: str) -> None:
        """Map HTTP error status codes to descriptive domain exceptions."""
        error_details = {}
        error_message = body
        with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            parsed = json.loads(body)
            if isinstance(parsed, dict) and "error" in parsed:
                err_dict = parsed["error"]
                error_message = err_dict.get("message", body)
                error_details = err_dict

        if status == HTTPStatus.UNAUTHORIZED:
            self._cached_token = None
            self._cached_token_expiry = 0.0
            msg = (
                "Authentication failed when contacting Agent Registry. "
                "Run 'gcloud auth application-default login' to refresh credentials."
            )
            raise AuthenticationError(msg)

        if status == HTTPStatus.FORBIDDEN:
            # Check for service disabled
            details_list = error_details.get("details", [])
            for item in details_list:
                if isinstance(item, dict) and item.get("reason") == "SERVICE_DISABLED":
                    msg = (
                        "The Agent Registry API (agentregistry.googleapis.com) is disabled. "
                        "Enable it with: gcloud services enable agentregistry.googleapis.com"
                    )
                    raise ServiceDisabledError(msg)

            msg = (
                f"Permission denied on Agent Registry ({endpoint}). "
                "Ensure your account has the 'roles/agentregistry.user' role on the project."
            )
            raise PermissionDeniedError(msg)

        if status == HTTPStatus.NOT_FOUND:
            msg = f"Agent Registry resource not found ({endpoint}): {error_message}"
            raise NotFoundError(msg)

        msg = f"Agent Registry request failed with HTTP {status}: {error_message}"
        raise RegistryError(msg)

    def list_skills(
        self,
        project: str,
        location: str = "global",
        publisher: str | None = None,
    ) -> list[RegistrySkillData]:
        """Fetch all registered skills in the specified project and location."""
        endpoint = f"projects/{project}/locations/{location}/skills"
        skills: list[RegistrySkillData] = []
        page_token: str | None = None

        while True:
            params: dict[str, str] = {}
            if page_token:
                params["pageToken"] = page_token

            data = self._request(endpoint, params=params or None)
            raw_skills = data.get("skills", [])
            for s in raw_skills:
                try:
                    parsed = RegistrySkillData.model_validate(s)
                    if not _matches_publisher(parsed.publisher, publisher):
                        continue
                    skills.append(parsed)
                except (ValidationError, ValueError):
                    continue

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return skills

    def get_skill(
        self,
        project: str,
        location: str,
        skill_id: str,
    ) -> RegistrySkillData:
        """Fetch a single skill resource from Agent Registry."""
        endpoint = f"projects/{project}/locations/{location}/skills/{skill_id}"
        data = self._request(endpoint)
        return RegistrySkillData.model_validate(data)

    def fetch_manifest(
        self,
        project: str,
        location: str = "global",
        publisher: str | None = None,
    ) -> RegistryManifest:
        """Compile a complete RegistryManifest snapshot for project and location."""
        raw_skills = self.list_skills(project=project, location=location, publisher=publisher)
        return RegistryManifest(
            project=project,
            location=location,
            publisher=publisher,
            fetched_at=datetime.now(UTC),
            skills=tuple(raw_skills),
        )


def _sanitize_path_segment(value: str) -> str:
    """Sanitize arbitrary string into a safe path segment slug."""
    return re.sub(r"[^\w.-]", "_", value.strip()).lstrip(".")


def _safe_resolve_subpath(
    root: Path,
    *segments: tuple[str, str],
    disallow_separators: bool = False,
    fallbacks: dict[str, str] | None = None,
) -> Path:
    """Resolve and validate a relative subpath under a root directory.

    Args:
        root: Base directory that must contain the resolved target.
        *segments: Tuples of (value, label) for each path segment.
        disallow_separators: If True, reject values containing path separators or absolute paths.
        fallbacks: Optional mapping of segment label to fallback string when slug is empty.

    Returns:
        The validated Path strictly contained within root.

    Raises:
        ValueError: If any segment is invalid, empty without a fallback, or escapes root.
    """
    resolved_root = root.resolve()
    current = resolved_root
    fallbacks = fallbacks or {}

    for value, label in segments:
        trimmed = value.strip()
        if disallow_separators and (
            Path(trimmed).is_absolute() or "/" in trimmed or "\\" in trimmed
        ):
            msg = f"Invalid {label} identifier: {value!r} escapes cache directory"
            raise ValueError(msg)

        slug = _sanitize_path_segment(trimmed)
        if not slug or slug in (".", ".."):
            if label in fallbacks:
                slug = fallbacks[label]
            elif disallow_separators:
                msg = f"Invalid {label} identifier: {value!r} escapes cache directory"
                raise ValueError(msg)
            else:
                msg = f"Invalid {label} identifier: {value!r}"
                raise ValueError(msg)

        current = current / slug

    target = current.resolve()
    if not target.is_relative_to(resolved_root) or (
        disallow_separators and target == resolved_root
    ):
        if segments:
            last_val, last_label = segments[-1]
            msg = f"Invalid {last_label} identifier: {last_val!r} escapes cache directory"
        else:
            msg = f"Invalid path escapes directory: {target}"
        raise ValueError(msg)

    return target


class RegistryCacheManager:
    """Manage local caching and payload hydration of Agent Registry skills."""

    REGISTRY_CACHE_RELPATH = Path(".reach") / "cache" / "registry"

    def __init__(self, cache_root: Path | str | None = None) -> None:
        """Initialize RegistryCacheManager with optional root directory."""
        if cache_root is not None:
            self.cache_root = resolve_path(cache_root)
        else:
            self.cache_root = resolve_path(self.REGISTRY_CACHE_RELPATH)

    @classmethod
    def for_workdir(cls, workdir: Path | str) -> Self:
        """Instantiate a RegistryCacheManager rooted inside a specific workspace directory."""
        return cls(cache_root=resolve_path(workdir) / cls.REGISTRY_CACHE_RELPATH)

    def location_dir(self, project: str, location: str) -> Path:
        """Return the directory containing manifest and cached skills for project and location."""
        return _safe_resolve_subpath(
            self.cache_root,
            (project, "project"),
            (location, "location"),
        )

    def manifest_path(
        self,
        project: str,
        location: str,
        publisher: str | None = None,
    ) -> Path:
        """Return the path to the cached .manifest.json file, optionally scoped by publisher."""
        loc_dir = self.location_dir(project, location)
        if publisher:
            slug = _sanitize_path_segment(publisher)
            return loc_dir / f".manifest.{slug}.json"
        return loc_dir / ".manifest.json"

    def skill_dir(
        self,
        project: str,
        location: str,
        skill_name: str,
        revision_slug: str = "default",
    ) -> Path:
        """Return the directory for an unpacked skill revision."""
        loc_dir = self.location_dir(project, location)
        return _safe_resolve_subpath(
            loc_dir,
            (skill_name, "skill name"),
            (revision_slug, "revision"),
            fallbacks={"revision": "default"},
        )

    def get_cached_manifest(
        self,
        project: str,
        location: str,
        publisher: str | None = None,
        max_age_seconds: int = 300,
    ) -> RegistryManifest | None:
        """Load cached manifest if present, compatible with publisher, and within TTL."""
        candidate_paths = [self.manifest_path(project, location, publisher)]
        if publisher is not None:
            candidate_paths.append(self.manifest_path(project, location, None))

        for manifest_file in candidate_paths:
            if not manifest_file.is_file():
                continue

            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
                manifest = RegistryManifest.model_validate(data)
                now = datetime.now(UTC)
                age = (now - manifest.fetched_at).total_seconds()
                if max_age_seconds >= 0 and age > max_age_seconds:
                    continue
                if manifest.publisher is not None:
                    if publisher is None:
                        continue
                    if not _matches_publisher(manifest.publisher, publisher):
                        continue
                elif publisher is not None:
                    filtered_skills = tuple(
                        s for s in manifest.skills if _matches_publisher(s.publisher, publisher)
                    )
                    return manifest.model_copy(
                        update={"skills": filtered_skills, "publisher": publisher}
                    )
                return manifest
            except (json.JSONDecodeError, ValidationError, OSError):
                continue

        return None

    def save_manifest(self, manifest: RegistryManifest) -> None:
        """Atomically persist a RegistryManifest to disk."""
        target_path = self.manifest_path(manifest.project, manifest.location, manifest.publisher)
        ensure_private_directory(target_path.parent)
        atomic_write_text(target_path, manifest.model_dump_json(indent=2) + "\n")

    def hydrate_skill_file(
        self,
        project: str,
        location: str,
        skill_data: RegistrySkillData,
    ) -> Path:
        """Generate local directory and SKILL.md file for a registry skill if absent.

        Args:
            project: Google Cloud project ID.
            location: Registry location.
            skill_data: RegistrySkillData object.

        Returns:
            Path to the enclosing directory of the hydrated skill.
        """
        skill_dir = ensure_private_directory(
            self.skill_dir(
                project=project,
                location=location,
                skill_name=skill_data.identifier,
                revision_slug=skill_data.revision_slug,
            )
        )
        skill_file = skill_dir / "SKILL.md"

        if not skill_file.is_file():
            body_content = (
                f"---\n"
                f"name: {skill_data.identifier}\n"
                f"description: {skill_data.description}\n"
                f"---\n\n"
                f"# {skill_data.identifier}\n\n"
                f"{skill_data.description}\n\n"
                f"<!-- Registry source: {skill_data.name} -->\n"
            )
            if skill_data.skill_id:
                body_content += f"<!-- Skill ID: {skill_data.skill_id} -->\n"

            atomic_write_text(skill_file, body_content)

        return skill_dir

    def resolve_skills(
        self,
        project: str,
        location: str = "global",
        publisher: str | None = None,
        fresh: bool = False,
        no_cache: bool = False,
        cache_ttl_seconds: int = 300,
        client: RegistryClient | None = None,
    ) -> list[Skill]:
        """Fetch or load skills from cache, hydrate local files, and return Skill models.

        Args:
            project: Google Cloud project ID.
            location: Registry location (e.g. 'global').
            publisher: Optional publisher filter.
            fresh: If True, bypass metadata TTL and fetch live.
            no_cache: If True, do not use or persist cache.
            cache_ttl_seconds: TTL in seconds for metadata cache validity.
            client: Optional pre-configured RegistryClient.

        Returns:
            List of validated Skill models.
        """
        manifest: RegistryManifest | None = None
        if not fresh and not no_cache:
            manifest = self.get_cached_manifest(
                project,
                location,
                publisher=publisher,
                max_age_seconds=cache_ttl_seconds,
            )

        if manifest is None:
            active_client = client or RegistryClient()
            try:
                manifest = active_client.fetch_manifest(
                    project=project,
                    location=location,
                    publisher=publisher,
                )
                if not no_cache:
                    self.save_manifest(manifest)
            except Exception as net_err:
                # If network fails, try falling back to stale cache
                stale = self.get_cached_manifest(
                    project,
                    location,
                    publisher=publisher,
                    max_age_seconds=-1,
                )
                if stale is not None:
                    manifest = stale
                else:
                    raise net_err

        skills: list[Skill] = []
        for s in manifest.skills:
            if not _matches_publisher(s.publisher, publisher):
                continue

            skill_path = self.hydrate_skill_file(project, location, s)
            model_invocable = s.state == "STATE_ACTIVE"

            meta = {
                "urn": s.skill_id or "",
                "publisher": s.publisher or "",
                "project": project,
                "location": location,
                "default_revision": s.default_revision or "",
                "state": s.state,
            }

            skills.append(
                Skill(
                    name=s.identifier,
                    description=s.description,
                    path=skill_path,
                    metadata=meta,
                    manifest_source=f"agent-registry://{project}/{location}",
                    model_invocable=model_invocable,
                ),
            )

        return sorted(skills, key=lambda sk: sk.name)

    def clean(self, project: str | None = None, dry_run: bool = False) -> tuple[int, list[Path]]:
        """Clean cached registry files, optionally scoped to a single project.

        Args:
            project: Optional project ID to limit cleanup scope.
            dry_run: If True, compute freed bytes and paths without deleting.

        Returns:
            Tuple of (total_bytes_reclaimed, list_of_paths_removed).
        """
        if project:
            target_dir = _safe_resolve_subpath(
                self.cache_root,
                (project, "project"),
                disallow_separators=True,
            )
        else:
            target_dir = self.cache_root.resolve()

        if not target_dir.exists():
            return 0, []

        total_bytes = 0
        removed_paths: list[Path] = []

        for p in target_dir.rglob("*"):
            if p.is_file():
                with contextlib.suppress(OSError):
                    total_bytes += p.stat().st_size

        removed_paths.append(target_dir)

        if not dry_run:
            shutil.rmtree(target_dir, ignore_errors=True)

        return total_bytes, removed_paths
