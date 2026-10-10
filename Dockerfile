# syntax=docker/dockerfile:1

# ── Stage 1: Build virtual environment with uv ──────────────────────────────
FROM python:3.12-slim AS builder

# Copy uv binary for fast package resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /build

# Configure virtual environment in /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Copy package metadata and source to install into isolated venv
COPY pyproject.toml README.md ./
COPY src/ src/

RUN uv venv /opt/venv && \
    uv pip install --no-cache-dir ".[antigravity-sdk]"

# ── Stage 2: Minimal Runtime ────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Install runtime dependencies (git for skill repository inspection, ca-certificates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy prebuilt virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Configure PATH to include virtual environment and Cloud Run sandbox launcher
ENV PATH="/opt/venv/bin:/usr/local/gcp/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Create unprivileged user (UID 10001) and prepare isolated workspace.
# Group root (GID 0) with mode 0775 allows both `--user reach` (UID 10001) and
# rootless/user-namespace sandboxes to write `.reach/` evaluation artifacts.
RUN groupadd -g 10001 reach && \
    useradd -u 10001 -g reach -m -d /home/reach reach && \
    mkdir -p /workspace && \
    chown -R reach:root /workspace && \
    chmod 0775 /workspace

WORKDIR /workspace

# Copy only resident skill catalog (avoid copying src/ or tests/ into runtime workspace)
COPY --chown=reach:reach .agents/ /workspace/.agents/

# Security note: Cloud Run's /usr/local/gcp/bin/sandbox launcher requires UID 0 in the
# outer container to initialize /var/run/netns network namespaces before dropping privileges
# to a non-root user inside the sandbox jail (and Cloud Run does not support runAsUser
# overrides). For standalone local Docker runs without `sandbox do`, pass `--user reach`.
CMD ["reach", "eval", "--agent", "antigravity-sdk", "--yes"]
