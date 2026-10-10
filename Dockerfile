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

# Prepare workspace directory
RUN mkdir -p /workspace

WORKDIR /workspace

# Copy skill catalog and configuration
COPY . /workspace

CMD ["reach", "eval", "--yes"]
