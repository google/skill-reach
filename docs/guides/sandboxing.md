# Sandboxing

When evaluating, probing, or optimizing skill catalogs, live agent runtimes (such as Claude Code, Goose, Pi, or Google Antigravity CLI) execute real tools, file operations, and shell commands on the host machine. If an agent probes a third-party, unvetted, or untrusted skill, instructions inside that skill's `SKILL.md` could attempt unauthorized filesystem access, network exfiltration, or script execution.

Reach implements safety gates to ensure that live agent probes on resident skills are never executed without explicit confirmation, while offering clean isolation patterns for automated pipelines and local development.

---

## Interactive Safety Confirmation Gate

By default, any Reach command that launches live agent probes (`reach eval`, `reach sweep`, `reach check` Stage 2, and `reach optimize`) displays an interactive security notice before any probe subprocess is spawned:

```
╭─ ⚠️  Security Notice: Agent Probes ─────────────────────────────────────────╮
│ Target Catalog: 18 skills resolved from 2 locations:                        │
│   • ./.agents/skills (3 project skills)                                     │
│   • ~/.claude/skills (15 global user skills)                                │
│                                                                             │
│ Runtime: pi (No built-in sandbox)                                           │
│ Probing skills executes real agent processes that can run shell commands    │
│ and file operations. Ensure you trust all resident skills.                  │
│                                                                             │
│ Tip: For unverified skills, consider running Reach inside a container       │
│ (e.g. Docker) or using the offline keyword driver (--agent keyword).        │
╰─────────────────────────────────────────────────────────────────────────────╯
Proceed with execution? [y/N]:
```

- **Interactive TTY**: Type `y` or `yes` to proceed. Entering `n`, hitting Enter, or pressing `Ctrl+C` immediately aborts execution with exit code `1`.
- **Non-Interactive (CI / Scripts)**: If Reach is executed without an interactive terminal (non-TTY) without an explicit bypass, it safely fails closed with exit code `2` and displays an actionable remedy.

---

## Safe Offline Evaluation: The Keyword Driver

To benchmark reachability or run CI checks without executing host code or calling external language models, use the **keyword driver**:

```bash
reach eval --agent keyword
reach check --agent keyword
reach sweep --agent keyword
```

The keyword driver is an in-memory lexical matching engine using compiled word-boundary regular expressions and BM25 scoring. It operates with zero subprocess calls, zero tool executions, zero network requests, and $0.00 token cost—automatically bypassing safety confirmation prompts.

## Agent Runtime Isolation & Memory Boundaries

When evaluating skill reachability, Reach enforces clean-room isolation around the agent runtime. Without isolation, an agent's routing decisions can be distorted by ambient host files, default bundled tools, global user profiles, or persistent auto-memory, leading to unreproducible benchmarks.

### Core Isolation Principles

Every Reach agent driver operates within four isolation boundaries:

1. **Catalog Residency Enforcement**: Reach tracks runtime skill discovery and tool execution events. If an agent attempts to invoke skills outside the active benchmark catalog, Reach detects and records a **residency leak** (`residency leak: <skill>`) or **tool leak** (`tool leak: <tool>`).
2. **Bundled Capability Suppression**: Default skills, plugins, and extensions bundled with the agent runtime are suppressed during evaluation so they do not compete with or shadow benchmark catalog skills.
3. **Ambient Context Neutralization**: Repository-level instruction files, developer preference hints, and persistent auto-memory are bypassed to prevent external prompt bias.
4. **Session & Workspace Scoping**: Probes execute in ephemeral working directories with isolated configuration paths to ensure probe independence and prevent cross-query state leakage.

---

### Agent-Specific Isolation Mechanisms

Reach tailors isolation mechanisms to the execution model of each supported agent runtime:

| Agent Runtime                                           | Bundled Skill & Plugin Suppression                                                                                          | Ambient Memory & Prompt Neutralization                                      | Workspace & Session Scoping                                                                       |
| :------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ |
| **Claude Code**<br>`claude-code`                        | Suppresses default bundled skills and built-in plugin mods via `--settings`                                                 | Bypasses ambient `CLAUDE.md`, project auto-memory, and system policy skills | Ephemeral workspace with isolated `CLAUDE_CONFIG_DIR`                                             |
| **Goose**<br>`goose`                                    | Scopes built-in extensions strictly to skill resolution (`--with-builtin skills`)                                           | Runs with `--no-profile` to bypass developer extensions and `.goosehints`   | Dedicated working home directory per probe worker                                                 |
| **Pi**<br>`pi`                                          | Restricts runtime tools strictly to resident skill definitions                                                              | Ignores ambient host configurations and ancestor instructions               | Dedicated per-worker session directories (`.reach_pi_sessions`)                                   |
| **Antigravity**<br>`antigravity-cli`, `antigravity-sdk` | Scopes tool declarations strictly to resident catalog schemas and denies `read_file` on isolated `.config` credential paths | Enforces turn limits, early-exit invariants, and cumulative token budgets   | Ephemeral workspace and isolated `HOME` (`settings.json` + `0600` ADC copy unlinked after probes) |
| **Keyword Driver**<br>`keyword`                         | N/A (zero tool execution)                                                                                                   | N/A (zero prompt context or LLM calls)                                      | Ephemeral skill staging without subprocess or tool execution                                      |

---

## Container Sandboxing

For live agent runtimes probing untrusted or community skills, run Reach inside a container sandbox. This restricts tool execution, filesystem modifications, and environment visibility to the container boundary.

### Running with Docker or Podman

Mount your skill catalog into an ephemeral container and forward your model provider API key:

```bash
docker run --rm -it \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -v "$PWD:/workspace" \
  -w /workspace \
  python:3.12-slim \
  bash -c "pip install skill-reach && reach eval"
```

/// tip

- **Podman**: Replace `docker` with `podman` (on macOS, run `podman machine init --now` first).
- **Apple Container (macOS 26+)**: Replace `docker` with `container` after running `container system start --enable-kernel-install`.
- **macOS Providers**: The `docker` CLI works identically with Docker Desktop, [Colima](https://github.com/abiosoft/colima) (`colima start`), or OrbStack.

///

### Docker Compose

For automated multi-container environments, declare a sandbox service in `docker-compose.yml`:

```yaml
services:
  reach-sandbox:
    image: python:3.12-slim
    volumes:
      - .:/workspace:ro
      - reach-scratch:/tmp
    working_dir: /workspace
    environment:
      - GEMINI_API_KEY
      - REACH_YES=1
    command: ["reach", "check", "--agent", "pi", "--yes"]

volumes:
  reach-scratch:
```

---

## Cloud-Native Sandboxing with Google Cloud Run

For production CI/CD workflows, automated pull request benchmarks, or enterprise agent fleets, [Google Cloud Run sandboxes](https://docs.cloud.google.com/run/docs/configuring/services/sandboxes) (Preview) provide a fast, secure, and isolated environment to evaluate untrusted skills and execute live agent tools in the second-generation execution environment.

### Why Use Cloud Run Sandboxes for Skill Evaluation

Standard container runners share the host kernel and instance resources. Cloud Run sandboxes add process-level isolation within your container, offering several critical security boundaries:

1. **Metadata Server Protection**: Sandboxes are strictly isolated from the Google Cloud metadata server (`http://metadata.google.internal`). Even if a prompt injection or untrusted skill attempts to query instance credentials, it cannot access the container's GCP IAM service account token.
2. **Network Egress Blocked by Default**: Outbound networking is disabled inside the sandbox unless explicitly permitted with `--allow-egress`. This prevents untrusted skills from exfiltrating data or dialing external servers without authorization.
3. **Secret & Environment Isolation**: Sandboxes do not inherit environment variables or secrets mounted in the host container. Only explicitly passed variables (`--env`) are visible to the agent process.
4. **Read-Only Filesystem with Isolated Mounts**: The host container filesystem is mounted read-only. Writable workspaces are restricted to ephemeral tmpfs overlays (`--write`) or designated bind mounts (`--mount`).
5. **Fast Local Launch**: Sandboxes spin up in milliseconds inside the active container instance without provisioning new virtual machines or deploying new container revisions.

### Building the Runner Container Image

To run Reach on Cloud Run, package your skill catalog or benchmark workspace into a container image using the multi-stage `Dockerfile` in the repository root:

```dockerfile
# syntax=docker/dockerfile:1

# ── Stage 1: Build virtual environment with uv ──────────────────────────────
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /build

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src/ src/

RUN uv venv /opt/venv && \
    uv pip install --no-cache-dir ".[antigravity-sdk]"

# ── Stage 2: Minimal Runtime ────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:/usr/local/gcp/bin:$PATH" \
    PYTHONUNBUFFERED=1

RUN groupadd -g 10001 reach && \
    useradd -u 10001 -g reach -m -d /home/reach reach && \
    mkdir -p /workspace && \
    chown -R reach:reach /workspace && \
    chmod 1777 /workspace

WORKDIR /workspace
COPY --chown=reach:reach .agents/ /workspace/.agents/

# Cloud Run's /usr/local/gcp/bin/sandbox requires UID 0 in the outer container
# to initialize /var/run/netns before dropping privileges inside the sandbox jail.
# For standalone local Docker runs without `sandbox do`, pass `--user reach`.
CMD ["reach", "eval", "--agent", "antigravity-sdk", "--yes"]
```

Build and push the image to Google Cloud Artifact Registry:

```bash
gcloud builds submit --tag "LOCATION-docker.pkg.dev/PROJECT_ID/REPO_NAME/reach-runner:latest"
```

/// tip
**Build Context Optimization**: Maintain a `.gcloudignore` and `.dockerignore` in your repository root so local virtual environments (`.venv/`), git history (`.git/`), and benchmark runs (`.reach/`) are not uploaded to Cloud Build or baked into the container image.
///

### Enabling the Sandbox Launcher

To enable sandboxes on Cloud Run, configure the second-generation execution environment with the sandbox launcher enabled via the `gcloud beta` CLI (`--sandbox-launcher`) or declaratively in YAML (`sandboxLauncher: true`). When enabled, Cloud Run injects the `sandbox` CLI binary inside the container at `/usr/local/gcp/bin/sandbox`.

#### Cloud Run Jobs (Batch Evaluation & Sweeps)

For scheduled regression tests, CI-triggered scaling sweeps, or overnight batch evaluations, Cloud Run Jobs run to completion and terminate. Declare the job specification in YAML (`deploy/cloudrun/job.yaml`):

```yaml
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: reach-eval-job
  annotations:
    run.googleapis.com/launch-stage: BETA
spec:
  template:
    spec:
      template:
        spec:
          maxRetries: 0
          timeoutSeconds: 1800
          containers:
            - name: reach-eval
              image: LOCATION-docker.pkg.dev/PROJECT_ID/REPO_NAME/reach-runner:latest
              sandboxLauncher: true
              volumeMounts:
                - name: gemini-secret
                  mountPath: /secrets
                  readOnly: true
              command:
                - /usr/local/gcp/bin/sandbox
              args:
                - do
                - --allow-egress
                - --write
                - --env
                - PATH=/opt/venv/bin:/usr/local/gcp/bin:/usr/local/bin:/usr/bin:/bin
                - --env
                - REACH_YES=1
                - --env
                - GEMINI_API_KEY_FILE=/secrets/GEMINI_API_KEY
                - --mount
                - type=bind,source=/workspace,destination=/workspace
                - --mount
                - type=bind,source=/secrets,destination=/secrets,readonly
                - --workdir
                - /workspace
                - --
                - reach
                - eval
                - --agent
                - antigravity-sdk
                - --auto
                - --yes
              # Scale cpu/memory (e.g. 4 vCPU / 4Gi) for large multi-worker sweeps
              resources:
                limits:
                  cpu: "2"
                  memory: 2Gi
              env:
                - name: REACH_YES
                  value: "1"
          volumes:
            - name: gemini-secret
              secret:
                secretName: GEMINI_API_KEY
                items:
                  - key: latest
                    path: GEMINI_API_KEY
```

Deploy and execute the job:

```bash
gcloud run jobs replace deploy/cloudrun/job.yaml --region us-central1
gcloud run jobs execute reach-eval-job --wait --region us-central1
```

/// tip
**Secret Safety in Sandboxes**: The Cloud Run sandbox launcher (`sandbox do`) logs its startup arguments to `/var/log/sandbox.log`. Avoid passing raw secrets via `--env KEY="$KEY"` or shell command substitutions (`$(cat ...)`). Instead, mount the secret as a read-only volume (`/secrets/GEMINI_API_KEY`) and pass `--env GEMINI_API_KEY_FILE=/secrets/GEMINI_API_KEY` so Reach reads the credential directly in Python without spawning subshells or exposing keys in CLI arguments or logs.
///

#### Cloud Run Services (Evaluation Webhook / API)

If hosting an automated skill verification service or pull request webhook triggered via HTTP:

/// note
Cloud Run Services require a container that listens for incoming HTTP requests on `$PORT` (default `8080`). Your service container runs a lightweight web framework (e.g. FastAPI) that handles webhook events and dispatches isolated evaluation runs using `sandbox do` via subprocess.
///

Deploy a Cloud Run service with the sandbox launcher enabled:

```bash
gcloud beta run deploy reach-eval-service \
  --image "LOCATION-docker.pkg.dev/PROJECT_ID/REPO_NAME/reach-service:latest" \
  --sandbox-launcher \
  --no-allow-unauthenticated \
  --region us-central1
```

Or declare the service in YAML (`service.yaml`):

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: reach-eval-service
  annotations:
    run.googleapis.com/launch-stage: BETA
spec:
  template:
    spec:
      containers:
        - name: reach-service
          image: LOCATION-docker.pkg.dev/PROJECT_ID/REPO_NAME/reach-service:latest
          sandboxLauncher: true
          resources:
            limits:
              cpu: "2"
              memory: 2Gi
          ports:
            - containerPort: 8080
```

### In-Container Execution with `sandbox do`

Once enabled, invoke the `sandbox do` command from within your container to run Reach or agent runtimes in an isolated sandbox. The `sandbox do` command automatically provisions the sandbox, executes the command, and cleans up the sandbox on exit:

/// important
**Environment Variable Isolation**: Cloud Run sandboxes do not inherit environment variables from the host container, including `PATH`. Always explicitly forward `--env PATH="$PATH"` so Reach and agent runtime binaries are discoverable.
///

```bash
# Execute reach eval in an ephemeral sandbox with outbound API access, writable workspace, and mounted secret
sandbox do --allow-egress --write \
  --env PATH="$PATH" \
  --env REACH_YES="1" \
  --env GEMINI_API_KEY_FILE="/secrets/GEMINI_API_KEY" \
  --mount type=bind,source=/workspace,destination=/workspace \
  --mount type=bind,source=/secrets,destination=/secrets,readonly \
  --workdir /workspace \
  -- reach eval --agent antigravity-sdk --yes
```

### Sandbox CLI Flag Reference

The following options configure process containment when running `sandbox do`:

| Option           | Type   | Description                                                                         |
| :--------------- | :----- | :---------------------------------------------------------------------------------- |
| `--allow-egress` | flag   | Permits outbound network connections (required for model APIs). Blocked by default. |
| `--write`        | flag   | Enables a writable temporary filesystem (tmpfs) overlay.                            |
| `--mount`        | string | Attaches host directory mounts (`type=bind,source=SRC,destination=DST[,readonly]`). |
| `-e, --env`      | string | Injects environment variables into the sandbox (host env is not inherited).         |
| `-w, --workdir`  | string | Specifies the working directory for command execution inside the sandbox.           |
| `--export-tar`   | string | Archives the modified overlay filesystem to a `.tar` file upon completion.          |
| `--import-tar`   | string | Pre-populates the sandbox overlay filesystem from a `.tar` archive on startup.      |
| `--sync-tar`     | string | Two-way archive sync (imports state on launch and exports updates on exit).         |

### Persisting Reports to Cloud Storage

Because Cloud Run containers and sandboxes are ephemeral, stream evaluation reports (`.reach/eval.json`, `report.html`) to Google Cloud Storage using `gcloud storage`:

```bash
# Upload evaluation artifacts to Cloud Storage
gcloud storage cp -r .reach/ "gs://my-evaluation-bucket/runs/$(date +%Y%m%d-%H%M%S)/"
```

---

## Bypassing Prompts in CI/CD

To run automated pipelines without interactive prompts, Reach provides three equivalent bypass mechanisms:

| Mechanism                | Syntax                        | Scope & Usage                                                                                                         |
| :----------------------- | :---------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| **CLI Flag**             | `reach eval --yes` (`-y`)     | Single execution across any probe command (`eval`, `sweep`, `check`, `optimize`).                                     |
| **Environment Variable** | `export REACH_YES=1`          | Current shell session, container runner, or CI/CD workflow pipeline.                                                  |
| **Repository Config**    | `[study]`<br>`trusted = true` | Repository-wide setting in `reach.toml`. Excluded from `config_fingerprint` to keep historical benchmarks comparable. |

/// note
`--yes` / `-y` specifically bypasses safety confirmation prompts. On `reach optimize`, `--force` / `-f` remains dedicated to force-applying candidate descriptions even when empirical recall does not improve.
///
