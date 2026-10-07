# Sandboxing & Execution Safety

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

If you want to benchmark skill reachability or run CI checks on unvetted skills without executing any host code or calling external language models, use the **deterministic keyword driver**:

```bash
reach eval --agent keyword
reach check --agent keyword
reach sweep --agent keyword
```

The keyword driver is an in-memory lexical matching engine that matches query terms against skill names and descriptions using compiled word-boundary regular expressions and BM25 scoring, simulating agent routing decisions without running subprocesses, executing tools, or making network calls.

### Why `--agent keyword` is Safe

1. **Zero Process Execution**: Operates purely in Python memory using compiled word-boundary regular expressions. It never invokes `subprocess` or shell commands.
2. **Zero Tool Execution**: Does not run scripts, shell tools, or instructions defined inside skills.
3. **Zero Network & Token Cost**: Requires no model credentials, produces no HTTP network calls, and costs $0.00.
4. **Automatic Safety Bypass**: Because no subprocesses or tools are invoked, Reach automatically skips safety prompts when `--agent keyword` is active.

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
| **Keyword Driver**<br>`keyword`                         | N/A (zero tool execution)                                                                                                   | N/A (zero prompt context or LLM calls)                                      | In-memory evaluation (zero disk mutations)                                                        |

---

## Containerized Sandboxing (Docker & Podman)

For live agent runtimes probing untrusted or community skills, run Reach inside a containerized sandbox. This restricts tool access, file system modifications, and environment variable visibility to the container boundary.

### Running with Docker

Mount your skill catalog into an ephemeral container and pass your model provider API key:

```bash
docker run --rm -it \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -v "$PWD:/workspace" \
  -w /workspace \
  python:3.13-slim \
  bash -c "pip install skill-reach && reach eval"
```

### Running with Docker Compose

Define an isolated sandbox service in `docker-compose.yml`:

```yaml
services:
  reach-sandbox:
    image: python:3.13-slim
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

### Enabling the Sandbox Launcher

To enable sandboxes, deploy your Cloud Run service or job with the `--sandbox-launcher` flag or configure `sandboxLauncher: true` in your container specification. This injects the `sandbox` CLI binary at `/usr/local/gcp/bin/sandbox`.

#### Cloud Run Jobs (Batch Evaluation & Sweeps)

For scheduled regression tests or CI-triggered scaling sweeps, deploy Reach as a Cloud Run Job:

```bash
# Create a Cloud Run Job with the sandbox launcher enabled
gcloud beta run jobs create reach-eval-job \
  --image "LOCATION-docker.pkg.dev/PROJECT_ID/REPO_NAME/reach-runner:latest" \
  --sandbox-launcher \
  --set-env-vars "REACH_YES=1" \
  --region us-central1
```

Or declare the job in YAML (`job.yaml`):

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
          containers:
            - name: reach-eval
              image: LOCATION-docker.pkg.dev/PROJECT_ID/REPO_NAME/reach-runner:latest
              sandboxLauncher: true
              env:
                - name: REACH_YES
                  value: "1"
```

Deploy the job configuration:

```bash
gcloud run jobs replace job.yaml
gcloud run jobs execute reach-eval-job --wait
```

#### Cloud Run Services (Evaluation Webhook / API)

If hosting an automated skill verification service or pull request webhook:

```bash
# Deploy a Cloud Run service with the sandbox launcher enabled
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
          ports:
            - containerPort: 8080
```

### In-Container Execution with `sandbox do`

Once enabled, invoke the `sandbox do` command from within your container to run Reach or agent runtimes in an isolated sandbox. The `sandbox do` command automatically provisions the sandbox, executes the command, and cleans up the sandbox on exit:

```bash
# Execute reach eval in an ephemeral sandbox with outbound API access and writable workspace
sandbox do --allow-egress --write \
  --env GEMINI_API_KEY="$GEMINI_API_KEY" \
  --env REACH_YES="1" \
  --mount type=bind,source=/workspace,destination=/workspace \
  --workdir /workspace \
  -- reach eval --yes
```

#### Python Subprocess Wrapper

If your evaluation runner is written in Python, execute the sandbox binary via `subprocess`:

```python
import subprocess
import sys


def run_sandboxed_reach(cmd: list[str], api_key: str, workspace: str) -> int:
    """Execute Reach commands inside an isolated Cloud Run sandbox."""
    sandbox_cmd = [
        "sandbox",
        "do",
        "--allow-egress",
        "--write",
        f"--env=GEMINI_API_KEY={api_key}",
        "--env=REACH_YES=1",
        f"--mount=type=bind,source={workspace},destination=/workspace",
        "--workdir=/workspace",
        "--",
        *cmd,
    ]
    result = subprocess.run(sandbox_cmd, stdout=sys.stdout, stderr=sys.stderr)
    return result.returncode


# Example: execute check inside sandbox
exit_code = run_sandboxed_reach(["reach", "check", "--yes"], api_key="...", workspace="/workspace")
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

## Bypassing Prompts in CI/CD & Automated Scripts

To run automated pipelines without interactive prompts, Reach provides three equivalent bypass mechanisms:

### 1. CLI Flag (`--yes` / `-y`)

Pass `--yes` or `-y` to any command that probes agents:

```bash
reach eval --yes
reach sweep -y
reach check --yes
reach optimize --skill my-skill --yes
```

> [!NOTE]
> `--yes` / `-y` specifically bypasses safety confirmation prompts. On `reach optimize`, `--force` / `-f` remains exclusively dedicated to force-applying candidate descriptions even when empirical recall does not improve.

### 2. Environment Variable (`REACH_YES=1`)

Export `REACH_YES=1` in your shell or CI workflow configuration:

```bash
export REACH_YES=1
reach eval
```

### 3. Repository Configuration (`trusted = true`)

For trusted private codebases where all skills are vetted, set `trusted = true` under `[study]` in `reach.toml`:

```toml
[study]
trusted = true
```

Setting `trusted = true` bypasses safety confirmation for all runs using that configuration. Reach excludes `trusted` from the experiment `config_fingerprint`, ensuring your historical diffs, run records, and benchmarks remain fully comparable.
