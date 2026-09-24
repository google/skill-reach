# `skill-reach`

An evaluation suite for AI agent skill routing, collision detection, and description optimization.

While standard evaluation suites measure skill execution _after_ invocation, `skill-reach` measures the discovery and routing phase: determining whether incoming user prompts route to the intended skill in the presence of competing descriptions.

---

## The Routing Problem

Modern AI agents (including Claude Code, Google Antigravity, and OpenAI Codex) select skills dynamically. When a user enters a prompt, the agent scans installed skill names and descriptions in its system prompt and decides which skill to activate.

```
User Prompt ──► [ Agent System Prompt: Skill Catalog ] ──► Selected Skill
                       ├── skill-a (Cloud Run deployer)
                       ├── skill-b (Docker container builder)   ◄── Collision / Misroute?
                       └── skill-c (Kubernetes manifest helper)
```

When two or more skills describe overlapping tasks, the model can silently misroute requests. For example, a database migration request might land on a generic SQL helper, or a specialized security scanner might sit idle because a broader utility claimed the topic.

Textual similarity alone (like BM25 or embedding cosine distance) cannot reliably predict these failures. Two skills with similar boilerplate can route cleanly if their distinguishing keywords are clear, while a single vague phrase in an unrelated skill can hijack prompts. `skill-reach` runs empirical probes against real agent runtimes to measure what the model actually decides.

---

## 30-Second Quickstart

Install `reach` with `uv`:

```bash
uv tool install skill-reach
```

Run a static check on your skills directory without making any API calls:

```bash
reach lint ./my-skills
```

Find which skills compete for the same vocabulary:

```bash
reach overlap ./my-skills
```

```text
 skill                           overlap   nearest rival
 ────────────────────────────────────────────────────────────────────────
 google-agents-cli-deploy           0.69   google-agents-cli-scaffold
 google-agents-cli-scaffold         0.63   google-agents-cli-deploy
 google-agents-cli-adk-code         0.60   google-agents-cli-deploy
```

Measure selection accuracy against rivals using an offline mock runtime:

```bash
reach eval ./my-skills/google-agents-cli-deploy --agent keyword
```

To probe live models, point `--agent` at `claude-code`, `antigravity-cli`, or `antigravity-sdk`.

---

## Supported Agents & Ecosystems

Reach provides two levels of integration across the AI agent ecosystem:

### 1. Live Empirical Probing (`reach eval`, `reach check`, `reach optimize`)

- **Anthropic Claude Code** (`--agent claude-code`): Live multi-turn evaluation and prompt listing budget checks via the `claude` CLI. Supports Google Cloud Model Garden on Agent Platform.
- **Google Antigravity** (`--agent antigravity-cli`, `--agent antigravity-sdk`): Live sandboxed evaluation and structured output validation via the `agy` CLI or native Python SDK. Supports Gemini API or Google Cloud Agent Platform.
- **Goose** (`--agent goose`): Live autonomous agent evaluation via the `goose` CLI ([`aaif-goose/goose`](https://github.com/aaif-goose/goose)).
- **Pi Agent Harness** (`--agent pi`): Live single-turn and multi-turn evaluation via the headless `pi` CLI ([`earendil-works/pi`](https://github.com/earendil-works/pi)) with progressive disclosure skill loading.
- **Lexical Baseline**: `--agent keyword` (in-memory lexical BM25 matching driver that evaluates prompt routing without subprocesses, tools, or API tokens).

### 2. Multi-Agent Discovery & Static Analysis (`reach lint`, `reach overlap`, `--global`)

Reach automatically discovers, lints, and inspects skill definitions across:

- **Universal Agent Skills Standard** (`.agents/skills/`, `~/.agents/skills/`)
- **Anthropic Claude Code** (`.claude/skills/`, `~/.claude/skills/`)
- **Google Antigravity** (`.agents/skills/`, `~/.agents/skills/`)
- **Goose** (`.agents/skills/`, `~/.agents/skills/`)
- **Cursor** (`.cursor/skills/`, `~/.cursor/skills/`)
- **GitHub Copilot / VS Code** (`.github/skills/`, `~/.copilot/skills/`)
- **Pi** (`.pi/skills/`, `~/.pi/agent/skills/`)
- **OpenAI Codex** (`.agents/skills/`, `~/.agents/skills/`)

---

## Capabilities

### Manifest Validation ([`reach lint`](cli/lint.md))

Check `SKILL.md` frontmatter schemas, naming conventions, and runtime listing budgets.

### Collision Detection ([`reach overlap`](cli/overlap.md))

Calculate vocabulary overlap across skills to surface competing rivals before testing.

### Routing Evaluation ([`reach eval`](cli/eval.md))

Empirically probe agent runtimes to score selection accuracy, recall, and confidence intervals.

### Description Optimization ([`reach optimize`](cli/optimize.md))

Synthesize, benchmark, and apply candidate rewrites that disambiguate rival skills with multi-round hill climbing, holdout validation, and query boundary curation.

### CI Gates ([`reach check`](cli/check.md))

Run a two-stage pipeline combining fast static linting with empirical regression tests.

### Regression Analysis ([`reach diff`](cli/diff.md))

Compare evaluation runs to measure routing deltas against the statistical noise floor.

### Dataset Management ([`reach query`](cli/query.md))

Draft synthetic queries, export to CSV for human review, and import benchmark sets.

### Reporting ([`reach view`](cli/view.md))

Generate terminal scorecards or standalone interactive HTML reports from run artifacts.

---

## How It Works

- **Progressive disclosure awareness**: Agents only load skill names and descriptions during the initial prompt scan (Level 1). Full markdown instructions are only read if a skill is activated (Level 2). `skill-reach` focuses on this critical Level 1 selection surface.
- **Empirical decisions over word counts**: Lexical analysis ranks likely competitors, but empirical probes test actual model routing behavior.
- **Zero heavy infrastructure**: Runs locally as a lightweight Python CLI via `uv` or `pipx`. Supports offline execution modes (`--dry-run`, `--agent keyword`) for instant CI runs without token costs.
- **Reproducible artifacts**: Every evaluation run produces an immutable JSON artifact recording configuration fingerprints, corpus digests, query hashes, and complete confusion pairs.

---

## Next Steps

- Follow the [Getting Started](getting-started.md) tutorial to run your first evaluation.
- Browse the [CLI Reference](cli/index.md) for detailed flag tables and options for every command.
- Read [How Reachability Works](concepts/how-it-works.md) for background on metrics, scoring intervals, and experimental methodology.
- Read [Progressive Disclosure](concepts/progressive-disclosure.md) to understand how agent runtimes manage skill context limits.
