# Configuration Reference

`skill-reach` can be configured via a project-local `reach.toml` file, environment variables, or CLI arguments.

When you run any `reach` command, it loads the bundled base configuration, overlays your project-level `reach.toml` (if present), and applies any environment variables or CLI flags passed at runtime.

To customize settings for your repository, copy the included template to your project root:

```bash
cp reach.example.toml reach.toml
```

---

## The `reach.toml` Specification

Place a `reach.toml` file in your repository root, or specify a custom path with `--config <path>`:

```toml
# ==============================================================================
# General Settings
# ==============================================================================
[general]
default_agent = "antigravity-cli"


# ==============================================================================
# Catalog Assembly
# ==============================================================================
[catalog]
mode = "neighborhood"    # Assembly strategy: all, neighborhood, singleton
rivals = 10              # Number of competitor skills sampled around the target
scorer = "hybrid"        # Rival selection metric: hybrid, dense, bm25
seed = 0                 # Deterministic seed for reproducible catalog sampling
size = 20                # Total resident skills installed per probe trial

# ==============================================================================
# CI/CD Quality Gate (`reach check`)
# ==============================================================================
[check]
budget = 50              # Probe limit budget for Stage 2 empirical evaluation
max_misroute = 0.10      # Maximum permitted fraction of misrouted queries
max_redundancy = 0.25    # Maximum acceptable skill redundancy (excess calls >= 0.0, optional)
min_accuracy = 0.80      # Minimum overall routing accuracy (0.0 - 1.0)
min_efficiency = 0.80    # Minimum observed step efficiency MRR (0.0 - 1.0, optional)
min_entrypoint = 0.80    # Minimum observed entrypoint accuracy (0.0 - 1.0, optional)
min_f1 = 0.85            # Minimum observed skill selection F1 score (0.0 - 1.0, optional)
min_reachability = 0.90  # Minimum observed trajectory reachability (0.0 - 1.0, optional)
min_recall = 0.80        # Minimum recall rate for target skills (0.0 - 1.0)
since = "HEAD~1"         # Default git base reference when running --changed
strict = true            # When true, Stage 1 static lint warnings fail the build

# ==============================================================================
# A/B Diffing & Noise Floor Calibration
# ==============================================================================
[diff]
confidence = 0.95
noise_inflation = 1.265
power = 0.80

# ==============================================================================
# Discovery Precedence Order
# ==============================================================================
[discovery]
# Candidate paths checked in order; the first one found with skills in the workspace is used.
# Supports "." (single skill at root), "skills" (generic project folder), explicit directory paths (e.g. ".agents/skills"), or agent/client names ("claude-code", "cursor", "github", "copilot", "pi", "goose", "codex", "agents").
precedence = [
    ".",
    "skills",
    ".agents/skills",
    "claude-code",
    "cursor",
    "github",
    "pi",
    "goose",
]

# ==============================================================================
# Static Linting
# ==============================================================================
[lint]
max_description_length = 1024
max_name_length = 64
min_description_length = 20
mutual_handoff_lexical_threshold = 0.35
mutual_handoff_similarity_threshold = 0.75

[lint.rules]
catalog-budget-overflow = "warn"
description-too-short = "warn"
duplicate-capability = "warn"
duplicate-name = "error"
invalid-name-format = "error"
invalid-yaml = "error"
listing-overflow = "warn"
lockfile-drift = "warn"
missing-description = "error"
missing-mutual-handoff = "warn"
missing-name = "error"
name-mismatch = "error"
reserved-name-collision = "warn"
unbounded-attractor = "warn"
unknown-skill-reference = "warn"
unresolved-declared-dependency = "warn"
unresolved-placeholder = "warn"

# ==============================================================================
# Description Optimization
# ==============================================================================
[optimize]
adversarial_count = 5
auto_queries = true
budget = 30
holdout = 0.2
iterations = 1
positive_count = 5
review = false
review_timeout = 600.0
seed = 42
temperature = 0.7
with_handoff = false
workers = 4

# ==============================================================================
# Lexical Overlap & Vocabulary Rewrite Heuristics
# ==============================================================================
[overlap]
claim_limit = 8
contender_band = 0.90
material_share = 0.01
min_claim_length = 3
min_claim_uses = 2

# ==============================================================================
# Probe Planning & Rate Limiting
# ==============================================================================
[plan]
attempts = 5             # Replicate probes per query for confidence interval sizing
backoff_s = 5.0          # Exponential backoff duration in seconds
pause_s = 0.0            # Delay between successive probes to prevent rate limits
retries = 2              # Retry attempts on transient API failures

# ==============================================================================
# Query Synthesis & Difficulty
# ==============================================================================
[query]
adversarial_count = 1
count = 3
distinctive_idf_floor = 0.693147
top_rivals = 3

# ==============================================================================
# Semantic, Lexical & Hybrid Retrieval
# ==============================================================================
[retrieval]
bm25_b = 0.75
bm25_k1 = 1.5
model = "minishlab/potion-retrieval-32M"
rrf_k = 60
scorer = "hybrid"
similarity_threshold = 0.92

# ==============================================================================
# Runtime Execution
# ==============================================================================
[runtime]
early_exit = true        # Abort multi-turn execution immediately when target skill is observed
max_turns = 3            # Maximum turns for multi-turn evaluations
timeout_s = 200          # Maximum seconds to wait for an agent probe response

# ==============================================================================
# Study Inputs, Workspaces & Artifacts
# ==============================================================================
[study]
# catalog = "auto"             # Target catalog scope: "auto" (default), "all", "neighborhood:<skill>", or "singleton:<skill>"
# out = ".reach/eval.json"     # Destination for evaluation run results
# partial = false              # Enforce query set coverage across all catalog skills
# queries = ".reach/queries.json"  # Labeled evaluation queries benchmark file
# rescope = false              # Prevent cross-catalog label reuse without --rescope
# skills = "skills"            # Optional explicit override (omit to use [discovery].precedence)
# tag = ""                     # Semantic label for run tracking (e.g. "v1-baseline")
# workdir = "work"             # Custom persistent workspace (omit to use isolated ephemeral tempdir)

# ==============================================================================
# Agent Profiles & Executables
# ==============================================================================
[agents.antigravity-cli]
default_model = "gemini-3.8-flash"
executable = "agy"
models = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]
skills_dir = ".agents/skills"
user_skills_dir = ".agents/skills"

[agents.antigravity-sdk]
default_model = "gemini-3.8-flash"
models = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]
skills_dir = ".agents/skills"

[agents.claude-code]
default_model = "claude-sonnet-5"
executable = "claude"
models = ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"]
skills_dir = ".claude/skills"
user_skills_dir = ".claude/skills"

[agents.goose]
default_model = "gemini-3.6-flash"
executable = "goose"
models = ["gemini-3.6-flash", "gemini-3.1-flash-lite"]
skills_dir = ".agents/skills"
user_skills_dir = ".agents/skills"

[agents.pi]
default_model = "gemini-3.8-flash"
default_provider = "google"
executable = "pi"
models = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]
skills_dir = ".pi/skills"
user_skills_dir = ".pi/agent/skills"

# ==============================================================================
# Model Context Windows & Listing Budgets
# ==============================================================================
[models.gemini-3-8-flash]
effort = "low"

[models.claude-sonnet-5]
chars_per_token = 3.0
completion_window = 200_000
context_window = 1_000_000
effort = "low"
```

---

## Sections & Options Reference

### `[catalog]`

Controls how competing skills are selected and installed into the active catalog during empirical probes.

| Key      | Type    | Default          | Description                                                                                                         |
| :------- | :------ | :--------------- | :------------------------------------------------------------------------------------------------------------------ |
| `mode`   | String  | `"neighborhood"` | Assembly mode: `all` (install whole catalog), `neighborhood` (target + nearest rivals), `singleton` (target alone). |
| `size`   | Integer | `20`             | Total number of resident skills in the synthetic catalog.                                                           |
| `rivals` | Integer | `10`             | Number of competing skills selected by vocabulary or semantic proximity.                                            |
| `seed`   | Integer | `0`              | Deterministic random seed for catalog permutation.                                                                  |
| `scorer` | String  | `"hybrid"`       | Scoring method for selecting nearest rivals: `hybrid`, `dense`, or `bm25`.                                          |

### `[check]`

Default thresholds enforced by `reach check` in continuous integration.

| Key                | Type    | Default    | Description                                                                  |
| :----------------- | :------ | :--------- | :--------------------------------------------------------------------------- |
| `budget`           | Integer | `50`       | Maximum empirical probes executed during CI evaluations.                     |
| `max_misroute`     | Float   | `0.10`     | Fail gate if queries misroute to competitor skills above this fraction.      |
| `max_redundancy`   | Float   | `None`     | Fail gate if skill redundancy exceeds this fraction (excess calls).          |
| `min_accuracy`     | Float   | `0.80`     | Fail gate if overall routing accuracy drops below this threshold.            |
| `min_efficiency`   | Float   | `None`     | Fail gate if observed step efficiency MRR drops below this threshold.        |
| `min_entrypoint`   | Float   | `None`     | Fail gate if observed entrypoint accuracy drops below this threshold.        |
| `min_f1`           | Float   | `None`     | Fail gate if observed skill selection F1 score drops below this threshold.   |
| `min_reachability` | Float   | `None`     | Fail gate if observed trajectory reachability drops below this threshold.    |
| `min_recall`       | Float   | `0.80`     | Fail gate if target skill recall drops below this threshold.                 |
| `since`            | String  | `"HEAD~1"` | Default git revision comparison target when `--changed` is passed.           |
| `strict`           | Boolean | `true`     | When true, Stage 1 static lint warnings cause the check to exit with code 1. |

### `[diff]`

Statistical parameters for A/B evaluation diffing and noise floor estimation.

| Key               | Type  | Default | Description                                                                    |
| :---------------- | :---- | :------ | :----------------------------------------------------------------------------- |
| `confidence`      | Float | `0.95`  | Two-sided confidence level ($0 < c < 1$) for statistical intervals.            |
| `power`           | Float | `0.80`  | Statistical power ($1 - \beta$) for sample size planning.                      |
| `noise_inflation` | Float | `1.265` | Inflation multiplier applied to standard errors for empirical over-dispersion. |

### `[discovery]`

Controls directory and client skill store search precedence for auto-discovering skills.

| Key          | Type            | Default                                                                               | Description                                            |
| :----------- | :-------------- | :------------------------------------------------------------------------------------ | :----------------------------------------------------- |
| `precedence` | Array of String | `[".", "skills", ".agents/skills", "claude-code", "cursor", "github", "pi", "goose"]` | Ordered search locations when resolving skill corpora. |

### `[general]`

| Key             | Type   | Default             | Description                                                                                                                                       |
| :-------------- | :----- | :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| `default_agent` | String | `"antigravity-cli"` | Default agent driver when `--agent` is omitted from CLI commands (`antigravity-cli`, `antigravity-sdk`, `claude-code`, `goose`, `pi`, `keyword`). |

### `[lint]` & `[lint.rules]`

Controls static frontmatter and budget thresholds.

| Key                                   | Type    | Default  | Description                                                                                          |
| :------------------------------------ | :------ | :------- | :--------------------------------------------------------------------------------------------------- |
| `max_description_length`              | Integer | `1024`   | Maximum allowable character length for description before triggering `listing-overflow`.             |
| `min_description_length`              | Integer | `20`     | Minimum character length before triggering `description-too-short`.                                  |
| `max_name_length`                     | Integer | `64`     | Maximum character length for skill name.                                                             |
| `catalog_budget_chars`                | Integer | `30000`  | Maximum allowable resident listing budget in characters before triggering `catalog-budget-overflow`. |
| `mutual_handoff_similarity_threshold` | Float   | `0.75`   | Minimum semantic similarity between neighbors before requiring reciprocal handoffs.                  |
| `mutual_handoff_lexical_threshold`    | Float   | `0.35`   | Minimum lexical competition score before requiring reciprocal handoffs.                              |
| `rules.<rule-name>`                   | String  | (varies) | Severity override for any static lint rule: `"error"`, `"warn"`, or `"ignore"`.                      |

### `[optimize]`

Parameters for closed-loop skill description optimization.

| Key                 | Type    | Default | Description                                                                   |
| :------------------ | :------ | :------ | :---------------------------------------------------------------------------- |
| `adversarial_count` | Integer | `5`     | Number of adversarial negative near-miss queries to synthesize per round.     |
| `auto_queries`      | Boolean | `true`  | Automatically synthesize positive and adversarial queries when none provided. |
| `budget`            | Integer | `30`    | Maximum empirical probe budget allocated across candidate evaluations.        |
| `holdout`           | Float   | `0.2`   | Fraction of queries held out for generalization validation (`0.0` - `0.9`).   |
| `iterations`        | Integer | `1`     | Number of iterative hill-climbing refinement rounds (1-10).                   |
| `positive_count`    | Integer | `5`     | Number of positive in-scope trigger queries to synthesize per round.          |
| `review`            | Boolean | `false` | Launch interactive browser boundary review for drafted queries before probes. |
| `review_timeout`    | Float   | `600.0` | Maximum timeout in seconds waiting for interactive browser query review.      |
| `seed`              | Integer | `42`    | Pseudo-random seed for train/test query splitting and reproducible runs.      |
| `temperature`       | Float   | `0.7`   | Sampling temperature for candidate rewrite generation.                        |
| `with_handoff`      | Boolean | `false` | Synthesize and stage reciprocal Layer-2 `SKILL.md` Routing Notes.             |
| `workers`           | Integer | `4`     | Number of parallel probe workers (inherits from `[plan].workers` if unset).   |

### `[overlap]`

Parameters governing lexical competition and vocabulary rewrite suggestions.

| Key                | Type    | Default | Description                                                                 |
| :----------------- | :------ | :------ | :-------------------------------------------------------------------------- |
| `contender_band`   | Float   | `0.90`  | Proximity ratio (within 90% of top score) defining close competitor skills. |
| `material_share`   | Float   | `0.01`  | Minimum contribution share (1%) for a rival term to be reported as ceded.   |
| `claim_limit`      | Integer | `8`     | Maximum number of suggested unclaimed terms extracted from a skill body.    |
| `min_claim_length` | Integer | `3`     | Minimum character length for suggested unclaimed body terms.                |
| `min_claim_uses`   | Integer | `2`     | Minimum frequency in the skill body to qualify as a claim candidate.        |

### `[plan]`

Controls probe replication and network resilience.

| Key         | Type    | Default | Description                                                                           |
| :---------- | :------ | :------ | :------------------------------------------------------------------------------------ |
| `attempts`  | Integer | `5`     | Number of repeated trials per query to compute statistical confidence intervals.      |
| `retries`   | Integer | `2`     | Number of times to retry a probe if an agent throws a transient API or network error. |
| `backoff_s` | Float   | `5.0`   | Initial backoff time in seconds between retries.                                      |
| `pause_s`   | Float   | `0.0`   | Delay between consecutive probes to respect provider rate limits.                     |
| `workers`   | Integer | `1`     | Number of concurrent worker threads executing probes in parallel.                     |

### `[query]`

Parameters for synthetic query drafting and leakage detection.

| Key                     | Type    | Default    | Description                                                               |
| :---------------------- | :------ | :--------- | :------------------------------------------------------------------------ |
| `count`                 | Integer | `3`        | Number of positive queries drafted per target skill.                      |
| `adversarial_count`     | Integer | `1`        | Number of negative near-miss queries drafted per target skill.            |
| `top_rivals`            | Integer | `3`        | Maximum competitor skills injected into synthesis prompts.                |
| `distinctive_idf_floor` | Float   | `0.693147` | Minimum IDF threshold ($\ln(2)$) for distinctive terms in leak detection. |

### `[registry]`

Parameters for Google Cloud Agent Registry integration.

| Key                 | Type    | Default    | Description                                                                         |
| :------------------ | :------ | :--------- | :---------------------------------------------------------------------------------- |
| `project`           | String  | `None`     | Default Google Cloud project ID hosting the Agent Registry.                         |
| `location`          | String  | `"global"` | Agent Registry regional endpoint location (`global`, `us`, `eu`).                   |
| `publisher`         | String  | `None`     | Optional publisher filter (e.g. `cloud.google.com`).                                |
| `cache_ttl_seconds` | Integer | `300`      | Local cache TTL in seconds for remote registry skill metadata before re-validating. |

### `[retrieval]`

Parameters for dense embedding models, BM25 lexical scoring, and hybrid reciprocal rank fusion (RRF).

| Key                    | Type    | Default                            | Description                                                              |
| :--------------------- | :------ | :--------------------------------- | :----------------------------------------------------------------------- |
| `scorer`               | String  | `"hybrid"`                         | Rival retrieval algorithm (`hybrid`, `dense`, `bm25`).                   |
| `model`                | String  | `"minishlab/potion-retrieval-32M"` | Sentence transformer embedding model for dense similarity ranking.       |
| `rrf_k`                | Integer | `60`                               | Smoothing constant $k$ used in Reciprocal Rank Fusion ($1 / (k + r)$).   |
| `similarity_threshold` | Float   | `0.92`                             | Cosine similarity threshold for identifying near-duplicate capabilities. |
| `bm25_k1`              | Float   | `1.5`                              | Lucene BM25 term-frequency saturation parameter ($k_1 > 0$).             |
| `bm25_b`               | Float   | `0.75`                             | Lucene BM25 document length normalization parameter ($0 \le b \le 1$).   |

### `[runtime]`

| Key                | Type             | Default | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| :----------------- | :--------------- | :------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `timeout_s`        | Integer          | `200`   | Process execution timeout in seconds before aborting an unresponsive probe.                                                                                                                                                                                                                                                                                                                                                                                              |
| `max_turns`        | Integer          | `3`     | Maximum conversation turns to execute and evaluate per probe.                                                                                                                                                                                                                                                                                                                                                                                                            |
| `early_exit`       | Boolean          | `true`  | When true, aborts probe execution immediately when the target skill is invoked.                                                                                                                                                                                                                                                                                                                                                                                          |
| `blocked_env_vars` | Sequence[String] | `None`  | Explicit list of ambient environment variables to strip from child agent processes. When omitted, Reach's default sensitive credentials are stripped (with automatic exemption of `GOOGLE_APPLICATION_CREDENTIALS` when Google Cloud Model Garden or Google Enterprise mode is active; runner configuration variables such as `CLAUDE_CODE_USE_VERTEX`, `ANTHROPIC_VERTEX_PROJECT_ID`, and `CLOUD_ML_REGION` are preserved). Set to `[]` to allow all ambient variables. |

### `[study]`

Controls default file paths for benchmark queries, skill roots, workspaces, and evaluation artifacts.

| Key            | Type                                | Default  | Description                                                                                                                |
| :------------- | :---------------------------------- | :------- | :------------------------------------------------------------------------------------------------------------------------- |
| `anchor`       | Integer / Sequence[String] / String | `None`   | Anchor skills cohort evaluated across all scales (count, skill names list, or `"all"`).                                    |
| `auto_queries` | Boolean                             | `true`   | Automatically synthesize and backfill benchmark queries for unqueried anchor skills during scaling sweeps.                 |
| `catalog`      | String                              | `"auto"` | Target catalog scope: `"auto"` (derives from dataset/target), `"all"`, `"neighborhood:<skill>"`, or `"singleton:<skill>"`. |
| `early_stop`   | Boolean                             | `true`   | When true, terminates scaling sweeps early if $F_1$ upper CI drops below threshold.                                        |
| `out`          | Path                                | `None`   | Destination file path for recorded evaluation artifacts.                                                                   |
| `partial`      | Boolean                             | `false`  | Allow query sets that evaluate only a subset of resident skills.                                                           |
| `queries`      | Path                                | `None`   | Path to labeled evaluation queries JSON benchmark file.                                                                    |
| `rescope`      | Boolean                             | `false`  | Permit evaluation against a catalog differing from the one labeled in.                                                     |
| `scales`       | Sequence[Integer]                   | `None`   | Pre-configured catalog sizes for scaling sweeps (e.g. `[10, 25, 50, 100]`).                                                |
| `skills`       | Path                                | `None`   | Explicit path override to local skills directory (bypasses `[discovery].precedence` if set).                               |
| `tag`          | String                              | `""`     | Semantic run tracking label (e.g. `"v1-baseline"`).                                                                        |
| `trusted`      | Boolean                             | `false`  | When true, trusts resident skills and bypasses interactive safety confirmation prompts.                                    |
| `workdir`      | Path                                | `None`   | Custom persistent workspace path (defaults to isolated ephemeral temp directory).                                          |

> [!CAUTION]
> **Risk of Bypassing Safety Confirmation**
> Setting `trusted = true` bypasses interactive safety confirmation prompts across all commands that launch live agent probes. **Only enable `trusted = true` in private repositories where all skill manifests and instructions have been vetted and reviewed.** Never enable `trusted = true` on repositories that evaluate untrusted or community-contributed skills.

---

## Configuration Precedence

Settings resolve in the following order (highest precedence wins):

1. **Explicit CLI flags** (e.g. `--min-recall 0.90`)
2. **Project-local `reach.toml`** (in the current working directory, or specified by `--config`)
3. **Bundled default `reach.toml`**

Configuration values in `reach.toml` can also interpolate environment variables dynamically using `${VAR}` syntax (e.g. `skills = "${REACH_SKILL_ROOT}"`).

---

## Environment Variables

| Variable              | Description                                                                                       |
| :-------------------- | :------------------------------------------------------------------------------------------------ |
| `REACH_NO_BROWSER`    | Set to `"1"` or `"true"` to bypass interactive browser review for drafted queries.                |
| `REACH_YES`           | Set to `"1"` or `"true"` to bypass interactive safety confirmation prompts in CI/CD and scripts.  |
| `REACH_FORCE`         | Set to `"1"` or `"true"` as an alias to bypass interactive safety confirmation prompts.           |
| `GITHUB_STEP_SUMMARY` | When set (in GitHub Actions), `reach check` automatically writes markdown summaries to this file. |
| `NO_MKDOCS_2_WARNING` | Set to `"1"` to suppress upstream MkDocs 2.0 console notices during documentation builds.         |

---

## Provider API Keys & Authentication

Reach automatically routes model API keys to the corresponding environment variables expected by each runtime agent:

| Provider            | Credentials / Injected Environment Variables | Support Tier               |
| :------------------ | :------------------------------------------- | :------------------------- |
| `google` / `gemini` | `GEMINI_API_KEY`, `GOOGLE_API_KEY`           | Tested (Primary reference) |
| `google-cloud`      | Application Default Credentials (ADC)        | Tested (Agent Platform)    |

> [!NOTE]
> **Provider Support Status**: Google Gemini (via Developer API keys) and Google Cloud Agent Platform / Model Garden (via Application Default Credentials) are the tested and benchmarked authentication paths for Reach. For Claude Code, configure Google Cloud Model Garden on Agent Platform (`CLAUDE_CODE_USE_VERTEX=1`).

If an unrecognized provider name is specified via options, Reach raises an error rather than mapping credentials to an unintended provider. For custom, local, or self-hosted model engines (such as Ollama, vLLM, or Mistral), set the provider's expected environment variables directly in your shell or CI workflow.

---

## Execution Security & Trust Boundary

Reach executes agent command-line interfaces (CLIs) and tools directly with the host permissions of the running user. Child agent processes inherit the host environment so tools, compilers, language runtimes, and local developer configuration remain operational.

> [!WARNING]
> **Evaluating Untrusted Skills**: Agent runtimes possess tool-use capabilities that can access the local filesystem and network. When benchmarking or evaluating skills from untrusted third-party repositories, public pull requests, or external registries, run Reach inside an isolated container (such as Docker), a disposable virtual machine, or a dedicated CI runner.
