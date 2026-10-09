# Configuration Reference

`skill-reach` can be configured via a project-local `reach.toml` file, environment variables, or CLI arguments.

When you run any `reach` command, it loads the bundled base configuration, overlays your project-level `reach.toml` (if present), and applies any environment variables or CLI flags passed at runtime.

To customize settings for your repository, copy the included template to your project root (which is gitignored by default so your local paths and credentials remain private):

```bash
cp reach.example.toml reach.toml
```

---

## The `reach.toml` Specification

Place a `reach.toml` file in your repository root, or specify a custom path with `--config <path>`. All sections and keys are optional—include only the settings you want to override from [reach.example.toml](https://github.com/google/skill-reach/blob/main/reach.example.toml):

```toml
[general]
default_agent = "antigravity-cli"

[catalog]
# mode = "neighborhood"  # Assembly strategy: all, neighborhood, singleton, sweep
rivals = 10              # Number of competitor skills sampled around the target
scorer = "hybrid"        # Rival selection metric: hybrid, dense, bm25
seed = 0                 # Deterministic seed for reproducible catalog sampling
size = 20                # Total resident skills installed per probe trial

[check]
budget = 50              # Probe limit budget for Stage 2 empirical evaluation
max_misroute = 0.10      # Maximum permitted fraction of misrouted queries
min_accuracy = 0.80      # Minimum overall routing accuracy (0.0 - 1.0)
min_recall = 0.80        # Minimum recall rate for target skills (0.0 - 1.0)
# Optional multi-step trajectory quality thresholds (unset / None by default):
# max_redundancy = 0.25
# min_efficiency = 0.80
# min_entrypoint = 0.80
# min_f1 = 0.85
# min_reachability = 0.90
since = "HEAD~1"         # Default git base reference when running --changed
strict = true            # When true, Stage 1 static lint warnings fail the build

[discovery]
# Supports "." (single skill at root), "skills" (generic project folder), explicit
# paths (e.g. ".agents/skills"), or client profile names ("agents", "antigravity-cli",
# "antigravity-sdk", "claude-code", "codex", "copilot", "cursor", "github", "goose", "pi").
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
empty-skill-directory = "info"
invalid-name-format = "error"
invalid-yaml = "error"
listing-overflow = "warn"
lockfile-drift = "info"
missing-description = "error"
missing-mutual-handoff = "warn"
missing-name = "error"
name-mismatch = "error"
reserved-name-collision = "warn"
unbounded-attractor = "warn"
unknown-skill-reference = "warn"
unresolved-declared-dependency = "warn"
unresolved-placeholder = "warn"

[plan]
# Mode-dependent default when omitted: 5 (formal eval/check/diff), 3 (quick eval), 1 (sweep/keyword)
# attempts = 5
backoff_s = 5.0
pause_s = 0.0
retries = 2

[runtime]
early_exit = true        # Abort multi-turn execution immediately when target skill is observed
max_turns = 3            # Maximum turns for multi-turn evaluations
timeout_s = 200          # Maximum seconds to wait for an agent probe response

# Bundled Agent Profiles (customizable per agent under [agents.<name>])
[agents.antigravity-cli]
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
models = ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"]
skills_dir = ".claude/skills"
user_skills_dir = ".claude/skills"

[agents.goose]
default_model = "gemini-3.8-flash"
models = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]
skills_dir = ".agents/skills"
user_skills_dir = ".agents/skills"

[agents.pi]
default_model = "gemini-3.8-flash"
default_provider = "google"
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

# Model Context Windows & Listing Budgets (dots in model names replaced with hyphens)
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

### `[agents.<name>]`

Configures built-in or custom agent runtime profiles (`AgentProfile`), including supported model identifiers and project/user skill discovery directories. Note that CLI binary overrides (`executable`) belong under `[runtime.options]`, not `[agents.<name>]`.

| Key                | Type             | Default | Description                                                                                  |
| :----------------- | :--------------- | :------ | :------------------------------------------------------------------------------------------- |
| `default_model`    | String           | `""`    | Default model identifier used when probing with this agent runtime.                          |
| `default_provider` | String / `None`  | `None`  | Optional default model provider name (e.g. `"google"` for `pi`).                             |
| `models`           | Sequence[String] | `()`    | Model identifiers supported by this agent runtime for `--model` auto-routing.                |
| `skills_dir`       | String           | `""`    | Relative project workspace directory where this agent discovers resident skills.             |
| `user_skills_dir`  | String / `None`  | `None`  | Relative path under `~` where this agent discovers user-global skills when `--global` is on. |

### `[catalog]`

Controls how competing skills are selected and installed into the active catalog during empirical probes.

| Key      | Type    | Default          | Description                                                                                                                                      |
| :------- | :------ | :--------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`   | String  | `"neighborhood"` | Assembly mode: `all` (install whole catalog), `neighborhood` (target + nearest rivals), `singleton` (target alone), or `sweep` (scaling curves). |
| `size`   | Integer | `20`             | Total number of resident skills in the synthetic catalog.                                                                                        |
| `rivals` | Integer | `10`             | Number of competing skills selected by vocabulary or semantic proximity.                                                                         |
| `seed`   | Integer | `0`              | Deterministic random seed for catalog permutation.                                                                                               |
| `scorer` | String  | `"hybrid"`       | Scoring method for selecting nearest rivals: `hybrid`, `dense`, or `bm25`.                                                                       |

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

Controls directory and client skill store search precedence for auto-discovering skills. Supports `"."`, `"skills"`, explicit relative paths, or any of the 10 built-in client profiles (`"agents"`, `"antigravity-cli"`, `"antigravity-sdk"`, `"claude-code"`, `"codex"`, `"copilot"`, `"cursor"`, `"github"`, `"goose"`, `"pi"`).

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
| `rules.<rule-name>`                   | String  | (varies) | Severity override for any static lint rule: `"error"`, `"warn"`, `"info"`, or `"ignore"`.            |

/// note | Interaction with `[catalog]` in `reach.toml`
When a project's `reach.toml` explicitly includes a `[catalog]` table (with the default `scorer = "hybrid"`) and omits `catalog_budget_chars` under `[lint]`, `load_config()` sets `lint.catalog_budget_chars = None` unless `catalog_budget_chars` is explicitly specified in `[lint]`.
///

### `[models.<name>]`

Configures context window limits, character-to-token ratios, and reasoning effort tiers (`ModelProfile`). Dots in model names are replaced with hyphens in table headers (e.g., `[models.gemini-3-8-flash]`).

| Key                       | Type            | Default     | Description                                                                                      |
| :------------------------ | :-------------- | :---------- | :----------------------------------------------------------------------------------------------- |
| `context_window`          | Integer         | `1_000_000` | Total model context window in tokens.                                                            |
| `completion_window`       | Integer         | `65_536`    | Maximum output completion window in tokens (`200_000` for Claude 4.5/5 models).                  |
| `chars_per_token`         | Float           | `4.0`       | Estimated characters per token used for catalog budget calculations (`3.0` for Claude models).   |
| `listing_budget_fraction` | Float           | `0.01`      | Fraction of `context_window` allocated to the resident skill listing table (`0 < f \le 1`).      |
| `budget_fraction_places`  | Integer         | `4`         | Decimal rounding precision when formatting listing budget fractions (`1`–`10`).                  |
| `effort`                  | String / `None` | `None`      | Default reasoning effort level (`"minimal"`, `"low"`, `"medium"`, `"high"`, `"max"`, `"xhigh"`). |

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

| Key         | Type    | Default | Description                                                                                                                                                      |
| :---------- | :------ | :------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `attempts`  | Integer | `5`     | Replicate trials per query (`5` for formal `eval`/`check`/`diff`, `3` for quick mode, `1` for `sweep` and deterministic `keyword` unless explicitly overridden). |
| `retries`   | Integer | `2`     | Number of times to retry a probe if an agent throws a transient API or network error.                                                                            |
| `backoff_s` | Float   | `5.0`   | Initial backoff time in seconds between retries.                                                                                                                 |
| `pause_s`   | Float   | `0.0`   | Delay between consecutive probes to respect provider rate limits.                                                                                                |
| `workers`   | Integer | `1`     | Number of concurrent worker threads executing probes in parallel.                                                                                                |

### `[query]`

Parameters for synthetic query drafting and leakage detection.

| Key                     | Type    | Default    | Description                                                               |
| :---------------------- | :------ | :--------- | :------------------------------------------------------------------------ |
| `count`                 | Integer | `3`        | Number of positive queries drafted per target skill.                      |
| `adversarial_count`     | Integer | `1`        | Number of negative near-miss queries drafted per target skill.            |
| `top_rivals`            | Integer | `3`        | Maximum competitor skills injected into synthesis prompts.                |
| `distinctive_idf_floor` | Float   | `0.693147` | Minimum IDF threshold ($\ln(2)$) for distinctive terms in leak detection. |

### `[registry]`

Parameters for Google Cloud Agent Registry integration. When `--skills` or `[study].skills` is set without an explicit `--registry` or `--project` CLI flag, local skills take precedence over `[registry].project`. For `antigravity-sdk`, `[registry].project` and `[registry].location` also provide the default Agent Platform project and location when no Gemini API key is configured.

| Key                 | Type    | Default    | Description                                                                             |
| :------------------ | :------ | :--------- | :-------------------------------------------------------------------------------------- |
| `project`           | String  | `None`     | Default Google Cloud project ID hosting the Agent Registry and Agent Platform fallback. |
| `location`          | String  | `"global"` | Agent Registry regional endpoint location (`global`, `us`, `eu`).                       |
| `publisher`         | String  | `None`     | Optional publisher filter (e.g. `cloud.google.com`).                                    |
| `cache_ttl_seconds` | Integer | `300`      | Local cache TTL in seconds for remote registry skill metadata before re-validating.     |

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

| Key                | Type             | Default | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| :----------------- | :--------------- | :------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `timeout_s`        | Integer          | `200`   | Process execution timeout in seconds before aborting an unresponsive probe.                                                                                                                                                                                                                                                                                                                                                                                                                |
| `max_turns`        | Integer          | `3`     | Maximum conversation turns to execute and evaluate per probe.                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `early_exit`       | Boolean          | `true`  | When true, aborts probe execution immediately when the target skill is invoked.                                                                                                                                                                                                                                                                                                                                                                                                            |
| `blocked_env_vars` | Sequence[String] | `None`  | Explicit list of ambient environment variables to strip from child agent processes. When omitted, Reach's default sensitive credentials are stripped (with automatic exemption of `GOOGLE_APPLICATION_CREDENTIALS` when Google Cloud Model Garden, Agent Platform, or Antigravity ADC mode is active; runner configuration variables such as `AGY_ADC_AUTH`, `GOOGLE_GENAI_USE_ENTERPRISE`, `CLAUDE_CODE_USE_VERTEX`, `ANTHROPIC_VERTEX_PROJECT_ID`, and `CLOUD_ML_REGION` are preserved). |

Agent-specific driver options are configured under `[runtime.options]`. See [Driver Options (`[runtime.options]`)](#driver-options-runtimeoptions) and the [Sandboxing Guide](guides/sandboxing.md) for full driver configuration details.

### `[study]`

Controls default file paths for benchmark queries, skill roots, workspaces, and evaluation artifacts.

| Key                    | Type                                | Default  | Description                                                                                                                                                     |
| :--------------------- | :---------------------------------- | :------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `anchor`               | Integer / Sequence[String] / String | `None`   | Anchor skills cohort evaluated across all scales (count, skill names list, or `"all"`).                                                                         |
| `auto_queries`         | Boolean                             | `true`   | Automatically synthesize benchmark queries on cold start or for explicitly requested unqueried skills; existing non-empty query sets in sweeps are used as-is.  |
| `bootstrap_iterations` | Integer                             | `200`    | Number of bootstrap replicates for curve confidence intervals (min: 10).                                                                                        |
| `bootstrap_seed`       | Integer / `None`                    | `42`     | Random seed for reproducible bootstrap resamples and curve perturbation.                                                                                        |
| `catalog`              | String                              | `"auto"` | Target catalog scope: `"auto"` (derives from dataset/target), `"all"`, `"neighborhood:<skill>"`, or `"singleton:<skill>"`.                                      |
| `catalog_replicates`   | Integer                             | `1`      | Number of independent distractor catalog permutations to probe per scale step (1–20) for replicate collision detection.                                         |
| `out`                  | Path                                | `None`   | Destination file path for recorded evaluation artifacts (`.jsonl` paths stream per-probe records in `reach eval` while `reach sweep` uses `.reach/sweep.json`). |
| `partial`              | Boolean                             | `false`  | Allow query sets that evaluate only a subset of resident skills.                                                                                                |
| `queries`              | Path                                | `None`   | Path to labeled evaluation queries JSON benchmark file.                                                                                                         |
| `rescope`              | Boolean                             | `false`  | Permit evaluation against a catalog differing from the one labeled in.                                                                                          |
| `scales`               | Sequence[Integer]                   | `None`   | Pre-configured catalog sizes for scaling sweeps (e.g. `[10, 25, 50, 100]`).                                                                                     |
| `skills`               | Path                                | `None`   | Explicit path override to local skills directory (bypasses `[discovery].precedence` if set).                                                                    |
| `tag`                  | String                              | `""`     | Semantic run tracking label (e.g. `"v1-baseline"`).                                                                                                             |
| `trusted`              | Boolean                             | `false`  | When true, trusts resident skills and bypasses interactive safety confirmation prompts.                                                                         |
| `workdir`              | Path                                | `None`   | Custom persistent workspace path (defaults to isolated ephemeral temp directory).                                                                               |

/// warning | Risk of Bypassing Safety Confirmation
Setting `trusted = true` bypasses interactive safety confirmation prompts across all commands that launch live agent probes, and agent runtimes execute with the host permissions of the running user. Only enable `trusted = true` in private repositories where all skill manifests have been reviewed; when evaluating untrusted third-party skills, run Reach inside an isolated container or VM (see the [Sandboxing & Safety Guide](guides/sandboxing.md)).
///

---

## Driver Options (`[runtime.options]`)

Driver-specific settings can be passed in `reach.toml` under `[runtime.options]` or via `--opt KEY=VALUE` on the CLI. All agent drivers inherit the base `AgentOptions` fields, CLI drivers add `CliOptions`, and Vertex-enabled drivers add `VertexOptions`.

### Common Runtime Options (`AgentOptions`, `CliOptions`, `VertexOptions`)

| Key                  | Base Class      | Type                    | Default            | Description                                                                                          |
| :------------------- | :-------------- | :---------------------- | :----------------- | :--------------------------------------------------------------------------------------------------- |
| `model`              | `AgentOptions`  | String                  | Driver default     | Model identifier to evaluate.                                                                        |
| `effort`             | `AgentOptions`  | String / `None`         | `None`             | Reasoning effort level (`"low"`, `"medium"`, `"high"`).                                              |
| `provider`           | `AgentOptions`  | String / `None`         | `None`             | Model provider override (e.g., `"google"`, `"anthropic"`, `"vertex"`).                               |
| `max_turns`          | `AgentOptions`  | Integer `>= 1`          | `3`                | Maximum number of conversation turns permitted per probe.                                            |
| `early_exit`         | `AgentOptions`  | Boolean                 | `true`             | Terminate the session immediately once the target skill is invoked.                                  |
| `allowed_tools`      | `AgentOptions`  | Sequence[String] / None | `None`             | Explicit allowlist of runtime tools exposed during probes.                                           |
| `blocked_env_vars`   | `AgentOptions`  | Sequence[String] / None | `None`             | Environment variable names stripped from agent subprocesses (overrides defaults).                    |
| `api_key`            | `AgentOptions`  | String / `None`         | `None`             | Explicit API key override for direct provider authentication.                                        |
| `use_symlinks`       | `AgentOptions`  | Boolean                 | `true` (`false`\*) | Symlink skill directories into isolated workspaces (`false` by default on `antigravity-*` runtimes). |
| `isolate_config_dir` | `AgentOptions`  | Boolean                 | `true`             | Isolate the agent's home/config directory inside the ephemeral probe workspace.                      |
| `auto_clean`         | `AgentOptions`  | Boolean                 | `false`            | Automatically delete isolated session artifacts after each probe completes.                          |
| `json_schema`        | `AgentOptions`  | String / `None`         | `None`             | Optional JSON schema string or object constraining structured agent output.                          |
| `executable`         | `CliOptions`    | String                  | Driver binary      | Path or command name for the CLI agent binary (`antigravity-cli`, `claude-code`, `goose`, `pi`).     |
| `extra_args`         | `CliOptions`    | Sequence[String]        | `()`               | Extra command-line arguments appended to every CLI subprocess invocation.                            |
| `vertex`             | `VertexOptions` | Boolean / `None`        | `None`             | Explicitly enable (`true`) or disable (`false`) Google Cloud Vertex AI / ADC authentication.         |
| `project`            | `VertexOptions` | String / `None`         | Inferred           | Google Cloud project ID for Vertex AI / Agent Platform requests.                                     |
| `location`           | `VertexOptions` | String / `None`         | `"global"` / None  | Google Cloud region or `"global"` endpoint location for Vertex AI / Agent Platform requests.         |

### Google Antigravity CLI (`antigravity-cli`)

| Key            | Type    | Default              | Description                                                                                                      |
| :------------- | :------ | :------------------- | :--------------------------------------------------------------------------------------------------------------- |
| `executable`   | String  | `"agy"`              | Path or command name for the Antigravity CLI binary.                                                             |
| `model`        | String  | `"gemini-3.8-flash"` | Model identifier to evaluate.                                                                                    |
| `vertex`       | Boolean | `None`               | Explicitly enable (`true`) or disable (`false`) Vertex AI / Agent Platform ADC authentication (`AGY_ADC_AUTH`).  |
| `project`      | String  | Inferred             | Google Cloud project ID hosting the target Agent Platform endpoints or Agent Registry.                           |
| `location`     | String  | `"global"`           | Google Cloud region or `"global"` endpoint location for Agent Platform requests.                                 |
| `use_symlinks` | Boolean | `false`              | Copy skill directories by default so Antigravity's workspace file watcher indexes resident `SKILL.md` manifests. |

### Google Antigravity SDK (`antigravity-sdk`)

| Key            | Type    | Default              | Description                                                                                                        |
| :------------- | :------ | :------------------- | :----------------------------------------------------------------------------------------------------------------- |
| `model`        | String  | `"gemini-3.8-flash"` | Model identifier to evaluate.                                                                                      |
| `vertex`       | Boolean | `None`               | Explicitly enable (`true`) or disable (`false`) Vertex AI / Agent Platform ADC authentication (`vertexai`).        |
| `project`      | String  | Inferred             | Google Cloud project ID hosting the target Agent Platform endpoints or Agent Registry.                             |
| `location`     | String  | `"global"`           | Google Cloud region or `"global"` endpoint location for Agent Platform requests.                                   |
| `use_symlinks` | Boolean | `false`              | Copy skill directories by default so Antigravity's workspace file indexer discovers resident `SKILL.md` manifests. |

### Claude Code (`claude-code`)

| Key                             | Type                     | Default                  | Description                                                                                                            |
| :------------------------------ | :----------------------- | :----------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| `executable`                    | String                   | `"claude"`               | Path or command name for the Claude Code CLI binary.                                                                   |
| `model`                         | String                   | `"claude-sonnet-5"`      | Model identifier to evaluate.                                                                                          |
| `vertex`                        | Boolean                  | `None`                   | Explicitly enable (`true`) or disable (`false`) Google Cloud Model Garden / Vertex AI mode (`CLAUDE_CODE_USE_VERTEX`). |
| `project`                       | String                   | `None`                   | Google Cloud project ID injected as `ANTHROPIC_VERTEX_PROJECT_ID`.                                                     |
| `location`                      | String                   | `None`                   | Google Cloud region injected as `CLOUD_ML_REGION`.                                                                     |
| `disable_bundled_skills`        | Boolean                  | `true`                   | When true, disables default runtime-bundled skills, built-in plugin mods, and ambient instruction files (`CLAUDE.md`). |
| `skill_overrides`               | Mapping[String, String]  | Built-in skills disabled | Explicit skill activation overrides passed to Claude Code to suppress bundled skills.                                  |
| `enabled_plugins`               | Mapping[String, Boolean] | Built-in mods disabled   | Explicit plugin activation states to suppress default runtime mods.                                                    |
| `skill_listing_budget_fraction` | Float                    | `None`                   | Fraction of total model context window allocated to the resident skill listing table (e.g. `0.05` for 5%).             |
| `skill_listing_max_desc_chars`  | Integer                  | `None`                   | Maximum character length for each individual skill description before description truncation.                          |
| `setting_sources`               | String                   | `"project"`              | Setting sources passed via `--setting-sources` (`"project"`, `"user"`, etc.).                                          |

### Goose (`goose`)

| Key            | Type    | Default              | Description                                                                       |
| :------------- | :------ | :------------------- | :-------------------------------------------------------------------------------- |
| `executable`   | String  | `"goose"`            | Path or command name for the Goose CLI binary.                                    |
| `model`        | String  | `"gemini-3.8-flash"` | Model identifier to evaluate.                                                     |
| `provider`     | String  | Inferred             | Model provider identifier (e.g. `"google"`, `"anthropic"`, `"openai"`).           |
| `no_profile`   | Boolean | `true`               | When true, bypasses user developer profiles and ambient configurations.           |
| `with_builtin` | String  | `"skills"`           | Built-in extensions to enable (restricted to `"skills"` for isolated evaluation). |

### Pi (`pi`)

| Key          | Type   | Default              | Description                                 |
| :----------- | :----- | :------------------- | :------------------------------------------ |
| `executable` | String | `"pi"`               | Path or command name for the Pi CLI binary. |
| `model`      | String | `"gemini-3.8-flash"` | Model identifier to evaluate.               |
| `provider`   | String | `"google"`           | Model provider identifier.                  |

### Keyword (`keyword`)

The deterministic `keyword` baseline driver scores resident skills by word-boundary lexical matching and BM25 without external subprocesses or model APIs:

| Key     | Type   | Default     | Description                                                                   |
| :------ | :----- | :---------- | :---------------------------------------------------------------------------- |
| `model` | String | `"keyword"` | Baseline lexical scorer identifier recorded in evaluation provenance.         |
| `scope` | String | `"user"`    | Skill scope label applied when staging resident skills (`"user"`, `"local"`). |

---

## Configuration Precedence

Settings resolve in the following order (highest precedence wins):

1. **Explicit CLI flags** (e.g. `--min-recall 0.90`)
2. **Project-local `reach.toml`** (in the current working directory, or specified by `--config`)
3. **Bundled default `reach.toml`**

Path fields in `[study]` (`skills`, `queries`, `workdir`, `out`) and entries in `[discovery].precedence` also interpolate environment variables dynamically using `${VAR}` or `$VAR` syntax (e.g. `skills = "${REACH_SKILL_ROOT}"`).

---

## Environment Variables

| Variable                      | Description                                                                                                          |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| `REACH_NO_BROWSER`            | Set to `"1"`, `"true"`, or `"yes"` to bypass interactive browser review for drafted queries.                         |
| `REACH_YES`                   | Set to `"1"`, `"true"`, or `"yes"` to bypass interactive safety confirmation prompts in CI/CD and scripts.           |
| `AGY_ADC_AUTH`                | Set to `"true"` or `"1"` to enable Google Cloud ADC / Vertex AI authentication for `antigravity-cli`.                |
| `GOOGLE_GENAI_USE_ENTERPRISE` | Set to `"true"` or `"1"` to enable Google Cloud Agent Platform ADC authentication for `antigravity-sdk`.             |
| `GOOGLE_GENAI_USE_VERTEXAI`   | Set to `"true"` or `"1"` to enable Vertex AI authentication for `google-genai` and Antigravity runtimes.             |
| `GOOGLE_CLOUD_PROJECT`        | Default Google Cloud project ID for Agent Platform, Vertex AI, and Agent Registry operations.                        |
| `GOOGLE_CLOUD_QUOTA_PROJECT`  | Fallback Google Cloud project ID checked when `GOOGLE_CLOUD_PROJECT` is unset.                                       |
| `GOOGLE_CLOUD_LOCATION`       | Default Google Cloud location (e.g. `"global"`) for Agent Platform and Vertex AI requests.                           |
| `CLAUDE_CODE_USE_VERTEX`      | Set to `"1"` or `"true"` to route `claude-code` through Google Cloud Model Garden on Vertex AI.                      |
| `ANTHROPIC_VERTEX_PROJECT_ID` | Google Cloud project ID used by `claude-code` when Vertex AI mode is enabled (falls back to `GOOGLE_CLOUD_PROJECT`). |
| `CLOUD_ML_REGION`             | Google Cloud region used by `claude-code` when Vertex AI mode is enabled (defaults to `"global"`).                   |
| `GITHUB_STEP_SUMMARY`         | When set (in GitHub Actions), `reach check` automatically appends markdown gate summaries to this file.              |
| `NO_MKDOCS_2_WARNING`         | Set to `"1"` to suppress upstream MkDocs 2.0 console notices during documentation builds.                            |

---

## Provider API Keys & Authentication

Reach automatically routes model API keys and ADC settings to the corresponding environment variables expected by each runtime agent:

| Provider            | Credentials / Injected Environment Variables                                                                                     | Support Tier                 |
| :------------------ | :------------------------------------------------------------------------------------------------------------------------------- | :--------------------------- |
| `google` / `gemini` | `GEMINI_API_KEY`, `GOOGLE_API_KEY`                                                                                               | Tested (Primary reference)   |
| `google-cloud`      | Application Default Credentials (ADC), `AGY_ADC_AUTH`, `GOOGLE_GENAI_USE_ENTERPRISE`, `GOOGLE_CLOUD_PROJECT`                     | Tested (Agent Platform)      |
| `anthropic`         | `ANTHROPIC_API_KEY` (or Vertex AI Model Garden via `CLAUDE_CODE_USE_VERTEX=1`, `ANTHROPIC_VERTEX_PROJECT_ID`, `CLOUD_ML_REGION`) | Supported                    |
| `openai`            | `OPENAI_API_KEY`                                                                                                                 | Supported (CLI pass-through) |

/// note | Provider Support Status
Google Gemini (via Developer API keys) and Google Cloud Agent Platform / Model Garden (via Application Default Credentials) are the primary benchmarked authentication paths for Reach. For `antigravity-cli` and `antigravity-sdk`, set `vertex = true` under `[runtime.options]` or export `AGY_ADC_AUTH=true` / `GOOGLE_GENAI_USE_ENTERPRISE=true`. For Claude Code, configure Google Cloud Model Garden on Vertex AI (`CLAUDE_CODE_USE_VERTEX=1` or `vertex = true`) or set `ANTHROPIC_API_KEY`.
///

If an unrecognized provider name is specified via options, Reach raises an error rather than mapping credentials to an unintended provider. For custom, local, or self-hosted model engines (such as Ollama, vLLM, or Mistral), set the provider's expected environment variables directly in your shell or CI workflow.
