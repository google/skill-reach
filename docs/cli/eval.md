# `reach eval`

Measure whether a catalog's skills are reachable when resident alongside their rivals. Uses `.reach/queries.{json,yaml,yml}` when present, or synthesizes queries automatically when `--auto` or a single-skill `[TARGET]` is provided.

/// warning | Agent Execution Safety
`reach eval` executes live agent subprocesses that can run tools and shell commands on the host system. When testing third-party or untrusted skills, execute inside an isolated container sandbox (e.g. Docker or [Google Cloud Run sandboxes](../guides/sandboxing.md)) or use `--agent keyword` (an in-memory lexical matching engine that matches query terms against skill manifests via compiled regexes and BM25 scoring without running subprocesses or external APIs). Automated non-interactive environments must explicitly pass `--yes` / `-y` or set `REACH_YES=1` to bypass the safety confirmation gate.
///

---

## Synopsis

```bash
reach eval [TARGET] [OPTIONS]
```

---

## Key Scenarios

/// tab | Run a quick evaluation on a target skill
Quick single-skill evaluation defaults to 3 attempts per query (override with `--attempts`):

```bash
reach eval cloud-deploy --agent claude-code
```

///

/// tab | Auto-draft and evaluate full catalog
Synthesize queries for all resident skills and immediately probe the catalog in one step:

```bash
reach eval ./skills --auto
```

Tune the number of drafted queries per skill with `--count` (default: 3):

```bash
reach eval ./skills --auto --count 5
```

///

/// tab | Evaluate skills from Google Cloud Agent Registry
Evaluate reachability directly on skills pulled from Google Cloud Agent Registry:

```bash
reach eval --project your-project-id --location global --auto
```

///

/// tab | Dry-run without model calls
Simulate catalog assembly, prompt generation, and probe loops without executing live model API calls:

```bash
reach eval cloud-deploy --dry-run
```

///

/// tab | Direct skill directory path
Specify a direct path to a skill directory (parent corpus is auto-detected):

```bash
reach eval ./skills/cloud-deploy --agent claude-code
```

///

/// tab | Parallel execution
Speed up evaluation with concurrent probe execution:

```bash
reach eval cloud-deploy --workers 4
```

///

---

## Options

### Study & Corpus

| Option           | Type          | Description                                                                                                                                                   |
| :--------------- | :------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `[TARGET]`       | String / Path | One skill to evaluate by name or directory (requests a quick run), or a skill corpus directory when paired with `--auto`.                                     |
| `--query`        | String        | Probe this exact question rather than drafting one (repeatable).                                                                                              |
| `--expected`     | String        | The skill every `--query` should reach; defaults to the skill named.                                                                                          |
| `--skills`, `-s` | Path          | Root directory containing skills or a catalog tree.                                                                                                           |
| `--queries`      | Path          | Labeled evaluation queries file (`.json`, `.yaml`/`.yml`, `.jsonl`, or `.csv`). Defaults to `.reach/queries.*` if present, or requires `--auto` / `[TARGET]`. |
| `--workdir`      | Path          | Temporary workspace to install the competitive catalog into.                                                                                                  |
| `--catalog`      | String        | Catalog identifier; defaults to the query set's catalog ID.                                                                                                   |
| `--partial`      | Flag          | Allow a query set that targets only a subset of the catalog's skills.                                                                                         |
| `--rescope`      | Flag          | Probe the query set against a catalog other than the one it was labeled in.                                                                                   |
| `--tag`          | String        | Short semantic label for this run (e.g. `v1-baseline`), displayed in headers.                                                                                 |
| `--run-dir`      | Path          | Directory containing default queries, workdir, and output paths.                                                                                              |

### Agent Registry Options

| Option            | Type   | Default    | Description                                                               |
| :---------------- | :----- | :--------- | :------------------------------------------------------------------------ |
| `--project`, `-p` | String | None       | Google Cloud project ID hosting the Agent Registry.                       |
| `--location`      | String | `"global"` | Agent Registry location endpoint (`global`, `us`, `eu`).                  |
| `--publisher`     | String | None       | Filter registry skills by publisher identifier (e.g. `cloud.google.com`). |
| `--registry`      | Flag   | `false`    | Target Google Cloud Agent Registry instead of local workspace.            |
| `--fresh`         | Flag   | `false`    | Bypass cached metadata and re-fetch latest skill definitions.             |
| `--no-cache`      | Flag   | `false`    | Run without reading or writing local disk cache.                          |

### Catalog Assembly

| Option           | Type    | Default | Description                                                                         |
| :--------------- | :------ | :------ | :---------------------------------------------------------------------------------- |
| `--mode`         | Choice  | `all`   | Assembly mode: `all` (whole catalog), `neighborhood` (closest rivals), `singleton`. |
| `--catalog-size` | Integer | -       | Number of skills per neighborhood catalog.                                          |
| `--rivals`       | Integer | -       | Number of top-ranked rival skills per neighborhood catalog.                         |
| `--seed`         | Integer | -       | Random seed for catalog filler selection.                                           |

### Runtime & Execution

| Option                             | Type    | Default                  | Description                                                                                    |
| :--------------------------------- | :------ | :----------------------- | :--------------------------------------------------------------------------------------------- |
| `--agent`                          | Choice  | `from reach.toml`        | Target runtime: `antigravity-cli`, `antigravity-sdk`, `claude-code`, `goose`, `keyword`, `pi`. |
| `--model`, `-m`                    | String  | Default model            | Target model identifier.                                                                       |
| `--effort`, `-e`                   | String  | Default effort           | Reasoning effort level (e.g. `low`, `medium`, `high`).                                         |
| `--timeout`                        | Integer | -                        | Seconds allowed per probe attempt.                                                             |
| `--max-turns`, `-T`                | Integer | `3`                      | Maximum conversation turns to execute and evaluate.                                            |
| `--early-exit` / `--no-early-exit` | Flag    | `true`                   | Terminate multi-turn probe immediately when target skill is invoked.                           |
| `--opt`, `-O`                      | String  | -                        | Agent runtime option as `key=value` (repeatable).                                              |
| `--attempts`, `-n`                 | Integer | `5` (`3` for quick eval) | Number of probe attempts per query for consistency estimation.                                 |
| `--retries`                        | Integer | `2`                      | Retry attempts for failed model invocations.                                                   |
| `--backoff`                        | Float   | `5.0`                    | Initial backoff time in seconds before the first retry.                                        |
| `--pause`                          | Float   | `0.0`                    | Seconds to pause between probes to respect rate limits.                                        |
| `--workers`, `-j`                  | Integer | `1`                      | Number of concurrent probes to run.                                                            |
| `--auto`                           | Flag    | `false`                  | Automatically draft queries for all skills and probe the catalog in one step.                  |
| `--reasoning`                      | Flag    | `false`                  | Display model reasoning / thought traces directly beneath misrouted collision rows.            |
| `--no-resume`                      | Flag    | `false`                  | Re-probe everything, ignoring results already in output.                                       |
| `--append-across-arms`             | Flag    | `false`                  | Add probes to an `--out` recorded under a different configuration.                             |
| `--allow-truncation`               | Flag    | `false`                  | Probe a catalog too wide for the runtime's skill listing without error.                        |
| `--yes`, `-y`                      | Flag    | `false`                  | Bypass interactive safety confirmation prompts.                                                |

### Query Generation

| Option                | Type    | Default            | Description                                                                                  |
| :-------------------- | :------ | :----------------- | :------------------------------------------------------------------------------------------- |
| `--skill`             | String  | All                | Filter evaluation queries or auto-drafting to specific skill name(s) (repeatable).           |
| `--count`             | Integer | `3`                | Number of queries to draft per target skill.                                                 |
| `--generator-model`   | String  | `gemini-3.8-flash` | Model used to draft synthetic queries.                                                       |
| `--generator-agent`   | Choice  | Auto-detected      | Agent driver used to draft synthetic queries, overriding the probe agent.                    |
| `--generator-arm`     | Choice  | `None`             | Which generation prompt to use (`content` or `framing`).                                     |
| `--top-rivals`        | Integer | `None` (All)       | Maximum number of top-ranked rival skills to include in generation prompts (`3` via config). |
| `--draft-concurrency` | Integer | `1`                | Targets to draft concurrently.                                                               |
| `--adversarial`       | Flag    | `false`            | Synthesize near-miss adversarial negative queries sharing target vocabulary.                 |
| `--adversarial-count` | Integer | `1`                | Number of adversarial negative queries per target.                                           |

### Recording & Artifacts

| Option           | Type   | Default            | Description                                                                          |
| :--------------- | :----- | :----------------- | :----------------------------------------------------------------------------------- |
| `--out`, `-o`    | Path   | `.reach/eval.json` | Destination for the evaluation artifact.                                             |
| `--records`      | Path   | -                  | JSONL file to stream raw probe results to.                                           |
| `--format`       | Choice | `text`             | Output format: `text`, `json`, `jsonl`, `csv`.                                       |
| `--save`         | Path   | -                  | Save a quick run's query set, citations, and artifact to DIR before scratch cleanup. |
| `--dry-run`      | Flag   | `false`            | Simulate drafting and probing without executing model calls or saving results.       |
| `--global`, `-g` | Flag   | `false`            | Discover and inspect skills from user global configuration (`~/`).                   |
| `--quiet`, `-q`  | Flag   | `false`            | Suppress terminal progress and summary view.                                         |
| `--verbose`      | Flag   | `false`            | Display full hexadecimal hash digests alongside badges.                              |
| `--config`       | Path   | -                  | Path to custom `reach.toml` configuration file.                                      |
