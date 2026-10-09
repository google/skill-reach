# `reach sweep`

Measure reachability decay and capacity limits across catalog sizes.

/// warning | Scaling Safety
Scaling sweeps execute repeated live agent probe iterations across varying catalog sizes. Ensure all resident skills in the scaling neighborhood are trusted, or execute the sweep inside an isolated sandbox container (e.g. Docker or [Google Cloud Run sandboxes](../guides/sandboxing.md)). Pass `--yes` / `-y` or set `REACH_YES=1` to bypass interactive confirmation.
///

---

## Synopsis

```bash
reach sweep [SKILLS] [OPTIONS]
```

---

## Key Scenarios

/// tab | Run whole-corpus capacity sweep (default)
Determine optimal skill installation capacity across the entire library:

```bash
reach sweep ./skills --agent antigravity-cli --model gemini-3.8-flash
```

///

/// tab | Run single-skill rival decay sweep
Evaluate reachability and skill collisions for a specific target skill:

```bash
reach sweep ./skills --target cloud-deploy --agent claude-code
```

///

/// tab | Bootstrap replicates and seed
Configure stratified query-cluster bootstrap iterations, random seed, and Owen-scrambled catalog replicates:

```bash
reach sweep ./skills --bootstrap-iterations 500 --catalog-replicates 3 --seed 42
```

///

/// tab | Export sweep metrics to JSON or CSV
Export full capacity and decomposition metrics for plotting and regression tracking:

```bash
reach sweep ./skills --format json > sweep.json
```

///

---

/// note | Anchor Cohort Identification & Simpson's Paradox
By default, `reach sweep` evaluates a fixed anchor cohort across all library sizes. This holds target-skill difficulty constant to cleanly isolate distractor interference from target-set composition shift. Using `--anchor all` subjects the decay curve to composition bias as peripheral skills enter at larger catalog scales.
///

/// tip | Knee Uncertainty, Right-Censoring & Replicate Collision Diagnostics
`ScalingStudy` artifacts (`sweep.json`) record the discrete bootstrap knee distribution (`knee_scale_pmf`), total significant degradation probability (`drop_probability`), immediate cliff probability (`cliff_probability` at $K_0$), right-censoring indicator (`knee_upper_censored`, rendered as `[low, >K_max]` in text and `>K_max` in CSV), baseline intra-skill correlation (`skill_icc`), and per-query replicate collision sensitivity (`replicate_collisions` when `--catalog-replicates > 1`).
///

## Options

| Option                                         | Type          | Default                       | Description                                                                                                                                                        |
| :--------------------------------------------- | :------------ | :---------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SKILLS`, `--skills`, `-s`                     | Path / String | `.`                           | Path to skill directory or corpus to sweep, or a single skill name/directory to evaluate (discovered if omitted).                                                  |
| `--target`                                     | String        | None                          | Target skill to evaluate across scaling steps (omitted for whole-corpus capacity evaluation).                                                                      |
| `--queries`                                    | Path          | `.reach/queries.json`         | Path to labeled evaluation queries file (defaults to `.reach/queries.json`).                                                                                       |
| `--scales`                                     | String        | Adaptive (`10,25,50,100...N`) | Comma-separated list of catalog sizes to evaluate.                                                                                                                 |
| `--anchor`                                     | String        | Scale medoids                 | Anchor skills cohort evaluated across all scales. Accepts integer count (e.g. 10), comma-separated skill names, or `all` for full-corpus expansion.                |
| `--rivals-share`                               | Float         | `0.5`                         | Proportion of distractor skills selected as nearest rivals.                                                                                                        |
| `--workers`, `-j`                              | Integer       | `1`                           | Number of concurrent probe execution workers.                                                                                                                      |
| `--attempts`, `-n`                             | Integer       | `1` (`from reach.toml`)       | Number of probe execution attempts per query at each scale step.                                                                                                   |
| `--retries`                                    | Integer       | `2`                           | Retry attempts for failed model invocations.                                                                                                                       |
| `--backoff`                                    | Float         | `5.0`                         | Initial backoff time in seconds before the first retry.                                                                                                            |
| `--pause`                                      | Float         | `0.0`                         | Seconds to pause between probes to respect rate limits.                                                                                                            |
| `--bootstrap-iterations`                       | Integer       | `200`                         | Number of bootstrap replicates for curve confidence intervals (min: 10).                                                                                           |
| `--seed`                                       | Integer       | `42`                          | Random seed for reproducible bootstrap resamples and curve perturbation.                                                                                           |
| `--catalog-replicates`                         | Integer       | `1`                           | Number of independent distractor catalog permutations to probe per scale step (1–20) for replicate collision detection.                                            |
| `--noise-floor`                                | Float         | Dynamic (paired McNemar)      | Minimum pass rate drop to trigger knee detection (defaults to paired McNemar variance scaled by cluster design effect).                                            |
| `--agent`                                      | Choice        | `from reach.toml`             | Agent runtime to execute scaling probes: `antigravity-cli`, `antigravity-sdk`, `claude-code`, `goose`, `keyword`, `pi`.                                            |
| `--model`, `-m`                                | String        | Default model                 | Target model identifier.                                                                                                                                           |
| `--effort`, `-e`                               | String        | Default effort                | Reasoning effort level (e.g. `low`, `medium`, `high`).                                                                                                             |
| `--timeout`                                    | Integer       | `200`                         | Seconds allowed per probe attempt.                                                                                                                                 |
| `--max-turns`, `-T`                            | Integer       | `3`                           | Maximum conversation turns to execute and evaluate.                                                                                                                |
| `--early-exit` / `--no-early-exit`             | Boolean       | `true`                        | Terminate multi-turn probe immediately when target skill is invoked.                                                                                               |
| `--opt`, `-O`                                  | String        | -                             | Agent runtime option as `key=value` (repeatable).                                                                                                                  |
| `--global`, `-g`                               | Flag          | `false`                       | Discover and inspect skills from user's global configuration (`~/`).                                                                                               |
| `--project`, `-p`                              | String        | None                          | Google Cloud project ID hosting the Agent Registry.                                                                                                                |
| `--location`                                   | String        | `global`                      | Agent Registry location (default: `global`).                                                                                                                       |
| `--publisher`                                  | String        | None                          | Filter skills by publisher identifier (e.g. `cloud.google.com`).                                                                                                   |
| `--registry`                                   | Flag          | `false`                       | Target the Google Cloud Agent Registry instead of local workspace.                                                                                                 |
| `--fresh`                                      | Flag          | `false`                       | Bypass cached metadata and fetch latest revision pointers from registry.                                                                                           |
| `--no-cache`                                   | Flag          | `false`                       | Run without reading or persisting local disk cache.                                                                                                                |
| `--format`                                     | Choice        | `text`                        | Output format: `text`, `json`, `csv`.                                                                                                                              |
| `--out`, `-o`                                  | Path          | `.reach/sweep.json`           | File path to write results (format inferred from file extension or `--format`).                                                                                    |
| `--workdir`                                    | Path          | Temp                          | Working directory for probe execution.                                                                                                                             |
| `--allow-truncation` / `--no-allow-truncation` | Boolean       | `true`                        | Probe scaling steps even if catalogs exceed runtime listing budget (default: `true`).                                                                              |
| `--auto-queries` / `--no-auto-queries`         | Boolean       | `true`                        | Automatically synthesize queries on cold start or for explicitly requested unqueried skills; existing non-empty query sets in corpus-wide sweeps are reused as-is. |
| `--yes`, `-y`                                  | Flag          | `false`                       | Bypass interactive safety confirmation prompts.                                                                                                                    |
| `--config`, `-c`                               | Path          | -                             | Path to reach.toml configuration file.                                                                                                                             |
