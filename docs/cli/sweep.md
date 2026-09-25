# `reach sweep`

Measure reachability decay and capacity limits across catalog sizes.

> [!WARNING]
> **Scaling Safety**
> Scaling sweeps execute repeated live agent probe iterations across varying catalog sizes. Ensure all resident skills in the scaling neighborhood are trusted, or execute the sweep inside an isolated sandbox container (e.g. Docker or [Google Cloud Run sandboxes](../guides/sandboxing.md)). Pass `--yes` / `-y` or set `REACH_YES=1` to bypass interactive confirmation.

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
Evaluate reachability and distractor shadowing for a specific target skill:

```bash
reach sweep ./skills --target cloud-deploy --agent claude-code
```

///

/// tab | Bootstrap replicates and seed
Configure cluster-bootstrap iterations and random seed for curve uncertainty:

```bash
reach sweep ./skills --bootstrap-iterations 500 --seed 42
```

///

/// tab | Export sweep metrics to JSON or CSV
Export full capacity and decomposition metrics for plotting and regression tracking:

```bash
reach sweep ./skills --format json > sweep.json
```

///

---

> [!NOTE]
> **Anchor Cohort Identification & Simpson's Paradox**
> By default, `reach sweep` evaluates a fixed anchor cohort across all library sizes. This holds target-skill difficulty constant to cleanly isolate distractor interference from target-set composition shift. Using `--anchor all` subjects the decay curve to composition bias as peripheral skills enter at larger catalog scales.

## Options

| Option                                         | Type    | Default                       | Description                                                                                                                                         |
| :--------------------------------------------- | :------ | :---------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SKILLS`                                       | Path    | `.`                           | Path to skill directory or corpus to sweep (discovered if omitted).                                                                                 |
| `--target`                                     | String  | None                          | Target skill to evaluate across scaling steps (omitted for whole-corpus capacity evaluation).                                                       |
| `--queries`                                    | Path    | `.reach/queries.json`         | Path to labeled evaluation queries file (defaults to `.reach/queries.json`).                                                                        |
| `--scales`                                     | String  | Adaptive (`10,25,50,100...N`) | Comma-separated list of catalog sizes to evaluate.                                                                                                  |
| `--anchor`                                     | String  | Scale medoids                 | Anchor skills cohort evaluated across all scales. Accepts integer count (e.g. 10), comma-separated skill names, or `all` for full-corpus expansion. |
| `--rivals-share`                               | Float   | `0.5`                         | Proportion of distractor skills selected as nearest rivals.                                                                                         |
| `--workers`, `-j`                              | Integer | `1`                           | Number of concurrent probe execution workers.                                                                                                       |
| `--attempts`, `-a`                             | Integer | `5`                           | Number of probe execution attempts per query at each scale step.                                                                                    |
| `--bootstrap-iterations`                       | Integer | `200`                         | Number of bootstrap replicates for curve confidence intervals (min: 10).                                                                            |
| `--seed`                                       | Integer | `42`                          | Random seed for reproducible bootstrap resamples and curve perturbation.                                                                            |
| `--noise-floor`                                | Float   | `0.05`                        | Minimum pass rate drop to trigger knee detection.                                                                                                   |
| `--agent`                                      | Choice  | `keyword`                     | Agent runtime to execute scaling probes.                                                                                                            |
| `--model`, `-m`                                | String  | Default model                 | Target model identifier.                                                                                                                            |
| `--global`, `-g`                               | Flag    | `false`                       | Discover and inspect skills from user's global configuration (`~/`).                                                                                |
| `--project`, `-p`                              | String  | None                          | Google Cloud project ID hosting the Agent Registry.                                                                                                 |
| `--location`                                   | String  | `global`                      | Agent Registry location (default: `global`).                                                                                                        |
| `--publisher`                                  | String  | None                          | Filter skills by publisher identifier (e.g. `cloud.google.com`).                                                                                    |
| `--registry`                                   | Flag    | `false`                       | Target the Google Cloud Agent Registry instead of local workspace.                                                                                  |
| `--fresh`                                      | Flag    | `false`                       | Bypass cached metadata and fetch latest revision pointers from registry.                                                                            |
| `--no-cache`                                   | Flag    | `false`                       | Run without reading or persisting local disk cache.                                                                                                 |
| `--format`                                     | Choice  | `text`                        | Output format: `text`, `json`, `csv`.                                                                                                               |
| `--out`, `-o`                                  | Path    | `.reach/sweep.json`           | File path to write results (format inferred from file extension or `--format`).                                                                     |
| `--workdir`                                    | Path    | Temp                          | Working directory for probe execution.                                                                                                              |
| `--allow-truncation` / `--no-allow-truncation` | Boolean | `true`                        | Probe scaling steps even if catalogs exceed runtime listing budget (default: `true`).                                                               |
| `--auto-queries` / `--no-auto-queries`         | Boolean | `true`                        | Automatically synthesize missing queries and refresh updated queries for anchor skills (default: `true`).                                           |
| `--yes`, `-y`                                  | Flag    | `false`                       | Bypass interactive safety confirmation prompts.                                                                                                     |
| `--config`, `-c`                               | Path    | -                             | Path to reach.toml configuration file.                                                                                                              |
