# `reach view`

Read back an evaluation artifact (default: `.reach/eval.json`) or scaling sweep study (`.reach/sweep.json`) and render human-readable scorecards, scaling curves, CSVs, or interactive standalone HTML reports.

---

## Synopsis

```bash
reach view [ARTIFACT] [OPTIONS]
```

---

## Key Scenarios

/// tab | Terminal scorecard
Print summary scores, precision, recall, and top confusion pairs for the latest evaluation:

```bash
reach view
```

Output:

```text
all: 58 skills, 35 probes

 skill                             reached   recall    95% CI   absorbed   precision
 ───────────────────────────────────────────────────────────────────────────────────
 google-agents-cli-adk-code            5/5     100%   57-100%                   100%
 google-agents-cli-deploy              5/5     100%   57-100%                   100%
 google-agents-cli-eval                5/5     100%   57-100%                   100%
 google-agents-cli-observability       5/5     100%   57-100%                   100%
 google-agents-cli-publish             5/5     100%   57-100%                   100%
 google-agents-cli-scaffold            5/5     100%   57-100%                   100%
 google-agents-cli-workflow            5/5     100%   57-100%                   100%

              51 resident skills had no query and took no traffic
consistency 100% [65-100%]   top-1 100.0% ± 4.7pp   abstention 0% [0-10%] (0% false)
macro-F1 100.0%  over 7 observed labels  [5 replicates, 7 repeated queries]
```

///

/// tab | Interactive HTML report
Generate a self-contained, standalone HTML diagnostic workbench report with an interactive confusion matrix, skill-level scores, search filtering, and expandable queries:

```bash
reach view --format html > report.html
open report.html
```

The interactive workbench includes:

- **Confusion matrix cross-filtering**: Click any hit or miss cell in the confusion matrix to instantly filter queries down to the matching expected/invoked skill pair, with an active filter banner and one-click reset.
- **Dynamic table and query search**: Filter skills and query cards in real time by ID, skill name, or query prompt text.
- **Expand/collapse controls**: Expand all query cards at once to inspect full prompt text and difficulty ranks, or collapse them for a high-level overview.
- **Air-gapped and self-contained**: Zero external script or CDN dependencies; safe to open offline or in air-gapped environments.

///

/// tab | Inspect query-level probe outcomes
List individual queries, difficulty ranks, and probe outcomes in the terminal:

```bash
reach view --show-queries
```

Output:

```text
● google-agents-cli-adk-code  5/5 reached (100%)

 query              reached   rank   leak           selected   text
 ───────────────────────────────────────────────────────────────────────────────────────────────
 q-adk-code   5/5 [57-100%]   1/58   names target   ✓ match    Write custom agent code with ADK

● google-agents-cli-deploy  5/5 reached (100%)

 query              reached   rank   leak           selected   text
 ───────────────────────────────────────────────────────────────────────────────────────────────
 q-deploy-1   5/5 [57-100%]   1/58   names target   ✓ match    Deploy agent service to Cloud Run

● google-agents-cli-eval  5/5 reached (100%)

 query              reached   rank   leak           selected   text
 ───────────────────────────────────────────────────────────────────────────────────────────────
 q-eval-1     5/5 [57-100%]   1/58   names target   ✓ match    Run an evaluation on skill suite
```

///

/// tab | Scaling sweep study
Render recorded capacity decay, knee metrics, and scaling curves from a sweep artifact:

```bash
reach view .reach/sweep.json
```

Or export to CSV format:

```bash
reach view .reach/sweep.json --format csv > sweep.csv
```

///

---

## Options

| Option            | Type   | Default            | Description                                                                                                                                                                           |
| :---------------- | :----- | :----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ARTIFACT`        | Path   | `.reach/eval.json` | Path to the evaluation or sweep artifact JSON file. If omitted, defaults to `.reach/eval.json` (or `.reach/sweep.json`).                                                              |
| `--out`, `-o`     | Path   | -                  | Destination path to write the rendered report (defaults to stdout).                                                                                                                   |
| `--open`, `-O`    | Flag   | `false`            | Open the rendered HTML report directly in the default web browser.                                                                                                                    |
| `--show-queries`  | Flag   | `false`            | Display individual scored query records below the summary scorecard.                                                                                                                  |
| `--queries`, `-q` | Path   | -                  | Subset query set file used to slice the recorded evaluation run.                                                                                                                      |
| `--filter-skill`  | String | `()`               | Glob pattern(s) matching target skill names to slice the recorded run.                                                                                                                |
| `--filter-id`     | String | `()`               | Glob pattern(s) matching query IDs to slice the recorded run.                                                                                                                         |
| `--format`        | Choice | `text`             | Output format: `text` (terminal table), `html` (standalone interactive HTML), `json` (raw artifact JSON), `jsonl` (scored query records as JSON lines), or `csv` (sweep metrics CSV). |
| `--verbose`       | Flag   | `false`            | Display full hexadecimal hash digests alongside badges.                                                                                                                               |
