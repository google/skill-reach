# `reach query`

Synthesize, convert, and inspect labeled evaluation query datasets across JSON, JSONL, and CSV formats.

An evaluation dataset consists of realistic user prompts paired with ground-truth target skill expectations and difficulty metrics.

---

## Synopsis

```bash
# Synthesize queries for all skills (defaults to saving in .reach/queries.json)
reach query

# Synthesize directly in JSONL or CSV format
reach query -o queries.jsonl
reach query -o queries.csv

# Synthesize for a specific skill corpus or target skill
reach query ./my-skills --count 3 --adversarial

# Synthesize queries with interactive browser review before saving
reach query ./my-skills --review

# Convert an existing query dataset between formats offline (0 LLM tokens)
reach query .reach/queries.json -o queries.csv
reach query queries.csv -o .reach/queries.json
reach query queries.json -o queries.jsonl

# Inspect query difficulty, leak detection, and citations
reach query --queries .reach/queries.json --leaks --citations
```

---

## Options

### Synthesis & Targets

| Option                | Type    | Default            | Description                                                                                          |
| :-------------------- | :------ | :----------------- | :--------------------------------------------------------------------------------------------------- |
| `[TARGET]`            | Path    | Auto-discovered    | Skill directory to draft queries for, or existing query file (`.json`, `.jsonl`, `.csv`) to convert. |
| `--skills`            | Path    | Auto-discovered    | Path to skills directory or catalog.                                                                 |
| `--skill`             | String  | All                | Specific skill name(s) to draft queries for (repeatable).                                            |
| `--count`             | Integer | `3`                | Number of queries to draft per target skill.                                                         |
| `--generator-model`   | String  | `gemini-3.8-flash` | Model used to draft synthetic queries.                                                               |
| `--generator-agent`   | String  | Auto-detected      | Agent driver used to draft synthetic queries, overriding the probe agent.                            |
| `--adversarial`       | Flag    | `false`            | Synthesize near-miss negative queries sharing target vocabulary.                                     |
| `--adversarial-count` | Integer | `1`                | Number of adversarial negative queries per target.                                                   |
| `--review`            | Flag    | `false`            | Launch interactive browser review for drafted queries before saving.                                 |

### Output & Formats

| Option           | Type   | Default               | Description                                                                                                                    |
| :--------------- | :----- | :-------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| `--out`, `-o`    | Path   | `.reach/queries.json` | Where to write the query set (format auto-inferred from `.json`, `.jsonl`, or `.csv`).                                         |
| `--format`, `-f` | Choice | `json`                | Explicit output format: `json`, `jsonl`, or `csv`. When `--out` is omitted with `csv` or `jsonl`, output is printed to stdout. |
| `--force`        | Flag   | `false`               | Overwrite destination query set file if it already exists.                                                                     |
| `--dry-run`      | Flag   | `false`               | Preview prompts and token budget without making model calls.                                                                   |

### Inspection & View

| Option        | Type   | Default         | Description                                                                                     |
| :------------ | :----- | :-------------- | :---------------------------------------------------------------------------------------------- |
| `--queries`   | Path   | -               | Query set path to inspect or convert.                                                           |
| `--leaks`     | Flag   | `false`         | Add a column flagging potential target skill name leakage in query text.                        |
| `--citations` | Flag   | `false`         | Add a column quoting each query's grounding passage (requires `.reach/queries-citations.json`). |
| `--agent`     | Choice | Auto-discovered | Agent runtime used for skill corpus discovery.                                                  |

### Agent Registry Options

| Option            | Type   | Default    | Description                                                               |
| :---------------- | :----- | :--------- | :------------------------------------------------------------------------ |
| `--project`, `-p` | String | None       | Google Cloud project ID hosting the Agent Registry.                       |
| `--location`      | String | `"global"` | Agent Registry location endpoint (`global`, `us`, `eu`).                  |
| `--publisher`     | String | None       | Filter registry skills by publisher identifier (e.g. `cloud.google.com`). |
| `--registry`      | Flag   | `false`    | Target Google Cloud Agent Registry instead of local workspace.            |
| `--fresh`         | Flag   | `false`    | Bypass cached metadata and re-fetch latest skill definitions.             |
| `--no-cache`      | Flag   | `false`    | Run without reading or writing local disk cache.                          |

### Field Mapping (Custom CSV / JSONL)

| Option                       | Type   | Default | Description                                                      |
| :--------------------------- | :----- | :------ | :--------------------------------------------------------------- |
| `--text-column`              | String | -       | Column holding query text.                                       |
| `--id-column`                | String | -       | Column holding query ID (numbered automatically when absent).    |
| `--kind-column`              | String | -       | Column holding query kind.                                       |
| `--expected-skill-column`    | String | -       | Column holding expected target skill name.                       |
| `--acceptable-skills-column` | String | -       | Column holding neutral router or helper skills.                  |
| `--notes-column`             | String | -       | Column holding per-query notes.                                  |
| `--separator`                | String | `,`     | Delimiter separating multiple skills in a cell.                  |
| `--id-prefix`                | String | -       | Prefix prepended to query IDs to prevent collisions.             |
| `--catalog`                  | String | `all`   | Target catalog ID to associate with imported queries.            |
| `--notes`                    | String | `""`    | Provenance notes or reviewer comments describing this query set. |

---

## Subcommands

### `reach query draft`

Draft synthetic evaluation queries targeting resident skills without converting existing datasets.

```bash
reach query draft [TARGET] [OPTIONS]

# Draft queries with interactive browser review before saving
reach query draft ./my-skills --review
```

### `reach query view`

Render an evaluation query set as a terminal table with difficulty ranks, leakage flags, and citations.

```bash
reach query view <QUERIES> [OPTIONS]
```

---

## Format Conversion & Exchange

Reach supports offline format conversion directly through `reach query` without calling an LLM:

```bash
# Export JSON to CSV for spreadsheet review in Google Sheets or Excel
reach query .reach/queries.json -o queries.csv

# Print CSV rows directly to stdout
reach query .reach/queries.json -f csv

# Export JSON to JSONL for streaming evaluation pipelines
reach query .reach/queries.json -o queries.jsonl

# Import edited CSV back into canonical Reach JSON
reach query queries.csv -o .reach/queries.json --catalog all

# Map external CSV datasets with non-standard column headers
reach query external.csv -o .reach/queries.json --text-column prompt --expected-skill-column tool
```

`reach eval` also natively accepts `.yaml`/`.yml`, `.jsonl`, and `.csv` files via `--queries`, so conversion is purely optional:

```bash
reach eval --queries queries.yaml
reach eval --queries queries.jsonl
reach eval --queries queries.csv
```

---

## Query Schema (`Query`)

Each entry in a `.reach/queries.json` (or `.yaml` / `.jsonl` / `.csv` dataset) conforms to the [`Query`](../api/models.md) model (when hand-authoring JSON or YAML files, top-level `catalog_id` defaults to `"all"` and missing query `id`s are numbered automatically):

| Field               | Type                | Default      | Description                                                                                                                                               |
| :------------------ | :------------------ | :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                | `str`               | Auto (`q-N`) | Unique identifier for the evaluation query (numbered automatically as `q-001`, `q-002`, ... when omitted).                                                |
| `text`              | `str`               | —            | Realistic user prompt text presented to the agent runtime. Accepts `"query"` as a validation alias and always serializes to `"text"` on save.             |
| `kind`              | `QueryKind \| None` | `None`       | Structural category (`implicit`, `contextual`, `neighbor_negative`, or `out_of_scope`).                                                                   |
| `expected_skill`    | `str \| None`       | `None`       | Ground-truth target skill expected to be invoked, or `None` for out-of-scope queries.                                                                     |
| `acceptable_skills` | `tuple[str, ...]`   | `()`         | Optional neutral helper or router skills (e.g. `finding-google-skills`) that consume turns at runtime but are neither rewarded as TP nor penalized as FP. |
| `notes`             | `str`               | `""`         | Author notes, rationale, or difficulty context.                                                                                                           |
