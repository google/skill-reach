# `reach lint`

Validate skill manifests, frontmatter schemas, naming conventions, and runtime listing budget limits.

---

## Synopsis

```bash
reach lint [SKILLS] [OPTIONS]
```

---

## Key Scenarios

/// tab | Auto-discovery by precedence
Automatically inspect the project using the configured precedence order:

```bash
reach lint
```

///

/// tab | Targeting a directory or file
Validate all skills in a specific directory or a standalone skill manifest positionally:

```bash
reach lint ./skills
reach lint ./SKILL.md
```

///

/// tab | Strict mode for CI
Fail with exit code `1` if any warnings or errors are discovered:

```bash
reach lint ./skills --strict
```

///

/// tab | Targeting specific skills
Filter lint checks to a specific skill:

```bash
reach lint ./skills --skill database-migrate
```

///

/// tab | Explain a specific lint rule
Get detailed background and remediation instructions for a rule:

```bash
reach lint --explain listing-overflow
```

///

/// tab | Output formats
Generate GitHub annotations or JSON diagnostics:

```bash
reach lint ./skills --format github
reach lint ./skills --format json
```

///

---

## Options

| Option                 | Type   | Default         | Description                                                                                          |
| :--------------------- | :----- | :-------------- | :--------------------------------------------------------------------------------------------------- |
| `[SKILLS]`, `--skills` | Path   | Auto-discovered | Path to a skill directory, `SKILL.md` file, or catalog tree (discovered from precedence if omitted). |
| `--skill`              | String | -               | Filter lint diagnostics to specific skill name(s) (repeatable).                                      |
| `--strict`             | Flag   | `false`         | Fail with exit code 1 if any warnings are detected.                                                  |
| `--explain`            | String | -               | Display detailed explanation and remedy for a specific lint rule and exit.                           |
| `--ignore`             | String | -               | Disable specific lint rule(s) (repeatable).                                                          |
| `--error`              | String | -               | Treat specific lint rule(s) as an error (repeatable).                                                |
| `--warn`               | String | -               | Treat specific lint rule(s) as a warning (repeatable).                                               |
| `--agent`              | Choice | -               | Agent runtime to query for installed skill locations (`claude-code`, `antigravity-cli`, etc.).       |
| `--global`, `-g`       | Flag   | `false`         | Discover and inspect skills from user global configuration (`~/`).                                   |
| `--format`             | Choice | `text`          | Output format: `text`, `concise`, `github`, `json`, `jsonl`, `csv`.                                  |
| `--config`             | Path   | -               | Path to custom `reach.toml` configuration file.                                                      |

---

## Built-in Lint Rules

| Rule ID                          | Default Severity | Description                                                                                  |
| :------------------------------- | :--------------- | :------------------------------------------------------------------------------------------- |
| `invalid-yaml`                   | Error            | SKILL.md contains missing or unparseable YAML frontmatter.                                   |
| `missing-name`                   | Error            | Frontmatter does not declare a skill `name`.                                                 |
| `missing-description`            | Error            | Frontmatter has no `description` or the description is empty.                                |
| `invalid-name-format`            | Error            | Skill name does not adhere to lowercase kebab-case convention (max 64 chars).                |
| `name-mismatch`                  | Error            | Frontmatter `name` differs from the parent directory name.                                   |
| `duplicate-name`                 | Error            | Multiple skills in the corpus declare the same `name`.                                       |
| `duplicate-capability`           | Warning          | Skill description has high semantic overlap (> 92%) with another resident skill.             |
| `description-too-short`          | Warning          | Description is under 20 characters and lacks actionable routing criteria.                    |
| `listing-overflow`               | Warning          | Description exceeds warning threshold (1,024 characters) and risks prompt listing elision.   |
| `reserved-name-collision`        | Warning          | Skill name collides with a built-in agent tool or command primitive.                         |
| `unresolved-placeholder`         | Warning          | Description contains unresolved template markers (`TODO`, `FIXME`, `<FILL_IN>`).             |
| `unresolved-declared-dependency` | Warning          | Declared dependency skill in `metadata` or `allowed-tools` is missing from resident catalog. |
| `lockfile-drift`                 | Warning          | Local SKILL.md content digest does not match pinned `computedHash` in `skills-lock.json`.    |
| `unbounded-attractor`            | Warning          | Description contains overly broad phrases that aggressively attract out-of-scope queries.    |
| `unknown-skill-reference`        | Warning          | Negative routing handoff (`use <other-skill>`) references a skill missing from the catalog.  |
| `missing-mutual-handoff`         | Warning          | High-similarity or one-way competing neighbor lacks reciprocal `Don't use for...` handoff.   |
| `catalog-budget-overflow`        | Warning          | Resident skill catalog exceeds runtime listing character budget causing description elision. |
