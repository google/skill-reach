# `reach doctor`

Inspect and diagnose local development environment, runtime agent binaries, API credentials, and skill catalogs.

---

## Synopsis

```bash
reach doctor [PATH] [OPTIONS]
```

---

## Diagnostics Checked

1. **Environment**: Verifies Python `>= 3.12` runtime/platform details and checks whether optional `Semantic Scoring (model2vec)` (`skill-reach[semantic]`) is installed.
2. **Runtimes**: Checks for installed agent CLI executables (`claude`, `agy` / `antigravity`, `goose`, `pi`), the `Antigravity SDK` (`google.antigravity`), and the built-in `Keyword Runtime (BM25)`.
3. **Credentials**: Verifies configured API keys (`GEMINI_API_KEY` or `GOOGLE_API_KEY`), Google Cloud Application Default Credentials (`GOOGLE_APPLICATION_CREDENTIALS` or standard `gcloud` ADC paths plus active Vertex flags `AGY_ADC_AUTH`, `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_GENAI_USE_ENTERPRISE`), and Google Cloud Agent Registry project and cache state.
4. **Skills**: Discovers and counts resident skills across workspace paths (respecting `reach.toml` `[discovery].precedence`, `[study].skills`, root `SKILL.md`, and canonical symlink deduplication), plus optional user-global directories (`~/`) when `--global` (`-g`) is enabled.
5. **Configuration**: Validates the TOML syntax, Pydantic schema, and active settings in `reach.toml`.

`reach doctor` exits `0` when all checks pass or warn (`OK` / `WARN`) and exits `1` when any check fails (`FAIL`), making it suitable as a preflight health gate in CI/CD pipelines.

---

## Key Scenarios

/// tab | Run environment diagnostics
Check local environment health and report status:

```bash
reach doctor
```

///

/// tab | Verbose diagnostics with remedies
Include detailed remediation steps for warnings and missing dependencies:

```bash
reach doctor --verbose
```

///

/// tab | Inspect workspace and global skills
Include user-level skill directories (`~/.agents/skills`, `~/.claude/skills`, etc.) alongside workspace skills:

```bash
reach doctor --global
```

///

/// tab | Export structured JSON or CSV diagnostics
Emit machine-readable diagnostic reports for CI/CD preflight checks:

```bash
reach doctor --format json
```

///

---

## Options

| Option                   | Type   | Default           | Description                                                        |
| :----------------------- | :----- | :---------------- | :----------------------------------------------------------------- |
| `[PATH]`, `--path`, `-p` | Path   | Current directory | Target workspace directory to inspect.                             |
| `--verbose`, `-v`        | Flag   | `false`           | Display detailed diagnostics and recommended remediation steps.    |
| `--quiet`, `-q`          | Flag   | `false`           | Mute the terminal table view.                                      |
| `--global`, `-g`         | Flag   | `false`           | Discover and inspect skills from user global configuration (`~/`). |
| `--format`               | Choice | `text`            | Output format: `text`, `json`, `jsonl`, `csv`.                     |
| `--config`, `-c`         | Path   | `reach.toml`      | Path to `reach.toml` configuration file.                           |
