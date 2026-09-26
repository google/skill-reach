# `reach doctor`

Inspect and diagnose local development environment, runtime agent binaries, API credentials, and skill catalogs.

---

## Synopsis

```bash
reach doctor [PATH] [OPTIONS]
```

---

## Diagnostics Checked

1. **Python Environment**: Verifies Python `>= 3.12` and runtime platform details.
2. **Agent Runtime Drivers**: Checks for installed agent CLI executables (`claude`, `agy`, `goose`, `pi`) and SDK packages (`google.antigravity`, `keyword`).
3. **Credentials & Keys**: Checks for configured API key (`GEMINI_API_KEY`), Google Cloud Application Default Credentials (ADC), and Agent Registry target configuration.
4. **Skill Directories & Cache**: Scans for resident skills across standard workspace paths (`.agents/skills`, `.claude/skills`, etc.) and cached Agent Registry bundles.
5. **Project Configuration**: Validates the syntax, schema, and active settings in `reach.toml`.

---

## Key Scenarios

/// tab | Run environment diagnostics
Check local environment health and report status:

```bash
reach doctor
```

///

/// tab | Verbose diagnostics with remedies
Include detailed remediation steps for missing keys and CLIs:

```bash
reach doctor --verbose
```

///

---

## Options

| Option           | Type   | Default           | Description                                                     |
| :--------------- | :----- | :---------------- | :-------------------------------------------------------------- |
| `PATH`, `--path` | `PATH` | Current directory | Target workspace directory to inspect.                          |
| `--verbose`      | `flag` | `false`           | Display detailed diagnostic info and recommended action panels. |
