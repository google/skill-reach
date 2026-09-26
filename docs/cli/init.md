# `reach init`

Scaffold a new Reach project configuration and initialize skill directories.

---

## Synopsis

```bash
reach init [PATH] [OPTIONS]
```

---

## Behavior

1. Detects or creates standard skill directory locations (`.agents/skills/`).
2. Detects available agent CLI binaries and environment variables to choose an appropriate default runtime.
3. Generates a tailored `reach.toml` file with sensible defaults.
4. Creates `.reach/` workspace cache directory for local query sets and artifacts.

---

## Key Scenarios

/// tab | Initialize with auto-detection
Scaffold a project detecting local skills and available agent CLI:

```bash
reach init
```

///

/// tab | Specify a default agent runtime
Explicitly set the default agent driver:

```bash
reach init --agent claude-code
reach init --agent antigravity-cli
```

///

/// tab | Custom skills directory
Configure an explicit skill path:

```bash
reach init --skills ./custom-skills
```

///

/// tab | Force overwrite
Overwrite an existing `reach.toml`:

```bash
reach init --force
```

///

---

## Options

| Option                 | Type   | Default           | Description                                               |
| :--------------------- | :----- | :---------------- | :-------------------------------------------------------- |
| `PATH`, `--path`, `-p` | `PATH` | Current directory | Project root directory to initialize.                     |
| `--agent`              | `TEXT` | Auto-detected     | Default agent runtime to configure in `reach.toml`.       |
| `--skills`, `-s`       | `PATH` | `.agents/skills`  | Directory path where skills are stored.                   |
| `--force`, `-f`        | `flag` | `false`           | Overwrite existing `reach.toml` configuration if present. |
| `--quiet`, `-q`        | `flag` | `false`           | Suppress summary output.                                  |
