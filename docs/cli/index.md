# CLI Reference

The `reach` command-line interface provides tools for the complete skill measurement and optimization lifecycle.

---

## Command Overview

| Command                             | Purpose                                                                                  |
| :---------------------------------- | :--------------------------------------------------------------------------------------- |
| [`reach check`](check.md)           | Execute two-stage CI/CD quality gate combining linting and empirical assertions.         |
| [`reach clean`](clean.md)           | Clean cached Agent Registry payloads, sandboxes, and evaluation artifacts.               |
| [`reach cluster`](cluster.md)       | Partition skill catalogs into cohesive subagent scopes to prevent routing decay.         |
| [`reach completion`](completion.md) | Generate or install shell tab completion scripts for bash, zsh, and fish.                |
| [`reach diff`](diff.md)             | Change one factor (description, rival, scope), hold queries fixed, and report the delta. |
| [`reach doctor`](doctor.md)         | Inspect and diagnose local environment, agent binaries, credentials, and catalogs.       |
| [`reach eval`](eval.md)             | Measure whether a catalog's skills are reachable, writing a query set if none exists.    |
| [`reach init`](init.md)             | Scaffold reach.toml configuration and initialize skill directories.                      |
| [`reach lint`](lint.md)             | Validate skill manifests, frontmatter schemas, naming, and runtime limits.               |
| [`reach optimize`](optimize.md)     | Optimize a skill's description using candidate synthesis and empirical probes.           |
| [`reach overlap`](overlap.md)       | Find which of your installed skills compete to answer the same requests.                 |
| [`reach query`](query.md)           | Draft, export, import, and view query datasets.                                          |
| [`reach sweep`](sweep.md)           | Execute multi-scale catalog evaluation sweeps to measure reachability decay.             |
| [`reach view`](view.md)             | Read back a recorded run and render what it measured (terminal or HTML).                 |

---

## Global Options

The following flags apply across `reach` commands:

- `--help`, `-h`: Display command help and exit.
- `--version`: Display application version and exit.
- `--global`, `-g`: Discover and inspect skills from user global configuration (`~/.agents/skills`, `~/.claude/skills`, `~/.cursor/skills`, etc.). Can be used as a top-level flag (`reach --global lint`) or subcommand flag (`reach lint --global`).
- `--quiet`, `-q`: Suppress interactive progress bars and spinners.
- `--verbose`: Display full cryptographic hashes and detailed logs.

---

## Supported Agent Runtimes

When running commands that interact with an agent runtime (`check`, `eval`, `optimize`, `sweep`), select the target runtime via `--agent <runtime>`:

- `claude-code`: Drives Anthropic's Claude Code CLI.
- `antigravity-cli`: Drives the Google Antigravity Agent API / CLI.
- `antigravity-sdk`: Uses the Google Antigravity Python SDK directly.
- `goose`: Drives the Goose agent CLI ([`aaif-goose/goose`](https://github.com/aaif-goose/goose)).
- `pi`: Drives the headless Pi coding agent CLI (`earendil-works/pi`).
- `keyword`: Fast rule-based lexical matching runtime for instant baseline scores.
