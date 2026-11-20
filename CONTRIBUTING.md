# How to Contribute

We welcome contributions and patches to `skill-reach`! Please follow these guidelines before submitting code.

## Contributor License Agreement

Contributions to this project must be accompanied by a [Contributor License Agreement](https://cla.developers.google.com/) (CLA). You (or your employer) retain copyright to your contributions; the agreement gives permission to use and redistribute your work within the project.

- **Individuals**: If you are writing original code you own, sign the [Individual CLA](https://developers.google.com/open-source/cla/individual).
- **Corporations**: If you are contributing on behalf of your employer, sign the [Corporate CLA](https://developers.google.com/open-source/cla/corporate).
- **Googlers**: If you are a Google employee, please contribute using your `@google.com` account.

You only need to submit a CLA once across Google open source projects. Check your active agreements or sign a new one at <https://cla.developers.google.com/>.

## Community Guidelines

This project adheres to [Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Before Opening a Pull Request

- **Open an issue first**: For anything beyond a small bug fix or typo, open an issue first to discuss design. This is especially important for new agent runtimes or changes to statistical modules (`reach.uncertainty`, `reach.metrics`, `reach.diff`).
- **AI-Assisted Contributions**: If a change was substantially drafted or reviewed by an AI coding agent, please state so in the pull request description.

## Development Workflow & Local Verification

1. Fork the repository and create a dedicated feature branch for your changes.
1. Install Python dependencies with `uv sync --all-groups` and Node tools with `npm install`.
1. Follow Test-Driven Development (TDD), adding tests in `tests/` for all bug fixes and features.
1. Run the full local pre-flight suite before opening your pull request:

```sh
# 1. Run the test suite
uv run pytest -q

# 2. Python lint and format checks
uv run ruff check .
uv run ruff format --check .

# 3. Static type checks
uvx ty check

# 4. Web assets and markdown formatting checks
npx @biomejs/biome ci
npx prettier --check "**/*.md"

# 5. Strict documentation build
uv run mkdocs build --strict
```

## Code Reviews & Commit Guidelines

- All submissions, including those from project members, require review through a GitHub pull request. Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/) for pull request workflows.
- Keep pull requests scoped to a single logical change with clear, well-formed commit messages (preferably following [Conventional Commits](https://www.conventionalcommits.org/)).
