# Python API Reference

`skill-reach` provides a programmatic Python API for embedding reachability measurement, manifest linting, and automated description optimization into custom tools or workflows.

---

## Module Index

| Module                                | Description                                                                                                                |
| :------------------------------------ | :------------------------------------------------------------------------------------------------------------------------- |
| [`reach.artifact`](artifact.md)       | Evaluation artifact schema, scoring metrics, and confusion matrix data structures.                                         |
| [`reach.catalog`](catalog.md)         | Catalog discovery, filesystem scanning, and YAML frontmatter parsing.                                                      |
| [`reach.check`](check.md)             | Two-stage CI/CD quality gate orchestration, git diff scoping, and threshold enforcement.                                   |
| [`reach.cluster`](cluster.md)         | Modularity-based skill clustering for subagent catalog scoping.                                                            |
| [`reach.config`](config.md)           | Configuration loading and Pydantic settings models.                                                                        |
| [`reach.diff`](diff.md)               | A/B evaluation comparison, noise floor estimation, and effect size reporting.                                              |
| [`reach.lint`](lint.md)               | Static analysis engine for validating skill manifests and listing budgets.                                                 |
| [`reach.metrics`](metrics.md)         | Accuracy, precision, recall, F1, and multi-step trajectory evaluation metrics.                                             |
| [`reach.models`](models.md)           | Domain representations: `Skill`, `Catalog`, `Query`, and `ProbeResult`.                                                    |
| [`reach.optimize`](optimize.md)       | Synthesis and empirical optimization routines for skill descriptions.                                                      |
| [`reach.overlap`](overlap.md)         | BM25 vocabulary competition ranking, nearest-rival extraction, and pairwise collision detection.                           |
| [`reach.queries`](queries.md)         | Labeled evaluation query set loading, serialization, provenance recording, and digest generation.                          |
| [`reach.registry`](registry.md)       | Google Cloud Agent Registry REST client, ADC token resolution, and cache manager.                                          |
| [`reach.retrieval`](retrieval.md)     | BM25 lexical scoring, dense embeddings, and reciprocal rank fusion (RRF).                                                  |
| [`reach.run`](run.md)                 | Execution engine for conducting evaluation runs and probe batches.                                                         |
| [`reach.runtime`](runtime.md)         | Agent execution runtime interfaces, CLI subprocess template drivers, Antigravity domain bridges, and trajectory telemetry. |
| [`reach.sweep`](sweep.md)             | Multi-scale catalog scaling sweeps, capacity knee detection, and loss decomposition.                                       |
| [`reach.uncertainty`](uncertainty.md) | Wilson score confidence intervals, sample size sizing, and power analysis.                                                 |

---

## Quick Example

Programmatically load a catalog, run static linting, and inspect issues:

```python
from pathlib import Path
from reach.catalog import load_skills
from reach.lint import lint_tree

skills = load_skills(Path("./skills"))
report = lint_tree(Path("./skills"))

for issue in report.issues:
    print(f"[{issue.rule}] {issue.skill}: {issue.message}")
```
