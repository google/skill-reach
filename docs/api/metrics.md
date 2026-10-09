# `reach.metrics`

Compute standard evaluation, classification, routing, and multi-step trajectory metrics for skill selection probes.

---

## Multi-Step Trajectory Routing

Modern agent runtimes execute multi-turn conversational trajectories where a task may require entering via one skill, performing a precursor handoff, and terminating at the target capability:

```mermaid
flowchart LR
    Query["User Query"] --> E["Step 1: Entrypoint Skill<br/>(Entrypoint Accuracy)"]
    E -->|Handoff| P["Step 2: Precursor Skill"]
    P -->|Handoff| T["Step 3: Target Capability<br/>(Trajectory Reachability)"]
```

---

## Trajectory Metrics

Reach evaluates multi-step trajectories against the ground-truth capability target $T$ over the scored trajectory ($\vec{s}$ after stripping neutral `acceptable_skills`):

| Metric                      |       Symbol        | Definition & Meaning                                                                                                                                                                      |
| :-------------------------- | :-----------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Entrypoint Accuracy**     | $A_{\text{entry}}$  | Fraction of queries where the **very first** scored skill invoked matches target $T$ ($\vec{s}_1 = T$, or $\vec{s} = \emptyset$ for `out_of_scope` queries).                              |
| **Trajectory Reachability** |  $R_{\text{traj}}$  | Fraction of queries where target $T$ is reached anywhere in the scored trajectory ($T \in \vec{s}$, or $\vec{s} = \emptyset$ for `out_of_scope` queries).                                 |
| **Step Efficiency (MRR)**   |    $\text{MRR}$     | Reciprocal rank $\frac{1}{\text{step}}$ of the first scored invocation of $T$ for positive queries (`None` / excluded from macro-averages on `out_of_scope` queries).                     |
| **Skill Selection F1**      |        $F_1$        | Harmonic mean of set-level precision (target reached / unique scored skills invoked) and recall (target reached) for positive queries (`None` on `out_of_scope` queries).                 |
| **Skill Redundancy**        | $\text{Redundancy}$ | Excess scored invocations beyond the target: $\max(0, \text{len}(\vec{s}) - 1)$ for positive queries, or $\text{len}(\vec{s})$ for `out_of_scope` queries where any invocation is excess. |

---

## API Reference

<!-- prettier-ignore -->
::: reach.metrics
    options:
      show_root_heading: false
      members:
        - classification_report
        - ClassificationReport
        - classify_invocation_pattern
        - ClassMetrics
        - collisions
        - confusion
        - consistency
        - decompose_pass_rate_drop
        - DecompositionResult
        - score_trajectory
        - TrajectoryScore
