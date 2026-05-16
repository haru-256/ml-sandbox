# Remove GraphStepMixin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `GraphStepMixin` from `projects/recsys-candidate-generation/` without duplicating the shared graph `training_step` / `validation_step` logic or moving it into `libs/ml_sandbox_libs`.

**Architecture:** Replace mixin inheritance with project-local module-level helper functions in `src/models/_graph_steps.py`. `LightGCNModule` and `UltraGCNModule` inherit only from `BaseModule` and expose explicit Lightning hooks that delegate to the helpers, while model-specific computation remains in `_compute_step_outputs()`.

**Tech Stack:** Python 3.12, PyTorch Lightning, PyTorch Geometric `HeteroData`, `uv`, `ruff`, `mypy`, `pytest`.

---

## Context

Current implementation:

- `src/models/_graph_step_mixin.py` defines `GraphStepMixin.training_step()` and `GraphStepMixin.validation_step()`.
- `LightGCNModule(GraphStepMixin, BaseModule)` and `UltraGCNModule(GraphStepMixin, BaseModule)` inherit these hooks.
- The shared hooks call `self._compute_step_outputs(typed_batch)`, which is model-specific.
- Existing behavior is covered by:
  - `src/tests/test_models/test_lightgcn.py::test_lightgcn_module_training_step_returns_scalar_loss`
  - `src/tests/test_models/test_lightgcn.py::test_lightgcn_module_validation_step_returns_scalar_loss`
  - `src/tests/test_models/test_ultragcn.py::test_ultragcn_module_training_step_returns_scalar_loss`
  - `src/tests/test_models/test_ultragcn.py::test_ultragcn_module_validation_step_returns_scalar_loss`

## Approach Comparison

### Recommended: module-level helper functions

Create `src/models/_graph_steps.py` with `graph_training_step(module, batch, batch_idx)` and `graph_validation_step(module, batch, batch_idx)`. The helpers keep all shared logging, metric, batch conversion, and retrieval-input logic in one project-local place. Each Lightning module gets short, explicit hook methods that are easy to discover and delegate to the helpers.

Why this is best:

- Removes mixin inheritance and the confusing MRO.
- Avoids duplicating the body of `training_step` / `validation_step`.
- Keeps the abstraction stateless and project-local.
- Makes the dependency contract explicit with a `Protocol` instead of implicit inherited behavior.
- Requires only minimal source changes and no `libs/ml_sandbox_libs` changes.

### Alternative: standalone `GraphStepHelper` class

Instantiate `self.graph_steps = GraphStepHelper(self)` in each module and delegate hooks to it. This also avoids mixin inheritance, but adds object state and lifecycle questions without a benefit because the helper does not own meaningful mutable state. It is more indirection than the module-level function approach.

### Alternative: copy shared hook code into each module

Paste the shared hook bodies into `LightGCNModule` and `UltraGCNModule`. This is the most explicit locally, but it violates the requirement not to duplicate `training_step` / `validation_step` logic and creates future drift risk in logging and retrieval metrics.

## File Structure

- Create: `projects/recsys-candidate-generation/src/models/_graph_steps.py`
  - Owns project-local graph Lightning step helper functions and their protocol contract.
- Modify: `projects/recsys-candidate-generation/src/models/lightgcn.py`
  - Replace mixin import with helper imports.
  - Change class inheritance to `BaseModule` only.
  - Add explicit delegate `training_step()` and `validation_step()` methods.
- Modify: `projects/recsys-candidate-generation/src/models/ultragcn.py`
  - Same as `lightgcn.py`.
- Delete: `projects/recsys-candidate-generation/src/models/_graph_step_mixin.py`
  - The mixin class no longer exists.
- Tests: no new tests are required unless existing tests reveal a behavior gap. The existing step tests validate the public Lightning hook contract without depending on implementation internals.

## Acceptance Criteria

1. `GraphStepMixin` no longer exists in `projects/recsys-candidate-generation/`.
2. `LightGCNModule` and `UltraGCNModule` inherit from `BaseModule` only.
3. Shared graph step logic exists once in a project-local helper module.
4. No files under `libs/ml_sandbox_libs` are changed.
5. Existing training and validation behavior is preserved, including monitor logging and retrieval metric updates.
6. `make lint` and `make test` pass from `projects/recsys-candidate-generation/`; the expected test count is 66 passing tests.

## Task 1: Add project-local graph step helper functions

**Files:**

- Create: `projects/recsys-candidate-generation/src/models/_graph_steps.py`
- Read for reference only: `projects/recsys-candidate-generation/src/models/_graph_step_mixin.py`

- [ ] **Step 1: Create `_graph_steps.py` with stateless helper functions**

Create `projects/recsys-candidate-generation/src/models/_graph_steps.py` with this complete content:

```python
"""Project-local graph Lightning step helpers for candidate generation models."""

from typing import Protocol

import torch
from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsBipartiteGraphBatch,
    to_bipartite_graph_batch,
)
from ml_sandbox_libs.training import ExperimentMonitor, summarize_pos_neg_scores
from ml_sandbox_libs.utils.metrics import RetrievalMetrics, create_retrieval_inputs
from torch_geometric.data import HeteroData


class GraphStepModule(Protocol):
    """Protocol for graph modules that delegate shared Lightning steps.

    Implementers provide model-specific loss and score computation through
    `_compute_step_outputs()` while the helper functions handle common batch
    conversion, logging, and retrieval metric updates.
    """

    monitor: ExperimentMonitor
    retrieval_metrics: RetrievalMetrics

    def _compute_step_outputs(
        self,
        batch: AmazonReviewsBipartiteGraphBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss, positive scores, and negative scores for a graph batch.

        Args:
            batch: Typed bipartite graph batch.

        Returns:
            Tuple containing scalar loss, positive scores, and negative scores.
        """


def graph_training_step(
    module: GraphStepModule,
    batch: HeteroData,
    batch_idx: int,
) -> torch.Tensor:
    """Run the shared graph training step for a candidate-generation module.

    Args:
        module: Module that owns monitor state and model-specific step computation.
        batch: Raw sampled bipartite graph batch from the dataloader.
        batch_idx: Lightning batch index used for logging.

    Returns:
        Scalar loss tensor produced by the module-specific step computation.
    """
    typed_batch = to_bipartite_graph_batch(batch)
    loss, pos_scores, neg_scores = module._compute_step_outputs(typed_batch)
    module.monitor.logging_step(
        {"loss": loss.item(), **summarize_pos_neg_scores(pos_scores, neg_scores)},
        stage="train",
        batch_idx=batch_idx,
        batch_size=typed_batch.src_index.size(0),
    )
    return loss


def graph_validation_step(
    module: GraphStepModule,
    batch: HeteroData,
    batch_idx: int,
) -> torch.Tensor:
    """Run the shared graph validation step for a candidate-generation module.

    Args:
        module: Module that owns monitor state, retrieval metrics, and
            model-specific step computation.
        batch: Raw sampled bipartite graph batch from the dataloader.
        batch_idx: Lightning batch index used for logging.

    Returns:
        Scalar loss tensor produced by the module-specific step computation.
    """
    typed_batch = to_bipartite_graph_batch(batch)
    loss, pos_scores, neg_scores = module._compute_step_outputs(typed_batch)
    scores, target, _ = create_retrieval_inputs(pos_scores, neg_scores)
    module.retrieval_metrics.update(scores, target)
    module.monitor.logging_step(
        {
            "loss": loss.item(),
            **summarize_pos_neg_scores(pos_scores, neg_scores),
            **module.retrieval_metrics.metric_dict(),
        },
        stage="val",
        batch_idx=batch_idx,
        batch_size=typed_batch.src_index.size(0),
    )
    return loss
```

- [ ] **Step 2: Run targeted syntax/import validation**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run python -m compileall src/models/_graph_steps.py
```

Expected: command exits 0 and reports successful compilation for `_graph_steps.py`.

## Task 2: Update `LightGCNModule` to delegate explicit hooks

**Files:**

- Modify: `projects/recsys-candidate-generation/src/models/lightgcn.py:1-22`
- Modify: `projects/recsys-candidate-generation/src/models/lightgcn.py:298-409`
- Test: `projects/recsys-candidate-generation/src/tests/test_models/test_lightgcn.py`

- [ ] **Step 1: Replace the mixin import**

In `projects/recsys-candidate-generation/src/models/lightgcn.py`, replace:

```python
from ._graph_step_mixin import GraphStepMixin
```

with:

```python
from torch_geometric.data import HeteroData

from ._graph_steps import graph_training_step, graph_validation_step
```

Keep the existing `from .base import CandidateGenerationModelBase` import below the helper import.

- [ ] **Step 2: Remove mixin inheritance**

In `projects/recsys-candidate-generation/src/models/lightgcn.py`, replace:

```python
class LightGCNModule(GraphStepMixin, BaseModule):
```

with:

```python
class LightGCNModule(BaseModule):
```

- [ ] **Step 3: Add explicit Lightning hook delegates**

Insert these methods after `forward()` and before `_compute_step_outputs()` in `LightGCNModule`:

```python
    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Perform a single graph training step."""
        return graph_training_step(self, batch, batch_idx)

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Perform a single graph validation step and update retrieval metrics."""
        return graph_validation_step(self, batch, batch_idx)
```

- [ ] **Step 4: Run LightGCN step tests**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest \
  src/tests/test_models/test_lightgcn.py::test_lightgcn_module_training_step_returns_scalar_loss \
  src/tests/test_models/test_lightgcn.py::test_lightgcn_module_validation_step_returns_scalar_loss \
  -v
```

Expected: both selected tests pass.

## Task 3: Update `UltraGCNModule` to delegate explicit hooks

**Files:**

- Modify: `projects/recsys-candidate-generation/src/models/ultragcn.py:1-20`
- Modify: `projects/recsys-candidate-generation/src/models/ultragcn.py:406-551`
- Test: `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py`

- [ ] **Step 1: Replace the mixin import**

In `projects/recsys-candidate-generation/src/models/ultragcn.py`, replace:

```python
from ._graph_step_mixin import GraphStepMixin
```

with:

```python
from torch_geometric.data import HeteroData

from ._graph_steps import graph_training_step, graph_validation_step
```

Keep the existing `from .base import CandidateGenerationModelBase` import below the helper import.

- [ ] **Step 2: Remove mixin inheritance**

In `projects/recsys-candidate-generation/src/models/ultragcn.py`, replace:

```python
class UltraGCNModule(GraphStepMixin, BaseModule):
```

with:

```python
class UltraGCNModule(BaseModule):
```

- [ ] **Step 3: Add explicit Lightning hook delegates**

Insert these methods after `forward()` and before `_batch_ids()` in `UltraGCNModule`:

```python
    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Perform a single graph training step."""
        return graph_training_step(self, batch, batch_idx)

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Perform a single graph validation step and update retrieval metrics."""
        return graph_validation_step(self, batch, batch_idx)
```

- [ ] **Step 4: Run UltraGCN step tests**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest \
  src/tests/test_models/test_ultragcn.py::test_ultragcn_module_training_step_returns_scalar_loss \
  src/tests/test_models/test_ultragcn.py::test_ultragcn_module_validation_step_returns_scalar_loss \
  -v
```

Expected: both selected tests pass.

## Task 4: Delete the mixin module and verify no references remain

**Files:**

- Delete: `projects/recsys-candidate-generation/src/models/_graph_step_mixin.py`
- Search scope: `projects/recsys-candidate-generation/`

- [ ] **Step 1: Delete the obsolete mixin file**

Remove:

```text
projects/recsys-candidate-generation/src/models/_graph_step_mixin.py
```

- [ ] **Step 2: Verify `GraphStepMixin` is gone**

Run from repository root:

```bash
rg "GraphStepMixin|_graph_step_mixin" projects/recsys-candidate-generation
```

Expected: no matches.

- [ ] **Step 3: Verify helper references are limited and readable**

Run from repository root:

```bash
rg "graph_training_step|graph_validation_step|GraphStepModule" projects/recsys-candidate-generation/src/models
```

Expected: matches only in `src/models/_graph_steps.py`, `src/models/lightgcn.py`, and `src/models/ultragcn.py`.

## Task 5: Run formatter, lint, and full package tests

**Files:**

- Validate package: `projects/recsys-candidate-generation/`

- [ ] **Step 1: Run formatter**

Run from `projects/recsys-candidate-generation/`:

```bash
make fmt
```

Expected: command exits 0. If `ruff` formats imports or wrapping, inspect the diff and keep only related changes.

- [ ] **Step 2: Run lint**

Run from `projects/recsys-candidate-generation/`:

```bash
make lint
```

Expected: command exits 0 with `ruff` and `mypy` passing.

- [ ] **Step 3: Run the full test suite**

Run from `projects/recsys-candidate-generation/`:

```bash
make test
```

Expected: command exits 0 and all 66 tests pass.

- [ ] **Step 4: Inspect final diff for scope**

Run from repository root:

```bash
git diff -- projects/recsys-candidate-generation/src/models/_graph_steps.py \
  projects/recsys-candidate-generation/src/models/_graph_step_mixin.py \
  projects/recsys-candidate-generation/src/models/lightgcn.py \
  projects/recsys-candidate-generation/src/models/ultragcn.py
```

Expected: diff contains only project-local helper creation, mixin deletion, import/inheritance updates, and delegate hook additions.

## Implementation Notes

- Do not change `libs/ml_sandbox_libs`.
- Do not add a `GraphStepHelper` class unless lint/type checking reveals a concrete issue with the module-level helper approach.
- Do not add tests that assert class inheritance internals or method source locations; those would be brittle implementation-detail tests.
- Keep `_compute_step_outputs()` in each model module. It is the model-specific seam and should not move into the helper.
- If `mypy` complains that `self` does not satisfy `GraphStepModule`, prefer widening the helper protocol minimally over adding `cast()` calls in model hooks. The target is readable model code.

## Self-Review

- Spec coverage: all requested constraints are covered by the file structure, acceptance criteria, and tasks.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: helper, import, and delegate names are consistent across all tasks.

## Implementation Log

<!-- Implementer appends one line per attempt: [YYYY-MM-DD] attempt #N -> STATUS | commit-or-failure-signature -->

## Review Findings

<!-- This template is also defined in commands/plan-v2.md. Keep them in sync on every edit. -->

### Reviewer Raw Findings

<!-- Planner V2 copies @reviewer_v2's structured findings verbatim here when invoking @reviewer_v2 during a workflow. Direct /review-*-v2 calls do not write here. Raw findings are review input, not implementation instructions. -->

### Planner V2 Adjudication

<!-- Planner V2 appends adjudication tables for v2 workflow reviews. Only ACCEPT rows are implementation instructions: | ID | Severity | Decision | Reason | Action | -->

## Deviations from Plan

<!-- Implementer documents intentional deviations and reasons. -->

## Open Questions

<!-- Any agent adds questions for planner_v2 or oracle_v2. -->
