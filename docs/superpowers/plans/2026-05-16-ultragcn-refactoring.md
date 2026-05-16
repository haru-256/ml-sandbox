# UltraGCN Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the accepted UltraGCN/graph-model refactoring items while preserving current behavior and excluding F2.

**Architecture:** Keep all changes project-local under `projects/recsys-candidate-generation/`. Extract shared graph step boilerplate into a local mixin used by `LightGCNModule` and `UltraGCNModule`, add small private validation/co-occurrence helpers, and DRY repeated test scaffolding without moving anything into `libs/ml_sandbox_libs`.

**Tech Stack:** Python 3.12, PyTorch, PyTorch Geometric `HeteroData`, Lightning modules, OmegaConf, pytest, `uv`/`make`. No new dependencies.

---

## Scope and Constraints

- Implement accepted items: F1, F3, F4, F8, F9, F10, F11, F13.
- Do not implement F2; wait for a third occurrence.
- Do not change `libs/ml_sandbox_libs`.
- Work from package root `projects/recsys-candidate-generation/` for commands.
- Preserve current public behavior and keep all existing tests passing; expected test count is 65.

## File Structure

- Create `projects/recsys-candidate-generation/src/models/_graph_step_mixin.py`
  - Project-local mixin for graph `training_step` / `validation_step` boilerplate.
  - Owns `to_bipartite_graph_batch`, `_compute_step_outputs` invocation, monitor logging, retrieval input creation, and metric updates.
- Modify `projects/recsys-candidate-generation/src/models/lightgcn.py`
  - Remove duplicated graph step methods and imports now owned by the mixin.
  - Make `LightGCNModule` inherit the mixin before `BaseModule`.
- Modify `projects/recsys-candidate-generation/src/models/ultragcn.py`
  - Remove duplicated graph step methods and imports now owned by the mixin.
  - Make `UltraGCNModule` inherit the mixin before `BaseModule`.
  - Extract `_build_item_pairs_for_user()` from `_build_sparse_item_cooccurrence()`.
- Modify `projects/recsys-candidate-generation/src/config/validation.py`
  - Add scalar validation helpers and use them only in `validate_ultragcn_config`.
- Modify `projects/recsys-candidate-generation/src/models/factory.py`
  - Replace two datamodule guards with one typed guard helper.
- Modify `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py`
  - Add `_fake_optimizer()` helper, `ultragcn_module` fixture, and named expected co-occurrence values.
- Modify `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py`
  - Add one module-scope `_DummyModule` and use it in duplicated monkeypatch sites.

## Implementation Tasks

### Task 1: Extract shared graph step mixin (F1)

**Files:**
- Create: `projects/recsys-candidate-generation/src/models/_graph_step_mixin.py`
- Modify: `projects/recsys-candidate-generation/src/models/lightgcn.py`
- Modify: `projects/recsys-candidate-generation/src/models/ultragcn.py`

- [ ] **Step 1: Add the project-local mixin file**

Create `src/models/_graph_step_mixin.py` with this content:

```python
"""Shared graph Lightning step helpers for project-local graph modules."""

from typing import Protocol

import torch
from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsBipartiteGraphBatch,
    to_bipartite_graph_batch,
)
from ml_sandbox_libs.training import ExperimentMonitor, summarize_pos_neg_scores
from ml_sandbox_libs.utils.metrics import RetrievalMetrics, create_retrieval_inputs
from torch_geometric.data import HeteroData


class _GraphStepModuleProtocol(Protocol):
    """Protocol for graph modules that use the shared step implementation."""

    monitor: ExperimentMonitor
    retrieval_metrics: RetrievalMetrics

    def _compute_step_outputs(
        self,
        batch: AmazonReviewsBipartiteGraphBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss, positive scores, and negative scores for a typed graph batch."""


class GraphStepMixin:
    """Shared training and validation steps for bipartite graph modules.

    Classes using this mixin must provide `monitor`, `retrieval_metrics`, and
    `_compute_step_outputs()`.
    """

    def training_step(self: _GraphStepModuleProtocol, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Perform a single graph training step."""
        typed_batch = to_bipartite_graph_batch(batch)
        loss, pos_scores, neg_scores = self._compute_step_outputs(typed_batch)
        self.monitor.logging_step(
            {"loss": loss.item(), **summarize_pos_neg_scores(pos_scores, neg_scores)},
            stage="train",
            batch_idx=batch_idx,
            batch_size=typed_batch.src_index.size(0),
        )
        return loss

    def validation_step(
        self: _GraphStepModuleProtocol,
        batch: HeteroData,
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single graph validation step and update retrieval metrics."""
        typed_batch = to_bipartite_graph_batch(batch)
        loss, pos_scores, neg_scores = self._compute_step_outputs(typed_batch)
        scores, target, _ = create_retrieval_inputs(pos_scores, neg_scores)
        self.retrieval_metrics.update(scores, target)
        self.monitor.logging_step(
            {
                "loss": loss.item(),
                **summarize_pos_neg_scores(pos_scores, neg_scores),
                **self.retrieval_metrics.metric_dict(),
            },
            stage="val",
            batch_idx=batch_idx,
            batch_size=typed_batch.src_index.size(0),
        )
        return loss
```

- [ ] **Step 2: Update `LightGCNModule` imports and inheritance**

In `src/models/lightgcn.py`, remove `to_bipartite_graph_batch`, `summarize_pos_neg_scores`, `create_retrieval_inputs`, and `HeteroData` imports that are only used by `training_step` / `validation_step`. Keep `AmazonReviewsBipartiteGraphBatch`, `ExperimentMonitor`, and `RetrievalMetrics` imports.

Add this local import near `from .base import CandidateGenerationModelBase`:

```python
from ._graph_step_mixin import GraphStepMixin
```

Change the class header:

```python
class LightGCNModule(GraphStepMixin, BaseModule):
```

- [ ] **Step 3: Delete duplicated LightGCN step methods**

In `src/models/lightgcn.py`, delete the full `training_step()` and `validation_step()` methods currently between `_compute_step_outputs()` and `configure_optimizers()`. Do not change `_compute_step_outputs()`.

- [ ] **Step 4: Update `UltraGCNModule` imports and inheritance**

In `src/models/ultragcn.py`, remove `to_bipartite_graph_batch`, `summarize_pos_neg_scores`, `create_retrieval_inputs`, and `HeteroData` imports that are only used by `training_step` / `validation_step`. Keep `AmazonReviewsBipartiteGraphBatch`, `ExperimentMonitor`, and `RetrievalMetrics` imports.

Add this local import near `from .base import CandidateGenerationModelBase`:

```python
from ._graph_step_mixin import GraphStepMixin
```

Change the class header:

```python
class UltraGCNModule(GraphStepMixin, BaseModule):
```

- [ ] **Step 5: Delete duplicated UltraGCN step methods**

In `src/models/ultragcn.py`, delete the full `training_step()` and `validation_step()` methods currently between `_compute_step_outputs()` and `configure_optimizers()`. Do not change `_compute_step_outputs()`.

- [ ] **Step 6: Run graph model step tests**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest src/tests/test_models/test_lightgcn.py src/tests/test_models/test_ultragcn.py -v
```

Expected: all selected tests pass. Do not add `@override` to `GraphStepMixin` methods; the mixin implements a project-local step contract instead of overriding methods from its own ancestor.

### Task 2: Add UltraGCN scalar validation helpers (F3)

**Files:**
- Modify: `projects/recsys-candidate-generation/src/config/validation.py`

- [ ] **Step 1: Add private helper functions above `validate_lightgcn_neighbor_config`**

Insert after `is_graph_model()`:

```python
def _require_positive_scalar(cfg: DictConfig, field_name: str) -> None:
    """Require `cfg.model.<field_name>` to be greater than zero."""
    value = cfg.model[field_name]
    if value <= 0:
        raise ValueError(f"UltraGCN requires positive {field_name}, got {value}.")


def _require_non_negative_scalar(cfg: DictConfig, field_name: str) -> None:
    """Require `cfg.model.<field_name>` to be greater than or equal to zero."""
    value = cfg.model[field_name]
    if value < 0:
        raise ValueError(f"UltraGCN requires non-negative {field_name}, got {value}.")
```

- [ ] **Step 2: Replace duplicated UltraGCN scalar checks**

Replace the body of `validate_ultragcn_config()` after the docstring with:

```python
    _require_positive_scalar(cfg, "out_dim")
    _require_positive_scalar(cfg, "constraint_weight")
    _require_positive_scalar(cfg, "negative_weight")
    _require_non_negative_scalar(cfg, "item_constraint_weight")
    _require_positive_scalar(cfg, "item_constraint_top_k")
    _require_non_negative_scalar(cfg, "l2_weight")
```

Do not change `validate_lightgcn_neighbor_config()`.

- [ ] **Step 3: Run validation/factory tests**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest src/tests/test_models/test_factory.py -v
```

Expected: all selected tests pass; existing error-message substrings such as `positive` and `non-negative` remain present.

### Task 3: Split item-pair construction helper (F4)

**Files:**
- Modify: `projects/recsys-candidate-generation/src/models/ultragcn.py`

- [ ] **Step 1: Add `_build_item_pairs_for_user()` above `_build_sparse_item_cooccurrence()`**

Insert this helper after `_coalesce_item_pairs()`:

```python
def _build_item_pairs_for_user(interacted_items: torch.Tensor) -> torch.Tensor:
    """Build directed non-self item pairs for one user's unique interacted items.

    Args:
        interacted_items: Unique item ids for one user. Shape: ``(I,)``.

    Returns:
        Directed item-pair index tensor with shape ``(2, I * (I - 1))``.
    """
    item_count = interacted_items.numel()
    src_items = interacted_items.repeat_interleave(item_count)
    dst_items = interacted_items.repeat(item_count)
    non_self = src_items != dst_items
    return torch.stack([src_items[non_self], dst_items[non_self]], dim=0)
```

- [ ] **Step 2: Use the helper in `_build_sparse_item_cooccurrence()`**

Replace this block inside the loop:

```python
        src_items = interacted_items.repeat_interleave(item_count)
        dst_items = interacted_items.repeat(item_count)
        non_self = src_items != dst_items
        pair_index = torch.stack([src_items[non_self], dst_items[non_self]], dim=0)
```

with:

```python
        pair_index = _build_item_pairs_for_user(interacted_items)
```

Keep the outer chunking loop unchanged.

- [ ] **Step 3: Run UltraGCN co-occurrence tests**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest src/tests/test_models/test_ultragcn.py::test_build_ultragcn_constraint_weights_does_not_allocate_dense_item_matrix src/tests/test_models/test_ultragcn.py::test_build_ultragcn_constraint_weights_uses_unique_items_per_user_for_sparse_cooccurrence -v
```

Expected: both selected tests pass.

### Task 4: DRY UltraGCN test scaffolding (F8, F10, F11)

**Files:**
- Modify: `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py`

- [ ] **Step 1: Add `_fake_optimizer()` helper**

Insert after `_positive_neighbor_weights()`:

```python
def _fake_optimizer() -> Any:
    """Return a minimal optimizer strategy stub for module tests."""
    return cast(
        Any,
        SimpleNamespace(
            configure_optimizers=lambda _params: {"optimizer": object()},
            lr_scheduler_step=lambda *_args: None,
        ),
    )
```

- [ ] **Step 2: Add `ultragcn_module` fixture after `ultragcn` fixture**

```python
@pytest.fixture
def ultragcn_module(
    ultragcn_constraint_weights: UltraGCNConstraintWeights,
) -> UltraGCNModule:
    """Create an UltraGCNModule with shared default test arguments."""
    return UltraGCNModule(
        num_users=20,
        num_items=30,
        out_dim=8,
        constraint_weights=ultragcn_constraint_weights,
        negative_weight=1.0,
        item_constraint_weight=0.1,
        l2_weight=1e-4,
        eval_top_k=3,
        optimizer=_fake_optimizer(),
    )
```

The shared `eval_top_k=3` value is safe for `test_ultragcn_module_summary_runs` because `UltraGCNModule.summary()` does not read retrieval metric configuration.

- [ ] **Step 3: Replace duplicated module setup in four tests**

Update these tests to accept and use `ultragcn_module`:

```python
def test_ultragcn_module_training_step_returns_scalar_loss(
    bipartite_batch: HeteroData,
    ultragcn_module: UltraGCNModule,
) -> None:
    """Runs a training step on a sampled bipartite graph batch."""
    loss = ultragcn_module.training_step(bipartite_batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
```

```python
def test_ultragcn_module_validation_step_returns_scalar_loss(
    bipartite_batch: HeteroData,
    ultragcn_module: UltraGCNModule,
) -> None:
    """Runs a validation step and updates retrieval metrics."""
    loss = ultragcn_module.validation_step(bipartite_batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
```

```python
def test_ultragcn_module_summary_runs(ultragcn_module: UltraGCNModule) -> None:
    """Builds a torchinfo summary with synthetic triplet inputs."""
    model_summary = ultragcn_module.summary(batch_size=2)

    assert model_summary.total_params > 0
```

```python
def test_ultragcn_module_item_constraint_loss_returns_zero_when_disabled(
    ultragcn_constraint_weights: UltraGCNConstraintWeights,
) -> None:
    """Skips item-neighbor lookup when the item-item constraint is disabled."""
    module = UltraGCNModule(
        num_users=20,
        num_items=30,
        out_dim=8,
        constraint_weights=ultragcn_constraint_weights,
        negative_weight=1.0,
        item_constraint_weight=0.0,
        l2_weight=1e-4,
        eval_top_k=3,
        optimizer=_fake_optimizer(),
    )

    loss = module._item_constraint_loss(torch.tensor([10_000], dtype=torch.long))

    assert loss.ndim == 0
    assert loss.item() == 0.0
```

This removes the four duplicated inline `SimpleNamespace(...)` optimizer constructions while keeping the disabled-item-constraint test's special `item_constraint_weight=0.0` setup explicit.

- [ ] **Step 4: Name expected co-occurrence values in the sparse co-occurrence test**

In `test_build_ultragcn_constraint_weights_uses_unique_items_per_user_for_sparse_cooccurrence`, replace inline math in `pytest.approx(...)` with named variables after the `item_degree` assertion:

```python
    item0_item1_weight = 2.0 / torch.sqrt(torch.tensor(4.0 * 4.0)).item()
    item0_item2_weight = 2.0 / torch.sqrt(torch.tensor(4.0 * 3.0)).item()
    item1_item2_weight = 1.0 / torch.sqrt(torch.tensor(4.0 * 3.0)).item()
    item2_item0_weight = 2.0 / torch.sqrt(torch.tensor(3.0 * 4.0)).item()
    item2_item1_weight = 1.0 / torch.sqrt(torch.tensor(3.0 * 4.0)).item()
```

Then assert:

```python
    assert _positive_neighbor_weights(weights, item_id=0) == pytest.approx(
        {1: item0_item1_weight, 2: item0_item2_weight}
    )
    assert _positive_neighbor_weights(weights, item_id=1) == pytest.approx(
        {0: item0_item1_weight, 2: item1_item2_weight}
    )
    assert _positive_neighbor_weights(weights, item_id=2) == pytest.approx(
        {0: item2_item0_weight, 1: item2_item1_weight}
    )
```

- [ ] **Step 5: Run UltraGCN tests**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest src/tests/test_models/test_ultragcn.py -v
```

Expected: all selected tests pass.

### Task 5: Unify factory test dummy module (F9)

**Files:**
- Modify: `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py`

- [ ] **Step 1: Add module-scope `_DummyModule`**

Insert after imports:

```python
class _DummyModule:
    """Capture constructor kwargs for factory tests."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
```

- [ ] **Step 2: Replace local `DummyModule` classes with closures using `_DummyModule`**

Apply this replacement inside each test function that has a local `captured_kwargs` dictionary: `test_model_creators_use_pad_idx_from_datamodule`, `test_create_lightgcn_module_uses_embedding_loss_factory_with_bpr`, and `test_create_ultragcn_module_builds_constraint_weights`. Define `fake_module` as a nested function inside each test so it closes over that test's local `captured_kwargs` dictionary.

For each location that currently defines:

```python
    class DummyModule:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
```

replace it with:

```python
    def fake_module(**kwargs: Any) -> _DummyModule:
        module = _DummyModule(**kwargs)
        captured_kwargs.update(module.kwargs)
        return module
```

Then replace monkeypatch calls such as:

```python
    monkeypatch.setattr(factory, module_name, DummyModule)
```

with:

```python
    monkeypatch.setattr(factory, module_name, fake_module)
```

Apply the same pattern for `LightGCNModule` and `UltraGCNModule` monkeypatches.

- [ ] **Step 3: Run factory tests**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest src/tests/test_models/test_factory.py -v
```

Expected: all selected tests pass.

### Task 6: Merge datamodule guard functions (F13)

**Files:**
- Modify: `projects/recsys-candidate-generation/src/models/factory.py`
- Test coverage: `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py`

- [ ] **Step 1: Import `TypeVar`**

Change the top imports in `src/models/factory.py` to include:

```python
from typing import TypeVar
```

- [ ] **Step 2: Add a datamodule type alias and type variable near imports**

After the imports, add:

```python
Datamodule = AmazonReviewsSeqRecDataModule | AmazonReviewsBipartiteGraphDataModule
DatamoduleT = TypeVar("DatamoduleT", bound=Datamodule)
```

- [ ] **Step 3: Replace the two guard functions with one helper**

Delete `_require_seq_rec_datamodule()` and `_require_bipartite_graph_datamodule()`. Add this replacement in the same location:

```python
def _require_datamodule_type(
    datamodule: Datamodule,
    expected_type: type[DatamoduleT],
    expected_type_name: str,
    model_name: str,
) -> DatamoduleT:
    """Return a datamodule of the expected type or raise a clear type error.

    Args:
        datamodule: Datamodule selected by the project data factory.
        expected_type: Concrete datamodule class required by the model.
        expected_type_name: Stable public datamodule class name for error messages.
        model_name: Model name being constructed.

    Returns:
        Datamodule narrowed to `expected_type`.

    Raises:
        TypeError: If the selected datamodule is not compatible with the model.
    """
    if not isinstance(datamodule, expected_type):
        raise TypeError(f"{model_name} requires {expected_type_name}")
    return datamodule
```

- [ ] **Step 4: Update `create_model_module()` call sites**

Change the function signature to use the alias:

```python
def create_model_module(
    cfg: DictConfig,
    datamodule: Datamodule,
    optimizer: AdamWCosine,
) -> BaseModule:
```

Replace each sequential model guard with:

```python
seq_rec_datamodule = _require_datamodule_type(
    datamodule,
    AmazonReviewsSeqRecDataModule,
    "AmazonReviewsSeqRecDataModule",
    model_name="TwoTower",
)
```

Use the corresponding `model_name` for `SASRec`, `gSASRec`, and `SimpleX`.

Replace graph model guards with:

```python
bipartite_datamodule = _require_datamodule_type(
    datamodule,
    AmazonReviewsBipartiteGraphDataModule,
    "AmazonReviewsBipartiteGraphDataModule",
    model_name="LightGCN",
)
```

and the same shape with `model_name="UltraGCN"` for UltraGCN.

- [ ] **Step 5: Update dispatch test monkeypatches**

In `test_create_model_module_dispatches_to_matching_creator`, replace:

```python
    monkeypatch.setattr(factory, "_require_seq_rec_datamodule", lambda dm, **_: dm)
    monkeypatch.setattr(factory, "_require_bipartite_graph_datamodule", lambda dm, **_: dm)
```

with:

```python
    monkeypatch.setattr(factory, "_require_datamodule_type", lambda dm, *_args, **_kwargs: dm)
```

- [ ] **Step 6: Run factory tests**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest src/tests/test_models/test_factory.py -v
```

Expected: all selected tests pass, including existing type-error message checks.

### Task 7: Final formatting, linting, and verification

**Files:**
- All files changed by Tasks 1-6.

- [ ] **Step 1: Run formatter**

Run from `projects/recsys-candidate-generation/`:

```bash
make fmt
```

Expected: formatter completes successfully. Review formatter changes to ensure they are limited to touched project files.

- [ ] **Step 2: Run lint/type checks**

Run from `projects/recsys-candidate-generation/`:

```bash
make lint
```

Expected: lint and mypy checks pass. If mypy reports a generic type issue in `_require_datamodule_type`, keep the single helper but adjust type annotations locally; do not reintroduce the two old guard functions.

- [ ] **Step 3: Run all package tests**

Run from `projects/recsys-candidate-generation/`:

```bash
make test
```

Expected: all 65 existing tests pass.

- [ ] **Step 4: Run the required combined verification command**

Run from `projects/recsys-candidate-generation/`:

```bash
make fmt && make lint && make test
```

Expected: all commands pass in sequence.

## Verification

Primary required verification from `projects/recsys-candidate-generation/`:

```bash
make fmt && make lint && make test
```

Expected final result: formatter succeeds, lint/type checks pass, and all 65 tests pass.

## Self-Review

- Spec coverage: Tasks 1-6 map directly to F1, F3, F4, F8, F9, F10, F11, and F13. F2 is explicitly excluded.
- Placeholder scan: No TBD/placeholder implementation steps remain; each code-changing task includes exact file paths and concrete snippets.
- Type consistency: The graph mixin depends on existing `_compute_step_outputs()` signatures returning `(loss, pos_scores, neg_scores)`; factory guard call sites preserve existing error message class names.

## Implementation Log
<!-- Implementer appends one line per attempt: [YYYY-MM-DD] attempt #N -> STATUS | commit-or-failure-signature -->
- [2026-05-16] attempt #1 -> DONE | All 8 refactoring items (F1, F3, F4, F8, F9, F10, F11, F13) applied successfully. 65 tests pass. No commit.

## Review Findings
<!-- This template is also defined in commands/plan-v2.md. Keep them in sync on every edit. -->

### Reviewer Raw Findings
<!-- Planner V2 copies @reviewer_v2's structured findings verbatim here when invoking @reviewer_v2 during a workflow. Direct /review-*-v2 calls do not write here. Raw findings are review input, not implementation instructions. -->

#### [2026-05-16] PLAN -> REQUEST_CHANGES
Critical issues:
- F1: BLOCKER | HIGH | correctness
  - **Evidence:** Plan lines 508-529 (Task 6 Step 3) defines `_require_datamodule_type` with `f"{model_name} requires {expected_type.__name__}"`. Test file `src/tests/test_models/test_factory.py` lines 338-357 and 360-383 monkeypatch `factory.AmazonReviewsSeqRecDataModule` / `factory.AmazonReviewsBipartiteGraphDataModule` with local classes `FakeSeqRecDataModule` / `FakeGraphDataModule`. Their `__name__` values are `"FakeSeqRecDataModule"` and `"FakeGraphDataModule"`, but the existing `pytest.raises(match=...)` patterns expect `"AmazonReviewsSeqRecDataModule"` and `"AmazonReviewsBipartiteGraphDataModule"`.
  - **Why it matters:** Two tests (`test_create_model_module_rejects_bipartite_datamodule_for_seq_rec_model` and `test_create_model_module_rejects_seq_rec_datamodule_for_graph_models`) will fail with a `TypeError` match failure, violating the constraint that all 65 tests must pass.
  - **Recommended action:** Either (a) hard-code the expected class name strings in `_require_datamodule_type` (e.g., pass a `expected_type_name: str` parameter alongside `expected_type`), or (b) update the test `match=` patterns to use the monkeypatched class names. Option (a) preserves the original error message semantics better.
  - **Must fix before merge:** yes
- F2: MAJOR | MEDIUM | plan
  - **Evidence:** Plan lines 422-468 (Task 5 Step 2) says "For each location that currently defines: `class DummyModule:` … replace it with: [closure]" and then "Apply the same pattern for `LightGCNModule` and `UltraGCNModule` monkeypatches." It never states that `fake_module` must be defined *inside* each of the three test functions (`test_model_creators_use_pad_idx_from_datamodule`, `test_create_lightgcn_module_uses_embedding_loss_factory_with_bpr`, `test_create_ultragcn_module_builds_constraint_weights`) to capture each function's local `captured_kwargs`.
  - **Why it matters:** If an implementer places `fake_module` at module scope, `captured_kwargs` will be undefined, causing a `NameError` at test collection time. The plan is underspecified for the executability acceptance criterion.
  - **Recommended action:** Add an explicit sentence: "Define `fake_module` as a nested function inside each test to close over the test-local `captured_kwargs` dict." Also list each test function name that needs the replacement.
  - **Must fix before merge:** yes
Non-blocking suggestions:
- S1: MAJOR | HIGH | maintainability
  - **Evidence:** Plan lines 90-125 (Task 1 Step 1) places `@override` on `GraphStepMixin.training_step` and `GraphStepMixin.validation_step`. `GraphStepMixin` does not inherit from any class that defines these methods; `BaseModule` (from `libs/ml_sandbox_libs/src/ml_sandbox_libs/models/base/module.py` line 9) inherits from `L.LightningModule` but `GraphStepMixin` is not in that hierarchy. Mypy will reject both `@override` decorators.
  - **Why it matters:** The plan's mitigation ("If mypy rejects @override … remove the two decorators") is reactive — it bakes in a guaranteed first-build failure of `make lint`, which should be avoidable.
  - **Recommended action:** Remove the two `@override` decorators from `GraphStepMixin` proactively. Add a comment noting they are deliberately omitted because the mixin implements the step contract rather than overriding its own ancestor. The child classes (`LightGCNModule`, `UltraGCNModule`) will still have `@override` on their own methods (e.g., `configure_optimizers`, `summary`) via `BaseModule`.
  - **Must fix before merge:** uncertain (the plan acknowledges the issue, but proactively removing the decorators saves a build cycle)
- S2: MINOR | LOW | correctness
  - **Evidence:** Plan line 310 (`ultragcn_module` fixture) uses `eval_top_k=3`. The existing `test_ultragcn_module_summary_runs` at line 241 uses `eval_top_k=5`. The `summary()` method (ultragcn.py line 583) does not reference `self.eval_top_k` at all, so the value is irrelevant.
  - **Why it matters:** No behavioral impact, but a reviewer might flag the inconsistency.
  - **Recommended action:** Add a brief note in Task 4 Step 2 confirming `eval_top_k=3` is safe because `summary()` ignores it.
- S3: NIT | LOW | docs
  - **Evidence:** The plan-rendered code snippets for the mixin (lines 55-124) and the `_require_datamodule_type` function (lines 507-528) lack module-level docstrings. While the plan snippets are meant as instruction, the implementer should include them per the repo's Google-style docstring requirement.
  - **Why it matters:** `make lint` (ruff) may or may not enforce module docstrings depending on configuration. Low risk.
  - **Recommended action:** Add a brief note: "Include a module-level docstring at the top of new files per project convention."

### Planner V2 Adjudication
<!-- Planner V2 appends adjudication tables for v2 workflow reviews. Only ACCEPT rows are implementation instructions: | ID | Severity | Decision | Reason | Action | -->

#### [2026-05-16] PLAN review adjudication

| ID | Severity | Decision | Reason | Action |
|----|----------|----------|--------|--------|
| F1 | BLOCKER | ACCEPT | Would break existing factory tests because monkeypatched fake classes change `__name__`; concrete evidence and violates 65-test acceptance criterion. | Updated Task 6 to pass a stable `expected_type_name` string and preserve existing error-message semantics. |
| F2 | MAJOR | ACCEPT | Plan wording was ambiguous enough to permit a non-working module-scope closure. | Updated Task 5 to require nested `fake_module` definitions inside each affected test and listed the test names. |
| S1 | MAJOR | ACCEPT | Avoidable mypy/lint failure; fix is local and proportionate. | Removed `@override` from the planned mixin snippet and replaced reactive mitigation with a proactive note. |
| S2 | MINOR | ACCEPT | Harmless but clarifying reduces reviewer churn. | Added a note that `eval_top_k=3` is safe because `summary()` does not read it. |
| S3 | NIT | REJECT | The new mixin snippet already includes a module-level docstring; `_require_datamodule_type` is not a new file and has a function docstring. | No action. |

## Deviations from Plan
<!-- Implementer documents intentional deviations and reasons. -->

## Open Questions
<!-- Any agent adds questions for planner_v2 or oracle_v2. -->
