# UltraGCN Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address accepted UltraGCN review findings before merge while keeping deferred scalability/performance work out of this PR.

**Architecture:** Keep fixes local to `projects/recsys-candidate-generation`. Replace the factory's NumPy-backed edge tensor conversion with direct Polars-to-Torch stacking, correct the README example so it reflects UltraGCN's internal loss, and add focused regression tests for currently untested UltraGCN constraint paths.

**Tech Stack:** Python 3.12, PyTorch, Polars, pytest, Hydra/OmegaConf, `uv`/`make` package workflow.

---

## File Structure

- Modify `projects/recsys-candidate-generation/src/models/factory.py`
  - Build UltraGCN `edge_index` from writable Torch tensors instead of a transposed NumPy view.
- Modify `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py`
  - Strengthen the existing UltraGCN factory test to assert `edge_index` shape, dtype, and ordering through the constraint builder boundary.
- Modify `projects/recsys-candidate-generation/README.md`
  - Remove misleading `loss=bpr` from the UltraGCN example because `UltraGCNModule` owns its custom loss and factory does not call `loss.factory`.
- Modify `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py`
  - Add tests for `item_constraint_weight=0` early return and `num_items < item_constraint_top_k` padding.

## Implementation Tasks

### Task 1: Replace UltraGCN factory edge conversion

**Files:**
- Modify: `projects/recsys-candidate-generation/src/models/factory.py:266-270`
- Modify: `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py:261-309`

- [ ] **Step 1: Strengthen the factory test around constructed `edge_index`**

In `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py`, update `test_create_ultragcn_module_builds_constraint_weights` so it captures the `edge_index` passed into `build_ultragcn_constraint_weights`:

```python
def test_create_ultragcn_module_builds_constraint_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Builds UltraGCN constraints from train edges in the prepared graph datamodule."""
    cfg = OmegaConf.create(
        {
            "data": {"eval_top_k": 10},
            "model": {
                "name": "UltraGCN",
                "out_dim": 16,
                "constraint_weight": 1.0,
                "negative_weight": 1.0,
                "item_constraint_weight": 0.1,
                "item_constraint_top_k": 2,
                "l2_weight": 1e-4,
            },
        }
    )
    datamodule = cast(
        Any,
        SimpleNamespace(
            num_users=3,
            num_items=4,
            all_df=__import__("polars").DataFrame(
                {
                    "split": ["train", "train", "valid"],
                    "user_index": [0, 1, 2],
                    "item_index": [0, 1, 2],
                }
            ),
        ),
    )
    optimizer = cast(Any, SimpleNamespace())
    captured_kwargs: dict[str, Any] = {}
    captured_edge_index: dict[str, torch.Tensor] = {}

    class DummyModule:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    def fake_build_ultragcn_constraint_weights(
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
        constraint_weight: float,
        item_constraint_top_k: int,
    ) -> UltraGCNConstraintWeights:
        captured_edge_index["value"] = edge_index
        return build_ultragcn_constraint_weights(
            edge_index=edge_index,
            num_users=num_users,
            num_items=num_items,
            constraint_weight=constraint_weight,
            item_constraint_top_k=item_constraint_top_k,
        )

    monkeypatch.setattr(factory, "UltraGCNModule", DummyModule)
    monkeypatch.setattr(
        factory,
        "build_ultragcn_constraint_weights",
        fake_build_ultragcn_constraint_weights,
    )

    factory.create_ultragcn_module(cfg, datamodule, optimizer)

    edge_index = captured_edge_index["value"]
    assert edge_index.dtype == torch.long
    assert edge_index.shape == (2, 2)
    assert torch.equal(edge_index, torch.tensor([[0, 1], [0, 1]], dtype=torch.long))
    assert captured_kwargs["num_users"] == 3
    assert captured_kwargs["num_items"] == 4
    assert captured_kwargs["out_dim"] == 16
    assert captured_kwargs["optimizer"] is optimizer
    assert captured_kwargs["constraint_weights"].user_indices.tolist() == [0, 1]
    assert captured_kwargs["constraint_weights"].item_indices.tolist() == [0, 1]
```

If the file does not already import `torch` or `UltraGCNConstraintWeights`, add them:

```python
import torch

from models.ultragcn import UltraGCNConstraintWeights, build_ultragcn_constraint_weights
```

- [ ] **Step 2: Run the focused factory test and verify current behavior**

Run from `projects/recsys-candidate-generation`:

```bash
uv run pytest src/tests/test_models/test_factory.py::test_create_ultragcn_module_builds_constraint_weights -v
```

Expected: PASS before and after the implementation; this test protects the edge tensor contract while the implementation removes the warning-prone conversion path.

- [ ] **Step 3: Replace `torch.as_tensor(...to_numpy().T)` with direct Torch stacking**

In `projects/recsys-candidate-generation/src/models/factory.py`, replace:

```python
    edge_index = torch.as_tensor(
        train_df.select(["user_index", "item_index"]).to_numpy().T,
        dtype=torch.long,
    )
```

with:

```python
    edge_index = torch.stack(
        [
            train_df["user_index"].to_torch().to(torch.long),
            train_df["item_index"].to_torch().to(torch.long),
        ],
        dim=0,
    )
```

- [ ] **Step 4: Re-run the focused factory test**

Run from `projects/recsys-candidate-generation`:

```bash
uv run pytest src/tests/test_models/test_factory.py::test_create_ultragcn_module_builds_constraint_weights -v
```

Expected: PASS.

### Task 2: Add focused UltraGCN constraint regression tests

**Files:**
- Modify: `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py`

- [ ] **Step 1: Add test for `num_items < item_constraint_top_k` padding**

Append this test near the existing `build_ultragcn_constraint_weights` tests:

```python
def test_build_ultragcn_constraint_weights_pads_item_neighbors_when_top_k_exceeds_items() -> None:
    """Pads item-neighbor tensors when top-k is larger than the item vocabulary."""
    edge_index = torch.tensor(
        [
            [0, 0],
            [0, 1],
        ],
        dtype=torch.long,
    )

    weights = build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=1,
        num_items=2,
        constraint_weight=1.0,
        item_constraint_top_k=4,
    )

    assert weights.item_neighbor_indices.shape == (2, 4)
    assert weights.item_neighbor_weights.shape == (2, 4)
    assert torch.equal(
        weights.item_neighbor_indices[:, 2:],
        torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
    )
    assert torch.equal(weights.item_neighbor_weights[:, 2:], torch.zeros((2, 2)))
```

- [ ] **Step 2: Run the new padding test**

Run from `projects/recsys-candidate-generation`:

```bash
uv run pytest src/tests/test_models/test_ultragcn.py::test_build_ultragcn_constraint_weights_pads_item_neighbors_when_top_k_exceeds_items -v
```

Expected: PASS.

- [ ] **Step 3: Add test for `item_constraint_weight=0` early return**

Append this test near the existing `UltraGCNModule` tests:

```python
def test_ultragcn_module_item_constraint_loss_returns_zero_when_disabled(
    ultragcn_constraint_weights: UltraGCNConstraintWeights,
) -> None:
    """Skips item-neighbor lookup when the item-item constraint is disabled."""
    optimizer = cast(
        Any,
        SimpleNamespace(
            configure_optimizers=lambda _params: {"optimizer": object()},
            lr_scheduler_step=lambda *_args: None,
        ),
    )
    module = UltraGCNModule(
        num_users=20,
        num_items=30,
        out_dim=8,
        constraint_weights=ultragcn_constraint_weights,
        negative_weight=1.0,
        item_constraint_weight=0.0,
        l2_weight=1e-4,
        eval_top_k=3,
        optimizer=optimizer,
    )

    loss = module._item_constraint_loss(torch.tensor([10_000], dtype=torch.long))

    assert loss.ndim == 0
    assert loss.item() == 0.0
```

- [ ] **Step 4: Run the new early-return test**

Run from `projects/recsys-candidate-generation`:

```bash
uv run pytest src/tests/test_models/test_ultragcn.py::test_ultragcn_module_item_constraint_loss_returns_zero_when_disabled -v
```

Expected: PASS. The deliberately out-of-range item id confirms the method returns before indexing constraint tensors.

### Task 3: Correct the UltraGCN README example

**Files:**
- Modify: `projects/recsys-candidate-generation/README.md:152`

- [ ] **Step 1: Remove the ignored loss override from the UltraGCN example**

In `projects/recsys-candidate-generation/README.md`, replace:

```md
uv run python src/fit.py model=UltraGCN loss=bpr
```

with:

```md
uv run python src/fit.py model=UltraGCN
```

- [ ] **Step 2: Verify the surrounding explanation remains accurate**

Confirm the paragraph immediately after the examples still says UltraGCN precomputes graph degree and item-item constraints and learns embeddings. No additional README change is needed unless that paragraph has drifted.

### Task 4: Run package verification

**Files:**
- Verify package: `projects/recsys-candidate-generation`

- [ ] **Step 1: Run formatter, lint, and tests**

Run from `projects/recsys-candidate-generation`:

```bash
make fmt && make lint && make test
```

Expected: all commands exit 0.

- [ ] **Step 2: If full output is noisy, also run the focused UltraGCN/factory tests**

Run from `projects/recsys-candidate-generation`:

```bash
uv run pytest src/tests/test_models/test_ultragcn.py src/tests/test_models/test_factory.py -v
```

Expected: all selected tests PASS.

## Acceptance Criteria

- UltraGCN factory no longer builds `edge_index` through `train_df.select(...).to_numpy().T` and instead stacks `Series.to_torch()` tensors.
- UltraGCN README example does not include `loss=bpr`.
- `item_constraint_weight=0` early-return behavior is covered by a focused unit test.
- `num_items < item_constraint_top_k` padding behavior is covered by a focused unit test.
- `make fmt && make lint && make test` passes from `projects/recsys-candidate-generation`.

## Non-Goals

- No sparse or streaming rewrite for item-item co-occurrence construction in this PR.
- No vectorization of the nested co-occurrence loops in this PR.
- No caching/device-state refactor for UltraGCN constraints in this PR.
- No changes to `libs/ml_sandbox_libs`.
- No dependency additions.

## Implementation Log
<!-- Implementer appends one line per attempt: [YYYY-MM-DD] attempt #N -> STATUS | commit-or-failure-signature -->

## Review Findings
<!-- This template is also defined in commands/plan-v2.md. Keep them in sync on every edit. -->

### Reviewer Raw Findings
<!-- Planner V2 copies @reviewer_v2's structured findings verbatim here when invoking @reviewer_v2 during a workflow. Direct /review-*-v2 calls do not write here. Raw findings are review input, not implementation instructions. -->

#### 2026-05-15 CODE_REVIEW -> REQUEST_CHANGES
Critical issues:
- F3 (MAJOR): `torch.as_tensor(train_df.select(["user_index", "item_index"]).to_numpy().T)` may trigger a non-writable NumPy array warning; replace with `torch.stack([train_df["user_index"].to_torch().to(torch.long), train_df["item_index"].to_torch().to(torch.long)], dim=0)`.
Non-blocking suggestions:
- F1 (MAJOR): `cooccurrence = torch.zeros((num_items, num_items))` allocates a dense O(N²) matrix; track as known limitation for large item vocabularies.
- F2 (MAJOR): Nested Python loops dominate one-time co-occurrence construction time; vectorize later if needed.
- F4 (MINOR): `self.constraint_weights.to(device)` is called twice per batch; consider caching device-moved constraints.
- F5 (MINOR): README UltraGCN example includes misleading `loss=bpr`, which UltraGCN ignores.
- F6 (MINOR): `_item_constraint_loss` early return for `item_constraint_weight=0` is untested.
- F7 (MINOR): `num_items < top_k` item-neighbor padding path is untested.

### Planner V2 Adjudication
<!-- Planner V2 appends adjudication tables for v2 workflow reviews. Only ACCEPT rows are implementation instructions: | ID | Severity | Decision | Reason | Action | -->

| ID | Severity | Decision | Reason | Action |
|----|----------|----------|--------|--------|
| F3 | MAJOR | ACCEPT | Code uses the warning-prone NumPy conversion path; the direct `Series.to_torch()` stack preserves shape/dtype/order and avoids shared NumPy memory risk. | Replace edge construction in `src/models/factory.py`; strengthen factory test around `edge_index`. |
| F1 | MAJOR | DEFER | Dense O(N²) allocation exists, but reviewer states it is not blocking for current dataset scope; a sparse/streaming design is larger than this PR. | Track as known limitation/follow-up. |
| F2 | MAJOR | DEFER | Nested loops exist, but this is one-time preprocessing and vectorization is a broader performance refactor. | Track with F1 as follow-up constraint preprocessing optimization. |
| F4 | MINOR | DEFER | Duplicate `.to(device)` calls exist, but fixing cleanly introduces cached device state or helper-signature churn without current correctness impact. | Track as follow-up if profiling shows batch overhead. |
| F5 | MINOR | ACCEPT | README example is misleading because `create_ultragcn_module` does not call `loss.factory` and `UltraGCNModule` owns its loss. | Remove `loss=bpr` from the UltraGCN command. |
| F6 | MINOR | ACCEPT | Early return exists and is a simple public regression path for disabled item constraints. | Add focused `_item_constraint_loss` zero-weight test. |
| F7 | MINOR | ACCEPT | Padding branch exists and is a small edge case with user-visible config behavior. | Add focused `num_items < item_constraint_top_k` padding test. |

## Deviations from Plan
<!-- Implementer documents intentional deviations and reasons. -->

## Open Questions
<!-- Any agent adds questions for planner_v2 or oracle_v2. -->

- Follow-up: decide whether UltraGCN constraint preprocessing needs sparse/streaming co-occurrence construction and vectorization for item vocabularies beyond the current dataset scope.
- Follow-up: decide whether profiling justifies caching device-moved `UltraGCNConstraintWeights` across training batches.
