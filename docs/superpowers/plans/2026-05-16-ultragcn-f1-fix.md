# UltraGCN F1 Sparse Constraint Weights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dense `num_items x num_items` UltraGCN item co-occurrence allocation while preserving the existing item-neighbor tensor contract.

**Architecture:** Keep the fix local to `projects/recsys-candidate-generation/src/models/ultragcn.py`. Build item-item co-occurrence counts as sparse COO edges, coalesce duplicate `(src_item, dst_item)` pairs, normalize sparse values, and materialize only the required dense outputs `item_neighbor_indices` and `item_neighbor_weights` with shape `(num_items, item_constraint_top_k)`.

**Tech Stack:** Python 3.12, PyTorch CPU tensors, pytest, `uv`/`make` package workflow. No new dependencies.

---

## Approach

Use a sparse COO co-occurrence pipeline with per-row top-k extraction:

1. Keep degree and link-weight computation unchanged.
2. Deduplicate `(user, item)` interactions for co-occurrence only, preserving raw edge counts for `item_degree`.
3. For each user's unique item set, generate directed item pairs using `repeat_interleave`/`repeat` instead of nested Python item loops.
4. Flush generated pairs through `torch.sparse_coo_tensor(...).coalesce()` in chunks so duplicate pairs are summed without allocating an `N²` matrix.
5. Normalize sparse co-occurrence values by item degrees.
6. Loop over sparse rows to write top-k neighbors into `(num_items, K)` output tensors initialized to self-neighbor/zero-weight defaults.

This changes memory from `O(num_items²)` to approximately `O(num_nonzero_item_pairs + num_items * K)` and preserves the module's current public output contract.

## Trade-offs Evaluated

| Option | Decision | Reason |
| --- | --- | --- |
| A. Sparse COO/CSR with batched pair generation and coalesce | **Recommended** | Avoids dense `N²` allocation, uses built-in duplicate summation via `coalesce()`, needs no dependencies, and remains readable. Sparse top-k still needs a row loop, but only over non-empty sparse rows. |
| B. Fully vectorized pair generation for all users at once | Rejected | Fast for small data but can create a very large all-pairs tensor for high-degree users. Chunked sparse flushing is safer. |
| C. Incremental top-k per item using Python min-heaps | Rejected | Bounded memory, but significantly more Python bookkeeping, harder to validate, and slower for one-time construction unless carefully optimized. |
| D. PyG utility | Rejected | `torch_geometric.data.Data.coalesce()`/utility coalescing helps deduplicate edge indices, but it does not provide the complete user-grouped co-occurrence + normalized top-k pipeline. Adding `torch_sparse`-specific code would increase coupling. |
| E. Per-item batch processing against all other items | Rejected | Avoids full dense memory but tends toward repeated scans or dense per-batch slices; more complex and likely slower than sparse pair accumulation. |

## File Structure

- Modify `projects/recsys-candidate-generation/src/models/ultragcn.py`
  - Add private helper functions for sparse item co-occurrence and sparse top-k neighbor extraction.
  - Replace dense `cooccurrence`, dense `denom`, and dense `torch.topk(item_weights, dim=1)` logic inside `build_ultragcn_constraint_weights`.
- Modify `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py`
  - Add a regression test that fails if the builder attempts to allocate a dense `(num_items, num_items)` zero tensor.
  - Add a behavior test that verifies sparse co-occurrence preserves duplicate-edge semantics and normalized top-k weights.

## Implementation Tasks

### Task 1: Add regression tests for sparse item constraints

**Files:**
- Modify: `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py`

- [ ] **Step 1: Add a helper for neighbor weight assertions**

Insert this helper after the imports in `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py`:

```python
def _positive_neighbor_weights(weights: UltraGCNConstraintWeights, item_id: int) -> dict[int, float]:
    """Return positive item-neighbor weights keyed by neighbor id."""
    result: dict[int, float] = {}
    for neighbor_id, weight in zip(
        weights.item_neighbor_indices[item_id].tolist(),
        weights.item_neighbor_weights[item_id].tolist(),
        strict=True,
    ):
        if weight > 0:
            result[int(neighbor_id)] = float(weight)
    return result
```

- [ ] **Step 2: Add a test that blocks dense item-item allocation**

Append this test near the existing `build_ultragcn_constraint_weights` tests:

```python
def test_build_ultragcn_constraint_weights_does_not_allocate_dense_item_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoids allocating a dense num_items x num_items co-occurrence matrix."""
    num_items = 128
    original_zeros = torch.zeros

    def guarded_zeros(*args: Any, **kwargs: Any) -> torch.Tensor:
        shape = args[0] if args else kwargs.get("size")
        if tuple(shape) == (num_items, num_items):
            raise AssertionError("dense item-item allocation is not allowed")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 1, 2, 2],
            [0, 1, 2, 1, 3, 2, 4],
        ],
        dtype=torch.long,
    )

    weights = build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=3,
        num_items=num_items,
        constraint_weight=1.0,
        item_constraint_top_k=3,
    )

    assert weights.item_neighbor_indices.shape == (num_items, 3)
    assert weights.item_neighbor_weights.shape == (num_items, 3)
```

- [ ] **Step 3: Add a test for duplicate-edge semantics and normalized sparse weights**

Append this test near `test_build_ultragcn_constraint_weights_keeps_top_item_neighbors`:

```python
def test_build_ultragcn_constraint_weights_uses_unique_items_per_user_for_sparse_cooccurrence() -> None:
    """Counts each user's unique item pair once while preserving raw item degrees."""
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 2, 2],
            [0, 1, 1, 0, 1, 2, 0, 2],
        ],
        dtype=torch.long,
    )

    weights = build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=3,
        num_items=3,
        constraint_weight=1.0,
        item_constraint_top_k=2,
    )

    assert torch.equal(weights.item_degree, torch.tensor([3.0, 3.0, 2.0]))
    assert _positive_neighbor_weights(weights, item_id=0) == pytest.approx(
        {
            1: 2.0 / torch.sqrt(torch.tensor(4.0 * 4.0)).item(),
            2: 2.0 / torch.sqrt(torch.tensor(4.0 * 3.0)).item(),
        }
    )
    assert _positive_neighbor_weights(weights, item_id=1) == pytest.approx(
        {
            0: 2.0 / torch.sqrt(torch.tensor(4.0 * 4.0)).item(),
            2: 1.0 / torch.sqrt(torch.tensor(4.0 * 3.0)).item(),
        }
    )
    assert _positive_neighbor_weights(weights, item_id=2) == pytest.approx(
        {
            0: 2.0 / torch.sqrt(torch.tensor(3.0 * 4.0)).item(),
            1: 1.0 / torch.sqrt(torch.tensor(3.0 * 4.0)).item(),
        }
    )
```

- [ ] **Step 4: Run the new tests and verify they fail on the current implementation**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest src/tests/test_models/test_ultragcn.py::test_build_ultragcn_constraint_weights_does_not_allocate_dense_item_matrix src/tests/test_models/test_ultragcn.py::test_build_ultragcn_constraint_weights_uses_unique_items_per_user_for_sparse_cooccurrence -v
```

Expected before implementation: the dense-allocation test fails with `AssertionError: dense item-item allocation is not allowed`.

### Task 2: Replace dense co-occurrence with sparse coalesced pairs

**Files:**
- Modify: `projects/recsys-candidate-generation/src/models/ultragcn.py`

- [ ] **Step 1: Add private sparse helper functions**

Insert these helpers above `build_ultragcn_constraint_weights`:

```python
def _unique_user_item_lists(
    user_indices: torch.Tensor,
    item_indices: torch.Tensor,
    num_users: int,
    num_items: int,
) -> list[torch.Tensor]:
    """Group unique valid item ids by user for co-occurrence counting.

    Args:
        user_indices: Raw user ids from train edges. Shape: ``(E,)``.
        item_indices: Raw item ids from train edges. Shape: ``(E,)``.
        num_users: Number of valid user ids.
        num_items: Number of valid item ids.

    Returns:
        A list of 1D item-id tensors, one for each user that has at least one valid item.
    """
    valid_edges = (
        (user_indices >= 0)
        & (user_indices < num_users)
        & (item_indices >= 0)
        & (item_indices < num_items)
    )
    if not valid_edges.any():
        return []

    valid_users = user_indices[valid_edges]
    valid_items = item_indices[valid_edges]
    unique_keys = torch.unique(valid_users * num_items + valid_items, sorted=True)
    unique_users = torch.div(unique_keys, num_items, rounding_mode="floor")
    unique_items = unique_keys.remainder(num_items)
    _, per_user_counts = torch.unique_consecutive(unique_users, return_counts=True)
    return list(torch.split(unique_items, per_user_counts.tolist()))


def _coalesce_item_pairs(
    pair_chunks: list[torch.Tensor],
    num_items: int,
) -> torch.Tensor:
    """Coalesce directed item-pair chunks into sparse COO co-occurrence counts.

    Args:
        pair_chunks: List of ``(2, P)`` directed item-pair tensors.
        num_items: Number of item ids in the sparse square matrix.

    Returns:
        Coalesced sparse COO tensor with shape ``(num_items, num_items)``.
    """
    if not pair_chunks:
        empty_indices = torch.empty((2, 0), dtype=torch.long)
        empty_values = torch.empty((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(
            empty_indices,
            empty_values,
            (num_items, num_items),
            dtype=torch.float32,
        ).coalesce()

    pair_index = torch.cat(pair_chunks, dim=1)
    pair_values = torch.ones(pair_index.size(1), dtype=torch.float32)
    return torch.sparse_coo_tensor(
        pair_index,
        pair_values,
        (num_items, num_items),
        dtype=torch.float32,
    ).coalesce()


def _build_sparse_item_cooccurrence(
    user_indices: torch.Tensor,
    item_indices: torch.Tensor,
    num_users: int,
    num_items: int,
    max_pairs_per_chunk: int = 1_000_000,
) -> torch.Tensor:
    """Build sparse directed item co-occurrence counts without dense item-item memory.

    Args:
        user_indices: Raw user ids from train edges. Shape: ``(E,)``.
        item_indices: Raw item ids from train edges. Shape: ``(E,)``.
        num_users: Number of valid user ids.
        num_items: Number of valid item ids.
        max_pairs_per_chunk: Maximum generated item pairs before coalescing a chunk.

    Returns:
        Coalesced sparse COO tensor with directed co-occurrence counts.
    """
    coalesced_chunks: list[torch.Tensor] = []
    pending_pairs: list[torch.Tensor] = []
    pending_pair_count = 0

    for interacted_items in _unique_user_item_lists(
        user_indices=user_indices,
        item_indices=item_indices,
        num_users=num_users,
        num_items=num_items,
    ):
        item_count = interacted_items.numel()
        if item_count < 2:
            continue
        src_items = interacted_items.repeat_interleave(item_count)
        dst_items = interacted_items.repeat(item_count)
        non_self = src_items != dst_items
        pair_index = torch.stack([src_items[non_self], dst_items[non_self]], dim=0)
        pending_pairs.append(pair_index)
        pending_pair_count += pair_index.size(1)

        if pending_pair_count >= max_pairs_per_chunk:
            coalesced_chunks.append(_coalesce_item_pairs(pending_pairs, num_items))
            pending_pairs = []
            pending_pair_count = 0

    if pending_pairs:
        coalesced_chunks.append(_coalesce_item_pairs(pending_pairs, num_items))

    if not coalesced_chunks:
        return _coalesce_item_pairs([], num_items)
    if len(coalesced_chunks) == 1:
        return coalesced_chunks[0]

    merged_indices = torch.cat([chunk.indices() for chunk in coalesced_chunks], dim=1)
    merged_values = torch.cat([chunk.values() for chunk in coalesced_chunks])
    return torch.sparse_coo_tensor(
        merged_indices,
        merged_values,
        (num_items, num_items),
        dtype=torch.float32,
    ).coalesce()


def _sparse_topk_item_neighbors(
    cooccurrence: torch.Tensor,
    item_degree: torch.Tensor,
    item_constraint_top_k: int,
    num_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert sparse co-occurrence counts to fixed-width top-k neighbor tensors.

    Args:
        cooccurrence: Coalesced sparse COO item-item co-occurrence counts.
        item_degree: Raw item degree tensor. Shape: ``(num_items,)``.
        item_constraint_top_k: Number of neighbors to output per item.
        num_items: Number of item ids.

    Returns:
        Tuple of ``(item_neighbor_indices, item_neighbor_weights)`` with shapes
        ``(num_items, item_constraint_top_k)``.
    """
    top_indices = torch.arange(num_items, dtype=torch.long).unsqueeze(1).repeat(
        1,
        item_constraint_top_k,
    )
    top_weights = torch.zeros((num_items, item_constraint_top_k), dtype=torch.float32)

    cooccurrence = cooccurrence.coalesce()
    if cooccurrence._nnz() == 0:
        return top_indices, top_weights

    rows, cols = cooccurrence.indices()
    denom = torch.sqrt((item_degree[rows] + 1.0) * (item_degree[cols] + 1.0)).clamp_min(1.0)
    weights = cooccurrence.values() / denom
    unique_rows, row_counts = torch.unique_consecutive(rows, return_counts=True)

    offset = 0
    for row, row_count in zip(unique_rows.tolist(), row_counts.tolist(), strict=True):
        row_slice = slice(offset, offset + row_count)
        row_weights = weights[row_slice]
        row_cols = cols[row_slice]
        selected_count = min(item_constraint_top_k, row_weights.numel())
        selected_weights, selected_positions = torch.topk(row_weights, k=selected_count)
        top_indices[row, :selected_count] = row_cols[selected_positions]
        top_weights[row, :selected_count] = selected_weights
        offset += row_count

    return top_indices, top_weights
```

- [ ] **Step 2: Replace dense logic in `build_ultragcn_constraint_weights`**

In `projects/recsys-candidate-generation/src/models/ultragcn.py`, replace lines 109-139 with:

```python
    cooccurrence = _build_sparse_item_cooccurrence(
        user_indices=user_indices,
        item_indices=item_indices,
        num_users=num_users,
        num_items=num_items,
    )
    top_indices, top_weights = _sparse_topk_item_neighbors(
        cooccurrence=cooccurrence,
        item_degree=item_degree,
        item_constraint_top_k=item_constraint_top_k,
        num_items=num_items,
    )
```

The return block remains unchanged:

```python
    return UltraGCNConstraintWeights(
        user_degree=user_degree,
        item_degree=item_degree,
        user_indices=user_indices,
        item_indices=item_indices,
        constraint_weight=constraint_weight,
        link_weights=link_weights.to(torch.float32),
        item_neighbor_indices=top_indices.to(torch.long),
        item_neighbor_weights=top_weights.to(torch.float32),
    )
```

- [ ] **Step 3: Run focused tests**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest src/tests/test_models/test_ultragcn.py -v
```

Expected: all UltraGCN tests pass.

### Task 3: Format, lint, and run package verification

**Files:**
- Verify package only; no additional file changes expected unless format/lint requires local adjustments.

- [ ] **Step 1: Run formatter**

Run from `projects/recsys-candidate-generation/`:

```bash
make fmt
```

Expected: command exits 0. If formatting changes files, inspect that only intended local files changed.

- [ ] **Step 2: Run lint**

Run from `projects/recsys-candidate-generation/`:

```bash
make lint
```

Expected: command exits 0. Fix only issues caused by this UltraGCN change.

- [ ] **Step 3: Run tests**

Run from `projects/recsys-candidate-generation/`:

```bash
make test
```

Expected: command exits 0.

- [ ] **Step 4: Run the required focused test command**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest src/tests/test_models/test_ultragcn.py -v
```

Expected: command exits 0.

- [ ] **Step 5: Report verification evidence**

Report the exact commands and pass/fail outcomes. Do not commit unless the user explicitly requests a commit.

## Acceptance Criteria

- `build_ultragcn_constraint_weights` no longer allocates a dense `(num_items, num_items)` tensor.
- The function still returns `item_neighbor_indices` and `item_neighbor_weights` with shape `(num_items, item_constraint_top_k)`.
- Item co-occurrence semantics remain: duplicate interactions by the same user count once for co-occurrence; raw edges still determine `item_degree`.
- Isolated items keep self-neighbor indices and zero weights, matching current behavior.
- No new dependencies and no changes outside `projects/recsys-candidate-generation` implementation/tests, except this plan document.
- `make fmt && make lint && make test` passes from `projects/recsys-candidate-generation/`.
- `uv run pytest src/tests/test_models/test_ultragcn.py -v` passes from `projects/recsys-candidate-generation/`.

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
