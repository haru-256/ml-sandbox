# Candidate Generation Factory Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `recsys-candidate-generation` の factory 周辺と factory tests を局所的に整理し、UltraGCN 追加後の見通しを改善する。

**Architecture:** 過度な抽象化は避ける。`training_step` / `validation_step` の共通化、`BaseModule` への移動、`GraphStepMixin` の復活、UltraGCN constraint logic の shared libs 移動は行わない。変更は project-local な `models/factory.py`、`config/validation.py`、`test_factory.py`、必要最小限の config コメントに限定する。

**Tech Stack:** Python 3.12, PyTorch, Hydra/OmegaConf, Polars, pytest, `uv`/`make` package workflow.

---

## Scope and Constraints

- Implement:
  - R1: `models/factory.py` の `match/case` dispatch を seq-rec / graph creator map に整理する。
  - R2: `create_ultragcn_module()` 内の train edge extraction を private helper に切り出す。
  - R3: `test_model_creators_use_pad_idx_from_datamodule` を seq-rec / LightGCN / UltraGCN の焦点別テストに分割する。
  - R4: `validation.py` の scalar validation error message を `cfg.model.name` ベースにする。
  - R5: `ultragcn.yaml` の `num_neighbors` に、UltraGCN 本体では message passing に使わないが graph DataModule の batch sampling 設定として残している旨のコメントを追加する。
- Do not implement:
  - `training_step` / `validation_step` の共通化。
  - `GraphStepMixin` の復活。
  - `BaseModule` や `libs/ml_sandbox_libs` への移動。
  - `LightGCN` / `UltraGCN` の `_compute_step_outputs` 統合。
  - UltraGCN constraint builder の shared libs 移動。
- Work from package root `projects/recsys-candidate-generation/` for commands.
- Preserve current behavior; expected final package test count is 66.

## File Structure

- Modify `projects/recsys-candidate-generation/src/models/factory.py`
  - Add private `_extract_ultragcn_train_edge_index()` helper.
  - Add creator maps for seq-rec and graph models.
  - Rewrite `create_model_module()` to use maps instead of the long `match/case` chain.
- Modify `projects/recsys-candidate-generation/src/config/validation.py`
  - Make scalar validation helpers use `cfg.model.name` in error messages.
- Modify `projects/recsys-candidate-generation/src/config/model/ultragcn.yaml`
  - Add a short comment explaining `num_neighbors` role for the graph DataModule.
- Modify `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py`
  - Split the large creator kwargs test into focused tests.
  - Keep dispatch tests and type-error tests passing.

## Implementation Tasks

### Task 1: Make UltraGCN train-edge extraction explicit

**Files:**
- Modify: `projects/recsys-candidate-generation/src/models/factory.py`
- Test: `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py`

- [ ] **Step 1: Add a private helper above `create_ultragcn_module()`**

In `src/models/factory.py`, add this helper immediately above `create_ultragcn_module()`:

```python
def _extract_ultragcn_train_edge_index(
    datamodule: AmazonReviewsBipartiteGraphDataModule,
) -> torch.Tensor:
    """Extract train split user-item edges for UltraGCN constraints.

    Args:
        datamodule: Prepared bipartite graph datamodule whose `all_df` contains
            `split`, `user_index`, and `item_index` columns.

    Returns:
        Edge index tensor with shape `(2, E)` and dtype `torch.long`.
    """
    train_df = datamodule.all_df.filter(pl.col("split") == "train")
    return torch.stack(
        [
            train_df["user_index"].to_torch().to(torch.long),
            train_df["item_index"].to_torch().to(torch.long),
        ],
        dim=0,
    )
```

- [ ] **Step 2: Use the helper in `create_ultragcn_module()`**

Replace:

```python
    train_df = datamodule.all_df.filter(pl.col("split") == "train")
    edge_index = torch.stack(
        [
            train_df["user_index"].to_torch().to(torch.long),
            train_df["item_index"].to_torch().to(torch.long),
        ],
        dim=0,
    )
```

with:

```python
    edge_index = _extract_ultragcn_train_edge_index(datamodule)
```

- [ ] **Step 3: Run the focused UltraGCN factory test**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest src/tests/test_models/test_factory.py::test_create_ultragcn_module_builds_constraint_weights -v
```

Expected: PASS. This verifies the helper still filters train edges only and preserves edge-index ordering.

### Task 2: Make validation helper messages model-aware

**Files:**
- Modify: `projects/recsys-candidate-generation/src/config/validation.py`
- Test: `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py`

- [ ] **Step 1: Update `_require_positive_scalar()`**

Replace:

```python
def _require_positive_scalar(cfg: DictConfig, field_name: str) -> None:
    """Require `cfg.model.<field_name>` to be greater than zero."""
    value = cfg.model[field_name]
    if value <= 0:
        raise ValueError(f"UltraGCN requires positive {field_name}, got {value}.")
```

with:

```python
def _require_positive_scalar(cfg: DictConfig, field_name: str) -> None:
    """Require `cfg.model.<field_name>` to be greater than zero."""
    value = cfg.model[field_name]
    if value <= 0:
        raise ValueError(f"{cfg.model.name} requires positive {field_name}, got {value}.")
```

- [ ] **Step 2: Update `_require_non_negative_scalar()`**

Replace:

```python
def _require_non_negative_scalar(cfg: DictConfig, field_name: str) -> None:
    """Require `cfg.model.<field_name>` to be greater than or equal to zero."""
    value = cfg.model[field_name]
    if value < 0:
        raise ValueError(f"UltraGCN requires non-negative {field_name}, got {value}.")
```

with:

```python
def _require_non_negative_scalar(cfg: DictConfig, field_name: str) -> None:
    """Require `cfg.model.<field_name>` to be greater than or equal to zero."""
    value = cfg.model[field_name]
    if value < 0:
        raise ValueError(f"{cfg.model.name} requires non-negative {field_name}, got {value}.")
```

- [ ] **Step 3: Run factory tests**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest src/tests/test_models/test_factory.py -v
```

Expected: PASS. Existing tests should still match substrings such as `positive` and `non-negative` if they assert error messages.

### Task 3: Split creator kwargs tests by model family

**Files:**
- Modify: `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py`

- [ ] **Step 1: Replace `test_model_creators_use_pad_idx_from_datamodule` with a seq-rec focused test**

Delete the current parametrized `test_model_creators_use_pad_idx_from_datamodule` function and replace it with this seq-rec test:

```python
@pytest.mark.parametrize(
    ("creator_name", "module_name", "loss_factory_name"),
    [
        ("create_two_tower_module", "TwoTowerModule", "create_score_loss"),
        ("create_sasrec_module", "SASRecModule", "create_score_loss"),
        ("create_gsasrec_module", "gSASRecModule", "create_score_loss"),
        ("create_simplex_module", "SimpleXModule", "create_embedding_loss"),
    ],
)
def test_seqrec_model_creators_use_pad_idx_from_datamodule(
    monkeypatch: pytest.MonkeyPatch,
    creator_name: str,
    module_name: str,
    loss_factory_name: str,
) -> None:
    """Seq-rec creators use the datamodule padding index and configured loss."""
    cfg = OmegaConf.create(
        {
            "data": {
                "neg_sample_size": 3,
                "max_seq_len": 20,
                "eval_top_k": 10,
            },
            "device": {"float16": False},
            "loss": {"name": "ccl"},
            "model": {
                "out_dim": 16,
                "user_id_dim": 8,
                "item_id_dim": 12,
                "hidden_dims": [32, 16],
                "normalization": "layer",
                "activation": "relu",
                "dropout": 0.1,
                "num_heads": 2,
                "num_blocks": 2,
                "attn_dropout": 0.1,
                "ffn_dropout": 0.1,
                "user_id_weight": 0.5,
                "user_history_pooling": "mean",
            },
        }
    )
    optimizer = cast(Any, SimpleNamespace())
    datamodule = cast(
        Any,
        SimpleNamespace(
            user2index={"u": 0},
            item2index={"i": 0},
            num_users=1,
            num_items=1,
            item_pad_idx=17,
        ),
    )
    sentinel_loss = object()
    captured_kwargs: dict[str, Any] = {}

    def fake_loss_factory(*_args: object, **_kwargs: object) -> object:
        return sentinel_loss

    def fake_module(**kwargs: Any) -> _DummyModule:
        module = _DummyModule(**kwargs)
        captured_kwargs.update(module.kwargs)
        return module

    monkeypatch.setattr(factory, loss_factory_name, fake_loss_factory)
    monkeypatch.setattr(factory, module_name, fake_module)

    creator = getattr(factory, creator_name)
    creator(cfg, datamodule, optimizer)

    assert captured_kwargs["pad_idx"] == datamodule.item_pad_idx
    assert captured_kwargs["loss_fn"] is sentinel_loss
```

- [ ] **Step 2: Add a LightGCN focused kwargs test**

Add below the seq-rec test:

```python
def test_lightgcn_creator_uses_graph_datamodule_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LightGCN creator uses graph datamodule sizes and embedding loss."""
    cfg = OmegaConf.create(
        {
            "data": {"eval_top_k": 10},
            "loss": {"name": "ccl"},
            "model": {
                "name": "LightGCN",
                "out_dim": 16,
                "num_layers": 2,
                "num_neighbors": [9, 4],
            },
        }
    )
    datamodule = cast(Any, SimpleNamespace(num_users=1, num_items=1))
    optimizer = cast(Any, SimpleNamespace())
    sentinel_loss = object()
    captured_kwargs: dict[str, Any] = {}

    def fake_create_embedding_loss(cfg_arg: object) -> object:
        assert cfg_arg is cfg
        return sentinel_loss

    def fake_module(**kwargs: Any) -> _DummyModule:
        module = _DummyModule(**kwargs)
        captured_kwargs.update(module.kwargs)
        return module

    monkeypatch.setattr(factory, "create_embedding_loss", fake_create_embedding_loss)
    monkeypatch.setattr(factory, "LightGCNModule", fake_module)

    factory.create_lightgcn_module(cfg, datamodule, optimizer)

    assert "pad_idx" not in captured_kwargs
    assert captured_kwargs["loss_fn"] is sentinel_loss
    assert captured_kwargs["num_users"] == datamodule.num_users
    assert captured_kwargs["num_items"] == datamodule.num_items
    assert captured_kwargs["out_dim"] == cfg.model.out_dim
    assert captured_kwargs["num_layers"] == cfg.model.num_layers
    assert captured_kwargs["eval_top_k"] == cfg.data.eval_top_k
    assert captured_kwargs["optimizer"] is optimizer
```

- [ ] **Step 3: Add an UltraGCN focused kwargs test**

Add below the LightGCN test:

```python
def test_ultragcn_creator_uses_graph_datamodule_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UltraGCN creator uses graph datamodule sizes and internal loss settings."""
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
            num_users=1,
            num_items=1,
            all_df=__import__("polars").DataFrame(
                {"split": ["train"], "user_index": [0], "item_index": [0]}
            ),
        ),
    )
    optimizer = cast(Any, SimpleNamespace())
    captured_kwargs: dict[str, Any] = {}

    def fake_module(**kwargs: Any) -> _DummyModule:
        module = _DummyModule(**kwargs)
        captured_kwargs.update(module.kwargs)
        return module

    monkeypatch.setattr(factory, "UltraGCNModule", fake_module)

    factory.create_ultragcn_module(cfg, datamodule, optimizer)

    assert "pad_idx" not in captured_kwargs
    assert "loss_fn" not in captured_kwargs
    assert captured_kwargs["num_users"] == datamodule.num_users
    assert captured_kwargs["num_items"] == datamodule.num_items
    assert captured_kwargs["out_dim"] == cfg.model.out_dim
    assert captured_kwargs["negative_weight"] == cfg.model.negative_weight
    assert captured_kwargs["item_constraint_weight"] == cfg.model.item_constraint_weight
    assert captured_kwargs["l2_weight"] == cfg.model.l2_weight
    assert captured_kwargs["eval_top_k"] == cfg.data.eval_top_k
```

- [ ] **Step 4: Remove redundant old BPR test if duplicated**

After adding `test_lightgcn_creator_uses_graph_datamodule_kwargs`, inspect `test_create_lightgcn_module_uses_embedding_loss_factory_with_bpr`. If it is now fully duplicated except for `loss.name == "bpr"`, keep it only if it still verifies a unique behavior. If kept, trim duplicated assertions to the unique checks:

```python
    assert captured_kwargs["loss_fn"] is sentinel_loss
    assert captured_kwargs["optimizer"] is optimizer
```

Do not fold it back into the large seq-rec test.

- [ ] **Step 5: Run factory tests**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run pytest src/tests/test_models/test_factory.py -v
```

Expected: PASS.

### Task 4: Add `num_neighbors` comment for UltraGCN config

**Files:**
- Modify: `projects/recsys-candidate-generation/src/config/model/ultragcn.yaml`

- [ ] **Step 1: Add a clarifying comment**

Replace:

```yaml
num_neighbors: [10, 5]
```

with:

```yaml
# Used by the shared bipartite graph DataModule / LinkNeighborLoader for batch sampling.
# UltraGCN itself does not run message passing; graph constraints are precomputed from train edges.
num_neighbors: [10, 5]
```

- [ ] **Step 2: Verify Hydra composition for UltraGCN**

Run from `projects/recsys-candidate-generation/`:

```bash
uv run python src/fit.py --cfg job model=UltraGCN
```

Expected: command exits 0 and prints the composed config including `model.name: UltraGCN`.

### Task 5: Final verification

**Files:**
- Verify all touched files.

- [ ] **Step 1: Run package verification**

Run from `projects/recsys-candidate-generation/`:

```bash
make fmt && make lint && make test
```

Expected: all commands pass. Final package test count should remain 66 unless duplicate test trimming intentionally changes the count.

## Acceptance Criteria

- `create_model_module()` no longer contains a long 6-case `match` chain.
- UltraGCN train edge extraction is in a private helper, not inline in `create_ultragcn_module()`.
- `test_factory.py` no longer has one giant creator kwargs test with seq-rec / LightGCN / UltraGCN assertion branches.
- UltraGCN scalar validation messages use `cfg.model.name`, not a hard-coded string in helper functions.
- `ultragcn.yaml` explains why `num_neighbors` remains present.
- No `GraphStepMixin`, no BaseModule/shared-libs step abstraction, no constraint logic movement to `libs`.
- `make fmt && make lint && make test` passes from `projects/recsys-candidate-generation/`.

## Non-Goals

- Do not optimize UltraGCN neighbor sampling or create a separate UltraGCN datamodule.
- Do not change training/validation step implementations.
- Do not move UltraGCN constraint preprocessing outside `models/ultragcn.py`.
- Do not modify `libs/ml_sandbox_libs`.

## Implementation Log
<!-- Implementer appends one line per attempt: [YYYY-MM-DD] attempt #N -> STATUS | commit-or-failure-signature -->
[2026-05-16] attempt #1 -> DONE | all tasks completed, 66 tests pass

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
