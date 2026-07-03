# Apply PR159 Fixes to Recsys Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the four PR #159 local Docker/W&B and training-policy fixes from `projects/recsys-candidate-generation` to `projects/recsys-ranking`.

**Architecture:** Keep `projects/recsys-ranking` self-contained and mirror only the candidate-generation patterns that are valid for ranking. Docker support should replicate the monorepo layout inside the image so editable local path dependencies resolve, while `vertex-job-runner` is excluded from the runtime Docker dependency set via a dependency group. Checkpointing should be an explicit Hydra contract, disabled by default and stored under `save_dir/checkpoints` only when enabled.

**Tech Stack:** Python 3.12, uv 0.11.6, Makefile targets, Docker/Compose, Lightning, Hydra, WandB, PyTorch/TorchVision.

## Global Constraints

- Do not commit, push, rebase, reset, or revert unless explicitly asked.
- Preserve existing user changes, including the current `projects/recsys-candidate-generation/Makefile` diff.
- Run Python commands from package roots and through `make` / `uv`; do not run direct `python`, `pip`, or `pytest`.
- Keep changes scoped to `projects/recsys-ranking` unless a lock/update command requires a package-local file update.
- Follow the candidate-generation policy: checkpointing defaults to off; if enabled, checkpoints are local under `save_dir/checkpoints`.
- Docker local training should require `WANDB_API_KEY` and should not reintroduce GCS env or mounted-volume plumbing.

---

## File Structure

- Modify `projects/recsys-ranking/pyproject.toml`: move `vertex-job-runner` from project dependencies to `dependency-groups.vertex`; keep the editable source entry; evaluate GPU index alignment with PR #159.
- Modify `projects/recsys-ranking/uv.lock`: regenerate from ranking package root after dependency changes.
- Create `projects/recsys-ranking/Dockerfile`: ranking-specific Docker build using the same monorepo layout pattern as candidate-generation.
- Create `projects/recsys-ranking/compose.yaml`: dev shell service with `additional_contexts`, required `WANDB_API_KEY`, default image name, and GPU reservation.
- Create `projects/recsys-ranking/.env.example`: local Docker env sample with `WANDB_API_KEY`, `IMAGE_URI`, and `EXPERIMENT_NAME`.
- Modify `projects/recsys-ranking/.gitignore`: ignore `.env` if not already ignored.
- Modify `projects/recsys-ranking/Makefile`: add `docker-build`, `docker-run`, `docker-up`, `docker-exec`, and `docker-down` targets.
- Modify `projects/recsys-ranking/src/config/config.yaml`: add required `enable_checkpointing: false`.
- Modify `projects/recsys-ranking/src/fit.py`: add opt-in `ModelCheckpoint`, `enable_checkpointing`, and `default_root_dir`.
- Modify `projects/recsys-ranking/src/tests/test_fit.py`: add checkpointing-on/off tests for ranking's `val_ndcg` monitor.
- Modify `projects/recsys-ranking/README.md`: document local Docker usage, checkpointing behavior, and dependency group behavior.

---

### Task 1: Dependency Policy and Lockfile

**Files:**
- Modify: `projects/recsys-ranking/pyproject.toml`
- Modify: `projects/recsys-ranking/uv.lock`

**Interfaces:**
- Consumes: existing `[tool.uv.sources].vertex-job-runner` local path source.
- Produces: a `vertex` dependency group available through `uv sync --group vertex` while normal Docker sync can use `--no-dev --extra=gpu` without installing `vertex-job-runner`.

- [x] **Step 1: Move `vertex-job-runner` out of project dependencies**

In `projects/recsys-ranking/pyproject.toml`, remove this line from `[project].dependencies`:

```toml
  "vertex-job-runner",
```

Add this dependency group:

```toml
[dependency-groups]
dev = ["notebook~=7.5.5", "ipywidgets>=8.1.5", "tqdm~=4.67.1"]
test = ["pytest~=9.0.3", "pytest-mock~=3.15.1", "deepdiff~=8.6.0"]
lint = ["ruff~=0.12.11", "mypy~=1.17.1"]
vertex = ["vertex-job-runner"]
```

Keep this source mapping unchanged:

```toml
vertex-job-runner = { path = "../../apps/vertex-job-runner", editable = true }
```

- [x] **Step 2: Decide GPU index alignment deliberately**

Ranking currently uses:

```toml
[[tool.uv.index]]
name = "pytorch-gpu"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

Candidate-generation PR #159 uses `cu128`, but ranking has no PyG extension wheels. Prefer a minimal first pass: keep ranking on `cu126` unless `uv lock` or Docker build proves a TorchVision mismatch. If changing to `cu128`, update `pyproject.toml`, regenerate `uv.lock`, and include that as an explicit dependency-alignment change in the final summary.

- [x] **Step 3: Regenerate lockfile**

Run from the ranking package root:

```sh
cd projects/recsys-ranking
make lock
```

Expected: `uv lock` exits 0 and `uv.lock` reflects `vertex-job-runner` under a `vertex` dependency group instead of normal package dependencies.

- [x] **Step 4: Inspect the dependency diff**

Run:

```sh
git diff -- projects/recsys-ranking/pyproject.toml projects/recsys-ranking/uv.lock
```

Expected: no unrelated package churn beyond the dependency group / lock resolution implied by this task.

---

### Task 2: Ranking Docker and WandB Local Runtime

**Files:**
- Create: `projects/recsys-ranking/Dockerfile`
- Create: `projects/recsys-ranking/compose.yaml`
- Create: `projects/recsys-ranking/.env.example`
- Modify: `projects/recsys-ranking/.gitignore`
- Modify: `projects/recsys-ranking/Makefile`

**Interfaces:**
- Consumes: `ml_sandbox_libs=../../libs/ml_sandbox_libs` as an additional Docker build context.
- Produces: `make docker-build`, `make docker-run`, `make docker-up`, `make docker-exec`, and `make docker-down` from `projects/recsys-ranking`.

- [x] **Step 1: Create ranking Dockerfile**

Create `projects/recsys-ranking/Dockerfile`:

```dockerfile
# Using multi-stage build to slim down the image
FROM --platform=linux/amd64 nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04@sha256:d02c4310b6d57ca0b16cd80298bdb33a74187baafe2eccd8a6a16180ddc90802

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.6@sha256:b1e699368d24c57cda93c338a57a8c5a119009ba809305cc8e86986d4a006754 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Replicate monorepo structure so pyproject.toml's relative path deps resolve.
# pyproject.toml references ../../libs/ml_sandbox_libs, which is outside the build
# context. It is provided via an additional build context.
# vertex-job-runner is in the 'vertex' dependency group and excluded from Docker builds.
WORKDIR /workspace/projects/recsys-ranking

COPY --from=ml_sandbox_libs . /workspace/libs/ml_sandbox_libs
COPY pyproject.toml uv.lock .python-version ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra=gpu

COPY ./src ./src

# Disable runtime uv sync to prevent uv from checking missing editable dependencies at runtime.
ENV UV_NO_SYNC=1

WORKDIR /workspace/projects/recsys-ranking/src
CMD ["uv", "run", "fit.py"]
```

- [x] **Step 2: Create compose config**

Create `projects/recsys-ranking/compose.yaml`:

```yaml
services:
  job:
    build:
      context: .
      dockerfile: Dockerfile
      additional_contexts:
        - ml_sandbox_libs=../../libs/ml_sandbox_libs
    command: ["tail", "-f", "/dev/null"]
    platform: linux/amd64
    image: ${IMAGE_URI:-recsys-ranking:latest}
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY is required}
      - EXPERIMENT_NAME=${EXPERIMENT_NAME:-local-experiment}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
```

- [x] **Step 3: Create env example and ignore local env**

Create `projects/recsys-ranking/.env.example`:

```dotenv
# Required: wandb API key for experiment tracking
# Get yours from: https://wandb.ai/authorize
WANDB_API_KEY=your-wandb-api-key-here

# Optional: Docker image name (used by compose.yaml)
IMAGE_URI=recsys-ranking:latest

# Optional: experiment name (used by Cloud Logging in Vertex AI, not in local Docker)
EXPERIMENT_NAME=local-experiment
```

Ensure `projects/recsys-ranking/.gitignore` contains:

```gitignore
results
.env
```

- [x] **Step 4: Add Makefile Docker targets**

Append to `projects/recsys-ranking/Makefile` after the `install` target:

```make
# Docker
DOCKER_IMAGE ?= recsys-ranking:latest
# Additional build context for monorepo local path dep (required by Dockerfile)
DOCKER_BUILD_CONTEXTS := ml_sandbox_libs=../../libs/ml_sandbox_libs

.PHONY: docker-build
docker-build: ## Build the Docker image for local training
	docker build --platform linux/amd64 --build-context $(DOCKER_BUILD_CONTEXTS) -t $(DOCKER_IMAGE) .

.PHONY: docker-run
docker-run: ## Run training directly in Docker (requires WANDB_API_KEY; Hydra args via DOCKER_ARGS)
	@test -n "$$WANDB_API_KEY" || (echo "ERROR: WANDB_API_KEY is not set" && exit 1)
	docker run --rm --gpus all --platform linux/amd64 \
		-e WANDB_API_KEY=$$WANDB_API_KEY \
		-e EXPERIMENT_NAME=$${EXPERIMENT_NAME:-local-experiment} \
		$(DOCKER_IMAGE) uv run fit.py $(DOCKER_ARGS)

.PHONY: docker-up
docker-up: ## Start a dev shell container (use 'make docker-exec' to run training inside)
	docker compose up -d

.PHONY: docker-exec
docker-exec: ## Run training inside the dev shell container
	docker compose exec job uv run fit.py

.PHONY: docker-down
docker-down: ## Stop the dev shell container
	docker compose down
```

- [x] **Step 5: Verify generated Docker commands without building**

Run from the ranking package root:

```sh
cd projects/recsys-ranking
make -n docker-build
WANDB_API_KEY=dummy docker compose config
```

Expected:
- `make -n docker-build` prints `docker build --platform linux/amd64 --build-context ml_sandbox_libs=../../libs/ml_sandbox_libs -t recsys-ranking:latest .`
- `docker compose config` exits 0.
- Compose output includes `WANDB_API_KEY: dummy`.
- Compose output does not include `GCS_PATH`, `MOUNTED_GCS_PATH`, or volume mounts.

---

### Task 3: Opt-In Checkpointing for Ranking

**Files:**
- Modify: `projects/recsys-ranking/src/config/config.yaml`
- Modify: `projects/recsys-ranking/src/fit.py`
- Modify: `projects/recsys-ranking/src/tests/test_fit.py`

**Interfaces:**
- Consumes: Hydra key `cfg.enable_checkpointing`.
- Produces: `fit.create_trainer(cfg, save_dir)` that disables checkpointing by default and adds `ModelCheckpoint(monitor="val_ndcg")` only when enabled.

- [x] **Step 1: Add the config contract**

In `projects/recsys-ranking/src/config/config.yaml`, add:

```yaml
enable_checkpointing: false
```

Place it next to `debug: true`, matching candidate-generation.

- [x] **Step 2: Add failing checkpoint tests**

Extend `projects/recsys-ranking/src/tests/test_fit.py` with:

```python
import pathlib
```

Add these tests:

```python
def test_create_trainer_checkpointing_disabled(mocker: MockerFixture) -> None:
    """Trainer does not create checkpoint callbacks unless explicitly enabled."""
    mocker.patch("fit.WandbLogger")

    cfg = OmegaConf.create(
        {
            "model": {"name": "test_model"},
            "device": {"accelerator": "cpu"},
            "debug": False,
            "log": {"log_every_n_steps": 10},
            "optimizer": {"gradient_clip_val": 1.0},
            "enable_checkpointing": False,
        }
    )
    save_dir = pathlib.Path("/tmp/test_save_dir")

    trainer = fit.create_trainer(cfg, save_dir)

    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callbacks = [
        c for c in cast(Any, trainer).callbacks if isinstance(c, ModelCheckpoint)
    ]
    assert len(checkpoint_callbacks) == 0


def test_create_trainer_checkpointing_enabled(mocker: MockerFixture) -> None:
    """Trainer creates a local checkpoint callback when explicitly enabled."""
    mocker.patch("fit.WandbLogger")

    cfg = OmegaConf.create(
        {
            "model": {"name": "test_model"},
            "device": {"accelerator": "cpu"},
            "debug": False,
            "log": {"log_every_n_steps": 10},
            "optimizer": {"gradient_clip_val": 1.0},
            "enable_checkpointing": True,
        }
    )
    save_dir = pathlib.Path("/tmp/test_save_dir")

    trainer = fit.create_trainer(cfg, save_dir)

    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callbacks = [
        c for c in cast(Any, trainer).callbacks if isinstance(c, ModelCheckpoint)
    ]
    assert len(checkpoint_callbacks) == 1

    checkpoint_cb = checkpoint_callbacks[0]
    assert (
        pathlib.Path(cast(Any, checkpoint_cb).dirpath).resolve()
        == (save_dir / "checkpoints").resolve()
    )
    assert checkpoint_cb.monitor == "val_ndcg"
    assert checkpoint_cb.mode == "max"
    assert pathlib.Path(cast(Any, trainer).default_root_dir).resolve() == save_dir.resolve()
```

- [x] **Step 3: Run the failing targeted tests**

Run:

```sh
cd projects/recsys-ranking
uv run pytest src/tests/test_fit.py -q
```

Expected before implementation: checkpoint tests fail because `ModelCheckpoint` is still created by Lightning default behavior and `default_root_dir` is not set to `save_dir`.

- [x] **Step 4: Implement checkpointing policy**

In `projects/recsys-ranking/src/fit.py`, change:

```python
from lightning.pytorch.callbacks import EarlyStopping
```

to:

```python
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
```

Replace the inline callbacks list in `create_trainer` with:

```python
    enable_checkpointing = cfg.enable_checkpointing
    callbacks: list[L.Callback] = [
        EarlyStopping(monitor="val_ndcg", mode="max", patience=3),
    ]
    if enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir / "checkpoints",
                monitor="val_ndcg",
                mode="max",
                save_top_k=1,
            )
        )
```

Pass these arguments to `L.Trainer`:

```python
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        default_root_dir=save_dir,
```

- [x] **Step 5: Run targeted tests**

Run:

```sh
cd projects/recsys-ranking
uv run pytest src/tests/test_fit.py -q
```

Expected: all tests in `src/tests/test_fit.py` pass.

---

### Task 4: README and Full Verification

**Files:**
- Modify: `projects/recsys-ranking/README.md`

**Interfaces:**
- Consumes: Makefile Docker targets, `.env.example`, checkpoint config key, and dependency group from Tasks 1-3.
- Produces: user-facing instructions for local Docker, checkpoint behavior, and Vertex dependency usage.

- [x] **Step 1: Add Local Docker docs**

Add a `## Local Docker` section near the training section in `projects/recsys-ranking/README.md`:

````markdown
## Local Docker

Dockerfile から image を build して、ローカルで学習を実行できます。

### 前提

- Docker（GPU アクセスのため `--gpus all` をサポート）
- `WANDB_API_KEY` 環境変数（[wandb.ai/authorize](https://wandb.ai/authorize) から取得）
- Dev shell では `WANDB_API_KEY` を `.env` に設定するか、`make docker-up` 実行時の shell で export してください。

### image を build して学習を直接実行

```sh
export WANDB_API_KEY=your-api-key
make docker-build
make docker-run
```

Hydra override も渡せます:

```sh
make docker-run DOCKER_ARGS="model=DeepFM data.batch_size=32"
```

### Dev shell としての利用

```sh
cp .env.example .env
# .env の WANDB_API_KEY を設定
make docker-up
make docker-exec
make docker-down
```

### 環境変数

| 変数 | 必須 | 説明 |
|------|------|------|
| `WANDB_API_KEY` | yes | wandb 認証用 API key |
| `IMAGE_URI` | optional | Docker image 名（default: `recsys-ranking:latest`。`make docker-run` では `DOCKER_IMAGE` 変数を使用） |
| `EXPERIMENT_NAME` | optional | Cloud Logging 用（ローカル Docker では使用されない） |
````

- [x] **Step 2: Document checkpointing**

In the configuration area, add:

```markdown
model checkpoint は default では保存しません。
必要な場合は `enable_checkpointing=true` を指定すると、`save_dir` 配下の `checkpoints/` に保存します。
```

- [x] **Step 3: Update dependency wording**

In the dependency / shared library sections, avoid saying normal install always depends on `vertex-job-runner`. State that Vertex AI job support is available through the `vertex` dependency group.

Use wording like:

```markdown
Vertex AI custom training job 連携が必要な場合は `vertex` dependency group で `vertex-job-runner` を追加します。
```

- [x] **Step 4: Run package verification**

Run from the ranking package root:

```sh
cd projects/recsys-ranking
make install
make fmt
make lint
make test
make -n docker-build
WANDB_API_KEY=dummy docker compose config
git diff --check
```

Expected:
- `make install`, `make fmt`, `make lint`, and `make test` exit 0.
- Docker dry-run uses `--build-context ml_sandbox_libs=../../libs/ml_sandbox_libs`.
- Compose config exits 0 with `WANDB_API_KEY=dummy`.
- `git diff --check` exits 0.

- [x] **Step 5: Review final diff**

Run:

```sh
git diff --stat
git diff -- projects/recsys-ranking
```

Expected:
- All new/changed files are under `projects/recsys-ranking`.
- Existing `projects/recsys-candidate-generation/Makefile` user diff is still present and unchanged unless the user explicitly asked to revisit it.

---

## Review Gates

1. After Task 1, inspect dependency churn before continuing. If `uv.lock` changes broad transitive versions unexpectedly, pause and judge whether `cu126` / `cu128` or tool-version alignment caused it.
2. After Task 2, run only dry-run Docker checks first. Do not require a real image build unless the user asks or a reviewer needs it.
3. After Task 3, targeted tests must pass before README changes.
4. After Task 4, delegate a fresh `auditor` review for the full ranking diff.

## Suggested Herdr Execution

Use one-shot coder delegation because this is scoped implementation but touches multiple files:

```sh
HERDR_AGENT_REUSE=never herdr-agent coder "Implement docs/superpowers/plans/2026-07-03-apply-pr159-fixes-to-recsys-ranking.md for projects/recsys-ranking only. Preserve existing user changes. Do not commit."
```

Then review with a fresh auditor:

```sh
HERDR_AGENT_REUSE=never herdr-agent auditor "Review the current projects/recsys-ranking diff against docs/superpowers/plans/2026-07-03-apply-pr159-fixes-to-recsys-ranking.md. Report concrete findings with severity and evidence."
```
