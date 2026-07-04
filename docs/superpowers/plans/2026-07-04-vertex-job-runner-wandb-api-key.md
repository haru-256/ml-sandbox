# Vertex Job Runner Wandb Api Key Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `vertex-job-runner` to pass a W&B API key into Vertex AI custom training containers while removing the unused `GCS_URI` container environment variable.

**Architecture:** Keep the existing `Settings` → `job.run(environment_variables=...)` boundary. Add an optional secret setting loaded from `VRUN_WANDB_API_KEY`, inject it as `WANDB_API_KEY` only when provided, and keep `gcs_uri` only for Vertex AI SDK `staging_bucket` / `base_output_dir` because those still require a GCS bucket.

**Tech Stack:** Python 3.12, Pydantic Settings, Google Cloud Vertex AI SDK, Typer CLI, pytest via `make test`, ruff/mypy via `make lint`.

---

## File Structure

- Modify `apps/vertex-job-runner/src/vertex_job_runner/settings.py`: add optional `wandb_api_key` setting loaded from `VRUN_WANDB_API_KEY`; use `pydantic.SecretStr` so dry-run/config dumps do not expose the raw key.
- Modify `apps/vertex-job-runner/src/vertex_job_runner/job.py`: construct the environment variable dict explicitly, remove `GCS_URI`, and add `WANDB_API_KEY` when configured.
- Modify `apps/vertex-job-runner/tests/test_cli.py`: keep CLI dry-run coverage stable and add direct job tests for env var injection/removal.
- Modify `apps/vertex-job-runner/README.md`: document `VRUN_WANDB_API_KEY`, clarify that `gcs_uri` is for Vertex AI staging/output only, and remove any implication that `GCS_URI` is passed to the container.
- Modify `apps/vertex-job-runner/samples/.env.local`: add a commented `VRUN_WANDB_API_KEY` example and keep `VRUN_GCS_URI` only as SDK staging/output config.
- Modify `projects/recsys-candidate-generation/README.md`: document setting `VRUN_WANDB_API_KEY` before `vrun run`.
- Modify `projects/recsys-ranking/README.md`: add the same Vertex AI/W&B env guidance without inventing a `[tool.vrun]` block.

## Chosen Design and Alternatives

1. **Recommended: optional secret setting + conditional env injection.** `WANDB_API_KEY` is not stored in `pyproject.toml`; users export `VRUN_WANDB_API_KEY`, and `job.py` passes `WANDB_API_KEY` into the container only when set. This is minimal and avoids leaking secrets.
2. **Plain string setting.** Simpler code, but dry-run/config output can reveal the API key. Reject this because the key is a secret.
3. **Remove `gcs_uri` entirely.** Reject for this change because `CustomContainerTrainingJob(staging_bucket=...)` and `job.run(base_output_dir=...)` still use it. The unused thing is the `GCS_URI` container env var, not the SDK bucket configuration.

## Acceptance Criteria

- `VRUN_WANDB_API_KEY` can be read by `Settings` without putting a secret in `pyproject.toml`.
- `run_custom_training_job()` passes `WANDB_API_KEY` to Vertex AI container environment variables when configured.
- `run_custom_training_job()` no longer passes `GCS_URI` to the container.
- `gcs_uri` remains available for Vertex AI SDK `staging_bucket` and `base_output_dir`.
- README updates explain the `VRUN_WANDB_API_KEY` workflow for both recsys projects.
- `make lint` and `make test` pass in `apps/vertex-job-runner`.

---

### Task 1: Add settings support for W&B API key

**Files:**
- Modify: `apps/vertex-job-runner/src/vertex_job_runner/settings.py`

- [ ] **Step 1: Add `SecretStr` import**

Change the Pydantic import from:

```python
from pydantic import Field, field_validator
```

to:

```python
from pydantic import Field, SecretStr, field_validator
```

- [ ] **Step 2: Add optional setting**

Insert this field after `experiment_name` and before `command`:

```python
    wandb_api_key: SecretStr | None = Field(
        default=None,
        description="The W&B API key to pass to the training container",
    )
```

This maps to `VRUN_WANDB_API_KEY` because `SettingsConfigDict(env_prefix="VRUN_")` is already configured.

- [ ] **Step 3: Run focused static check**

Run from `apps/vertex-job-runner`:

```bash
make lint
```

Expected: may fail before later tasks if existing tests or code need updates, but no import/type error should remain in `settings.py` after this task.

---

### Task 2: Pass WANDB_API_KEY and remove GCS_URI env var

**Files:**
- Modify: `apps/vertex-job-runner/src/vertex_job_runner/job.py`

- [ ] **Step 1: Build environment variables before `job.run`**

Add this block after `CustomContainerTrainingJob(...)` creation and before `job.run(...)`:

```python
    environment_variables = {
        "PROJECT": settings.project,
        "EXPERIMENT_NAME": settings.experiment_name,
    }
    if settings.wandb_api_key is not None:
        environment_variables["WANDB_API_KEY"] = settings.wandb_api_key.get_secret_value()
```

- [ ] **Step 2: Replace inline environment dict**

Replace:

```python
        environment_variables={
            "PROJECT": settings.project,
            "EXPERIMENT_NAME": settings.experiment_name,
            "GCS_URI": settings.gcs_uri,
        },
```

with:

```python
        environment_variables=environment_variables,
```

Keep these existing SDK arguments unchanged:

```python
        staging_bucket=settings.gcs_uri,
        base_output_dir=settings.gcs_uri,
```

- [ ] **Step 3: Run focused static check**

Run from `apps/vertex-job-runner`:

```bash
make lint
```

Expected: pass, or report only pre-existing issues unrelated to this change. Fix any new type/lint issue introduced by this task.

---

### Task 3: Add job-level tests for env injection

**Files:**
- Modify: `apps/vertex-job-runner/tests/test_cli.py`

- [ ] **Step 1: Add imports for mocking and settings**

At the top of `test_cli.py`, add:

```python
from unittest.mock import MagicMock, patch

from vertex_job_runner.job import run_custom_training_job
from vertex_job_runner.settings import Settings
```

- [ ] **Step 2: Add a settings fixture helper**

Append this helper after `strip_ansi`:

```python
def make_settings(**overrides: object) -> Settings:
    """Create Settings for job unit tests."""
    values = {
        "project": "test-project",
        "location": "us-central1",
        "image_uri": "us-docker.pkg.dev/test/image:latest",
        "gcs_uri": "gs://test-bucket/vertex/",
        "service_account": "trainer@test-project.iam.gserviceaccount.com",
        "experiment_name": "test-experiment",
        "command": ["uv", "run", "python", "src/fit.py"],
        "machine_type": "g2-standard-4",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 1,
        "args": ["model=SASRec"],
    }
    values.update(overrides)
    return Settings(**values)
```

- [ ] **Step 3: Add test for WANDB env injection and GCS_URI removal**

Append this test:

```python
def test_run_custom_training_job_passes_wandb_api_key_without_gcs_env() -> None:
    settings = make_settings(wandb_api_key="secret-wandb-key")
    mock_job = MagicMock()
    mock_job.resource_name = "projects/test/locations/us-central1/trainingPipelines/123"

    with patch("vertex_job_runner.job.aiplatform.CustomContainerTrainingJob", return_value=mock_job):
        resource_name = run_custom_training_job(settings)

    assert resource_name == "projects/test/locations/us-central1/trainingPipelines/123"
    _, kwargs = mock_job.run.call_args
    assert kwargs["environment_variables"] == {
        "PROJECT": "test-project",
        "EXPERIMENT_NAME": "test-experiment",
        "WANDB_API_KEY": "secret-wandb-key",
    }
    assert "GCS_URI" not in kwargs["environment_variables"]
```

- [ ] **Step 4: Add test for absent WANDB key**

Append this test:

```python
def test_run_custom_training_job_omits_wandb_api_key_when_unset() -> None:
    settings = make_settings()
    mock_job = MagicMock()
    mock_job.resource_name = "projects/test/locations/us-central1/trainingPipelines/456"

    with patch("vertex_job_runner.job.aiplatform.CustomContainerTrainingJob", return_value=mock_job):
        run_custom_training_job(settings)

    _, kwargs = mock_job.run.call_args
    assert kwargs["environment_variables"] == {
        "PROJECT": "test-project",
        "EXPERIMENT_NAME": "test-experiment",
    }
```

- [ ] **Step 5: Run focused tests**

Run from `apps/vertex-job-runner`:

```bash
make test
```

Expected: all tests pass. If `SecretStr` input needs an explicit construction in tests, update only the helper/test input while preserving the production field type.

---

### Task 4: Update vertex-job-runner docs and samples

**Files:**
- Modify: `apps/vertex-job-runner/README.md`
- Modify: `apps/vertex-job-runner/samples/.env.local`

- [ ] **Step 1: Update README config item wording**

In `apps/vertex-job-runner/README.md`, keep `gcs_uri` in the `[tool.vrun]` example, but word it as SDK staging/output storage:

```md
- `gcs_uri`: Vertex AI SDK の staging bucket と base output dir に利用する GCS URI
```

- [ ] **Step 2: Add W&B env var example**

In the environment variable section, extend the shell example to include:

```sh
export VRUN_WANDB_API_KEY="your-wandb-api-key"
```

Add this paragraph below the example:

```md
`VRUN_WANDB_API_KEY` を設定すると、Vertex AI の training container には `WANDB_API_KEY` として渡されます。`pyproject.toml` には API key を書かず、shell・CI secret・Secret Manager などから環境変数として渡してください。
```

- [ ] **Step 3: Clarify caution section**

Replace the caution bullet mentioning `gcs_uri` with:

```md
- `service_account`、`project`、SDK staging/output 用の `gcs_uri` は実際の環境に合わせて設定してください。
- W&B を使う training container では、実行前に `VRUN_WANDB_API_KEY` を設定してください。これは container 内では `WANDB_API_KEY` として参照されます。
```

- [ ] **Step 4: Update sample env**

In `apps/vertex-job-runner/samples/.env.local`, keep any `VRUN_GCS_URI` sample as staging/output configuration and add:

```sh
# W&B API key for training containers. Do not commit real values.
# VRUN_WANDB_API_KEY=your-wandb-api-key
```

---

### Task 5: Update recsys project docs

**Files:**
- Modify: `projects/recsys-candidate-generation/README.md`
- Modify: `projects/recsys-ranking/README.md`

- [ ] **Step 1: Update candidate generation Vertex AI section**

After the `[tool.vrun]` example in `projects/recsys-candidate-generation/README.md`, add:

```md
W&B を使う Vertex AI job では、API key を `pyproject.toml` に書かず、実行時に `VRUN_WANDB_API_KEY` として渡します。

```sh
export VRUN_WANDB_API_KEY="your-wandb-api-key"
uv run vrun run
```

`vertex-job-runner` はこの値を training container の `WANDB_API_KEY` として設定します。
```

- [ ] **Step 2: Add ranking Vertex AI note**

In `projects/recsys-ranking/README.md`, after the Local Docker environment variable table, add:

```md
## Vertex AI

Vertex AI custom training job で W&B を使う場合は、API key を `pyproject.toml` に書かず、`vertex-job-runner` 実行時の環境変数として渡します。

```sh
export VRUN_WANDB_API_KEY="your-wandb-api-key"
uv run vrun run
```

`vertex-job-runner` はこの値を training container の `WANDB_API_KEY` として設定します。`[tool.vrun]` の project 固有設定を追加する場合も、secret は設定ファイルに保存しないでください。
```

---

### Task 6: Final verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Format**

Run from `apps/vertex-job-runner`:

```bash
make fmt
```

Expected: formatter completes successfully.

- [ ] **Step 2: Lint**

Run from `apps/vertex-job-runner`:

```bash
make lint
```

Expected: lint and mypy pass.

- [ ] **Step 3: Test**

Run from `apps/vertex-job-runner`:

```bash
make test
```

Expected: pytest passes.

- [ ] **Step 4: Check diff scope**

Run from repository root:

```bash
git diff -- apps/vertex-job-runner projects/recsys-candidate-generation/README.md projects/recsys-ranking/README.md docs/superpowers/plans/2026-07-04-vertex-job-runner-wandb-api-key.md
```

Expected: diff only contains W&B API key support, `GCS_URI` container env removal, and related docs/tests.

## Implementation Log
<!-- Implementer appends one line per attempt: [YYYY-MM-DD] attempt #N -> STATUS | commit-or-failure-signature -->
[2026-07-04] attempt #1 -> DONE | no commit (user instructed not to commit)
[2026-07-04] attempt #2 -> DONE | resolved accepted findings F1/F4/F5; no commit (user instructed not to commit)

## Review Findings
<!-- This template is also defined in commands/plan.md. Keep them in sync on every edit. -->

### Reviewer Raw Findings
<!-- Orchestrator copies @reviewer's structured findings verbatim here when invoking @reviewer during a workflow. Direct /review-* calls do not write here. Raw findings are review input, not implementation instructions. -->

#### [2026-07-04] implementation -> REQUEST_CHANGES
Critical issues:
- F1: MAJOR / HIGH / scope — Evidence: `infra/terraform/modules/vertex_ai_training/main.tf` diff hunk: `+  project       = var.project_id` in `google_artifact_registry_repository.ml_sandbox`; not mentioned anywhere in the plan's File Structure or Acceptance Criteria. Why it matters: The plan scope is W&B API key support + `GCS_URI` container env removal. This Terraform edit is an unrelated infra change; bundling it violates the repo's "変更範囲は依頼範囲に限定" rule and makes review/rollback granularity worse. It also touches IAM-adjacent infra (project scoping of a registry), which per the escalation policy warrants isolation. Recommended action: Revert this hunk from the working tree and ship it as a separate commit/PR with its own justification (e.g., fixing a missing project scoping on the registry resource). Must fix before merge: yes

Non-blocking suggestions:
- F2: MINOR / HIGH / docs (maintainability) — Evidence: `apps/vertex-job-runner/src/vertex_job_runner/job.py:8-9` — docstring remains `"""Run a custom training job to Vertex AI."""` after the function was materially changed (now injects `WANDB_API_KEY` conditionally and no longer passes `GCS_URI`). Why it matters: AGENTS.md requires docstrings be updated when behavior substantively changes; a reader cannot tell from the docstring what env vars are injected or under what condition. Recommended action: Expand the docstring (Google style) to state that `PROJECT`/`EXPERIMENT_NAME` are always injected, `WANDB_API_KEY` is injected only when `settings.wandb_api_key` is set, and `GCS_URI` is intentionally not passed to the container (only to SDK staging/output). Must fix before merge: no
- F3: MINOR / MEDIUM / test — Evidence: `apps/vertex-job-runner/tests/test_cli.py` — both new tests assert only `mock_job.run.call_args` kwargs; neither captures `aiplatform.CustomContainerTrainingJob(...)` constructor args. Why it matters: Acceptance criterion #4 ("`gcs_uri` remains available for Vertex AI SDK `staging_bucket` and `base_output_dir`") is not directly asserted; a future regression that drops `staging_bucket=settings.gcs_uri` would not be caught by these tests. Recommended action: Add an assertion on the `CustomContainerTrainingJob` call kwargs (`staging_bucket`, `container_uri`) and on `run(...)`'s `base_output_dir` to lock the SDK-side `gcs_uri` contract. Must fix before merge: no
- F4: MINOR / MEDIUM / security (docs) — Evidence: `apps/vertex-job-runner/README.md:107-109` and recsys READMEs document `VRUN_WANDB_API_KEY` flow without noting that Vertex AI container environment variables are visible in the Vertex AI job detail / audit logs. Why it matters: Users may assume the key is hidden because it is a `SecretStr` on the runner side; once injected as a container env var it is observable in Vertex AI surfaces. The plan's chosen design accepts this, but the docs should state it so users can judge whether to use Secret Manager instead. Recommended action: Add a one-line caution that `WANDB_API_KEY` is passed as a container environment variable and is therefore visible in Vertex AI job details; for higher-security needs, use Secret Manager. Must fix before merge: no
- F5: NIT / LOW / docs — Evidence: `projects/recsys-ranking/README.md` new "## Vertex AI" section does not reference the `apps/vertex-job-runner` README for the full `[tool.vrun]` setup, unlike the candidate-generation README which says "実際の job 実行時は `apps/vertex-job-runner` 側の README も参照してください。" Why it matters: Minor consistency gap between the two recsys docs. Recommended action: Add the same cross-reference line in the ranking README's Vertex AI section. Must fix before merge: no

#### [2026-07-04] implementation -> REQUEST_CHANGES
Critical issues:
- F1: MAJOR / HIGH / scope — UNRESOLVED. Evidence: `infra/terraform/modules/vertex_ai_training/main.tf` diff hunk `+  project       = var.project_id` in `google_artifact_registry_repository.ml_sandbox`; still present in working tree. Plan `Deviations from Plan` and `Open Questions` sections are empty — no ownership explanation recorded. Why it matters: Orchestrator adjudication explicitly required either removing the hunk or reporting ownership. Neither happened; the implementer returned an empty report. This violates the repo's "変更範囲は依頼範囲に限定" rule and the plan's own scope, and touches IAM/infra-adjacent state. Recommended action: Either revert the Terraform hunk from this working tree, or add a `Deviations from Plan` entry stating this is pre-existing user-owned work that the workflow did not introduce (with evidence). Do not merge the W&B change with an unexplained infra edit. Must fix before merge: yes

Non-blocking suggestions:
- F2: MINOR / HIGH / docs — RESOLVED. Evidence: `apps/vertex-job-runner/src/vertex_job_runner/job.py:8-17` docstring now documents `PROJECT`/`EXPERIMENT_NAME` always injected, `WANDB_API_KEY` conditional, and intentional `GCS_URI` omission, with `Args:`/`Returns:`. No further action.
- F3: MINOR / MEDIUM / test — RESOLVED. Evidence: `apps/vertex-job-runner/tests/test_cli.py` both new tests now assert `ctor_kwargs["staging_bucket"]`, `ctor_kwargs["container_uri"]`, and `run_kwargs["base_output_dir"]`, locking the SDK-side `gcs_uri` contract. Tests pass. No further action.
- F4: MINOR / MEDIUM / security (docs) — UNRESOLVED. Evidence: `apps/vertex-job-runner/README.md:107-110` added paragraph mentions Secret Manager only as a source for the env var, not as an alternative for higher-security needs, and does not state that the injected container env var is visible in Vertex AI job detail / audit surfaces. Recommended action: Add a one-line caution that `WANDB_API_KEY` is passed as a container environment variable and is therefore visible in Vertex AI job details; for higher-security needs, use Secret Manager. Must fix before merge: no
- F5: NIT / LOW / docs — UNRESOLVED. Evidence: `projects/recsys-ranking/README.md` new `## Vertex AI` section does not include the cross-reference line `実際の job 実行時は apps/vertex-job-runner 側の README も参照してください。` that exists in the candidate-generation README. Recommended action: Append the same cross-reference line at the end of the ranking README's Vertex AI section. Must fix before merge: no

#### [2026-07-04] implementation -> APPROVE
[2026-07-04] implementation -> APPROVE | no findings

### Orchestrator Adjudication
<!-- Orchestrator appends adjudication tables for workflow reviews. Only ACCEPT rows are implementation instructions: | ID | Severity | Decision | Reason | Action | -->

#### [2026-07-04] implementation -> REQUEST_CHANGES

| ID | Severity | Decision | Reason | Action |
|----|----------|----------|--------|--------|
| F1 | MAJOR | ACCEPT | Concrete out-of-scope Terraform diff violates the plan scope and repo change-scope rule; fix is proportionate. | Remove the Terraform hunk if introduced by this workflow; if it is a pre-existing/user-owned change, leave it untouched and report ownership. |
| F2 | MINOR | ACCEPT | Repo AGENTS requires docstrings to stay current for materially changed behavior. | Expand `run_custom_training_job` docstring with injected env vars and intentional `GCS_URI` omission. |
| F3 | MINOR | ACCEPT | Acceptance criterion says `gcs_uri` must remain available for SDK staging/output; adding assertions is low-risk. | Assert constructor `staging_bucket` and run `base_output_dir` in job tests. |
| F4 | MINOR | ACCEPT | Secret exposure surface is user-relevant security documentation and aligned with the chosen env-var design. | Add docs caution that Vertex AI container env vars can be visible in job details/logging surfaces; mention Secret Manager for higher-security needs. |
| F5 | NIT | ACCEPT | Consistency fix is tiny and stays within documentation scope. | Add vertex-job-runner README cross-reference to ranking README Vertex AI section. |

#### [2026-07-04] implementation -> REQUEST_CHANGES follow-up

| ID | Severity | Decision | Reason | Action |
|----|----------|----------|--------|--------|
| F1 | MAJOR | ACCEPT | Still unresolved and blocking; review evidence shows out-of-scope Terraform diff remains unexplained. | Fresh implementer must either remove the Terraform hunk if workflow-owned or document it under Deviations from Plan as pre-existing/user-owned with evidence. |
| F2 | MINOR | ACCEPT | Already resolved; no further action needed. | No action. |
| F3 | MINOR | ACCEPT | Already resolved; no further action needed. | No action. |
| F4 | MINOR | ACCEPT | Security docs remain incomplete for env-var exposure surface. | Add Vertex AI job detail/audit visibility caution and Secret Manager alternative guidance. |
| F5 | NIT | ACCEPT | Ranking README cross-reference remains missing. | Add `apps/vertex-job-runner` README cross-reference line. |

## Deviations from Plan
<!-- Implementer documents intentional deviations and reasons. -->

## Open Questions
<!-- Any agent adds questions for orchestrator or oracle. -->
