# Pytest Mock Addition

plan body not authored in this workflow

## Implementation Log
<!-- Implementer appends one line per attempt: [YYYY-MM-DD] attempt #N -> STATUS | commit-or-failure-signature -->
- [2026-07-04] attempt #2 -> DONE | no commit (F3/F4 only)

## Review Findings
<!-- This template is also defined in commands/plan.md. Keep them in sync on every edit. -->

### Reviewer Raw Findings
<!-- Orchestrator copies @reviewer's structured findings verbatim here when invoking @reviewer during a workflow. Direct /review-* calls do not write here. Raw findings are review input, not implementation instructions. -->

#### [2026-07-04] implementation -> REQUEST_CHANGES
Critical issues:
- F1: BLOCKER / MAJOR — Evidence: `apps/vertex-job-runner/src/vertex_job_runner/job.py` diff: `environment_variables` no longer includes `"GCS_URI": settings.gcs_uri`; `tests/test_cli.py:91` asserts `"GCS_URI" not in run_kwargs["environment_variables"]`. Why it matters: Any deployed training container that reads `GCS_URI` from its env will break at runtime; this is a silent breaking change with no deprecation path or migration warning in `README.md`. Recommended action: Either restore `GCS_URI` to container env for backward compat and add `WANDB_API_KEY` alongside, or add an explicit breaking note and confirm no downstream training image relies on it. Must fix before merge: yes
- F2: MAJOR — Evidence: `git diff` touches `settings.py`, `job.py`, `README.md`, `samples/.env.local` — none of which are "pytest-mock dependency addition and test migration." Original `test_cli.py` had no `unittest.mock` usage to migrate. Why it matters: The review scope as stated would not catch the breaking change in F1; the PR bundles a feature + breaking change under a dep-addition label. Recommended action: Split into two PRs or retitle/add breaking changes section. Must fix before merge: yes

Non-blocking suggestions:
- F3: MINOR — Evidence: `apps/vertex-job-runner/pyproject.toml:24` uses `"pytest-mock>=3.15.1"` while `pytest~=9.0.3` and `pytest-console-scripts~=1.4.1` use `~=`. Recommended action: Use `pytest-mock~=3.15` for consistency with sibling test deps. Must fix before merge: no
- F4: MINOR — Evidence: `tests/test_cli.py:116` `def test_env_var_override(monkeypatch: Any) -> None:` annotates the fixture as `Any` instead of `pytest.MonkeyPatch`. Recommended action: Change to `monkeypatch: pytest.MonkeyPatch` and add `import pytest`. Must fix before merge: no
- F5: NIT — Evidence: `README.md` 注意事項 now says "SDK staging/output 用の `gcs_uri`" but never states the container no longer receives `GCS_URI` env var. Recommended action: Add one line documenting that `GCS_URI` is no longer passed to the training container. Must fix before merge: no

#### [2026-07-04] implementation -> APPROVE
[2026-07-04] implementation -> APPROVE | no findings

### Orchestrator Adjudication
<!-- Orchestrator appends adjudication tables for workflow reviews. Only ACCEPT rows are implementation instructions: | ID | Severity | Decision | Reason | Action | -->

#### [2026-07-04] implementation -> REQUEST_CHANGES

| ID | Severity | Decision | Reason | Action |
|----|----------|----------|--------|--------|
| F1 | BLOCKER | REJECT | The `GCS_URI` removal belongs to the immediately preceding user-requested W&B/GCS_URI workflow and already passed final review; it was not introduced by this pytest-mock task. | No action in this task. |
| F2 | MAJOR | REJECT | The broader W&B/docs diffs are pre-existing uncommitted changes from the prior approved workflow, not scope creep introduced by adding pytest-mock. | No action in this task. |
| F3 | MINOR | ACCEPT | This directly concerns the newly added pytest-mock dependency and matches existing dependency style. | Change specifier to `pytest-mock~=3.15` and refresh lockfile. |
| F4 | MINOR | ACCEPT | This is a small test typing cleanup in the same file and aligns fixture typing style with `MockerFixture`. | Use `pytest.MonkeyPatch` for the `monkeypatch` fixture. |
| F5 | NIT | DEFER | Potential docs note for the prior GCS_URI behavior change, but outside this pytest-mock dependency task and not blocking the requested change. | Track as follow-up only. |

## Deviations from Plan
<!-- Implementer documents intentional deviations and reasons. -->

## Open Questions
<!-- Any agent adds questions for orchestrator or oracle. -->
- [missing-plan] [2026-07-04] No plan authored for docs/superpowers/plans/2026-07-04-pytest-mock-addition.md; skeleton created to preserve the audit trail; plan body not fabricated.
- [defer] [2026-07-04] F5: Consider adding a docs note that `GCS_URI` is no longer passed to the training container in a separate docs-focused follow-up.
