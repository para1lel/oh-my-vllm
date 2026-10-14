# Codex and OMP collaboration

Codex owns architecture, project integration, pull-request review, GPU execution, and end-to-end acceptance.
OMP implements assigned details and complete English/Chinese documents in its separate checkout.
Keep [active requirements](requirements.md), numerical tolerances, checkpoint bytes, and source-bound evidence equal.

## Issues and review

Record a new problem as a GitHub Issue, with its trigger, observed result, acceptance criteria, and responsible actor.
Use one scoped `omp/issue-N-description` branch for each OMP Issue.
The shared GitHub account labels comments with `[Codex]` or `[OMP/deepseek-flash]`.
Discuss assignments, findings, and corrections through Issues, pull requests, and their comments.

OMP requests independent Codex review before each milestone commit.
Correct findings, update the two handoff documents, and complete applicable checks with hooks enabled.
Stage intentional paths. Keep the required commit attribution in [contribution rules](../CONTRIBUTING.md).
Push the OMP branch and submit a draft pull request.

Before integration, rebase the pull request onto current `main`.
Repeat affected checks after rebase. Review the final head and base identities.
Squash its commits into one new commit whose only parent is current `main`.
Advance `main` by fast-forward and promptly synchronize GitHub.
If `main` changes, repeat rebase, checks, and review before integration.

Close an Issue after its correction is reviewed and integrated with applicable evidence.
Incomplete final-source acceptance stays open.

## Model and dispatch

The collaborator uses OMP `18.10.3` and official `deepseek/deepseek-flash`.
Default reasoning is `high`. Complex CUDA tasks can use `max`.
Credentials stay in the user credential store. Do not put them in source, command arguments, logs, Issues, or pull requests.

Text roles use the official model. Cross-provider retry fallback is disabled.
Image, search-model, speech, and dictation roles are not enabled for this development workflow.
Host configuration, proxy selection, and checkout paths stay in ignored `LOCAL.md`.
The independent checkout has separate build and runtime-cache output.

There is no API cost cap.
The [official peak periods](https://api-docs.deepseek.com/zh-cn/quick_start/pricing) use Beijing time: weekday 09:00–12:00 and 14:00–18:00, except legal holidays.
The [dispatch policy](../development/collaboration.py) checks `Asia/Shanghai` immediately before each new Issue task starts.
Unverified weekday holidays conservatively keep the peak block active.

OMP can finish an existing Issue across a pricing boundary. A resumed task must use its recorded native session.
Codex handles new work during peak periods.
The policy governs task starts. It does not prove how the provider bills requests across a boundary.
Local-model OMP acceptance clients do not use this paid-provider dispatch policy.

Use the [entry point](../scripts/omp-collaborator.py) through the environment wrapper:

```bash
scripts/with-env.sh python scripts/omp-collaborator.py --checkout "$OMP_CHECKOUT" --issue 4
scripts/with-env.sh python scripts/omp-collaborator.py --checkout "$OMP_CHECKOUT" --issue 4 --continue-task "Apply the reviewed corrections on Issue 4."
```

The controller verifies repository and branch scope, prevents concurrent dispatch, and keeps process birth and native session records.
It repeats scope and clean-state checks after GitHub access, then checks pricing again before launch.
The isolated supervisor verifies descendant cleanup after natural task completion or controller failure.
Proxy configuration comes from the environment or `OH_MY_VLLM_GITHUB_PROXY`, with host values in `LOCAL.md`.

Original task logs stay in the checkout's ignored `.local` directory.
Stop owned GPU processes and temporary services after runs. Keep the user-authorized CPU tutorial preview.
