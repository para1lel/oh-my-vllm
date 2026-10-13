# Test ownership

Test critical behavior through independent assertions. Do not copy implementation steps into expected results.
Keep scheduler/cache ownership, commit/rollback, speculation, sampling, protocol faults, and numerical boundary contracts.
Do not duplicate ordinary paths covered by integration or end-to-end tests.

Move shared builders and references into support modules before moving tests.

## Layers

Directories identify responsibility. Markers identify execution conditions.
Register markers and reject unknown markers.
Keep unit, integration, numerical, tooling, and acceptance selections explicit.

Use small deterministic state/property cases without a new dependency when the existing tools are sufficient.
Keep original failed records, case selections, and source identities. Never reuse past evidence as current-source acceptance.

## Environment and cleanup

Use the environment wrapper. GPU tests use an idle pinned B200 through the GPU wrapper.
Agent workloads run in disposable sandboxes with new work directories, not the inference repository.
Remove their files, copied or installed tools, containers, and owned images after testing.

Keep critical fault tests even when ordinary end-to-end paths pass.
Stop owned workers, children, and temporary service ports after each run.
