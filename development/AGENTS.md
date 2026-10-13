# Development tools

Tools must not become inference dependencies. Keep experiments and raw measurements in external storage.
Install Python tools with `uv` in the selected project environment.
Use [development](../docs/development.md) for environment selection and [testing](../docs/testing.md) for acceptance.

## GPU ownership

Set `OH_MY_VLLM_MODEL` or supply `--model`. Use `scripts/with-gpu.sh` to wait for an idle B200 and pin its UUID.
Stop owned GPU processes and children after success, failure, cancellation, or handoff.
Check process birth identities and `nvidia-smi`. Stop only task-owned processes.

Keep a service active only for an explicit user request.
Record its purpose in the handoff and host identity in `LOCAL.md`.

## Dependencies and evidence

Use compatible stable dependencies in dependency order and pin the combination after compatibility tests.
TileFoundry is a development dependency from the pinned fork. Fix simple incompatibilities directly.
Discuss substantial fork changes first. Do not open upstream pull requests.

The permitted adaptation branch does not authorize branches in the main project.
Keep measurement source, environment, configuration, module, and checkpoint identities.
Do not select favorable samples or change theoretical bounds to hide overhead.
