# Tutorial and cleanup checks

## Tutorial and cleanup

Use the checks in [code-journey](../../code-journey/README.md#verification).
Examine desktop/mobile screenshots and all thirteen routes after tutorial changes.

Run each GPU program through `scripts/with-gpu.sh`.
Stop all owned processes and descendants after completion or failure.
Make sure that GPU processes stopped, listeners closed, and IPC cleanup completed.
Keep only explicitly user-requested services active. Put their host details in `LOCAL.md`.
