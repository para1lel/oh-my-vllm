# Project principles

Build single-B200 Qwen3.8-27B-FP8 inference. Rust owns service, scheduling, and logical KV. Python owns GPU computation.
Use ZMQ DEALER and msgpack. Keep the [runtime contracts](docs/requirements.md).

Read Git status, the [handoff](docs/handoff.md), and applicable directory instructions before work.
Follow [CONTRIBUTING.md](CONTRIBUTING.md). Add directory instructions when local contracts appear.

Keep root guides short. Put details near their owner and maintain complete Chinese companions.
Use clear names and small modules. Add comments only for facts the code cannot express clearly.

Keep tests few and focused on behavior. Tell the user why each additional test is useful before adding it.
Run applicable existing tests yourself. Do not store test results in project files.

Get an independent sub-agent review at each delivery and correct its findings.
Keep hooks enabled, commit intentional paths, and synchronize the working branch with GitHub after each delivery.

Keep host settings in ignored `LOCAL.md`. Stop owned workers and temporary services after use.
