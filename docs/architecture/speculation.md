# Speculative decoding concepts

## MTP

Draft row `p` combines target hidden row `p-1` and token row `p`.
Row zero is absent.
The uniform RoPE offset keeps relative positions.
Rust allocates draft/state capacity. Python never invents logical blocks.

A 784-token boundary hidden state waits if its required page is unavailable.

MTP4 proposes four drafts and verifies them against target sampling.
Grouped verification has at most five queries with per-query causal lengths.
The single-token decode path has its own implementation.
The worker keeps accepted target output and the correct GDN/convolution snapshot.
Rejected scheduled drafts roll back computed progress. Unscheduled drafts have no computed progress to roll back.

Grammar simulation uses each speculative prefix and rolls back before it commits kept output.

## DSpark

The optional mode loads the local DSpark checkpoint through project-owned code.
It validates configuration, tensor names, shapes, BF16 dtype, and stable source-file identities before execution.
The loaded configuration and weight hashes identify the bytes that supplied the tensors.
Checkpoint Python code does not execute.

The draft uses five GQA layers and zero-based target layer outputs `(5,19,33,47,61)`.
Each feature is the BF16 layer output before the subsequent normalization.
MTP4 and ordinary computation do not calculate these features.
The shared target embedding and vocabulary head supply DSpark token representations and base logits.
The draft projects concatenated target features into context KV for its five layers.

DSpark context pages use the same Rust-owned FA page IDs and 784-token layout, in a different physical cache.
Only committed target input rows enter this context.
Rejected input rows can stay in target FA after the committed cursor and are overwritten before subsequent use.
DSpark context receives only kept input rows.
Seven proposal rows use bidirectional draft attention and temporary KV.
They do not enter persistent context.

The Markov head uses the previous draft token to adjust each row's base logits.
The confidence head limits the candidate prefix through cumulative confidence.
The initial `--dspark-confidence-threshold` setting is `0.2`.
A first confidence product less than the threshold returns zero candidates and uses one target token for that step.
Set it to `0.0` to keep all valid rows up to seven for fixed-count experiments.
Output and context budgets can make that prefix shorter.

Draft block 7 and training block 16 are different from target block 784.

Verification with greedy sampling compares draft tokens with the target tokens that have maximum scores.
For stochastic output, the worker keeps each candidate's full conditional proposal distribution `q`.
Target distribution `p` and proposal distribution `q` use the configured temperature, penalties, top-k/top-p, and grammar masks.

Accept candidate `x` with probability `min(1,p(x)/q(x))`.
After rejection, sample from normalized `max(p-q,0)`. After full acceptance, sample the bonus token from `p`.
Draft sampling has a different random generator. Target sampling keeps its own generator.

The existing GDN snapshot/rollback logic keeps the state after committed input, with at most eight verification rows.
Speculative state is BF16. The target and draft keep specified BF16 rounding points.
Grammar simulation rolls back before committed output changes persistent grammar state.
Cancellation and preemption release target and draft request state.

The worker selects one mode when it starts. It keeps that mode until it stops.
See [ADR-009](../decisions/ADR-009-dspark-optional-mode.md).
