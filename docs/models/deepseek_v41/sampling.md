# DeepSeek DSpark sampling contract

## Scope

This version 1 contract defines project service behavior, not the local checkpoint's generation loop.
The [state contract](state.md) defines predictor mathematics and persistent state.
The [CPU oracle](../../../development/model/deepseek_v41/sampling_reference.py) defines exact rejection sampling and independent random draws.
The inference runtime does not use this oracle.
Qwen keeps its existing sampling and RNG behavior.

## Positions and pending tokens

Positions are zero-based language-model token positions.
Processed length counts target inputs with materialized state.
A pending token is already accepted and emitted, but its target computation is not complete.
Its position equals the processed length.
At most one such token exists between generation steps.

For a prompt of length `N>=2`, target prefill computes positions `0..N-1` and samples pending token `t` at position `N`.
Predictor seed consumes target features at positions `0..N-2`.
The proposal call appends the boundary feature at `p=N-1` and receives `t` as its first input token.
Thus, predictor main-state length becomes `N`.

Five draft attention rows use positions `N..N+4`.
Their logits predict proposals `y1..y5` at output positions `N+1..N+5`.
An attention-row position is not its predicted-token position.
Each Markov bias uses the preceding token: `t`, then `y1..y4`.

The [FP64 predictor](../../../development/model/deepseek_v41/reference_dspark.py) separates `append_main`, `project_features`, and `forward`.
It does not infer service positions from tensor shape.
Its main-window updates do not execute predictor FFNs.
Seed chunks keep only bounded windows, with absolute positions.

For a one-token prompt, first use ordinary target decode.
Start proposals only when the boundary feature has `p>0`.
Near the context limit, use ordinary decode if all five predictor attention positions cannot fit.
Do not shorten the bidirectional predictor block and claim the same mathematics.

## Verification and state commit

Let `L` be the pending-token position before verification.
Select a verification count `K` from zero through five.
Target computation consumes `[t,y1,...,yK]` at positions `L..L+K`.
Target row `i` predicts the token at `L+i+1`.
It uses private grammar and penalty history with the preceding proposed tokens.

If `a` drafts pass acceptance, publish those drafts and one correction or bonus token.
The processed target length becomes `L+1+a`.
The final published token is pending at that position.
The oracle's `processed_rows` is `1+a`, not the number of accepted drafts.
Do not process the correction token with features from the rejected proposal.

Before the next proposal, append accepted target features from `L` through `L+a-1` to the predictor main windows.
Hold the feature at `L+a` for that proposal's boundary append.
For `a=0`, the intermediate append is empty.
Rejected target rows and predictor proposal rows never enter accepted main windows.

If a stop condition ends output earlier, discard subsequent output and state.
Never commit target state beyond the accepted token history.
Each state group must select the same accepted prefix before publication.
Ring slots, compressor partials, and source identities follow the [restoration rules](state.md#accepted-state-and-restoration).

Let `R` count the output tokens still permitted after the pending token.
It includes the total-context bound.
If `R=0`, finish without verification.
Otherwise use `K<=min(5,R-1)` so a correction or bonus always has reserved space.
Stops can reduce the publication count further.

## Probability rules

Apply the same request settings, grammar, penalties, and stop-token masks to target and proposal logits.
These transforms use their respective accepted or private proposed prefixes.
Store each complete proposal distribution that actually generated its token.
Do not reconstruct it from later logits or store only the sampled probability.
Markov bias precedes these transforms.

The mathematical oracle uses finite, nonnegative FP64 probability rows with unit sum.
It accepts a sum error of at most `1e-12`, then normalizes the rows in FP64.
The implementation must separately validate its probability rounding against this reference.
No probability rounding tolerance or model-logit limit is set by this document.

For proposal `y_i`, accept when `u_i < min(1,p_i(y_i)/q_i(y_i))`.
A sampled proposal with `q_i(y_i)=0` is invalid input.
A zero target probability rejects even when the uniform draw is zero.
At the first rejection, sample from normalized `max(p_i-q_i,0)`.
Do not subtract a reduced proposal probability or add a correction penalty.
Zero residual mass on a rejection is an error, not permission to use the target distribution instead.

If all `K` drafts pass, sample the bonus from target row `K`.
For `K=0`, this is ordinary target sampling.
Categorical sampling uses an ascending-token CDF and a uniform draw in `[0,1)`.
Zero-mass tokens cannot be selected.
Greedy decoding accepts only matching target argmax tokens and emits the first mismatch or final bonus.
Exact argmax ties choose the lower token ID.

## RNG allocation

DeepSeek uses Philox4x32-10 with the two words of its unsigned 64-bit request seed as the key.
The counter words are `[position_low32,position_high32,cycle,stream]`.
Position names the sampled output token, not the target input row.
The request-local unsigned 32-bit cycle counts generation decisions.
Cycle 0 is the first target sample after prefill.
The next proposal and its verification share cycle 1, then cycle 2, and so on.

| Stream | ID | Purpose |
|---|---|---|
| Target | 0 | Ordinary target sample and initial prefill output |
| Proposal | 1 | One categorical draw per proposed output position |
| Accept | 2 | One acceptance draw per verified proposal position |
| Correction | 3 | One categorical draw at the first rejected position |
| Bonus | 4 | One target draw after a nonempty all-accepted prefix |

Use the first two result words for `((word0<<21) | (word1>>11))*2^-53`.
No batch index, rank ID, worker epoch, or execution order enters this address.
Unused draws cause no mutable counter updates.
A completed decision advances the cycle once, independent of its accepted length.
A new proposal block always uses the new cycle, even if its positions overlap discarded proposals.
Check counter overflow instead of wrapping.

Graph recording does not consume RNG state.
All ranks use the request owner's address, not separate rank draws.
Request failure aborts the transaction.
It does not publish a partial decision.
This contract provides deterministic addressing, not identical token paths between ordinary and speculative decoding.

## Confidence scheduling

First validate fixed-length verification.
The default verification cap is five.
Output and context bounds can reduce it.
Raw confidence is not an acceptance probability.
The initial measured fixed-five path uses no calibrated confidence threshold.

An optional later policy can select a shorter prefix with `sigmoid(clamp(raw/Tc,-60,60))`, where `Tc>0`.
It stops before the first score below its configured threshold.
The threshold and `Tc` must appear in the measurement identity.
The score for row `i` depends only on main features and tokens before proposal `i`.

Do not use the selected token at row `i` to choose whether to verify that row.
Do not use future proposal tokens or acceptance draws for that choice.
Such selection could bias sampling.
Confidence never changes `p`, `q`, or the acceptance formula.

## Validation boundary

Small CPU checks cover all five rejection positions, all-accept, zero drafts, residual mass, zero probabilities, greedy ties, and Philox known vectors.
They do not prove runtime grammar, stopping, distributed rollback, or RNG integration.
Phase 6 must validate those paths before DSpark service acceptance.
