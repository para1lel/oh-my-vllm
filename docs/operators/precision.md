# Operator numerical limits

The [microbench design](microbench.md) gives fixed inputs and independent PyTorch fixtures.
This document gives accuracy policy `operator-precision-v1`.
The [limit proposal tool](../../development/operators/budgets.py) applies this policy to the stored FP64 reference arrays.
It writes a new external file for review.
It does not read candidate outputs or change approved limits.

## Field limits

Use `u16 = 1/256` and `u8 = 1/16` for the BF16 and E4M3 normal rounding units.
These units supply engineering accuracy limits.
The limits are not worst-case rounding-error proofs.
Projection conditioning, nonlinear computation, and repeated layers can amplify rounding error.
Keep different limits for projected values, residuals, and persistent state.

`P` is the maximum absolute FP64 reference value for the applicable field and sequence.
Compare each chain hidden/logit field independently for each request and draft step.
Use only the magnitude of that step in each comparison.

| Field | Normalized L2 limit | Maximum absolute error |
|---|---|---|
| Residual | `u16` | `u16 * P` |
| Final hidden and hidden cache | `2*u16` | `4*u16 * P` |
| Output logits | `4*u16` | `8*u16 * P` |
| MTP context projection | `4*u16` | `8*u16 * P` |
| Vision hidden | `4*u16` | `8*u16 * P` |
| FFN value | `u8` | `2*u8 * P` |
| FA value | `0.1` | `2*u8 * P` |
| GDN value | `0.1` | `2*u8 * P_j` for channel `j` |
| BF16 KV, recurrent state, convolution state | `u8` | `2*u8 * P` |
| FP8 KV codes | `2*u8` | `3*u8 * P` |
| KV scales | `u8` | `2*u8 * P` |
| One MTP draft hidden/logits | `u8 + 4*u16` | `2*u8 * P` |
| Four-draft chain hidden/logits | `2*u8` | `4*u8 * P` |

BF16-only paths have smaller limits than complete FP8 sublayers.
The one-draft limit includes BF16 input projection, normalization, and vocabulary projection before and after the FP8 sublayers.
The chain limit is higher because draft steps repeat.
Persistent-state limits are the same for these paths.

For GDN output projection `y = W*x`, calculate `R_j = sqrt(sum_i (W_ji*x_i)^2)` from the FP64 reference.
For each sequence, use `P_j = max(P, max_t R_tj)`.
Store one vector of `max_t R_tj` values per sequence in the reference manifest.
The limit proposal tool calculates `P_j` from these values.
Apply each channel limit to all its output rows.

Only channels with conditioning scales above the original sequence peak have larger limits.
This engineering policy includes projection sensitivity.
It is not a worst-case bound for coherent input errors.
The normalized L2 and persistent-state limits still apply.

Use an RMS denominator floor of `1e-6`.
Use a maximum absolute-error floor of `1e-7` when the table gives a smaller limit.
For reference RMS at most `1e-6`, maximum absolute error must also be at most `1e-6`.
All conditions must pass.
Reject nonfinite values.
Use bit equality for integer outputs, RNG counters, assigned slots, and untouched storage.

## Cache encoding

The scale limits must also pass the [cache consistency checks](../../development/operators/numerics.py).
Check the K scale against `max(32 * max(abs(1 + k_norm)), 1e-10) / 448`.
Check each V scale against `max(max(abs(candidate_BF16_V)), 1e-10) / 448`.
The maximum permitted relative scale error is four FP32 epsilons.
Scales must be positive.

For each candidate BF16 value `v`, decoded FP8 code `q`, and scale `s`, check:

```text
abs(q*s - v) <= (u8 + 4*FP32_epsilon)*abs(v) + 2^-10*s
```

The final term is half of an E4M3 subnormal step.
The field limits and these encoding checks must all pass.
The BF16 KV comparison still checks the computed values against FP64.

## Probability limits

Output sampling cases keep penalties, masks, and truncation.
MTP draft cases use temperature 1 without top-k, top-p, or min-p truncation.
They keep penalties, masks, and stochastic selection.
The output cases check discontinuous truncation behavior independently of the MTP computation.

| Probability path | Normalized L2 limit | Maximum absolute error and total variation |
|---|---|---|
| BF16 output | `0.03125` | `0.0078125` |
| MTP | `0.5` | `0.125` |

Total variation is half the sum of absolute probability errors in one vocabulary row.
Apply its limit to each row, with each verification row checked independently.
Apply the relative and maximum limits to selected probabilities also.

Smooth MTP probabilities must also stay in a softmax perturbation envelope.
Let `e` be the approved logit absolute-error limit after repetition-penalty sensitivity and temperature scaling.
For reference probability `p`, use:

```text
r = exp(2*e)
lower = p / (p + (1-p)*r)
upper = p*r / (1-p+p*r)
```

Use a tolerance of 32 FP32 epsilons at these endpoints.
The independent probability limits still apply when this envelope is wide.

Probabilities must be nonnegative. Masked token probabilities must be zero.
Each row sum must differ from one by at most 32 FP32 epsilons.
Greedy rows must be one-hot.
The selected probability must equal the probability at the selected token.

## Fixed random selections

Choose seeds from the FP64 distribution before candidate evaluation.
Store these seeds and the expected counter updates in the fixture.
For each stochastic sample, store the CDF endpoints around the selected token.
Use the same reference-selected token when observing candidate endpoints in the native adjusted-score order.
This order includes penalties, masks, and token tie handling.

Set the endpoint-error budget to the smaller of the probability total-variation cap and one third of the reference bin width.
The reference draw's distance from each endpoint must be more than that budget plus 32 FP32 epsilons.
The distance from each positive-mass CDF boundary must also be more than `0.01`.
The candidate must pass the two endpoint-error limits and exact token selection.
This checks a stable subset of the permitted probability errors.
Permitted logit errors can still change token order.

Acceptance fixtures also record distances from acceptance ratios and residual or bonus CDF boundaries.
These fixed binary-fraction probabilities do not depend on upstream projection errors.
Exact token, count, slot, and RNG requirements stay mandatory.

Examine a failed limit before any change to fixtures or policy.
Do not increase limits automatically to accept a candidate.
