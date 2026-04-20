# FLM Checkpoint vs. VDM-Style ELBO: Findings

## Summary

This note summarizes a sequence of experiments run on the released `lm1b_flm.ckpt`
checkpoint in `flm-og` to understand whether we can derive a meaningful
VDM-style CE upper bound / ELBO-like quantity from the existing FLM model.

The main conclusion is:

- The released FLM checkpoint is not zero-shot compatible with a VP-style
  Gaussian latent family.
- The large zero-shot mismatch was mostly caused by changing the corruption law,
  not by changing the loss formula itself.
- Once we restore the original FLM-style noise scale, the checkpoint's
  unweighted CE returns to roughly the native FLM baseline.
- After that, the remaining problem is the weight, not the model.
- The current `snr_prime`-style weighting is too aggressive across the interval
  and especially problematic near both ends, causing the weighted CE to blow up.


## Native Baseline

First, the native FLM branch was restored to its proper behavior by removing an
accidental SNR weighting that had been added on top of the original FLM loss.

Baseline LM1B eval on the released checkpoint:

- `val/nll = 2.136755806431304`
- `val/ppl = 8.471908487070243`

This is the reference point for all comparisons below.


## Gaussian / Gamma Variants Tested

Several VDM-style branches were evaluated to see which, if any, were compatible
enough with the released FLM checkpoint to support a useful zero-shot
upper-bound-style diagnostic.

### 1. `gamma_schedule=linear`

This branch used a normalized Gaussian latent family with:

- `gamma(t)` linear in FM physical time
- `alpha(t) = sqrt(sigmoid(-gamma(t)))`
- `sigma(t) = sqrt(sigmoid(gamma(t)))`

Result:

- `val/ce_unweighted ≈ 9.33`
- `val/ce_upper_bound ≈ 340275.99`

This was far from the native baseline and the weighted objective was unusable.

### 2. `gamma_schedule=flm_matched`

This branch kept the same normalized Gaussian latent family, but defined
`gamma(t)` using the FLM warp shape.

Result:

- `val/ce_unweighted ≈ 8.51`
- `val/ce_upper_bound ≈ 403586.54`

This slightly improved unweighted CE relative to `linear`, but the weighted
objective was even worse.

### 3. `gamma_schedule=old_checkpoint_compatible`

This branch kept:

- the same sampled scalar used by the FLM checkpoint
- the same conditioning scalar `tau`
- the same signal coefficient `alpha = _tau_to_t(tau)`

but still used a VP-style Gaussian latent family:

- `sigma = sqrt(1 - alpha^2)`

Result:

- best `val/ce_unweighted` was around `8.32 - 8.41`
- best weighted values were still around `16.7 - 20.3` on truncated intervals

This showed that matching the time variable helped, but not nearly enough.


## Diagnosing the Large Unweighted Mismatch

The surprising part was that `old_checkpoint_compatible` still had a very large
unweighted gap relative to native FLM:

- native FLM: `val/nll ≈ 2.14`
- old checkpoint compatible Gaussian branch: `val/ce_unweighted ≈ 8.3`

The core issue turned out to be the corruption law.

### Native FLM Corruption

Native FLM uses interpolation-style corruption:

- `x_t = alpha * x + (1 - alpha) * noise`

where `x` is the one-hot token target in vocab space.

### VP-Style Gaussianized Corruption

The Gaussian branch used:

- `z_t = alpha * x + sqrt(1 - alpha^2) * noise`

Both branches share the same signal coefficient `alpha`, but the noise scale is
very different:

- native FLM noise scale: `1 - alpha`
- VP-style Gaussian noise scale: `sqrt(1 - alpha^2)`

For moderate and large `alpha`, `sqrt(1 - alpha^2)` is much larger than
`1 - alpha`, which means the Gaussian branch injects far more noise at the same
conditioning value.

This produces:

- much lower effective SNR
- much harder denoising problem
- strong zero-shot degradation even before weighting

This explains the large unweighted mismatch.


## Diagnostic Branch: `old_checkpoint_flm_noise`

To test whether the corruption mismatch was the real culprit, a new diagnostic
branch was added:

- `gamma_schedule=old_checkpoint_flm_noise`

This branch keeps:

- the old checkpoint's sampled `tau`
- the same `tau` conditioning
- `alpha = _tau_to_t(tau)`

but changes the noise scale to match the FLM-style corruption family:

- `sigma = 1 - alpha`

This branch is not VP-normalized Gaussian VDM, but it is much closer to the
latent family the checkpoint was originally trained on.

### Results

Full interval `[0.01, 0.99]`, uniform sampling:

- `val/ce_unweighted = 2.1045827865600586`
- `val/ce_weighted = 82.37704467773438`
- `val/ce_upper_bound = 82.37716141942067`

Full interval `[0.01, 0.99]`, exact importance sampling with
`q(tau) ∝ snr_prime(tau)`:

- `val/ce_unweighted = 2.017965793609619`
- `val/ce_weighted = 82.42192077636719`
- `val/ce_upper_bound = 82.42198773871056`
- `val/proposal_density_mean = 2.3324882984161377`
- `val/importance_correction_mean = 1.000253438949585`

Trimmed interval `[0.05, 0.90]`:

- `val/ce_unweighted = 2.0561954975128174`
- `val/ce_weighted = 57.49897384643555`
- `val/ce_upper_bound = 57.49895816536096`

### Interpretation

This was the key result.

Once the corruption law was matched more faithfully, the checkpoint's unweighted
CE returned to essentially the native FLM baseline:

- native FLM baseline: `2.1386`
- `old_checkpoint_flm_noise`: `2.06 - 2.11`

So the model itself is not the issue. The giant zero-shot mismatch was mostly
caused by the wrong latent family.

The exact-importance result is also important:

- the weighted bound stayed essentially unchanged (`82.377` vs `82.422`)
- therefore the importance-sampling implementation is behaving correctly
- and the large bound is a property of the target integral itself, not a Monte
  Carlo artifact from uniform sampling


## Interval Sweeps

Before the FLM-noise-compatible branch was introduced, interval sweeps were run
for the older branch to understand whether the bad behavior came mostly from the
endpoints.

### Sweep over `t_max` for `old_checkpoint_compatible`

With `t_min=0.01`:

- `t_max=0.90`
  - `val/ce_unweighted ≈ 8.33`
  - `val/ce_upper_bound ≈ 16.71`
- `t_max=0.95`
  - `val/ce_unweighted ≈ 8.37`
  - `val/ce_upper_bound ≈ 17.66`
- `t_max=0.99`
  - `val/ce_unweighted ≈ 8.41`
  - `val/ce_upper_bound ≈ 20.30`

Reducing `t_max` helped the weighted bound somewhat, but did not fix the
unweighted mismatch.

### Sweep over `t_min` for `old_checkpoint_compatible`

With `t_max=0.90`:

- `t_min=0.001`
  - `val/ce_unweighted ≈ 8.32`
  - `val/ce_upper_bound ≈ 19.21`
- `t_min=0.01`
  - `val/ce_unweighted ≈ 8.33`
  - `val/ce_upper_bound ≈ 16.71`
- `t_min=0.05`
  - `val/ce_unweighted ≈ 8.39`
  - `val/ce_upper_bound ≈ 14.68`

Increasing `t_min` also improved the weighted bound, but again did not remove
the underlying unweighted mismatch.

These sweeps established that truncation could reduce pathology, but could not
explain the full gap.


## The Real Remaining Problem: The Weight

Once `old_checkpoint_flm_noise` restored the unweighted CE to near-baseline,
the remaining problem became very sharp:

- unweighted CE is good
- weighted CE is still huge

For the trimmed interval `[0.05, 0.90]`:

- `val/ce_unweighted = 2.0562`
- `val/ce_weighted = 57.4990`

So the remaining failure mode is not corruption mismatch anymore. It is the
weight itself.

For the FLM-noise-compatible branch:

- `snr = alpha^2 / (1 - alpha)^2`
- `snr_prime = 2 * alpha * alpha_prime / (1 - alpha)^3`

The cubic denominator makes the clean end especially expensive, but it turns
out the objective is not dominated by a single infinitesimal spike only.


## Importance Sampling Interpretation

Exact importance sampling does not change the ELBO-like target. It only changes
how we estimate it.

If the target is

- `E_{tau ~ Uniform}[snr_prime(tau) * CE(tau)]`

and we instead sample from a proposal `q(tau)` with full support and apply the
exact correction factor `1 / q(tau)` (plus the interval-width constant), then
the estimator remains unbiased for the same quantity.

That is exactly what happened here:

- uniform sampling produced `val/ce_upper_bound ≈ 82.377`
- exact importance sampling produced `val/ce_upper_bound ≈ 82.422`

So importance sampling did not "fail". It did what it should:

- preserve the target
- improve the estimator design
- reveal that the true weighted objective is still intrinsically large


## Binning Analysis

To understand where the weighted objective was accumulating mass, binwise
diagnostics were added over the sampled `tau` axis for the full-interval
`old_checkpoint_flm_noise` branch.

For each `tau` bin, the following were logged:

- fraction of samples
- mean `tau`
- mean unweighted CE
- mean `snr_prime`
- mean weighted CE

### Uniform-Sampling Diagnostic

Uniform run `668599`:

- lowest bin, mean `tau ≈ 0.068`
  - `ce ≈ 4.09`
  - `snr_prime ≈ 48.54`
  - `ce_weighted ≈ 212.33`
- middle bins, mean `tau ≈ 0.314 - 0.686`
  - `ce ≈ 1.54 - 2.19`
  - `snr_prime ≈ 21.23 - 27.43`
  - `ce_weighted ≈ 39.36 - 68.53`
- upper-clean bins:
  - mean `tau ≈ 0.811`
    - `ce ≈ 1.44`
    - `snr_prime ≈ 39.96`
    - `ce_weighted ≈ 57.42`
  - highest bin, mean `tau ≈ 0.933`
    - `ce ≈ 1.34`
    - `snr_prime ≈ 129.73`
    - `ce_weighted ≈ 170.69`

### Exact-Importance Diagnostic

Importance-sampled run `668600`:

- lowest bin, mean `tau ≈ 0.052`
  - `ce ≈ 4.44`
  - `snr_prime ≈ 59.53`
  - `ce_weighted ≈ 181.29`
- middle bins, mean `tau ≈ 0.312 - 0.689`
  - `ce ≈ 1.54 - 2.21`
  - `snr_prime ≈ 20.91 - 27.57`
  - `ce_weighted ≈ 62.90 - 90.10`
- upper-clean bins:
  - mean `tau ≈ 0.817`
    - `ce ≈ 1.44`
    - `snr_prime ≈ 41.03`
    - `ce_weighted ≈ 58.65`
  - highest bin, mean `tau ≈ 0.952`
    - `ce ≈ 1.32`
    - `snr_prime ≈ 194.86`
    - `ce_weighted ≈ 53.94`

### Interpretation

The weighted CE is not ruined by one infinitesimal endpoint spike only.

Instead:

- the noisy end contributes because CE itself is large
- the clean end contributes because `snr_prime` is large
- even the middle remains expensive because `snr_prime` never becomes small
- exact importance sampling confirms the same broad pattern rather than changing
  the answer

So the weighted objective is broadly heavy, with especially strong
contributions from both ends of the interval.


## Final Conclusions

1. The released FLM checkpoint is not zero-shot compatible with a VP-style
   normalized Gaussian latent family.
2. The large zero-shot mismatch in the Gaussian branches was mostly caused by
   changing the corruption law, not by a mysterious failure of the checkpoint.
3. Matching the original FLM-style noise scale restores the checkpoint's
   unweighted CE to near-baseline quality.
4. Therefore the checkpoint itself is not the main issue anymore.
5. The remaining bottleneck is the weighting.
6. Exact importance sampling preserves the same large weighted objective, so
   the problem is not estimator bias or poor sampling of the interval.
7. The current `snr_prime`-style weighting is too aggressive across much of the
   interval and especially problematic near both ends.
8. If the goal is to derive an ELBO-like quantity from the released FLM
   checkpoint, the next challenge is to derive or justify a better weight for
   the FLM-style corruption family rather than continuing to import the VDM
   weight unchanged.


## Most Plausible Next Steps

The next useful directions are:

1. Derive the appropriate continuous-time bound directly for the FLM-style
   interpolation corruption family.
2. As diagnostics, test softened or clipped versions of the weight to
   understand how sensitive the objective is to endpoint amplification.
3. If a practical zero-shot diagnostic is needed immediately, use the
   unweighted CE under `old_checkpoint_flm_noise` as a corruption-compatible
   measure, while treating the current weighted objective as numerically
   unstable / poorly calibrated for this checkpoint.
