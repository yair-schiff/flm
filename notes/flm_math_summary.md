# Mathematical summary for VDM-consistent FLM experiments

## Goal

Instantiate FLM with a VDM-consistent Gaussian corruption on one-hot token representations, train with a CE-based upper bound on NLL, and evaluate with the tighter L2-based VDM bound.

This file is intended as a theory reference for implementation. Keep code changes minimal and preserve the existing FLM time convention.

---

## Time convention

Use the existing FLM time convention:

- \(t=0\): noise
- \(t=1\): clean data

Do **not** flip time in code.

This is equivalent to diffusion-time via the transformation \(s = 1-t\), but we avoid explicit reparameterization to minimize code changes.

Therefore, schedules should be defined so that:

- \(\alpha_t\) increases with \(t\)
- \(\sigma_t\) decreases with \(t\)
- \(\mathrm{SNR}(t)\) increases with \(t\)

Hence:
\[
\mathrm{SNR}'(t) \ge 0.
\]

---

## Data representation

Let the clean token be represented as a one-hot vector:
\[
x \in \Delta^{V-1}.
\]

The denoiser predicts a probability vector:
\[
\hat x_\theta(z_t,t)\in \Delta^{V-1}.
\]

This should come from a softmax over logits.

---

## VDM-consistent corruption under FLM time

Use a Gaussian latent construction:
\[
z_t = \alpha_t x + \sigma_t \epsilon,
\qquad
\epsilon \sim \mathcal N(0,I).
\]

Use a variance-preserving schedule such that
\[
\alpha_t^2 + \sigma_t^2 = 1.
\]

A valid choice is any monotone schedule satisfying:
- \(\alpha_0 \approx 0,\; \sigma_0 \approx 1\)
- \(\alpha_1 \approx 1,\; \sigma_1 \approx 0\)

For example, parameterize through \(\sigma_t\) and set
\[
\alpha_t = \sqrt{1-\sigma_t^2}.
\]

Define
\[
\mathrm{SNR}(t) = \frac{\alpha_t^2}{\sigma_t^2}.
\]

Because of the FLM time convention, \(\mathrm{SNR}(t)\) should be increasing in \(t\).

---

## CE-based NLL upper bound

For one-hot \(x\), define the token-level cross-entropy
\[
\mathrm{CE}(x,\hat x_\theta)
=
-\log \langle x,\hat x_\theta(z_t,t)\rangle.
\]

Under the FLM time convention, the CE-based upper bound is
\[
U_{CE}(\theta)
=
\mathbb E_{x,\epsilon,t}
\left[
\mathrm{SNR}'(t)\,
\mathrm{CE}(x,\hat x_\theta(z_t,t))
\right].
\]

Equivalently,
\[
U_{CE}(\theta)
=
\mathbb E_{x,\epsilon,t}
\left[
\mathrm{SNR}'(t)\,
\bigl(-\log \langle x,\hat x_\theta(z_t,t)\rangle\bigr)
\right].
\]

If time is sampled from density
\[
w(t)\propto \mathrm{SNR}'(t),
\]
then the explicit weighting can be absorbed into time sampling.

So there are two equivalent implementation modes:

1. **uniform time sampling** with explicit weight \(\mathrm{SNR}'(t)\)
2. **SNR-reweighted time sampling** with unweighted CE

These should differ only by a constant factor / reparameterization.

---

## L2-based tighter bound

Evaluate the same denoiser under
\[
U_{L2}(\theta)
=
\mathbb E_{x,\epsilon,t}
\left[
\frac12 \mathrm{SNR}'(t)\,
\|x-\hat x_\theta(z_t,t)\|_2^2
\right].
\]

This is the tighter VDM-style quantity corresponding to the same latent family.

Important:
- this is an evaluation metric even if training uses CE
- CE and L2 must be computed on the same corrupted latent family

---

## Pointwise inequality

For simplex-valued \(x,\hat x_\theta\),
\[
\|x-\hat x_\theta\|_2^2 \le 2\,\mathrm{KL}(x\|\hat x_\theta).
\]

For one-hot \(x\),
\[
\mathrm{KL}(x\|\hat x_\theta)=\mathrm{CE}(x,\hat x_\theta).
\]

Therefore, pointwise,
\[
\frac12 \mathrm{SNR}'(t)\,\|x-\hat x_\theta\|_2^2
\;\le\;
\mathrm{SNR}'(t)\,\mathrm{CE}(x,\hat x_\theta).
\]

Averaging yields
\[
U_{L2}(\theta)\le U_{CE}(\theta).
\]

Thus the CE objective is a valid upper bound on the tighter L2 quantity for the same model and same latent samples.

---

## Slack

Define
\[
\mathrm{slack}=U_{CE}-U_{L2}\ge 0.
\]

This should be logged using:

- the same minibatch
- the same sampled times
- the same sampled Gaussian noise
- the same model outputs

Pointwise slack for one-hot target \(x=e_y\) is
\[
\mathrm{SNR}'(t)
\left[
-\log p_y
-\frac12\|e_y-\hat x_\theta\|_2^2
\right],
\]
where \(p_y\) is the predicted probability of the true token.

Slack should be nonnegative up to numerical error.

---

## Shared population minimizer

For one-hot targets, both CE and L2 have the same Bayes-optimal denoiser:
\[
\hat x^*(z_t,t)=\mathbb E[x\mid z_t,t].
\]

Because \(x\) is one-hot, this is exactly the posterior token probability vector:
\[
\hat x^*(z_t,t)
=
\bigl(
p(x=e_1\mid z_t,t),\dots,p(x=e_V\mid z_t,t)
\bigr).
\]

Therefore:

- CE and L2 share the same population minimizer
- any empirical difference between CE training and L2 training is due to optimization / finite-sample effects, not a different optimum

---

## Experimental hypothesis

Even though \(U_{L2}\) is tighter, CE may optimize better in practice.

So compare:

- **train with CE**, evaluate with CE and L2
- **train with L2**, evaluate with CE and L2

using the same:

- architecture
- optimizer
- data
- tokenization
- corrupted latent family

---

## Important implementation constraints

1. Keep the existing FLM time convention.
2. Do not introduce explicit \(1-t\) reparameterizations unless absolutely necessary.
3. CE and L2 must be computed on the same corrupted latent family.
4. Slack must use the same sampled \(t\) and same sampled \(\epsilon\).
5. Existing FLM behavior should remain available behind config flags.
6. The denoiser output for these experiments must be a probability vector over tokens.

---

## Practical implementation target

We want config-driven support for:

- a VDM-consistent Gaussian interpolant
- CE-based upper-bound training
- L2-based evaluation
- optional L2-based training
- slack logging
- optional SNR-reweighted time sampling

Default behavior of the original repo should remain unchanged.
