# FLM / FLMVDM Time-Warping Notes

This note explains how the time-warping code in this repo relates to Equation 25 in the FLM paper and how the newer VP-native `FLMVDM` path differs from the legacy FLM path.

---

## 1. The core quantity from the paper

Equation 25 in the FLM paper standardizes decoding progress by converting the decoding error probability \(P_e\) into a unit-interval time:

$$
\tau \;=\; 1 - \frac{P_e}{1 - 1/|V|}.
$$

Here \(|V|\) denotes the vocabulary size, i.e. the number of token classes over which the argmax decoder competes.

Equivalently, if

$$
q_c = \Pr(\text{argmax decoder returns the correct token}),
$$

then \(P_e = 1 - q_c\), so

$$
\tau \;=\; \frac{q_c - 1/|V|}{1 - 1/|V|}.
$$

That is the form used throughout the code.

---

## 2. Legacy FLM: \(\tau(t)\)

For the original FLM Gaussian interpolation, the latent is

$$
x_t = (1-t)\,\epsilon + t\,e_y,
\qquad
t \in [0,1],
$$

where \(e_y\) is the one-hot clean token and \(\epsilon \sim \mathcal N(0, I)\).

For the correct coordinate \(c\) and an incorrect coordinate \(u\),

$$
x_{t,c} = t + (1-t)\epsilon_c,
\qquad
x_{t,u} = (1-t)\epsilon_u.
$$

So the standardized correct-vs-incorrect margin is

$$
m_c(t) = \frac{t}{1-t}.
$$

To see where the margin comes from, compare the correct coordinate to one incorrect coordinate:

$$
x_{t,c} > x_{t,u}
\quad\Longleftrightarrow\quad
t + (1-t)\epsilon_c > (1-t)\epsilon_u.
$$

Since \(1-t > 0\) for interior points,

$$
\epsilon_u < \epsilon_c + \frac{t}{1-t}.
$$

So if we condition on the correct-coordinate noise value \(\epsilon_c = z\), then each incorrect token is beaten with probability

$$
\Phi\!\left(z + \frac{t}{1-t}\right).
$$

There are \(|V|-1\) incorrect tokens. Conditional on \(\epsilon_c = z\), the noises \(\epsilon_u\) for those incorrect coordinates are i.i.d., so the probability that **all** of them are below the correct coordinate is

$$
\Phi\!\left(z + \frac{t}{1-t}\right)^{|V|-1}.
$$

Averaging over the random correct-coordinate noise \(Z \sim \mathcal N(0,1)\) gives the exact correct-decoding probability:

$$
q_c(t)
=
\mathbb E_Z\left[
\Phi\!\left(Z + \frac{t}{1-t}\right)^{|V|-1}
\right].
$$

Then Equation 25 becomes

$$
\tau(t)
=
\frac{q_c(t) - 1/|V|}{1 - 1/|V|}.
$$

### Where this lives in code

The quantity we need in order to build the warp is the scalar function

$$
t \mapsto \tau(t).
$$

That is not evaluated analytically in closed form in the code. Instead, the repo first computes

$$
t \mapsto q_c(t),
$$

then converts it to \(\tau(t)\), then builds lookup tables from sampled values on a dense grid.

The core legacy implementation is in `utils.py`:

```python
def compute_alpha_exact(gamma, K, ...):
    sigma = 1.0 - gamma
    m_c = gamma / sigma
    ...
    q_c = ...
    alpha = K / (K - 1.0) * (q_c - 1.0 / K)
```

Important naming mismatch:

- legacy `gamma` in that function is really paper \(t\)
- legacy `alpha` is really paper \(\tau\)

More explicitly, the function does the following:

1. Build the standardized margin:

   $$
   m_c(t) = \frac{t}{1-t}.
   $$

2. Rewrite the Gaussian expectation over \(Z \sim \mathcal N(0,1)\) as a form suitable for Gauss-Hermite quadrature.

3. Approximate

   $$
   q_c(t)
   =
   \mathbb E_Z\left[
   \Phi\!\left(Z + m_c(t)\right)^{|V|-1}
   \right]
   $$

   by a finite weighted sum over quadrature nodes.

4. Standardize the resulting \(q_c(t)\) into \(\tau(t)\) via Equation 25.

In code, the quadrature step is the part that uses:

```python
x, w = hermgauss(n_gh)
z_nodes = np.sqrt(2.0) * x
L_cu = log_ndtr(z_expanded + m_c_expanded)
log_prod_c = (K - 1) * L_cu
q_c = np.sum(w * np.exp(log_prod_c), axis=-1)
```

This is a Gauss-Hermite approximation to the exact Gaussian integral above.

#### How the lookup table is built

After `compute_alpha_exact` can evaluate \(\tau(t)\) on any requested grid, the repo tabulates it on a dense uniform grid of \(t\)-values:

```python
gamma_vals = np.linspace(0.0, 1.0, n_points)   # really t-values
alpha_vals = compute_alpha_exact(gamma_vals, K=K)   # really tau(t)
```

Then it fits two cubic splines:

```python
def build_luts(K, n_points=10000):
    gamma_vals = np.linspace(0.0, 1.0, n_points)
    alpha_vals = compute_alpha_exact(gamma_vals, K=K)
    lut_g2a = CubicSpline(gamma_vals, alpha_vals)   # t -> tau
    lut_a2g = CubicSpline(unique_alpha, unique_gamma)  # tau -> t
```

So legacy FLM uses:

- `lut_g2a`: \(t \mapsto \tau\)
- `lut_a2g`: \(\tau \mapsto t\)

In other words, the runtime warp is not recomputing the Gaussian integral over and over. It precomputes the curve once, then uses spline interpolation during training and sampling.

#### Why the method is smooth / differentiable

There are two layers here:

1. **Mathematical smoothness.**
   The function

   $$
   q_c(t)=\mathbb E_Z\left[\Phi(Z+m_c(t))^{|V|-1}\right]
   $$

   is a smooth function of \(t\) on the interior \(t \in (0,1)\), because it is built from smooth operations: addition, division away from the endpoint, Gaussian CDFs, exponentiation, and expectation.

2. **Implementation of derivatives.**
   The code does **not** backpropagate through the SciPy quadrature or spline fitting. Instead, it fits a cubic spline and then explicitly differentiates that spline when it needs a derivative such as \(dt/d\tau\). So the runtime derivative comes from the spline object itself, via `lut.derivative()`.

So in practice the method is "differentiable" in the sense that the warp is represented by a smooth spline with an explicit derivative, not in the sense of using end-to-end autograd through the quadrature computation.

---

## 3. VP-native FLMVDM: \(\tau(\gamma)\)

The refactored `FLMVDM` path no longer uses a hidden physical FLM time internally. It is now:

1. sample \(\tau\) uniformly
2. map \(\tau \mapsto \gamma\) with a VP lookup table
3. corrupt with VP coefficients derived from \(\gamma\)

The corruption is

$$
z_\tau = \alpha(\gamma(\tau))\,x + \sigma(\gamma(\tau))\,\epsilon,
$$

with the VP parameterization

$$
\alpha(\gamma) = \sqrt{\sigma(-\gamma)},
\qquad
\sigma(\gamma) = \sqrt{\sigma(\gamma)},
\qquad
\mathrm{SNR}(\gamma) = \frac{\alpha(\gamma)^2}{\sigma(\gamma)^2} = e^{-\gamma}.
$$

So the standardized margin becomes

$$
m_c(\gamma)
=
\frac{\alpha(\gamma)}{\sigma(\gamma)}
=
\sqrt{\mathrm{SNR}(\gamma)}
=
e^{-\gamma/2}.
$$

Then the VP analogue of the paper's decoding-progress curve is

$$
q_c(\gamma)
=
\mathbb E_Z\left[
\Phi\!\left(Z + e^{-\gamma/2}\right)^{|V|-1}
\right],
$$

and

$$
\tau(\gamma)
=
\frac{q_c(\gamma) - 1/|V|}{1 - 1/|V|}.
$$

This is implemented in `utils.py`:

```python
def compute_vp_tau_exact(gamma, K, ...):
    m_c = np.exp(-0.5 * gamma)
    ...
    q_c = ...
    tau = K / (K - 1.0) * (q_c - 1.0 / K)
```

This is structurally the same procedure as legacy FLM:

1. define the correct-vs-incorrect margin
2. compute \(q_c\) by Gauss-Hermite quadrature
3. standardize \(q_c\) into \(\tau\)
4. tabulate the result on a dense grid
5. build splines for the forward and inverse maps

### VP lookup tables

The VP-native LUT builder is:

```python
def build_vp_luts(K, gamma_min, gamma_max, n_points=10000):
    gamma_vals = np.linspace(gamma_min, gamma_max, n_points)
    tau_vals = compute_vp_tau_exact(gamma_vals, K=K)
    lut_gamma2tau = CubicSpline(gamma_vals, tau_vals)
    lut_tau2gamma = CubicSpline(tau_augmented, gamma_augmented)
```

So the refactored VP path uses:

- `lut_gamma2tau`: \(\gamma \mapsto \tau\)
- `lut_tau2gamma`: \(\tau \mapsto \gamma\)

The same spline logic also gives an explicit derivative:

- `d_tau_to_gamma(...)` evaluates \(d\gamma / d\tau\) by differentiating the inverse spline

This is the quantity used later in the VP loss weight.

The inverse LUT is endpoint-pinned so that:

$$
\tau = 0 \mapsto \gamma_{\max},
\qquad
\tau = 1 \mapsto \gamma_{\min}.
$$

That avoids spline extrapolation at the exact endpoints.

---

## 4. How training uses the VP warp

In `algo.py`, `FLMVDM` now samples \(\tau\) directly:

```python
def _sample_tau_coordinates(self, B, accum_step):
    tau_t = self._sample_t_interval(...)
    gamma = self._tau_to_gamma(tau_t)
    dgamma_dtau = self._d_gamma_by_d_tau(tau_t)
    return tau_t, gamma, dgamma_dtau
```

Then corruption is performed from \(\gamma\), not from a hidden physical \(t\):

```python
alpha = torch.sigmoid(-gamma).sqrt()
sigma = torch.sigmoid(gamma).sqrt()
x_t = alpha * target_data + sigma * noise
```

The Jacobian-aware weight is

$$
\frac{d\,\mathrm{SNR}}{d\tau}
=
\frac{d\,e^{-\gamma}}{d\tau}
=
-e^{-\gamma}\frac{d\gamma}{d\tau}.
$$

That is exactly what the code uses:

```python
snr_prime_tau = torch.exp(-gamma) * (-dgamma_dtau)
loss_weight = snr_prime_tau
```

So the current `FLMVDM` training objective is a VP-native weighted surrogate expressed in uniformly sampled \(\tau\).

---

## 5. Interpreting `gamma_min` and `gamma_max`

In the refactored VP path:

- `gamma_max` is the noisy endpoint
- `gamma_min` is the clean endpoint

because

$$
\mathrm{SNR}(\gamma) = e^{-\gamma}.
$$

So:

- larger positive \(\gamma\) means lower SNR and more noise
- more negative \(\gamma\) means higher SNR and cleaner latents

With endpoint pinning, the VP warp satisfies:

$$
\tau = 0 \Rightarrow \gamma = \gamma_{\max},
\qquad
\tau = 1 \Rightarrow \gamma = \gamma_{\min}.
$$

Numerically, they set the range of VP noise levels the model will ever see.

---

## 6. The analogue of \(P_e\) in both worlds

In both FLM and VP/VDM, the decoding error probability is

$$
P_e = 1 - q_c.
$$

Once \(q_c\) is known, the standardized-progress relation is linear:

$$
\tau = 1 - \frac{P_e}{1 - 1/K}
\qquad\Longleftrightarrow\qquad
P_e = (1 - 1/|V|)(1 - \tau).
$$

So:

- in legacy FLM, the repo visualizes \(P_e(t)\)
- in VP/VDM, the natural analogue is \(P_e(\gamma)\) or \(P_e(\mathrm{SNR})\)

The main combined explorer `scripts/flm_vdm_schedule_explorer.py` now includes a
dedicated tau/error comparison tab. Those views plot:

- \(\tau(t)\) and \(t(\tau)\) for legacy FLM
- \(\tau(\mathrm{SNR})\), \(\mathrm{SNR}(\tau)\), \(\tau(\gamma)\), and \(\gamma(\tau)\) for VP
- \(P_e\) and \(q_c\) in both parameterizations

---

## 7. Practical summary

There are really two different warps in the repo now:

### Legacy FLM

```text
t  -->  q_c(t)  -->  tau(t)
```

and the inverse LUT gives:

```text
tau  -->  t
```

### VP-native FLMVDM

```text
gamma  -->  q_c(gamma)  -->  tau(gamma)
```

and the inverse LUT gives:

```text
tau  -->  gamma  -->  (alpha, sigma, SNR)
```

That is the main conceptual shift introduced by the refactor.

---

## 8. Main files to read

- `utils.py`
  - `compute_alpha_exact`
  - `build_luts`
  - `compute_vp_tau_exact`
  - `build_vp_luts`
  - `tau_to_gamma`
  - `gamma_to_tau`
  - `d_tau_to_gamma`

- `algo.py`
  - `FLMVDM.__init__`
  - `_sample_tau_coordinates`
  - `corrupt_continuous`
  - `generate_samples`

- `scripts/flm_vdm_schedule_explorer.py`
  - main combined Streamlit explorer for schedule, warp, and tau/error views

- `scripts/flm_vdm_tau_error_explorer.py`
  - focused companion app for just the tau/error comparison views
