# Goal

Modify the FLM codebase to test the following hypothesis:

1. Instantiate FLM with a **VDM-consistent Gaussian interpolant** applied to one-hot token representations.
2. Train the denoiser with a **cross-entropy objective** corresponding to an **upper bound on negative log-likelihood (NLL)**.
3. Evaluate the same model with the tighter **L2-based VDM upper bound**.
4. Compare optimization behavior between CE training and L2 training.

---

# Required workflow (IMPORTANT)

Before making any code changes:

1. Inspect the repository structure.
2. Identify:
   - interpolant / corruption implementation
   - denoiser output head
   - training loss
   - time sampling
   - validation / logging
   - checkpoint selection
3. Propose a minimal implementation plan:
   - files to edit
   - config additions
   - how Gaussian corruption will be inserted
   - how CE and L2 will be computed on identical samples
   - how validation NLL/PPL will be reported
4. Only after presenting this plan should code changes begin.

---

# Time convention

Use the existing FLM time convention unchanged:

- t = 0: noise  
- t = 1: clean data  

Do NOT introduce explicit (1 - t) reparameterizations unless absolutely necessary.

All schedules (alpha_t, sigma_t, SNR(t)) should be defined consistently with this convention.

---

# High-level theory to preserve

We instantiate FLM so that its latent family matches a VDM-style Gaussian corruption on one-hot vectors.

Clean token:
x ∈ Δ^(V-1) (one-hot)

Noisy latent:
z_t = alpha_t * x + sigma_t * epsilon  
epsilon ~ N(0, I)

Variance-preserving schedule:
alpha_t^2 + sigma_t^2 = 1

SNR:
SNR(t) = alpha_t^2 / sigma_t^2, with SNR'(t) ≥ 0

Denoiser output:
x_hat_theta(z_t, t) ∈ Δ^(V-1)

---

# Required code changes

## 1. Add Gaussian interpolant mode

Add config option:
interpolant_type = "vdm_gaussian"

Behavior:
- represent tokens as one-hot
- sample Gaussian noise
- compute:
  z_t = alpha_t * x + sigma_t * epsilon

Keep existing FLM interpolants unchanged.

---

## 2. Add SNR schedule utilities

Implement:
- snr(t)
- snr_prime(t)
- alpha(t)
- sigma(t)

Constraints:
- monotone increasing SNR
- consistent with variance-preserving constraint

---

## 3. Time sampling modes

Add config:
time_sampling = "uniform" | "snr_reweighted"

- uniform:
  - sample t ~ Uniform(0,1)
  - apply weight SNR'(t)

- snr_reweighted:
  - sample t proportional to SNR'(t)
  - no explicit weighting in loss

These should be equivalent up to a constant.

---

## 4. Denoiser output

Ensure model can return:
- logits
- softmax probabilities

Needed for:
- CE training
- L2 evaluation

---

## 5. CE-based training objective

For one-hot target x = e_y:

CE = -log p_theta(y | z_t, t)

Training loss:

Uniform sampling:
E[SNR'(t) * CE]

SNR-reweighted sampling:
E[CE]

Config:
train_loss = "ce_upper_bound"

---

## 6. L2-based objective

Compute:
||x - x_hat_theta(z_t, t)||^2

Evaluation:
U_L2 = E[ (1/2) * SNR'(t) * ||x - x_hat_theta||^2 ]

Optional training mode:
train_loss = "l2_vdm"

---

## 7. Slack computation

Compute on the same minibatch:

slack = U_CE - U_L2 ≥ 0

Must use:
- same t
- same noise
- same latent

We can use this as a sanity check that everything looks good.
But we do not need to report / log this in actual experimenation. 

---

## 8. Validation metrics and reporting

### Primary metrics
- val/ce_upper_bound
- val/l2_bound

### NLL upper bounds
- val/nll_upper_ce = val/ce_upper_bound
- val/nll_upper_l2 = val/l2_bound

### Perplexity upper bounds

If loss is averaged per token:

PPL = exp(NLL)

Log:
- val/ppl_upper_ce
- val/ppl_upper_l2

These must be clearly labeled as upper bounds.

---

## 9. Checkpoint selection

Default:
best_metric = val/l2_bound

Allow alternatives:
- val/ce_upper_bound

---

## 10. Experiments

### Experiment A
- Gaussian interpolant
- CE training

### Experiment B
- Gaussian interpolant
- L2 training

### Experiment C
- Native FLM baseline

Keep all else fixed:
- architecture
- optimizer
- data
- tokenization

---

## 11. Logging

Log:
- CE loss
- L2 loss
- SNR stats
- probability of true token

---

## 12. Correctness constraints

1. Same latent family for CE and L2
2. Same sampled t and noise for comparisons
3. Per-token averaging before computing PPL
4. No changes to default repo behavior
5. Safe log clipping
6. Config-driven behavior only

---

## 13. Sanity checks

Verify:

- CE ≥ L2
- slack ≥ 0
- training is stable

---

## 14. Deliverables

After implementation:

1. List of modified files
2. Explanation of changes
3. Config flags added
4. Example commands:
   - CE training
   - L2 training
   - evaluation
5. Validation metric outputs
6. Confirmation sanity checks passed

---

## Coding style

- Minimal edits
- No large refactors
- Reuse existing code paths
- Add concise comments where needed
- Preserve backward compatibility
