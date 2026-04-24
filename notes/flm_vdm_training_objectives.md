# FLMVDM Training Objective Goals

We want FLMVDM to train from useful language noise levels without making likelihood diagnostics unreadable.

## Problem

The exact tau-sampled VLB estimator uses `exp(-gamma) * (-dgamma/dtau)`. This is principled, but the weights can become huge and make training or evaluation unstable and hard to compare.

## Lessons

- [VDM](https://papers.nips.cc/paper/2021/file/b578f2a52a0229873fefc2a4b06377fa-Paper.pdf): the continuous VLB is an SNR integral, and tau sampling needs a Jacobian correction.
- [FLM](https://arxiv.org/html/2602.16813v2): language benefits from sampling noise levels near meaningful decoding progress.
- [LangFlow](https://arxiv.org/html/2604.11748v3): continuous language models improve when the noise schedule or proposal follows information gain.

## Direction

Separate the training proposal from the evaluation target:

- sample training noise from an information-shaped proposal over `gamma`;
- support a bounded-VLB mode with importance weights capped by construction;
- support an unweighted proposal-CE mode as a generation-first ablation;
- always log explicit VLB diagnostics separately from training-loss diagnostics.
