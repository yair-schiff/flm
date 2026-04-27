## Eval Launch Runbook

This note exists to avoid repeating the same eval-launch mistakes.

### Safe default for LM1B checkpoint evals

Use these defaults unless there is a specific reason not to:

- `--partition=kuleshov`
- `--constraint=a5000`
- `--ntasks-per-node=1`
- `--gres=gpu:1`
- `loader.batch_size=8`
- `loader.eval_batch_size=8`
- `loader.num_workers=0`
- `sampling.num_sample_batches=0`
- `eval.generate_samples=false`
- `eval.compute_generative_perplexity=false`

### Why

- The checked-in 4-GPU / batch-64 eval setup is too memory-heavy for these
  checkpoint evaluations and can OOM immediately.
- If we only want perplexity / CE-style validation metrics, sample generation
  and generative perplexity should stay off.
- One-GPU evals are simpler, cheaper, and sufficient for comparing
  `gamma_schedule`, `time_sampling`, and weighting variants.

### Before launching a new sweep

Check these first:

- Are `eval.generate_samples=false` and
  `eval.compute_generative_perplexity=false` set?
- Is `sampling.num_sample_batches=0` set?
- Is the eval batch size still `8` or smaller?
- Are we on a single GPU rather than the older 4-GPU script defaults?

If a run still OOMs after that, reduce `loader.eval_batch_size` from `8` to `4`
before changing anything else.
