import json
import os
import math
import itertools
import functools
import argparse
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
torch.load = functools.partial(torch.load, weights_only=False)
from torch.distributed import init_process_group, destroy_process_group
import wandb
import algo
import dataloader
import utils

import numpy as np
from datetime import datetime

import uuid

# Allow torch.load(weights_only=True) to safely unpickle Hydra configs stored in checkpoints
torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig, omegaconf.base.ContainerMetadata, omegaconf.base.Metadata])

omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
    if 'hf' in config.algo.backbone:
        return diffusion_model(
            config, tokenizer=tokenizer).to('cuda')

    return diffusion_model.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        weights_only=False)


@L.pytorch.utilities.rank_zero_only
def _print_config(
        config: omegaconf.DictConfig,
        resolve: bool = True,
        save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            '{}/config_tree.txt'.format(
                config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [
            ('train', train_ds), ('valid', valid_ds)]:
        print(f'Printing {dl_type} dataloader batch.')
        batch = next(iter(dl))
        print('Batch input_ids.shape', batch['input_ids'].shape)
        first = batch['input_ids'][0, :k]
        last = batch['input_ids'][0, -k:]
        print(f'First {k} tokens:', tokenizer.decode(first))
        print('ids:', first)
        print(f'Last {k} tokens:', tokenizer.decode(last))
        print('ids:', last)


def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
    logger.info('Starting Sample Eval.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    all_samples = []

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps)
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            samples = model.restore_model_and_sample(
                num_steps=config.sampling.steps)
            model.metrics.record_entropy(samples)
            text_samples = model.tokenizer.batch_decode(samples)
            model.metrics.record_generative_perplexity(
                text_samples, config.model.length, model.device)
            all_samples.extend(list(text_samples))

    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    generative_ppl = 0.
    entropy = 0.
    if not config.sampling.semi_ar:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        print('Generative perplexity:', generative_ppl)
        print('Sample entropy:', entropy)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, 'w') as f:
        json.dump({'generative_ppl': generative_ppl,
                   'entropy': entropy,
                   'generated_seqs': all_samples}, f, indent=4)
    print('Samples saved at:', samples_path)


def _generate_samples_with_tc(diffusion_model, config, logger,
                              tokenizer):
    logger.info('Starting Sample Eval.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    all_samples = []

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for i in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps)
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            assert config.loader.eval_batch_size % config.sampling.duplicate == 0
            different_in_batch = config.loader.eval_batch_size // config.sampling.duplicate
            samples = model.restore_model_and_sample(
                num_steps=config.sampling.steps, duplicate=config.sampling.duplicate)
            model.metrics.record_entropy(samples)
            text_samples = model.tokenizer.batch_decode(samples)
            model.metrics.record_generative_perplexity(
                text_samples, config.model.length, model.device)
            model.metrics.record_tc([i*different_in_batch + j for _ in range(
                config.sampling.duplicate) for j in range(different_in_batch)], samples)
            all_samples.extend(list(text_samples))

    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    generative_ppl = 0.
    entropy = 0.
    if not config.sampling.semi_ar:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        avg_tc, avg_joints, avg_marginals = model.metrics.tc.compute()
        print('Generative perplexity:', generative_ppl)
        print('Sample entropy:', entropy)
        print('Total average correlation:', avg_tc)
        print('Average joint entropy:', avg_joints)
        print('Average marginal entropy:', avg_marginals)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, 'w') as f:
        json.dump({'generative_ppl': generative_ppl,
                   'entropy': entropy,
                   'avg_tc': avg_tc,
                   'avg_joints': avg_joints,
                   'avg_marginals': avg_marginals,
                   'generated_seqs': all_samples}, f, indent=4)
    print('Samples saved at:', samples_path)


@torch.inference_mode()
def generate_reflow_dataset(diffusion_model, config, logger, tokenizer):
    # TODO: implement with lightning_module.test with pseudo-data
    logger.info('Generating samples.')
    model = _load_from_checkpoint(diffusion_model=diffusion_model,
                                  config=config,
                                  tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    # if model.ema:
    #   model.ema.store(itertools.chain(
    #       model.backbone.parameters(),
    #       model.noise.parameters()))
    #   model.ema.copy_to(itertools.chain(
    #       model.backbone.parameters(),
    #       model.noise.parameters()))
    #   model.backbone.eval()
    #   model.noise.eval()

    test_ds = dataloader.get_pseudo_dataloader(config, tokenizer, model)
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=None,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=None)
    trainer.test(model, test_ds)
    return


def _eval_ppl(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Perplexity Eval.')

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed)
    trainer.validate(model, valid_ds)


def _eval_continuous_likelihood(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Continuous Likelihood Eval.')

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed)

    likelihood_cfg = config.eval
    all_log_probs = []
    all_nlls = []
    all_base_log_probs = []
    all_divergence_integrals = []
    all_token_counts = []
    num_batches = 0
    ambient_dim = int(model.num_tokens * model.vocab_size)
    bits_per_nat = 1.0 / math.log(2.0)

    model._eval_mode()
    try:
        for batch_idx, batch in enumerate(valid_ds):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            token_counts = attention_mask.sum(dim=1).to(torch.float32)

            details = model.continuous_log_likelihood(
                input_ids,
                num_steps=likelihood_cfg.likelihood_num_steps,
                endpoint_eps=likelihood_cfg.likelihood_endpoint_eps,
                trace_method=likelihood_cfg.likelihood_trace_method,
                trace_samples=likelihood_cfg.likelihood_trace_samples,
                solver=likelihood_cfg.likelihood_solver,
                noise=likelihood_cfg.likelihood_noise,
                seed=likelihood_cfg.likelihood_seed + batch_idx,
                return_details=True,
            )

            all_log_probs.append(details['log_prob'].detach().cpu())
            all_nlls.append(details['nll'].detach().cpu())
            all_base_log_probs.append(details['base_log_prob'].detach().cpu())
            all_divergence_integrals.append(
                details['divergence_integral'].detach().cpu())
            all_token_counts.append(token_counts.detach().cpu())
            num_batches += 1

            batch_mean_log_prob = details['log_prob'].mean().item()
            batch_mean_nll = details['nll'].mean().item()
            batch_mean_token_nll = (
                details['nll'] / token_counts.clamp_min(1.0)
            ).mean().item()
            batch_mean_base_log_prob = (
                details['base_log_prob'].mean().item())
            batch_mean_divergence_integral = (
                details['divergence_integral'].mean().item())
            batch_mean_log_prob_per_dim = (
                batch_mean_log_prob / ambient_dim)
            batch_mean_nll_per_dim = batch_mean_nll / ambient_dim
            batch_mean_bits_per_dim = (
                batch_mean_nll_per_dim * bits_per_nat)
            batch_mean_continuous_pseudo_ppl_per_dim = math.exp(
                batch_mean_nll_per_dim)
            batch_mean_base_log_prob_per_dim = (
                batch_mean_base_log_prob / ambient_dim)
            batch_mean_divergence_integral_per_dim = (
                batch_mean_divergence_integral / ambient_dim)
            print(
                f"Processed likelihood batch {batch_idx + 1}: "
                f"mean log_prob={batch_mean_log_prob:.4f}, "
                f"mean log_prob/dim={batch_mean_log_prob_per_dim:.6f}, "
                f"mean nll={batch_mean_nll:.4f}, "
                f"mean nll/dim={batch_mean_nll_per_dim:.6f}, "
                f"mean bits/dim={batch_mean_bits_per_dim:.6f}, "
                "mean continuous_pseudo_ppl/dim="
                f"{batch_mean_continuous_pseudo_ppl_per_dim:.6f}, "
                f"mean token_nll={batch_mean_token_nll:.4f}, "
                f"mean base_log_prob={batch_mean_base_log_prob:.4f}, "
                f"mean base_log_prob/dim="
                f"{batch_mean_base_log_prob_per_dim:.6f}, "
                f"mean divergence_integral={batch_mean_divergence_integral:.4f}, "
                f"mean divergence_integral/dim="
                f"{batch_mean_divergence_integral_per_dim:.6f}"
            )
    finally:
        model._train_mode()

    log_probs = torch.cat(all_log_probs, dim=0)
    nlls = torch.cat(all_nlls, dim=0)
    base_log_probs = torch.cat(all_base_log_probs, dim=0)
    divergence_integrals = torch.cat(all_divergence_integrals, dim=0)
    token_counts = torch.cat(all_token_counts, dim=0).clamp_min(1.0)
    token_nlls = nlls / token_counts
    log_probs_per_dim = log_probs / ambient_dim
    nlls_per_dim = nlls / ambient_dim
    bits_per_dim = nlls_per_dim * bits_per_nat
    continuous_pseudo_ppl_per_dim = torch.exp(nlls_per_dim)
    base_log_probs_per_dim = base_log_probs / ambient_dim
    divergence_integrals_per_dim = divergence_integrals / ambient_dim

    payload = {
        'density_type': 'continuous_log_density',
        'comparison_note': (
            'Continuous density in ambient one-hot space. '
            'Not directly comparable to discrete language-model perplexity. '
            'continuous_pseudo_ppl_per_dim is exp(nll_per_dim), useful only '
            'for comparing continuous-density runs to each other.'
        ),
        'ambient_dim': ambient_dim,
        'num_batches': num_batches,
        'num_sequences': int(log_probs.numel()),
        'total_tokens': float(token_counts.sum().item()),
        'mean_log_prob': float(log_probs.mean().item()),
        'mean_nll': float(nlls.mean().item()),
        'mean_log_prob_per_dim': float(log_probs_per_dim.mean().item()),
        'mean_nll_per_dim': float(nlls_per_dim.mean().item()),
        'mean_bits_per_dim': float(bits_per_dim.mean().item()),
        'mean_continuous_pseudo_ppl_per_dim': float(
            continuous_pseudo_ppl_per_dim.mean().item()),
        'mean_token_nll': float(token_nlls.mean().item()),
        'mean_base_log_prob': float(base_log_probs.mean().item()),
        'mean_base_log_prob_per_dim': float(
            base_log_probs_per_dim.mean().item()),
        'mean_divergence_integral': float(
            divergence_integrals.mean().item()),
        'mean_divergence_integral_per_dim': float(
            divergence_integrals_per_dim.mean().item()),
        'trace_method': likelihood_cfg.likelihood_trace_method,
        'trace_samples': int(likelihood_cfg.likelihood_trace_samples),
        'solver': likelihood_cfg.likelihood_solver,
        'endpoint_eps': float(likelihood_cfg.likelihood_endpoint_eps),
        'sequence_log_probs': log_probs.tolist(),
        'sequence_nlls': nlls.tolist(),
        'sequence_log_probs_per_dim': log_probs_per_dim.tolist(),
        'sequence_nlls_per_dim': nlls_per_dim.tolist(),
        'sequence_bits_per_dim': bits_per_dim.tolist(),
        'sequence_continuous_pseudo_ppl_per_dim': (
            continuous_pseudo_ppl_per_dim.tolist()),
        'sequence_token_counts': token_counts.tolist(),
        'sequence_token_nlls': token_nlls.tolist(),
        'sequence_base_log_probs': base_log_probs.tolist(),
        'sequence_base_log_probs_per_dim': base_log_probs_per_dim.tolist(),
        'sequence_divergence_integrals': divergence_integrals.tolist(),
        'sequence_divergence_integrals_per_dim': (
            divergence_integrals_per_dim.tolist()),
    }

    output_path = likelihood_cfg.likelihood_output_path
    with fsspec.open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print('Likelihood summary:')
    print(f"  sequences: {payload['num_sequences']}")
    print(f"  total_tokens: {payload['total_tokens']}")
    print(f"  ambient_dim: {payload['ambient_dim']}")
    print(f"  mean_log_prob: {payload['mean_log_prob']:.6f}")
    print(
        f"  mean_log_prob_per_dim: "
        f"{payload['mean_log_prob_per_dim']:.6f}"
    )
    print(f"  mean_nll: {payload['mean_nll']:.6f}")
    print(f"  mean_nll_per_dim: {payload['mean_nll_per_dim']:.6f}")
    print(f"  mean_bits_per_dim: {payload['mean_bits_per_dim']:.6f}")
    print(
        "  mean_continuous_pseudo_ppl_per_dim: "
        f"{payload['mean_continuous_pseudo_ppl_per_dim']:.6f}"
    )
    print(f"  mean_token_nll: {payload['mean_token_nll']:.6f}")
    print(f"  mean_base_log_prob: {payload['mean_base_log_prob']:.6f}")
    print(
        f"  mean_base_log_prob_per_dim: "
        f"{payload['mean_base_log_prob_per_dim']:.6f}"
    )
    print(
        "  mean_divergence_integral: "
        f"{payload['mean_divergence_integral']:.6f}"
    )
    print(
        "  mean_divergence_integral_per_dim: "
        f"{payload['mean_divergence_integral_per_dim']:.6f}"
    )


def _eval_discrete_elbo(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Discrete ELBO Eval.')

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed)

    elbo_cfg = config.eval
    all_elbos = []
    all_sample_elbos = []
    all_iw_elbos = []
    all_nll_upper_bounds = []
    all_sample_nll_upper_bounds = []
    all_iw_nll_upper_bounds = []
    all_continuous_log_probs = []
    all_decoder_log_probs = []
    all_encoder_log_probs = []
    all_encoder_entropies = []
    all_token_counts = []
    num_batches = 0

    model._eval_mode()
    try:
        for batch_idx, batch in enumerate(valid_ds):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            token_counts = attention_mask.sum(dim=1).to(torch.float32)

            details = model.discrete_elbo(
                input_ids,
                mc_samples=elbo_cfg.elbo_mc_samples,
                num_steps=elbo_cfg.elbo_num_steps,
                endpoint_eps=elbo_cfg.elbo_endpoint_eps,
                trace_method=elbo_cfg.elbo_trace_method,
                trace_samples=elbo_cfg.elbo_trace_samples,
                solver=elbo_cfg.elbo_solver,
                noise=elbo_cfg.elbo_noise,
                seed=elbo_cfg.elbo_seed + batch_idx,
                return_details=True,
            )

            all_elbos.append(details['elbo'].detach().cpu())
            all_sample_elbos.append(details['sample_elbo'].detach().cpu())
            all_iw_elbos.append(details['iw_elbo'].detach().cpu())
            all_nll_upper_bounds.append(
                details['nll_upper_bound'].detach().cpu())
            all_sample_nll_upper_bounds.append(
                details['sample_nll_upper_bound'].detach().cpu())
            all_iw_nll_upper_bounds.append(
                details['iw_nll_upper_bound'].detach().cpu())
            all_continuous_log_probs.append(
                details['continuous_log_prob'].detach().cpu())
            all_decoder_log_probs.append(
                details['decoder_log_prob'].detach().cpu())
            all_encoder_log_probs.append(
                details['encoder_log_prob'].detach().cpu())
            all_encoder_entropies.append(
                details['encoder_entropy'].detach().cpu())
            all_token_counts.append(token_counts.detach().cpu())
            num_batches += 1

            batch_nll_upper = details['nll_upper_bound']
            batch_iw_nll_upper = details['iw_nll_upper_bound']
            batch_token_nll_upper = (
                batch_nll_upper / token_counts.clamp_min(1.0))
            batch_iw_token_nll_upper = (
                batch_iw_nll_upper / token_counts.clamp_min(1.0))
            batch_corpus_token_nll_upper = (
                batch_nll_upper.sum() / token_counts.sum().clamp_min(1.0))
            batch_iw_corpus_token_nll_upper = (
                batch_iw_nll_upper.sum() / token_counts.sum().clamp_min(1.0))
            batch_ppl_upper = math.exp(batch_corpus_token_nll_upper.item())
            batch_iw_ppl_upper = math.exp(
                batch_iw_corpus_token_nll_upper.item())
            print(
                f"Processed ELBO batch {batch_idx + 1}: "
                f"mean elbo={details['elbo'].mean().item():.4f}, "
                f"mean sample_elbo={details['sample_elbo'].mean().item():.4f}, "
                f"mean iw_elbo={details['iw_elbo'].mean().item():.4f}, "
                f"mean nll_upper={batch_nll_upper.mean().item():.4f}, "
                f"mean iw_nll_upper={batch_iw_nll_upper.mean().item():.4f}, "
                "mean sequence_token_nll_upper="
                f"{batch_token_nll_upper.mean().item():.4f}, "
                "mean sequence_iw_token_nll_upper="
                f"{batch_iw_token_nll_upper.mean().item():.4f}, "
                "batch corpus_token_nll_upper="
                f"{batch_corpus_token_nll_upper.item():.4f}, "
                "batch corpus_iw_token_nll_upper="
                f"{batch_iw_corpus_token_nll_upper.item():.4f}, "
                f"batch ppl_upper={batch_ppl_upper:.4f}, "
                f"batch iw_ppl_upper={batch_iw_ppl_upper:.4f}, "
                "mean continuous_log_prob="
                f"{details['continuous_log_prob'].mean().item():.4f}, "
                "mean decoder_log_prob="
                f"{details['decoder_log_prob'].mean().item():.4f}, "
                "mean encoder_log_prob="
                f"{details['encoder_log_prob'].mean().item():.4f}, "
                "mean encoder_entropy="
                f"{details['encoder_entropy'].mean().item():.4f}"
            )
    finally:
        model._train_mode()

    elbos = torch.cat(all_elbos, dim=0)
    sample_elbos = torch.cat(all_sample_elbos, dim=0)
    iw_elbos = torch.cat(all_iw_elbos, dim=0)
    nll_upper_bounds = torch.cat(all_nll_upper_bounds, dim=0)
    sample_nll_upper_bounds = torch.cat(all_sample_nll_upper_bounds, dim=0)
    iw_nll_upper_bounds = torch.cat(all_iw_nll_upper_bounds, dim=0)
    continuous_log_probs = torch.cat(all_continuous_log_probs, dim=0)
    decoder_log_probs = torch.cat(all_decoder_log_probs, dim=0)
    encoder_log_probs = torch.cat(all_encoder_log_probs, dim=0)
    encoder_entropies = torch.cat(all_encoder_entropies, dim=0)
    token_counts = torch.cat(all_token_counts, dim=0).clamp_min(1.0)
    sequence_token_nll_upper_bounds = nll_upper_bounds / token_counts
    sequence_sample_token_nll_upper_bounds = (
        sample_nll_upper_bounds / token_counts)
    sequence_iw_token_nll_upper_bounds = iw_nll_upper_bounds / token_counts
    total_nll_upper_bound = float(nll_upper_bounds.sum().item())
    total_iw_nll_upper_bound = float(iw_nll_upper_bounds.sum().item())
    total_tokens = float(token_counts.sum().item())
    corpus_token_nll_upper_bound = total_nll_upper_bound / total_tokens
    corpus_iw_token_nll_upper_bound = total_iw_nll_upper_bound / total_tokens
    ppl_upper_bound = math.exp(corpus_token_nll_upper_bound)
    iw_ppl_upper_bound = math.exp(corpus_iw_token_nll_upper_bound)

    payload = {
        'bound_type': 'discrete_token_elbo',
        'comparison_note': (
            'ELBO lower bound on discrete sequence log likelihood using the '
            'FLM endpoint latent q(y|x)=N((1-eps) one_hot(x), eps^2 I). '
            'The main ELBO uses exact Gaussian encoder entropy for lower '
            'variance. The IW fields use log-mean-exp across mc_samples for '
            'a tighter bound when mc_samples > 1. Reported token NLL and PPL '
            'values are upper bounds and depend on endpoint_eps plus Monte '
            'Carlo estimation settings.'
        ),
        'num_batches': num_batches,
        'num_sequences': int(elbos.numel()),
        'total_tokens': total_tokens,
        'mean_elbo': float(elbos.mean().item()),
        'mean_sample_elbo': float(sample_elbos.mean().item()),
        'mean_iw_elbo': float(iw_elbos.mean().item()),
        'mean_nll_upper_bound': float(nll_upper_bounds.mean().item()),
        'mean_sample_nll_upper_bound': float(
            sample_nll_upper_bounds.mean().item()),
        'mean_iw_nll_upper_bound': float(iw_nll_upper_bounds.mean().item()),
        'mean_sequence_token_nll_upper_bound': float(
            sequence_token_nll_upper_bounds.mean().item()),
        'mean_sequence_sample_token_nll_upper_bound': float(
            sequence_sample_token_nll_upper_bounds.mean().item()),
        'mean_sequence_iw_token_nll_upper_bound': float(
            sequence_iw_token_nll_upper_bounds.mean().item()),
        'corpus_token_nll_upper_bound': float(corpus_token_nll_upper_bound),
        'corpus_iw_token_nll_upper_bound': float(
            corpus_iw_token_nll_upper_bound),
        'ppl_upper_bound': float(ppl_upper_bound),
        'iw_ppl_upper_bound': float(iw_ppl_upper_bound),
        'mean_continuous_log_prob': float(
            continuous_log_probs.mean().item()),
        'mean_decoder_log_prob': float(decoder_log_probs.mean().item()),
        'mean_encoder_log_prob': float(encoder_log_probs.mean().item()),
        'mean_encoder_entropy': float(encoder_entropies.mean().item()),
        'mc_samples': int(elbo_cfg.elbo_mc_samples),
        'trace_method': elbo_cfg.elbo_trace_method,
        'trace_samples': int(elbo_cfg.elbo_trace_samples),
        'solver': elbo_cfg.elbo_solver,
        'endpoint_eps': float(elbo_cfg.elbo_endpoint_eps),
        'sequence_elbos': elbos.tolist(),
        'sequence_sample_elbos': sample_elbos.tolist(),
        'sequence_iw_elbos': iw_elbos.tolist(),
        'sequence_nll_upper_bounds': nll_upper_bounds.tolist(),
        'sequence_sample_nll_upper_bounds': (
            sample_nll_upper_bounds.tolist()),
        'sequence_iw_nll_upper_bounds': iw_nll_upper_bounds.tolist(),
        'sequence_token_counts': token_counts.tolist(),
        'sequence_token_nll_upper_bounds': (
            sequence_token_nll_upper_bounds.tolist()),
        'sequence_sample_token_nll_upper_bounds': (
            sequence_sample_token_nll_upper_bounds.tolist()),
        'sequence_iw_token_nll_upper_bounds': (
            sequence_iw_token_nll_upper_bounds.tolist()),
        'sequence_continuous_log_probs': continuous_log_probs.tolist(),
        'sequence_decoder_log_probs': decoder_log_probs.tolist(),
        'sequence_encoder_log_probs': encoder_log_probs.tolist(),
        'sequence_encoder_entropies': encoder_entropies.tolist(),
    }

    output_path = elbo_cfg.elbo_output_path
    with fsspec.open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print('ELBO summary:')
    print(f"  sequences: {payload['num_sequences']}")
    print(f"  total_tokens: {payload['total_tokens']}")
    print(f"  mean_elbo: {payload['mean_elbo']:.6f}")
    print(f"  mean_sample_elbo: {payload['mean_sample_elbo']:.6f}")
    print(f"  mean_iw_elbo: {payload['mean_iw_elbo']:.6f}")
    print(
        "  mean_nll_upper_bound: "
        f"{payload['mean_nll_upper_bound']:.6f}"
    )
    print(
        "  mean_sample_nll_upper_bound: "
        f"{payload['mean_sample_nll_upper_bound']:.6f}"
    )
    print(
        "  mean_iw_nll_upper_bound: "
        f"{payload['mean_iw_nll_upper_bound']:.6f}"
    )
    print(
        "  mean_sequence_token_nll_upper_bound: "
        f"{payload['mean_sequence_token_nll_upper_bound']:.6f}"
    )
    print(
        "  mean_sequence_iw_token_nll_upper_bound: "
        f"{payload['mean_sequence_iw_token_nll_upper_bound']:.6f}"
    )
    print(
        "  corpus_token_nll_upper_bound: "
        f"{payload['corpus_token_nll_upper_bound']:.6f}"
    )
    print(
        "  corpus_iw_token_nll_upper_bound: "
        f"{payload['corpus_iw_token_nll_upper_bound']:.6f}"
    )
    print(f"  ppl_upper_bound: {payload['ppl_upper_bound']:.6f}")
    print(f"  iw_ppl_upper_bound: {payload['iw_ppl_upper_bound']:.6f}")
    print(
        "  mean_continuous_log_prob: "
        f"{payload['mean_continuous_log_prob']:.6f}"
    )
    print(
        "  mean_decoder_log_prob: "
        f"{payload['mean_decoder_log_prob']:.6f}"
    )
    print(
        "  mean_encoder_log_prob: "
        f"{payload['mean_encoder_log_prob']:.6f}"
    )
    print(
        "  mean_encoder_entropy: "
        f"{payload['mean_encoder_entropy']:.6f}"
    )
    print(f"Saved likelihoods to: {output_path}")


@torch.inference_mode()
def generate_reflow_dataset_with_perturbed_rect(diffusion_model, config, logger, tokenizer):
    logger.info('Generating samples.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    train_ds, _ = dataloader.get_dataloaders(
        config, tokenizer, skip_valid=True)

    # i is given by random sequence of train_ds's N
    shuffled_indices = np.random.permutation(len(train_ds.dataset))

    eval_batch_size = config.loader.eval_batch_size
    generate_samples = config.sampling.num_reflow_samples

    x0s = []
    xTs = []
    ts = []

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    for j in range(generate_samples // eval_batch_size):
        if config.sampling.semi_ar:
            raise NotImplementedError(
                "Semi-AR sampling is not implemented. Please use standard sampling.")
        else:
            assert eval_batch_size == 1
            x0 = train_ds.dataset[shuffled_indices[j *
                                                   eval_batch_size:(j+1)*eval_batch_size]]['input_ids']
            x0 = torch.from_numpy(x0).to(model.device)
            x1 = torch.randint(0, 50258, x0.shape,
                               device=model.device, dtype=x0.dtype)
            rand_t = torch.randint(
                0, x0.shape[1], (1, ), device=model.device).float().item() / x0.shape[1]
            num_step = max(int(config.sampling.steps * (1 - rand_t)), 1)
            # random interpolate between x1 and x0
            # =y_given_t where y0=noise, y1=data
            xt = torch.where(rand_t > torch.rand(
                x1.shape, device=model.device), x0, x1)

            samples = model.restore_model_and_sample(
                num_steps=num_step, xT=xt.clone(), given_t=rand_t)
            x0s.append(samples.clone())
            xTs.append(xt.clone())
            ts.append(rand_t)
        if j % 500 == 0:
            print(f"Generated {(j+1) * eval_batch_size} samples")
    x0s = torch.cat(x0s, dim=0)
    xTs = torch.cat(xTs, dim=0)
    ts = torch.tensor(ts, device=model.device)

    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    x0s = x0s.cpu().numpy()
    xTs = xTs.cpu().numpy()
    ts = ts.cpu().numpy()

    save_path = config.data.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    xT_path = os.path.join(save_path, 'xT.npy')
    x0_path = os.path.join(save_path, 'x0.npy')
    ts_path = os.path.join(save_path, 'ts.npy')

    np.save(x0_path, x0s)
    np.save(xT_path, xTs)
    np.save(ts_path, ts)


def _train(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Training.')
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wid = config.wandb.get('id')
        if not wid or len(str(wid)) > 16:
            wid = str(uuid.uuid4().hex[:8])
        config.wandb.id = wid
        if config.wandb.get('name'):
            config.wandb.name = f"{config.wandb.name}_{wid}"
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)

    if (config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(
            config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    if config.training.finetune_path != '':
        assert utils.fsspec_exists(config.training.finetune_path)
        model = diffusion_model.load_from_checkpoint(
            config.training.finetune_path,
            tokenizer=tokenizer,
            config=config,
            weights_only=False)
    else:
        model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    # Force weights_only=False to allow full checkpoint restore (PyTorch 2.6 defaults torch.load to weights_only=True)
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
    if config.algo.name == 'ar':
        diffusion_model = algo.AR
    elif config.algo.name == 'mdlm':
        diffusion_model = algo.MDLM
    elif config.algo.name == 'duo_base':
        diffusion_model = algo.DUO_BASE
    elif config.algo.name == 'duo':
        diffusion_model = algo.DUO
    elif config.algo.name == 'flm':
        diffusion_model = algo.FLM
    elif config.algo.name == 'flm_vdm':
        diffusion_model = algo.FLMVDM
    elif config.algo.name == 'fmlm_twomodel':
        diffusion_model = algo.FMLM_TwoModel
    elif config.algo.name == 'fmlm_twostage':
        diffusion_model = algo.FMLM_TwoStage
    elif config.algo.name == 'fmlm':
        diffusion_model = algo.FMLM
    elif config.algo.name == 'd3pm':
        diffusion_model = algo.D3PMAbsorb
    elif config.algo.name == 'sedd':
        diffusion_model = algo.SEDDAbsorb
    elif config.algo.name == 'distillation':
        diffusion_model = algo.Distillation
    elif config.algo.name == 'rectification':
        diffusion_model = algo.Rectification
    else:
        raise ValueError(
            f'Invalid algorithm name: {config.algo.name}')
    kwargs = {'diffusion_model': diffusion_model,
              'config': config,
              'tokenizer': tokenizer,
              'logger': logger}
    if config.mode == 'sample_eval':
        _generate_samples(**kwargs)
    elif config.mode == 'sample_eval_recon':
        _generate_samples(**kwargs)
    elif config.mode == 'sample_eval_with_tc':
        _generate_samples_with_tc(**kwargs)
    elif config.mode == 'ppl_eval':
        _eval_ppl(**kwargs)
    elif config.mode == 'likelihood_eval':
        _eval_continuous_likelihood(**kwargs)
    elif config.mode == 'elbo_eval':
        _eval_discrete_elbo(**kwargs)
    elif config.mode == 'generate_reflow_data':
        generate_reflow_dataset(diffusion_model, config, logger, tokenizer)
    elif config.mode == 'generate_reflow_data_with_perturbed_rect':
        generate_reflow_dataset_with_perturbed_rect(**kwargs)
    else:
        _train(**kwargs)


if __name__ == '__main__':
    # allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
