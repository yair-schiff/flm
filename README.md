<h1 align="center">Flow Map Language Models:<br>One-step Language Modeling via Continuous Denoising</h1>

<div align="center">
  
**[Chanhyuk Lee](https://david3684.github.io)**<sup>1</sup>, **[Jaehoon Yoo](https://sites.google.com/view/jaehoon-yoo/홈)**<sup>1</sup>, **[Manan Agarwal](https://mananag007.github.io)**<sup>2</sup>, **[Sheel Shah](https://sheelfshah.github.io)**<sup>2</sup>, **[Jerry Huang](https://jrrhuang.github.io/)**<sup>2</sup>, \
**[Aditi Raghunathan](https://www.cs.cmu.edu/~aditirag/)**<sup>2</sup>, **[Seunghoon Hong](https://maga33.github.io/)**<sup>1</sup>, **[Nicholas M. Boffi](https://nmboffi.github.io/)**<sup>†2</sup>, **[Jinwoo Kim](https://jw9730.github.io/)**<sup>†1</sup>



<sup>1</sup>KAIST &nbsp; <sup>2</sup>Carnegie Mellon University &nbsp; <sup>†</sup>Equal advising
</div>

<div align="center">

**[[Project Page]](https://one-step-lm.github.io/)** | **[[Paper]](https://arxiv.org/abs/2602.16813)** | **[[Blog]](https://one-step-lm.github.io/blog/index.html)**

</div>

## News

- **[2026-04]** We released LM1B/OpenWebText checkpoints for FLM and FMLM. 

## TL;DR

<p align="center">
  <img src="figures/overview.gif" width="100%">
</p>

<p align="center">
  <img src="figures/overview.png" width="100%">
</p>

We introduce **Flow Language Model (FLM)** and its flow-map distilled variant **Flow Map Language Model (FMLM)**, enabling **one-step parallel text generation** through continuous denoising. 

## Overview

FLM applies the benefits of continuous image generation to discrete state spaces by encoding text as one-hot vectors and using flow matching to directly map noise to one-hot data. Unlike discrete diffusion, FLM **gradually denoises all tokens in parallel**, allowing it to represent a superposition of sequences while capturing correlations between tokens — a fundamental bottleneck for discrete diffusion in the few-step regime.

## How to Run

### Install Dependencies

Recommended cluster / workstation path:

```bash
chmod +x scripts/bootstrap_env.sh scripts/verify_env.sh
scripts/bootstrap_env.sh
scripts/verify_env.sh
```

This installs the project into the `flm_og` conda env prefix by default and
includes an env-local `cuda-nvcc` so `flash-attn` can be built reliably.

Manual path:

```bash
pip install torch>=2.3.0
pip install -r requirements.txt
# Install flash-attn separately matching your python / torch version (see https://github.com/Dao-AILab/flash-attention/releases)
CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin:$PATH" \
PIP_NO_CACHE_DIR=1 pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
```

Our DiT backbone supports `torch.compile` with `max-autotune` for faster training. Enable it by setting the environment variable before running any script:

```bash
export DIT_USE_COMPILE=TRUE
```

With the option, we are able to train OpenWebText experiments with 512 batch size on 8 H100 (80GB VRAM), with local batch size of 32.

When using the provided launch scripts, `setup_env.sh` will activate `flm_og`
by default unless `FLM_CONDA_ENV` is set explicitly.

### Training

Before running, update `data.cache_dir` in the scripts to point to your dataset location. If the directory is empty, the dataset will be automatically downloaded and preprocessed.

Set `algo.teacher_path` to your pre-trained FLM checkpoint before running FMLM distillation.


| Model | Dataset     | Script                                                                     |
| ----- | ----------- | -------------------------------------------------------------------------- |
| FLM   | LM1B        | [scripts/train_lm1b_flm.sh](scripts/train_lm1b_flm.sh)                     |
| FMLM  | LM1B        | [scripts/train_lm1b_fmlm_denoiser.sh](scripts/train_lm1b_fmlm_denoiser.sh) |
| FLM   | OpenWebText | [scripts/train_owt_flm.sh](scripts/train_owt_flm.sh)                       |
| FMLM  | OpenWebText | [scripts/train_owt_fmlm_denoiser.sh](scripts/train_owt_fmlm_denoiser.sh)   |

### Evaluation

Set `CKPT_PATH` in the script to your trained checkpoint before running.


| Model | Dataset     | Script                                                       |
| ----- | ----------- | ------------------------------------------------------------ |
| FLM   | LM1B        | [scripts/gen_ppl_lm1b_flm.sh](scripts/gen_ppl_lm1b_flm.sh)   |
| FMLM  | LM1B        | [scripts/gen_ppl_lm1b_fmlm.sh](scripts/gen_ppl_lm1b_fmlm.sh) |
| FLM   | OpenWebText | [scripts/gen_ppl_owt_flm.sh](scripts/gen_ppl_owt_flm.sh)     |
| FMLM  | OpenWebText | [scripts/gen_ppl_owt_fmlm.sh](scripts/gen_ppl_owt_fmlm.sh)   |

## Checkpoints
### Pretrained Checkpoints

Pretrained FLM and FMLM checkpoints are available at [here](https://drive.google.com/drive/folders/1fNAx4LP2RwPBdqDQFQ_gRrYZI9u3Vq15?usp=drive_link).


| Model | Dataset     | Checkpoint       |
| ----- | ----------- | ---------------- |
| FLM   | LM1B        | `lm1b_flm.ckpt`  |
| FMLM  | LM1B        | `lm1b_fmlm.ckpt` |
| FLM   | OpenWebText | `owt_flm.ckpt`   |
| FMLM  | OpenWebText | `owt_fmlm.ckpt`  |


Set `eval.checkpoint_path` (or `algo.teacher_path` for distillation) to the downloaded checkpoint path when running evaluation or distillation scripts.

### Baseline Checkpoints

Reproduced baseline checkpoints for LM1B are available at [here](https://drive.google.com/drive/folders/1TJO3aFWqI7ukbmjciZ6krAUFlAak1itl?usp=drive_link).

For other checkpoints, mostly for OpenWebText, refer to [Duo](https://github.com/s-sahoo/duo), [SDTT](https://github.com/jdeschena/sdtt), [RDLM](https://github.com/harryjo97/RDLM), [di4c](https://github.com/sony/di4c) repositories.

## BibTeX

```bibtex
@article{lee2026flow,
    title={Flow Map Language Models: One-step Language Modeling via Continuous Denoising},
    author={Chanhyuk Lee and Jaehoon Yoo and Manan Agarwal
            and Sheel Shah and Jerry Huang
            and Aditi Raghunathan and Seunghoon Hong
            and Nicholas M. Boffi and Jinwoo Kim},
    journal={arXiv preprint arXiv:2602.16813},
    year={2026},
}
```

---

## Acknowledgements

This repository is built upon the codebases of **[Duo](https://github.com/s-sahoo/duo)** and **[ReDi](https://github.com/Ugness/ReDi)**.
