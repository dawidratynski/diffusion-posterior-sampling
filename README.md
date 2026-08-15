# Diffusion Posterior Sampling for General Noisy Inverse Problems (ICLR 2023 spotlight)

![result-gif1](./figures/motion_blur.gif)
![result-git2](./figures/super_resolution.gif)
<!-- See more results in the [project-page](https://jeongsol-kim.github.io/dps-project-page) -->

## Abstract
In this work, we extend diffusion solvers to efficiently handle general noisy (non)linear inverse problems via the approximation of the posterior sampling. Interestingly, the resulting posterior sampling scheme is a blended version of the diffusion sampling with the manifold constrained gradient without strict measurement consistency projection step, yielding more desirable generative path in noisy settings compared to the previous studies.

![cover-img](./figures/cover.jpg)


## Prerequisites
- python 3.8

- pytorch 1.11.0

- CUDA 11.3.1

- nvidia-docker (if you use GPU in docker container)

It is okay to use lower version of CUDA with proper pytorch version.

Ex) CUDA 10.2 with pytorch 1.7.0

<br />

## Getting started 

### 1) Clone the repository

```
git clone https://github.com/DPS2022/diffusion-posterior-sampling

cd diffusion-posterior-sampling
```

<br />

### 2) Download pretrained checkpoint
From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoint "ffhq_10m.pt" and paste it to ./models/
```
mkdir models
mv {DOWNLOAD_DIR}/ffqh_10m.pt ./models/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.

:speaker: Checkpoint for imagenet is uploaded.

<br />


### 3) Set environment
### [Option 1] Local environment setting

We use the external codes for motion-blurring and non-linear deblurring.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse

git clone https://github.com/LeviBorodenko/motionblur motionblur
```

Install dependencies

```
conda create -n DPS python=3.8

conda activate DPS

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

<br />

### [Option 2] Build Docker image

Install docker engine, GPU driver and proper cuda before running the following commands.

Dockerfile already contains command to clone external codes. You don't have to clone them again.

--gpus=all is required to use local GPU device (Docker >= 19.03)

```
docker build -t dps-docker:latest .

docker run -it --rm --gpus=all dps-docker
```

<br />

### 4) Inference

```
python3 sample_condition.py \
--model_config=configs/model_config.yaml \
--diffusion_config=configs/diffusion_config.yaml \
--task_config={TASK-CONFIG};
```


:speaker: For imagenet, use configs/imagenet_model_config.yaml

<br />

## Possible task configurations

```
# Linear inverse problems
- configs/super_resolution_config.yaml
- configs/gaussian_deblur_config.yaml
- configs/motion_deblur_config.yaml
- configs/inpainting_config.yaml

# Non-linear inverse problems
- configs/nonlinear_deblur_config.yaml
- configs/phase_retrieval_config.yaml
```

### Structure of task configurations
You need to write your data directory at data.root. Default is ./data/samples which contains three sample images from FFHQ validation set.

```
conditioning:
    method: # check candidates in guided_diffusion/condition_methods.py
    params:
        scale: 0.5

data:
    name: ffhq
    root: ./data/samples/

measurement:
    operator:
        name: # check candidates in guided_diffusion/measurements.py

noise:
    name:   # gaussian or poisson
    sigma:  # if you use name: gaussian, set this.
    (rate:) # if you use name: poisson, set this.
```

---

# Thesis fork: DPS with a CycleGAN/UVCGAN operator

**Goal: generating labelled data, not recovering lattices.** Real measurements
are prohibitively expensive to collect. Synthetic images can be generated
without limit *with known lattice parameters*, but look too clean. So: take a
synthetic image `s`, produce a realistic `r̂` that keeps `s`'s parameters, and
use `(r̂, params)` as training data downstream. The question is whether DPS does
this better than applying a cycle-model directly.

Domain **A = synth**, **B = real**, matching `train_crystals.py` in the sibling
`MIMUW_Licencjat_Mat` repo. DPS **inverts** its operator, so producing R from S
means the operator points R→S:

| | role |
|---|---|
| diffusion prior | trained on **real** — the distribution being sampled from |
| `gen_ba` (real→synth) | the **DPS forward operator** |
| `gen_ab` (synth→real) | the **baseline** DPS is compared against |
| measurement `y` | the known synthetic image `s` (exact, hence `noise: clean`) |

DPS samples `x ~ p_R(x) · p(s | G_{R→S}(x))` — "a realistic image whose
synth-domain projection is my known `s`".

Why this should beat direct translation: `G_{R→S}` is trained to *discard*
imperfections, so its null space **is** the imperfection manifold. DPS resamples
that null space from the real-image prior rather than collapsing it to one
deterministic output — so one synthetic input yields many distinct realistic
variants, which a deterministic generator cannot provide.

⚠️ The prior now sits on the **scarce** side (~15k crops from ~2957 source
images, so far fewer independent samples than the count suggests). Memorisation
is the risk to watch: a prior that reproduces training images makes the
augmentation worthless. Check nearest neighbours against the training set before
trusting samples.

Everything runs at **160×160**: the 150px PNGs are resized, matching UVCGAN2's
`shape: (3, 160, 160)`. The prior and the operator must share a resolution and
the `[-1, 1]` pixel convention or the data-consistency term is meaningless.

## Local setup (CPU, no GPU needed)

```
uv sync          # py3.11 + CPU torch
uv run pytest    # ~15s, verifies the pipeline including autograd through the operator
```

The upstream `requirements.txt` is a 2022 `pip freeze` with no wheels for modern
Python and is missing torch/torchvision/lpips/scikit-image; `pyproject.toml`
supersedes it. `torch` sits in the `cpu` dependency group, so on Colab/a cluster
you use the environment's CUDA torch and install only the project deps.

Develop and test locally, run at full scale remotely. Keep logic in git and the
notebook limited to clone + install + invoke. For fast local iteration lower
`timestep_respacing` (not `steps` — the linear schedule rescales betas by
`1000/steps` and produces `beta > 1`).

## Quick end-to-end check

```
./scripts/smoke_e2e.sh ../dataset
```

Trains a prior, runs DPS through the analytic stand-in operator, and evaluates —
the same scripts and configs as a real run, ~7 min on CPU. Output quality is
meaningless at that scale; the point is to catch wiring and format errors before
spending GPU time or an overnight run.

## Data hygiene

The pipeline emits several crops per source photo, so the train/val split must
be at **source** level. Splitting on filenames scatters siblings across both
sides: on the sample dataset that leaked 1969 of 1970 real val sources into
train, which inflates val metrics and makes the memorisation check meaningless.

```
uv run python scripts/fix_split.py ../dataset/real ../dataset/synth --dry_run
```

Re-splits in place, no regeneration. Idempotent, and skips domains that are
already clean. (`data_processing/train_val_split.py` in the sibling repo is
fixed too, for future regenerations.)

## Training the prior

```
uv run python scripts/train_diffusion.py \
    --model_config configs/crystal_model_config.yaml \
    --data_root /path/to/dataset/real/train \
    --out_dir ./models/crystal_real \
    --batch_size 16 --amp --resume
```

**`real/train`, not synth** — the prior models the distribution being generated.

Real is the scarce side, so pretrain on synth first (unlimited data, no
memorisation risk) and finetune from it. `--init_from` takes a bare
`model_ema_*.pt` and starts a fresh optimiser and step counter, unlike
`--resume` which continues a run:

```
uv run python scripts/train_diffusion.py ... \
    --data_root ../dataset/synth/train --out_dir ./models/synth_pretrain
uv run python scripts/train_diffusion.py ... \
    --data_root ../dataset/real/train --out_dir ./models/real_finetune \
    --init_from ./models/synth_pretrain/model_ema_150000.pt --lr 2e-5
```

Stop the finetune on the **memorisation** metric, not on loss.

[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) runs this on Colab.
It contains no logic — clone, install, mount Drive, call the same CLIs.

> **EMA decay and short runs.** `model_ema_*.pt` is what `model_path` loads, and
> a fixed decay of 0.9999 has a ~10k-update horizon: after 300 steps 97% of the
> exported average is still the random initialisation, and it predicts at chance
> at every timestep while the training log looks healthy. The decay therefore
> ramps as `(1+n)/(10+n)` capped at `--ema_rate`. `smoke_e2e.sh` asserts the
> exported checkpoint beats chance, so this cannot regress silently.

Writes `ckpt_latest.pt` (resume state) and `model_ema_XXXXXX.pt` (a bare
state_dict for `model_path`). **Always pass `--resume` on Colab** — sessions get
killed. Sample from the EMA weights, not the raw ones.

Objective is epsilon-MSE only, hence `learn_sigma: False` plus a `fixed_*`
`model_var_type`. The learned-variance hybrid VLB is not implemented.

## Generating augmented data (the deliverable)

Emits realistic images plus `manifest.csv` mapping each output back to its
source — that mapping is what carries the labels.

```
# product: y IS the synthetic image (needs a trained G_{R->S})
uv run python scripts/generate_augmented.py ... \
    --mode generate --synth_root ../dataset/synth/val \
    --out_dir ./results/dps --method dps --samples_per_input 4

# rehearsal: y = A(real image), works with any operator including the stand-in
uv run python scripts/generate_augmented.py ... \
    --mode validate --reference_root ../dataset/real/val \
    --out_dir ./results/dps --method dps --samples_per_input 4
```

`--mode generate` requires a real trained generator: `y` has to lie in the
operator's actual output distribution, and the analytic stand-in emits
sinusoid-like images rather than true synth-domain ones, so the data-consistency
term would be unsatisfiable by construction. `--mode validate` derives `y` with
the same operator DPS inverts, so it is self-consistent for any operator.

`--method uvcgan` runs the direct-translation baseline through the same
interface. It is deterministic, so `--samples_per_input > 1` is rejected rather
than silently emitting duplicates. Both modes write the same manifest format,
so `evaluate.py` consumes either.

## Evaluation

```
uv run python scripts/evaluate.py \
    --result_dir ./results/aug_dps --result_dir ./results/aug_uvcgan \
    --real_root /path/to/dataset/real/val \
    --real_train_root /path/to/dataset/real/train \
    --out_csv ./results/per_image.csv
```

There is no ground-truth output — given a synthetic input there is no single
correct realistic image — so reconstruction error is the wrong frame. Four axes
instead:

| axis | metric | reading |
|---|---|---|
| **label validity** | lattice spacing / angle error vs the source | **decisive.** Drift ⇒ mislabelled pairs ⇒ the data is worse than useless |
| realism | radial-spectrum distance to real; peak prominence | prominence should land *near* the real value — below is over-degraded, above is too clean |
| diversity | pairwise RMSE across variants of one input | structurally 0 for a deterministic generator; DPS's main claim |
| memorisation | NN distance to the real training set | compared against the real-to-real distance as a floor |

Label validity is the one to lead with: PSNR/FID cannot see it, and a method
that adds convincing imperfections while shifting the spacing scores *well* on
every generic metric while producing unusable data.

Metrics live in [`util/lattice_metrics.py`](util/lattice_metrics.py) and are
tested against synthetic gratings with exactly known period and orientation.

Calibrated on this dataset:

- The lattice fundamental sits at radius ~5–6 (period 25–30px) in **both**
  domains. Real carries 42% of its power at radius ≤3 (illumination gradients),
  which the search annulus excludes.
- Integer FFT bins at radius 5 quantise spacing into ~17% steps, so peak finding
  zero-pads 4× by default.
- Spacing error is **exact** (median 0.0000, p90 ≤2.4%) under heavy noise,
  real-matched contrast shifts, and 30% occlusion by other structure — so >5%
  error is real drift, not measurement noise.
- Structures are 2D lattices, not 1D fringes. `lattice_signature` /
  `signature_distance` compare whole reciprocal lattices, catching distortion of
  a secondary vector that `dominant_peak` alone reports as zero error.
- Prominence is aggregated by **median**: it is heavily right-tailed (real val
  median vs. mean differ by an order of magnitude), so means track outliers.

## Validation experiment

`sample_condition.py` reconstructs a held-out **real** image from its own
synth-domain projection, so there is a ground truth to score against:

```
uv run python sample_condition.py \
    --model_config=configs/crystal_model_config.yaml \
    --diffusion_config=configs/crystal_diffusion_config.yaml \
    --task_config=configs/crystal_cyclegan_config.yaml
```

This benchmark structurally favours the UVCGAN baseline — `gen_ab` was trained
with a cycle loss to invert `gen_ba`. Report that caveat; a DPS win here is
strong, a loss is not damning.

Both configs default to `framework: stub`, a seeded dummy generator so the whole
pipeline runs before real weights exist. Swap to `framework: uvcgan2` + `path:`
when they arrive.

## Changes to upstream

- `create_model` no longer swallows checkpoint-load errors. It used to catch
  every exception and silently fall back to random weights, so a bad path or an
  architecture mismatch produced an untrained model that still sampled.
- `fixed_small` variance took `log(0) = -inf` at `t=0`; now clipped like
  `fixed_large` and `posterior_log_variance_clipped` already were.
- `np.float` → `float` (removed in numpy 1.24).
- `motionblur` imports made lazy, so the repo imports without cloning `bkse`
  or `motionblur`.
- `noise: clean` now works with `ps`/`mcg` conditioning. It previously hit
  `NotImplementedError`, but is the correct setting when the measurement is a
  generated image known exactly. The L2 gradient does not depend on sigma.

## Citation
If you find our work interesting, please consider citing

```
@inproceedings{
chung2023diffusion,
title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
author={Hyungjin Chung and Jeongsol Kim and Michael Thompson Mccann and Marc Louis Klasky and Jong Chul Ye},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=OnD9zGAGT0k}
}
```

