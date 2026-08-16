'''Generate realistic (R-like) images from synthetic inputs with known parameters.

This is the project deliverable: cheap labelled data. Each synthetic input has
known lattice parameters; each generated output inherits them, so the pair
(output image, source parameters) becomes a training example for the downstream
project that lacks real data.

Two methods, same interface, so the comparison is apples-to-apples:

  --method dps      x ~ p_R(x) . p(s | G_{R->S}(x))
                    Diffusion prior on REAL, conditioned so the synth-domain
                    projection matches the input. Stochastic: --samples_per_input
                    > 1 gives genuinely different realistic variants of one input.

  --method uvcgan   x = G_{S->R}(s)
                    The direct-translation baseline. Deterministic, so
                    --samples_per_input > 1 is pointless and is rejected.

Two modes, because the measurement can come from two places:

  --mode generate (--synth_root)
      y IS the synthetic image. This is the product. It requires a real trained
      G_{R->S}, because y must lie in the operator's actual output distribution.

  --mode validate (--reference_root)
      y = A(real image), and the real image is recorded as the source. Self
      consistent for ANY operator, including the untrained `spectral` stand-in,
      so the full pipeline can be exercised before UVCGAN exists. The generated
      image should recover the reference it came from.

Both modes write the same manifest format, so scripts/evaluate.py consumes
either without knowing which was used.

    python scripts/generate_augmented.py \
        --model_config configs/crystal_model_config.yaml \
        --diffusion_config configs/crystal_diffusion_config.yaml \
        --task_config configs/crystal_cyclegan_config.yaml \
        --synth_root /path/to/dataset/synth/val \
        --out_dir ./results/augmented --method dps --samples_per_input 4

Writes images plus manifest.csv mapping every output back to its source file,
which is what carries the labels.
'''
import argparse
import csv
import os

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import matplotlib.pyplot as plt
import torch
import yaml

from data.dataloader import get_dataset
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from util.img_utils import clear_color
from util.logger import get_logger


def load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', type=str, required=True)
    p.add_argument('--diffusion_config', type=str, required=True)
    p.add_argument('--task_config', type=str, required=True)
    p.add_argument('--mode', choices=['generate', 'validate'], default='generate')
    p.add_argument('--synth_root', type=str, default=None,
                   help='mode=generate: synthetic PNGs used directly as y.')
    p.add_argument('--reference_root', type=str, default=None,
                   help='mode=validate: real PNGs; y = A(reference).')
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--method', choices=['dps', 'uvcgan'], default='dps')
    p.add_argument('--samples_per_input', type=int, default=1,
                   help='DPS only. >1 exploits the stochasticity a GAN lacks.')
    p.add_argument('--limit', type=int, default=None,
                   help='Only process the first N synthetic inputs.')
    p.add_argument('--scale', type=float, default=None,
                   help='Override the conditioning scale from the task config.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logger = get_logger()
    torch.manual_seed(args.seed)

    if args.method == 'uvcgan' and args.samples_per_input != 1:
        raise ValueError(
            'The uvcgan baseline is deterministic, so --samples_per_input > 1 '
            'would just duplicate identical images. Use --method dps for '
            'multiple variants per input.')

    roots = {'generate': args.synth_root, 'validate': args.reference_root}
    expected = {'generate': '--synth_root', 'validate': '--reference_root'}
    input_root = roots[args.mode]
    if input_root is None:
        raise ValueError(f'--mode {args.mode} requires {expected[args.mode]}.')
    if roots[{'generate': 'validate', 'validate': 'generate'}[args.mode]]:
        raise ValueError(
            f'--mode {args.mode} uses {expected[args.mode]}; the other root is '
            'ignored, which is probably not what you meant.')

    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f'Device: {device} / method: {args.method}')

    task_config = load_yaml(args.task_config)
    measure_config = task_config['measurement']

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, 'generated')
    os.makedirs(img_dir, exist_ok=True)

    # The synthetic inputs. image_size must match both the prior and the
    # generator; taken from the model config so the three cannot drift apart.
    model_config = load_yaml(args.model_config)
    image_size = model_config['image_size']
    dataset = get_dataset(name='crystal', root=input_root,
                          image_size=image_size, augment=False)
    # Spread the subset evenly over the sorted file list instead of taking the
    # first N. Crops are named "<photo>_sample_<k>", so the first N files are
    # all crops of the alphabetically-first photo(s): --limit 8 gave 2 distinct
    # scenes, --limit 32 gave 7. Striding gives N distinct scenes and makes small
    # runs representative rather than a study of one micrograph.
    if args.limit is not None and args.limit < len(dataset):
        stride = len(dataset) / args.limit
        indices = [int(i * stride) for i in range(args.limit)]
    else:
        indices = list(range(len(dataset)))
    n_inputs = len(indices)
    logger.info(f'{n_inputs} inputs from {input_root} (mode={args.mode})')

    # The real->synth operator is needed by DPS (it is what DPS inverts) and by
    # validate mode (it derives y from the reference). Built once and shared:
    # constructing it twice would hold a trained generator in GPU memory twice.
    # The uvcgan baseline in generate mode needs none of it.
    operator = None
    if args.method == 'dps' or args.mode == 'validate':
        operator = get_operator(device=device, **measure_config['operator'])
        if operator.direction != 'ba':
            raise ValueError(
                f"DPS needs the real->synth operator, but the task config gives "
                f"direction: {operator.direction!r}. DPS inverts its operator, "
                "so generating R from S requires direction: ba.")

    # In validate mode the measurement is derived from the reference by the same
    # operator DPS inverts, so y is guaranteed to lie in A's output distribution.
    # In generate mode the input already IS the measurement.
    to_measurement = None
    if args.mode == 'validate':
        to_measurement = lambda ref: operator.forward(ref).detach()

    if args.method == 'uvcgan':
        # Baseline: the S -> R generator applied directly. Note this is the
        # OPPOSITE direction to the DPS operator, so flip it explicitly.
        op_cfg = dict(measure_config['operator'])
        op_cfg['direction'] = 'ab'
        generator = get_operator(device=device, **op_cfg)
        generate = lambda s: generator.forward(s).detach()
    else:
        model = create_model(**model_config).to(device).eval()
        noiser = get_noise(**measure_config['noise'])

        cond_config = task_config['conditioning']
        params = dict(cond_config['params'])
        if args.scale is not None:
            params['scale'] = args.scale
        logger.info(f"Conditioning: {cond_config['method']} scale={params['scale']}")

        cond_method = get_conditioning_method(
            cond_config['method'], operator, noiser, **params)
        sampler = create_sampler(**load_yaml(args.diffusion_config))

        def generate(s):
            x_start = torch.randn(s.shape, device=device).requires_grad_()
            return sampler.p_sample_loop(
                model=model, x_start=x_start, measurement=s,
                measurement_cond_fn=cond_method.conditioning,
                record=False, save_root=args.out_dir).detach()

    manifest_path = os.path.join(args.out_dir, 'manifest.csv')
    with open(manifest_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        # source_image is what carries the known lattice parameters.
        writer.writerow(['output_image', 'source_image', 'method',
                         'sample_index', 'seed'])

        for n, i in enumerate(indices):
            source_path = dataset.fpaths[i]
            s = dataset[i].unsqueeze(0).to(device)
            if to_measurement is not None:
                s = to_measurement(s)
            logger.info(f'[{n + 1}/{n_inputs}] {os.path.basename(source_path)}')

            for k in range(args.samples_per_input):
                out = generate(s)
                fname = f'{n:05d}_{k:02d}.png'
                plt.imsave(os.path.join(img_dir, fname), clear_color(out))
                writer.writerow([fname, source_path, args.method, k, args.seed])
            fh.flush()

    logger.info(f'Wrote {args.out_dir} (manifest: {manifest_path})')


if __name__ == '__main__':
    main()
