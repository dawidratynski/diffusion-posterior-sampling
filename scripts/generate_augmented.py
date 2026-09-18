'''Generate realistic (R-like) images, for the three-way S->R model comparison.

THE THREE MODELS UNDER TEST
---------------------------
All three map a synth-domain image to a realistic one; they differ in how.

  uvcgan          x = G_{S->R}(s)
                  Direct translation. Deterministic, so --samples_per_input > 1
                  is pointless and rejected.
                  --method uvcgan

  dps + uvcgan    x ~ p_R(x) . p(s | G_{R->S}(x))
                  DPS with the TRAINED real->synth generator as its operator.
                  --method dps --dps_framework uvcgan2

  dps + analytic  as above with the analytic SpectralIdealizer as the operator.
                  The "no trained translation model" ablation.
                  --method dps --dps_framework spectral

The operator is named on the command line, never inherited silently from the
task config: which operator ran decides what the experiment measured, and a
plausible-but-wrong one yields plausible-but-wrong numbers with nothing in the
output to reveal it.

TWO INPUT MODES
---------------
  --input_mode synth (--synth_root)
      y IS a synthetic image. This is the product: cheap labelled data, since
      each output inherits its source's known lattice parameters.

  --input_mode roundtrip (--real_root)
      y = G_{R->S}(real), and the real image is recorded as ground truth, so
      paired metrics (PSNR/SSIM/LPIPS) become available. Which R->S model builds
      y is --rs_framework, INDEPENDENT of the operator DPS inverts. That
      separation is the point: every SR model must see the same y for the
      comparison to be apples-to-apples, including dps+analytic, whose operator
      then does not match the y it is given. That mismatch is not a flaw in the
      experiment -- it is what the ablation measures.

Writes generated/, measurement/ (roundtrip only) and manifest.csv. The manifest
records, per output: the input whose lattice defines the label, the ground truth
if any, and the measurement actually fed to the model.

    python scripts/generate_augmented.py \
        --model_config configs/crystal_model_config.yaml \
        --diffusion_config configs/crystal_diffusion_config.yaml \
        --task_config configs/crystal_cyclegan_config.yaml \
        --input_mode roundtrip --real_root /path/to/dataset/real/val \
        --rs_framework uvcgan2 --uvcgan_path /path/to/uvcgan_model \
        --method dps --dps_framework uvcgan2 \
        --out_dir ./results/rt_dps_uvcgan --label dps_uvcgan \
        --samples_per_input 4 --limit 50
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
from util.img_utils import to_display
from util.logger import get_logger

FRAMEWORKS = ('uvcgan2', 'spectral', 'cyclegan_resnet', 'torchscript')


def load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', type=str, required=True)
    p.add_argument('--diffusion_config', type=str, required=True)
    p.add_argument('--task_config', type=str, required=True)

    p.add_argument('--input_mode', choices=['synth', 'roundtrip'],
                   default='synth')
    p.add_argument('--synth_root', type=str, default=None,
                   help='input_mode=synth: synthetic PNGs used directly as y.')
    p.add_argument('--real_root', type=str, default=None,
                   help='input_mode=roundtrip: real PNGs; y = G_RS(real).')
    p.add_argument('--rs_framework', choices=FRAMEWORKS, default=None,
                   help='roundtrip: which real->synth model builds y. Keep this '
                        'fixed across the models being compared.')

    p.add_argument('--method', choices=['dps', 'uvcgan'], default='dps')
    p.add_argument('--dps_framework', choices=FRAMEWORKS, default=None,
                   help='dps: which real->synth operator DPS inverts. This is '
                        'what distinguishes the two DPS variants.')
    p.add_argument('--uvcgan_path', type=str, default=None,
                   help='Model directory for any uvcgan2 framework selected '
                        'above. Applies to both --rs_framework and '
                        '--dps_framework and to --method uvcgan.')

    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--label', type=str, default=None,
                   help='Name for this model in the comparison tables. '
                        'Defaults to method[+dps_framework].')
    p.add_argument('--samples_per_input', type=int, default=1,
                   help='DPS only. >1 exploits the stochasticity a GAN lacks.')
    p.add_argument('--limit', type=int, default=None,
                   help='Process N inputs, strided across the sorted file list.')
    p.add_argument('--scale', type=float, default=None,
                   help='Override the conditioning scale from the task config.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def build_operator(measure_cfg, framework, direction, device, uvcgan_path):
    '''One real->synth (or synth->real) generator, with framework stated.'''
    cfg = dict(measure_cfg['operator'])
    cfg['framework'] = framework
    cfg['direction'] = direction

    # Framework-specific settings in the task config would be rejected by a
    # different framework's loader, so keep only what this one accepts.
    if framework != 'spectral':
        for k in ('gamma', 'min_period', 'max_period', 'target_mean',
                  'target_std', 'softness'):
            cfg.pop(k, None)
    if framework in ('uvcgan2', 'cyclegan_resnet', 'torchscript'):
        if not uvcgan_path:
            raise ValueError(
                f"framework '{framework}' needs weights: pass --uvcgan_path.")
        if not os.path.exists(uvcgan_path):
            raise FileNotFoundError(
                f'--uvcgan_path {uvcgan_path} does not exist. Run '
                'scripts/probe_uvcgan.py first to check the checkpoint loads.')
        cfg['path'] = uvcgan_path
    else:
        cfg.pop('path', None)

    return get_operator(device=device, **cfg)


def validate_args(args):
    if args.method == 'uvcgan' and args.samples_per_input != 1:
        raise ValueError(
            'The uvcgan baseline is deterministic, so --samples_per_input > 1 '
            'would just duplicate identical images. Use --method dps for '
            'multiple variants per input.')

    roots = {'synth': args.synth_root, 'roundtrip': args.real_root}
    expected = {'synth': '--synth_root', 'roundtrip': '--real_root'}
    other = {'synth': 'roundtrip', 'roundtrip': 'synth'}[args.input_mode]
    if roots[args.input_mode] is None:
        raise ValueError(
            f'--input_mode {args.input_mode} requires {expected[args.input_mode]}.')
    if roots[other]:
        raise ValueError(
            f'--input_mode {args.input_mode} uses {expected[args.input_mode]}; '
            f'{expected[other]} would be ignored, which is probably not meant.')

    if args.input_mode == 'roundtrip' and not args.rs_framework:
        raise ValueError(
            '--input_mode roundtrip needs --rs_framework: y = G_RS(real), and '
            'which R->S model builds y must be identical across the models you '
            'are comparing or the comparison is not like-for-like.')
    if args.method == 'dps' and not args.dps_framework:
        raise ValueError(
            '--method dps needs --dps_framework: the operator DPS inverts is '
            'what distinguishes the DPS variants, so it is never defaulted.')
    if args.input_mode == 'synth' and args.rs_framework:
        raise ValueError(
            '--rs_framework only applies to --input_mode roundtrip; in synth '
            'mode the input already is the measurement.')

    return roots[args.input_mode]


def main():
    args = parse_args()
    logger = get_logger()
    torch.manual_seed(args.seed)

    input_root = validate_args(args)
    label = args.label or (
        args.method if args.method == 'uvcgan'
        else f'dps_{args.dps_framework}')

    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu'))

    task_config = load_yaml(args.task_config)
    measure_config = task_config['measurement']

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, 'generated')
    os.makedirs(img_dir, exist_ok=True)
    meas_dir = os.path.join(args.out_dir, 'measurement')
    if args.input_mode == 'roundtrip':
        os.makedirs(meas_dir, exist_ok=True)

    model_config = load_yaml(args.model_config)
    image_size = model_config['image_size']
    dataset = get_dataset(name='crystal', root=input_root,
                          image_size=image_size, augment=False)
    # Spread the subset evenly over the sorted file list instead of taking the
    # first N. Crops are named "<photo>_sample_<k>", so the first N files are all
    # crops of the alphabetically-first photo(s): --limit 8 gave 2 distinct
    # scenes, --limit 32 gave 7. Striding gives N distinct scenes, and keeps the
    # subset identical across models so their outputs are directly comparable.
    if args.limit is not None and args.limit < len(dataset):
        stride = len(dataset) / args.limit
        indices = [int(i * stride) for i in range(args.limit)]
    else:
        indices = list(range(len(dataset)))
    n_inputs = len(indices)

    logger.info(f'Device: {device}')
    logger.info(f'Model under test: {label} '
                f'(method={args.method}'
                + (f', operator={args.dps_framework}' if args.method == 'dps' else '')
                + ')')
    logger.info(f'{n_inputs} inputs from {input_root} '
                f'(input_mode={args.input_mode}'
                + (f', y = G_RS[{args.rs_framework}](real)'
                   if args.input_mode == 'roundtrip' else '')
                + ')')

    # The R->S model that BUILDS the measurement. Deliberately separate from the
    # operator DPS inverts: holding it fixed across models is what makes the
    # comparison like-for-like.
    to_measurement = None
    if args.input_mode == 'roundtrip':
        rs_op = build_operator(measure_config, args.rs_framework, 'ba',
                               device, args.uvcgan_path)
        to_measurement = lambda ref: rs_op.forward(ref).detach()

    # Recorded in the manifest so a sweep's comparison table carries the swept
    # parameter as a real column, instead of leaving it encoded in directory
    # names for a reader to decode.
    used_scale = ''

    if args.method == 'uvcgan':
        sr_op = build_operator(measure_config, 'uvcgan2', 'ab', device,
                               args.uvcgan_path)
        generate = lambda s: sr_op.forward(s).detach()
    else:
        # The operator DPS inverts. Reused as the measurement builder only when
        # rs_framework happens to match, which build_operator does not assume.
        operator = build_operator(measure_config, args.dps_framework, 'ba',
                                  device, args.uvcgan_path)
        model = create_model(**model_config).to(device).eval()
        noiser = get_noise(**measure_config['noise'])

        cond_config = task_config['conditioning']
        params = dict(cond_config['params'])
        if args.scale is not None:
            params['scale'] = args.scale
        used_scale = params['scale']
        logger.info(f"Conditioning: {cond_config['method']} scale={used_scale}")

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
        writer.writerow([
            'output_image',
            'source_image',       # whose lattice defines the label
            'ground_truth',       # what the output should reconstruct (roundtrip)
            'measurement_image',  # the y actually fed to the model
            'label', 'method', 'dps_framework', 'rs_framework', 'scale',
            'sample_index', 'seed',
        ])

        for n, i in enumerate(indices):
            source_path = dataset.fpaths[i]
            s = dataset[i].unsqueeze(0).to(device)

            ground_truth, meas_rel = '', ''
            if to_measurement is not None:
                # In roundtrip the real image is both the label source (G_RS
                # preserves the lattice) and the reconstruction target.
                ground_truth = source_path
                s = to_measurement(s)
                meas_rel = f'{n:05d}.png'
                plt.imsave(os.path.join(meas_dir, meas_rel), to_display(s))

            logger.info(f'[{n + 1}/{n_inputs}] {os.path.basename(source_path)}')

            for k in range(args.samples_per_input):
                out = generate(s)
                fname = f'{n:05d}_{k:02d}.png'
                plt.imsave(os.path.join(img_dir, fname), to_display(out))
                writer.writerow([
                    fname, source_path, ground_truth, meas_rel,
                    label, args.method, args.dps_framework or '',
                    args.rs_framework or '', used_scale, k, args.seed,
                ])
            fh.flush()

    logger.info(f'Wrote {args.out_dir} (manifest: {manifest_path})')


if __name__ == '__main__':
    main()
