'''Drive the whole three-model comparison, skipping what is not yet available.

The point of this script is that NOTHING needs editing when the UVCGAN weights
arrive. Point --uvcgan_path at them and the two gated models join the comparison;
leave it unset and the analytic model runs alone. Everything is resumable, so a
re-run after the weights land redoes only the missing work.

MODELS
  uvcgan          direct S->R translation (needs weights)
  dps_uvcgan      DPS inverting the trained R->S generator (needs weights)
  dps_analytic    DPS inverting the analytic operator (runs today)

STAGES
  --stage sweep   Conditioning-scale sweep for each DPS variant, with the full
                  metric suite per scale. The table this writes is what
                  justifies the chosen scale in the report.

  --stage final   Both experiments for every available model, at the chosen
                  scales, then metrics and figures:

                    roundtrip  real -> G_RS -> model. Has ground truth, so
                               PSNR/SSIM/LPIPS apply.
                    synth      synthetic image -> model. The actual product; no
                               ground truth, so distributional metrics only.

    python scripts/run_comparison.py --stage sweep \
        --data_root /content/data_raw/dataset --model_config ... \
        --out_dir /content/results/sweep --uvcgan_path /content/drive/.../uvcgan

    python scripts/run_comparison.py --stage final \
        --data_root /content/data_raw/dataset --model_config ... \
        --out_dir /content/results/final \
        --scale_dps_analytic 2.0 --scale_dps_uvcgan 1.0
'''
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from util.logger import get_logger

logger = get_logger()
HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    {'label': 'uvcgan', 'method': 'uvcgan', 'dps_framework': None,
     'needs_weights': True, 'stochastic': False},
    {'label': 'dps_uvcgan', 'method': 'dps', 'dps_framework': 'uvcgan2',
     'needs_weights': True, 'stochastic': True},
    {'label': 'dps_analytic', 'method': 'dps', 'dps_framework': 'spectral',
     'needs_weights': False, 'stochastic': True},
]


def available(uvcgan_path, rs_framework=None):
    have = bool(uvcgan_path) and os.path.exists(uvcgan_path)
    usable = [m for m in MODELS if have or not m['needs_weights']]
    skipped = [m['label'] for m in MODELS if m not in usable]
    if skipped:
        logger.warning(
            f'No UVCGAN weights at {uvcgan_path!r} -- skipping {skipped}. '
            'Re-run with --uvcgan_path once they exist; completed work is kept.')

    # Fail here rather than inside the first generation: the R->S model that
    # builds the round-trip measurement is needed even when every model that
    # *needs weights* has been skipped, so "no weights" does not by itself make
    # the run viable.
    if rs_framework and rs_framework != 'spectral' and not have:
        raise ValueError(
            f'--rs_framework {rs_framework} needs UVCGAN weights to build the '
            'round-trip measurement, but none were found at '
            f'{uvcgan_path!r}. Either pass --uvcgan_path, or use '
            '--rs_framework spectral to run the analytic-R->S variant.')
    return usable, have


def run(cmd):
    logger.info('  $ ' + ' '.join(str(c) for c in cmd[-8:]))
    # check=False: the raise below reports the failing command, which is more
    # useful here than CalledProcessError's traceback through this driver.
    result = subprocess.run([sys.executable] + cmd, cwd=os.path.dirname(HERE),
                            check=False)
    if result.returncode != 0:
        raise RuntimeError(f'failed: {" ".join(str(c) for c in cmd)}')


def _fingerprint(path):
    '''Content hash of a config file, so editing it invalidates cached results.

    The configs name the checkpoint, so this also catches a changed prior --
    provided the checkpoint PATH changes. Checkpoints are named
    model_ema_<step>.pt, so in practice it does; overwriting a checkpoint in
    place would not be detected.
    '''
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def run_signature(model, args, input_mode, scale, samples, limit):
    '''Everything that determines the output of one generation run.'''
    return {
        'label': model['label'],
        'method': model['method'],
        'dps_framework': model['dps_framework'],
        'input_mode': input_mode,
        'rs_framework': args.rs_framework if input_mode == 'roundtrip' else None,
        'scale': scale if model['method'] == 'dps' else None,
        'samples': samples if model['stochastic'] else 1,
        'limit': limit,
        'uvcgan_path': args.uvcgan_path or None,
        'model_config': _fingerprint(args.model_config),
        'diffusion_config': _fingerprint(args.diffusion_config),
        'task_config': _fingerprint(args.task_config),
    }


def _cached_run_is_valid(out_dir, sig):
    '''Is an existing result dir complete AND produced by these settings?

    Two ways a naive "does manifest.csv exist" check goes wrong, both of which
    have already cost a run of experiments in this project:

      - an interrupted run leaves a partial manifest (it is flushed per source),
        which would then be evaluated as if complete;
      - a manifest produced with a different prior, scale or subset size looks
        identical from the outside and would be silently reused.
    '''
    manifest = os.path.join(out_dir, 'manifest.csv')
    sig_path = os.path.join(out_dir, 'run_config.json')
    if not os.path.exists(manifest):
        return False, None

    if not os.path.exists(sig_path):
        return False, 'no run_config.json (predates provenance tracking)'
    with open(sig_path) as fh:
        old = json.load(fh)
    if old != sig:
        changed = sorted(k for k in set(old) | set(sig)
                         if old.get(k) != sig.get(k))
        return False, f'settings changed: {changed}'

    with open(manifest) as fh:
        n_rows = sum(1 for _ in csv.DictReader(fh))
    expected = sig['limit'] * sig['samples']
    if n_rows != expected:
        return False, f'incomplete: {n_rows} of {expected} rows'
    return True, None


def generate(model, out_dir, args, input_mode, scale, samples, limit):
    '''One model, one experiment. Skips if already done, so re-runs are cheap.'''
    sig = run_signature(model, args, input_mode, scale, samples, limit)
    valid, why = _cached_run_is_valid(out_dir, sig)
    if valid:
        logger.info(f'  {model["label"]}: already present, skipping')
        return True
    if why:
        logger.warning(f'  {model["label"]}: regenerating -- {why}')

    cmd = [os.path.join(HERE, 'generate_augmented.py'),
           '--model_config', args.model_config,
           '--diffusion_config', args.diffusion_config,
           '--task_config', args.task_config,
           '--out_dir', out_dir,
           '--label', model['label'],
           '--method', model['method'],
           '--input_mode', input_mode,
           '--limit', str(limit)]

    if input_mode == 'roundtrip':
        cmd += ['--real_root', os.path.join(args.data_root, 'real', 'val'),
                '--rs_framework', args.rs_framework]
    else:
        cmd += ['--synth_root', os.path.join(args.data_root, 'synth', 'val')]

    if model['method'] == 'dps':
        cmd += ['--dps_framework', model['dps_framework'],
                '--scale', str(scale),
                '--samples_per_input', str(samples if model['stochastic'] else 1)]
    if args.uvcgan_path:
        cmd += ['--uvcgan_path', args.uvcgan_path]

    run(cmd)
    # Written only after the run succeeds, so a crash leaves no signature and
    # the partial output is regenerated rather than trusted.
    with open(os.path.join(out_dir, 'run_config.json'), 'w') as fh:
        json.dump(sig, fh, indent=2, sort_keys=True)
    return True


def evaluate(result_dirs, args, out_csv, summary_csv, paired):
    cmd = [os.path.join(HERE, 'evaluate.py'),
           '--real_root', os.path.join(args.data_root, 'real', 'val'),
           '--real_train_root', os.path.join(args.data_root, 'real', 'train'),
           '--out_csv', out_csv, '--out_summary_csv', summary_csv]
    for d in result_dirs:
        cmd += ['--result_dir', d]
    if args.fid_kid:
        cmd += ['--fid_kid']
    # LPIPS needs a per-output target, so it only applies to the round trip.
    if paired and args.lpips:
        cmd += ['--lpips']
    run(cmd)


def stage_sweep(args):
    '''Scale sweep per DPS variant. Each variant gets its own optimum.'''
    usable, _ = available(args.uvcgan_path, args.rs_framework)
    dps_models = [m for m in usable if m['method'] == 'dps']
    scales = [float(s) for s in args.scales.split(',')]

    logger.info(f'Sweeping {[m["label"] for m in dps_models]} over {scales}')
    logger.info(f'  {args.sweep_limit} refs x {args.sweep_samples} variants '
                f'x {len(scales)} scales per model')

    for model in dps_models:
        dirs = []
        for scale in scales:
            d = os.path.join(args.out_dir, f'{model["label"]}_scale{scale}')
            generate(model, d, args, 'roundtrip', scale,
                     args.sweep_samples, args.sweep_limit)
            dirs.append(d)
        evaluate(dirs, args,
                 os.path.join(args.out_dir, f'{model["label"]}_per_image.csv'),
                 os.path.join(args.out_dir, f'{model["label"]}_sweep.csv'),
                 paired=True)
        logger.info(f'{model["label"]}: table at '
                    f'{args.out_dir}/{model["label"]}_sweep.csv')


def stage_final(args):
    usable, _ = available(args.uvcgan_path, args.rs_framework)
    chosen = {'dps_analytic': args.scale_dps_analytic,
              'dps_uvcgan': args.scale_dps_uvcgan}

    for input_mode in ('roundtrip', 'synth'):
        logger.info(f'=== {input_mode} ===')
        dirs = []
        for model in usable:
            d = os.path.join(args.out_dir, input_mode, model['label'])
            generate(model, d, args, input_mode,
                     chosen.get(model['label'], 1.0),
                     args.samples, args.limit)
            dirs.append(d)

        evaluate(dirs, args,
                 os.path.join(args.out_dir, f'{input_mode}_per_image.csv'),
                 os.path.join(args.out_dir, f'{input_mode}_metrics.csv'),
                 paired=(input_mode == 'roundtrip'))

        if not args.no_figures:
            cmd = [os.path.join(HERE, 'make_figures.py'),
                   '--out_dir', os.path.join(args.out_dir, 'figures', input_mode),
                   '--rows_per_grid', str(args.rows_per_grid),
                   '--limit', str(args.figure_rows)]
            for d in dirs:
                cmd += ['--result_dir', d]
            run(cmd)

    logger.info(f'\nEverything under {args.out_dir}')
    logger.info('  roundtrip_metrics.csv / synth_metrics.csv -> report tables')
    logger.info('  figures/<mode>/grids  -> report figures')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['sweep', 'final'], required=True)
    p.add_argument('--data_root', type=str, required=True,
                   help='Dataset root containing real/ and synth/.')
    p.add_argument('--model_config', type=str, required=True)
    p.add_argument('--diffusion_config', type=str, required=True)
    p.add_argument('--task_config', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--uvcgan_path', type=str, default=None,
                   help='UVCGAN model directory. Absent => the two models that '
                        'need it are skipped and everything else still runs.')
    p.add_argument('--rs_framework', type=str, default='uvcgan2',
                   help='Which R->S model builds the round-trip measurement. '
                        'Held FIXED across models so the comparison is '
                        'like-for-like. Set to spectral to also run the '
                        'analytic-R->S variant of the experiment.')

    p.add_argument('--scales', type=str, default='0.3,1.0,2.0,3.0',
                   help='sweep: conditioning scales to try.')
    p.add_argument('--sweep_limit', type=int, default=24)
    p.add_argument('--sweep_samples', type=int, default=2)

    p.add_argument('--scale_dps_analytic', type=float, default=2.0)
    p.add_argument('--scale_dps_uvcgan', type=float, default=1.0)
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--figure_rows', type=int, default=12)
    p.add_argument('--rows_per_grid', type=int, default=6)
    p.add_argument('--no_figures', action='store_true')

    p.add_argument('--lpips', action='store_true', default=True)
    p.add_argument('--no_lpips', dest='lpips', action='store_false')
    p.add_argument('--fid_kid', action='store_true', default=True)
    p.add_argument('--no_fid_kid', dest='fid_kid', action='store_false')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    {'sweep': stage_sweep, 'final': stage_final}[args.stage](args)


if __name__ == '__main__':
    main()
