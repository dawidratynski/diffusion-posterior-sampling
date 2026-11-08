'''Drive the whole five-model comparison, skipping what is not yet available.

The point of this script is that NOTHING needs editing when UVCGAN weights
arrive. There are two UVCGAN models, trained with different adversarial losses
(standard GAN and Wasserstein GAN), and each gates its own pair of models
independently: point --uvcgan_gan_path / --uvcgan_wgan_path at whichever exist
and those models join the comparison. With neither, the analytic model runs
alone. Everything is resumable, so a re-run after weights land redoes only the
missing work.

MODELS
  uvcgan_gan        direct S->R translation, GAN-trained UVCGAN
  uvcgan_wgan       direct S->R translation, WGAN-trained UVCGAN
  dps_uvcgan_gan    DPS inverting the GAN UVCGAN's R->S generator
  dps_uvcgan_wgan   DPS inverting the WGAN UVCGAN's R->S generator
  dps_analytic      DPS inverting the analytic operator (needs no weights)

STAGES
  --stage sweep   Conditioning-scale sweep for each DPS variant: a coarse log
                  grid, then a refinement around its winner, on synth/TRAIN
                  inputs (disjoint from the synth/val inputs of the final
                  run). The scale is chosen by a fixed rule, identical for
                  every model (see select_scale), and written to
                  chosen_scales.json. The per-model tables are what justify
                  the choice in the report.

  --stage final   Every experiment for every available model, at the chosen
                  scales, then metrics and figures:

                    synth      synthetic image -> model. The actual product; no
                               ground truth, so distributional metrics only.
                               Builder-independent, so it runs once, first.
                    roundtrip_<builder>   (one per --rs_models entry)
                               real -> G_RS -> model, where G_RS is that
                               builder, held fixed across all five models.
                               Has ground truth, so PSNR/SSIM/LPIPS apply.
                               Running it once per UVCGAN removes the bias
                               any single builder has toward its own pair.

    python scripts/run_comparison.py --stage sweep \
        --data_root /content/data_raw/dataset --model_config ... \
        --out_dir /content/results/sweep \
        --uvcgan_gan_path /content/drive/.../uvcgan_gan \
        --uvcgan_wgan_path /content/drive/.../uvcgan_wgan

    python scripts/run_comparison.py --stage final \
        --data_root /content/data_raw/dataset --model_config ... \
        --out_dir /content/results/final --rs_models uvcgan_gan,uvcgan_wgan \
        --scale_dps_analytic 2.0 --scale_dps_uvcgan_gan 1.0 \
        --scale_dps_uvcgan_wgan 1.0
'''
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import yaml

from util.logger import get_logger

logger = get_logger()
HERE = os.path.dirname(os.path.abspath(__file__))

# `weights` names which trained UVCGAN a model depends on (None: needs none).
# The two UVCGANs share an architecture and differ only in adversarial loss, so
# both load through the same uvcgan2 framework; only the directory differs.
WEIGHTS = ('gan', 'wgan')
MODELS = [
    {'label': 'uvcgan_gan', 'method': 'uvcgan', 'dps_framework': None,
     'weights': 'gan', 'stochastic': False},
    {'label': 'uvcgan_wgan', 'method': 'uvcgan', 'dps_framework': None,
     'weights': 'wgan', 'stochastic': False},
    {'label': 'dps_uvcgan_gan', 'method': 'dps', 'dps_framework': 'uvcgan2',
     'weights': 'gan', 'stochastic': True},
    {'label': 'dps_uvcgan_wgan', 'method': 'dps', 'dps_framework': 'uvcgan2',
     'weights': 'wgan', 'stochastic': True},
    {'label': 'dps_analytic', 'method': 'dps', 'dps_framework': 'spectral',
     'weights': None, 'stochastic': True},
]
DPS_LABELS = [m['label'] for m in MODELS if m['method'] == 'dps']

# Round-trip measurement builders: y = G_RS(real). Value: (framework, weights).
RS_MODELS = {
    'uvcgan_gan': ('uvcgan2', 'gan'),
    'uvcgan_wgan': ('uvcgan2', 'wgan'),
    'spectral': ('spectral', None),
}


def weight_paths(args):
    '''{'gan': path-or-None, 'wgan': path-or-None}, None where absent.'''
    given = {'gan': args.uvcgan_gan_path, 'wgan': args.uvcgan_wgan_path}
    return {k: (p if p and os.path.exists(p) else None)
            for k, p in given.items()}


def resolve_rs_models(spec, paths):
    '''Parse a comma-separated builder list; 'auto' = every trained UVCGAN
    whose weights exist, or the analytic operator when there are none.'''
    if spec == 'auto':
        found = [name for name, (_, w) in RS_MODELS.items()
                 if w and paths[w]]
        return found or ['spectral']
    names = [s.strip() for s in spec.split(',') if s.strip()]
    unknown = [n for n in names if n not in RS_MODELS]
    if unknown or not names:
        raise ValueError(f'unknown round-trip builder(s) {unknown}; choose '
                         f"from {sorted(RS_MODELS)} or 'auto'.")
    return list(dict.fromkeys(names))  # dedupe, keep order


def check_uvcgan_identity(paths):
    '''Fail fast if a UVCGAN directory is not the model its flag claims.

    The GAN and WGAN runs produce identically named uvcgan2 directories (the
    loss is not part of uvcgan2's label), so they are told apart only by how
    they were renamed on upload. A swap would load, run and evaluate perfectly
    while mislabelling every GAN-vs-WGAN result. The loss IS recorded in the
    model's config.json, so check it -- before any GPU time is spent.
    '''
    for w, path in paths.items():
        if not path:
            continue
        cfg_path = os.path.join(path, 'config.json')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f'--uvcgan_{w}_path {path} has no config.json, so it is not a '
                'uvcgan2 model directory (point at the folder that contains '
                'config.json and net_gen_ab.pth, not the training outdir).')
        with open(cfg_path) as fh:
            loss = json.load(fh).get('loss')
        is_wgan = loss == 'wgan'
        if (w == 'wgan') != is_wgan:
            raise ValueError(
                f'--uvcgan_{w}_path {path} was trained with loss={loss!r}, '
                f'which is {"" if is_wgan else "not "}the Wasserstein loss -- '
                'the GAN and WGAN directories look swapped (or the wrong run '
                'was uploaded). Fix the folder names on Drive.')
        logger.info(f'{w.upper()} UVCGAN: {path} (loss={loss})')


def available(args):
    '''Returns (usable models, weight paths, round-trip builders).'''
    paths = weight_paths(args)
    check_uvcgan_identity(paths)
    usable = [m for m in MODELS if m['weights'] is None or paths[m['weights']]]
    for w in WEIGHTS:
        if paths[w]:
            continue
        skipped = [m['label'] for m in MODELS if m['weights'] == w]
        given = getattr(args, f'uvcgan_{w}_path')
        logger.warning(
            f'No {w.upper()} UVCGAN weights at {given!r} -- skipping '
            f'{skipped}. Re-run with --uvcgan_{w}_path once they exist; '
            'completed work is kept.')

    rs_models = resolve_rs_models(args.rs_models, paths)

    # Fail here rather than inside the first generation: the R->S model that
    # builds the round-trip measurement is needed even when every model that
    # needs weights has been skipped, so "no weights" does not by itself make
    # the run viable. The sweep runs on synth inputs and needs no builder.
    needed = rs_models if args.stage == 'final' else []
    for rs in needed:
        _, rs_weights = RS_MODELS[rs]
        if rs_weights and not paths[rs_weights]:
            raise ValueError(
                f'Round-trip builder {rs} needs the {rs_weights.upper()} '
                f'UVCGAN, but no weights were found at '
                f'--uvcgan_{rs_weights}_path. Pass that path, drop {rs} from '
                "the builder list, or use 'auto'.")
    return usable, paths, rs_models


def run(cmd):
    logger.info('  $ ' + ' '.join(str(c) for c in cmd[-8:]))
    # check=False: the raise below reports the failing command, which is more
    # useful here than CalledProcessError's traceback through this driver.
    result = subprocess.run([sys.executable] + cmd, cwd=os.path.dirname(HERE),
                            check=False)
    if result.returncode != 0:
        raise RuntimeError(f'failed: {" ".join(str(c) for c in cmd)}')


def _fingerprint(path):
    """Hash of a config's PARSED content, so editing it invalidates caches.

    Parsed, not raw bytes: a comment-only edit used to change the hash and
    throw away every cached run (up to ~22 GPU-h). The configs name the prior
    checkpoint, so a changed prior is caught provided its PATH changes
    (checkpoints are model_ema_<step>.pt, so in practice it does).
    """
    with open(path) as fh:
        content = yaml.safe_load(fh)
    blob = json.dumps(content, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _weights_fingerprint(model_dir):
    """Size + mtime of the files a uvcgan2 load actually reads.

    The directory path alone missed weights replaced in place (e.g. an early
    checkpoint uploaded first, then the final model into the same folder):
    every cached result from the old weights was silently reused.
    """
    if not model_dir:
        return None
    files = sorted(glob.glob(os.path.join(model_dir, 'config.json'))
                   + glob.glob(os.path.join(model_dir, 'net_*gen_*.pth'))
                   + glob.glob(os.path.join(model_dir, 'checkpoints',
                                            '*_net_*gen_*.pth')))
    stat = [(os.path.relpath(f, model_dir), os.path.getsize(f),
             int(os.path.getmtime(f))) for f in files]
    return hashlib.sha256(json.dumps(stat).encode()).hexdigest()[:16]


def _prior_path(model_config):
    with open(model_config) as fh:
        return (yaml.safe_load(fh) or {}).get('model_path')


def _code_version():
    """Git commit of this repo, for provenance only (never compared)."""
    try:
        return subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(HERE),
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# Recorded in run_config.json but not part of cache validity.
INFO_KEYS = ('code_version',)


def run_signature(model, args, paths, input_mode, rs_model, scale, samples,
                  limit, split='val'):
    """Everything that determines the output of one generation run.

    Only what actually affects THIS run goes in, so e.g. GAN weights arriving
    does not invalidate a finished dps_analytic synth run. `rs_model` is the
    round-trip builder, ignored in synth mode. `split` is which data split the
    inputs come from: the sweep tunes on train, the final runs test on val.
    """
    roundtrip = input_mode == 'roundtrip'
    rs_weights = RS_MODELS[rs_model][1] if roundtrip else None
    is_dps = model['method'] == 'dps'
    rs_path = paths[rs_weights] if roundtrip and rs_weights else None
    model_path = paths[model['weights']] if model['weights'] else None
    return {
        'label': model['label'],
        'method': model['method'],
        'dps_framework': model['dps_framework'],
        'input_mode': input_mode,
        'split': split,
        'rs_model': rs_model if roundtrip else None,
        'rs_path': rs_path,
        'rs_weights': _weights_fingerprint(rs_path),
        'model_path': model_path,
        'model_weights': _weights_fingerprint(model_path),
        'scale': scale if is_dps else None,
        'samples': samples if model['stochastic'] else 1,
        'limit': limit,
        # The diffusion prior and sampler only matter to DPS models.
        'model_config': _fingerprint(args.model_config) if is_dps else None,
        'diffusion_config': (_fingerprint(args.diffusion_config)
                             if is_dps else None),
        'task_config': _fingerprint(args.task_config),
    }


def _same_settings(old, sig):
    keys = (set(old) | set(sig)) - set(INFO_KEYS)
    return sorted(k for k in keys if old.get(k) != sig.get(k))


def _read_manifest(out_dir):
    path = os.path.join(out_dir, 'manifest.csv')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _cached_run_is_valid(out_dir, sig):
    """Is an existing result dir complete AND produced by these settings?

    Ways a naive "does manifest.csv exist" check goes wrong, each of which has
    cost (or could cost) a run of experiments:

      - an interrupted run leaves a partial manifest (it is flushed per
        sample), which would then be evaluated as if complete;
      - a manifest produced with a different prior, scale, subset size or
        weights looks identical from the outside and would be silently reused;
      - a manifest whose images did not all reach disk (Drive sync on a hard
        disconnect) passes a row count and crashes evaluation later.
    """
    rows = _read_manifest(out_dir)
    sig_path = os.path.join(out_dir, 'run_config.json')
    if rows is None:
        return False, None
    if not os.path.exists(sig_path):
        return False, 'no completed run_config.json'
    with open(sig_path) as fh:
        old = json.load(fh)
    changed = _same_settings(old, sig)
    if changed:
        return False, f'settings changed: {changed}'
    expected = sig['limit'] * sig['samples']
    if len(rows) != expected:
        return False, f'incomplete: {len(rows)} of {expected} rows'
    missing = [r['output_image'] for r in rows if not os.path.exists(
        os.path.join(out_dir, 'generated', r['output_image']))]
    if missing:
        return False, f'{len(missing)} image(s) missing on disk'
    return True, None


def generate(model, out_dir, args, paths, input_mode, rs_model, scale, samples,
             limit, split='val'):
    """One model, one experiment. Skips if already done, so re-runs are cheap.

    An interrupted run RESUMES: a provisional signature (run_config.partial
    .json) is written before generation starts, and if it matches on the next
    attempt the generator appends to the existing manifest, skipping samples
    already on disk. Per-sample seeding makes the resumed output identical to
    an uninterrupted one. Anything that does not match starts from scratch.
    """
    sig = run_signature(model, args, paths, input_mode, rs_model, scale,
                        samples, limit, split)
    valid, why = _cached_run_is_valid(out_dir, sig)
    if valid:
        logger.info(f'  {model["label"]}: already present, skipping')
        return True

    final_sig = os.path.join(out_dir, 'run_config.json')
    partial_sig = os.path.join(out_dir, 'run_config.partial.json')
    resume = False
    if os.path.exists(partial_sig):
        with open(partial_sig) as fh:
            resume = not _same_settings(json.load(fh), sig)
    if resume:
        n_done = len(_read_manifest(out_dir) or [])
        logger.warning(f'  {model["label"]}: resuming interrupted run '
                       f'({n_done} of {limit * sig["samples"]} samples done)')
    else:
        if why:
            logger.warning(f'  {model["label"]}: regenerating -- {why}')
        # Start clean: a stale completed signature next to a new manifest
        # would otherwise pass as valid if this run died at the very end.
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(final_sig):
        os.remove(final_sig)
    with open(partial_sig, 'w') as fh:
        json.dump(sig, fh, indent=2, sort_keys=True)

    cmd = [os.path.join(HERE, 'generate_augmented.py'),
           '--model_config', args.model_config,
           '--diffusion_config', args.diffusion_config,
           '--task_config', args.task_config,
           '--out_dir', out_dir,
           '--label', model['label'],
           '--method', model['method'],
           '--input_mode', input_mode,
           '--limit', str(limit)]
    if resume:
        cmd += ['--resume']

    if input_mode == 'roundtrip':
        rs_framework, rs_weights = RS_MODELS[rs_model]
        cmd += ['--real_root', os.path.join(args.data_root, 'real', split),
                '--rs_framework', rs_framework]
        if rs_weights:
            cmd += ['--rs_path', paths[rs_weights]]
    else:
        cmd += ['--synth_root', os.path.join(args.data_root, 'synth', split)]

    if model['method'] == 'dps':
        cmd += ['--dps_framework', model['dps_framework'],
                '--scale', str(scale),
                '--samples_per_input', str(samples if model['stochastic'] else 1)]
    if model['weights']:
        cmd += ['--uvcgan_path', paths[model['weights']]]

    run(cmd)
    # Completed signature written only after the run succeeds; the partial one
    # is then dropped.
    with open(final_sig, 'w') as fh:
        json.dump({**sig, 'code_version': _code_version()}, fh, indent=2,
                  sort_keys=True)
    os.remove(partial_sig)
    return True


def _stamp(result_dirs, **extra):
    """What a derived output (metrics, figures) was computed from: the
    completed signature of every member run, plus the settings and code
    version. Equal stamp => the output is still current."""
    runs = {}
    for d in result_dirs:
        with open(os.path.join(d, 'run_config.json')) as fh:
            sig = json.load(fh)
        # e.g. 'synth/dps_analytic' or 'dps_analytic/scale1': unique within
        # one evaluation and independent of where out_dir is mounted.
        key = '/'.join(os.path.normpath(d).split(os.sep)[-2:])
        runs[key] = {k: v for k, v in sig.items() if k not in INFO_KEYS}
    return {'runs': runs, 'code_version': _code_version(), **extra}


def _is_current(outputs, stamp_path, stamp):
    if not all(os.path.exists(o) for o in outputs):
        return False
    if not os.path.exists(stamp_path):
        return False
    with open(stamp_path) as fh:
        return json.load(fh) == stamp


def _write_stamp(stamp_path, stamp):
    with open(stamp_path, 'w') as fh:
        json.dump(stamp, fh, indent=2, sort_keys=True)


def evaluate(result_dirs, args, out_csv, summary_csv, paired, half):
    """Score result dirs. Skipped when nothing it depends on has changed:
    re-evaluating every cached run on every notebook session cost ~2-3 h of
    runtime per session for identical numbers."""
    stamp = _stamp(result_dirs, paired=paired, half=half,
                   fid_kid=args.fid_kid, lpips=bool(paired and args.lpips))
    stamp_path = summary_csv + '.stamp.json'
    if _is_current([out_csv, summary_csv], stamp_path, stamp):
        logger.info(f'  metrics up to date: {os.path.basename(summary_csv)}')
        return
    cmd = [os.path.join(HERE, 'evaluate.py'),
           '--real_root', os.path.join(args.data_root, 'real', 'val'),
           '--real_train_root', os.path.join(args.data_root, 'real', 'train'),
           '--reference_half', half,
           '--out_csv', out_csv, '--out_summary_csv', summary_csv]
    for d in result_dirs:
        cmd += ['--result_dir', d]
    if args.fid_kid:
        cmd += ['--fid_kid']
    # LPIPS needs a per-output target, so it only applies to the round trip.
    if paired and args.lpips:
        cmd += ['--lpips']
    run(cmd)
    _write_stamp(stamp_path, stamp)


# ---------------------------------------------------------------- selection --
# The conditioning scale is chosen by this rule, fixed in advance and applied
# identically to every DPS model, so no model is tuned "harder" than another.
# docs/methods.md §4.1 states it for the report; keep the two in sync.
#
#   1. Admissible: label validity holds and the prior is not copying.
#        signature_err_median <= SIG_ERR_MAX      (lattice preserved)
#        nn_dist_median >= NN_FLOOR_RATIO * floor (not memorising)
#   2. Among admissible scales, the most realistic: lowest KID.
#   3. KID differences smaller than the best scale's subset spread are noise,
#      so among scales within that spread pick the best label validity.
#   Without KID (--no_fid_kid), realism falls back to prominence closeness to
#   real, |log(prominence / real prominence)|.
#
# Diversity is reported but deliberately not a criterion: it has a sensible
# range rather than a direction, and a near-zero value already shows up as
# poor realism. It only raises a warning if it collapses.
SIG_ERR_MAX = 0.05        # signature-error floor on synth sources: p90 0.013 (methods §5.1)
NN_FLOOR_RATIO = 0.98     # "well below the floor" = more than 2% under it
DIVERSITY_COLLAPSE = 0.108   # two crops of the SAME real photo (methods §5.5)


def _num(row, key):
    '''A float from a CSV row, or None for missing/empty/non-finite.'''
    v = row.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def select_scale(rows):
    '''Apply the rule above to one model's sweep table.

    rows: dicts with at least 'scale' and the metric columns evaluate.py
    writes. Returns {'scale', 'reason', 'warnings'}.
    '''
    rows = [r for r in rows if _num(r, 'scale') is not None]
    if not rows:
        raise ValueError('sweep table has no rows with a scale')
    warnings = []

    def sig(r):
        v = _num(r, 'signature_err_median')
        return v if v is not None else math.inf

    def admissible(r):
        if sig(r) > SIG_ERR_MAX:
            return False
        nn, floor = _num(r, 'nn_dist_median'), _num(r, 'nn_floor')
        return nn is None or floor is None or nn >= NN_FLOOR_RATIO * floor

    ok = [r for r in rows if admissible(r)]
    if not ok:
        best = min(rows, key=sig)
        warnings.append(
            f'NO scale passed the admissibility check (signature error <= '
            f'{SIG_ERR_MAX}, NN distance >= {NN_FLOOR_RATIO} x floor). Fell '
            'back to the best label validity, which tends to be the STRONGEST '
            'guidance -- i.e. outputs that copy the input rather than look '
            'real. Not usable as-is: inspect the scale ladder and set a scale '
            'explicitly.')
        reason = 'fallback: lowest signature error (none admissible)'
    elif all(_num(r, 'kid') is not None for r in ok):
        kid_best = min(ok, key=lambda r: _num(r, 'kid'))
        tol = _num(kid_best, 'kid') + (_num(kid_best, 'kid_std') or 0.0)
        near = [r for r in ok if _num(r, 'kid') <= tol]
        best = min(near, key=lambda r: (sig(r), _num(r, 'kid')))
        reason = (f'{len(ok)}/{len(rows)} admissible; {len(near)} within KID '
                  f'noise of the best ({_num(kid_best, "kid"):.5f} + '
                  f'{(_num(kid_best, "kid_std") or 0.0):.5f}); lowest '
                  'signature error among those')
    else:
        def prom_gap(r):
            p, real = _num(r, 'prominence_median'), _num(r, 'real_prominence_median')
            return abs(math.log(p / real)) if p and real else math.inf
        best = min(ok, key=prom_gap)
        reason = (f'{len(ok)}/{len(rows)} admissible; no KID, so closest '
                  'prominence to real')

    scale = _num(best, 'scale')
    tried = sorted(_num(r, 'scale') for r in rows)
    if scale in (tried[0], tried[-1]) and len(tried) > 1:
        warnings.append(
            f'chosen scale {scale:g} is at the edge of the grid {tried} -- the '
            'optimum may lie beyond it.')
    div = _num(best, 'diversity_rmse')
    if div is not None and div < DIVERSITY_COLLAPSE:
        warnings.append(
            f'diversity {div:.3f} is below two crops of one real photo '
            f'({DIVERSITY_COLLAPSE}) -- variants have nearly collapsed.')
    return {'scale': scale, 'reason': reason, 'warnings': warnings}


def refine_scales(coarse, chosen):
    '''Scales for the second sweep stage, around the coarse winner.

    The geometric midpoint toward each neighbour. At an edge of the grid, one
    step further out instead, by the same ratio as the last interval -- the
    optimum may be off-grid, and the operators differ enough (trained
    generators versus the analytic one) that it plausibly is.
    '''
    grid = sorted(set(coarse))
    i = grid.index(chosen)
    new = []
    if i > 0:
        new.append(math.sqrt(grid[i - 1] * chosen))
    elif len(grid) > 1:
        new.append(chosen * chosen / grid[1])
    if i < len(grid) - 1:
        new.append(math.sqrt(chosen * grid[i + 1]))
    elif len(grid) > 1:
        new.append(chosen * chosen / grid[-2])
    # 3 significant figures: readable directory names, and stable across
    # re-runs so the cached runs are found again.
    return sorted({float(f'{s:.3g}') for s in new} - set(grid))


def _read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _sweep_round(model, scales, args, paths, tag):
    '''Generate (cached) and evaluate one model at the given scales.'''
    dirs = []
    for scale in scales:
        d = os.path.join(args.out_dir, model['label'], f'scale{scale:g}')
        # Synth inputs from the TRAIN split: the final runs use synth/val, and
        # the split is at source-photo level, so tuning and test images are
        # disjoint down to the photo.
        generate(model, d, args, paths, 'synth', None, scale,
                 args.sweep_samples, args.sweep_limit, split='train')
        dirs.append(d)
    table = os.path.join(args.out_dir, f'{model["label"]}_{tag}.csv')
    # Reference half A: the final experiments are scored against half B, so
    # the realism target the scale is tuned on is not the one it is reported
    # against (disjoint down to the source photo).
    evaluate(dirs, args,
             os.path.join(args.out_dir, model['label'], f'{tag}_per_image.csv'),
             table, paired=False, half='A')
    return table


def stage_sweep(args):
    '''Scale sweep per DPS variant, each selected by the same fixed rule.

    Two rounds per model: a coarse log grid, then the geometric midpoints
    around its winner (or one step past the edge if the winner is at one).
    The final choice is made over both rounds together and written to
    chosen_scales.json, which section 14 of the notebook reads.

    Runs on synth inputs -- the deliverable's own input distribution -- so it
    needs no round-trip builder. Every metric the rule uses is unpaired.
    '''
    usable, paths, _ = available(args)
    dps_models = [m for m in usable if m['method'] == 'dps']
    coarse = sorted(float(s) for s in args.scales.split(','))

    logger.info(f'Sweeping {[m["label"] for m in dps_models]} over {coarse}'
                + (' + refinement' if args.refine else ''))
    logger.info(f'  {args.sweep_limit} synth/train inputs x '
                f'{args.sweep_samples} variants per scale')

    chosen_path = os.path.join(args.out_dir, 'chosen_scales.json')
    chosen = {}
    if os.path.exists(chosen_path):
        # Keep entries for models not swept this time (e.g. weights not yet
        # present), so a partial run never erases an earlier choice.
        with open(chosen_path) as fh:
            chosen = json.load(fh)

    for model in dps_models:
        label = model['label']
        logger.info(f'=== {label}: coarse ===')
        table = _sweep_round(model, coarse, args, paths, 'sweep_coarse')
        first = select_scale(_read_csv(table))
        scales = coarse
        if args.refine:
            extra = refine_scales(coarse, first['scale'])
            logger.info(f'=== {label}: refine around {first["scale"]:g} '
                        f'-> {extra} ===')
            scales = sorted(coarse + extra)
        # Re-evaluated together, so every scale is judged against the same
        # reference statistics in one table. Generation is cached.
        table = _sweep_round(model, scales, args, paths, 'sweep')
        pick = select_scale(_read_csv(table))
        pick['coarse_choice'] = first['scale']
        pick['scales_tried'] = scales
        pick['table'] = os.path.basename(table)
        # Which prior/sampler the choice was made with, so a stale pick (prior
        # changed since) is detectable when section 14 reads this file.
        pick['prior_model_path'] = _prior_path(args.model_config)
        pick['model_config'] = _fingerprint(args.model_config)
        pick['diffusion_config'] = _fingerprint(args.diffusion_config)
        chosen[label] = pick

        logger.info(f'{label}: scale {pick["scale"]:g} -- {pick["reason"]}')
        for w in pick['warnings']:
            logger.warning(f'{label}: {w}')

    with open(chosen_path, 'w') as fh:
        json.dump(chosen, fh, indent=2, sort_keys=True)
    logger.info(f'Chosen scales: {chosen_path}')
    for label, pick in sorted(chosen.items()):
        flag = '  (!) see warnings' if pick.get('warnings') else ''
        logger.info(f'  {label:<16} {pick["scale"]:g}{flag}')


def stage_final(args):
    usable, paths, rs_models = available(args)
    chosen = {label: getattr(args, f'scale_{label}') for label in DPS_LABELS}

    # Synth first: it is the deliverable and builder-independent, so if a
    # session dies mid-run the headline result is the one already finished.
    # Then one round trip per measurement builder, each keyed by it.
    experiments = ([('synth', 'synth', None)]
                   + [(f'roundtrip_{rs}', 'roundtrip', rs) for rs in rs_models])
    logger.info(f'Experiments: {[e[0] for e in experiments]}')
    for name, input_mode, rs in experiments:
        logger.info(f'=== {name} ===')
        dirs = []
        for model in usable:
            d = os.path.join(args.out_dir, name, model['label'])
            generate(model, d, args, paths, input_mode, rs,
                     chosen.get(model['label']),
                     args.samples, args.limit)
            dirs.append(d)

        evaluate(dirs, args,
                 os.path.join(args.out_dir, f'{name}_per_image.csv'),
                 os.path.join(args.out_dir, f'{name}_metrics.csv'),
                 paired=(input_mode == 'roundtrip'), half='B')

        if not args.no_figures:
            fig_dir = os.path.join(args.out_dir, 'figures', name)
            stamp = _stamp(dirs, rows=args.rows_per_grid,
                           limit=args.figure_rows)
            stamp_path = fig_dir + '.stamp.json'
            if _is_current([fig_dir], stamp_path, stamp):
                logger.info(f'  figures up to date: {name}')
                continue
            # Start clean: panel files are numbered by column, so adding a
            # model used to leave the previous run's panels mixed in.
            if os.path.isdir(fig_dir):
                shutil.rmtree(fig_dir)
            cmd = [os.path.join(HERE, 'make_figures.py'),
                   '--out_dir', fig_dir,
                   '--rows_per_grid', str(args.rows_per_grid),
                   '--limit', str(args.figure_rows)]
            for d in dirs:
                cmd += ['--result_dir', d]
            run(cmd)
            _write_stamp(stamp_path, stamp)

    logger.info(f'\nEverything under {args.out_dir}')
    logger.info('  roundtrip_<builder>_metrics.csv / synth_metrics.csv '
                '-> report tables')
    logger.info('  figures/<experiment>/grids  -> report figures')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['sweep', 'final'], required=True)
    p.add_argument('--data_root', type=str, required=True,
                   help='Dataset root containing real/ and synth/.')
    p.add_argument('--model_config', type=str, required=True)
    p.add_argument('--diffusion_config', type=str, required=True)
    p.add_argument('--task_config', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--uvcgan_gan_path', type=str, default=None,
                   help='GAN-trained UVCGAN model directory. Absent => '
                        'uvcgan_gan and dps_uvcgan_gan are skipped and '
                        'everything else still runs.')
    p.add_argument('--uvcgan_wgan_path', type=str, default=None,
                   help='WGAN-trained UVCGAN model directory. Absent => '
                        'uvcgan_wgan and dps_uvcgan_wgan are skipped.')
    p.add_argument('--rs_models', type=str, default='auto',
                   help='final: comma-separated R->S models that build the '
                        f'round-trip measurement, from {sorted(RS_MODELS)}. '
                        'One full round trip runs per builder, each held '
                        'FIXED across all five models so it is like-for-like, '
                        'into roundtrip_<builder>/. auto = every UVCGAN whose '
                        'weights exist (spectral if none).')
    p.add_argument('--scales', type=str, default='0.1,0.3,1,3,10',
                   help='sweep: the COARSE grid, roughly log-spaced. The '
                        'operators differ enough that one model\'s optimum '
                        'says little about another\'s, so start wide.')
    p.add_argument('--refine', action='store_true', default=True,
                   help='sweep: add a second round around each coarse winner '
                        '(default on).')
    p.add_argument('--no_refine', dest='refine', action='store_false')
    p.add_argument('--sweep_limit', type=int, default=24,
                   help='sweep: synth/train inputs per scale (distinct photos).')
    p.add_argument('--sweep_samples', type=int, default=2,
                   help='sweep: variants per input; >= 2 so diversity is '
                        'measurable.')

    # Separate decisions: each DPS variant inverts a different operator.
    p.add_argument('--scale_dps_analytic', type=float, default=2.0)
    p.add_argument('--scale_dps_uvcgan_gan', type=float, default=1.0)
    p.add_argument('--scale_dps_uvcgan_wgan', type=float, default=1.0)
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
