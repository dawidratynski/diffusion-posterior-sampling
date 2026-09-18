'''Compare generated-data methods on the axes that actually decide usefulness.

There is no ground-truth output to score against -- given a synthetic input
there is no single correct realistic image -- so reconstruction error is the
wrong frame. What matters instead:

  1. LABEL VALIDITY (the decisive one). Does the generated image still carry the
     lattice parameters of the synthetic input it came from? If the spacing
     drifts, the (image, params) pairs are mislabelled and the augmented data is
     worse than useless. Nothing else matters if this fails.

  2. REALISM. Does the output distribution look like real measurements? Reported
     as radial-spectrum distance to a held-out real set, plus peak prominence:
     synthetic lattices are sharp, real ones are broadened by defects and
     overlapping structure, so prominence landing near the real value means
     realistic imperfection was added rather than noise piled on.

  3. DIVERSITY. How different are repeated samples from one input? This is
     structurally zero for a deterministic generator and is DPS's main claim.

  4. MEMORISATION (--real_train_root). The prior is trained on scarce real data.
     A prior that regurgitates training images scores wonderfully on realism
     while making the augmentation worthless, so check it explicitly.

    python scripts/evaluate.py \
        --result_dir ./results/aug_dps --result_dir ./results/aug_uvcgan \
        --real_root /path/to/dataset/real/val \
        --real_train_root /path/to/dataset/real/train \
        --out_csv ./results/per_image.csv
'''
import argparse
import csv
import os
from collections import Counter, defaultdict
from glob import glob

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import numpy as np
from PIL import Image

from util.image_metrics import (
    InceptionFeatures,
    fid_is_reliable,
    frechet_distance,
    kernel_distance,
    psnr,
    ssim,
)
from util.lattice_metrics import (
    angle_difference,
    lattice_params,
    lattice_signature,
    power_spectrum,
    profile_distance,
    radial_profile,
    signature_distance,
    spacing_error,
)
from util.logger import get_logger

logger = get_logger()


# Generated images come out at the model resolution (160px) while the source and
# reference PNGs on disk are 150px. Lattice spacing is measured IN PIXELS, so
# comparing across those sizes injects a flat 160/150 - 1 = 6.7% spacing error
# into identical content -- larger than the ~5% threshold that is supposed to
# mean real drift. Everything is therefore resized to one size before measuring.
DEFAULT_IMAGE_SIZE = 160


def read_gray(path, image_size=DEFAULT_IMAGE_SIZE):
    img = Image.open(path).convert('L')
    if image_size and img.size != (image_size, image_size):
        # Bilinear, matching CrystalDataset so the pipeline and the metrics
        # resample identically.
        img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0


def load_dir(root, limit=None, exclude=None):
    """Sorted PNGs, subsampled by even stride rather than truncation.

    Crops are named "<photo>_sample_<k>", so paths[:limit] collapses onto the
    alphabetically-first photos: a 500-image reference drawn that way covers 100
    distinct scenes instead of 500.

    `exclude` drops paths before striding, so the returned set is still `limit`
    images. Used to keep the realism reference disjoint from the images being
    scored -- see collect_evaluated_sources.
    """
    paths = sorted(glob(os.path.join(root, '**', '*.png'), recursive=True))
    if exclude:
        excl = {os.path.abspath(p) for p in exclude}
        paths = [p for p in paths if os.path.abspath(p) not in excl]
    if not paths:
        raise ValueError(f'No PNGs under {root}')
    if limit and limit < len(paths):
        stride = len(paths) / limit
        return [paths[int(i * stride)] for i in range(limit)]
    return paths


def collect_evaluated_sources(result_dirs):
    """Every input and ground-truth image referenced by the runs being scored.

    These are excluded from the realism reference. In the round trip the sources
    are drawn from the same real/val split the reference comes from, so without
    this the generated images would be compared against a distribution
    containing the very images they were reconstructed from -- optimistic for
    FID/KID and for the spectral distance. In synth mode the sources are
    synthetic and never appear in a real reference, so nothing is excluded.
    """
    found = set()
    for rd in result_dirs:
        manifest = os.path.join(rd, 'manifest.csv')
        if not os.path.exists(manifest):
            continue
        with open(manifest) as fh:
            for row in csv.DictReader(fh):
                for key in ('source_image', 'ground_truth'):
                    p = (row.get(key) or '').strip()
                    if p:
                        found.add(os.path.abspath(p))
    return found


def summarise_reference(paths, image_size=DEFAULT_IMAGE_SIZE):
    '''Mean radial profile and prominence distribution of a reference set.'''
    profiles, prominences = [], []
    for p in paths:
        img = read_gray(p, image_size)
        profiles.append(radial_profile(power_spectrum(img)))
        prominences.append(lattice_params(img)['prominence'])
    n = min(len(x) for x in profiles)
    # Median, not mean: prominence is heavily right-tailed (on the real val
    # split, mean 3811 with std 11968), so a mean is set by a few outliers and
    # is not comparable across sets of different size.
    return {
        'profile': np.mean([x[:n] for x in profiles], axis=0),
        'prominence_median': float(np.median(prominences)),
        'prominence_iqr': float(np.subtract(*np.percentile(prominences, [75, 25]))),
        'n': len(paths),
    }


def feature_matrix(paths, size=32, image_size=DEFAULT_IMAGE_SIZE):
    '''Contrast-normalised low-res features for nearest-neighbour comparison.

    Downsampled and per-image standardised so the match is on structure rather
    than brightness, and so the all-pairs distance stays cheap.
    '''
    feats = []
    for p in paths:
        # Resize rather than stride: striding a 150px image by 4 covers only 83%
        # of it while striding 160px by 5 covers all of it, so strided features
        # are not comparable across the two sizes.
        img = np.asarray(
            Image.fromarray((read_gray(p, image_size) * 255).astype(np.uint8))
            .resize((size, size), Image.BILINEAR), dtype=np.float64)
        small = img - img.mean()
        norm = np.linalg.norm(small)
        feats.append((small / norm if norm > 0 else small).ravel())
    return np.stack(feats)


def nearest_neighbour_distances(query_feats, ref_feats):
    '''Min distance from each query to the reference set (unit-norm vectors).'''
    # ||a-b||^2 = 2 - 2<a,b> for unit vectors, so max similarity = min distance.
    sims = query_feats @ ref_feats.T
    return np.sqrt(np.maximum(2.0 - 2.0 * sims.max(axis=1), 0.0))


def evaluate_result_dir(result_dir, image_size=DEFAULT_IMAGE_SIZE, n_peaks=3,
                        lpips_model=None):
    '''Per-image records from one generate_augmented.py output directory.'''
    manifest = os.path.join(result_dir, 'manifest.csv')
    if not os.path.exists(manifest):
        raise FileNotFoundError(
            f'{manifest} not found. Result dirs must come from '
            'scripts/generate_augmented.py, which records the source mapping.')

    with open(manifest) as fh:
        rows = list(csv.DictReader(fh))

    source_cache = {}
    records = []
    for row in rows:
        out_path = os.path.join(result_dir, 'generated', row['output_image'])
        src_path = row['source_image']

        if src_path not in source_cache:
            if not os.path.exists(src_path):
                raise FileNotFoundError(
                    f'Source image {src_path} from the manifest is missing; '
                    'the label mapping cannot be verified.')
            ref = read_gray(src_path, image_size)
            source_cache[src_path] = (lattice_params(ref),
                                      lattice_signature(ref, k=n_peaks))
        src, src_sig = source_cache[src_path]

        img = read_gray(out_path, image_size)
        gen = lattice_params(img)
        gen_sig = lattice_signature(img, k=n_peaks)

        # Paired metrics need a per-output target, which exists only in the
        # round trip (real -> G_RS -> SR model), where the original real image
        # is what the output should have reconstructed. In the synth -> real
        # direction there is no single correct answer, so these stay absent
        # rather than being computed against something arbitrary.
        paired = {}
        gt_path = (row.get('ground_truth') or '').strip()
        if gt_path:
            if not os.path.exists(gt_path):
                raise FileNotFoundError(
                    f'ground_truth {gt_path} from the manifest is missing; '
                    'paired metrics cannot be computed.')
            gt = read_gray(gt_path, image_size)
            paired['psnr'] = psnr(img, gt)
            paired['ssim'] = ssim(img, gt)
            if lpips_model is not None:
                paired['lpips'] = lpips_model(img, gt)

        records.append({
            # `label` names the model in the comparison tables; older manifests
            # predate it, so fall back to the method.
            'label': row.get('label') or row['method'],
            'method': row['method'],
            # The swept hyperparameter, so a sweep table has it as a column
            # rather than encoded in directory names.
            'scale': row.get('scale', ''),
            'dps_framework': row.get('dps_framework', ''),
            'rs_framework': row.get('rs_framework', ''),
            'ground_truth': gt_path,
            **paired,
            'output_image': row['output_image'],
            'source_image': src_path,
            'sample_index': int(row['sample_index']),
            'src_spacing_px': src['spacing_px'],
            'gen_spacing_px': gen['spacing_px'],
            'spacing_rel_error': spacing_error(gen['spacing_px'], src['spacing_px']),
            # Whole reciprocal lattice, not just the strongest vector: these are
            # 2D lattices, and distorting a secondary vector while preserving the
            # primary one shows up as zero error in spacing_rel_error alone.
            'signature_error': signature_distance(src_sig, gen_sig),
            'src_angle_deg': src['angle_deg'],
            'gen_angle_deg': gen['angle_deg'],
            'angle_error_deg': angle_difference(gen['angle_deg'], src['angle_deg']),
            'src_prominence': src['prominence'],
            'gen_prominence': gen['prominence'],
            '_path': out_path,
            '_profile': radial_profile(power_spectrum(img)),
        })
    return records


def diversity(records, image_size=DEFAULT_IMAGE_SIZE):
    '''Mean pairwise RMSE between variants sharing a source image.'''
    by_source = defaultdict(list)
    for r in records:
        by_source[r['source_image']].append(r['_path'])

    scores = []
    for paths in by_source.values():
        if len(paths) < 2:
            continue
        imgs = [read_gray(p, image_size) for p in paths]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                scores.append(float(np.sqrt(np.mean((imgs[i] - imgs[j]) ** 2))))
    return float(np.mean(scores)) if scores else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--result_dir', action='append', required=True,
                   help='Output of generate_augmented.py. Repeat to compare.')
    p.add_argument('--real_root', type=str, required=True,
                   help='Held-out REAL images: the realism reference.')
    p.add_argument('--real_train_root', type=str, default=None,
                   help='Real TRAINING images, for the memorisation check.')
    p.add_argument('--out_csv', type=str, default=None,
                   help='Per-image metrics.')
    p.add_argument('--out_summary_csv', type=str, default=None,
                   help='The comparison table itself, one row per model. This '
                        'is what goes into the report.')
    p.add_argument('--lpips', action='store_true',
                   help='Perceptual paired metric (round trip only). Downloads '
                        'weights on first use.')
    p.add_argument('--fid_kid', action='store_true',
                   help='Distributional metrics against the real reference. '
                        'Downloads InceptionV3 weights on first use. KID is the '
                        'one to trust below a few thousand images; FID is '
                        'reported for familiarity and flagged when unreliable.')
    p.add_argument('--limit_reference', type=int, default=500,
                   help='Realism reference set size. Bounded because each image '
                        'costs an FFT.')
    p.add_argument('--limit_memorisation', type=int, default=None,
                   help='Training images for the NN check. Defaults to ALL of '
                        'them: a generated image can copy any training image, '
                        'so sampling the reference under-detects copying. The '
                        'features are 32x32, so the full set is cheap.')
    p.add_argument('--n_peaks', type=int, default=3,
                   help='Reciprocal-lattice peaks compared per image.')
    p.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE,
                   help='All images are resized to this before measuring. Must '
                        'match the model resolution, else pixel-space lattice '
                        'spacing is not comparable between source and output.')
    return p.parse_args()


def main():
    args = parse_args()

    # Build the reference AFTER reading the manifests, so the images being
    # scored can be held out of the distribution they are scored against.
    evaluated = collect_evaluated_sources(args.result_dir)

    logger.info(f'Reference (real) set: {args.real_root} '
                f'(all images measured at {args.image_size}px)')
    real_paths = load_dir(args.real_root, args.limit_reference,
                          exclude=evaluated)
    # Count what was actually removed rather than inferring it from directory
    # names, which breaks for a trailing slash or images in a subdirectory
    # (load_dir globs recursively).
    n_excluded = sum(1 for p in load_dir(args.real_root)
                     if os.path.abspath(p) in evaluated)
    if n_excluded:
        logger.info(f'  excluded {n_excluded} image(s) that these runs were '
                    'derived from, keeping the reference disjoint')
    real = summarise_reference(real_paths, args.image_size)
    logger.info(f'  {real["n"]} images, '
                f'prominence median {real["prominence_median"]:.1f} '
                f'(IQR {real["prominence_iqr"]:.1f})')

    ref_feats = None
    if args.real_train_root:
        train_paths = load_dir(args.real_train_root, args.limit_memorisation)
        logger.info(f'Memorisation reference: {len(train_paths)} training images')
        ref_feats = feature_matrix(train_paths, image_size=args.image_size)
        # Baseline: how close distinct real images get to each other. Generated
        # images landing much closer than this indicates copying.
        self_sims = ref_feats @ ref_feats.T
        np.fill_diagonal(self_sims, -np.inf)
        real_nn = np.sqrt(np.maximum(2.0 - 2.0 * self_sims.max(axis=1), 0.0))
        logger.info(f'Real-to-real NN distance: median {np.median(real_nn):.4f} '
                    f'(memorisation floor)')

    # Built once and shared: constructing these per image would dominate runtime.
    lpips_model = None
    if args.lpips:
        from util.image_metrics import LPIPS
        logger.info('Loading LPIPS (first use downloads weights)')
        lpips_model = LPIPS()

    inception, real_feats = None, None
    if args.fid_kid:
        logger.info('Loading InceptionV3 for FID/KID')
        inception = InceptionFeatures()
        real_feats = inception(read_gray(p, args.image_size) for p in real_paths)
        logger.info(f'  reference features: {real_feats.shape}')

    all_records, summaries = [], []
    for result_dir in args.result_dir:
        records = evaluate_result_dir(result_dir, args.image_size, args.n_peaks,
                                      lpips_model=lpips_model)
        all_records.extend(records)
        model = records[0]['label']

        n = min(len(r['_profile']) for r in records)
        mean_profile = np.mean([r['_profile'][:n] for r in records], axis=0)

        # Split by how strong the SOURCE lattice is. Where the reference is so
        # noisy that the lattice is barely detectable by eye, the "true" spacing
        # is itself uncertain, so a large error there measures the metric's
        # limits rather than the method's. Reporting both keeps that honest.
        src_prom = [r['src_prominence'] for r in records]
        thr = float(np.median(src_prom))
        strong = [r for r in records if r['src_prominence'] >= thr]
        weak = [r for r in records if r['src_prominence'] < thr]

        summary = {
            'result_dir': result_dir,
            'src_prom_threshold': thr,
            'spacing_err_strong': float(np.nanmedian(
                [r['spacing_rel_error'] for r in strong])) if strong else None,
            'spacing_err_strong_p90': float(np.nanpercentile(
                [r['spacing_rel_error'] for r in strong], 90)) if strong else None,
            'spacing_err_weak': float(np.nanmedian(
                [r['spacing_rel_error'] for r in weak])) if weak else None,
            'model': model,
            'scale': records[0].get('scale', ''),
            'dps_framework': records[0].get('dps_framework', ''),
            'rs_framework': records[0].get('rs_framework', ''),
            'n_images': len(records),
            'n_sources': len({r['source_image'] for r in records}),
            'spacing_err_median': float(np.nanmedian(
                [r['spacing_rel_error'] for r in records])),
            'spacing_err_p90': float(np.nanpercentile(
                [r['spacing_rel_error'] for r in records], 90)),
            'signature_err_median': float(np.nanmedian(
                [r['signature_error'] for r in records])),
            'angle_err_median': float(np.nanmedian(
                [r['angle_error_deg'] for r in records])),
            'prominence_median': float(np.median(
                [r['gen_prominence'] for r in records])),
            'src_prominence_median': float(np.median(
                [r['src_prominence'] for r in records])),
            'spectrum_dist_to_real': profile_distance(mean_profile, real['profile']),
            'diversity_rmse': diversity(records, args.image_size),
        }

        # Paired metrics exist only where the manifest carried a ground truth.
        for key in ('psnr', 'ssim', 'lpips'):
            vals = [r[key] for r in records if key in r and np.isfinite(r[key])]
            summary[f'{key}_median'] = float(np.median(vals)) if vals else None

        if inception is not None:
            gen_feats = inception(read_gray(r['_path'], args.image_size)
                                  for r in records)
            kid_mean, kid_std = kernel_distance(
                gen_feats, real_feats,
                subset_size=min(100, len(gen_feats), len(real_feats)))
            summary['kid'] = kid_mean
            summary['kid_std'] = kid_std
            summary['fid'] = frechet_distance(gen_feats, real_feats)
            summary['fid_reliable'] = fid_is_reliable(len(gen_feats),
                                                      len(real_feats))

        if ref_feats is not None:
            nn = nearest_neighbour_distances(
                feature_matrix([r['_path'] for r in records],
                               image_size=args.image_size), ref_feats)
            summary['nn_dist_median'] = float(np.median(nn))
            summary['nn_dist_min'] = float(np.min(nn))

        summaries.append(summary)

    # Prefer the label the generating run recorded. Fall back to the directory
    # name when several dirs share one -- a `scale` sweep is all one model.
    counts = Counter(s['model'] for s in summaries)
    for s in summaries:
        label = (s['model'] if counts[s['model']] == 1
                 else os.path.basename(os.path.normpath(s['result_dir'])))
        s['label'] = label[:14]

    print()
    width = 32 + 15 * len(summaries) + 20
    print('=' * width)
    print(f'{"metric":<32}' + ''.join(f'{s["label"]:>15}' for s in summaries))
    print('=' * width)

    def row(label, key, fmt='{:.4f}', note=''):
        vals = ''.join(
            f'{(fmt.format(s[key]) if s.get(key) is not None else "n/a"):>15}'
            for s in summaries)
        print(f'{label:<32}{vals}   {note}')

    print('-- label validity (decisive) '.ljust(width, '-'))
    row('spacing rel. error (median)', 'spacing_err_median', note='lower better')
    row('spacing rel. error (p90)', 'spacing_err_p90', note='lower better')
    row('lattice signature error', 'signature_err_median',
        note='all peaks, not just the top one')
    row('  strong-lattice refs (median)', 'spacing_err_strong',
        note='the trustworthy half')
    row('  strong-lattice refs (p90)', 'spacing_err_strong_p90')
    row('  weak-lattice refs (median)', 'spacing_err_weak',
        note='ground truth itself is uncertain here')
    row('angle error deg (median)', 'angle_err_median', '{:.2f}', 'lower better')
    print('-- realism '.ljust(width, '-'))
    row('spectrum dist to real', 'spectrum_dist_to_real', note='lower better')
    row('peak prominence (median)', 'prominence_median', '{:.1f}',
        f'real={real["prominence_median"]:.1f} <- target')
    row('  (source images)', 'src_prominence_median', '{:.1f}', 'for reference')
    if any(s.get('psnr_median') is not None for s in summaries):
        print('-- reconstruction (round trip only) '.ljust(width, '-'))
        row('PSNR dB (median)', 'psnr_median', '{:.2f}', 'higher better')
        row('SSIM (median)', 'ssim_median', '{:.4f}', 'higher better')
        row('LPIPS (median)', 'lpips_median', '{:.4f}', 'lower better')
    if any('kid' in s for s in summaries):
        print('-- distribution vs real '.ljust(width, '-'))
        row('KID', 'kid', '{:.5f}', 'lower better; trust this one')
        row('  subset spread', 'kid_std', '{:.5f}',
            'noise scale, NOT a standard error (subsets overlap)')
        reliable = all(s.get('fid_reliable') for s in summaries)
        row('FID', 'fid', '{:.2f}',
            'lower better' if reliable else 'UNRELIABLE at this n -- see KID')
    print('-- diversity '.ljust(width, '-'))
    row('pairwise RMSE across variants', 'diversity_rmse', note='higher = more varied')
    if ref_feats is not None:
        print('-- memorisation '.ljust(width, '-'))
        row('NN dist to real train (median)', 'nn_dist_median',
            note=f'floor={np.median(real_nn):.4f}')
        row('NN dist to real train (min)', 'nn_dist_min', note='inspect if tiny')
    print('=' * width)
    print('Prominence is not "lower better": it should land NEAR the real value.')
    print('Below it means over-degraded, above means too clean.')
    print()

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        # Union of keys: paired metrics are present only for round-trip runs, so
        # the first record's keys are not necessarily the full set.
        fields = list(dict.fromkeys(
            k for r in all_records for k in r if not k.startswith('_')))
        with open(args.out_csv, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_records)
        logger.info(f'Per-image metrics: {args.out_csv}')

    if args.out_summary_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_summary_csv)),
                    exist_ok=True)
        fields = list(dict.fromkeys(k for s in summaries for k in s))
        with open(args.out_summary_csv, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summaries)
        logger.info(f'Comparison table: {args.out_summary_csv}')


if __name__ == '__main__':
    main()
