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


def load_dir(root, limit=None):
    paths = sorted(glob(os.path.join(root, '**', '*.png'), recursive=True))
    if not paths:
        raise ValueError(f'No PNGs under {root}')
    return paths[:limit] if limit else paths


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


def evaluate_result_dir(result_dir, image_size=DEFAULT_IMAGE_SIZE, n_peaks=3):
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

        records.append({
            'method': row['method'],
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
    p.add_argument('--out_csv', type=str, default=None)
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

    logger.info(f'Reference (real) set: {args.real_root} '
                f'(all images measured at {args.image_size}px)')
    real_paths = load_dir(args.real_root, args.limit_reference)
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

    all_records, summaries = [], []
    for result_dir in args.result_dir:
        records = evaluate_result_dir(result_dir, args.image_size, args.n_peaks)
        all_records.extend(records)
        method = records[0]['method']

        n = min(len(r['_profile']) for r in records)
        mean_profile = np.mean([r['_profile'][:n] for r in records], axis=0)

        summary = {
            'result_dir': result_dir,
            'method': method,
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

        if ref_feats is not None:
            nn = nearest_neighbour_distances(
                feature_matrix([r['_path'] for r in records],
                               image_size=args.image_size), ref_feats)
            summary['nn_dist_median'] = float(np.median(nn))
            summary['nn_dist_min'] = float(np.min(nn))

        summaries.append(summary)

    # Several result dirs can share a method -- a `scale` sweep is all 'dps' --
    # so fall back to the directory name to keep the columns distinguishable.
    counts = Counter(s['method'] for s in summaries)
    for s in summaries:
        label = (s['method'] if counts[s['method']] == 1
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
    row('angle error deg (median)', 'angle_err_median', '{:.2f}', 'lower better')
    print('-- realism '.ljust(width, '-'))
    row('spectrum dist to real', 'spectrum_dist_to_real', note='lower better')
    row('peak prominence (median)', 'prominence_median', '{:.1f}',
        f'real={real["prominence_median"]:.1f} <- target')
    row('  (source images)', 'src_prominence_median', '{:.1f}', 'for reference')
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
        fields = [k for k in all_records[0] if not k.startswith('_')]
        with open(args.out_csv, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_records)
        logger.info(f'Per-image metrics: {args.out_csv}')


if __name__ == '__main__':
    main()
