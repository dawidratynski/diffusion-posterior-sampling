'''Re-split an already-generated dataset so crops never straddle train/val.

The upstream splitter shuffles individual crop files, but the pipeline emits
several crops per source photo (`REAL_SAMPLES_PER_IMAGE`), so siblings of a
"held-out" image end up in training. Measured on the sample dataset: 1969 of
1970 real val sources also appeared in train.

That inflates every val metric, and it specifically breaks the memorisation
check -- a val image whose siblings were trained on looks memorised whether or
not the model memorised anything, so the number stops meaning what it should.

This regroups the existing files by source photo and re-splits at that level.
Lossless: files are moved between train/ and val/, never regenerated.

    python scripts/fix_split.py ../dataset/real --dry_run
    python scripts/fix_split.py ../dataset/real ../dataset/synth

Filenames are `<source>_sample_<k>.png`, so the source key is the name with the
`_sample_<k>` suffix stripped -- which covers both the real (`12_sample_3.png`)
and synth (`images_0-010.npy_@_7_sample_0.png`) conventions.
'''
import argparse
import os
import random
import re
import shutil
from collections import defaultdict

SAMPLE_SUFFIX = re.compile(r'_sample_\d+\.png$')


def source_key(filename: str) -> str:
    stripped = SAMPLE_SUFFIX.sub('', filename)
    if stripped == filename:
        # No _sample_N suffix: treat the file as its own source rather than
        # silently lumping every such file into one giant group.
        return filename
    return stripped


def collect(root: str) -> dict:
    groups = defaultdict(list)
    for split in ('train', 'val'):
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            raise FileNotFoundError(f'{d} does not exist')
        for fn in os.listdir(d):
            if fn.endswith('.png'):
                groups[source_key(fn)].append((split, fn))
    return groups


def leakage(groups: dict) -> int:
    return sum(1 for files in groups.values()
               if len({s for s, _ in files}) > 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('roots', nargs='+',
                   help='Domain dirs, each containing train/ and val/.')
    p.add_argument('--ratio', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--force', action='store_true',
                   help='Re-split even a domain that already has no leakage.')
    args = p.parse_args()

    for root in args.roots:
        groups = collect(root)
        n_files = sum(len(v) for v in groups.values())
        before = leakage(groups)
        print(f'\n{root}')
        print(f'  {n_files} files, {len(groups)} source photos')
        print(f'  sources split across train and val: {before}'
              f'{"  <-- leakage" if before else ""}')

        if before == 0 and not args.force:
            # One crop per source (as for synth): the split is already sound, so
            # reshuffling would churn thousands of files and discard a valid
            # split for nothing.
            print('  already clean, skipping (use --force to re-split anyway)')
            continue

        keys = sorted(groups)
        random.Random(args.seed).shuffle(keys)
        n_train = int(len(keys) * args.ratio)
        target = {k: ('train' if i < n_train else 'val')
                  for i, k in enumerate(keys)}

        moves = [(cur, want, fn)
                 for k, files in groups.items()
                 for cur, fn in files
                 if (want := target[k]) != cur]

        train_files = sum(len(groups[k]) for k in keys[:n_train])
        print(f'  -> {n_train} sources / {train_files} files train, '
              f'{len(keys) - n_train} sources / {n_files - train_files} files val')
        print(f'  {len(moves)} files to move')

        if args.dry_run:
            continue

        for cur, want, fn in moves:
            shutil.move(os.path.join(root, cur, fn),
                        os.path.join(root, want, fn))

        after = leakage(collect(root))
        print(f'  sources split across train and val after: {after}')
        if after:
            raise RuntimeError(f'Re-split failed: {after} sources still leak.')


if __name__ == '__main__':
    main()
