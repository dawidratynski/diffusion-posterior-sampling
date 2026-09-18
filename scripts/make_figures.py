'''Build the report figures from one or more generate_augmented.py result dirs.

Produces three things, because the right layout for a report is rarely known in
advance and re-running generation to change a figure is expensive:

  grids/comparison_NN.png     rows = source images, columns = [reference,
                              measurement, model 1, model 2, ...]. The
                              side-by-side that goes in the report.

  grids/variants_<model>_NN.png   rows = source images, columns = repeated
                              samples from ONE model. Only meaningful for a
                              stochastic model; this is what shows that DPS
                              produces genuinely different imperfections rather
                              than one answer with noise on top.

  panels/<source_id>/NN_<name>.png    every cell of every grid as a separate
                              file, numbered so lexical order is the intended
                              column order. Use these to lay the figure out by
                              hand, or to pull a single image into the text.

Result dirs are given in the order they should appear as columns, and each
model's name comes from its own manifest, so the figure cannot silently mislabel
a column. All dirs must come from runs sharing --limit and the same input root:
the strided subset is then identical across them, which is what makes a row a
like-for-like comparison.

    python scripts/make_figures.py \
        --result_dir results/rt_uvcgan \
        --result_dir results/rt_dps_uvcgan \
        --result_dir results/rt_dps_analytic \
        --out_dir /content/drive/MyDrive/.../figures/roundtrip \
        --rows_per_grid 6 --limit 24
'''
import argparse
import csv
import os

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from util.logger import get_logger

logger = get_logger()
PANEL_PX = 160


def read_manifest(result_dir):
    path = os.path.join(result_dir, 'manifest.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found; result dirs must come from '
            'scripts/generate_augmented.py.')
    with open(path) as fh:
        return list(csv.DictReader(fh))


def load_panel(path, size=PANEL_PX):
    img = Image.open(path).convert('RGB')
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0


def collect(result_dirs):
    '''Group every model's outputs by the source image they came from.

    Returns (labels, sources, table) where table[source][label] is the ordered
    list of that model's variant filepaths.
    '''
    labels, table, order = [], {}, []
    measurement_of = {}

    for rd in result_dirs:
        rows = read_manifest(rd)
        label = rows[0].get('label') or rows[0]['method']
        if label in labels:
            raise ValueError(
                f'Two result dirs both labelled {label!r}. Pass --label to '
                'generate_augmented.py so each column is named distinctly.')
        labels.append(label)

        for row in rows:
            src = row['source_image']
            if src not in table:
                table[src] = {}
                order.append(src)
            table[src].setdefault(label, []).append(
                os.path.join(rd, 'generated', row['output_image']))
            meas = (row.get('measurement_image') or '').strip()
            if meas and src not in measurement_of:
                measurement_of[src] = os.path.join(rd, 'measurement', meas)

    # A row is only a fair comparison if every model saw this source.
    complete = [s for s in order if len(table[s]) == len(labels)]
    dropped = len(order) - len(complete)
    if dropped:
        logger.warning(
            f'{dropped} source(s) missing from some result dir and skipped. '
            'That means the runs did not use the same --limit and input root, '
            'so their rows would not be comparable.')
    return labels, complete, table, measurement_of


def save_panels(out_dir, sources, labels, table, measurement_of):
    '''Every cell as its own file, numbered to preserve column order.'''
    root = os.path.join(out_dir, 'panels')
    index = []
    for i, src in enumerate(sources):
        sid = f'{i:05d}'
        d = os.path.join(root, sid)
        os.makedirs(d, exist_ok=True)

        Image.open(src).convert('RGB').resize(
            (PANEL_PX, PANEL_PX), Image.BILINEAR).save(
            os.path.join(d, '00_reference.png'))
        col = 1
        if src in measurement_of:
            Image.open(measurement_of[src]).convert('RGB').resize(
                (PANEL_PX, PANEL_PX), Image.BILINEAR).save(
                os.path.join(d, '01_measurement.png'))
            col = 2
        for label in labels:
            for k, p in enumerate(table[src][label]):
                suffix = '' if len(table[src][label]) == 1 else f'_var{k}'
                Image.open(p).convert('RGB').resize(
                    (PANEL_PX, PANEL_PX), Image.BILINEAR).save(
                    os.path.join(d, f'{col:02d}_{label}{suffix}.png'))
            col += 1
        index.append({'panel_dir': os.path.relpath(d, out_dir),
                      'source_image': src})

    with open(os.path.join(out_dir, 'index.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['panel_dir', 'source_image'])
        w.writeheader()
        w.writerows(index)
    logger.info(f'  panels: {len(sources)} source dirs under {root}')


def grid(path, rows, col_titles, row_images, dpi=160):
    '''One figure: row_images[r][c] is an HxWx3 array in [0, 1].'''
    ncol = len(col_titles)
    fig, ax = plt.subplots(rows, ncol,
                           figsize=(1.7 * ncol, 1.78 * rows), dpi=dpi,
                           squeeze=False)
    for r in range(rows):
        for c in range(ncol):
            ax[r][c].imshow(row_images[r][c])
            ax[r][c].axis('off')
            if r == 0:
                ax[r][c].set_title(col_titles[c], fontsize=8)
    plt.tight_layout(pad=0.3)
    # A visible seam between cells; without it adjacent rows of a textured
    # micrograph read as one continuous image in print.
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_comparison_grids(out_dir, sources, labels, table, measurement_of,
                          rows_per_grid):
    d = os.path.join(out_dir, 'grids')
    os.makedirs(d, exist_ok=True)
    has_meas = all(s in measurement_of for s in sources)
    titles = (['reference'] + (['measurement'] if has_meas else []) + labels)

    n = 0
    for start in range(0, len(sources), rows_per_grid):
        chunk = sources[start:start + rows_per_grid]
        images = []
        for src in chunk:
            row = [load_panel(src)]
            if has_meas:
                row.append(load_panel(measurement_of[src]))
            # Variant 0 only: this grid compares models, not samples.
            row += [load_panel(table[src][label][0]) for label in labels]
            images.append(row)
        n += 1
        grid(os.path.join(d, f'comparison_{n:02d}.png'), len(chunk), titles, images)
    logger.info(f'  grids: {n} comparison figure(s) under {d}')


def save_variant_grids(out_dir, sources, labels, table, rows_per_grid):
    '''Repeated samples from one model -- only informative if it is stochastic.'''
    d = os.path.join(out_dir, 'grids')
    os.makedirs(d, exist_ok=True)
    for label in labels:
        n_var = min(len(table[s][label]) for s in sources)
        if n_var < 2:
            continue
        titles = ['reference'] + [f'{label} #{k}' for k in range(n_var)]
        n = 0
        for start in range(0, len(sources), rows_per_grid):
            chunk = sources[start:start + rows_per_grid]
            images = [[load_panel(src)]
                      + [load_panel(table[src][label][k]) for k in range(n_var)]
                      for src in chunk]
            n += 1
            grid(os.path.join(d, f'variants_{label}_{n:02d}.png'),
                 len(chunk), titles, images)
        logger.info(f'  grids: {n} variant figure(s) for {label}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result_dir', action='append', required=True,
                   help='Repeat, in the order the models should appear as '
                        'columns. Each is labelled from its own manifest.')
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--rows_per_grid', type=int, default=6,
                   help='Sources per grid figure. Keeps each figure readable at '
                        'page width instead of one unusable tall image.')
    p.add_argument('--limit', type=int, default=None,
                   help='Use only the first N sources (after the shared subset).')
    p.add_argument('--no_panels', action='store_true',
                   help='Skip the per-cell files; grids only.')
    args = p.parse_args()

    labels, sources, table, measurement_of = collect(args.result_dir)
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        raise ValueError('No sources common to all result dirs.')

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info(f'{len(sources)} sources x {len(labels)} models -> {args.out_dir}')
    logger.info(f'  columns: {" | ".join(labels)}')

    save_comparison_grids(args.out_dir, sources, labels, table, measurement_of,
                          args.rows_per_grid)
    save_variant_grids(args.out_dir, sources, labels, table, args.rows_per_grid)
    if not args.no_panels:
        save_panels(args.out_dir, sources, labels, table, measurement_of)
    logger.info('Done.')


if __name__ == '__main__':
    main()
