"""Integration tests for the evaluation pipeline.

Builds fake result directories where the right answer is known by construction:
one 'method' preserves the lattice of its source, another shifts it. The
evaluation must separate them, since that judgement is what the thesis rests on.
"""
import csv
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
# evaluate.py imports its sibling _bootstrap, so scripts/ must be importable.
sys.path.insert(0, os.path.abspath(_SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "evaluate", os.path.join(_SCRIPTS, "evaluate.py"))
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)

N = 160


def grating(period, angle_deg=0.0, noise=0.0, seed=0, size=N):
    yy, xx = np.mgrid[0:size, 0:size]
    theta = np.radians(angle_deg)
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    img = np.sin(2 * np.pi * proj / period)
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip((img + 1) / 2, 0, 1)


def write_png(path, arr):
    plt.imsave(path, np.stack([arr] * 3, -1))


def build_result_dir(tmp_path, name, method, period_fn, n_sources=3, variants=1):
    """Fake generate_augmented.py output. period_fn maps source period -> output."""
    root = tmp_path / name
    (root / "generated").mkdir(parents=True)
    src_dir = tmp_path / f"{name}_sources"
    src_dir.mkdir(exist_ok=True)

    rows = []
    for i in range(n_sources):
        period = 8.0 + 2.0 * i
        src_path = src_dir / f"src_{i}.png"
        write_png(src_path, grating(period))
        for k in range(variants):
            out = f"{i:05d}_{k:02d}.png"
            write_png(root / "generated" / out,
                      grating(period_fn(period), noise=0.15, seed=i * 10 + k))
            rows.append([out, str(src_path), method, k, 0])

    with open(root / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["output_image", "source_image", "method", "sample_index", "seed"])
        w.writerows(rows)
    return root


def test_faithful_method_scores_near_zero_spacing_error(tmp_path):
    root = build_result_dir(tmp_path, "good", "faithful", lambda p: p)
    records = evaluate.evaluate_result_dir(root)
    assert len(records) == 3
    errs = [r["spacing_rel_error"] for r in records]
    assert max(errs) < 0.07, f"faithful method flagged as drifting: {errs}"


def test_drifting_method_is_caught(tmp_path):
    """The failure this whole metric exists to catch: plausible but mislabelled."""
    root = build_result_dir(tmp_path, "bad", "drifting", lambda p: p * 1.5)
    records = evaluate.evaluate_result_dir(root)
    errs = [r["spacing_rel_error"] for r in records]
    assert min(errs) > 0.2, f"lattice drift went undetected: {errs}"


def test_faithful_and_drifting_are_separable(tmp_path):
    good = evaluate.evaluate_result_dir(
        build_result_dir(tmp_path, "g", "faithful", lambda p: p))
    bad = evaluate.evaluate_result_dir(
        build_result_dir(tmp_path, "b", "drifting", lambda p: p * 1.5))
    assert (np.median([r["spacing_rel_error"] for r in good])
            < np.median([r["spacing_rel_error"] for r in bad]))


def test_diversity_zero_for_identical_variants(tmp_path):
    """A deterministic generator must score no diversity."""
    root = tmp_path / "det"
    (root / "generated").mkdir(parents=True)
    src = tmp_path / "s.png"
    write_png(src, grating(10.0))

    img = grating(10.0, noise=0.1)
    rows = []
    for k in range(3):
        out = f"00000_{k:02d}.png"
        write_png(root / "generated" / out, img)  # identical every time
        rows.append([out, str(src), "det", k, 0])
    with open(root / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["output_image", "source_image", "method", "sample_index", "seed"])
        w.writerows(rows)

    records = evaluate.evaluate_result_dir(root)
    assert evaluate.diversity(records) == pytest.approx(0.0, abs=1e-6)


def test_diversity_positive_for_varied_variants(tmp_path):
    root = build_result_dir(tmp_path, "var", "dps", lambda p: p, n_sources=2,
                            variants=3)
    records = evaluate.evaluate_result_dir(root)
    assert evaluate.diversity(records) > 0.01


def test_memorisation_check_flags_copies(tmp_path):
    """Copied images sit far closer to the training set than novel ones do."""
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    train_paths = []
    for i in range(6):
        p = train_dir / f"{i}.png"
        write_png(p, grating(8.0 + i, angle_deg=13 * i, noise=0.1, seed=i))
        train_paths.append(str(p))

    ref = evaluate.feature_matrix(train_paths)

    copies = evaluate.feature_matrix(train_paths[:3])          # exact duplicates

    novel_path = tmp_path / "n.png"
    write_png(novel_path, grating(30.0, angle_deg=77.0, noise=0.4, seed=99))
    novel = evaluate.feature_matrix([str(novel_path)])

    d_copy = evaluate.nearest_neighbour_distances(copies, ref)
    d_novel = evaluate.nearest_neighbour_distances(novel, ref)
    assert d_copy.max() < 1e-6, "exact copies should have ~zero NN distance"
    assert d_novel.min() > d_copy.max()


def test_missing_manifest_is_an_explicit_error(tmp_path):
    (tmp_path / "generated").mkdir()
    with pytest.raises(FileNotFoundError, match="manifest"):
        evaluate.evaluate_result_dir(tmp_path)


def test_source_and_output_size_mismatch_does_not_fake_drift(tmp_path):
    """Generated images are 160px; source PNGs on disk are 150px.

    Lattice spacing is measured in pixels, so without a common measurement size
    identical content scores 160/150 - 1 = 6.7% spacing error -- above the ~5%
    that is supposed to mean real drift. This is invisible whenever the method
    under test is bad enough to swamp it.
    """
    import PIL.Image

    root = tmp_path / "res"
    (root / "generated").mkdir(parents=True)
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    rows = []
    for i in range(4):
        arr = (grating(9.0 + i, angle_deg=15 * i, size=150) * 255).astype(np.uint8)
        src = src_dir / f"s{i}.png"
        PIL.Image.fromarray(arr).convert("RGB").save(src)          # 150px source
        PIL.Image.fromarray(arr).convert("RGB").resize(
            (160, 160), PIL.Image.BILINEAR
        ).save(root / "generated" / f"{i:05d}_00.png")             # 160px output
        rows.append([f"{i:05d}_00.png", str(src), "identity_resized", 0, 0])

    with open(root / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["output_image", "source_image", "method", "sample_index", "seed"])
        w.writerows(rows)

    records = evaluate.evaluate_result_dir(root, image_size=160)
    errs = [r["spacing_rel_error"] for r in records]
    assert max(errs) < 0.02, f"size mismatch leaked into spacing error: {errs}"


def test_read_gray_normalises_size_and_range():
    import tempfile

    import PIL.Image
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.png")
        PIL.Image.fromarray(
            (grating(10.0, size=150) * 255).astype(np.uint8)
        ).convert("RGB").save(p)
        img = evaluate.read_gray(p, image_size=160)
        assert img.shape == (160, 160)
        assert 0.0 <= img.min() and img.max() <= 1.0
