"""Tests for the standard image metrics.

Each metric is checked against a case where the right answer is known by
construction: identical inputs, a known-worse variant, or two sets drawn from
the same distribution.
"""
import numpy as np
import pytest

from util.image_metrics import (
    fid_is_reliable,
    frechet_distance,
    kernel_distance,
    psnr,
    ssim,
)

N = 64


def img(seed=0, noise=0.0, shift=0.0):
    rng = np.random.default_rng(seed)
    _, xx = np.mgrid[0:N, 0:N]
    base = 0.5 + 0.3 * np.sin(2 * np.pi * xx / 12.0)
    if noise:
        base = base + rng.normal(0, noise, base.shape)
    return np.clip(base + shift, 0, 1)


# ---------------------------------------------------------------- paired ----

def test_psnr_infinite_for_identical():
    a = img()
    assert psnr(a, a) == float("inf")


def test_psnr_decreases_with_corruption():
    a = img()
    scores = [psnr(img(noise=n, seed=1), a) for n in (0.01, 0.05, 0.2)]
    assert scores[0] > scores[1] > scores[2], scores


def test_psnr_matches_closed_form():
    a = img()
    b = np.clip(a + 0.1, 0, 1)
    mse = np.mean((a - b) ** 2)
    assert psnr(b, a) == pytest.approx(10 * np.log10(1.0 / mse), rel=1e-9)


def test_ssim_one_for_identical():
    a = img()
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-6)


def test_ssim_decreases_with_corruption():
    a = img()
    scores = [ssim(img(noise=n, seed=1), a) for n in (0.01, 0.05, 0.2)]
    assert scores[0] > scores[1] > scores[2], scores


def test_paired_metrics_accept_grayscale_and_rgb():
    """Generated PNGs read back as RGB(A); references may be grayscale."""
    gray = img()
    rgb = np.stack([gray] * 3, -1)
    rgba = np.concatenate([rgb, np.ones((N, N, 1))], -1)
    assert psnr(rgb, gray) == float("inf")
    assert psnr(rgba, gray) == float("inf")
    assert ssim(rgb, gray) == pytest.approx(1.0, abs=1e-6)


# -------------------------------------------------------- distributional ----

def _feats(n, dim=64, loc=0.0, seed=0):
    return np.random.default_rng(seed).normal(loc, 1.0, (n, dim))


def test_frechet_zero_for_same_features():
    f = _feats(200)
    assert frechet_distance(f, f) == pytest.approx(0.0, abs=1e-6)


def test_frechet_grows_with_separation():
    a = _feats(300, seed=0)
    near = _feats(300, loc=0.2, seed=1)
    far = _feats(300, loc=1.0, seed=2)
    assert frechet_distance(a, near) < frechet_distance(a, far)


def test_kid_near_zero_for_same_distribution():
    """The property FID lacks at small n: unbiased, so ~0 for like vs like."""
    a = _feats(200, seed=0)
    b = _feats(200, seed=1)
    mean, std = kernel_distance(a, b, n_subsets=20, subset_size=100)
    assert abs(mean) < 5 * std + 1e-3, f"mean {mean}, std {std}"


def test_kid_grows_with_separation():
    a = _feats(200, seed=0)
    near, _ = kernel_distance(a, _feats(200, loc=0.2, seed=1),
                              n_subsets=20, subset_size=100)
    far, _ = kernel_distance(a, _feats(200, loc=1.0, seed=2),
                             n_subsets=20, subset_size=100)
    assert near < far


def test_kid_is_deterministic_for_a_fixed_seed():
    a, b = _feats(200, seed=0), _feats(200, loc=0.3, seed=1)
    m1, _ = kernel_distance(a, b, n_subsets=10, subset_size=50, seed=7)
    m2, _ = kernel_distance(a, b, n_subsets=10, subset_size=50, seed=7)
    assert m1 == m2


def test_kid_unequal_sizes_matches_brute_force():
    """The estimator must handle m != n exactly (models differ in n)."""
    from util.image_metrics import _mmd2_unbiased
    rng = np.random.default_rng(0)
    d = 8
    x, y = rng.normal(size=(7, d)), rng.normal(0.3, 1, size=(11, d))

    def k(a, b):
        return (a @ b / d + 1) ** 3

    m, n = len(x), len(y)
    brute = (sum(k(x[i], x[j]) for i in range(m) for j in range(m) if i != j)
             / (m * (m - 1))
             + sum(k(y[i], y[j]) for i in range(n) for j in range(n) if i != j)
             / (n * (n - 1))
             - 2 * sum(k(x[i], y[j]) for i in range(m) for j in range(n))
             / (m * n))
    assert np.isclose(_mmd2_unbiased(x, y), brute)


def test_kid_uses_the_whole_reference_not_a_prefix():
    """A 50-image model used to be scored against the first 50 references only."""
    gen, ref = _feats(50, seed=0), _feats(500, seed=1)
    shifted = ref.copy()
    shifted[:100] += 1.0          # unrepresentative prefix (sorted file order)
    clean, _ = kernel_distance(gen, ref, n_subsets=5)
    skewed, _ = kernel_distance(gen, shifted, n_subsets=5)
    prefix_only, _ = kernel_distance(gen, shifted[:50], n_subsets=5)
    assert skewed < prefix_only / 5   # whole reference dilutes the prefix
    assert abs(skewed - clean) < abs(prefix_only - clean)


def test_kid_does_not_reward_or_penalise_repeated_variants():
    """4 variants per source must not count as 4 independent samples."""
    gen, ref = _feats(50, seed=0), _feats(300, seed=1)
    once, _ = kernel_distance(gen, ref, n_subsets=5)
    groups = np.repeat(np.arange(50), 4)
    four, _ = kernel_distance(np.repeat(gen, 4, axis=0), ref, n_subsets=5,
                              groups_a=groups)
    assert np.isclose(once, four)


def test_fid_reliability_flag_tracks_feature_dimension():
    """At our sample sizes FID cannot estimate a 2048-d covariance at all."""
    assert not fid_is_reliable(200, 500)
    assert not fid_is_reliable(2047, 10000)
    assert fid_is_reliable(2048, 2048)
