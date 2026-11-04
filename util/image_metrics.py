'''Standard image metrics: paired (PSNR/SSIM/LPIPS) and distributional (FID/KID).

These complement the domain-specific lattice and spectral metrics in
`lattice_metrics.py`. They are the numbers a reader of an image-translation
paper expects to see; the lattice metrics are the ones that actually decide
whether the generated data is usable. Both belong in the report.

WHICH APPLIES WHERE
  paired (PSNR, SSIM, LPIPS)
      Need a ground truth for each output, so they apply ONLY to the round-trip
      experiment (real -> G_RS -> SR model), where the original real image is
      the target. In the synth -> real direction there is no ground truth --
      given a synthetic input there is no single correct realistic image -- so
      applying them there would be meaningless.

  distributional (FID, KID)
      Compare two SETS of images without pairing, so they apply in both
      directions. This is the right frame for "does the output look like real
      data" when no per-image target exists.

A NOTE ON FID AT THIS SAMPLE SIZE
  FID estimates a 2048-dimensional covariance. With n < 2048 that covariance is
  rank-deficient and FID is badly biased upward -- at n = 200 the bias dwarfs
  any difference between decent models. KID's unbiased MMD estimator has no such
  problem and is the metric to trust here; FID is computed and reported because
  readers look for it, with `fid_reliable` flagging when n makes it meaningless.

  Absolute values are not comparable to published figures either way: those use
  the TF-ported InceptionV3, this uses torchvision's. Comparisons BETWEEN the
  models here are valid, comparisons to numbers in other papers are not.
'''
import warnings

import numpy as np

# Torch-dependent pieces are imported lazily so the lattice metrics, which need
# only numpy, keep working in environments without torch.


def _to_hwc(img):
    '''Accept HxW or HxWxC in [0, 1]; return HxWx3.'''
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f'Expected HxW or HxWx3, got {img.shape}')
    return img


def psnr(generated, target, data_range: float = 1.0) -> float:
    '''Peak signal-to-noise ratio, dB.

        PSNR = 10 log10(MAX^2 / MSE)

    Pixel-wise, so it rewards matching the target exactly. For a stochastic
    method that is partly the wrong thing to ask -- DPS is meant to produce a
    plausible sample, not the one true answer -- which is why it is reported
    alongside SSIM and LPIPS rather than alone.
    '''
    a, b = _to_hwc(generated), _to_hwc(target)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(generated, target, data_range: float = 1.0) -> float:
    '''Structural similarity, in [-1, 1], higher better.

    Compares local luminance, contrast and structure rather than raw pixel
    differences, so it tolerates the small intensity shifts that pixel metrics
    punish heavily. Computed on the grayscale image: these micrographs carry no
    real colour, so per-channel SSIM would just average three copies.

    Uses the Wang et al. (2004) parameterisation -- an 11x11 Gaussian window
    with sigma 1.5 and the population covariance -- rather than scikit-image's
    defaults (a 7x7 uniform window with the sample covariance). Both are
    self-consistent for comparing models here, but only the former produces
    numbers comparable to SSIM values reported elsewhere.
    '''
    from skimage.metrics import structural_similarity

    a, b = _to_hwc(generated).mean(-1), _to_hwc(target).mean(-1)
    return float(structural_similarity(
        a, b, data_range=data_range,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False))


class LPIPS:
    '''Learned Perceptual Image Patch Similarity, lower better.

    Distance in the feature space of a pretrained CNN, calibrated against human
    similarity judgements. It is the paired metric that best tracks "looks like
    the same thing" for textured images, where PSNR is dominated by
    high-frequency detail no method can reproduce exactly.

    Weights download on first use. Constructed lazily and reused, because
    building it per image would dominate the runtime.
    '''

    def __init__(self, net: str = 'alex', device=None):
        import lpips as lpips_pkg
        import torch

        self.torch = torch
        self.device = device or torch.device('cpu')
        self.model = lpips_pkg.LPIPS(net=net).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def __call__(self, generated, target) -> float:
        torch = self.torch
        # lpips expects NCHW in [-1, 1].
        a = torch.from_numpy(_to_hwc(generated)).permute(2, 0, 1)[None].float()
        b = torch.from_numpy(_to_hwc(target)).permute(2, 0, 1)[None].float()
        a, b = a * 2 - 1, b * 2 - 1
        with torch.no_grad():
            d = self.model(a.to(self.device), b.to(self.device))
        return float(d.item())


class InceptionFeatures:
    '''2048-d pool3 features from torchvision InceptionV3, for FID and KID.'''

    def __init__(self, device=None, batch_size: int = 32):
        import torch
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.torch = torch
        self.device = device or torch.device('cpu')
        self.batch_size = batch_size
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                             aux_logits=True)
        model.fc = torch.nn.Identity()   # expose the 2048-d pooled features
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # ImageNet normalisation, as the weights expect.
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, images) -> np.ndarray:
        '''images: iterable of HxW or HxWx3 arrays in [0, 1]. -> (N, 2048).'''
        torch = self.torch
        feats = []
        batch = []

        def flush():
            if not batch:
                return
            x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float()
            x = torch.nn.functional.interpolate(
                x, size=(299, 299), mode='bilinear', align_corners=False)
            x = (x - self.mean) / self.std
            with torch.no_grad():
                feats.append(self.model(x.to(self.device)).cpu().numpy())
            batch.clear()

        for img in images:
            batch.append(_to_hwc(img))
            if len(batch) == self.batch_size:
                flush()
        flush()
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 2048))


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    '''FID between two feature sets.

        FID = ||mu_a - mu_b||^2 + Tr(S_a + S_b - 2 (S_a S_b)^{1/2})

    i.e. the squared Wasserstein-2 distance between Gaussians fitted to the two
    feature sets. See the module docstring on why this is unreliable below a few
    thousand samples.
    '''
    from scipy import linalg

    mu_a, mu_b = feats_a.mean(0), feats_b.mean(0)
    sigma_a = np.cov(feats_a, rowvar=False)
    sigma_b = np.cov(feats_b, rowvar=False)

    diff = mu_a - mu_b
    # sqrtm of a product of PSD matrices can come back with tiny imaginary parts
    # from numerical error; discard them once confirmed negligible.
    # scipy deprecated `disp`, and without it sqrtm returns the bare array, so
    # accept either shape rather than pinning a scipy version.
    covmean = linalg.sqrtm(sigma_a.dot(sigma_b))
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            # Below feature_dim samples the covariances are singular and sqrtm
            # can return a substantially complex result. Return NaN rather than
            # raising: FID is already flagged unreliable at that sample size
            # (see fid_is_reliable), so killing the whole evaluation over it
            # would lose every other metric for no benefit.
            warnings.warn(
                'FID: sqrtm returned a substantially complex matrix, which '
                'happens when the feature covariance is singular. Reporting '
                'NaN; use KID at this sample size.', RuntimeWarning,
                stacklevel=2)
            return float('nan')
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b)
                 - 2 * np.trace(covmean))


def _poly_kernel(x, y, d):
    return (x @ y.T / d + 1) ** 3


def _mmd2_unbiased(x, y, groups_x=None):
    '''Unbiased MMD^2 for sets of DIFFERENT sizes m and n.

        E k(x,x') + E k(y,y') - 2 E k(x,y),

    each self-term averaged over pairs of DISTINCT samples. With `groups_x`,
    the x-x term only uses pairs from different groups: several samples drawn
    from one source (the 4 DPS variants of one input) are correlated, so their
    mutual kernel values would inflate E k(x,x') -- and only for stochastic
    models, since a deterministic one has one sample per source.
    '''
    d = x.shape[1]
    m, n = len(x), len(y)
    kxx = _poly_kernel(x, x, d)
    if groups_x is None:
        mask = ~np.eye(m, dtype=bool)
    else:
        g = np.asarray(groups_x)
        mask = g[:, None] != g[None, :]
    if not mask.any():
        raise ValueError('KID needs samples from at least 2 distinct sources')
    kyy = _poly_kernel(y, y, d)
    np.fill_diagonal(kyy, 0)
    return (kxx[mask].mean() + kyy.sum() / (n * (n - 1))
            - 2 * _poly_kernel(x, y, d).mean())


def kernel_distance(feats_a: np.ndarray, feats_b: np.ndarray,
                    n_subsets: int = 100, subset_size: int | None = None,
                    seed: int = 0, groups_a=None) -> tuple:
    '''KID: unbiased MMD^2 with the polynomial kernel, plus its spread.

        k(x, y) = (x.y / d + 1)^3

    feats_a are the generated images, feats_b the reference. groups_a, if
    given, names each generated image's source (e.g. its source image path),
    so that variants of one source are not treated as independent samples.

    Returns (estimate, spread).

    The ESTIMATE is computed once on the FULL sets, with the unbiased
    estimator for unequal sizes. (An earlier version trimmed both sets to the
    shorter length, so a 50-image model was scored against the first 50
    reference images and a 200-image one against the first 200 -- different,
    alphabetically-first slices of the reference, which made KID incomparable
    across models.)

    The SPREAD is the standard deviation across n_subsets resamples: half of
    the generated SOURCES (all their variants) against `subset_size`
    reference images, the same subset_size for every model, so the spread
    means the same thing for every model. Treat it as an order-of-magnitude
    "differences smaller than this are noise", NOT a standard error: the
    resamples overlap heavily, so it understates the true uncertainty.

    Lower is better; 0 means identical distributions.
    '''
    rng = np.random.default_rng(seed)
    if groups_a is None:
        groups_a = np.arange(len(feats_a))
    groups_a = np.asarray(groups_a)
    if len(groups_a) != len(feats_a):
        raise ValueError('groups_a must give one source per generated image')
    if len(feats_b) < 2:
        raise ValueError('KID needs at least 2 reference samples')

    estimate = _mmd2_unbiased(feats_a, feats_b, groups_a)

    sources = np.unique(groups_a)
    k_src = max(2, len(sources) // 2)
    k_ref = min(subset_size or 100, len(feats_b) // 2)
    vals = []
    if len(sources) >= 3 and k_ref >= 2:
        for _ in range(n_subsets):
            pick = np.isin(groups_a, rng.choice(sources, k_src, replace=False))
            ref = rng.choice(len(feats_b), k_ref, replace=False)
            vals.append(_mmd2_unbiased(feats_a[pick], feats_b[ref],
                                       groups_a[pick]))
    return float(estimate), (float(np.std(vals)) if vals else 0.0)


def fid_is_reliable(n_generated: int, n_reference: int,
                    feature_dim: int = 2048) -> bool:
    '''Whether FID has enough samples to estimate its covariance at all.

    Below feature_dim samples the covariance is rank-deficient and FID is
    dominated by that bias rather than by any real difference between the sets.
    '''
    return min(n_generated, n_reference) >= feature_dim
