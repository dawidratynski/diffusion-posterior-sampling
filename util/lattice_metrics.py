'''Fourier-domain lattice metrics for the crystal data.

The generated images are only useful as training data if they keep the lattice
parameters of the synthetic input they came from -- a method that adds
convincing imperfections but shifts the spacing produces mislabelled data, which
is worse than no data. Generic image metrics (PSNR, FID) cannot see that, so
these measure it directly.

Everything works on the 2D power spectrum: a repeating lattice of period p
pixels puts a peak at radius N/p in an N-pixel FFT, and the peak's angle gives
the lattice orientation.
'''
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    '''HxW, HxWx3 or HxWx4 in any range -> float HxW.'''
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    if img.ndim != 2:
        raise ValueError(f'Expected a 2D or 3D image, got shape {img.shape}')
    return img


def power_spectrum(img: np.ndarray, window: bool = True,
                   pad_factor: int = 1) -> np.ndarray:
    '''Centred 2D power spectrum.

    A Hann window is applied by default: these images are crops, so the implicit
    periodic tiling in the FFT creates hard edge discontinuities whose spectral
    leakage (a cross artefact through the origin) is easily mistaken for lattice
    structure.

    `pad_factor` zero-pads before the transform, interpolating the spectrum onto
    a finer grid. This matters for peak location: the lattice here sits at radius
    ~5 in a 150px crop, where integer bins quantise the spacing into ~17% steps
    (150/5 = 30 vs 150/6 = 25). Measured on the real data, padding by 4 cuts the
    crop-to-crop spacing spread from 8.4% to 7.6% and then saturates. Use 1 for
    radial profiles, where padding only adds interpolation ripple.
    '''
    img = to_gray(img)
    img = img - img.mean()

    h, w = img.shape
    if window:
        img = img * np.outer(np.hanning(h), np.hanning(w))

    if pad_factor > 1:
        padded = np.zeros((h * pad_factor, w * pad_factor))
        padded[:h, :w] = img
        img = padded

    return np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2


def _radius_grid(shape):
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2), (cy, cx)


def radial_profile(ps: np.ndarray, n_bins: int | None = None) -> np.ndarray:
    '''Radially averaged power spectrum.

    Rotation-invariant texture summary. Comparing the mean profile of a
    generated set against a real set says whether the two have similar
    structure across scales, without needing an Inception network.
    '''
    r, _ = _radius_grid(ps.shape)
    r = r.astype(int)
    n_bins = n_bins or (min(ps.shape) // 2)

    mask = r < n_bins
    total = np.bincount(r[mask].ravel(), weights=ps[mask].ravel(), minlength=n_bins)
    count = np.bincount(r[mask].ravel(), minlength=n_bins)
    return total[:n_bins] / np.maximum(count[:n_bins], 1)


def dominant_peak(ps: np.ndarray, min_period: float = 4.0,
                  max_period: float | None = None,
                  pad_factor: int = 1) -> dict:
    '''Strongest non-DC peak, as lattice spacing / orientation / prominence.

    Searches an annulus: the inner bound drops DC and slow illumination
    gradients (which otherwise dominate every real micrograph), the outer bound
    drops near-Nyquist pixel noise.

    Returns spacing in pixels, angle in degrees mod 180 (a lattice direction has
    no sign, and the spectrum is symmetric anyway), and prominence -- the peak
    relative to the median power in its annulus. Prominence is a sharpness
    measure: perfect synthetic lattices give tall narrow peaks, real
    measurements with defects and mixed structures give broader weaker ones, so
    it is a realism signal rather than an error.
    '''
    h, w = ps.shape
    # Padding scales every radius, so convert back to cycles over the original
    # image before reporting a spacing in original pixels.
    n = min(h, w) / pad_factor
    max_period = max_period or (n / 4.0)

    r, (cy, cx) = _radius_grid(ps.shape)
    r = r / pad_factor
    # period p <-> radius n/p, so the period bounds invert into radius bounds.
    annulus = (r >= n / max_period) & (r <= n / min_period)
    if not annulus.any():
        raise ValueError(f'Empty search annulus for a {h}x{w} spectrum.')

    masked = np.where(annulus, ps, -np.inf)
    iy, ix = np.unravel_index(np.argmax(masked), ps.shape)

    dy, dx = (iy - cy) / pad_factor, (ix - cx) / pad_factor
    radius = float(np.hypot(dy, dx))
    baseline = float(np.median(ps[annulus]))

    return {
        'spacing_px': n / radius if radius > 0 else np.inf,
        'angle_deg': float(np.degrees(np.arctan2(dy, dx)) % 180.0),
        'prominence': float(ps[iy, ix] / baseline) if baseline > 0 else np.inf,
        'radius_px': radius,
    }


def top_k_peaks(ps: np.ndarray, k: int = 4, min_period: float = 4.0,
                max_period: float | None = None, pad_factor: int = 1,
                suppress_frac: float = 0.5) -> list:
    '''The k strongest distinct peaks, via greedy non-maximum suppression.

    These are 2D lattices, not 1D fringes: the synthetic spectra show a full
    reciprocal lattice with several orders, so a method could preserve one
    lattice vector while distorting another and still look perfect to
    `dominant_peak`. Each returned peak is a dict like dominant_peak's.

    `suppress_frac` is the exclusion radius around an accepted peak, as a
    fraction of its own radius; it stops the ± mirror pair and the immediate
    shoulder of one peak being counted as separate lattice vectors.
    '''
    h, w = ps.shape
    n = min(h, w) / pad_factor
    max_period = max_period or (n / 4.0)

    r, (cy, cx) = _radius_grid(ps.shape)
    r = r / pad_factor
    available = (r >= n / max_period) & (r <= n / min_period)
    if not available.any():
        raise ValueError(f'Empty search annulus for a {h}x{w} spectrum.')
    baseline = float(np.median(ps[available]))

    yy, xx = np.mgrid[0:h, 0:w]
    peaks = []
    for _ in range(k):
        if not available.any():
            break
        iy, ix = np.unravel_index(np.argmax(np.where(available, ps, -np.inf)),
                                  ps.shape)
        dy, dx = (iy - cy) / pad_factor, (ix - cx) / pad_factor
        radius = float(np.hypot(dy, dx))
        peaks.append({
            'spacing_px': n / radius if radius > 0 else np.inf,
            'angle_deg': float(np.degrees(np.arctan2(dy, dx)) % 180.0),
            'prominence': float(ps[iy, ix] / baseline) if baseline > 0 else np.inf,
            'radius_px': radius,
        })
        # Suppress this peak and its centrosymmetric mirror.
        excl = max(suppress_frac * radius * pad_factor, 2.0)
        for sy, sx in ((iy, ix), (2 * cy - iy, 2 * cx - ix)):
            available &= np.hypot(yy - sy, xx - sx) > excl
    return peaks


def lattice_params(img: np.ndarray, pad_factor: int = 4, **kwargs) -> dict:
    '''Dominant lattice vector. Padded by default -- see power_spectrum.'''
    ps = power_spectrum(img, pad_factor=pad_factor)
    return dominant_peak(ps, pad_factor=pad_factor, **kwargs)


def lattice_signature(img: np.ndarray, k: int = 4, pad_factor: int = 4,
                      **kwargs) -> list:
    ps = power_spectrum(img, pad_factor=pad_factor)
    return top_k_peaks(ps, k=k, pad_factor=pad_factor, **kwargs)


def signature_distance(sig_a: list, sig_b: list) -> float:
    '''Mean relative spacing mismatch after greedily matching peaks by angle.

    Compares whole reciprocal lattices rather than one vector, so distortion of
    a secondary lattice direction is visible.
    '''
    if not sig_a or not sig_b:
        return np.nan
    remaining = list(sig_b)
    errors = []
    for pa in sig_a:
        best = min(remaining,
                   key=lambda pb: angle_difference(pa['angle_deg'],
                                                   pb['angle_deg']))
        errors.append(spacing_error(best['spacing_px'], pa['spacing_px']))
        if len(remaining) > 1:
            remaining.remove(best)
    return float(np.nanmean(errors))


def angle_difference(a: float, b: float) -> float:
    '''Smallest angle between two undirected lattice orientations, in [0, 90].'''
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def spacing_error(generated: float, source: float) -> float:
    '''Relative lattice-spacing error. This is the label-validity metric.'''
    if not np.isfinite(source) or source == 0:
        return np.nan
    return abs(generated - source) / source


def profile_distance(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    '''Normalised L1 distance between two radial profiles.

    Each profile is normalised to unit sum first, so this compares the shape of
    the spectrum (how power distributes across scales) rather than overall
    contrast, which differs trivially between domains.
    '''
    n = min(len(profile_a), len(profile_b))
    a, b = profile_a[:n].copy(), profile_b[:n].copy()
    a /= max(a.sum(), 1e-12)
    b /= max(b.sum(), 1e-12)
    return float(np.abs(a - b).sum())
