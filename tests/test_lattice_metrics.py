"""Tests for the lattice metrics.

These use synthetic gratings with known period and orientation, so the expected
answers are exact rather than eyeballed.
"""
import numpy as np
import pytest

from util.lattice_metrics import (
    angle_difference,
    lattice_params,
    power_spectrum,
    profile_distance,
    radial_profile,
    spacing_error,
)

N = 160


def grating(period, angle_deg=0.0, size=N, amplitude=1.0, phase=0.0):
    """Sinusoidal lattice with an exactly known period and orientation."""
    yy, xx = np.mgrid[0:size, 0:size]
    theta = np.radians(angle_deg)
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    return amplitude * np.sin(2 * np.pi * proj / period + phase)


@pytest.mark.parametrize("period", [6.0, 8.0, 10.0, 16.0, 20.0])
def test_recovers_known_spacing(period):
    got = lattice_params(grating(period))["spacing_px"]
    assert abs(got - period) / period < 0.06, f"expected ~{period}, got {got:.2f}"


@pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, 90.0, 135.0])
def test_recovers_known_orientation(angle):
    got = lattice_params(grating(10.0, angle_deg=angle))["angle_deg"]
    # A grating at angle a has its spectral peak along a; undirected, so mod 180.
    assert angle_difference(got, angle) < 6.0, f"expected ~{angle}, got {got:.1f}"


def test_spacing_invariant_to_phase_and_amplitude():
    """Shifting or rescaling the pattern must not move the measured spacing."""
    base = lattice_params(grating(12.0))["spacing_px"]
    shifted = lattice_params(grating(12.0, phase=1.3))["spacing_px"]
    scaled = lattice_params(grating(12.0, amplitude=0.25))["spacing_px"]
    assert abs(base - shifted) < 0.5
    assert abs(base - scaled) < 0.5


def test_prominence_drops_when_noise_is_added():
    """Prominence is the realism signal: clean lattice sharp, degraded broader."""
    rng = np.random.default_rng(0)
    clean = grating(10.0)
    noisy = clean + rng.normal(0, 1.5, clean.shape)
    assert (lattice_params(clean)["prominence"]
            > lattice_params(noisy)["prominence"])


def test_dc_and_gradient_are_ignored():
    """A brightness offset plus a ramp must not be mistaken for a lattice."""
    yy, xx = np.mgrid[0:N, 0:N]
    img = grating(9.0) + 5.0 + 0.05 * xx + 0.03 * yy
    assert abs(lattice_params(img)["spacing_px"] - 9.0) / 9.0 < 0.06


def test_windowing_suppresses_edge_leakage():
    """Unwindowed crops leak a cross through the origin that mimics structure."""
    img = grating(9.0, angle_deg=37.0)
    ps_win = power_spectrum(img, window=True)
    ps_raw = power_spectrum(img, window=False)
    cy, cx = N // 2, N // 2
    # Power along the axes away from the true peak = leakage.
    leak_win = ps_win[cy, cx + 5:].sum() + ps_win[cy + 5:, cx].sum()
    leak_raw = ps_raw[cy, cx + 5:].sum() + ps_raw[cy + 5:, cx].sum()
    assert leak_win < leak_raw


def test_spacing_error_is_relative():
    assert spacing_error(10.0, 10.0) == 0.0
    assert spacing_error(11.0, 10.0) == pytest.approx(0.1)
    assert np.isnan(spacing_error(10.0, 0.0))


def test_angle_difference_wraps():
    assert angle_difference(0.0, 179.0) == pytest.approx(1.0)
    assert angle_difference(10.0, 100.0) == pytest.approx(90.0)
    assert angle_difference(45.0, 45.0) == 0.0


def test_profile_distance_zero_for_identical_and_positive_otherwise():
    a = radial_profile(power_spectrum(grating(10.0)))
    b = radial_profile(power_spectrum(grating(10.0, phase=0.7)))
    c = radial_profile(power_spectrum(grating(20.0)))
    assert profile_distance(a, a) == pytest.approx(0.0)
    assert profile_distance(a, b) < profile_distance(a, c)


def test_profile_distance_ignores_contrast():
    """Normalised, so a pure contrast change must not register as difference."""
    a = radial_profile(power_spectrum(grating(10.0, amplitude=1.0)))
    b = radial_profile(power_spectrum(grating(10.0, amplitude=4.0)))
    assert profile_distance(a, b) < 1e-9


def two_d_lattice(p1, p2, a1=0.0, a2=90.0, size=N):
    """Two superposed gratings -> a 2D reciprocal lattice, like the synth data."""
    return grating(p1, a1, size) + grating(p2, a2, size)


def test_padding_improves_spacing_resolution():
    """At radius ~5 integer bins quantise spacing into ~17% steps."""
    # 27.3 sits between the 150/5=30 and 150/6=25 bins.
    period = 27.3
    unpadded = lattice_params(grating(period), pad_factor=1)["spacing_px"]
    padded = lattice_params(grating(period), pad_factor=8)["spacing_px"]
    assert abs(padded - period) < abs(unpadded - period)
    assert abs(padded - period) / period < 0.03


def test_top_k_finds_both_lattice_vectors():
    """A 2D lattice has two independent vectors; one peak cannot see both."""
    from util.lattice_metrics import lattice_signature

    sig = lattice_signature(two_d_lattice(10.0, 16.0, 0.0, 90.0), k=2)
    assert len(sig) == 2
    spacings = sorted(p["spacing_px"] for p in sig)
    assert abs(spacings[0] - 10.0) / 10.0 < 0.08
    assert abs(spacings[1] - 16.0) / 16.0 < 0.08


def test_top_k_peaks_are_distinct():
    from util.lattice_metrics import lattice_signature

    sig = lattice_signature(two_d_lattice(9.0, 14.0, 20.0, 100.0), k=3)
    angles = [p["angle_deg"] for p in sig]
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            assert angle_difference(angles[i], angles[j]) > 5.0


def test_signature_distance_catches_secondary_vector_distortion():
    """The failure dominant_peak alone would miss."""
    from util.lattice_metrics import lattice_signature, signature_distance

    ref = lattice_signature(two_d_lattice(10.0, 16.0, 0.0, 90.0), k=2)
    same = lattice_signature(two_d_lattice(10.0, 16.0, 0.0, 90.0), k=2)
    # Primary vector preserved, secondary stretched by 25%.
    skewed = lattice_signature(two_d_lattice(10.0, 20.0, 0.0, 90.0), k=2)

    assert signature_distance(ref, same) < 0.05
    # Averaged over 2 peaks, one perfect and one 25% off -> ~0.125.
    assert signature_distance(ref, skewed) > 0.10

    # The point: dominant_peak alone sees the untouched primary vector and
    # reports no error at all, so this distortion is invisible to it.
    d_ref = lattice_params(two_d_lattice(10.0, 16.0, 0.0, 90.0))
    d_skew = lattice_params(two_d_lattice(10.0, 20.0, 0.0, 90.0))
    assert spacing_error(d_skew["spacing_px"], d_ref["spacing_px"]) < 0.01
