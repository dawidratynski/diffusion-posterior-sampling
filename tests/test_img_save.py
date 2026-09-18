"""Tests for the image-saving convention.

Anything saved and later measured must go through a FIXED affine map, not
per-image min-max normalisation. These guard two bugs that silently corrupted
results rather than raising.
"""
import numpy as np
import pytest
import torch

from util.img_utils import clear_color, normalize_np, to_display


def test_to_display_preserves_brightness_differences():
    """The bug: min-max normalisation made every output span the full range,
    so brightness and contrast differences between outputs vanished."""
    dark = torch.full((1, 3, 8, 8), -0.6)
    bright = torch.full((1, 3, 8, 8), 0.4)

    assert to_display(dark).mean() < to_display(bright).mean()
    assert to_display(dark).mean() == pytest.approx(0.2, abs=1e-6)     # (-0.6+1)/2
    assert to_display(bright).mean() == pytest.approx(0.7, abs=1e-6)   # ( 0.4+1)/2


def test_to_display_is_the_exact_inverse_of_the_dataset_mapping():
    """CrystalDataset maps PNG [0,1] -> [-1,1]; saving must invert it exactly,
    or a generated PNG is not on the same scale as the source it is compared to.
    """
    png_values = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    as_tensor = torch.from_numpy(png_values)[None, None].repeat(1, 3, 1, 1) * 2 - 1
    recovered = to_display(as_tensor)[..., 0]
    assert np.allclose(recovered, png_values, atol=1e-6)


def test_to_display_does_not_mutate_its_input():
    x = torch.linspace(-1, 1, 48).reshape(1, 3, 4, 4).clone()
    before = x.clone()
    to_display(x)
    assert torch.equal(before, x)


def test_to_display_clips_out_of_range_values():
    x = torch.tensor([-3.0, 3.0]).reshape(1, 1, 1, 2).repeat(1, 3, 1, 1)
    out = to_display(x)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_normalize_np_does_not_mutate_its_input():
    """clear_color reaches the array via .numpy(), which shares storage with the
    tensor on CPU -- so the in-place version rewrote the caller's tensor. The
    measurement was saved and then sampled from, so DPS conditioned on a
    rescaled y while the operator produced [-1,1]."""
    arr = np.linspace(-1.0, 1.0, 16).reshape(4, 4)
    before = arr.copy()
    normalize_np(arr)
    assert np.array_equal(before, arr)


def test_clear_color_does_not_mutate_the_source_tensor():
    x = torch.linspace(-1, 1, 48).reshape(1, 3, 4, 4).clone()
    before = x.clone()
    clear_color(x)
    assert torch.equal(before, x), 'saving an image must not alter it'


def test_normalize_np_handles_a_constant_image():
    """max == 0 after mean subtraction divided by zero."""
    out = normalize_np(np.full((4, 4), 0.5))
    assert np.isfinite(out).all()
