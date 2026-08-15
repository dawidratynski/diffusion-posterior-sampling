"""Tests for the CycleGAN-as-operator path.

The gradient tests are the important ones: DPS needs grad_x ||y - A(x)||, and
the failure mode when that breaks is silent (guidance becomes zero and sampling
quietly degrades to unconditional) rather than a crash.
"""
import numpy as np
import pytest
import torch

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.cyclegan_loader import StubGenerator, load_generator
from guided_diffusion.measurements import get_noise, get_operator

DEVICE = torch.device("cpu")
SIZE = 160  # must match UVCGAN2's (3, 160, 160)


def make_op(direction="ba", **kw):
    # 'ba' (real->synth) is the DPS operator for this project: DPS inverts its
    # operator, and the goal is generating real-like images from synthetic ones.
    return get_operator(name="cyclegan", device=DEVICE, framework="stub",
                        direction=direction, **kw)


def test_stub_shape_and_range():
    op = make_op()
    x = torch.rand(2, 3, SIZE, SIZE) * 2 - 1
    y = op.forward(x)
    assert y.shape == x.shape
    assert y.min() >= -1 and y.max() <= 1, "generator must stay in [-1, 1]"


def test_operator_is_deterministic():
    """DPS's likelihood approximation assumes a deterministic A."""
    op = make_op()
    x = torch.rand(1, 3, SIZE, SIZE) * 2 - 1
    assert torch.allclose(op.forward(x), op.forward(x))


def test_gradient_reaches_input():
    """The whole method depends on this."""
    op = make_op()
    x = (torch.rand(1, 3, SIZE, SIZE) * 2 - 1).requires_grad_()
    y = torch.rand(1, 3, SIZE, SIZE) * 2 - 1
    norm = torch.linalg.norm(y - op.forward(x))
    (grad,) = torch.autograd.grad(norm, x)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, "generator detached the autograd graph"


def test_generator_params_frozen_but_graph_intact():
    """Frozen weights must not mean a severed graph."""
    op = make_op()
    assert all(not p.requires_grad for p in op.generator.parameters())
    x = (torch.rand(1, 3, SIZE, SIZE) * 2 - 1).requires_grad_()
    assert op.forward(x).requires_grad, "forward ran without grad tracking"


def test_directions_differ():
    ab = load_generator("stub", direction="ab")
    ba = load_generator("stub", direction="ba")
    x = torch.rand(1, 3, 32, 32) * 2 - 1
    assert not torch.allclose(ab(x), ba(x))


def test_range_check_rejects_0_1_input():
    """Catches the [0,1] vs [-1,1] mismatch, which is otherwise silent."""
    op = make_op()
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        op.forward(torch.rand(1, 3, 32, 32) * 255)


def test_injected_generator_overrides_loader():
    gen = StubGenerator(seed=123)
    op = get_operator(name="cyclegan", device=DEVICE, direction="ba", generator=gen)
    assert op.generator is gen


def test_direction_must_be_stated_explicitly():
    """Wrong direction still runs and still makes images -- just the wrong ones."""
    with pytest.raises(TypeError, match="direction"):
        get_operator(name="cyclegan", device=DEVICE, framework="stub")


def test_clean_noise_supported_for_known_measurement():
    """y is a generated synthetic image, known exactly -- no measurement noise."""
    op = make_op()
    noiser = get_noise(name="clean")
    cond = get_conditioning_method("ps", op, noiser, scale=0.3)

    x_prev = (torch.rand(1, 3, 64, 64) * 2 - 1).requires_grad_()
    _, distance = cond.conditioning(
        x_prev=x_prev, x_t=torch.randn(1, 3, 64, 64),
        x_0_hat=torch.tanh(x_prev),
        measurement=torch.rand(1, 3, 64, 64) * 2 - 1,
    )
    assert torch.isfinite(distance).all()


def test_dps_conditioning_step_with_cyclegan_operator():
    """One full DPS guidance step end to end."""
    op = make_op()
    noiser = get_noise(name="gaussian", sigma=0.05)
    cond = get_conditioning_method("ps", op, noiser, scale=0.3)

    x_prev = (torch.rand(1, 3, SIZE, SIZE) * 2 - 1).requires_grad_()
    x_0_hat = torch.tanh(x_prev)
    x_t = torch.randn(1, 3, SIZE, SIZE)
    measurement = noiser(op.forward(torch.rand(1, 3, SIZE, SIZE) * 2 - 1))

    out, distance = cond.conditioning(
        x_prev=x_prev, x_t=x_t, x_0_hat=x_0_hat, measurement=measurement
    )
    assert out.shape == x_t.shape
    assert torch.isfinite(out).all()
    assert distance.item() > 0


def test_spectral_operator_is_differentiable_and_bounded():
    """The untrained stand-in must satisfy the same operator contract."""
    op = get_operator(name="cyclegan", device=DEVICE, framework="spectral",
                      direction="ba")
    x = (torch.rand(1, 3, SIZE, SIZE) * 2 - 1).requires_grad_()
    y = op.forward(x)
    assert y.shape == x.shape
    assert y.min() >= -1 and y.max() <= 1

    (grad,) = torch.autograd.grad(torch.linalg.norm(y), x)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_spectral_operator_preserves_lattice_spacing():
    """It must strip imperfections WITHOUT moving the lattice."""
    from util.lattice_metrics import lattice_params

    _, xx = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    clean = torch.sin(2 * np.pi * xx / 24.0) * 0.8
    noisy = (clean + torch.randn(SIZE, SIZE) * 0.6).clamp(-1, 1)
    x = noisy[None, None].repeat(1, 3, 1, 1)

    op = get_operator(name="cyclegan", device=DEVICE, framework="spectral",
                      direction="ba")
    out = op.forward(x).detach()[0].permute(1, 2, 0).numpy()

    assert abs(lattice_params(out)["spacing_px"] - 24.0) / 24.0 < 0.05


def test_spectral_operator_increases_periodicity():
    """Prominence must rise: that is the 'idealisation' it stands in for."""
    from util.lattice_metrics import lattice_params

    _, xx = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    noisy = (torch.sin(2 * np.pi * xx / 24.0) * 0.8
             + torch.randn(SIZE, SIZE) * 0.6).clamp(-1, 1)
    x = noisy[None, None].repeat(1, 3, 1, 1)

    op = get_operator(name="cyclegan", device=DEVICE, framework="spectral",
                      direction="ba")
    out = op.forward(x).detach()[0].permute(1, 2, 0).numpy()

    before = lattice_params(noisy.numpy())["prominence"]
    after = lattice_params(out)["prominence"]
    assert after > before, f"prominence fell: {before:.1f} -> {after:.1f}"


def test_spectral_rejects_wrong_direction():
    with pytest.raises(ValueError, match="real -> synth"):
        get_operator(name="cyclegan", device=DEVICE, framework="spectral",
                     direction="ab")
