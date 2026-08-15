"""CPU smoke tests: verify the DPS pipeline is wired correctly without a GPU.

These use a randomly-initialised tiny UNet at low resolution with few timesteps,
so they run in seconds. They check *plumbing* (shapes, dtypes, gradient flow),
not sample quality.
"""
import torch

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model

DEVICE = torch.device("cpu")
IMG = 64


def tiny_model():
    return create_model(
        image_size=IMG,
        num_channels=32,
        num_res_blocks=1,
        channel_mult="1,2",
        learn_sigma=False,
        class_cond=False,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
    ).to(DEVICE).eval()


def tiny_sampler(respacing="10"):
    # Keep steps=1000: the `linear` schedule rescales betas by 1000/steps, so a
    # small `steps` produces betas > 1. Shorten sampling via timestep_respacing.
    return create_sampler(
        sampler="ddpm",
        steps=1000,
        noise_schedule="linear",
        model_mean_type="epsilon",
        model_var_type="fixed_large",  # matches learn_sigma=False
        dynamic_threshold=False,
        clip_denoised=True,
        rescale_timesteps=False,
        timestep_respacing=respacing,
    )


def test_model_forward():
    model = tiny_model()
    x = torch.randn(1, 3, IMG, IMG)
    t = torch.tensor([0])
    out = model(x, t)
    assert out.shape == (1, 3, IMG, IMG)


def test_operator_gradient_flows():
    """DPS needs d||y - A(x0_hat)|| / d x_prev. Any operator must not break autograd."""
    op = get_operator(name="gaussian_blur", kernel_size=9, intensity=1.0, device=DEVICE)
    x_prev = torch.randn(1, 3, IMG, IMG, requires_grad=True)
    x_0_hat = x_prev * 0.5  # stand-in for the denoiser output
    y = torch.randn(1, 3, IMG, IMG)
    norm = torch.linalg.norm(y - op.forward(x_0_hat))
    (grad,) = torch.autograd.grad(norm, x_prev)
    assert grad.shape == x_prev.shape
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, "operator detached the graph"


def test_full_dps_loop(tmp_path):
    model = tiny_model()
    sampler = tiny_sampler(respacing="10")
    op = get_operator(name="gaussian_blur", kernel_size=9, intensity=1.0, device=DEVICE)
    noiser = get_noise(name="gaussian", sigma=0.05)
    cond = get_conditioning_method("ps", op, noiser, scale=0.3)

    ref = torch.randn(1, 3, IMG, IMG)
    y_n = noiser(op.forward(ref))
    x_start = torch.randn_like(ref).requires_grad_()

    (tmp_path / "progress").mkdir()
    sample = sampler.p_sample_loop(
        model=model,
        x_start=x_start,
        measurement=y_n,
        measurement_cond_fn=cond.conditioning,
        record=False,
        save_root=str(tmp_path),
    )
    assert sample.shape == ref.shape
    assert torch.isfinite(sample).all()
