"""CPU tests for the diffusion trainer.

Checks that the objective is correct and that a checkpoint round-trips back
through create_model -- i.e. that what you train on Colab is loadable by the
sampler without further translation.
"""
import numpy as np
import PIL.Image
import pytest
import torch

from data.dataloader import get_dataset
from guided_diffusion.train_util import EMA, DiffusionTrainer
from guided_diffusion.unet import create_model

DEVICE = torch.device("cpu")


def tiny_model(image_size=32, learn_sigma=False):
    return create_model(
        image_size=image_size, num_channels=32, num_res_blocks=1,
        channel_mult="1,2", learn_sigma=learn_sigma, attention_resolutions="16",
        num_heads=1, num_head_channels=-1,
    )


def test_q_sample_matches_closed_form():
    tr = DiffusionTrainer("linear", 1000, DEVICE)
    x0 = torch.randn(4, 3, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 10, 500, 999])
    x_t = tr.q_sample(x0, t, noise)

    for i, ti in enumerate(t):
        a = tr.sqrt_alphas_cumprod[ti]
        b = tr.sqrt_one_minus_alphas_cumprod[ti]
        assert torch.allclose(x_t[i], a * x0[i] + b * noise[i], atol=1e-6)


def test_q_sample_endpoints():
    """t=0 is nearly clean; t=T-1 is nearly pure noise."""
    tr = DiffusionTrainer("linear", 1000, DEVICE)
    x0 = torch.randn(1, 3, 16, 16)
    noise = torch.randn_like(x0)

    near_clean = tr.q_sample(x0, torch.tensor([0]), noise)
    near_noise = tr.q_sample(x0, torch.tensor([999]), noise)
    assert (near_clean - x0).abs().mean() < (near_noise - x0).abs().mean()
    assert tr.sqrt_alphas_cumprod[999] < 0.05


def test_loss_decreases_on_a_single_batch():
    """Overfitting one batch is the standard 'is the objective wired up' check."""
    torch.manual_seed(0)
    model = tiny_model()
    tr = DiffusionTrainer("linear", 1000, DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)

    torch.manual_seed(0)
    first = tr.loss(model, x).item()
    for _ in range(30):
        opt.zero_grad()
        tr.loss(model, x).backward()
        opt.step()
    torch.manual_seed(0)
    last = tr.loss(model, x).item()

    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"


def test_learn_sigma_gives_actionable_error():
    model = tiny_model(learn_sigma=True)
    tr = DiffusionTrainer("linear", 1000, DEVICE)
    with pytest.raises(ValueError, match="learn_sigma"):
        tr.loss(model, torch.randn(1, 3, 32, 32))


def test_ema_tracks_and_lags():
    model = tiny_model()
    ema = EMA(model, decay=0.9)
    before = [p.clone() for p in ema.ema.parameters()]

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)

    moved = [(a - b).abs().mean().item()
             for a, b in zip(ema.ema.parameters(), before)]
    assert all(0 < m < 1.0 for m in moved), "EMA should move, but lag the model"


def test_checkpoint_roundtrips_through_create_model(tmp_path):
    """A saved EMA state_dict must load straight back via model_path."""
    model = tiny_model()
    ema = EMA(model, decay=0.5)
    path = tmp_path / "model_ema.pt"
    torch.save(ema.state_dict(), path)

    reloaded = create_model(
        image_size=32, num_channels=32, num_res_blocks=1, channel_mult="1,2",
        learn_sigma=False, attention_resolutions="16", num_heads=1,
        num_head_channels=-1, model_path=str(path),
    )
    for a, b in zip(reloaded.state_dict().values(), ema.state_dict().values()):
        assert torch.equal(a, b)


def test_create_model_raises_on_bad_checkpoint(tmp_path):
    """Upstream silently returned random weights here."""
    with pytest.raises(FileNotFoundError):
        tiny_model_path = str(tmp_path / "does_not_exist.pt")
        create_model(
            image_size=32, num_channels=32, num_res_blocks=1, channel_mult="1,2",
            learn_sigma=False, attention_resolutions="16", num_heads=1,
            num_head_channels=-1, model_path=tiny_model_path,
        )


def test_crystal_dataset_resizes_150_to_160(tmp_path):
    """Data on disk is 150px; UVCGAN2 and the prior both work at 160px."""
    for i in range(3):
        arr = (np.random.rand(150, 150, 3) * 255).astype(np.uint8)
        PIL.Image.fromarray(arr).save(tmp_path / f"{i}.png")

    ds = get_dataset(name="crystal", root=str(tmp_path), image_size=160)
    assert len(ds) == 3
    img = ds[0]
    assert img.shape == (3, 160, 160)
    assert -1.0 <= img.min() and img.max() <= 1.0


@pytest.mark.parametrize("var_type", ["fixed_small", "fixed_large", "learned_range"])
def test_variance_processors_are_finite(var_type):
    """fixed_small used to take log(0) at t=0 -> -inf."""
    from guided_diffusion.gaussian_diffusion import get_named_beta_schedule
    from guided_diffusion.posterior_mean_variance import get_var_processor

    betas = np.array(get_named_beta_schedule("linear", 1000), dtype=np.float64)
    proc = get_var_processor(var_type, betas=betas)
    x = torch.randn(1, 3, 8, 8)
    for ti in (0, 1, 500, 999):
        var, logvar = proc.get_variance(x, torch.tensor([ti]))
        assert torch.isfinite(var).all(), f"{var_type} var not finite at t={ti}"
        assert torch.isfinite(logvar).all(), f"{var_type} logvar not finite at t={ti}"


def test_ema_warmup_tracks_model_early():
    """Without warmup a short run exports an essentially random checkpoint."""
    model = tiny_model()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)

    warm = EMA(tiny_model(), decay=0.9999, warmup=True)
    fixed = EMA(tiny_model(), decay=0.9999, warmup=False)
    for _ in range(300):
        warm.update(model)
        fixed.update(model)

    def mean_gap(ema):
        return float(np.mean([(p - 1.0).abs().mean().item()
                              for p in ema.ema.parameters()]))

    # 0.9999**300 = 0.970, so the fixed-decay average barely moved.
    assert mean_gap(warm) < 0.2 * mean_gap(fixed)


def test_ema_decay_ramps_then_caps():
    ema = EMA(tiny_model(), decay=0.999)
    assert ema.current_decay() == pytest.approx(0.1)
    for _ in range(90):
        ema.num_updates += 1
    assert ema.current_decay() == pytest.approx(91 / 100)
    ema.num_updates = 10**6
    assert ema.current_decay() == pytest.approx(0.999)  # capped at target


def test_ema_update_count_survives_checkpoint_roundtrip(tmp_path):
    """Resuming must not reset the decay ramp."""
    model = tiny_model()
    ema = EMA(model, decay=0.9999)
    for _ in range(50):
        ema.update(model)

    path = tmp_path / "ck.pt"
    torch.save({"ema": ema.state_dict(), "ema_updates": ema.num_updates}, path)
    ck = torch.load(path, map_location="cpu", weights_only=True)

    restored = EMA(tiny_model(), decay=0.9999)
    restored.load_state_dict(ck["ema"])
    restored.num_updates = ck["ema_updates"]
    assert restored.current_decay() == pytest.approx(ema.current_decay())


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_amp_backward_works_with_gradient_checkpointing(use_checkpoint):
    """Reproduces the Colab crash:

        RuntimeError: Input type (c10::Half) and bias type (float)
                      should be the same

    CheckpointFunction.backward re-runs the forward, and without restoring the
    autocast state that recomputation is fp32 while the saved activations are
    fp16. Exercised on CPU via bfloat16 autocast, which hits the same code path.
    """
    model = create_model(
        image_size=32, num_channels=32, num_res_blocks=1, channel_mult="1,2",
        learn_sigma=False, attention_resolutions="16", num_heads=1,
        num_head_channels=-1, use_checkpoint=use_checkpoint,
    )
    tr = DiffusionTrainer("linear", 1000, DEVICE)
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        loss = tr.loss(model, x)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads)


def test_attention_block_honours_use_checkpoint():
    """Upstream hardcoded checkpointing on regardless of the flag."""
    from guided_diffusion.unet import AttentionBlock

    assert AttentionBlock(32, num_heads=1, use_checkpoint=False).use_checkpoint is False
    assert AttentionBlock(32, num_heads=1, use_checkpoint=True).use_checkpoint is True

    calls = []
    import guided_diffusion.unet as unet_mod
    real = unet_mod.checkpoint

    def spy(func, inputs, params, flag):
        calls.append(flag)
        return real(func, inputs, params, flag)

    unet_mod.checkpoint = spy
    try:
        AttentionBlock(32, num_heads=1, use_checkpoint=False)(torch.randn(1, 32, 8, 8))
        AttentionBlock(32, num_heads=1, use_checkpoint=True)(torch.randn(1, 32, 8, 8))
    finally:
        unet_mod.checkpoint = real

    assert calls == [False, True], f"flag not forwarded: {calls}"
