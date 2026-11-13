"""load_uvcgan2 against a minimal fake of the uvcgan2 API.

The real library is not a test dependency, so these fake exactly the calls the
loader makes -- Args.load, construct_model, model.find_last_checkpoint_epoch,
model.load, model.models -- as uvcgan2 defines them (eval/funcs.py,
cgan/model_base.py, cgan/uvcgan2.py at github.com/LS4GAN/uvcgan2). The loader
was also run against the real library on directories built with
train_crystals.py's config; see scripts/probe_uvcgan.py for the on-Colab check.
"""
import sys
import types

import pytest
import torch
from torch import nn

from guided_diffusion.cyclegan_loader import UnitRangeAdapter, load_generator


class Sigmoidish(nn.Module):
    """Stands in for a uvcgan2 generator: [0, 1] in, sigmoid out."""

    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        self.last_input = x.detach()
        return torch.sigmoid(self.w * (x - 0.5) * 4)


@pytest.fixture
def fake_uvcgan2(monkeypatch):
    state = {"loaded_epoch": "unset", "last_ckpt": 50, "avg": True,
             "model_args": {"lambda_a": 10.0, "avg_momentum": 0.9999}}

    class Model:
        # Mirrors uvcgan2's UVCGAN2: an explicit keyword list, no **kwargs, so
        # an unknown model_arg raises TypeError exactly like the real class.
        def __init__(self, savedir, config, is_train, device,
                     lambda_a=10.0, lambda_idt=0.5, avg_momentum=None):
            names = ["gen_ab", "gen_ba"]
            if state["avg"]:
                names += ["avg_gen_ab", "avg_gen_ba"]
            self.models = {n: Sigmoidish(n) for n in names}

        def find_last_checkpoint_epoch(self):
            return state["last_ckpt"]

        def load(self, epoch):
            state["loaded_epoch"] = epoch

    cfg_mod = types.ModuleType("uvcgan2.config")
    cfg_mod.Args = types.SimpleNamespace(
        load=lambda path: types.SimpleNamespace(
            config=types.SimpleNamespace(
                model="uvcgan2", model_args=dict(state["model_args"])),
            savedir=path))
    cgan_mod = types.ModuleType("uvcgan2.cgan")
    cgan_mod.CGAN_MODELS = {"uvcgan2": Model}
    # As uvcgan2.cgan.construct_model: model_args go straight to the class.
    cgan_mod.construct_model = lambda savedir, config, is_train, device: Model(
        savedir, config, is_train, device, **config.model_args)
    monkeypatch.setitem(sys.modules, "uvcgan2", types.ModuleType("uvcgan2"))
    monkeypatch.setitem(sys.modules, "uvcgan2.config", cfg_mod)
    monkeypatch.setitem(sys.modules, "uvcgan2.cgan", cgan_mod)
    return state


def _config(first="synth", second="real"):
    """config.json as uvcgan2 writes it: datasets in training order."""
    import json
    return json.dumps({"loss": "lsgan", "data": {"datasets": [
        {"dataset": {"name": "cyclegan", "domain": first}},
        {"dataset": {"name": "cyclegan", "domain": second}}]}})


def model_dir(tmp_path, final=True, first="synth", second="real"):
    (tmp_path / "config.json").write_text(_config(first, second))
    if final:
        for d in ("ab", "ba"):
            (tmp_path / f"net_gen_{d}.pth").write_bytes(b"")
    return str(tmp_path)


def test_prefers_ema_generator_and_final_weights(tmp_path, fake_uvcgan2):
    g = load_generator("uvcgan2", direction="ba", path=model_dir(tmp_path))
    assert isinstance(g, UnitRangeAdapter)
    assert g.generator.tag == "avg_gen_ba"
    assert fake_uvcgan2["loaded_epoch"] is None      # final weights
    assert g.source == "avg_gen_ba, final weights"


def test_falls_back_to_plain_generator_without_ema(tmp_path, fake_uvcgan2):
    fake_uvcgan2["avg"] = False
    g = load_generator("uvcgan2", direction="ab", path=model_dir(tmp_path))
    assert g.generator.tag == "gen_ab"


def test_use_avg_false_selects_raw_generator(tmp_path, fake_uvcgan2):
    g = load_generator("uvcgan2", direction="ab", path=model_dir(tmp_path),
                       use_avg=False)
    assert g.generator.tag == "gen_ab"


def test_uses_last_checkpoint_when_no_final_weights(tmp_path, fake_uvcgan2):
    g = load_generator("uvcgan2", direction="ab",
                       path=model_dir(tmp_path, final=False))
    assert fake_uvcgan2["loaded_epoch"] == 50
    assert "checkpoint epoch 50" in g.source


def test_explicit_epoch_is_honoured(tmp_path, fake_uvcgan2):
    load_generator("uvcgan2", direction="ab", path=model_dir(tmp_path),
                   epoch=100)
    assert fake_uvcgan2["loaded_epoch"] == 100


def test_reversed_domain_order_maps_to_the_other_generator(tmp_path,
                                                           fake_uvcgan2):
    """If training put REAL first, the model's gen_ab is real->synth.

    The pipeline asks by meaning (ba = real->synth), so it must get gen_ab.
    This is what a trained model with datasets [real, synth] looks like; the
    old hard-coded assumption returned the synth->real generator instead.
    """
    path = model_dir(tmp_path, first="real", second="synth")
    rs = load_generator("uvcgan2", direction="ba", path=path)
    sr = load_generator("uvcgan2", direction="ab", path=path)
    assert rs.generator.tag == "avg_gen_ab"
    assert sr.generator.tag == "avg_gen_ba"
    assert "domain a = real" in rs.source


def test_standard_domain_order_needs_no_note(tmp_path, fake_uvcgan2):
    g = load_generator("uvcgan2", direction="ba", path=model_dir(tmp_path))
    assert g.generator.tag == "avg_gen_ba"
    assert "[" not in g.source          # nothing assumed, nothing flipped


def test_unnamed_domains_fall_back_loudly(tmp_path, fake_uvcgan2):
    path = model_dir(tmp_path, first="A", second="B")
    g = load_generator("uvcgan2", direction="ba", path=path)
    assert g.generator.tag == "avg_gen_ba"
    assert "ASSUMING domain a = synth" in g.source


@pytest.mark.parametrize("epoch", [0, -1])
def test_non_positive_epoch_refused(tmp_path, fake_uvcgan2, epoch):
    """uvcgan2's load() silently loads NOTHING for epoch <= 0."""
    with pytest.raises(ValueError, match="positive"):
        load_generator("uvcgan2", direction="ab", path=model_dir(tmp_path),
                       epoch=epoch)


def test_adapter_only_converts_range():
    """The trained generator must see its input unmodified (no channel mixing
    or other preprocessing) -- only the [-1,1] -> [0,1] convention change."""
    gen = Sigmoidish("g")
    a = UnitRangeAdapter(gen)
    x = torch.zeros(1, 3, 4, 4)
    x[:, 0], x[:, 1], x[:, 2] = 0.9, 0.0, -0.3
    a(x)
    assert torch.allclose(gen.last_input, (x + 1) / 2)


def test_nothing_to_load_fails_loudly(tmp_path, fake_uvcgan2):
    fake_uvcgan2["last_ckpt"] = -1
    with pytest.raises(FileNotFoundError, match="no checkpoints"):
        load_generator("uvcgan2", direction="ab",
                       path=model_dir(tmp_path, final=False))


def test_not_a_model_directory_fails_loudly(tmp_path, fake_uvcgan2):
    """E.g. pointing at uvcgan2's OUTDIR instead of the model dir inside it."""
    with pytest.raises(FileNotFoundError, match="config.json"):
        load_generator("uvcgan2", direction="ab", path=str(tmp_path))


def test_adapter_maps_pipeline_range_to_generator_range():
    """[-1, 1] in -> generator sees [0, 1]; its [0, 1] out -> [-1, 1]."""
    gen = Sigmoidish("g")
    a = UnitRangeAdapter(gen)
    # Gray (R=G=B), like every dataset image, so the gray projection is a no-op.
    x = torch.linspace(-1, 1, 8 * 8).reshape(1, 1, 8, 8).expand(1, 3, 8, 8)
    y = a(x)
    assert gen.last_input.min() >= 0 and gen.last_input.max() <= 1
    assert torch.allclose(gen.last_input, (x + 1) / 2)
    assert y.min() >= -1 and y.max() <= 1
    assert torch.allclose(y, gen((x + 1) / 2) * 2 - 1)


def test_adapter_keeps_gradient_for_dps():
    a = UnitRangeAdapter(Sigmoidish("g")).requires_grad_(False)
    x = torch.zeros(1, 3, 8, 8, requires_grad=True)
    (g,) = torch.autograd.grad(a(x).sum(), x)
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_training_only_model_args_from_newer_uvcgan2_are_ignored_and_reported(
        tmp_path, fake_uvcgan2):
    """The WGAN run's config has `discriminator_steps: 5`, which the pinned
    uvcgan2 does not know: construct_model raised TypeError. It is a training
    setting, so loading must succeed and SAY it ignored it."""
    fake_uvcgan2["model_args"]["discriminator_steps"] = 5
    g = load_generator("uvcgan2", direction="ba", path=model_dir(tmp_path))
    assert g.generator.tag == "avg_gen_ba"
    assert "ignored" in g.source and "discriminator_steps" in g.source


def test_known_model_args_are_not_reported(tmp_path, fake_uvcgan2):
    g = load_generator("uvcgan2", direction="ba", path=model_dir(tmp_path))
    assert "ignored" not in g.source
