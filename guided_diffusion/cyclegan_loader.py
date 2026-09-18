'''Loading CycleGAN-family generators as DPS measurement operators.

Each backend returns a plain `nn.Module` mapping [-1, 1] images to [-1, 1]
images. Keeping loading separate from the operator itself means the analytic
operator and a trained one are interchangeable at the call site, which is what
lets the same pipeline evaluate both as DPS operators.

Every registered framework is a real operator. Loaders declare their parameters
explicitly and reject any others, so a config cannot name one framework while
carrying another's settings -- see load_generator().

Direction convention throughout: domain A = synth (simulated/ideal),
domain B = real (electron microscope), matching uvcgan_training/train_crystals.py
where domain 'a' is synth. So:

    ba  real  -> synth  the DPS forward operator
    ab  synth -> real   the direct-translation baseline DPS is compared against

The goal is generating realistic measurements from synthetic inputs with known
lattice parameters, so the diffusion prior lives on REAL and DPS samples

    x ~ p_R(x) . p(s | G_{R->S}(x))

i.e. "a realistic image whose synth-domain projection is my known s". Note the
DPS operator is `ba` even though the overall S -> R mapping is what is wanted:
DPS inverts its operator.
'''
import inspect

import torch
from torch import nn

__LOADER__ = {}


def register_loader(name: str):
    def wrapper(fn):
        if __LOADER__.get(name, None):
            raise NameError(f"Loader {name} is already registered!")
        __LOADER__[name] = fn
        return fn
    return wrapper


def load_generator(framework: str, direction: str = 'ab', **kwargs) -> nn.Module:
    if framework not in __LOADER__:
        raise NameError(f"Unknown framework '{framework}'. "
                        f"Available: {sorted(__LOADER__)}")
    if direction not in ('ab', 'ba'):
        raise ValueError(f"direction must be 'ab' or 'ba', got {direction!r}")

    # Reject parameters the chosen loader does not accept. Every loader used to
    # end in **kwargs, so a config carrying another framework's settings loaded
    # silently with those settings ignored. `framework: stub` alongside
    # `gamma: 0.4` ran the random-weight stub while looking like a configured
    # spectral operator, and a whole run of results was attributed to the wrong
    # operator. Failing loudly is the only way that class of error is visible.
    sig = inspect.signature(__LOADER__[framework])
    accepted = {n for n, prm in sig.parameters.items()
                if prm.kind is not inspect.Parameter.VAR_KEYWORD}
    unknown = set(kwargs) - accepted
    if unknown:
        raise TypeError(
            f"framework '{framework}' does not accept {sorted(unknown)}; it "
            f"accepts {sorted(accepted - {'direction'})}. Those parameters "
            "usually belong to a different framework -- check that the config's "
            "`framework:` is the operator you meant to use.")
    return __LOADER__[framework](direction=direction, **kwargs)


@register_loader('uvcgan2')
def load_uvcgan2(direction: str, path: str, epoch: int = -1):
    '''Load a UVCGAN2 (LS4GAN) generator.

    UVCGAN2 saves a whole model directory (weights *plus* the config needed to
    rebuild the architecture), so `path` is that directory, not a .pth file.

    The exact accessor differs between uvcgan2 versions, so probe rather than
    assume. If this fails, print the loaded object and adjust -- the generator
    is a submodule of the returned model wrapper.
    '''
    try:
        from uvcgan2.utils.funcs import load_model_from_path
    except ImportError:
        try:
            from uvcgan2.utils.eval import load_model_from_path
        except ImportError as e:
            raise ImportError(
                'uvcgan2 is not installed. Install it from '
                'https://github.com/LS4GAN/uvcgan2 (see '
                'uvcgan_training/clone_repo.py in the sibling repo).') from e

    model = load_model_from_path(path, epoch=epoch, device='cpu')

    attr = f'gen_{direction}'
    holder = getattr(model, 'models', model)
    gen = getattr(holder, attr, None)
    if gen is None:
        available = [a for a in dir(holder) if a.startswith('gen')]
        raise AttributeError(
            f"Could not find '{attr}' on the loaded UVCGAN2 model. "
            f"Generator-like attributes present: {available}. "
            'Adjust load_uvcgan2() to match your uvcgan2 version.')
    return gen


@register_loader('cyclegan_resnet')
def load_cyclegan_resnet(direction: str, path: str, ngf: int = 64,
                         n_blocks: int = 9):
    '''Load a junyanz pytorch-CycleGAN-and-pix2pix generator.

    That repo saves a bare generator state_dict (`latest_net_G_A.pth`), so the
    architecture has to be reconstructed from its `models/networks.py`; clone it
    next to this repo and make it importable.
    '''
    try:
        from models.networks import define_G
    except ImportError as e:
        raise ImportError(
            'Could not import models.networks.define_G. Clone '
            'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and put '
            'it on sys.path.') from e

    gen = define_G(input_nc=3, output_nc=3, ngf=ngf,
                   netG=f'resnet_{n_blocks}blocks', norm='instance',
                   use_dropout=False, init_type='normal', gpu_ids=[])
    sd = torch.load(path, map_location='cpu', weights_only=True)
    # DataParallel-saved checkpoints carry a 'module.' prefix.
    sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    gen.load_state_dict(sd)
    return gen


@register_loader('torchscript')
def load_torchscript(direction: str, path: str):
    '''Load a TorchScript-traced generator.

    The most robust option if whoever trains the GAN can export one: it carries
    its own architecture, needs none of their training code, and still supports
    autograd through the traced graph (which DPS requires).
    '''
    return torch.jit.load(path, map_location='cpu')


# A `stub` framework used to live here: a small randomly-initialised conv net,
# for exercising the plumbing before any real or analytic operator existed. It
# was removed because `spectral` supersedes it and it was actively dangerous --
# being registered and named in the default config, it silently became the
# operator for a whole run of experiments whose results were then attributed to
# the spectral operator. A random conv net BLURS the lattice where the intended
# operator SHARPENS it, so the mix-up inverted the experiment rather than
# perturbing it. Nothing should need a placeholder operator again; if something
# does, give it a name that cannot be mistaken for a real one.


class SpectralIdealizer(nn.Module):
    '''An analytic, untrained stand-in for the real -> synth generator.

    Real crops are a lattice buried in noise, aperiodic clutter and a strong
    illumination gradient; synthetic ones are clean, bright and near-perfectly
    periodic. This caricatures that mapping with three spectral operations:

      1. a high-pass that removes the illumination blob (42% of real power sits
         at radius <= 3, measured on the dataset),
      2. peak sharpening `F <- F * (|F|/max|F|)^gamma`, which raises coherent
         lattice peaks above the incoherent floor -- gamma=0 is the identity,
         larger gamma drives the output toward perfect periodicity. gamma is
         calibrated so the output's peak prominence matches the SYNTH domain,
         which is what a real G_{R->S} emits; measured on the real val split:

             gamma      0.0    0.15   0.25   0.40   0.60    1.00
             prominence 987    2433   4134   9496   31960   243547
             synth domain median: 12197

         0.4 lands closest. The original 1.0 overshot by 20x, making the
         stand-in far harsher than the operator it stands in for -- and it also
         started to disturb the lattice itself (median spacing error 0.023 at
         gamma 1.0 versus 0.000 at 0.4),
      3. renormalisation to the synthetic domain's intensity statistics.

    Purpose: end-to-end runs before UVCGAN weights exist. It needs no training,
    is deterministic, and is smooth in x (the sharpening weight is a continuous
    function of the spectrum, not a top-k mask, so no discontinuous jumps in the
    DPS gradient). Like a real G_{R->S} it discards imperfections, so its null
    space is the imperfection manifold DPS is meant to resample -- which is what
    makes it a useful rehearsal rather than just a placeholder.

    NOT a substitute for the trained model in any reported result.
    '''

    def __init__(self, gamma: float = 0.4, min_period: float = 4.0,
                 max_period: float = 37.5, target_mean: float = 0.47,
                 target_std: float = 0.38, softness: float = 1.5):
        super().__init__()
        self.gamma = gamma
        self.min_period = min_period
        self.max_period = max_period
        # Synthetic-domain statistics in [-1, 1]: measured mean 188/255 and
        # std 49/255 over the synth split.
        self.target_mean = target_mean
        self.target_std = target_std
        self.softness = softness

    def _bandpass(self, h, w, device, dtype):
        fy = torch.fft.fftshift(torch.fft.fftfreq(h, device=device)) * h
        fx = torch.fft.fftshift(torch.fft.fftfreq(w, device=device)) * w
        r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        n = min(h, w)
        # Smooth (sigmoid) edges rather than a hard annulus: a brick-wall mask
        # produces ringing and a discontinuous gradient.
        lo, hi = n / self.max_period, n / self.min_period
        band = (torch.sigmoid((r - lo) / self.softness)
                * torch.sigmoid((hi - r) / self.softness))
        return band.to(dtype)

    def forward(self, x):
        h, w = x.shape[-2:]
        spec = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))

        mag = spec.abs()
        peak = mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        weight = (mag / peak) ** self.gamma
        spec = spec * weight * self._bandpass(h, w, x.device, weight.dtype)

        out = torch.fft.ifft2(torch.fft.ifftshift(spec, dim=(-2, -1))).real

        mean = out.mean(dim=(-3, -2, -1), keepdim=True)
        std = out.std(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
        out = (out - mean) / std * self.target_std + self.target_mean
        return out.clamp(-1.0, 1.0)


@register_loader('spectral')
def load_spectral(direction: str, gamma: float = 0.4, min_period: float = 4.0,
                  max_period: float = 37.5, target_mean: float = 0.47,
                  target_std: float = 0.38, softness: float = 1.5):
    if direction != 'ba':
        raise ValueError(
            'SpectralIdealizer only models real -> synth (direction: ba). '
            'There is no analytic stand-in for the synth -> real direction; '
            'use the trained generator for the baseline.')
    return SpectralIdealizer(gamma=gamma, min_period=min_period,
                             max_period=max_period, target_mean=target_mean,
                             target_std=target_std, softness=softness)
