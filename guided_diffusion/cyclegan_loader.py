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


class UnitRangeAdapter(nn.Module):
    '''Run a [0, 1] generator inside this pipeline's [-1, 1] convention.

    uvcgan2 feeds training images through a bare `ToTensor()` (so [0, 1], no
    normalisation) and its generator presets end in a sigmoid (so [0, 1] out).
    This pipeline -- the diffusion prior, the measurements, the saved PNGs --
    is [-1, 1] throughout. Without this affine map the generator would receive
    inputs half outside its training range and return outputs the rest of the
    pipeline reads as dark, and nothing would fail: CycleGanOperator's range
    check (|x| <= 1.5) passes [0, 1] values. The map is affine, so it keeps the
    generator differentiable, which DPS requires.
    '''

    def __init__(self, generator: nn.Module, source: str = ''):
        super().__init__()
        self.generator = generator
        self.source = source  # human-readable provenance, printed by the probe

    def forward(self, x):
        # Range conversion ONLY. The trained generator must be used exactly as
        # trained (the comparison is standard DPS + UVCGAN vs UVCGAN), so no
        # other input processing -- e.g. projecting channels to gray -- belongs
        # here.
        return self.generator((x + 1.0) / 2.0) * 2.0 - 1.0


def _uvcgan2_generator_for(path: str, direction: str):
    '''Which of the model's own generators implements a pipeline direction.

    The pipeline names directions by MEANING: 'ab' = synth -> real, 'ba' =
    real -> synth. uvcgan2 names its generators by DATASET ORDER: gen_ab maps
    datasets[0] -> datasets[1]. They only coincide if synth was datasets[0]
    in training -- an assumption that used to be hard-coded. The model's
    config.json records each dataset's domain name, so read it instead.

    Returns (model_direction, note); note is '' when nothing needed saying.
    '''
    import json
    import os
    with open(os.path.join(path, 'config.json')) as fh:
        cfg = json.load(fh)
    try:
        domains = [d['dataset'].get('domain') for d in cfg['data']['datasets']]
    except (KeyError, TypeError, AttributeError):
        domains = []
    if sorted(domains) == ['real', 'synth']:
        if domains[0] == 'synth':
            return direction, ''
        flipped = {'ab': 'ba', 'ba': 'ab'}[direction]
        return flipped, (f'config.json has domain a = real, so pipeline '
                         f'{direction} is the model\'s gen_{flipped}')
    return direction, (f'domain names {domains} not recognised in config.json;'
                       ' ASSUMING domain a = synth')


@register_loader('uvcgan2')
def load_uvcgan2(direction: str, path: str, epoch: int | None = None,
                 use_avg: bool = True):
    '''Load one generator from a UVCGAN2 (LS4GAN) model directory.

    `path` is the directory uvcgan2's train() wrote: config.json, the final
    weights `net_gen_ab.pth` / `net_gen_ba.pth` (+ `net_avg_gen_*`), and
    periodic snapshots under checkpoints/NNNN_net_*.pth. Verified against
    uvcgan2 at github.com/LS4GAN/uvcgan2 (eval/funcs.py:start_model_eval).

    epoch: None = the final weights if present, else the newest checkpoint;
        an int = that checkpoint.
    use_avg: use the EMA generator (`avg_gen_*`) when the model has one. This
        is what uvcgan2's own inference uses (UVCGAN2.forward dispatches to
        'avg-ab'/'avg-ba' whenever avg_momentum is set), so it is the model the
        training run actually produced.

    Returns the generator wrapped in UnitRangeAdapter -- see its docstring.
    '''
    import os

    try:
        from uvcgan2.cgan import construct_model
        from uvcgan2.config import Args
    except ImportError as e:
        raise ImportError(
            'uvcgan2 is not installed. Install it from '
            'https://github.com/LS4GAN/uvcgan2 (the notebook does this in '
            'section 12).') from e

    if not os.path.exists(os.path.join(path, 'config.json')):
        raise FileNotFoundError(
            f'{path} has no config.json, so it is not a uvcgan2 model '
            'directory. Point --uvcgan_path at the directory train() wrote '
            '(it contains config.json and net_gen_ab.pth).')

    # `direction` is the pipeline's (ab = synth->real); `model_dir` is the
    # name of the uvcgan2 generator that implements it in THIS model.
    model_dir, dir_note = _uvcgan2_generator_for(path, direction)

    args = Args.load(path)
    model = construct_model(path, args.config, is_train=False, device='cpu')

    if epoch is None:
        if os.path.exists(os.path.join(path, f'net_gen_{model_dir}.pth')):
            source = 'final weights'
        else:
            epoch = model.find_last_checkpoint_epoch()
            if epoch <= 0:
                raise FileNotFoundError(
                    f'No final weights (net_gen_{model_dir}.pth) and no '
                    f'checkpoints under {path}/checkpoints.')
            source = f'checkpoint epoch {epoch} (no final weights)'
    else:
        # uvcgan2's ModelBase.load() returns WITHOUT loading anything for
        # epoch <= 0, which would hand back the random initialisation.
        if epoch <= 0:
            raise ValueError(f'epoch must be a positive checkpoint number or '
                             f'None (final weights), got {epoch}.')
        source = f'checkpoint epoch {epoch}'
    model.load(epoch)
    loss = getattr(args.config, 'loss', None)
    if loss:
        source += f', trained with loss={loss}'

    gens = model.models
    name = f'gen_{model_dir}'
    if use_avg and f'avg_gen_{model_dir}' in gens \
            and gens[f'avg_gen_{model_dir}'] is not None:
        name = f'avg_gen_{model_dir}'
    if name not in gens or gens[name] is None:
        raise AttributeError(
            f"'{name}' not found on the loaded UVCGAN2 model; it has "
            f'{sorted(k for k in gens if "gen" in k)}.')
    if dir_note:
        source += f' [{dir_note}]'
    return UnitRangeAdapter(gens[name], source=f'{name}, {source}')


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
        # clamp_min: d/dm m^gamma is infinite at m = 0 for gamma < 1, so an
        # exactly-zero bin (flat or perfectly periodic x0_hat) made the DPS
        # gradient NaN. 1e-6 of the peak is far below any bin that matters.
        weight = (mag / peak).clamp_min(1e-6) ** self.gamma
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
