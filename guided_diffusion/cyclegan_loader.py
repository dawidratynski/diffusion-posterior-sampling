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


def _drop_unknown_model_args(config):
    '''Remove model_args the installed uvcgan2 model class does not accept.

    model_args are the TRAINING loop's settings (loss weights, discriminator
    steps per generator step, ...); the generator architecture comes from
    config.generator. A model trained with a newer or modified uvcgan2 can
    carry settings the installed version lacks -- the WGAN run's
    `discriminator_steps: 5` (critic updates per generator update) made
    construct_model fail with a TypeError. They do not affect inference, so
    they are dropped and REPORTED (in the loaded generator's `source`, which
    the probe prints). An architecture change cannot slip through this way:
    loading the weights is strict about names and shapes.
    '''
    import inspect

    from uvcgan2.cgan import CGAN_MODELS

    cls = CGAN_MODELS.get(config.model)
    model_args = dict(config.model_args or {})
    if cls is None:
        return []
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return []
    dropped = sorted(k for k in model_args if k not in params)
    if dropped:
        config.model_args = {k: v for k, v in model_args.items()
                             if k not in dropped}
    return dropped


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
    dropped = _drop_unknown_model_args(args.config)
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
    if dropped:
        source += (f' [ignored training-only model_args unknown to the '
                   f'installed uvcgan2: {dropped}]')
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


# Gain of step 3, calibrated ONCE with SpectralIdealizer.calibrate_gain on 400
# strided real/train images at 160 px (median output std = target_std). The
# calibration set is the training split, disjoint from everything evaluated.
SPECTRAL_GAIN = 5.261

# Brightness quantiles of step 3: per-image mean brightness ([-1, 1]) at the
# 0, 5, ..., 100th percentiles, over 800 strided real/train and synth/train
# images at 160 px (SpectralIdealizer.calibrate_brightness). Measured rather
# than a normal fit: the two distributions are skewed in opposite directions
# (real +0.62, synth -0.76), which a normal fit misses by up to 0.14 in the
# tails. Both strictly increasing, so the mapping is monotone.
BRIGHTNESS_QUANTILES_REAL = (
    -0.9452, -0.7796, -0.7333, -0.7103, -0.6880, -0.6687, -0.6490, -0.6250,
    -0.5997, -0.5770, -0.5533, -0.5346, -0.5123, -0.4835, -0.4556, -0.4137,
    -0.3635, -0.3062, -0.2615, -0.1946, 0.2244)
BRIGHTNESS_QUANTILES_SYNTH = (
    -0.0958, 0.1889, 0.2417, 0.2868, 0.3275, 0.3619, 0.3942, 0.4296,
    0.4624, 0.4793, 0.4964, 0.5185, 0.5431, 0.5587, 0.5778, 0.5925,
    0.6048, 0.6180, 0.6347, 0.6516, 0.7089)


def _interp(v, xp, fp):
    '''Piecewise-linear map xp -> fp at v (1-D), extrapolating linearly past
    the ends; differentiable in v.'''
    i = torch.searchsorted(xp, v.detach().contiguous()).clamp(1, len(xp) - 1)
    x0, x1, y0, y1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    return y0 + (v - x0) * (y1 - y0) / (x1 - x0)


class SpectralIdealizer(nn.Module):
    '''An analytic, untrained stand-in for the real -> synth generator.

    Real crops are a lattice buried in noise, aperiodic clutter and a strong
    illumination gradient; synthetic ones are clean, bright and near-perfectly
    periodic. This caricatures that mapping in four steps (methods §2.2):

      0. separate brightness: subtract the image mean, so the spectrum has no
         DC term (steps 1-2 never see brightness; step 3 maps it separately).
      1. band-pass to lattice periods min_period..max_period px: removes the
         slow illumination blob (42% of real power sits at radius <= 3) and
         near-Nyquist pixel noise,
      2. peak sharpening F <- F * (|F| / max_band |F|)^gamma, relative to the
         strongest LATTICE peak: raises coherent lattice peaks above the
         incoherent floor. gamma=0 is the identity; larger gamma drives the
         output toward perfect periodicity. gamma is calibrated so the output's
         peak prominence matches the SYNTH domain, which is what a real G_{R->S}
         emits; measured on the real val split:

             gamma      0.0    0.15   0.25   0.40   0.60    1.00
             prominence 987    2433   4134   9496   31960   243547
             synth domain median: 12197

         0.4 lands closest; 1.0 overshot by 20x and started to disturb the
         lattice itself (median spacing error 0.023 vs 0.000 at 0.4),
      3. map to the synthetic domain's intensity: multiply by a FIXED gain
         (calibrated once so a typical real image comes out with synthetic
         contrast, target_std 0.38) and add the brightness of the SAME
         QUANTILE in the synthetic domain: an image at the k-th percentile of
         real brightness gets the k-th percentile of synthetic brightness.

    Why map brightness instead of discarding it: when step 0's mean was simply
    dropped, brightness was in the operator's null space, so DPS guidance put
    no constraint on it at all and each sample's brightness was whatever the
    prior produced -- and the prior is over-dispersed (variants of one input
    differed more than unrelated real photos), giving many near-black and
    near-white samples. The trained UVCGAN operator maps brightness itself,
    which is why its DPS samples did not show this. The quantile map gives the
    analytic operator the same kind of constraint, and keeps the real
    brightness distribution: guidance pulls a sample toward the real
    brightness whose quantile matches its synthetic input's.

    Why a fixed gain, not each image's own contrast: dividing by the image's
    std made the DPS guidance gradient scale with 1/contrast -- 13.6x larger
    on a faint early estimate than on a synthetic-looking image -- which drove
    pixels into the sampler's clamp (near-black outputs). With a fixed gain the
    output contrast is simply proportional to the input's lattice contrast and
    the spread is 2.1x. Step 2 is relative to the strongest LATTICE peak (not
    the overall maximum, which was the brightness term) for the same reason:
    so the result does not depend on brightness. Neither changes the output's
    shape, so the gamma calibration above still holds (real images: output
    prominence 18033 vs 17438 before, std 0.352 vs 0.353, mean 0.453 vs 0.455).

    Deterministic, differentiable and smooth in x (the sharpening weight is a
    continuous function of the spectrum, not a top-k mask). Like a real G_{R->S}
    it discards imperfections, so its null space is the imperfection manifold
    DPS is meant to resample.

    It is the "no trained translation model" ablation, not a substitute for
    the trained generators.
    '''

    def __init__(self, gamma: float = 0.4, min_period: float = 4.0,
                 max_period: float = 37.5, gain: float = SPECTRAL_GAIN,
                 softness: float = 1.5,
                 quantiles_real=BRIGHTNESS_QUANTILES_REAL,
                 quantiles_synth=BRIGHTNESS_QUANTILES_SYNTH):
        super().__init__()
        self.gamma = gamma
        self.min_period = min_period
        self.max_period = max_period
        self.gain = gain
        self.softness = softness
        self.register_buffer('q_real', torch.tensor(quantiles_real))
        self.register_buffer('q_synth', torch.tensor(quantiles_synth))

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

    def idealise(self, x):
        '''Steps 0-2: the brightness-free, sharpened lattice component.'''
        h, w = x.shape[-2:]
        x = x - x.mean(dim=(-3, -2, -1), keepdim=True)                    # 0
        band = self._bandpass(h, w, x.device, x.dtype)                     # 1
        spec = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        mag = spec.abs()
        peak = (mag * band).amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        # clamp_min: d/dm m^gamma is infinite at m = 0 for gamma < 1, so an
        # exactly-zero bin would make the DPS gradient NaN. 1e-6 of the peak
        # is far below any bin that matters.
        weight = (mag / peak).clamp_min(1e-6) ** self.gamma                # 2
        spec = spec * weight * band
        return torch.fft.ifft2(torch.fft.ifftshift(spec, dim=(-2, -1))).real

    def brightness(self, x):
        '''Step 3's brightness: the synthetic brightness at the same quantile
        as x's mean brightness among real images. Shape (N, 1, 1, 1).'''
        m = x.mean(dim=(-3, -2, -1))
        q_real = self.q_real.to(m.dtype)
        q_synth = self.q_synth.to(m.dtype)
        return _interp(m, q_real, q_synth).view(-1, 1, 1, 1)

    def forward(self, x):
        out = self.idealise(x) * self.gain + self.brightness(x)            # 3
        return out.clamp(-1.0, 1.0)

    @classmethod
    @torch.no_grad()
    def calibrate_gain(cls, real_images, target_std: float = 0.38, **kwargs):
        '''The step-3 gain that gives a typical real image synthetic contrast:
        target_std / median per-image std of idealise(real). How
        SPECTRAL_GAIN was obtained (real/train, 400 strided images, 160 px).'''
        stds = cls(**kwargs).idealise(real_images).std(dim=(-3, -2, -1))
        return float(target_std / stds.median())

    @staticmethod
    @torch.no_grad()
    def calibrate_brightness(real_images, synth_images, n_quantiles=21):
        '''The step-3 quantile tables: per-image mean brightness of each domain
        at evenly spaced percentiles. How BRIGHTNESS_QUANTILES_* were obtained
        (800 strided images from each train split, 160 px).'''
        q = torch.linspace(0, 1, n_quantiles, dtype=torch.float64)
        return tuple(
            tuple(float(v) for v in torch.quantile(
                imgs.mean(dim=(-3, -2, -1)).double(), q))
            for imgs in (real_images, synth_images))


@register_loader('spectral')
def load_spectral(direction: str, gamma: float = 0.4, min_period: float = 4.0,
                  max_period: float = 37.5, gain: float = SPECTRAL_GAIN,
                  softness: float = 1.5):
    if direction != 'ba':
        raise ValueError(
            'SpectralIdealizer only models real -> synth (direction: ba). '
            'There is no analytic stand-in for the synth -> real direction; '
            'use the trained generator for the baseline.')
    return SpectralIdealizer(gamma=gamma, min_period=min_period,
                             max_period=max_period, gain=gain,
                             softness=softness)
