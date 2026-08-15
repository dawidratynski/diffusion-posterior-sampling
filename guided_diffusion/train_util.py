'''Training utilities for the unconditional diffusion prior.

The upstream DPS repo is sampling-only: `guided_diffusion` here ships no
`train_util`/`resample`/`script_util`, and `SpacedDiffusion.training_losses`
delegates to a `super().training_losses` that does not exist. This module adds
the minimum needed to train a prior that `create_model` can then load back.

Scope: epsilon-prediction with a plain MSE loss, i.e. `learn_sigma: False` plus
a fixed sampling variance. The learned-variance (hybrid VLB) objective is
deliberately not implemented -- it buys sample quality at the cost of a much
fussier loss, and is not worth it on a Colab-sized compute budget.
'''
import copy

import numpy as np
import torch

from .gaussian_diffusion import get_named_beta_schedule


class DiffusionTrainer:
    '''Forward-process bookkeeping and the training loss.

    Betas come from the repo's own `get_named_beta_schedule`, so the schedule
    used to train is guaranteed identical to the one `create_sampler` builds at
    sampling time.
    '''

    def __init__(self, noise_schedule: str = 'linear', steps: int = 1000,
                 device=None):
        device = device or torch.device('cpu')
        betas = np.array(get_named_beta_schedule(noise_schedule, steps),
                         dtype=np.float64)
        assert (betas > 0).all() and (betas <= 1).all(), 'betas must be in (0, 1]'

        alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
        self.num_timesteps = int(betas.shape[0])
        self.sqrt_alphas_cumprod = torch.tensor(
            np.sqrt(alphas_cumprod), dtype=torch.float32, device=device)
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32, device=device)

    def sample_timesteps(self, batch_size: int, device) -> torch.Tensor:
        # Uniform sampling. Importance sampling over t only pays off for the
        # VLB term, which we do not use.
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        '''x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps.

        `GaussianDiffusion.q_sample` draws its own noise and does not return it,
        so it cannot supply the regression target; hence this variant.
        '''
        shape = (-1,) + (1,) * (x_start.ndim - 1)
        coef1 = self.sqrt_alphas_cumprod[t].view(shape)
        coef2 = self.sqrt_one_minus_alphas_cumprod[t].view(shape)
        return coef1 * x_start + coef2 * noise

    def loss(self, model, x_start: torch.Tensor) -> torch.Tensor:
        t = self.sample_timesteps(x_start.shape[0], x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        eps_pred = model(x_t, t)

        if eps_pred.shape != x_start.shape:
            raise ValueError(
                f'Model returned {tuple(eps_pred.shape)} for input '
                f'{tuple(x_start.shape)}. This trainer implements the '
                'epsilon-MSE objective only -- set learn_sigma: False in the '
                'model config (and a fixed_* model_var_type when sampling).')

        return torch.nn.functional.mse_loss(eps_pred, noise)


class EMA:
    '''Exponential moving average of model weights, with decay warmup.

    Diffusion sample quality depends on this heavily; the raw weights at any
    given step are noticeably worse than their EMA. Always sample from `.ema`.

    Warmup matters more than it looks. A fixed decay of 0.9999 has a horizon of
    ~10k updates, so a shorter run leaves the average dominated by the random
    initialisation: after 300 steps 0.9999**300 = 97% of the init survives, and
    the exported checkpoint predicts at chance level at every timestep even
    though the raw model has clearly learned. Even a 17k-step run still carries
    18% init.

    So the effective decay ramps as `(1 + n) / (10 + n)`, capped at `decay`:
    the EMA tracks closely early and lengthens its horizon as training goes on,
    which makes an exported checkpoint usable at any point.
    '''

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999,
                 warmup: bool = True):
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.num_updates += 1
        d = self.current_decay()
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)
