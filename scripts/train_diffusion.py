'''Train the unconditional diffusion prior used by DPS.

Train on the REAL domain. The project generates realistic measurements from
synthetic inputs, so the distribution being sampled is the real one; DPS solves
y = A(x) with A = G_{R->S}, x a realistic image and y a known synthetic image.

    python scripts/train_diffusion.py \
        --model_config configs/crystal_model_config.yaml \
        --data_root /path/to/dataset/real/train \
        --out_dir ./models/crystal_real

Caveat worth keeping in view: real data is the scarce side (~15k crops from
~2957 source images, so far fewer independent samples than the count suggests).
Watch for memorisation -- a prior that reproduces training images would make the
generated data useless as augmentation. Keep --no_augment off, prefer a smaller
model over a larger one, and run a nearest-neighbour check against the training
set before trusting samples.

Writes `ckpt_latest.pt` (full training state, for --resume) and
`model_ema_XXXXXX.pt` (a bare state_dict that `create_model(model_path=...)`
loads directly). Colab will disconnect, so --resume is the normal way to run.
'''
import argparse
import os
import time
from glob import glob

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataloader import get_dataset
from guided_diffusion.train_util import EMA, DiffusionTrainer, ValidationProbe
from guided_diffusion.unet import create_model
from util.logger import get_logger


def load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', type=str, required=True)
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--dataset', type=str, default='crystal')
    p.add_argument('--noise_schedule', type=str, default='linear')
    p.add_argument('--diffusion_steps', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--train_steps', type=int, default=200_000)
    p.add_argument('--ema_rate', type=float, default=0.9999)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--save_every', type=int, default=5_000)
    # Colab gives 2 vCPUs; a hardcoded 4 oversubscribes and torch warns about it.
    p.add_argument('--num_workers', type=int,
                   default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    p.add_argument('--device', type=str, default=None,
                   help="Defaults to cuda when available, else cpu.")
    p.add_argument('--amp', action='store_true',
                   help='Mixed precision. CUDA only; ignored on CPU.')
    p.add_argument('--resume', action='store_true',
                   help='Continue from ckpt_latest.pt in --out_dir if present.')
    p.add_argument('--init_from', type=str, default=None,
                   help='Bare state_dict (model_ema_*.pt) to initialise from, '
                        'for synth-pretrain -> real-finetune. Unlike --resume '
                        'this starts the step counter and optimiser fresh.')
    p.add_argument('--save_fp16', action='store_true',
                   help='Export model_ema_*.pt in half precision: exact same '
                        'file count at half the size, and load_state_dict casts '
                        'back to fp32 so create_model is unaffected. Measured '
                        'output deviation ~0.1%%. ckpt_latest.pt stays fp32 so '
                        'resume is bit-exact.')
    p.add_argument('--keep_last', type=int, default=0,
                   help='Keep only the N most recent model_ema_*.pt (0 = all). '
                        'Each is 4 bytes/param -- 330 MB at 82M params -- so a '
                        '40k-step run saving every 2000 writes 6.6 GB. Use a '
                        'small N for a pretrain whose intermediates you do not '
                        'need; keep all for a finetune you must sweep over.')
    p.add_argument('--val_root', type=str, default=None,
                   help='Held-out images for the per-timestep validation probe. '
                        'The running training loss plateaus long before quality '
                        'does, so without this there is no stopping signal.')
    p.add_argument('--val_every', type=int, default=1000)
    p.add_argument('--val_batch', type=int, default=16)
    p.add_argument('--no_augment', action='store_true')
    return p.parse_args()


def infinite(loader):
    while True:
        yield from loader


def main():
    args = parse_args()
    logger = get_logger()

    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f'Device: {device}')

    os.makedirs(args.out_dir, exist_ok=True)

    model_config = load_yaml(args.model_config)
    if model_config.get('learn_sigma', False):
        raise ValueError(
            'learn_sigma: True needs the hybrid VLB objective, which this '
            'trainer does not implement. Set learn_sigma: False here and use a '
            'fixed_small/fixed_large model_var_type in the diffusion config.')
    image_size = model_config['image_size']

    # model_path in the config points at the *output* of training; strip it so a
    # fresh run starts from random weights instead of trying to load itself.
    model_config = {k: v for k, v in model_config.items() if k != 'model_path'}
    if args.init_from:
        # Pretrain on synth (unlimited, so no memorisation risk) then finetune
        # on real. The real set is the scarce side, and cutting the number of
        # real-data gradient steps needed is the main lever on memorisation.
        model_config['model_path'] = args.init_from
    model = create_model(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model: {n_params / 1e6:.1f}M parameters at {image_size}px')
    if args.init_from:
        logger.info(f'Initialised from {args.init_from}')

    dataset = get_dataset(name=args.dataset, root=args.data_root,
                          image_size=image_size, augment=not args.no_augment)
    logger.info(f'Dataset: {len(dataset)} images from {args.data_root}')
    if len(dataset) < args.batch_size:
        # drop_last=True would make the loader yield nothing, and infinite()
        # would then spin forever at 100% CPU with no output and no error.
        raise ValueError(
            f'Only {len(dataset)} images but --batch_size {args.batch_size}; '
            'with drop_last the loader yields no batches and training hangs.')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        pin_memory=(device.type == 'cuda'))

    trainer = DiffusionTrainer(args.noise_schedule, args.diffusion_steps, device)

    probe = None
    if args.val_root:
        val_ds = get_dataset(name=args.dataset, root=args.val_root,
                             image_size=image_size, augment=False)
        # Strided, not the first N: crops are named "<photo>_sample_<k>", so
        # truncating would evaluate several crops of one or two photos.
        stride = len(val_ds) / min(args.val_batch, len(val_ds))
        idx = [int(i * stride) for i in range(min(args.val_batch, len(val_ds)))]
        val_x = torch.stack([val_ds[i] for i in idx]).to(device)
        probe = ValidationProbe(trainer, val_x)
        logger.info(f'Validation probe: {len(idx)} held-out images from '
                    f'{args.val_root}, timesteps {probe.timesteps}')
    ema = EMA(model, args.ema_rate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if args.amp and not use_amp:
        logger.info('--amp ignored: not on CUDA.')

    start_step = 0
    ckpt_path = os.path.join(args.out_dir, 'ckpt_latest.pt')
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        scaler.load_state_dict(ckpt['scaler'])
        start_step = ckpt['step']
        # Restore the EMA warmup counter, else resuming resets the decay ramp
        # and the average snaps back toward the current weights.
        ema.num_updates = ckpt.get('ema_updates', start_step)
        logger.info(f'Resumed from {ckpt_path} at step {start_step}')
    elif args.resume:
        logger.info(f'No checkpoint at {ckpt_path}; starting fresh.')

    def save(step):
        torch.save({'model': model.state_dict(), 'ema': ema.state_dict(),
                    'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'step': step, 'ema_updates': ema.num_updates}, ckpt_path)
        # Bare state_dict of the EMA weights: this is what you point
        # model_path at for sampling.
        ema_sd = ema.state_dict()
        if args.save_fp16:
            ema_sd = {k: (v.half() if v.is_floating_point() else v)
                      for k, v in ema_sd.items()}
        torch.save(ema_sd,
                   os.path.join(args.out_dir, f'model_ema_{step:06d}.pt'))
        if args.keep_last > 0:
            stale = sorted(glob(os.path.join(args.out_dir, 'model_ema_*.pt')))
            for old in stale[:-args.keep_last]:
                os.remove(old)
        logger.info(f'Saved checkpoint at step {step}')

    def run_probe(step):
        # On the EMA weights, which are what gets exported and sampled from.
        losses = probe(ema.ema)
        parts = '  '.join(f't={t}: {v:.4f}' for t, v in losses.items())
        logger.info(f'  val @ step {step}   {parts}')

    model.train()
    data = infinite(loader)
    running, t0 = 0.0, time.time()
    if probe is not None:
        run_probe(start_step)   # baseline to compare later probes against

    for step in range(start_step, args.train_steps):
        x = next(data).to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            loss = trainer.loss(model, x)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        ema.update(model)

        running += loss.item()
        if (step + 1) % args.log_every == 0:
            rate = args.log_every / (time.time() - t0)
            logger.info(f'step {step + 1}/{args.train_steps}  '
                        f'loss {running / args.log_every:.4f}  '
                        f'ema_decay {ema.current_decay():.5f}  '
                        f'{rate:.2f} it/s')
            running, t0 = 0.0, time.time()

        if probe is not None and (step + 1) % args.val_every == 0:
            run_probe(step + 1)
            t0 = time.time()   # do not bill probe time to the it/s estimate

        if (step + 1) % args.save_every == 0:
            save(step + 1)

    if args.train_steps % args.save_every != 0:
        save(args.train_steps)
    logger.info('Training finished.')


if __name__ == '__main__':
    main()
