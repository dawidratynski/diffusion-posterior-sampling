'''Inspect a UVCGAN checkpoint and prove the operator path works with it.

Run this the moment the weights arrive, BEFORE anything expensive. `load_uvcgan2`
was written against the uvcgan2 API without a real checkpoint to test on, so it
is the least-verified piece of the pipeline; the difference between finding that
out in ten seconds and finding it out three GPU-hours in is this script.

It reports what is actually in the directory, then walks the same path the
experiments take -- load both generators, check ranges, check the DPS gradient
survives -- and says exactly what to fix if a step fails.

    python scripts/probe_uvcgan.py --path /content/drive/MyDrive/.../uvcgan_model
'''
import argparse
import os

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import torch

SIZE = 160


def describe_directory(path):
    '''What is actually here? The loader assumes a uvcgan2 model directory.'''
    print(f'\n--- contents of {path} ---')
    if not os.path.exists(path):
        print('  DOES NOT EXIST')
        return False
    if os.path.isfile(path):
        print(f'  a FILE ({os.path.getsize(path) / 1e6:.1f} MB), not a directory.')
        print('  uvcgan2 saves a model DIRECTORY (weights plus the config needed')
        print('  to rebuild the architecture). If you were given a bare .pth,')
        print('  ask for the whole output directory, or use framework')
        print('  torchscript / cyclegan_resnet instead.')
        return False

    total = 0
    for root, _, files in os.walk(path):
        depth = root[len(path):].count(os.sep)
        if depth > 2:
            continue
        rel = os.path.relpath(root, path)
        print(f'  {rel}/' if rel != '.' else '  ./')
        for f in sorted(files)[:12]:
            size = os.path.getsize(os.path.join(root, f))
            total += size
            print(f'      {f:<44} {size / 1e6:>8.1f} MB')
        if len(files) > 12:
            print(f'      ... and {len(files) - 12} more')
    print(f'  total listed: {total / 1e6:.1f} MB')
    return True


def check_import():
    print('\n--- uvcgan2 importable? ---')
    for mod in ('uvcgan2.utils.funcs', 'uvcgan2.utils.eval'):
        try:
            __import__(mod)
            print(f'  OK: {mod}')
            return True
        except ImportError as e:
            print(f'  no: {mod} ({e})')
    print('  uvcgan2 is not installed. Install it from')
    print('  https://github.com/LS4GAN/uvcgan2 -- the loader needs its code to')
    print('  rebuild the generator architecture from the saved config.')
    return False


def check_load(path, direction, device):
    from guided_diffusion.measurements import get_operator
    print(f'\n--- load generator, direction={direction} ---')
    try:
        op = get_operator(name='cyclegan', device=device, framework='uvcgan2',
                          direction=direction, path=path)
    # Blind catches throughout this file are deliberate: the whole purpose is to
    # turn an unknown checkpoint's arbitrary failure into a readable diagnosis,
    # so narrowing them would defeat the script.
    except Exception as e:  # noqa: BLE001
        print(f'  FAILED: {type(e).__name__}: {e}')
        print('  If this is an AttributeError about gen_ab/gen_ba, the loaded')
        print('  object nests its generators differently in your uvcgan2')
        print('  version -- adjust load_uvcgan2() in')
        print('  guided_diffusion/cyclegan_loader.py to match the names printed')
        print('  in the error.')
        return None
    n = sum(p.numel() for p in op.generator.parameters())
    print(f'  OK: {type(op.generator).__name__}, {n / 1e6:.1f}M parameters')
    return op


def check_forward(op, direction):
    print(f'\n--- forward pass, direction={direction} ---')
    x = (torch.rand(1, 3, SIZE, SIZE) * 2 - 1).to(op.device)
    try:
        y = op.forward(x)
    except Exception as e:  # noqa: BLE001  -- diagnostic, see check_load
        print(f'  FAILED: {type(e).__name__}: {e}')
        return False
    print(f'  input  {tuple(x.shape)} in [{x.min():.2f}, {x.max():.2f}]')
    print(f'  output {tuple(y.shape)} in [{y.min():.2f}, {y.max():.2f}]')
    if y.shape != x.shape:
        print('  WARNING: shape changed. The prior and the operator must agree')
        print(f'  on resolution; the prior is {SIZE}px.')
        return False
    if y.min() < -1.5 or y.max() > 1.5:
        print('  WARNING: output is not in [-1, 1]. The whole pipeline uses that')
        print('  convention; a mismatch here is silent and ruins every result.')
        return False
    return True


def check_determinism(op):
    print('\n--- deterministic? ---')
    x = (torch.rand(1, 3, SIZE, SIZE) * 2 - 1).to(op.device)
    a, b = op.forward(x), op.forward(x)
    same = torch.allclose(a, b)
    print(f'  {"OK" if same else "NO -- two calls differ"}')
    if not same:
        print("  DPS's likelihood approximation assumes a deterministic A.")
        print('  Check the generator is in eval() mode and has no dropout.')
    return same


def check_gradient(op):
    '''The one that matters: DPS needs grad_x ||y - A(x)||.'''
    print('\n--- gradient reaches the input? ---')
    x = (torch.rand(1, 3, SIZE, SIZE) * 2 - 1).to(op.device).requires_grad_()
    y = (torch.rand(1, 3, SIZE, SIZE) * 2 - 1).to(op.device)
    try:
        norm = torch.linalg.norm(y - op.forward(x))
        (grad,) = torch.autograd.grad(norm, x)
    except Exception as e:  # noqa: BLE001  -- diagnostic, see check_load
        print(f'  FAILED: {type(e).__name__}: {e}')
        return False
    ok = torch.isfinite(grad).all() and grad.abs().sum() > 0
    print(f'  grad magnitude {grad.abs().mean():.3e}  finite={torch.isfinite(grad).all()}')
    if not ok:
        print('  ZERO OR NON-FINITE. DPS would silently degrade to unconditional')
        print('  sampling -- it still produces images, just ignoring the')
        print('  measurement. Check the generator is not wrapped in no_grad and')
        print('  that requires_grad_(False) was applied to PARAMETERS only.')
    else:
        print('  OK')
    return bool(ok)


def check_domain_statistics(op_ba, real_root, synth_root, limit=24):
    '''Does G_RS actually map real images into the synth domain?

    A generator can load, run and differentiate perfectly while mapping to the
    wrong place. Comparing output statistics against real synth images is the
    cheapest check that the direction convention (a=synth, b=real) matches ours.
    '''
    import glob

    import numpy as np
    from PIL import Image

    from util.lattice_metrics import lattice_params

    print('\n--- does G_RS land in the synth domain? ---')

    def load(root, n):
        paths = sorted(glob.glob(os.path.join(root, '*.png')))
        if not paths:
            print(f'  skipped: no PNGs in {root}')
            return None
        step = max(1, len(paths) // n)
        out = []
        for p in paths[::step][:n]:
            im = Image.open(p).convert('RGB')
            if im.size != (SIZE, SIZE):
                im = im.resize((SIZE, SIZE), Image.BILINEAR)
            out.append(np.asarray(im, dtype=np.float32) / 127.5 - 1)
        return out

    reals, synths = load(real_root, limit), load(synth_root, limit)
    if reals is None or synths is None:
        return

    def stats(imgs):
        return (float(np.median([a.mean() for a in imgs])),
                float(np.median([lattice_params(a)['prominence'] for a in imgs])))

    mapped = []
    for a in reals:
        t = torch.from_numpy(a).permute(2, 0, 1)[None].to(op_ba.device)
        mapped.append(op_ba.forward(t)[0].permute(1, 2, 0).detach().cpu().numpy())

    r_mean, r_prom = stats(reals)
    m_mean, m_prom = stats(mapped)
    s_mean, s_prom = stats(synths)
    print(f'  {"":<22}{"mean":>9}{"prominence":>13}')
    print(f'  {"real input":<22}{r_mean:>9.3f}{r_prom:>13.0f}')
    print(f'  {"G_RS(real)":<22}{m_mean:>9.3f}{m_prom:>13.0f}')
    print(f'  {"real synth images":<22}{s_mean:>9.3f}{s_prom:>13.0f}   <- target')
    close = abs(m_mean - s_mean) < 0.25
    print(f'  brightness {"matches" if close else "DOES NOT match"} the synth domain'
          + ('' if close else '  <-- check the direction convention: is domain '
                             "'a' synth in your uvcgan2 config?"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, required=True,
                   help='The uvcgan2 model directory.')
    p.add_argument('--real_root', type=str, default=None,
                   help='Optional: real PNGs, for the domain-statistics check.')
    p.add_argument('--synth_root', type=str, default=None,
                   help='Optional: synth PNGs, the target domain.')
    p.add_argument('--device', type=str, default='cpu',
                   help='cpu is fine and avoids competing for the GPU.')
    args = p.parse_args()
    device = torch.device(args.device)

    print('=' * 72)
    print('UVCGAN checkpoint probe')
    print('=' * 72)

    ok = describe_directory(args.path)
    if not ok:
        return 1
    if not check_import():
        return 1

    results = {}
    ops = {}
    for direction, what in (('ba', 'real->synth, the DPS operator'),
                            ('ab', 'synth->real, the baseline model')):
        print(f'\n{"=" * 72}\n{direction}: {what}\n{"=" * 72}')
        op = check_load(args.path, direction, device)
        ops[direction] = op
        if op is None:
            results[direction] = False
            continue
        results[direction] = (check_forward(op, direction)
                              and check_determinism(op)
                              and check_gradient(op))

    if ops.get('ba') and args.real_root and args.synth_root:
        check_domain_statistics(ops['ba'], args.real_root, args.synth_root)

    print(f'\n{"=" * 72}')
    for d, passed in results.items():
        print(f'  direction {d}: {"PASS" if passed else "FAIL"}')
    if all(results.values()):
        print('\nThe operator path works. The experiment notebook can run')
        print('unchanged with --uvcgan_path pointing here.')
        return 0
    print('\nFix the failures above before spending GPU time.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
