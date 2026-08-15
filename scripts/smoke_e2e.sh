#!/usr/bin/env bash
# Phase 1A: end-to-end rehearsal on the real dataset, small enough to finish on
# CPU in well under ten minutes.
#
# Trains a prior, runs DPS through the spectral stand-in operator, and evaluates
# -- the same four scripts and the same configs as the real run, only smaller.
# The point is to surface wiring, config and format errors before committing to
# an overnight run or GPU time. The OUTPUT QUALITY IS MEANINGLESS at this scale.
#
#   ./scripts/smoke_e2e.sh [DATASET_DIR] [WORK_DIR]
#
set -euo pipefail

DATASET="${1:-../dataset}"
WORK="${2:-/tmp/dps_smoke}"
PY="${PY:-.venv/bin/python}"

IMG=64          # small enough that the run is minutes, big enough for the
                # lattice (period 25-30px at 150px -> ~11-13px here) to survive
STEPS=150
N_IMAGES=4
VARIANTS=2
RESPACING=30

echo "=== Phase 1A smoke: dataset=$DATASET work=$WORK ==="
rm -rf "$WORK"; mkdir -p "$WORK"

for d in real/train real/val; do
    [ -d "$DATASET/$d" ] || { echo "missing $DATASET/$d"; exit 1; }
done

# --- configs: same shape as the real ones, smaller ---------------------------
cat > "$WORK/model.yaml" <<EOF
image_size: $IMG
num_channels: 64
num_res_blocks: 1
channel_mult: "1,2,2"
learn_sigma: False
class_cond: False
use_checkpoint: False
attention_resolutions: "16"
num_heads: 1
num_head_channels: -1
num_heads_upsample: -1
use_scale_shift_norm: True
dropout: 0.0
resblock_updown: True
use_fp16: False
use_new_attention_order: False
EOF

sed "s/^timestep_respacing:.*/timestep_respacing: $RESPACING/" \
    configs/crystal_diffusion_config.yaml > "$WORK/diffusion.yaml"

# Real operator stand-in: analytic, untrained, differentiable.
sed -e 's/^    framework: stub/    framework: spectral/' \
    -e "s|^  root:.*|  root: $DATASET/real/val|" \
    configs/crystal_cyclegan_config.yaml > "$WORK/task.yaml"

# --- 1. train the prior on REAL ---------------------------------------------
echo; echo "--- [1/5] training prior on real (${STEPS} steps @ ${IMG}px) ---"
$PY scripts/train_diffusion.py \
    --model_config "$WORK/model.yaml" \
    --data_root "$DATASET/real/train" \
    --out_dir "$WORK/prior" \
    --batch_size 8 --train_steps $STEPS \
    --log_every 100 --save_every $STEPS --num_workers 2

cp "$WORK/model.yaml" "$WORK/model_loaded.yaml"
printf 'model_path: %s\n' "$WORK/prior/model_ema_$(printf '%06d' $STEPS).pt" \
    >> "$WORK/model_loaded.yaml"

# --- 2. sanity: did the EXPORTED checkpoint actually learn? ------------------
# Guards a bug that cost a whole smoke run: with a fixed EMA decay of 0.9999 the
# exported model_ema_*.pt was 97% random initialisation after 300 steps and
# predicted at chance level, while the training log showed a healthy loss.
echo; echo "--- [2/5] verifying exported checkpoint beats chance ---"
$PY - "$WORK" "$DATASET/real/val" $IMG $STEPS <<'PYEOF'
import sys, torch, yaml
sys.path.insert(0, '.')
from guided_diffusion.unet import create_model
from guided_diffusion.train_util import DiffusionTrainer
from data.dataloader import get_dataset

work, val_root, img_size, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cfg = yaml.safe_load(open(f'{work}/model_loaded.yaml'))
model = create_model(**cfg).eval()
tr = DiffusionTrainer('linear', 1000, torch.device('cpu'))
ds = get_dataset(name='crystal', root=val_root, image_size=img_size, augment=False)
x = torch.stack([ds[i] for i in range(8)])

torch.manual_seed(0)
worst = 0.0
for t in (50, 250, 500, 999):
    tt = torch.full((8,), t)
    noise = torch.randn_like(x)
    with torch.no_grad():
        loss = torch.nn.functional.mse_loss(model(tr.q_sample(x, tt, noise), tt), noise)
    print(f'  t={t:>4}  eps-MSE {loss.item():.4f}')
    worst = max(worst, loss.item())

# Predicting nothing gives ~1.0. Even a barely-trained model must beat that.
assert worst < 0.9, (
    f'Exported checkpoint predicts at chance (worst eps-MSE {worst:.4f}). '
    'The EMA is probably dominated by the random init -- check EMA warmup.')
print(f'  OK: worst-case {worst:.4f} < 0.9 (chance is ~1.0)')
PYEOF

# --- 3. DPS in validate mode -------------------------------------------------
# y = A(real image); the generated image should recover the reference. Self
# consistent for the untrained stand-in, unlike generate mode.
echo; echo "--- [3/5] DPS (validate mode, spectral operator) ---"
$PY scripts/generate_augmented.py \
    --model_config "$WORK/model_loaded.yaml" \
    --diffusion_config "$WORK/diffusion.yaml" \
    --task_config "$WORK/task.yaml" \
    --mode validate --reference_root "$DATASET/real/val" \
    --out_dir "$WORK/results/dps" --method dps \
    --samples_per_input $VARIANTS --limit $N_IMAGES 2>&1 \
    | { grep -vE '^[[:space:]]*[0-9]+%|it/s\]$' || true; }

# --- 4. identity control -------------------------------------------------
# The reference images themselves, scored as if they were a method. Any real
# method should sit between this ceiling and pure noise; if DPS cannot beat it
# on realism something is wired wrong.
echo; echo "--- [4/5] identity control ---"
$PY - "$DATASET/real/val" "$WORK/results/control" $N_IMAGES <<'PYEOF'
import csv, os, shutil, sys
from glob import glob
src_root, out_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(os.path.join(out_dir, 'generated'), exist_ok=True)
paths = sorted(glob(os.path.join(src_root, '**', '*.png'), recursive=True))[:n]
with open(os.path.join(out_dir, 'manifest.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['output_image', 'source_image', 'method', 'sample_index', 'seed'])
    for i, p in enumerate(paths):
        name = f'{i:05d}_00.png'
        shutil.copy(p, os.path.join(out_dir, 'generated', name))
        w.writerow([name, p, 'identity', 0, 0])
print(f'wrote {len(paths)} control images')
PYEOF

# --- 5. evaluate ---------------------------------------------------------
echo; echo "--- [5/5] evaluate ---"
$PY scripts/evaluate.py \
    --result_dir "$WORK/results/dps" \
    --result_dir "$WORK/results/control" \
    --real_root "$DATASET/real/val" \
    --real_train_root "$DATASET/real/train" \
    --limit_reference 150 \
    --out_csv "$WORK/results/per_image.csv"

echo
echo "=== smoke passed; artefacts in $WORK ==="
echo "Quality is meaningless at ${STEPS} steps. What this proves is that train ->"
echo "generate -> evaluate run together on real data with real configs."
