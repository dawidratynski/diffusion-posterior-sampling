# Methods reference

Everything non-obvious in the pipeline, with formulas and the reasoning behind
the choices, so the report can be written without re-deriving it from the code.

Written for: whoever writes the thesis (i.e. you) — assumes familiarity with the
project but not with the implementation details.

---

## 1. Setup and notation

Two image domains, both 160×160 grayscale stored as RGB, pixel values in
$[-1,1]$:

| symbol | domain | description |
|---|---|---|
| $R$ | real | electron-microscope photographs. A lattice buried in noise, often not filling the frame, with breaks and overlapping structure. Scarce: ~2.9k source photos. |
| $S$ | synthetic | simulated images of how the structure *should* look. Clean, bright, near-perfectly periodic, effectively unlimited. |

Crops are 150×150 px covering 300 nm, resized to 160×160 to match UVCGAN2's
`shape: (3, 160, 160)`.

Two translation maps, with UVCGAN2's convention that domain `a` = synth:

$$G_{S\to R} = \texttt{gen\_ab}, \qquad G_{R\to S} = \texttt{gen\_ba}$$

### The goal

A downstream project lacks real data. Real photographs are prohibitively
expensive; synthetic ones are free and come with **known lattice parameters**.
If we can map $S \to R$ convincingly, we get cheap labelled data: the generated
image looks real, and its parameters are inherited from the synthetic input.

The thesis question is whether the $S\to R$ map is better done **directly** (a
trained translation model) or as **posterior sampling** with a diffusion prior on
$R$ conditioned through $G_{R\to S}$.

### The three models compared

| label | definition | stochastic |
|---|---|---|
| `uvcgan` | $x = G_{S\to R}(s)$ | no |
| `dps_uvcgan` | $x \sim p_R(x)\, p(s \mid G_{R\to S}(x))$, trained operator | yes |
| `dps_analytic` | as above with the analytic operator $A$ of §2 | yes |

Note the direction inversion: DPS's operator is $R\to S$ even though the goal is
$S\to R$, because **DPS inverts its operator**. The diffusion prior therefore
lives on $R$ — the scarce domain — which is what makes the memorisation check
(§5.6) necessary.

---

## 2. The analytic operator

`SpectralIdealizer` in `guided_diffusion/cyclegan_loader.py`.

### 2.1 Why it exists

Two reasons, and the second is the one that matters for the report:

1. **Practical.** It let the entire pipeline be built and validated before any
   trained generator existed.
2. **Scientific.** It is the *"no trained translation model"* ablation. If
   `dps_uvcgan` beats `dps_analytic`, the gain is attributable to having learned
   the $R\to S$ mapping rather than to DPS itself.

### 2.2 Definition

For an image $x \in [-1,1]^{C\times H\times W}$, write $\mathcal{F}$ for the 2-D
DFT and $F = \mathcal{F}(x)$, with frequencies centred (fftshift) so that the
DC component sits at the array centre. Let $r(u,v)=\sqrt{u^2+v^2}$ be the radius
in cycles-per-image of frequency bin $(u,v)$, and $n=\min(H,W)$.

**Step 1 — peak sharpening.** With $M = |F|$ and $M_{\max}$ its maximum over
spatial frequencies (per image and channel),

$$W_{\text{sharp}}(u,v) \;=\; \left(\frac{M(u,v)}{M_{\max}}\right)^{\gamma}$$

**Step 2 — band-pass.** With $\rho_{\min}, \rho_{\max}$ the smallest and largest
lattice periods to retain (defaults 4 px and 37.5 px) and $\sigma$ a softness
parameter (default 1.5):

$$W_{\text{band}}(u,v) \;=\; \mathrm{sigmoid}\!\left(\frac{r - n/\rho_{\max}}{\sigma}\right) \cdot \mathrm{sigmoid}\!\left(\frac{n/\rho_{\min} - r}{\sigma}\right)$$

**Step 3 — inverse transform.**

$$\tilde{x} \;=\; \Re\,\mathcal{F}^{-1}\!\big(F \cdot W_{\text{sharp}} \cdot W_{\text{band}}\big)$$

**Step 4 — domain renormalisation.** With $\mu,\sigma_x$ the mean and standard
deviation of $\tilde{x}$ over all channels and pixels, and $(\mu_S,\sigma_S)$ the
measured synth-domain statistics:

$$A(x) \;=\; \mathrm{clip}_{[-1,1]}\!\left(\frac{\tilde{x}-\mu}{\sigma_x}\,\sigma_S + \mu_S\right)$$

Defaults $\mu_S = 0.47$, $\sigma_S = 0.38$, from the synth split: mean pixel
value $188.2/255$ and std $49.1/255$, mapped to $[-1,1]$.

### 2.3 Why each step

**Peak sharpening** is the core idea. A lattice is coherent, so its energy
concentrates at a few reciprocal-lattice points; noise and aperiodic clutter
spread energy across all frequencies. Raising the normalised magnitude to a
power $\gamma>0$ therefore *suppresses the incoherent floor relative to the
coherent peaks* — it idealises. $\gamma=0$ is the identity (band-pass only);
larger $\gamma$ drives the output toward a pure sum of a few sinusoids.

Crucially this is a **smooth, data-dependent** weighting, not a hard top-$k$
mask. A discrete mask would make $A$ piecewise constant in $x$, giving
discontinuous jumps in the DPS gradient. Smoothness here is a requirement, not
an aesthetic choice (see §4).

**The band-pass** removes two things. Below $n/\rho_{\max}$ lie DC and slow
illumination gradients, which carry **42% of total power** in real micrographs
(measured: radius ≤3 of a 150-px FFT) and would otherwise dominate the
sharpening step's $M_{\max}$. Above $n/\rho_{\min}$ lies near-Nyquist pixel
noise. Sigmoid rather than brick-wall edges, because a sharp cutoff causes
ringing (Gibbs) and, again, a non-smooth gradient.

**Renormalisation** is needed because the two domains have very different
intensity statistics — real images are dark (mean $-0.52$ in $[-1,1]$), synth
bright (mean $+0.47$). Without it, $A$'s output would not resemble $S$ at all.

### 2.4 Calibration of $\gamma$

$\gamma$ was set so that the operator's **output** has the same peak prominence
(§5.1) as genuine synth images — i.e. so the analytic operator idealises by
about as much as the real $G_{R\to S}$ must. Measured on the real validation
split:

| $\gamma$ | 0.0 | 0.15 | 0.25 | **0.4** | 0.6 | 1.0 |
|---|---|---|---|---|---|---|
| output prominence | 987 | 2 433 | 4 134 | **9 496** | 31 960 | 243 547 |
| vs synth (12 197) | 0.1× | 0.2× | 0.3× | **0.8×** | 2.6× | 20× |
| lattice spacing preserved | .000 | .000 | .000 | **.000** | .010 | .023 |

$\gamma = 0.4$ lands closest. The initially chosen $\gamma=1.0$ overshot the
target domain by 20× and, worse, began to *move the lattice itself* (2.3% median
spacing error introduced by the operator), corrupting the very quantity the
label-validity metric measures.

### 2.5 Properties worth stating

- **Deterministic.** DPS's likelihood approximation assumes a deterministic $A$.
- **Differentiable.** Required; see §4.
- **Brightness lies in its null space.** Step 2 removes DC and step 4
  renormalises, so *any* input mean maps to the same output mean (verified:
  inputs ranging over mean $-0.92$ to $+0.11$ all map to output mean $\approx
  +0.44$). Consequently the data-consistency term exerts **zero gradient** on
  output brightness, which is supplied entirely by the prior. Observed
  decorrelation between input and output brightness is therefore expected
  behaviour, not a defect — and for augmentation it is desirable, since
  brightness is nuisance variation that should be sampled rather than copied.
- **It is not a faithful model of $G_{R\to S}$.** Its output is sinusoidal;
  true synth images are blob-like with sharp edges. It reproduces the *global
  periodicity* of the synth domain but not its local morphology. This is the
  limitation the `dps_analytic` ablation is designed to expose.

---

## 3. The diffusion prior

### 3.1 Objective

Standard DDPM with $\epsilon$-prediction. Forward process:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon\sim\mathcal{N}(0,I)$$

with a linear $\beta$ schedule over $T=1000$ steps. Training loss:

$$\mathcal{L} = \mathbb{E}_{t\sim\mathcal{U}\{1..T\},\,x_0,\,\epsilon}\big\|\epsilon_\theta(x_t,t)-\epsilon\big\|_2^2$$

**Learned variance (hybrid VLB) is deliberately not implemented.** It buys some
sample quality at the cost of a much fussier objective, and is not worth it at
this compute budget. Consequently `learn_sigma: False` and sampling uses a fixed
variance.

Architecture: the OpenAI guided-diffusion UNet, 82.4M parameters at 160 px.

### 3.2 Training schedule

Two stages, motivated by the data asymmetry — $S$ is unlimited, $R$ is scarce:

1. **Pretrain on synth**, 40k steps, lr $10^{-4}$, batch 16. No memorisation
   risk, and it teaches lattice structure and low-level image statistics.
2. **Finetune on real**, 10k steps, lr $2\times10^{-5}$. This is where
   memorisation could occur, so it is kept short and monitored (§5.6).

The finetune converged within ~1000 steps and a learning-rate sweep over
$\{5\times10^{-5}, 10^{-4}, 3\times10^{-4}\}$ moved the validation loss by
**<1%**, with $10^{-3}$ actively unstable. So the plateau is a property of the
data, not a tuning failure: at low $t$ the model must predict $\epsilon$ from an
almost-clean image, and real micrographs contain genuinely unpredictable
high-frequency texture. The synth probe floors around 0.056 and the real one
around 0.100 for exactly this reason.

### 3.3 EMA with decay warmup

Sampling uses an exponential moving average of the weights. The decay ramps:

$$d_n = \min\!\left(d_{\max},\ \frac{1+n}{10+n}\right)$$

A fixed $d_{\max}=0.9999$ has a horizon of ~10k updates, so a shorter run leaves
the average dominated by the random initialisation — after 300 steps, 97% of the
init survives and the exported checkpoint predicts at chance *even though the raw
model has clearly learned*. This bug is silent: the training log looks healthy.
The warmup makes an exported checkpoint usable at any point.

### 3.4 Why the training loss is not a progress signal

$\mathcal{L}$ averages over $t\sim\mathcal{U}\{1..T\}$. At high $t$, $x_t$ is
nearly pure noise, so predicting $\epsilon$ is close to returning the input —
trivially easy, and saturating within a few hundred updates. Those terms dominate
the average, which therefore plateaus long before the model stops improving.

Measured: training loss flat at ~0.008 from step 5 000 onward, while the
low-$t$ validation loss fell a further 11% over the same span.

The **validation probe** (`ValidationProbe`) fixes the batch, the noise draw and
the timesteps, so successive evaluations differ only by the model. It reports
$t \in \{25, 100, 400, 900\}$ separately. Read $t=25$; ignore $t=900$. At
initialisation all timesteps read $\approx 1.0$ (chance, since
$\epsilon\sim\mathcal{N}(0,I)$), which doubles as a correctness check.

---

## 4. Diffusion Posterior Sampling

At each reverse step, after the unconditional update, DPS applies a
data-consistency correction (Chung et al., ICLR 2023):

$$x_{t-1} \leftarrow x_{t-1}' - \zeta\,\nabla_{x_t}\big\|y - A(\hat{x}_0(x_t))\big\|_2$$

where $\hat{x}_0(x_t)$ is the posterior mean estimate of the clean image and
$\zeta$ is the conditioning **scale**.

Three consequences worth stating in the report:

**The operator must stay inside the autograd graph.** Its parameters are frozen
with `requires_grad_(False)`, but the forward pass is *not* run under
`no_grad()`. Running it under `no_grad()` zeroes the guidance and silently
degrades DPS to unconditional sampling — it still produces plausible images, just
ignoring the measurement entirely.

**$\zeta$ is schedule-dependent.** One guidance step is applied per timestep, so
total guidance scales with the number of sampling steps. A value tuned at 100
steps over-guides at 250 or 1000. Sweep and final runs therefore share one
schedule (250 steps). This also explains why the DPS paper's $\zeta=0.3$ at 1000
steps is consistent with larger values here.

**$\zeta$ is operator-dependent.** The two DPS variants invert different
operators and are tuned separately.

**It is tuned on the round trip and applied to both experiments.** The two
experiments present different measurement distributions — $G_{R\to S}(\text{real})$
versus a genuine synthetic image — so the optimum need not be identical. It is
tuned once, on the round trip, because that is where paired metrics exist to
judge it; sweeping both would double the sweep cost for a second-order effect.
Worth stating as an assumption rather than leaving implicit.

---

## 5. Metrics

Two classes, applied where each is meaningful:

- **Paired** metrics need a per-output ground truth, so they apply **only** to
  the round-trip experiment. In the $S\to R$ direction there is no single correct
  realistic image for a given synthetic input, so scoring against one would
  penalise a method for producing a valid sample.
- **Distributional and structural** metrics apply to both.

All images are resized to a common 160 px before measurement. This matters: the
lattice metrics are measured **in pixels**, so comparing a 160 px output against
a 150 px source injects a flat $160/150-1 = 6.7\%$ spacing error into *identical
content* — larger than the threshold that is supposed to indicate real drift.

### 5.0 Pixel convention — why generated PNGs are comparable to source PNGs

Everything in the pipeline works in $[-1,1]$. `CrystalDataset` maps a PNG through
$[0,255] \to [0,1] \to [-1,1]$, and generated images are written back with the
**exact inverse**, $(x+1)/2$ clipped to $[0,1]$ (`util.img_utils.to_display`).
A generated PNG and a dataset PNG therefore sit on the same scale and can be
compared directly.

This is worth stating because the obvious alternative is wrong. The upstream
display helper `clear_color` min-max normalises **per image**, rescaling every
output to span the full range. That would:

- erase brightness and contrast differences between outputs, understating
  diversity;
- put generated and source images on different scales, so PSNR/SSIM/LPIPS would
  measure a stretch the model never applied;
- mean the saved figures are not what the model produced.

Any number quoted from a run whose images were saved that way — including the
diversity reference points in §5.5 — is not comparable to one saved with the
fixed map. Measured round trip through the current convention: PSNR 54.3 dB,
SSIM 0.99998, the residual being 8-bit PNG quantisation alone.

### 5.1 Lattice metrics — the decisive ones

These exist because generic image metrics cannot see the failure that matters
most. A method that adds convincing imperfections **while shifting the lattice
spacing** scores well on PSNR, FID and everything else, while producing
systematically **mislabelled** training pairs — worse than no data at all.

**Power spectrum.** For grayscale $g$ (mean-subtracted),

$$P(u,v) = \big|\mathcal{F}\{\,g \cdot w_{\text{Hann}}\,\}(u,v)\big|^2$$

The **Hann window** $w_{\text{Hann}}(i,j) = \text{hann}(i)\,\text{hann}(j)$ is
essential: these images are crops, so the FFT's implicit periodic tiling creates
hard edge discontinuities whose spectral leakage forms a cross through the
origin — easily mistaken for lattice structure.

**Zero padding.** The transform is computed on the image zero-padded by a factor
of 4. The lattice sits at radius $\approx 5$ in a 150 px FFT, where integer bins
quantise the recoverable spacing into ~17% steps ($150/5=30$ vs $150/6=25$).
Padding interpolates the spectrum onto a finer grid. Measured effect:
crop-to-crop spacing spread falls from 8.4% to 7.6% and then saturates, so
padding 4 is used and the residual spread is physical variation between crops,
not measurement error.

**Dominant peak.** Search the annulus $n/\rho_{\max} \le r \le n/\rho_{\min}$
and take the maximum. The rationale for the bounds is the same as §2.2 (exclude
the illumination blob below, near-Nyquist noise above), but the *values* differ
slightly: the metric defaults to $\rho_{\min}=4$ px and $\rho_{\max}=n/4$
(= 40 px at $n=160$), while the operator uses 4 px and 37.5 px. The two were
calibrated independently and neither is sensitive to the difference; they are
not required to match, since one shapes an image and the other measures one. From its position
$(\Delta u, \Delta v)$ relative to the centre, with $r=\sqrt{\Delta u^2+\Delta v^2}$:

$$\text{spacing} = \frac{n}{r}, \qquad \text{angle} = \arctan2(\Delta v, \Delta u) \bmod 180°$$

Angle is taken mod 180° because a lattice direction is undirected and the
spectrum is centrosymmetric.

$$\text{prominence} = \frac{P(\text{peak})}{\mathrm{median}\,\{P(u,v) : (u,v)\in\text{annulus}\}}$$

**Prominence is a sharpness measure, and is not "lower better".** Perfect
synthetic lattices give tall narrow peaks; real measurements, broadened by
defects and overlapping structure, give lower ones. Generated output should land
*near* the real value — below means over-degraded, above means too clean.

Note that a high-prominence image looks *smooth* to the eye (energy concentrated
in few frequencies ⇒ close to a pure sinusoid), while real images look crisper
despite lower prominence. The two facts are consistent; they are not in tension.

**Multiple peaks.** These are 2-D lattices with several reciprocal orders, not
1-D fringes. A method could preserve one lattice vector while distorting another
and score a perfect single-peak error. `top_k_peaks` therefore extracts the $k$
strongest distinct peaks by greedy non-maximum suppression (each accepted peak
and its centrosymmetric mirror are excluded within a radius proportional to their
own), and

$$\text{signature error} = \frac{1}{k}\sum_{i} \frac{\big|\,\text{spacing}(\hat{p}_i) - \text{spacing}(p_i)\,\big|}{\text{spacing}(p_i)}$$

after matching generated peaks $\hat{p}$ to source peaks $p$ greedily by
**distance in reciprocal space**, i.e. minimising
$\|\mathbf{v}(\hat{p}) - \mathbf{v}(p)\| / \|\mathbf{v}(p)\|$ where
$\mathbf{v}(p) = r_p(\cos\theta_p, \sin\theta_p)$.

Matching by angle alone would be ambiguous: a single lattice direction
contributes a fundamental *and its harmonics at the same angle*. A square-wave
profile at 20° yields peaks at 24.1 px, 8.0 px and 4.8 px, all at 20°, so the
pairing would depend on list order rather than on geometry. Reciprocal-space
distance separates harmonics by radius while still keeping different directions
apart, and does not mask drift — a fundamental displaced even 50% remains far
closer to the source fundamental than to its own third harmonic.

**Errors.**

$$\varepsilon_{\text{spacing}} = \frac{|\text{spacing}_{\text{gen}} - \text{spacing}_{\text{src}}|}{\text{spacing}_{\text{src}}}, \qquad \Delta\theta = \min(|\theta_1-\theta_2| \bmod 180°,\ 180° - \cdot)$$

**Noise floor.** Validated by applying realistic degradations to synthetic images
whose lattice is known to be unchanged: additive noise at real-matched contrast,
brightness and contrast shifts, and occluding 30% of the frame with unrelated
texture. Median spacing error **0.0000** in every case, p90 ≤ 0.024. So the
metric is essentially exact, and errors above ~5% indicate genuine drift.

**Stratification by source lattice strength.** Reported separately for sources
above and below the median source prominence. Where the reference is so noisy
that the lattice is barely detectable by eye, the "true" spacing is itself
uncertain, so a large error there measures the *metric's* limits rather than the
method's. Every large error observed in practice traced to such references.

### 5.2 Spectral realism

**Radial profile.** $\bar{P}(r)$ = mean of $P$ over the annulus at integer radius
$r$; a rotation-invariant summary of how power distributes across scales.

**Profile distance.** With each profile normalised to unit sum (so the metric
compares *shape*, not overall contrast, which differs trivially between domains):

$$D(\bar{P}_1,\bar{P}_2) = \sum_r \left| \frac{\bar{P}_1(r)}{\sum \bar{P}_1} - \frac{\bar{P}_2(r)}{\sum \bar{P}_2} \right|$$

**Calibration.** The number is meaningless without its floor and ceiling:

| | value |
|---|---|
| real vs real, $n=200$ (floor) | 0.0522 ± 0.0138 |
| real vs real, $n=48$ (floor) | 0.1062 ± 0.0564 |
| synth vs real (ceiling, wrong domain) | 1.1962 |

Report results as a position between these, and note that differences smaller
than the floor's standard deviation at your $n$ are not significant.

### 5.3 Paired reconstruction metrics — round trip only

$$\text{PSNR} = 10\log_{10}\frac{\text{MAX}^2}{\text{MSE}}$$

Pixel-wise, so it partly asks the wrong question of a stochastic method: DPS is
meant to produce a plausible sample, not the one true answer. Reported alongside
the others, never alone.

**SSIM** compares local luminance, contrast and structure, tolerating small
intensity shifts that PSNR punishes. Computed on grayscale — these micrographs
carry no real colour, so per-channel SSIM would average three copies — using the
Wang et al. (2004) parameterisation: an $11\times11$ Gaussian window with
$\sigma = 1.5$ and the population covariance. scikit-image defaults to a
$7\times7$ uniform window with the sample covariance instead; both are
self-consistent for ranking models here, but only the former gives numbers
comparable to SSIM values reported elsewhere.

**LPIPS** is distance in the feature space of a pretrained CNN (AlexNet
backbone), calibrated against human similarity judgements. Of the three it best
tracks "looks like the same thing" for textured images, where PSNR is dominated
by high-frequency detail no method can reproduce exactly. Lower is better.

### 5.4 Distributional metrics

Both computed on 2048-d pool features from torchvision's InceptionV3.

**FID** — squared Wasserstein-2 distance between Gaussians fitted to the two
feature sets:

$$\text{FID} = \|\mu_a-\mu_b\|_2^2 + \operatorname{Tr}\!\left(\Sigma_a + \Sigma_b - 2(\Sigma_a\Sigma_b)^{1/2}\right)$$

**KID** — unbiased MMD² with the polynomial kernel
$k(x,y) = \left(\frac{x^\top y}{d}+1\right)^3$:

$$\widehat{\text{MMD}}^2 = \frac{1}{m(m-1)}\sum_{i\ne j}k(x_i,x_j) + \frac{1}{m(m-1)}\sum_{i\ne j}k(y_i,y_j) - \frac{2}{mn}\sum_{i,j}k(x_i,y_j)$$

averaged over random subsets, reported with the standard deviation across
subsets.

**The reference set excludes the images being scored.** In the round trip the
sources are drawn from the same `real/val` split the reference comes from, so
scoring against a reference containing the very images that were reconstructed
would be optimistic. `evaluate.py` removes them before striding, so the
reference stays at its full size and is disjoint from the run. In synth mode the
sources are synthetic and never appear in a real reference, so nothing is
removed.

**On the reported KID spread.** The point estimate is computed once on the full
sets. The `±` is the standard deviation across random subsets and should be read
as an order-of-magnitude "differences smaller than this are noise", **not** as a
standard error: at a few hundred images the subsets overlap heavily (100 drawn
from 200) and are far from independent, so the spread understates the true
uncertainty. Estimating it from disjoint subsets instead would permit only two
subsets at $n=200$ — too few to estimate a spread at all — so the overlapping
version is kept and labelled rather than replaced.

**Two further caveats that must appear in the report.**

1. **FID is unreliable at these sample sizes.** It estimates a 2048×2048
   covariance; below 2048 samples that covariance is rank-deficient and FID is
   dominated by the resulting bias rather than by any real difference between
   the sets. At $n=200$ this is severe. The evaluator flags it via
   `fid_reliable`, and it is reported only because readers look for it. **KID's
   estimator is unbiased at any $n$ and is the one to trust here.**
2. **Absolute values are not comparable to published figures.** Canonical FID
   uses the TF-ported InceptionV3; this uses torchvision's. Comparisons *between*
   the models here are valid; comparisons to numbers in other papers are not.
   (This would be true regardless, since the domain differs from any published
   benchmark.)

### 5.5 Diversity

Mean pairwise RMSE between variants generated from the same input. Structurally
**zero** for a deterministic translator, which is the property DPS claims over
it.

**Higher is not simply better** — it has a sensible range, like prominence.
Calibration on real data:

| | pairwise RMSE |
|---|---|
| two crops of the **same** real photo | 0.0875 |
| two **different** real photos | 0.1436 |

Variants differing by much more than 0.144 are more different from each other
than two unrelated real images are — over-dispersion rather than richer realism.

These reference points are measured on raw dataset PNGs, so they are only
comparable to generated images saved with the fixed affine map of §5.0. A
min-max-normalised save changes the measured value in a direction that depends
on the images, so the two cannot be mixed.

### 5.6 Memorisation

The prior is finetuned on scarce real data, and a prior that regurgitates
training images scores *well* on every realism metric while making the
augmentation worthless. So this is checked explicitly.

Each image is reduced to 32×32, mean-subtracted and $L^2$-normalised; distance to
the nearest training image is then $\sqrt{2-2\max_j \langle f, g_j\rangle}$.

The number is meaningless without its **floor**: the median nearest-neighbour
distance *among training images themselves*, measured at **1.1125**. Generated
images landing well below it indicates copying. Held-out real images sit at
1.1145 — statistically identical to the floor — confirming the split is clean and
the floor is correctly calibrated.

**A confound to note:** smooth, low-detail images sit closer to everything in
this feature space, so a low score can indicate blandness rather than copying.
Distinguish by looking at the nearest-neighbour pairs.

---

## 6. Experimental design

### 6.1 Two experiments

| | input | ground truth | metrics |
|---|---|---|---|
| **round trip** | $y = G_{R\to S}(r)$ for real $r$ | $r$ itself | all, including paired |
| **synth** | $y = s$, a real synthetic image | none | all except paired |

The round trip exists because it is the only way to obtain paired ground truth.
The synth experiment is the actual deliverable.

### 6.2 Biases that must be reported

**The round trip structurally favours `uvcgan`.** $G_{S\to R}$ was trained with a
cycle-consistency loss to invert $G_{R\to S}$ — it was explicitly optimised for
this task. A DPS win here is therefore strong evidence; a DPS loss is not
damning.

**`dps_analytic` receives a mismatched measurement.** All three models get the
same $y$, built by the same $R\to S$ model, because otherwise the comparison is
not like-for-like. But `dps_analytic` inverts an operator that did not produce
that $y$. This is not a flaw in the design — it *is* what the ablation measures:
how much is lost by not having a trained translation model.

### 6.3 Sampling hygiene

**Subsets are strided, not truncated.** Crops are named
`<photo>_sample_<k>`, so taking the first $N$ files collapses onto the
alphabetically-first photos: `--limit 8` yielded **2** distinct scenes and
`--limit 32` yielded 7. Striding across the sorted list yields $N$ distinct
scenes. This is not cosmetic — an early scale sweep on 2 scenes produced a
p90 spacing error of 0.49 that vanished to 0.05 on 24 scenes, and would have led
to the wrong hyperparameter choice.

**Train/val split is at source-photo level.** Several crops come from each
photo, so splitting on filenames scatters siblings across both sides: the
original split leaked 1 969 of 1 970 validation sources into training. This
invalidates every validation number and specifically breaks the memorisation
check, since a "held-out" image whose siblings were trained on looks memorised
regardless.

**All models see the identical subset**, which is what makes a figure row and a
metric table row a like-for-like comparison.

---

## 7. Things to state as limitations

- The analytic operator reproduces the synth domain's global periodicity but not
  its local morphology (sinusoidal vs blob-like). `dps_analytic` results should
  be read as a lower bound on what a *poor* operator achieves, not as a
  well-tuned alternative.
- `learn_sigma` is off, so sampling uses a fixed variance. This costs some
  sample quality relative to a full hybrid-VLB model.
- The sampling schedule is 250 steps rather than 1000, a compute compromise. If
  a reviewer asks, re-running a handful of images at 1000 steps and comparing is
  cheap.
- FID at $n \lesssim 2000$ is not trustworthy (§5.4); lead with KID.
- The synth dataset used 15 000 images from one of ten available shards. Since
  the shards are a size-based split of a randomly ordered set, this is a
  representative sample rather than a restricted region of parameter space — but
  more data would still improve the pretrain.
- The diffusion prior was trained on ~2.9k distinct real photographs. The
  memorisation check passes comfortably, but the conclusion is specific to this
  training budget.
