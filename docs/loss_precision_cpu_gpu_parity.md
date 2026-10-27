# CPU/GPU parity for loss precision under BF16 autocast

## Symptom

Running the same configuration on CPU and on GPU gave noticeably different
`mean_average_precision_at_r`, but only for `ProxyAnchorLoss`. `CircleLoss`,
`NTXentLoss` and `MultiSimilarityLoss` agreed; `ArcFaceLoss` agreed closely
enough not to stand out.

## Root cause

`training/engine.py` has exactly one autocast site, and it is not
CUDA-specific:

```python
# training/engine.py, _compute_batch_loss
with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
```

`self.device_type` comes from `torch.device(args.device).type`, so a CPU run
gets BF16 autocast too. On its own that is fine. The problem is how
pytorch-metric-learning detects autocast.

`c_f.to_dtype` is how pml pushes tensors to the embedding dtype. It
deliberately does nothing while autocast is active, so that autocast alone
decides precision:

```python
# pytorch_metric_learning/utils/common_functions.py
def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():   # <-- no device argument
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x
```

`torch.is_autocast_enabled()` with no argument reports the **CUDA** autocast
state only. Verified in this environment (torch 2.13.0):

```
inside CPU bf16 autocast  : False
  is_autocast_enabled('cpu') : True
  is_autocast_enabled('cuda'): False
```

So the guard holds on CUDA and silently lapses on CPU. Two places in a proxy
or classifier loss then diverge by device.

### 1. `cast_types` — the learnable weights

Every proxy/classifier loss rewrites its own weights on each forward:

```python
# pytorch_metric_learning/losses/proxy_anchor_loss.py
def cast_types(self, dtype, device):
    self.proxies.data = c_f.to_device(self.proxies.data, device=device, dtype=dtype)
```

- **CUDA**: guard holds, cast skipped, proxies stay FP32. This is ordinary
  mixed precision: BF16 matmuls against FP32 master weights.
- **CPU**: guard lapses, `self.proxies.data` is replaced with BF16
  *permanently*. From then on the weights, their gradients, and the AdamW
  moments are all BF16.

The optimizer-state half is the damaging part. BF16 has `eps = 0.0078125`, and
Adam's `beta2 = 0.999` decay is a 0.1% change — finer than BF16 can represent,
so `exp_avg_sq.mul_(0.999)` rounds back to itself every time:

```
Adam second-moment decay (exp_avg_sq *= 0.999, applied 50x):
  float32   1.00e-03 -> 9.5121e-04   decays correctly
  bfloat16  1.00e-03 -> 9.9945e-04   FROZEN - decay is below resolution
```

The second moment never decays. That is not reduced precision; it is a
different optimizer.

### 2. `convert_to_weights` — the accumulation dtype

```python
# pytorch_metric_learning/utils/loss_and_miner_utils.py
def convert_to_weights(indices_tuple, labels, dtype, using_ref=False):
    weights = torch.zeros(labels.shape[0], device=labels.device)   # FP32
    weights = c_f.to_dtype(weights, dtype=dtype)
```

ProxyAnchor adds this vector to its logits before the log-sum-exp:

```python
pos_term = lmu.logsumexp((self.alpha * pos_exp) + miner_weights, ...)
```

- **CUDA**: guard holds, `miner_weights` stays FP32, `bf16 + fp32` promotes,
  and the `logsumexp` accumulates in **FP32**.
- **CPU**: guard lapses, `miner_weights` becomes BF16, and the `logsumexp`
  accumulates in **BF16**.

The FP32 accumulation on CUDA is incidental — a side effect of type promotion
— but it is what the GPU runs actually do.

## Why ProxyAnchor was the visible one

`CircleLoss`, `NTXentLoss` and `MultiSimilarityLoss` have **zero learnable
parameters**, so `cast_types` has nothing to touch. In a controlled
reproduction (identical seed, data, init and batches; the only variable being
whether the cast is allowed) they came out bit-identical at every difficulty
and seed:

```
loss                  GPU mean  CPU mean  |Δ| mean  |Δ| max  seed std
CircleLoss              0.6560    0.6560    0.0000   0.0000    0.0102
NTXentLoss              0.6518    0.6518    0.0000   0.0000    0.0224
MultiSimilarityLoss     0.6245    0.6245    0.0000   0.0000    0.0188
ArcFaceLoss             0.7793    0.7852    0.0120   0.0139    0.0121
ProxyAnchorLoss         0.8307    0.8306    0.0014   0.0021    0.0138
```

That accounts for three of the five exactly. Between the two parameterised
losses, gradient magnitude relative to each weight's own BF16 rounding step
(measured at batch 16, 98 classes, dim 128) separates them:

```
loss              classes w/ gradient  |grad| below bf16 ULP     median |g|/ULP
ProxyAnchorLoss                100.0%                   2.8%            12.632
ArcFaceLoss                    100.0%                  63.3%             0.138
```

ProxyAnchor's proxies genuinely move every step (~12 ULP), so each step is
quantised and the trajectory compounds away from the FP32 run. ArcFace's `W`
is nearly static either way — most of its signal is already sub-ULP — so both
devices land in much the same place. ProxyAnchor is also the only one of the
five whose `logsumexp` accumulation dtype changed (site 2 above).

**Caveat.** The synthetic reproduction demonstrates the mechanism and the exact
CPU/GPU split cleanly, but at that scale ArcFace's swing was comparable to
ProxyAnchor's rather than clearly smaller. The ULP measurement is the
defensible differentiator; the real-run magnitude gap was not reproduced
synthetically.

## What a GPU run actually computes in

Measured on real CUDA (torch 2.13.0+cu130), through the actual pipeline —
`project_features` normalises before the loss sees anything:

```
loss                    embeddings  cos/logits      loss   weights  weight.grad
CircleLoss                 float32         n/a   float32      none         none
NTXentLoss                 float32         n/a   float32      none         none
MultiSimilarityLoss        float32         n/a   float32      none         none
ArcFaceLoss                float32    bfloat16   float32   float32      float32
ProxyAnchorLoss            float32    bfloat16   float32   float32      float32
```

The two parameterised losses keep FP32 master weights and FP32 gradients, and
their similarity matmul runs in BF16. Everything else on GPU is FP32.

### A second, independent divergence: autocast op policy is per-backend

The guard is not the only thing that differs by device. CPU and CUDA autocast
maintain **different op lists**, and two of those differences land here:

```
                          CUDA        CPU
F.normalize            float32    bfloat16     <- promoted on CUDA only
loss reduction         float32    bfloat16     <- ProxyAnchor, Circle, NTXent, MS
```

`models/retrieval_model.py:417` ends `project_features` with `F.normalize`,
inside the engine's autocast block. On CUDA that promotes to FP32, so the
embeddings reaching *every* loss are FP32; on CPU they stay BF16. And even when
fed BF16 embeddings directly, CUDA still returns an FP32 loss for all five.

This has nothing to do with `to_dtype` and is not affected by the shim. It
predates this change and remains after it. It is also why the earlier claim
that "all five compute in BF16 on GPU" was wrong: that table was produced by
simulating CUDA on CPU (below), and a CPU simulation cannot reproduce CUDA's
op list.

## The change

`losses/metric_losses.py` installs a device-aware replacement for the guard at
import time:

```python
def _device_aware_to_dtype(x, tensor=None, dtype=None):
    if torch.is_tensor(x) and torch.is_autocast_enabled(x.device.type):
        return x
    ...
```

`install_device_aware_autocast_guard()` is idempotent and only replaces the
upstream function, never a previous patch. Because pml calls `c_f.to_dtype`
through the module (and `to_device` resolves it from the same module globals),
one replacement covers every call site.

This was chosen over the two alternatives:

- **Force FP32 for classification losses** (disable autocast around the
  criterion). Fixes the divergence, but changes GPU results too: the loss
  *reduction* on GPU is already FP32, but the similarity matmul that produces
  `cos/logits` genuinely runs in BF16 there, and forcing FP32 would change it.
  Measured cost of *not* doing this is small — FP32 vs BF16 loss math moves
  MAP@R by ~0.001, against seed noise of ~0.014 — so it is not worth
  invalidating existing GPU numbers.
- **Match BF16 everywhere** (force the downcast on CUDA too). Technically
  possible, but it drags the optimizer state to BF16 and freezes Adam's second
  moment, as shown above. That exports a defect to the runs that are currently
  correct.

Nothing at the training call sites changed. `training/engine.py` and
`training/semi_supervised.py` are untouched by this fix.

### Removed

`NumericallyStableArcFaceLoss` is gone from `losses/metric_losses.py`, along
with the now-unused `import math` and `UpstreamArcFaceLoss` import. It forced
ArcFace's math to FP32 and clamped `acos` away from ±1, but it was never
registered in `LOSS_REGISTRY` and never referenced anywhere, so
`get_loss_class` always resolved `"ArcFaceLoss"` to upstream pml. The
protection had never been active, and removing it changes nothing.

One consequence worth recording: that class also guarded against a genuinely
exact ±1 cosine, where `acos` has an infinite derivative and a single step can
NaN the projection head. Keeping ArcFace's cosine in BF16 leaves that hazard
live. It has not been observed in practice, and addressing it would mean
changing GPU behaviour, so it was left alone deliberately.

## Verification

Verified against **real CUDA**, not a simulation. The shim does what it is
meant to do: the learnable weights, their gradients and the AdamW moments all
stay FP32 on CPU, matching CUDA, where before they were BF16.

```
loss              variant        weights   w.grad    exp_avg_sq
ArcFaceLoss       REAL CUDA      float32   float32   float32
                  CPU + shim     float32   float32   float32     <- fixed
                  CPU, upstream  bfloat16  bfloat16  bfloat16    <- the bug
ProxyAnchorLoss   REAL CUDA      float32   float32   float32
                  CPU + shim     float32   float32   float32
                  CPU, upstream  bfloat16  bfloat16  bfloat16
```

Also confirmed directly: the shim is idempotent, refuses to replace a
non-upstream patch, still casts outside autocast, and reaches every call site
(`to_device` resolves `to_dtype` from the same module globals). It installs
before any training, because `training/engine.py:22` imports
`losses.metric_losses` at module load.

### What the shim does *not* fix

CPU does **not** reproduce the CUDA path exactly. The dtypes of the embeddings
and of the loss itself still differ, for the op-list reason above. The
remaining weight gap is real but **bounded and non-compounding** — which is the
distinction that matters:

```
max |W_cpu - W_cuda| over 200 AdamW steps (identical data, init, seed)

                     before fix     after fix
ProxyAnchorLoss
  step   1            2.193e-03     2.000e-03
  step  25            6.215e-03     2.474e-03
  step  50            1.100e-02     2.456e-03
  step 100            1.499e-02     2.580e-03
  step 200            2.318e-02     2.526e-03
ArcFaceLoss
  step   1            2.193e-03     2.000e-03
  step  50            1.130e-02     2.816e-03
  step 200            2.326e-02     2.600e-03
```

Before the fix the gap compounds — 10x over 200 steps and still climbing, the
signature of a different optimizer. After the fix it is flat: the same ~2.5e-3
at step 200 as at step 1, roughly two AdamW steps at `lr=1e-3`, which the
optimizer's normalised step re-absorbs instead of accumulating. This floor
cannot be removed by making the guard device-aware; it would require matching
every autocast op policy between backends.

### The CPU-only simulation, and its limit

Making `to_dtype` a no-op — exactly what the upstream guard does when it
returns `True` — and comparing against CPU + shim gives a clean match for all
five losses, including bitwise-identical weights:

```
loss                          CUDA semantics                     CPU + shim   match
CircleLoss             loss=bfloat16,w=none,|w|=None  loss=bfloat16,w=none,|w|=None   YES
NTXentLoss             loss=bfloat16,w=none,|w|=None  loss=bfloat16,w=none,|w|=None   YES
MultiSimilarityLoss    loss=bfloat16,w=none,|w|=None  loss=bfloat16,w=none,|w|=None   YES
ArcFaceLoss          loss=float32,w=float32,|w|=5.575490    ...same...                YES
ProxyAnchorLoss     loss=bfloat16,w=float32,|w|=5.574285    ...same...                YES
```

This is worth keeping, because it isolates the guard cleanly. But it proves
only that the shim reproduces the *guard's* behaviour. It says nothing about
CUDA's autocast op list, so it must not be read as CPU/GPU equivalence — the
real-CUDA numbers above are the ones that speak to that.

### Test coverage

`tests/test_loss_parameter_precision.py`, 13 tests, all passing. The suite uses
`unittest`; `pytest` is not installed.

```
venv/bin/python -m unittest tests.test_loss_parameter_precision -v
```

`LossParameterPrecisionTest` (10) pins the guard:

- the upstream guard still ignores CPU autocast (the premise — if this ever
  fails, pml became device-aware and the shim can be dropped)
- the patched guard respects CPU autocast, still casts outside autocast, and
  installs idempotently
- an unprotected loss really does downcast its proxies (the shim is
  load-bearing)
- classification weights and AdamW moments stay FP32
- the loss arithmetic still runs in BF16
- CPU + shim matches the simulated CUDA guard bitwise, for all five losses
- losses without weights are untouched

`CudaLossParameterPrecisionTest` (3, skipped without a GPU) executes the CUDA
half rather than simulating it:

- classification weights, gradients and AdamW moments are FP32 on real CUDA
- CPU and CUDA agree on those dtypes — a dtype claim, which is what the fix
  guarantees, not a bitwise one
- the `F.normalize` backend asymmetry is pinned deliberately, so the divergence
  the shim does *not* address is not rediscovered later as a bug

One test was renamed while adding these. `test_cpu_training_matches_the_cuda_path`
is now `test_cpu_matches_the_simulated_cuda_guard`: both of its sides run on
CPU, and the old name asserting a CPU/CUDA match is what the "reproduces the
CUDA path exactly" claim was built on.

### Full suite

612 tests, 12 failures, 18 errors on WSL (609 before the three CUDA tests were
added) — all pre-existing and unrelated. Verified by diffing the failing-test
IDs against a baseline captured with the shim neutralised: the two sets are
**identical** except for the five subtests that are supposed to fail without
the shim (FP32 weights and optimizer state for ProxyAnchor and ArcFace, and the
simulated-guard match). Nothing else moves.

Discovery needs an explicit module list — `tests/` has no `__init__.py`, so
`unittest discover -s tests` fails with `Start directory is not importable`:

```bash
venv/bin/python -m unittest $(ls tests/test_*.py | sed 's|/|.|; s|\.py$||')
```

The dominant error is a config-validation failure
(`Unknown hoffer_entropy regularizer_params: ['normalize_embeddings']`), plus
assorted CLI-opt-in, slade, simmatch and slrmml failures.

**Counts are platform-dependent.** An earlier baseline of 583 tests / 8
failures / 14 errors was captured on Windows. At least
`test_tsne_visualization` hardcodes `run\test_tsne.csv`, so it passes on
Windows and fails on Linux. Compare failing-test *IDs*, not totals.

## What to expect

- **GPU results are unaffected.** The GPU path already behaved this way; the
  shim only changes what CPU does.
- **CPU ProxyAnchor results will move**, toward the GPU numbers. Previous CPU
  ProxyAnchor runs trained their proxies in BF16 with a frozen Adam second
  moment and should not be compared against new ones.
- **CPU ArcFace results will move slightly**, for the same reason.
- **Pair losses are unchanged by the shim** — they have no weights, and both
  divergence sites only affect losses that do. Note this is *not* the same as
  CPU/GPU equivalence: CPU still runs them in BF16 while CUDA runs them in
  FP32, for the op-list reason above. That difference is unrelated to this fix
  and predates it.
- **CPU and GPU will still not match bitwise**, for any loss. What the fix
  removes is the compounding drift; a small non-accumulating gap remains and is
  expected.

## If pytorch-metric-learning fixes this upstream

The shim is only needed while `c_f.to_dtype` calls `torch.is_autocast_enabled()`
without a device argument. To check, run:

```python
import torch
from pytorch_metric_learning.utils import common_functions as c_f
t = torch.zeros(4, dtype=torch.float32)
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    print(c_f.to_dtype(t, dtype=torch.bfloat16).dtype)
# bfloat16 -> bug still present, shim required
# float32  -> upstream is device-aware, shim can be deleted
```

When it prints `float32`, `install_device_aware_autocast_guard()`,
`_device_aware_to_dtype` and `_upstream_to_dtype` can all be deleted.
`test_upstream_guard_ignores_cpu_autocast` asserts the bug still exists, so the
suite fails loudly when upstream fixes it.
