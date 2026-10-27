# `ssmlpc`: the unlabeled term of Soleymani Baghshah & Bagheri Shouraki (2009)

Mahdieh Soleymani Baghshah and Saeed Bagheri Shouraki, *Semi-Supervised Metric
Learning Using Pairwise Constraints*, IJCAI 2009, pages 1217–1222.
[PDF](https://www.ijcai.org/Proceedings/09/Papers/205.pdf).

Only the paper's regularizer is implemented, which is what was asked for. It is
registered as `method_params.regularizer = "ssmlpc"` and combines with the
repository's supervised metric losses through the ordinary
`supervised_loss + w * regularizer` path, so the target ratio, `regularizer_max_ratio`,
GradNorm and gradient surgery all apply to it unchanged.

## What the paper optimizes, and what is ported

The full objective is the constrained trace ratio of eqs. (1)/(2):

```
W* = argmax_{W^T W = I}  tr(W^T S_b W) / (tr(W^T S_w W) + gamma * J(W))
```

`S_w` and `S_b` are the scatter matrices of the positive (must-link) and
negative (cannot-link) constraint sets, eqs. (3) and (4). `J(W)` is the only
term that reads unlabeled data.

| Paper | Here |
|---|---|
| `J(W)`, eqs. (5)–(6) | **implemented** |
| `S_w`, `S_b`, eqs. (3)–(4) | not ported; the repository's supervised metric loss takes their place |
| trace *ratio* form, eq. (2) | not ported; this is an additive regularizer |
| `W^T W = I` | not ported; the head is an unconstrained `Linear` + `F.normalize` |
| kernelized form, eqs. (12)–(14) | not ported; the DINOv2 trunk is the nonlinearity |
| Xiang et al. heuristic search | not applicable; this trains by SGD |

### Equation (5) — the reconstruction weights

```
min_S  sum_i || x_i - sum_{j in N_k(x_i)} s_ij x_j ||^2
s.t.   s_ij = 0 for x_j not in N_k(x_i),   sum_j s_ij = 1
```

This is Locally Linear Embedding's first step (Roweis & Saul, 2000). Because
the weights sum to one the residual is a difference of differences, so the
problem is a constrained least squares with local Gram matrix
`C_jl = (x_i - x_j) . (x_i - x_l)`, solved as `C w = 1` and rescaled to
`sum_j w_j = 1`. `solve_lle_reconstruction_weights` conditions `C` with Roweis
& Saul's `C += delta^2 * trace(C) * I / k`, exposed as
`reconstruction_regularization` (default `1e-3`); it binds whenever the `k`
local difference vectors do not span their own space, which for `k > d` is
always.

### Equation (6) — the penalty

```
J = sum_i || z_i - sum_{j in N_k(x_i)} s*_ij z_j ||^2  =  tr(Z E Z^T),
E = (I - S*)^T (I - S*)
```

The term is anchored on **nodes**, not edges: each summand needs one node and
every neighbor it is reconstructed from at the same time. That is why it uses
`GraphNeighborhoodBatchSampler` rather than the `GraphEdgeBatchSampler` the
Laplacian regularizers share. A kNN graph gives every node the same neighbor
count, so a batch's map is rectangular and the energy is a dense gather plus a
weighted sum along `k` — which keeps it bit-reproducible on CUDA, where a
float `index_add_` over shared neighbors would not be.

## Two properties to know before reading a training curve

**A constant embedding is a global minimizer.** With `sum_j s_ij = 1`,
`z_i - sum_j s_ij z_j = c (1 - sum_j s_ij) = 0`. The paper is not exposed to
this because `J` sits in the *denominator* of a ratio whose numerator pushes
cannot-link pairs apart, and because `W^T W = I` forbids a rank-deficient map.
Taking the term out of the ratio discards both guards, and nothing here
reinstates them. Measured on a fully collapsed embedding, eq. (6) is `3.7e-15`
while LRML's normalized Laplacian is `2.5e-3` — the Laplacian is nonzero only
because `1/sqrt(deg)` differs per node, so eq. (6) is if anything the *more*
exactly collapse-minimized of the two.

**The affine invariance does not survive the output normalization.** As pure
algebra eq. (6) is invariant to an affine map: the residual of `A x + b` is `A`
applied to the input's own residual, so an affine image of the data the weights
were fit on carries no more energy than the fit already left behind. That is a
materially weaker attractor than a Laplacian, which on the unit sphere reduces
to `-2 sum_ij w_ij <z_i, z_j>` plus a constant and therefore pays for
contraction directly. **But the retrieval head is `normalize(Wx + b)`, and the
normalization is not affine.** The invariance is real (`tests/test_ssmlpc.py::
test_the_penalty_is_affine_invariant`) and does not transfer to this head.

Measured, 6000 real iNat DINOv2-B rows, a random 768-d linear head, Adam at
`lr=1e-4`, **the regularizer alone with no supervised loss** — a stress test,
not a prediction for a real run:

| step | eq. (6) centroid norm | eq. (6) rank | Laplacian centroid norm | Laplacian rank |
|---:|---:|---:|---:|---:|
| 0 | 0.2204 | 511.6 | 0.2204 | 511.6 |
| 100 | 0.7760 | 460.6 | 0.8093 | 478.0 |
| 200 | 0.9012 | 465.7 | 0.9262 | 500.3 |
| 300 | 0.9473 | 477.1 | 0.9634 | 521.3 |

**Eq. (6) collapses at essentially the same rate as the Laplacian.** Anyone
hoping this term fixes what `scripts/reports/inat_lrml_diagnosis/report.md`
diagnosed should read that table first. On the same setup at initialization its
gradient is about **6x larger** than the normalized Laplacian's
(`9.6e-2` vs `1.5e-2`), mostly because the Laplacian's `1/sqrt(d_i d_j)`
deflates it by roughly the mean degree; there is no vanishing-gradient regime
here, so target-ratio calibration is safe to switch on.

`train/ssmlpc/centroid_norm` and `train/ssmlpc/residual_scale` are written to
`diagnostics.csv` on every diagnostic step so this is visible rather than
inferred. A centroid norm climbing toward 1 while the residual shrinks is the
failure.

## Parameters

`method_params.regularizer_params`:

| Key | Default | Meaning |
|---|---|---|
| `n_neighbors` | `10` | `k` of `N_k(x_i)`. Section 3.1: "the parameters of our algorithm are set to `k = 10`". |
| `neighbor_metric` | `"euclidean"` | LLE fits by least squares in the input space, so its neighborhoods are Euclidean. `"cosine"` reuses the ranking the Laplacian regularizers build, which lets a study hold the neighborhoods fixed and vary only the weights. On unit-norm rows the two are identical. |
| `reconstruction_regularization` | `1e-3` | Roweis & Saul's `delta^2`, relative to `trace(C)`. |
| `graph_source` | `"backbone_features"` | The paper: `S*` is the *input* data's, solved once and held fixed. Under a frozen backbone that is the cached DINOv2 matrix, which is literally eq. (5)'s `X` while `W` is the trained head. `"embeddings"` re-solves from the current projection on the engine's refresh schedule, matching this repository's Laplacian regularizers; available for comparability, not fidelity. |
| `graph_on` | `"all"` | Labeled + unlabeled nodes, or `"unlabeled"` only. |
| `graph_batch_size` | `None` | Sampled **anchors** per step; `None` matches the supervised batch size. A batch loads at most `graph_batch_size * (1 + k)` nodes before deduplication. |
| `reduction` | `"mean"` | Per-anchor mean. `"sum"` rescales the sampled batch to eq. (6)'s full trace, which is unbiased under uniform anchor sampling (asserted in `test_uniform_anchor_estimates_average_to_the_full_trace`). `"batch_sum"` is the raw batch sum, whose scale then depends on `graph_batch_size`. |
| `oracle_same_class_neighbors` | `false` | Shorthand for `graph_edge_ablation="oracle_pruned"`. |

`graph_source="backbone_features"` requires `--use_cache`; a dataset that still
yields images has no input representation to fit and says so rather than
silently fitting the weights on pixels.

### On `gamma`

Eq. (15) sets `gamma = 0.2` for `d > 5` and `0.02` for `d <= 5`. With
`d = 768` the applicable value is **0.2**, which is what
`configs/ssl_methods/ssmlpc.json` ships as `regularizer_weight`. Note this is
not directly comparable to the repository's other regularizer weights: the
paper's `gamma` balances `J` against a constraint *scatter trace* inside a
ratio, not against a metric loss's gradient. `ssmlpc.json` is the
scale-free alternative.

### Oracle ablation

`ssl_config.graph_edge_ablation` works as it does for LRML — `graph_edge_ablation_is_supported`
now admits both regularizers — with modes `off`, `normal` (audit purity only),
`oracle_pruned` and `random_pruned` (the neighborhood-size control, which drops
the same total number of neighbors uniformly).

It cannot reuse `apply_graph_edge_ablation`: that helper prunes a symmetric
adjacency whose weights carry no constraint, whereas eq. (5)'s weights must sum
to one over the neighbors that remain. So
`apply_reconstruction_neighbor_ablation` returns a *mask*, and the weights are
solved afterwards on the retained columns. Pruning after the solve would leave
eq. (5)'s constraint broken. Every mode after `off` reads hidden training
labels and logs a `SSMLPC NEIGHBOR ABLATION` warning; none of them is a
deployable configuration.

## Numerical notes

LLE weights are not convex combinations — they are signed, and 15% of them are
negative on real DINOv2 rows. Their magnitude is bounded only by the
conditioner, and an ill-conditioned local Gram produces large opposing weights:
measured `[-0.46, 1.00]` at `k=10` on 6000 iNat rows, but `[-9.0, 13.0]` at
`k=4` once the oracle filter has thinned the neighborhoods. `weight_range` and
`negative_weights` are printed when eq. (5) is solved; a wide range there means
a noisier gradient and is a reason to raise `reconstruction_regularization`
rather than to lower the weight.

Rows left with no usable neighbor have all-zero weights, are reported as
`isolated_nodes`, and are excluded from the anchor pool: such a row's residual
would be its own embedding rather than a reconstruction error, and would pull
it toward the origin.

## Files

| Path | Contents |
|---|---|
| `training/ssl/ssmlpc.py` | `SsmlpcRegularizer`, `reconstruction_energy` |
| `training/ssl/algorithms.py` | `solve_lle_reconstruction_weights`, `find_reconstruction_neighbors`, `build_lle_reconstruction_graph`, `apply_reconstruction_neighbor_ablation`, `faiss_flat_l2_search` |
| `training/ssl/data.py` | `GraphNeighborhoodBatchSampler`, `collate_graph_neighborhood_batch` |
| `configs/ssl_methods/ssmlpc.json` | The paper's fixed `gamma = 0.2` |
| `../configs/ssl_methods/ssmlpc.json` | Target ratio 0.1, `statistic="l2"`, `regularizer_max_ratio=1.0` |
| `tests/test_ssmlpc.py` | 43 tests |

## Not implemented

- The constraint scatters `S_w`/`S_b` and the trace-ratio form. Adding them
  would be a different method, not a regularizer.
- `W^T W = I`. The orthonormality constraint is the paper's structural guard
  against a degenerate map and has no counterpart here.
- Graph diagnostics rendering. `graph_diagnostics_mode` draws a symmetric
  adjacency; eq. (5)'s object is a signed, directed, row-normalized weight
  matrix and would need its own renderer.
