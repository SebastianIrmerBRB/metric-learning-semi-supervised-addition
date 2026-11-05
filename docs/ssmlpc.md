# `ssmlpc`: the unlabeled term of Soleymani Baghshah & Bagheri Shouraki (2009)

Mahdieh Soleymani Baghshah and Saeed Bagheri Shouraki, *Semi-Supervised Metric
Learning Using Pairwise Constraints*, IJCAI 2009, pages 1217–1222.
[PDF](https://www.ijcai.org/Proceedings/09/Papers/205.pdf).

The unlabeled reconstruction penalty is registered as `method_params.regularizer = "ssmlpc"` and combines with the
repository's supervised metric losses through the ordinary
`supervised_loss + w * regularizer` path, so the target ratio, `regularizer_max_ratio`,
GradNorm and gradient surgery all apply to it unchanged.

## Reconstruction regularization

The method preserves local linear reconstructions from input features in the
learned embedding. It combines this penalty with the configured supervised loss.

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

## Collapse and normalization

A constant embedding minimizes the penalty because every reconstruction row
sums to one: `c - sum_j s_ij c = 0`. Keep supervised separation active; the
reconstruction term alone does not prevent collapse.

The reconstruction identity is affine-invariant before output normalization.
The retrieval head applies `normalize(Wx + b)`, so that invariance does not
carry through the full head. Track `train/ssmlpc/centroid_norm` and
`train/ssmlpc/residual_scale` in `diagnostics.csv`: a centroid norm approaching
one while residuals shrink can indicate collapse.

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

LLE weights are signed and can include negative values. An ill-conditioned
local Gram matrix can produce large opposing weights. Inspect `weight_range`
and `negative_weights` when the weights are solved; a wide range can justify
raising `reconstruction_regularization`.

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
