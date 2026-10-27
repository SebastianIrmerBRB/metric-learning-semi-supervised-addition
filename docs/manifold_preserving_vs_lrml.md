# `mode="manifold_preserving"` vs. the original LRML regularizer

This document defines exactly what the new `mode="manifold_preserving"` option of
[`LRMLRegularizer`](../training/semi_supervised.py) changes relative to the
default `mode="lrml"`, checks the implementation against its source paper, and
records what was measured on real features.

## Bottom line

The graph *weights* change and nothing else does. Both modes build the same kind
of object — a symmetric kNN graph over the same node pool, sampled edge-wise by
the same `GraphEdgeBatchSampler`, reduced by the same estimator, and added to the
same supervised metric loss. `mode="lrml"` puts a constant on every edge;
`mode="manifold_preserving"` puts a density-scaled Gaussian on every edge.

The port of Ying et al.'s regularity term is **faithful to Equation (7)**,
including the two places where a naive reading would go wrong (the directed
$\beta_i$ asymmetry, and the definition of $\sigma$ from the *global* distance
extremes). What it inherits from the deep adapter — L2-normalized features, a
graph rebuilt from the current embedding each epoch — are documented deviations,
not errors.

The one thing that does not survive contact with the deep pipeline is
**numerical range**: the product of two Gaussians spans far more decades than a
float32 training energy can hold, and the top of the shipped search space
produced a graph that was 99.8% inert. That is what caused the
`ZeroDivisionError` in target-ratio calibration. See
[§8](#8-review-findings).

## Sources and scope

| Role | Source |
| --- | --- |
| Paper (new mode) | Shihui Ying, Zhijie Wen, Jun Shi, Yaxin Peng, Jigen Peng, Hong Qiao. *Manifold Preserving: An Intrinsic Approach for Semisupervised Distance Metric Learning*, IEEE TNNLS 29(7):2731–2742, 2018. [Local copy](../papers/Manifold_Preserving_An_Intrinsic_Approach_for_Semisupervised_Distance_Metric_Learning.pdf) |
| Paper (default mode) | Steven C. H. Hoi, Wei Liu, Shih-Fu Chang. *Semi-supervised distance metric learning for collaborative image retrieval and clustering*, 2010 |
| Graph construction | [`training/ssl/algorithms.py`](../training/ssl/algorithms.py) — `build_lrml_knn_graph`, `build_manifold_preserving_graph` |
| Regularizer and training energy | [`training/semi_supervised.py`](../training/semi_supervised.py) — `LRMLRegularizer`, `_build_lrml_graph` |
| Method config | [`configs/ssl_methods/lrml_manifold_preserving.json`](../configs/ssl_methods/lrml_manifold_preserving.json) |
| Search space | [`configs/runs/lrml/first/manifold_preserving.json`](../configs/runs/lrml/first/manifold_preserving.json) |
| Deep adaptation background | [`docs/lrml.md`](lrml.md) |

Ying et al. is the normative reference for everything under
`mode="manifold_preserving"`. The paper leaves the Parzen bandwidth $h$
unspecified; that is called out below as a repository choice, not a mismatch.

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $x_i$ | input sample $i$; $n$ samples total, the first $l$ labeled |
| $z_i = f_\theta(x_i)$ | the network embedding, L2-normalized by the projection head |
| $A$ | the paper's Mahalanobis metric matrix, $A \succ 0$ |
| $D^2_{ij}$ | $(x_i-x_j)^\top A (x_i-x_j)$ in the paper; $\lVert z_i - z_j\rVert^2$ here |
| $N(i)$ | the $k$ nearest neighbors of node $i$, excluding itself |
| $N_{ij}$ | $1$ if $j \in N(i)$, else $0$ |
| $d_{ij}$ | Euclidean distance used to *build* the graph |
| $S_{ij}$ | Gaussian similarity on the graph edge |
| $\beta_i$ | per-node density weight |
| $W$ | the graph weight matrix; $\bar W$ its symmetrization |
| $\deg_i$ | $\sum_j \bar W_{ij}$ |
| $\lambda$ | regularizer weight (`regularizer_weight`, or the target-ratio result) |

## 2. What both modes share

The repository replaces the linear Mahalanobis form $D^2_{ij} = (x_i-x_j)^\top A
(x_i-x_j)$ with the squared distance between network embeddings, and replaces
the paper's labeled loss term with the configured deep metric loss. Both modes
then optimize

$$
\mathcal J(\theta) \;=\; \alpha\,\mathcal L_{\text{sup}}(\theta)
\;+\; \lambda\,\widehat{\mathcal R}_G(\theta),
$$

where $\widehat{\mathcal R}_G$ is an unbiased estimate of the graph energy from a
uniformly sampled mini-batch of undirected edges $B$:

$$
\widehat{\mathcal R}_G
=
\frac{|E|}{|B|}\sum_{(i,j)\in B} \bar W_{ij}\,
\bigl\lVert \tilde z_i - \tilde z_j \bigr\rVert^2 ,
\qquad
\tilde z_i =
\begin{cases}
z_i / \sqrt{\deg_i} & \texttt{normalized\_laplacian=true}\\[2pt]
z_i & \texttt{normalized\_laplacian=false.}
\end{cases}
$$

The $|E|/|B|$ factor is `reduction="sum"`; `"mean"` divides by $|B|$ instead and
`"torch_sum"` leaves the raw sampled sum. Everything above this line — node
pool (`graph_on`), refresh schedule, edge sampler, optional Weston et al.
repulsion, target-ratio calibration — is identical between the two modes.

**Only $\bar W$ differs.**

## 3. `mode="lrml"` — Hoi et al.

The graph is the binary symmetric kNN union: an edge exists if either endpoint
ranks the other in its top $k$, and every edge pulls exactly as hard as every
other.

$$
W^{\text{bin}}_{ij} \;=\; \mathbb 1\bigl[\,j \in N(i) \;\lor\; i \in N(j)\,\bigr],
\qquad
\bar W = W^{\text{bin}} .
$$

The energy is the Laplacian quadratic form over each undirected edge once:

$$
\mathcal R^{\text{LRML}}_G
= \tfrac12 \sum_{i,j} \bar W_{ij} \lVert \tilde z_i - \tilde z_j\rVert^2
= \sum_{i<j} \bar W_{ij} \lVert \tilde z_i - \tilde z_j\rVert^2
= \operatorname{tr}\!\bigl(\tilde Z^\top L\, \tilde Z\bigr),
\quad L = D - \bar W .
$$

`edge_weighting="cosine"` is a repository-only variant that swaps the indicator
for Iscen's affinity $\max(\cos,0)^\gamma$; it is not in either paper.

## 4. `mode="manifold_preserving"` — Ying et al.

### 4.1 The paper's model

Equation (2) of the paper is a convex combination of a labeled triplet term and
a regularity term:

$$
\min_{A \succ 0}\;
(1-\lambda) \sum_{\mathcal T'} \bigl(D^2_{ij} - \gamma D^2_{ik}\bigr)
\;+\;
\lambda \underbrace{\sum_{i=1}^{n} \beta_i \sum_{j \in N(i)} S_{ij} D^2_{ij}}_{\textstyle \mathrm{Reg}(A)} .
$$

Only $\mathrm{Reg}$ is ported. Equation (7) rewrites it as a single sum over a
**directed** weight matrix:

$$
\mathrm{Reg}(A)
= \sum_{i=1}^{n} \beta_i \sum_{j\in N(i)} S_{ij} D^2_{ij}
= \sum_{i,j=1}^{n} W_{ij} D^2_{ij},
\qquad
W_{ij} = \beta_i\, N_{ij}\, S_{ij},
$$

i.e. $W = \bigl(\operatorname{diag}(\beta)\,S\bigr) \odot N$. Note $W \ne
W^\top$: the row scaling $\beta_i$ and the neighbor mask $N_{ij}$ are both
one-sided.

### 4.2 The two kernels (Section IV)

**Similarity.** A Gaussian on the Euclidean distance, with one global bandwidth
set from the *extremes of the whole distance distribution*:

$$
S_{ij} = \exp\!\left(-\frac{d_{ij}^2}{2\sigma^2}\right),
\qquad
\sigma = \min_{p \neq q} d_{pq} \;+\; \frac{1}{v}\Bigl(\max_{p,q} d_{pq} - \min_{p \neq q} d_{pq}\Bigr),
\qquad v = 10 .
$$

**Density.** A Parzen window over the same neighbor list, mapped through the
linear $f(p)=p$ and normalized by its maximum:

$$
p(x_i) = \frac{1}{\lvert N(i)\rvert\,h^m} \sum_{j\in N(i)} K_h\!\left(\frac{x_i - x_j}{h}\right),
\qquad
\beta_i = \frac{p(x_i)}{\max_q p(x_q)} ,
$$

with $K_h$ Gaussian, $\lvert N(i)\rvert = 10$, and $m$ the feature dimension.

### 4.3 What the implementation actually computes

**Symmetrization.** The edge sampler consumes undirected edges, so the directed
$W$ is stored as $\bar W = W + W^\top$. This is exact rather than a change of
graph, because $d^2_{ij}$ is symmetric:

$$
\sum_{i<j} \bar W_{ij}\, d^2_{ij}
= \sum_{i<j} \bigl(W_{ij} + W_{ji}\bigr) d^2_{ij}
= \sum_{i \neq j} W_{ij}\, d^2_{ij}
= \mathrm{Reg}.
$$

This also repairs a slip in the paper: Equation (8) writes $\mathrm{Reg} = 2
\operatorname{tr}\bigl(X(D_w - W)X^\top A\bigr)$ with $D_w = \operatorname{diag}(We)$,
which only equals Equation (7) when $W$ is symmetric. In general
$\sum_{ij} W_{ij} d^2_{ij} = \operatorname{tr}\bigl(A X (D_{\text{row}} +
D_{\text{col}} - W - W^\top) X^\top\bigr)$. Summing $W + W^\top$ over unique
pairs reproduces **Equation (7)**, the definition, so the code follows the
definition rather than the matrix rewrite.

**Density in the log domain.** $h^m$ and the Gaussian's normalizing constant are
identical for every node and cancel in $\beta_i = p(x_i)/\max_q p(x_q)$, so they
are dropped and the estimate is evaluated stably as

$$
\log \beta_i =
\operatorname*{logsumexp}_{j \in N(i)}\left(-\frac{d_{ij}^2}{2h^2}\right)
- \log \lvert N(i)\rvert
- \max_q \left[\;\cdot\;\right]_q .
$$

**Distances on the unit sphere.** Graph features are L2-normalized before the
search, so cosine and Euclidean neighbor rankings coincide and

$$
d_{ij}^2 = 2 - 2\langle z_i, z_j \rangle \in [0, 4].
$$

Both endpoints of $\sigma$ are then computed exactly, without an $O(n^2)$
distance matrix:

- $\min d$ is read off the kNN edges. This is exact: the globally closest pair
  are mutual nearest neighbors, so that pair is in the kNN edge set for any
  $k\ge 1$.
- $\max d$ uses $\max_j \lVert z_i - z_j\rVert^2 = 2 + 2\max_j \langle -z_i, z_j\rangle$,
  one inner-product search with negated queries.

**Bandwidth $h$.** The paper does not specify it. `density_bandwidth=null`
sets $h = \sigma$ so both kernels share one local scale; an explicit positive
value reproduces any other choice.

## 5. The two graphs side by side

| | `mode="lrml"` | `mode="manifold_preserving"` |
| --- | --- | --- |
| Directed weight | $\mathbb 1[j \in N(i)]$ | $\beta_i \exp\!\bigl(-d_{ij}^2/2\sigma^2\bigr)$ |
| Symmetrization | union, clipped back to $\{0,1\}$ | $W + W^\top$ (sums, not clipped) |
| Weight range | $\{1\}$ | $(0,\,2]$, spanning many decades |
| Node term | none | Parzen density $\beta_i$ |
| Scale parameter | none | $\sigma$ from global $\min d$, $\max d$, and $v$ |
| Default $k$ | 6 (Hoi et al.) | 10 (Ying et al. Section IV) |
| Default Laplacian | symmetric-normalized | unnormalized (Eq. 8) |
| Energy convention | $\tfrac12\sum_{ij} \bar W_{ij} d^2_{ij}$ | $\sum_{ij} W_{ij} d^2_{ij}$, exactly Eq. (7) |
| `edge_weighting` | `binary` or `cosine` | forced to `density_gaussian` |
| `gamma` | read when `cosine` | unread |

Selecting the mode rewrites two defaults, so the paper's configuration is what
you get from `{"mode": "manifold_preserving"}` alone:

```python
if mode == "manifold_preserving":
    defaults["n_neighbors"] = 10          # Section IV
    defaults["normalized_laplacian"] = False   # Equation (8)
```

Explicit values still win; the shipped
[`lrml_manifold_preserving.json`](../configs/ssl_methods/lrml_manifold_preserving.json)
sets `normalized_laplacian: true`, which is the LRML convention, not Eq. (8).

## 6. Paper → code map

| Paper | Code |
| --- | --- |
| Eq. (2) regularity term | `build_manifold_preserving_graph` + the shared edge energy |
| Eq. (3) $\beta_i = f(p(x_i))$ | `density_weights = np.exp(log_density - log_density.max())` |
| Eq. (7) $W_{ij} = \beta_i N_{ij} S_{ij}$ | `directed_values = density_weights[:, None] * similarities` |
| Eq. (7) directed sum | `adjacency = (directed + directed.T)`, summed over the upper triangle |
| Eq. (8) unnormalized $D_w - W$ | `normalized_laplacian=False` default for this mode |
| Section IV item 3), Parzen density with $\lvert N(i)\rvert=10$ | `special.logsumexp(...) - np.log(k)` |
| Section IV item 4), $S_{ij}$, $\sigma$, $v=10$ | `similarity_sigma = minimum_distance + (maximum_distance - minimum_distance) / similarity_v` |
| Section IV item 3), $p := p/\max p$ | `log_density - log_density.max()` |

## 7. What is *not* ported

These are deliberate consequences of reusing the deep adapter, and they are why
this mode should be described as *the paper's regularizer*, not a reproduction
of the paper's method.

1. **The labeled term.** Equation (1)'s $\sum (D^2_{ij} - \gamma D^2_{ik})$ and
   its adaptive $\gamma = 1/(1+\bar D)$ are replaced wholesale by the configured
   deep metric loss. (The regularizer's own `gamma` parameter is unrelated — it
   is the exponent of the repository's `edge_weighting="cosine"`, and is unread
   in this mode.)
2. **The optimizer.** Section III's intrinsic steepest descent on the SPD
   manifold, $A(t{+}1) = A(t)^{1/2}\exp(-\alpha G(t))A(t)^{1/2}$, has no
   counterpart: Adam updates network parameters. The positive-definiteness the
   paper's whole algorithm is built to preserve is not a constraint here.
3. **The $\lambda$ convex combination.** The paper couples the two terms as
   $(1-\lambda)$ and $\lambda$. Here `supervised_weight` and
   `regularizer_weight` are free, and normally set by target-ratio calibration.
4. **The static input-space graph.** The paper measures $d_{ij}$, $S_{ij}$,
   $\beta_i$, and $\sigma$ once, on raw inputs, and never moves them; only $A$ is
   learned. Here the graph is rebuilt from the *current embedding* every
   `update_interval_epochs`, so $\sigma$ and every weight are re-measured in a
   space the regularizer is itself deforming. Since $S_{ij}$ is largest for pairs
   that are already close, this is self-reinforcing in a way the paper's
   formulation is not.
5. **Row normalization.** The paper uses raw Euclidean distances; the projection
   head normalizes, capping $d \in [0,2]$ and changing both endpoints of
   $\sigma$.

## 8. Review findings

### 8.1 Correct

Verified against the paper text and by
[`tests/test_lrml_manifold_preserving.py`](../tests/test_lrml_manifold_preserving.py):

- Equation (7)'s directed energy equals the sampled undirected energy exactly,
  mutual-neighbor edges included.
- $\sigma$'s endpoints are the true global extremes, obtained without an
  $O(n^2)$ matrix.
- The density estimate is the paper's, with the constants that cancel under
  max-normalization dropped rather than approximated.
- The mode's defaults are the paper's ($k=10$, $v=10$, unnormalized Laplacian).
- The graph is invariant to input row scale, as it must be for a cosine-neighbor
  convention.

### 8.2 Fixed

**The Gaussian weights underflow float32, and calibration divided by the result.**
$\beta_i$ and $S_{ij}$ are both Gaussians, so their product spans twice the
decades either does. When $\sigma$ falls well below the typical neighbor
distance, the tail of the weight distribution drops below what float32 can
represent — and the training energy is evaluated in float32. Enough of a sampled
edge batch can then be *exactly* zero that the regularizer's gradient is exactly
zero, and

```python
configured = component_norms.supervised / component_norms.regularizer
```

in `_calibrate_regularizer_weight` raised `ZeroDivisionError`. The guard was
half-present already — the very next line tests `euclidean.regularizer > 0.0` —
but the anchor ratio was computed before it. Three changes:

- `training/engine.py` now defers calibration on a batch with no regularizer
  gradient, mirroring the existing "no supervised gradient" guard, instead of
  dividing by zero.
- `build_manifold_preserving_graph` reports
  `float32_underflow_edge_fraction` in its metadata and warns when it is
  non-zero, so an inert graph is visible at build time rather than as a crash
  several epochs later.
- the search space's `similarity_v` upper bound was lowered from 60 to 12
  (see the measurements in §9 — at $v \ge 20$ the graph starts dying).

**The alternating update dropped its edge weights.** In
`separate_optimizer_step_losses`, the attraction step computed
$\sum_e \lVert \tilde z_i - \tilde z_j\rVert^2$ and used the sampled `weights`
only for their count, so `contrastive_update_mode="alternating"` trained *any*
weighted graph as if it were binary. Latent for the shipped alternating config
(which is binary), live for both `edge_weighting="cosine"` and this new mode.
Now weighted, matching the joint path.

### 8.3 Open, by design — worth knowing before reading results

- **`normalized_laplacian: true` in the shipped config** is the LRML convention,
  not Ying et al.'s Equation (8). It also interacts badly with the weight range:
  `MIN_WEIGHTED_DEGREE = 1e-12` floors $\deg_i$ before $1/\sqrt{\deg_i}$, and
  with the paper's own $v=10$ that floor binds for real nodes (65 of a
  4,000-node Cars196 sample), silently changing their normalization rather than
  only guarding a division by zero. `false` — the paper's setting — avoids this
  entirely.
- **Uniform edge sampling over a wildly non-uniform weight distribution.** The
  sampler draws edges uniformly and the estimator is unbiased, but its variance
  scales with the weight spread. See §9.
- **`edge_weighting` and `gamma` are silently ignored** in this mode rather than
  rejected.

## 9. Measured on Cars196

Frozen DINOv2 ViT-B/14 features from `data/Cars196/backbone_cache`, $k=10$,
$h=\sigma$.

**Weight concentration** (4,000-node random subsample, 29,212 undirected edges).
"Effective edges" is $(\sum w)^2/\sum w^2$, the count of equally-weighted edges
that would carry the same variance:

| Graph | Weight range | Effective edges | Top 1% of edges carry |
| --- | --- | --- | --- |
| `lrml` binary | $\{1\}$ | 29,212 (100%) | 1.0% |
| `lrml` cosine, $\gamma=3$ | $[0.020,\,2]$ | 22,038 (75.4%) | — |
| manifold-preserving, $v=10$ | $[5.9\!\times\!10^{-24},\,2]$ | **99.6 (0.34%)** | **67.1%** |
| manifold-preserving, $v=3$ | $[2.9\!\times\!10^{-3},\,2]$ | 18,839 (64.5%) | 3.6% |

At the paper's own $v=10$ the regularizer is, in effect, a ~100-edge graph
wearing a 29,212-edge coat: the median edge weight is $6.8\times10^{-7}$ while
the top 0.1% of edges carry 38% of the total. $v$ is the single knob that
controls this, and it does so violently — it enters as $\sigma^{-2}$ inside an
exponential.

**Underflow** (full 8,054-image split):

| $v$ | $\sigma$ | median $w$ | float32 edges $\to 0$ | $P(\text{all 256 sampled edges} = 0)$ |
| --- | --- | --- | --- | --- |
| 1 | 1.601 | $9.0\times10^{-1}$ | 0.00% | 0 |
| 3 | 0.534 | $3.6\times10^{-1}$ | 0.00% | 0 |
| 10 | 0.160 | $2.9\times10^{-6}$ | 0.00% | 0 |
| 20 | 0.080 | $9.4\times10^{-24}$ | 4.85% (+5.7% denormal) | $\approx 0$ |
| 60 | 0.027 | $4.4\times10^{-195}$ | **99.77%** | **0.557** |

The old search space sampled $v \in [1, 60]$ log-uniformly, putting ~27% of
trials at $v \ge 20$ and ~10% at $v \ge 40$. At $v=60$ a calibration batch had a
coin-flip chance of containing no live edge at all — the crash. Note also that
$\min d = 0$ exactly on this data (duplicate images), so $\sigma = \max d / v$
and the paper's $\min D$ term contributes nothing here.

**Recommended range:** $v \le 12$. Below $v \approx 3$ the Gaussian is so flat
(weights within a factor of 2 of each other) that the mode degenerates towards
`lrml` binary; above $v \approx 12$ it degenerates towards a handful of
near-duplicate pairs and then towards nothing.

## 10. Configuration

```jsonc
"regularizer_params": {
  "mode": "manifold_preserving",  // selects Ying et al.'s graph
  "n_neighbors": 10,              // Section IV; also |N(i)| for the density
  "similarity_v": 10.0,           // sigma = min d + (max d - min d) / v
  "density_bandwidth": null,      // Parzen h; null reuses sigma
  "normalized_laplacian": true,   // Eq. (8) is false; see §8.3
  "graph_on": "all",
  "reduction": "torch_sum",
  "oracle_same_class_edges": false
}
```

Every other `LRMLRegularizer` option — `contrastive`, `graph_batch_size`,
`graph_batch_mode`, the refresh schedule, target-ratio calibration — behaves
exactly as documented in [`docs/lrml.md`](lrml.md), because this mode changes
only $\bar W$.

Graph diagnostics saved for this mode carry
`graph_kind="density_weighted_gaussian_knn"` plus `similarity_sigma`,
`density_bandwidth`, the `density_weight_{min,mean,max}` spread,
`edge_weight_{min,max}`, and `float32_underflow_edge_fraction`.
