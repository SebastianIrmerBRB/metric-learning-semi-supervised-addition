# LRML: Graph Construction, Batch Construction, and Deep Training

This document describes the LRML implementation in this repository, including
the exact CIFAR experiment selected by
[`configs/experiments/ssl/cifar_lrml.json`](../configs/experiments/ssl/cifar_lrml.json).
It separates three things that are easy to conflate:

1. the original linear Laplacian Regularized Metric Learning (LRML) objective;
2. this repository's nonlinear, stochastic deep adaptation of its Laplacian
   term; and
3. the two independent batch streams used to optimize that adaptation.

The shortest description of the default local algorithm is

$$
\boxed{
\mathcal J_t(\theta,\phi)
=
\alpha\,\mathcal L_{\mathrm{sup},t}(\theta,\phi)
+
\lambda\,\widehat{\mathcal R}_{G,t}(\theta)
}
$$

where a labeled, class-balanced batch supplies the supervised deep metric
loss and an independently sampled batch of graph edges supplies an unbiased
estimate of either the full graph energy or its global weighted mean. Under
the default `contrastive_update_mode="joint"`, both terms are evaluated in one
iteration, followed by one backward pass and one model-optimizer step. The
optional `"alternating"` mode instead makes the supervised, graph-attraction,
and random-pair-repulsion branches three ordered optimizer updates, matching
the material update boundary of Weston et al.'s Algorithm 1. LRML is a training
regularizer here; it does not create pseudo-labels, and the graph is not needed
at inference time.

## 1. Configuration Layers

The supplied JSON is an Optuna search space, not a complete runnable LRML
configuration. The full CIFAR recipe is assembled from three files:

| Layer | File | Role |
|---|---|---|
| Experiment | [`configs/experiments/class/cifar_lrml.json`](../configs/experiments/class/cifar_lrml.json) | Dataset protocol, label-budget scenario, loss/miner grid, epoch length, and links to the other two files |
| Base SSL method | [`configs/ssl_regularized_lrml.json`](../configs/ssl_regularized_lrml.json) | Selects `method="regularized"`, `regularizer="lrml"`, warmup, graph refresh policy, and fixed LRML defaults |
| HPO search | [`configs/experiments/ssl/cifar_lrml.json`](../configs/experiments/ssl/cifar_lrml.json) | Samples learning rates, loss weights, graph parameters, and both batch sizes |

The normal entry point for this exact recipe is conceptually:

```powershell
python main.py --experiment_config configs/experiments/class/cifar_lrml.json
```

Explicit command-line arguments can still override experiment-config values.
Using the supplied HPO JSON with a different experiment or base SSL config can
therefore produce a different dataset split, warmup, graph refresh schedule,
or fixed graph reduction.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $x_i$ | Input image or, in the original LRML paper, an input feature vector |
| $y_i$ | Known class label; it is available only for the labeled support set |
| $f_\theta$ | Deep embedding model (`DinoWrapper`) with trainable parameters $\theta$ |
| $z_i=f_\theta(x_i)\in\mathbb R^p$ | Current deep embedding |
| $N$ | Number of nodes in the selected graph pool |
| $k$ | Requested directed nearest neighbors per node (`n_neighbors`) |
| $W\in\mathbb R^{N\times N}$ | Symmetric, binary LRML adjacency matrix |
| $d_i=\sum_j W_{ij}$ | Degree of graph node $i$ |
| $D_G=\operatorname{diag}(d_1,\ldots,d_N)$ | Graph degree matrix |
| $L=D_G-W$ | Unnormalized graph Laplacian |
| $L_{\mathrm{sym}}=D_G^{-1/2}LD_G^{-1/2}$ | Symmetric-normalized graph Laplacian |
| $E=\{(i,j):i<j,\ W_{ij}>0\}$ | Undirected upper-triangle edge set |
| $M=|E|$ | Number of undirected graph edges |
| $B$ | Number of labeled examples in one supervised batch (`batch_size`) |
| $m$ | Examples per class in a supervised batch (`sampler_m`) |
| $b_G$ | Requested graph edges in one regularizer batch (`graph_batch_size`) |
| $b_t$ | Actual number of graph edges in regularizer batch $t$; normally $b_G$, but a permutation's final chunk can be smaller |
| $\alpha$ | `supervised_weight` |
| $\lambda$ | `regularizer_weight` |
| $\phi$ | Optional trainable parameters owned by a proxy/classification-style supervised loss |

The subscript $G$ on $D_G$ distinguishes the degree matrix from the original
paper's set of dissimilar constraints, denoted below by $\mathcal D$.

## 3. Original Linear LRML

The original method is from Hoi, Liu, and Chang, *Semi-Supervised Distance
Metric Learning for Collaborative Image Retrieval and Clustering* (2010).
It learns a single positive-semidefinite Mahalanobis matrix. It is not a neural
mini-batch algorithm.

### 3.1 Linear metric

Let $x_i\in\mathbb R^q$ and let

$$
U\in\mathbb R^{q\times r},
\qquad
A=UU^\top\succeq0.
$$

The learned squared distance is

$$
d_A^2(x_i,x_j)
=(x_i-x_j)^\top A(x_i-x_j)
=\left\|U^\top x_i-U^\top x_j\right\|_2^2.
$$

Thus LRML can be viewed either as learning the matrix $A$ or as learning the
linear embedding $z_i=U^\top x_i$.

### 3.2 Original graph

Let $\mathcal N_k(i)$ be the $k$ Euclidean nearest neighbors of $x_i$ in the
fixed input feature space. The paper constructs a binary symmetric-union graph:

$$
W_{ij}
=
\begin{cases}
1,&j\in\mathcal N_k(i)\ \text{or}\ i\in\mathcal N_k(j),\\
0,&\text{otherwise}.
\end{cases}
$$

With $D_G=\operatorname{diag}(W\mathbf 1)$ and $L=D_G-W$, the manifold
regularizer is

$$
\begin{aligned}
g(A)
&=\frac12\sum_{i=1}^N\sum_{j=1}^N
W_{ij}\left\|U^\top x_i-U^\top x_j\right\|_2^2\\
&=\operatorname{tr}\!\left(U^\top XLX^\top U\right)\\
&=\operatorname{tr}\!\left(XLX^\top A\right),
\end{aligned}
$$

where $X=[x_1,\ldots,x_N]\in\mathbb R^{q\times N}$. The factor $1/2$
removes the double count in the symmetric matrix. Equivalently,

$$
g(A)=\sum_{(i,j)\in E}W_{ij}\|z_i-z_j\|_2^2.
$$

The trace identity follows from

$$
\begin{aligned}
\operatorname{tr}(Z^\top LZ)
&=\sum_i d_i\|z_i\|_2^2-\sum_{i,j}W_{ij}z_i^\top z_j\\
&=\frac12\sum_{i,j}W_{ij}\|z_i-z_j\|_2^2,
\end{aligned}
$$

with $Z=[z_1^\top;\ldots;z_N^\top]$. The paper also recommends the
symmetric-normalized Laplacian

$$
L_{\mathrm{sym}}=D_G^{-1/2}(D_G-W)D_G^{-1/2}.
$$

Its edge form is

$$
\operatorname{tr}(Z^\top L_{\mathrm{sym}}Z)
=
\sum_{(i,j)\in E}W_{ij}
\left\|
\frac{z_i}{\sqrt{d_i}}-\frac{z_j}{\sqrt{d_j}}
\right\|_2^2.
$$

This is exactly why the local implementation multiplies every endpoint
embedding by $1/\sqrt{d_i}$ before computing an edge distance.

### 3.3 Full original objective

Let $\mathcal S$ and $\mathcal D$ be similar and dissimilar labeled pair sets.
The paper defines

$$
V_s(A)=\sum_{(i,j)\in\mathcal S}d_A^2(x_i,x_j),
\qquad
V_d(A)=-\sum_{(i,j)\in\mathcal D}d_A^2(x_i,x_j).
$$

The constrained LRML problem is

$$
\begin{aligned}
\min_{A\succeq0}\quad
&\operatorname{tr}(XLX^\top A)
+\gamma_s V_s(A)
+\gamma_d V_d(A)\\
\text{subject to}\quad&\log\det(A)\ge0.
\end{aligned}
$$

Because $V_d$ is negative, minimizing it pushes dissimilar pairs apart. The
log-determinant constraint prevents the trivial collapsed solution $A=0$.

Define the pair-scatter matrices

$$
C_S=\sum_{(i,j)\in\mathcal S}(x_i-x_j)(x_i-x_j)^\top,
\qquad
C_D=\sum_{(i,j)\in\mathcal D}(x_i-x_j)(x_i-x_j)^\top.
$$

Then all three terms are linear in $A$:

$$
B=XLX^\top+\gamma_s C_S-\gamma_d C_D,
\qquad
\mathcal J(A)=\operatorname{tr}(AB).
$$

The paper gives an SDP formulation and a faster log-determinant surrogate:

$$
\min_{A\succeq0}\operatorname{tr}(AB)-\mu\log\det(A).
$$

When $B\succ0$, stationarity gives

$$
B-\mu A^{-1}=0
\quad\Longrightarrow\quad
A^*=\mu B^{-1}.
$$

If $B$ is not positive definite, the paper proposes adding a ridge term:

$$
A^*=\mu\left(B+\gamma_I I\right)^{-1}.
$$

None of this SDP, matrix inversion, or explicit log-determinant machinery is
used by the deep implementation below. It is included to make clear which part
of original LRML was ported: the graph-Laplacian regularizer.

## 4. Deep LRML in This Repository

### 4.1 Replace the linear map with a network

The local adaptation replaces

$$
z_i=U^\top x_i
\quad\text{with}\quad
z_i=f_\theta(x_i).
$$

For a linear $f_\theta$, Euclidean distance in the output recovers the PSD
Mahalanobis matrix $A=UU^\top$. For nonlinear $f_\theta$, there is no single
global input-space matrix $A$; positive semidefiniteness is implicit in using
Euclidean distance in the learned embedding space.

`DinoWrapper` consists of a DINOv2 backbone and, when `feat_dim` is set, a
linear projection. Its ordinary output is L2-normalized:

$$
h_i=\operatorname{projection}(\operatorname{DINO}(x_i)),
\qquad
z_i=\frac{h_i}{\|h_i\|_2}.
$$

In the associated CIFAR experiment, omitted command-line settings retain the
defaults `dino_size="b"`, `backbone_tuning="full"`, `feat_dim=null`, and
`optim="adam"`. Consequently, the default model fine-tunes DINOv2-B and uses
its normalized 768-dimensional output without an extra learned projection.
Those defaults remain overridable.

### 4.2 Exact stochastic objective

At a post-warmup outer iteration $t$, joint mode minimizes

$$
\mathcal J_t(\theta,\phi)
=\alpha\,\ell_{\mathrm{DML}}
\left(\{(f_\theta(x_a),y_a):a\in\mathcal B^S_t\};\phi\right)
+\lambda\,\widehat{\mathcal R}_{G,t}(\theta).
$$

Here:

- $\mathcal B^S_t$ is a labeled M-per-class batch;
- $\ell_{\mathrm{DML}}$ is the selected supervised metric-learning loss;
- $\phi$ is present only for losses with learnable class weights or proxies;
- $\widehat{\mathcal R}_{G,t}$ uses an independent graph-edge batch; and
- unlabeled ground-truth labels are never passed to either term.

Alternating contrastive mode uses the same raw component estimators and weights
but applies them sequentially rather than differentiating this displayed sum;
section 9 writes that update explicitly.

The associated outer experiment runs separate scenarios for
`NTXentLoss:no_miner`, `MultiSimilarityLoss:MultiSimilarityMiner`,
`CircleLoss:no_miner`, `ArcFaceLoss:no_miner`, and
`ProxyAnchorLoss:no_miner`. Therefore there is deliberately no single expanded
formula for $\ell_{\mathrm{DML}}$ in the LRML configuration. Original LRML's
separate $\gamma_s V_s$ and $\gamma_d V_d$ terms are replaced by whichever of
these supervised losses is selected. The local $\alpha$ and $\lambda$ are not
numerically comparable to the paper's $\gamma_s$ and $\gamma_d$, especially
because the local graph term uses a mean reduction by default.

### 4.3 Shared embedding and graph normalization

Two normalization operations have different roles:

1. `DinoWrapper`, invoked through `forward_model_inputs`, L2-normalizes the
   embeddings returned to both losses and the graph builder. With unit vectors,
   $\|u-v\|_2^2=2-2u^\top v$, so maximum inner product and minimum Euclidean
   distance have identical rankings.
2. `normalized_laplacian=true` degree-scales endpoints by $1/\sqrt{d_i}$ in
   the graph energy. This is graph normalization, not vector L2 normalization.

`LRMLRegularizer.compute_loss` relies on this model-forward contract. The
FAISS graph helpers defensively L2-normalize a copy of their input, which is a
no-op for the standard model and keeps neighbor construction well defined for
other callers.

## 5. Graph Construction

Graph construction is implemented by `LRMLRegularizer` in
[`training/semi_supervised.py`](../training/semi_supervised.py) and `build_lrml_knn_graph` in
[`training/ssl/algorithms.py`](../training/ssl/algorithms.py).

### 5.1 Choose graph nodes

The labeled-support selection first partitions the current training subset
into positions $P_L$ and $P_U$. The LRML graph pool is

$$
P_G=
\begin{cases}
\operatorname{unique}(P_L\cup P_U),&\text{if `graph_on="all"`},\\
P_U,&\text{if `graph_on="unlabeled"`}.
\end{cases}
$$

Positions are sorted and assigned local graph IDs $0,\ldots,N-1$. Labels are
not returned by `LRMLGraphDataset`. Even when labeled nodes are included, $W$
is purely geometric and contains no same-class or different-class edges.

### 5.2 Extract a deterministic graph snapshot

At a graph rebuild, the model is temporarily put in evaluation mode and every
graph node is embedded without gradients using the deterministic feature
transform:

$$
q_i=f_{\bar\theta_e}(T_{\mathrm{eval}}(x_i)).
$$

$\bar\theta_e$ denotes the parameter snapshot at the start of epoch $e$.
The model invoked through the shared forward path returns a unit vector, and
FAISS receives it directly:

$$
\|q_i\|_2=1.
$$

`embedding_batch_size=32` controls only the throughput of this all-node
embedding pass. It is unrelated to both supervised `batch_size` and
`graph_batch_size`.

### 5.3 Directed exact kNN, then symmetric union

The implementation uses `faiss.IndexFlatIP`, so the search is exact inner
product search. It requests $k+1$ candidates, removes the self-hit, and keeps

$$
k'=\min(k,N-1)
$$

non-self neighbors per node. Define the directed adjacency

$$
A^{\rightarrow}_{ij}=\mathbf 1[j\in\mathcal N_{k'}(i)].
$$

The stored LRML graph is

$$
W_{ij}=\mathbf 1[A^{\rightarrow}_{ij}+A^{\rightarrow}_{ji}>0],
\qquad W_{ii}=0.
$$

It is therefore symmetric, binary, and sparse. Its degrees and edge count are

$$
d_i=\sum_jW_{ij},
\qquad
M=\frac12\sum_{i,j}W_{ij}=\frac{\operatorname{nnz}(W)}2.
$$

Because there are $Nk'$ directed arcs before symmetrization,

$$
\frac{Nk'}2\le M\le Nk',
$$

with the lower endpoint corresponding to fully mutual neighbors and the upper
endpoint to no mutual directed pairs.

### 5.4 Ying et al. manifold-preserving mode

`regularizer_params.mode="manifold_preserving"` keeps the LRML graph lifecycle
and edge sampler but replaces the binary adjacency with the regularizer from
Ying et al. (2018). For directed neighbors it constructs

$$
W^\rightarrow_{ij}
=
\beta_i
\exp\!\left(-\frac{\lVert q_i-q_j\rVert_2^2}{2\sigma^2}\right),
\qquad j\in\mathcal N_k(i),
$$

where $\beta_i$ is a Gaussian Parzen density divided by the maximum density.
Section IV of that paper defines

$$
\sigma
=
\min(D)+\frac{\max(D)-\min(D)}{v}.
$$

The implementation obtains both extrema exactly on the normalized graph
features without materializing a dense pairwise-distance matrix. The default
is the paper's $v=10$. Because the paper does not report the Parzen bandwidth
$h$, `density_bandwidth=null` uses $h=\sigma`; a positive numeric value makes
that otherwise missing experimental choice explicit.

The sampler requires symmetric storage, so it stores
$W=W^\rightarrow+(W^\rightarrow)^\top$. This does not alter Equation (7):

$$
\sum_{i,j}W^\rightarrow_{ij}d_{ij}^2
=
\sum_{i<j}
\left(W^\rightarrow_{ij}+W^\rightarrow_{ji}\right)d_{ij}^2.
$$

The mode defaults to ten neighbors and `normalized_laplacian=false`, matching
the paper's experiment and unnormalized Equation (8). The ready-to-use SSL
configuration is
[`configs/ssl_methods/lrml_manifold_preserving.json`](../configs/ssl_methods/lrml_manifold_preserving.json).
The standard LRML-only `edge_weighting` and cosine-exponent `gamma` settings
are not used in this mode.

Only the semisupervised graph regularizer is ported. As in the repository's
standard LRML adapter, the configured supervised DML loss replaces the linear
labeled objective. A neural embedding also replaces the explicit SPD matrix,
so the paper's Riemannian exponential-map update is not applicable to the
network optimizer.

### 5.5 Select the Laplacian

The supplied HPO file fixes `normalized_laplacian` to `true`, so the active
full-graph target is

$$
\mathcal R_G(\theta)
=\operatorname{tr}(Z^\top L_{\mathrm{sym}}Z)
=\sum_{(i,j)\in E}W_{ij}
\left\|
\frac{z_i}{\sqrt{d_i}}-\frac{z_j}{\sqrt{d_j}}
\right\|_2^2.
$$

If it were false, the same code would use endpoint scale $1$ and target
$\operatorname{tr}(Z^\top LZ)$.

### 5.6 Refresh schedule and stop-gradient boundary

The canonical config has `warmup_epochs=1`, `update_mode="every_n_epochs"`, and
`update_interval_epochs=1`:

- epoch 0 performs labeled-only supervised training and does not build
  an LRML graph;
- at the start of epoch 1, the graph is built from the then-current model;
- at the start of every later epoch, it is rebuilt from the latest model; and
- within an epoch, $W$, $D_G$, and the endpoint degree scales remain fixed
  while $\theta$ changes on every optimizer step.

Neighbor selection and degree computation are discrete preprocessing. Backprop
does not differentiate through

$$
W=G(f_{\bar\theta_e}(X)).
$$

In optimization notation, the epoch uses

$$
\nabla_\theta
\mathcal R_G(\theta;\operatorname{stopgrad}(W_e,D_{G,e})).
$$

This is an adaptive alternating graph procedure. Setting `update_mode="once"`
and constructing $W$ from a fixed feature representation would be closer to
the original paper's fixed input-space graph.

#### Sample-scoped refreshes

`update_mode="every_n_samples"` schedules the rebuild on consumed training
samples instead of on epoch boundaries. It is off unless the config selects it.

An epoch is not a fixed amount of training here: `length_before_new_iter` is
resolved from the fold's own labeled + unlabeled pool, so one `every_epoch`
rebuild covers roughly 46k samples of drift on semi-iNat and a fraction of that
on Cars196. The same configured cadence therefore lets $W$ go much staler on the
larger dataset. This mode fixes the amount of drift instead:

```json
{
  "update_mode": "every_n_samples",
  "update_interval_samples": 10000
}
```

The interval is in samples rather than steps so it stays independent of the
batch size a trial draws. Each run converts it once,
`interval_steps = max(1, round(update_interval_samples / batch_size))`, and logs
the result:

```
SSL refresh schedule: every 10000 training samples = 156 steps at batch_size=64
(721 steps per epoch over 46172 samples, 4.62 rebuilds per epoch)
```

When the interval is shorter than an epoch, the rebuild happens between two
steps of that epoch: the epoch keeps the batch budget its first loader reported,
and only which graph the remaining steps train against changes. The counter runs
across epoch boundaries, so an interval longer than an epoch simply rebuilds
every few epochs at the right sample count. Warm-up epochs, which build no
graph, do not count toward it.

Two combinations are rejected rather than silently degraded:
`graph_batch_mode="in_batch"`, which already builds one graph per step, and
`graph_diagnostics_mode="save"`, whose artifact bundles are named per epoch and
so cannot record more than one graph per epoch.

## 6. Supervised Batch Construction

The supervised stream is built by `make_train_loader` in
[`utils/`](../utils) using `pytorch_metric_learning.samplers.MPerClassSampler`.
It sees only the labeled support dataset.

For a sampled `batch_sampler` value `"B:m"`:

$$
\text{classes per supervised batch}=\frac{B}{m},
\qquad
|\mathcal B^S_t|=B.
$$

The sampler shuffles the available class IDs, chooses $B/m$ classes, and draws
exactly $m$ indices from each. It samples with replacement inside a class only
when that class has fewer than $m$ available examples. The code validates

$$
B>0,\qquad m>0,\qquad B\bmod m=0,
\qquad mC_L\ge B,
$$

where $C_L$ is the number of labeled classes.

Every attached choice has $m=8$:

| HPO value | $B$ examples | $m$ per class | Classes per batch $B/m$ |
|---|---:|---:|---:|
| `16:8` | 16 | 8 | 2 |
| `32:8` | 32 | 8 | 4 |
| `64:8` | 64 | 8 | 8 |
| `128:8` | 128 | 8 | 16 |
| `256:8` | 256 | 8 | 32 |

The associated CIFAR experiment requests `length_before_new_iter=22800`.
`MPerClassSampler` rounds this down to a complete multiple of $B$, so the
supervised-update/outer-iteration count $S$ is:

| $B$ | Sampled labeled occurrences per epoch | Outer iterations $S$ |
|---:|---:|---:|
| 16 | 22,800 | 1,425 |
| 32 | 22,784 | 712 |
| 64 | 22,784 | 356 |
| 128 | 22,784 | 178 |
| 256 | 22,784 | 89 |

These are sampled occurrences, not distinct labeled images. In the HPO
training split there are only 1,200 distinct labeled images, so the sampler
revisits them across an epoch. Changing $B$ changes not only batch composition
but also the number of outer iterations and, consequently, the number of graph
batches consumed per epoch. That is $S$ model-optimizer steps in joint mode and
$3S$ in alternating mode.

## 7. Graph-Edge Batch Construction

The regularizer stream is implemented by `GraphEdgeBatchSampler`,
`LRMLGraphDataset`, and `collate_graph_edge_batch` in
[`training/ssl/data.py`](../training/ssl/data.py).

### 7.1 Sample edges, not nodes

The sampler stores only the upper-triangle edge list

$$
E=\{(i,j):i<j,\ W_{ij}>0\}.
$$

It generates a random permutation of all $M$ edges and takes consecutive
chunks of at most $b_G$ edges. Once a permutation is exhausted, it creates a
new permutation. The LRML loader requests exactly

$$
\text{number of graph batches}=S=\text{number of supervised batches}.
$$

Thus every outer training iteration receives one batch from each stream. Sampling is
without replacement within one graph pass and begins a new shuffled pass only
after all edges have been consumed.

### 7.2 Deduplicate endpoints

A raw edge batch has up to $2b_t$ endpoint occurrences. If edges share a node,
the sampler loads that node only once. For example,

$$
\{(0,1),(0,2)\}
$$

loads nodes $[0,1,2]$ and carries the local edge map

$$
[(0,1),(0,2)].
$$

The collated batch is

```text
(images_of_unique_nodes, global_graph_node_ids, local_edge_indices)
```

Only the explicitly sampled edges are evaluated. If nodes 1 and 2 are also
adjacent in the full graph, edge $(1,2)$ is not accidentally added. A shared
endpoint uses one augmented image and one forward result for all of its edges
in a joint-mode batch, so its incident edge gradients accumulate naturally.
Alternating mode expands the selected endpoints back to pair occurrences before
its attraction forward; this preserves per-pair multiplicity in batch-dependent
training layers while still taking one batched attraction update.

Graph construction used deterministic evaluation views, but—unless frozen
feature caching is active—the edge batch loads the normal stochastic training
transform. The graph therefore says which underlying examples are neighbors,
while the differentiable energy is evaluated on their current augmented views.

When the optional repulsion of section 8.1 is enabled, the sampler appends
`random_nodes_per_batch` further nodes drawn uniformly from all $N$ graph nodes,
*after* the endpoints and deliberately **not** deduplicated against them. They
belong to no sampled edge, so the endpoints keep local indices
$0,\dots,U-1$ and the local edge map stays valid; the regularizer reads the
split as "anchors in front, uniform partner pool behind". These extra nodes are
loaded like any other. Joint mode forwards the complete deduplicated endpoint
and partner batch once; alternating mode forwards only the realized expanded
random pairs in its fresh repulsion pass. Either way, the extra loads buy a
partner drawn from the graph pool rather than from the current endpoints. The
pool is uniform over all $N$ nodes; the $W_{in}=0$ restriction of section 8.1 is
applied to the draw itself, not to what the sampler loads.

### 7.3 Independent units

`batch_size` and `graph_batch_size` deliberately have different units:

| Setting | Unit | Controls |
|---|---|---|
| `batch_size=B` | labeled examples | Supervised forward, class composition, and outer iterations per epoch |
| `sampler_m=m` | labeled examples per class | Positive/class-balanced structure inside the supervised batch |
| `graph_batch_size=b_G` | undirected graph edges | Number of pairwise LRML terms in each graph-branch update |
| `embedding_batch_size` | graph-node images | Throughput only for the no-gradient graph-rebuild pass |

The graph forward sees at most $2b_G$ distinct nodes and often fewer after
deduplication, plus the uniform partner pool of section 8.1 when the repulsion
is on. `graph_batch_size=null` would make $b_G$ numerically equal to $B$, but
the supplied HPO file overrides it with one of 16, 32, 64, or 128.

## 8. Sampled Laplacian Estimator

Let an undirected edge $e=(i,j)$ have weight $w_e=W_{ij}$ and define

$$
s_i=
\begin{cases}
d_i^{-1/2},&\text{normalized Laplacian},\\
1,&\text{unnormalized Laplacian},
\end{cases}
\qquad
\varepsilon_e(\theta)=w_e\|s_i z_i-s_j z_j\|_2^2.
$$

The full graph sum is

$$
\mathcal R_{\mathrm{sum}}(\theta)=\sum_{e\in E}\varepsilon_e(\theta).
$$

Let $\mathcal B^G_t\subset E$ be a uniform edge subset of actual size $b_t$.
For `reduction="sum"`, the code returns

$$
\widehat{\mathcal R}_{\mathrm{sum},t}
=\frac{M}{b_t}\sum_{e\in\mathcal B^G_t}\varepsilon_e.
$$

Every edge has inclusion probability $b_t/M$, hence

$$
\begin{aligned}
\mathbb E\left[\widehat{\mathcal R}_{\mathrm{sum},t}\right]
&=\frac{M}{b_t}\sum_{e\in E}
\Pr(e\in\mathcal B^G_t)\varepsilon_e\\
&=\sum_{e\in E}\varepsilon_e
=\mathcal R_{\mathrm{sum}}.
\end{aligned}
$$

For `reduction="mean"`, define the global upper-triangle graph weight

$$
W_+=\sum_{e\in E}w_e=\frac12\sum_{i,j}W_{ij}.
$$

The estimator is

$$
\widehat{\mathcal R}_{\mathrm{mean},t}
=\frac{M}{b_tW_+}
\sum_{e\in\mathcal B^G_t}\varepsilon_e,
$$

and therefore

$$
\mathbb E\left[\widehat{\mathcal R}_{\mathrm{mean},t}\right]
=\frac{\mathcal R_{\mathrm{sum}}}{W_+}.
$$

LRML's graph is binary, so $w_e=1$, $W_+=M$, and the configured mean estimator
simplifies to

$$
\boxed{
\widehat{\mathcal R}_{\mathrm{mean},t}
=\frac1{b_t}\sum_{e\in\mathcal B^G_t}
\|s_i z_i-s_j z_j\|_2^2
}.
$$

This stable mean prevents the regularizer's basic scale from growing linearly
with graph size. It is not the paper's full trace; it is the trace divided by
the total binary edge weight. The HPO-selected $\lambda$ scales this mean.

### 8.1 Optional Non-Neighbor Repulsion

The Laplacian term is attraction only: it is minimized by mapping every node to
one point. Nothing in LRML's unlabeled term prevents that; only the supervised
loss does. `contrastive` optionally adds the repulsive branch of the
semi-supervised embedding loss of Weston et al. (2012), *Deep Learning via
Semi-Supervised Embedding*, eq. (3), which is the margin loss of Hadsell et al.
(2006):

$$
L(f_i,f_j,W_{ij})=
\begin{cases}
\|f_i-f_j\|_2^2,&W_{ij}=1,\\
\max\left(0,\;m-\|f_i-f_j\|_2\right)^2,&W_{ij}=0.
\end{cases}
$$

The $W_{ij}=1$ branch is exactly the unnormalized Laplacian energy already
computed, so only the $W_{ij}=0$ branch is added. With $\mathcal P_t$ the
random pairs drawn at step $t$, the regularizer returns

$$
\widehat{\mathcal R}_t(\theta)
+\rho_t\sum_{(i,j)\in\mathcal P_t}\max\left(0,m-\|z_i-z_j\|_2\right)^2,
$$

where $\rho_t$ is the same per-pair scaling the attractive term uses:
$1/|\mathcal P_t|$ for `reduction="mean"` and $M/|\mathcal P_t|$ for
`reduction="sum"`. The repulsion carries no weight of its own, which is the
paper's own arrangement: eq. (9) has a single $\lambda$, and Algorithm 1 spends
it on both branches,

```
Make a gradient step for  λ L(g(x_i), g(x_j), 1)
Make a gradient step for  λ L(g(x_i), g(x_n), 0)
```

so $\lambda$ — `regularizer_weight` — scales both branches and nothing
rebalances them against each other. With the default pair count, one random
pair is drawn per evaluated edge and anchored on that edge. `contrastive_pairs`
does not change the expected branch scale: $\rho_t$ divides by the realized pair
count, so drawing more pairs lowers the repulsion estimator's variance without
raising its weight.

The optimizer treatment depends on `contrastive_update_mode`:

- `joint` (the default) adds attraction and repulsion to the supervised scalar,
  calls `backward()` once, and takes one model-optimizer step. In this mode,
  `regularizer_target_ratio` or GradNorm may weight the simultaneous objective.
- `alternating` first takes the supervised update, then freshly forwards and
  steps the attraction batch, then draws the anchored random pairs, freshly
  forwards them at the post-attraction weights, and takes the repulsion step.
  Both graph updates use the same fixed `regularizer_weight`. Target-ratio
  calibration and GradNorm are rejected because their measurements assume all
  component gradients exist at one parameter point and in one backward graph.

The alternating path restores the update boundary that matters for Adam
moments, weight decay, and training-mode state. It is still a batched analogue
of Algorithm 1: the paper samples one supervised example, one neighbor pair,
and one random pair, whereas the local path averages one mini-batch in each of
the three updates.

Notes on the term as implemented by `LRMLRegularizer`'s joint and alternating
paths:

- The hinge is on the Euclidean distance, not on its square, so $m$ is a
  distance in the embedding space. Embeddings are L2-normalized, so distances
  never exceed 2; a margin above 2 can never be satisfied and is warned about at
  construction. The default $m=1$ corresponds to a 60° angle.
- The repulsion reads the plain embeddings $z_i$, not the degree-scaled
  $s_iz_i$. The $1/\sqrt{d_i}$ factor belongs to the normalized Laplacian, while
  the margin is a distance in the embedding space itself.
- Consequently, `normalized_laplacian=true` is faithful to Hoi et al.'s
  normalized LRML variant but is not literally Weston's positive branch:
  Weston uses $\|z_i-z_j\|^2$, not
  $\|z_i/\sqrt{d_i}-z_j/\sqrt{d_j}\|^2$. Use
  `normalized_laplacian=false` for a paper-oriented Weston ablation.
- Setting `contrastive` to `false`, the default, restores the pure Laplacian
  regularizer exactly, including its empty batch diagnostics.
- Eq. (3)'s hinge is on the plain distance, $\max(0,m-\|f_i-f_j\|_2)^2$: in the
  chapter's own typesetting the $2$ inside the hinge sits *below* the baseline,
  as $W_{ij}$'s $ij$ does, so it is the norm's subscript and not an exponent —
  eq. (1) writes its squared norm with a raised $2$ instead. Its prose agrees
  ("dissimilar ones … have a distance of at least $m$"), and so does the loss it
  credits, Hadsell et al. (2006). So $m$ is a distance and the diameter warning
  is against 2 rather than 4.

#### Pair sampling

Algorithm 1 takes **two pairs per step, sharing their anchor**:

```
Pick a random pair of neighbors x_i, x_j.
Make a gradient step for  λ L(g(x_i), g(x_j), 1)
Pick a random unlabeled example x_n.
Make a gradient step for  λ L(g(x_i), g(x_n), 0)
```

so $(x_i,x_j)$ is an edge of the graph and $(x_i,x_n)$ is the same $x_i$ against a
random draw. The paper is loose about the asymmetry — the positive step says "a
random pair of *neighbors*", the negative one only "a random unlabeled example"
— but eq. (3) is not: its third argument is $W_{in}$, and the repulsion is the
$W_{in}=0$ branch. Drawing an $x_n$ the graph joins to $x_i$ and pushing it away
is training the pair on the wrong branch,

$$
W_{in}=1\quad\text{trained as}\quad L(f_i,f_n,0),
$$

against the very edge the attractive step pulls together.

`_sample_repulsion_pairs` therefore draws the partner uniformly from the pool
*restricted to* the anchor's non-neighbors: it tests each draw against $W$,
redraws the blocked ones up to `PARTNER_REDRAW_ATTEMPTS` times, and drops the
pair if the anchor exhausts them. A draw landing on the anchor itself is blocked
the same way; $L(f_i,f_i,0)$ is the degenerate case of the same mistake, a
constant $m^2$ with no gradient. How often this fires is set by the pool, not by
the paper: a partner drawn from all $N$ nodes is a neighbor with probability
$d_i/N$ — negligible, which is why the unrestricted draw survives at whole-pool
scale — while `graph_batch_mode="in_batch"` draws from one batch, where the same
ratio is $d_i/|\text{batch}|$ and percent-scale.

The anchors are the endpoints of the step's sampled edges, one edge per pair at
the default count, with a coin flip deciding which endpoint plays $x_i$ (the
graph stores each edge as (lower node, higher node), which is an artifact of
storage, not of the sampling).

- Under `graph_batch_mode="global"` the partner is a uniform draw over all $N$
  graph nodes, streamed into the batch by `GraphEdgeBatchSampler` (section 7.2).
  The cost is one extra image load per pool draw. Joint mode includes those
  nodes in its one graph forward; alternating mode expands the realized pairs
  into one fresh batched repulsion forward.
- Under `graph_batch_mode="in_batch"` only the step's own labeled and unlabeled
  samples are embedded, so the pool is that batch and the anchors are the
  endpoints of the per-batch kNN graph's edges.
- Under `graph_on="all"` the pool contains the labeled nodes as well, matching
  the node set the attractive term runs over. `graph_on="unlabeled"` reduces it
  to the unlabeled pool the paper describes.
- The realized pair count equals the requested one unless an anchor's pool
  holds nothing but its own neighbors, which only a batch-sized pool realistically
  does. It is logged as `train/lrml/contrastive_pairs` next to
  `train/lrml/contrastive_requested_pairs` and
  `train/lrml/contrastive_partner_pool` (the number of nodes the partner was
  drawn from, before the $W_{in}=0$ restriction). Runs with batch diagnostics
  enabled also get `train/lrml/contrastive_active_pairs` (pairs inside the
  margin) and `train/lrml/contrastive_term`, both of which read device memory and
  are therefore skipped otherwise.

Configuration keys, all under `method_params.regularizer_params`:

| Key | Default | Meaning |
| --- | --- | --- |
| `contrastive` | `false` | Enables eq. (3)'s $W_{ij}=0$ branch at the shared $\lambda$; there is no separate weight |
| `contrastive_margin` | `1.0` | $m$ of eq. (3), a distance in the L2-normalized embedding space |
| `contrastive_pairs` | `null` | Random pairs drawn per step, which is also the number of pool nodes added to each graph batch; `null` matches that step's edge count |
| `contrastive_update_mode` | `"joint"` | `"joint"` keeps one combined update; `"alternating"` takes supervised → attraction → repulsion as three real model-optimizer updates |

```json
{
  "method": "regularized",
  "method_params": {
    "regularizer": "lrml",
    "regularizer_params": {
      "n_neighbors": 6,
      "contrastive": true,
      "contrastive_margin": 1.0,
      "contrastive_update_mode": "alternating"
    }
  }
}
```

Alternating mode currently requires `graph_batch_mode="global"` and fixed
`supervised_weight`/`regularizer_weight` values. In particular, it cannot be
overlaid directly on [`configs/ssl_methods/lrml.json`](../configs/ssl_methods/lrml.json),
because that canonical Hoi-oriented config enables GradNorm. A runnable fixed-
weight example is
[`configs/ssl_methods/lrml_contrastive_alternating.json`](../configs/ssl_methods/lrml_contrastive_alternating.json).
That example deliberately retains the repository's normal 6-NN normalized,
dynamic graph for a controlled local ablation; it is not a literal Weston
recipe. For the latter, also use `normalized_laplacian=false`, the paper's 10-NN
experimental anchor, a fixed graph, and the reported five supervised-only
epochs.

## 9. Gradients and the Deep-Learning Update

For one sampled edge $e=(i,j)$, let

$$
\delta_e=s_i z_i-s_j z_j
$$

and let $c_t=M/b_t$ for `sum` or $c_t=M/(b_tW_+)$ for `mean`. Its contributions
to the embedding gradients are

$$
\frac{\partial\widehat{\mathcal R}_{G,t}}{\partial z_i}
=2c_tw_es_i\delta_e,
\qquad
\frac{\partial\widehat{\mathcal R}_{G,t}}{\partial z_j}
=-2c_tw_es_j\delta_e.
$$

Contributions add when a node occurs in several sampled edges. For the full
unnormalized graph this recovers

$$
\nabla_Z\operatorname{tr}(Z^\top LZ)=2LZ,
$$

and for the symmetric-normalized graph it recovers

$$
\nabla_Z\operatorname{tr}(Z^\top L_{\mathrm{sym}}Z)=2L_{\mathrm{sym}}Z.
$$

Backpropagation applies the model Jacobian:

$$
\nabla_\theta\mathcal J_t
=\alpha\nabla_\theta\mathcal L_{\mathrm{sup},t}
+\lambda\sum_i
\left(\frac{\partial z_i}{\partial\theta}\right)^\top
\frac{\partial\widehat{\mathcal R}_{G,t}}{\partial z_i}.
$$

Because `DinoWrapper` returns $z_i=h_i/\|h_i\|_2$, the graph gradient also
passes through the normalization Jacobian

$$
\frac{\partial z_i}{\partial h_i}
=\frac1{\|h_i\|_2}\left(I-z_iz_i^\top\right).
$$

In `contrastive_update_mode="joint"`, the training loop computes the supervised
and graph forwards under the same autocast context, adds the scalar losses,
calls `backward()` once, and steps the model optimizer once.

In `"alternating"` mode there is no single gradient $\nabla\mathcal J_t$ at one
parameter point. Writing `Step` for one call to the configured optimizer, the
outer iteration is instead

$$
\begin{aligned}
\theta_{t,1} &= \operatorname{Step}
  (\theta_{t,0},\;\alpha\mathcal L_{\mathrm{sup},t}),\\
\theta_{t,2} &= \operatorname{Step}
  (\theta_{t,1},\;\lambda\widehat{\mathcal R}_{\mathrm{attr},t}),\\
\theta_{t,3} &= \operatorname{Step}
  (\theta_{t,2},\;\lambda\widehat{\mathcal R}_{\mathrm{rep},t}).
\end{aligned}
$$

The attraction forward is evaluated at $\theta_{t,1}$ and the repulsion forward
at $\theta_{t,2}$; three calls to `backward()` followed by only one call to
`step()` would not have this property. A proxy/classification-style supervised
loss owns a second optimizer for $\phi$ and uses `classifier_lr`; that criterion
optimizer is stepped only during the supervised branch. The model learning-rate
scheduler advances once after the complete three-update outer iteration, so all
three model updates use that iteration's learning rate. The logged batch loss is
the detached sum of the three sequentially observed component values; it is a
diagnostic aggregate, not a jointly differentiated objective.

## 10. End-to-End Epoch Algorithm

The implemented training procedure can be summarized as follows:

```text
Build a reproducible labeled support P_L and hidden-label pool P_U.
Build one labeled M-per-class DataLoader from P_L.
Build an LRMLGraphDataset over P_L union P_U, or only P_U.

For epoch e = 0, ..., E - 1:
    if e < warmup_epochs:
        for each labeled supervised batch:
            J = supervised loss
            backward; optimizer step
        continue

    Embed every graph node with eval transform and the start-of-epoch model.
    Construct exact directed kNN; symmetrize to binary W.
    Compute degrees, edge list, and normalized-Laplacian endpoint scales.
    Shuffle the M undirected edges.

    for t = 1, ..., number of supervised batches S:
        draw one B-example, m-per-class labeled batch
        take one b_G-edge graph chunk and deduplicate its endpoints
        compute supervised loss L_sup,t
        if contrastive_update_mode == "alternating":
            backward alpha * L_sup,t; model optimizer step
            freshly forward the expanded sampled neighbor pairs
            backward lambda * R_hat_attr,t; model optimizer step
            draw the anchored random non-neighbor partners
            freshly forward the expanded random pairs
            backward lambda * R_hat_rep,t; model optimizer step
        else:
            compute scaled edge estimator R_hat_G,t
            J_t = alpha * L_sup,t + lambda * R_hat_G,t
            backward once; model optimizer step once
```

`CombinedTrainingLoader` performs the one-to-one pairing. Its length is the
supervised loader length, so the supervised stream defines the number of outer
iterations. The graph sampler is explicitly created with the same number of
batches. The model therefore receives one optimizer update per outer iteration
in joint mode and three in alternating mode.

## 11. Exact CIFAR Experiment Numbers

The associated experiment uses `dataset_protocol="cifar100_fine_class_disjoint"`.
The protocol pools CIFAR-100's official splits, reserves 60 fine classes
(36,000 images) for development, and holds out 40 fine classes for final test.
With `cv_k=1` and `cv_mode="superclass_balanced_group_kfold"`, the HPO training
run makes an 80/20 class holdout inside development:

```text
48 training fine classes * 600 images = 28,800 training images
12 validation fine classes * 600      =  7,200 validation images
```

The outer scenario overwrites the base label settings with
`class_subset_k_shot`, class fraction 1.0, and 25 labeled examples per selected
class. During HPO this gives

$$
|P_L|=48\cdot25=1{,}200,
\qquad
|P_U|=28{,}800-1{,}200=27{,}600.
$$

Therefore:

| `graph_on` | HPO graph nodes $N$ |
|---|---:|
| `all` | 28,800 |
| `unlabeled` | 27,600 |

`val_mode="match_train"` downsamples the validation evaluation set to match
the 1,200-example labeled support size. After HPO, `final_test_after_hpo=true`
trains the selected configuration on all 60 development classes. Its new
support has $60\cdot25=1{,}500$ labeled images and 34,500 unlabeled images, so
the final graph has either 36,000 or 34,500 nodes.

The supplied HPO file runs 60 random-search trials per fixed outer
loss/miner scenario and maximizes best validation MAP@R. Its spaces are:

| Parameter | Search |
|---|---|
| Model learning rate $\eta$ | log-uniform $[10^{-7},10^{-5}]$ |
| Proxy/classifier learning rate | log-uniform $[10^{-4},10^2]$ when the selected loss owns trainable proxy/class weights |
| $\alpha$ (`supervised_weight`) | log-uniform $[0.1,2.0]$ |
| $\lambda$ (`regularizer_weight`) | log-uniform $[10^{-3},10]$ |
| $k$ (`n_neighbors`) | $\{3,6,10,15,25\}$ |
| $b_G$ (`graph_batch_size`) | $\{16,32,64,128\}$ edges |
| Laplacian | symmetric-normalized only |
| Graph node pool | `all` or `unlabeled` |
| Supervised batch | `16:8`, `32:8`, `64:8`, `128:8`, or `256:8` |

The following relevant base settings are fixed rather than searched:

| Setting | Fixed value |
|---|---|
| Warmup | 5 labeled-only epochs |
| Graph refresh | every epoch after warmup |
| Graph reduction | global binary-edge mean |
| Neighbor-search L2 normalization | enabled |
| Graph-extraction batch | 32 nodes |
| Maximum unlabeled samples | no cap |
| Pseudo-labeling | none |

## 12. What Is and Is Not Faithful to Original LRML

The following parts are mathematically faithful adaptations:

- the binary symmetric-union kNN adjacency;
- the unnormalized and symmetric-normalized Laplacian edge energies;
- the identity between the edge sum and the Laplacian trace;
- inclusion of unlabeled examples through label-free graph geometry; and
- direct uniform edge sampling with inverse-inclusion scaling.

The following parts make this a deep LRML variant rather than a reproduction
of the original solver:

- $U^\top x$ is replaced by nonlinear $f_\theta(x)$;
- the original similar/dissimilar squared-distance terms are replaced by a
  configurable supervised DML loss;
- the graph is normally rebuilt from learned embeddings every epoch instead of
  remaining fixed in the original input feature space;
- the implementation optimizes with stochastic gradient descent/Adam rather
  than SDP or matrix inversion;
- there is no explicit $A$, PSD constraint, log-determinant constraint, or
  closed-form inverse;
- `reduction="mean"` optimizes trace divided by total graph weight, not the
  paper's raw trace;
- supervised and graph losses are combined in the same optimizer step by
  default; `contrastive_update_mode="alternating"` provides three ordered,
  batched updates for the Weston branch, but not the paper's one-example/one-pair
  stochastic granularity; and
- `contrastive` can add a repulsive term (section 8.1) that belongs to
  Weston et al. (2012) rather than to LRML. It is off by default, so the
  configured default remains the plain Laplacian regularizer.

The graph energy by itself has a constant-embedding collapse solution. L2
normalization prevents collapse to the zero vector but does not prevent every
sample from becoming the same unit vector. In the supplied search space,
$\alpha>0$, so the supervised loss is the main anti-collapse force; the optional
random-pair repulsion of section 8.1 is a second one that acts on the
unlabeled term itself.

One implementation boundary is also worth recording: `IndexFlatIP` relies on
the shared forward path returning normalized vectors. Under that contract,
inner-product and Euclidean rankings are equivalent. Alternative models must
retain that contract for the graph to keep its Euclidean-neighbor interpretation.

## 13. Source Map and Verification

- [`training/semi_supervised.py`](../training/semi_supervised.py): `LRMLRegularizer`, sampled
  energy scaling, graph lifecycle, and combined loss.
- [`training/ssl/algorithms.py`](../training/ssl/algorithms.py): exact FAISS kNN, symmetric
  binary adjacency construction, and Ying et al.'s density-weighted Gaussian graph.
- [`training/ssl/data.py`](../training/ssl/data.py): graph-node dataset, direct edge sampler,
  endpoint deduplication, and paired loader.
- [`training/ssl/embeddings.py`](../training/ssl/embeddings.py): deterministic no-gradient graph
  embedding extraction.
- [`training/engine.py`](../training/engine.py): warmup, loader
  selection, loss combination, backpropagation, and optimizer steps.
- [`models/retrieval_model.py`](../models/retrieval_model.py): DINOv2 embedding network and
  output L2 normalization.
- [`utils/`](../utils): supervised M-per-class loader and retrieval
  evaluation.
- [`tests/test_lrml.py`](../tests/test_lrml.py): graph rebuild cadence, edge
  coverage, endpoint deduplication, independent batch sizes, trace estimator,
  absence of incidental edges, and empirical joint-versus-alternating optimizer
  step/forward ordering.
- [`tests/test_lrml_manifold_preserving.py`](../tests/test_lrml_manifold_preserving.py):
  paper-mode defaults, density/Gaussian weights, and equivalence to Equation (7).
- [`docs/deep_lrml_sampling.md`](deep_lrml_sampling.md): shorter design note
  focused specifically on the independent graph-edge batch contract.

Primary reference:

- Steven C. H. Hoi, Wei Liu, and Shih-Fu Chang. 2010.
  [*Semi-Supervised Distance Metric Learning for Collaborative Image Retrieval
  and Clustering*](https://www.ee.columbia.edu/ln/dvmm/publications/10/TOMCCAP10_ssml.pdf),
  ACM TOMCCAP 6(3), Article 18. DOI:
  [10.1145/1823746.1823752](https://doi.org/10.1145/1823746.1823752).
- Shihui Ying, Zhijie Wen, Jun Shi, Yaxin Peng, Jigen Peng, and Hong Qiao.
  2018. *Manifold Preserving: An Intrinsic Approach for Semisupervised
  Distance Metric Learning*. IEEE TNNLS 29(7), 2731–2742. DOI:
  [10.1109/TNNLS.2017.2691005](https://doi.org/10.1109/TNNLS.2017.2691005).
