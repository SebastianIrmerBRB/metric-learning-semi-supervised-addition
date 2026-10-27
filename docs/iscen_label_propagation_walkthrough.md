# Iscen label propagation: step-by-step worked example

This guide explains the whole-pool `iscen_label_spreading` path exposed by
[`training/semi_supervised.py`](../training/semi_supervised.py). The method is
named *label spreading* in this repository, while Iscen et al. call the method
*label propagation*. Both names below refer to the same implementation.

The adapter and the numerical algorithm live in different files:

```text
IscenLabelSpreadingPseudoLabeler.generate_pseudo_labels
    -> extract embeddings in [labeled, unlabeled] order
    -> semi_supervised.iscen_label_spreading              # facade
    -> training.ssl.algorithms.iscen_label_spreading      # implementation
    -> PseudoLabelResult(position, mapped_label, confidence)
```

The full implementation audit, including how propagation is integrated into
metric-learning training, is in
[`docs/iscen_label_spreading.md`](iscen_label_spreading.md).

## Notation

| Symbol | Meaning | Shape |
| --- | --- | ---: |
| `N` | Number of labeled and unlabeled samples | scalar |
| `d` | Embedding dimension | scalar |
| `C` | Number of mapped training classes | scalar |
| `F` | Model embeddings | `N x d` |
| `t` | Targets; `-1` means unlabeled | `N` |
| `W` | Symmetric affinity matrix | `N x N` |
| `S` | Degree-normalized affinity | `N x N` |
| `Y` | Class-balanced label seeds | `N x C` |
| `Z` | Raw propagated class scores | `N x C` |
| `P` | Row-normalized class probabilities | `N x C` |

Class labels must already be mapped to the dense range `0, ..., C - 1`.

## Running example

The example has one labeled seed for each of two classes and two unlabeled
samples. It uses small, non-default values (`k=2`, `gamma=1`, and `alpha=0.8`)
to make the arithmetic readable.

| Row | Role | Target | Embedding |
| ---: | --- | ---: | --- |
| 0 | labeled | 0 | `[1.00, 0.00]` |
| 1 | labeled | 1 | `[0.00, 1.00]` |
| 2 | unlabeled | -1 | `[0.98, 0.20]` |
| 3 | unlabeled | -1 | `[0.20, 0.98]` |

Here is the complete public call, runnable from the repository root:

```python
import numpy as np

from training.semi_supervised import iscen_label_spreading

features = np.array(
    [
        [1.00, 0.00],  # labeled class 0
        [0.00, 1.00],  # labeled class 1
        [0.98, 0.20],  # unlabeled, near class 0
        [0.20, 0.98],  # unlabeled, near class 1
    ],
    dtype=np.float32,
)
targets = np.array([0, 1, -1, -1], dtype=np.int64)

probabilities, confidences = iscen_label_spreading(
    features=features,
    targets=targets,
    num_classes=2,
    n_neighbors=2,
    gamma=1.0,
    alpha=0.8,
    cg_rtol=1e-12,
    cg_max_iter=100,
    linear_solver="cg",
)

unlabeled_probabilities = probabilities[2:]
pseudo_labels = unlabeled_probabilities.argmax(axis=1)

print(np.round(probabilities, 6))
print(pseudo_labels)
print(np.round(confidences, 6))
```

Output:

```text
[[0.771413 0.228587]
 [0.228587 0.771413]
 [0.643480 0.356520]
 [0.356520 0.643480]]
[0 1]
[1.000000 1.000000 0.268394 0.268394]
```

The next sections derive those values.

## Step 1: order the pool and extract embeddings

The adapter concatenates dataset positions with all labeled positions first:

```python
ssl_positions = np.concatenate(
    [split.labeled_positions, split.unlabeled_positions]
)
features = extract_embeddings(
    model=model,
    dataset=train_dataset,
    positions=ssl_positions,
    # ...
)
targets = np.concatenate(
    [
        labels[split.labeled_positions],
        np.full(len(split.unlabeled_positions), -1, dtype=np.int64),
    ]
)
```

This ordering is important. If there are `L` labeled examples, rows `0:L` of
the result belong to labeled samples and rows `L:N` belong to the unlabeled
samples. The adapter can therefore recover the latter with a suffix slice.

For the example:

$$
t = [0, 1, -1, -1].
$$

## Step 2: L2-normalize the embeddings and build a cosine kNN graph

Every embedding is normalized before FAISS inner-product search:

$$
v_i = \frac{f_i}{\lVert f_i \rVert_2}.
$$

The normalized example embeddings are approximately

$$
V =
\begin{bmatrix}
1       & 0       \\
0       & 1       \\
0.979804 & 0.199960 \\
0.199960 & 0.979804
\end{bmatrix}.
$$

For each query `i`, the graph builder keeps its `k` closest non-self neighbors.
A retained directed edge has weight

$$
a_{i \rightarrow j}
= \left[\max\left(v_i^\top v_j, 0\right)\right]^\gamma.
$$

Clipping at zero prevents negative cosine similarities from becoming graph
edges. The directed graph is made symmetric by adding both directions:

$$
W_{ij} = a_{i \rightarrow j} + a_{j \rightarrow i},
\qquad W_{ii}=0.
$$

Consequently, a mutual kNN edge contains the sum of two directed weights. For
example, rows 0 and 2 choose one another, so
`W[0, 2] = 0.979804 + 0.979804 = 1.959608`.

With `k=2` and `gamma=1`, the example graph is

$$
W \approx
\begin{bmatrix}
0        & 0        & 1.959608 & 0.199960 \\
0        & 0        & 0.199960 & 1.959608 \\
1.959608 & 0.199960 & 0        & 0.783687 \\
0.199960 & 1.959608 & 0.783687 & 0
\end{bmatrix}.
$$

An abridged version of the production graph construction is:

```python
normalized = np.ascontiguousarray(features, dtype=np.float32).copy()
faiss.normalize_L2(normalized)

# Exact inner-product search retrieves each point plus k non-self neighbors.
similarities, neighbors = faiss_flat_ip_search(
    database=normalized,
    queries=normalized,
    k=min(n_neighbors, len(normalized) - 1) + 1,
    purpose="mixed label propagation",  # shared graph helper's log label
    faiss_module=faiss,
)

# Remove self matches and retain the first k neighbors for every query.
# `selected_similarities` and the sparse row/column indices come from that step.
directed_values = np.clip(selected_similarities, 0.0, None) ** gamma
directed = scipy.sparse.coo_matrix(
    (directed_values.ravel(), (neighbor_rows, query_columns)),
    shape=(len(features), len(features)),
).tocsr()
W = directed + directed.T
W.setdiag(0)
W.eliminate_zeros()
```

The real helper is `make_mixed_label_affinity`; its name reflects that the same
graph builder is shared with the mixed-label-propagation implementation.

## Step 3: remove components that cannot receive a label

The implementation finds the connected components of `W`. A component is
active only if it contains at least one labeled node:

$$
\mathcal A
= \{i : \text{the component containing } i
             \text{ contains a labeled seed}\}.
$$

There is no source of class mass inside a seedless component, so its exact
solution is zero for every class. The implementation excludes such components
from the solve, leaves their probability rows as `[0, ..., 0]`, and the adapter
omits those unlabeled samples before calling `argmax`.

All four nodes in the running example are connected, so all remain active.

## Step 4: symmetrically degree-normalize the graph

First compute each node's weighted degree:

$$
d_i = \sum_j W_{ij},
\qquad
D = \operatorname{diag}(d_0, \ldots, d_{N-1}).
$$

For the example:

$$
d \approx [2.159568, 2.159568, 2.943255, 2.943255].
$$

Then normalize both endpoints of every edge:

$$
S = D^{-1/2} W D^{-1/2}.
$$

This gives

$$
S \approx
\begin{bmatrix}
0        & 0        & 0.777270 & 0.079313 \\
0        & 0        & 0.079313 & 0.777270 \\
0.777270 & 0.079313 & 0        & 0.266265 \\
0.079313 & 0.777270 & 0.266265 & 0
\end{bmatrix}.
$$

Code:

```python
degrees = np.asarray(W.sum(axis=1), dtype=np.float64).ravel()
inverse_sqrt_degrees = np.zeros_like(degrees)
positive_degree = degrees > 0.0
inverse_sqrt_degrees[positive_degree] = (
    1.0 / np.sqrt(degrees[positive_degree])
)
D_inverse_sqrt = scipy.sparse.diags(inverse_sqrt_degrees)
S = (D_inverse_sqrt @ W @ D_inverse_sqrt).tocsr()
```

## Step 5: construct class-balanced label seeds

Let `n_c` be the number of labeled examples belonging to class `c`. The seed
matrix is

$$
Y_{ic} =
\begin{cases}
\frac{1}{n_c}, & \text{if node } i \text{ is labeled as class } c, \\
0,             & \text{otherwise}.
\end{cases}
$$

Thus every represented class contributes total seed mass one, even if the
number of labeled examples differs across classes. In the example each class
has one seed:

$$
Y =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 0 \\
0 & 0
\end{bmatrix}.
$$

The implementation constructs it as follows:

```python
labeled = targets != -1
labeled_indices = np.flatnonzero(labeled)
labeled_targets = targets[labeled]
class_seed_counts = np.bincount(
    labeled_targets,
    minlength=num_classes,
).astype(np.float64)

Y = np.zeros((len(features), num_classes), dtype=np.float64)
Y[labeled_indices, labeled_targets] = (
    1.0 / class_seed_counts[labeled_targets]
)
```

A class with no labeled example has an all-zero column in `Y`; label
propagation cannot discover that class.

## Step 6: solve the propagation system

Iscen propagation solves

$$
(I - \alpha S)Z = Y,
$$

or equivalently

$$
Z = (I - \alpha S)^{-1}Y
  = \sum_{r=0}^{\infty}\alpha^r S^rY.
$$

The series form gives the intuition: `S^r Y` moves seed information across
paths of length `r`, while `alpha^r` discounts longer paths.

For `alpha=0.8`, the example system matrix is

$$
I - 0.8S \approx
\begin{bmatrix}
1         & 0         & -0.621816 & -0.063451 \\
0         & 1         & -0.063451 & -0.621816 \\
-0.621816 & -0.063451 & 1         & -0.213012 \\
-0.063451 & -0.621816 & -0.213012 & 1
\end{bmatrix}.
$$

Solving once for the two columns of `Y` produces

$$
Z \approx
\begin{bmatrix}
1.912719 & 0.566783 \\
0.566783 & 1.912719 \\
1.389284 & 0.769732 \\
0.769732 & 1.389284
\end{bmatrix}.
$$

The repository uses a sparse solver rather than explicitly forming the inverse:

```python
system = (
    scipy.sparse.eye(len(active_indices), format="csr") - alpha * S
)
Z = solve_sparse_label_system(
    system,
    Y,
    rtol=cg_rtol,
    max_iter=cg_max_iter,
    name="Iscen label spreading",
    linear_solver=linear_solver,
    allow_nonconvergence=linear_solver != "cholmod",
)
```

The default solver is conjugate gradient with at most 20 iterations, matching
the reference implementation's deliberately truncated solve. The optional
`cholmod` path performs an exact sparse solve when that dependency is present.

## Step 7: clamp and normalize the propagated scores

A truncated numerical solve can produce small negative values. The
implementation first clamps them:

$$
\widetilde Z_{ic} = \max(Z_{ic}, 0).
$$

Every row with positive mass is then L1-normalized:

$$
P_{ic}
= \frac{\widetilde Z_{ic}}
        {\sum_{c'=0}^{C-1}\widetilde Z_{ic'}}.
$$

For the example:

$$
P \approx
\begin{bmatrix}
0.771413 & 0.228587 \\
0.228587 & 0.771413 \\
0.643480 & 0.356520 \\
0.356520 & 0.643480
\end{bmatrix}.
$$

Notice that even labeled rows are diffused distributions rather than clamped
one-hot rows. Only the unlabeled suffix is converted to pseudo-labels:

$$
\widehat y_i = \operatorname*{arg\,max}_c P_{ic}.
$$

Therefore row 2 receives pseudo-label `0`, and row 3 receives pseudo-label `1`.

## Step 8: compute entropy certainty

For every positive-mass row, first compute Shannon entropy using natural logs:

$$
H(P_i) = -\sum_{c=0}^{C-1}P_{ic}\log P_{ic},
$$

where `0 log(0)` is treated as zero. Iscen certainty is one minus entropy
normalized by the maximum possible `C`-class entropy:

$$
\omega_i^{\mathrm{raw}}
= 1 - \frac{H(P_i)}{\log C}.
$$

The implementation then applies the reference code's graph-wide rescaling:

$$
\omega_i
= \frac{\omega_i^{\mathrm{raw}}}
        {\max_j \omega_j^{\mathrm{raw}}},
$$

and finally forces every true-labeled sample's certainty to `1`.

For the running example:

```text
entropy H             = [0.537564, 0.537564, 0.651390, 0.651390]
raw certainty         = [0.224459, 0.224459, 0.060244, 0.060244]
rescaled certainty    = [1.000000, 1.000000, 0.268394, 0.268394]
```

The two pseudo-labels are correct but relatively uncertain because each still
receives substantial propagated mass from the other class.

An important consequence of the maximum rescaling is that a confidence such
as `0.8` is relative to the most certain node in the current graph; it is not a
probability calibration that is directly comparable across graphs or epochs.

## Step 9: omit unpropagatable rows and apply the confidence threshold

The algorithm uses an all-zero probability row as an explicit sentinel for an
unlabeled sample that received no usable class mass. The adapter removes those
rows before `argmax`:

```python
unlabeled_start = len(split.labeled_positions)
unlabeled_probabilities = probabilities[unlabeled_start:]

propagated = unlabeled_probabilities.sum(axis=1) > 0.0
unlabeled_probabilities = unlabeled_probabilities[propagated]

pseudo_labels = np.argmax(unlabeled_probabilities, axis=1)
pseudo_label_positions = split.unlabeled_positions[propagated]
unlabeled_confidences = confidences[unlabeled_start:][propagated]
```

Later, `filter_pseudo_labels` normally accepts a candidate only when

$$
\omega_i \geq \texttt{confidence\_threshold}.
$$

Sampler-capacity rescue can admit some otherwise rejected candidates when that
feature is configured. With a threshold of `0.25`, both example pseudo-labels
survive; with a threshold of `0.30`, neither survives.

## Default parameters

`IscenLabelSpreadingPseudoLabeler.DEFAULT_PARAMS` supplies these values unless
`method_params` overrides them:

| Parameter | Default | Effect |
| --- | ---: | --- |
| `n_neighbors` | `50` | Number of non-self kNN candidates per node |
| `gamma` | `3.0` | Sharpens positive cosine affinity weights |
| `alpha` | `0.99` | Controls how far labels diffuse through the graph |
| `cg_rtol` | `1e-6` | Relative CG stopping tolerance |
| `cg_max_iter` | `20` | Maximum CG iterations per class system |
| `linear_solver` | `"cg"` | Sparse linear-system backend |

A minimal SSL configuration is:

```json
{
  "method": "iscen_label_spreading",
  "update_mode": "every_epoch",
  "confidence_threshold": 0.0,
  "method_params": {
    "n_neighbors": 50,
    "gamma": 3.0,
    "alpha": 0.99,
    "cg_rtol": 1e-6,
    "cg_max_iter": 20,
    "linear_solver": "cg"
  }
}
```

## Implementation map

| Responsibility | File and symbol |
| --- | --- |
| Public façade | [`training/semi_supervised.py`](../training/semi_supervised.py), `iscen_label_spreading` |
| Pool ordering, embedding extraction, result slicing | [`training/semi_supervised.py`](../training/semi_supervised.py), `IscenLabelSpreadingPseudoLabeler` |
| Parameter validation and defaults | [`training/semi_supervised.py`](../training/semi_supervised.py), `validate_iscen_label_spreading_params` |
| Graph, diffusion, normalization, certainty | [`training/ssl/algorithms.py`](../training/ssl/algorithms.py), `iscen_label_spreading` |
| Cosine kNN affinity | [`training/ssl/algorithms.py`](../training/ssl/algorithms.py), `make_mixed_label_affinity` |
| Sparse solver dispatch | [`training/ssl/algorithms.py`](../training/ssl/algorithms.py), `solve_sparse_label_system` |
| Confidence filtering | [`training/ssl/pseudo_labels.py`](../training/ssl/pseudo_labels.py), `filter_pseudo_labels` |
| Focused regression tests | [`tests/test_iscen_label_spreading.py`](../tests/test_iscen_label_spreading.py) |

## Practical checks when results look wrong

1. Confirm that at least one labeled seed exists for every class you expect the
   method to predict. An unseeded class can never receive propagated mass.
2. Confirm the input order is `[labeled, unlabeled]`; the adapter relies on the
   unlabeled samples being a suffix.
3. Inspect whether unlabeled nodes lie in graph components containing a labeled
   seed. Seedless components are intentionally omitted.
4. Increase `cg_max_iter` or try `linear_solver="cholmod"` if logs report
   zero-mass rows inside seed-reachable components.
5. Remember that reciprocal kNN edges are summed, and `gamma` can make weak
   cosine edges extremely small.
6. Treat the returned certainty as a relative entropy score, not as the model's
   calibrated probability of being correct.
