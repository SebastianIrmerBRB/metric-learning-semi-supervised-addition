# Deep LRML graph batching

## Literature basis

Weston et al., *Deep Learning via Semi-Supervised Embedding* (2012), ports a
shallow graph regularizer to a neural network by sharing the encoder between a
supervised task and sampled pairwise embedding updates. Its Algorithm 1
alternates a randomly sampled labeled example with a randomly sampled neighbor
pair (and, for its contrastive loss, a plain random example sharing the
neighbor pair's anchor; it is not filtered to be a non-neighbor). Section 3.1 treats
the graph `W` as supplied structure and discusses k-nearest-neighbor graph
construction as the potentially expensive preprocessing step.

Yang et al., *Revisiting Semi-Supervised Learning with Graph Embeddings* (2016),
uses two explicit batch types in Algorithm 2:

- `N1` labeled instances for the supervised objective;
- `N2` sampled graph contexts for the unsupervised objective.

Its graph-context loss is not LRML's Laplacian energy, but its separation of
batch units is directly useful for a stochastic deep LRML port.

## Local batch contract

The LRML and SLRML regularizers therefore use two independent sizes:

- training `batch_size`: number of supervised examples;
- regularizer `graph_batch_size`: number of sampled undirected graph edges.

When `graph_batch_size` is `null`, it takes the same numeric value as the
supervised `batch_size`, analogous to choosing `N1 = N2`; the units remain
different. A graph batch initially contains at most
`2 * graph_batch_size` endpoint occurrences. The sampler deduplicates shared
endpoint nodes before dataset loading and attaches an explicit local edge map,
so the loss still evaluates exactly the sampled edges and never adds incidental
induced-subgraph edges.

Edges are sampled uniformly from the upper triangle of the symmetric graph. If
the graph has `M` edges, the batch contains `b` edges, and
`W_plus = sum_(i<j) W_ij`, the stochastic objectives are:

```text
trace estimate = (M / b) * sum_(i,j in batch) W_ij ||z_i - z_j||^2

global weighted mean estimate =
    (M / (b * W_plus)) * sum_(i,j in batch) W_ij ||z_i - z_j||^2
```

`reduction="sum"` selects the trace estimate. `reduction="mean"` selects the
stable global weighted mean.

## Graph construction policy

For the closest port of classical LRML, build `W` once from a fixed feature
space by using `update_mode="once"`. Rebuilding a kNN graph from the current
learned embedding every epoch is a valid deep adaptation, but it changes the
objective because the sampled graph itself then depends on the model.

The training loop pairs one supervised batch with one graph batch. The default
`contrastive_update_mode="joint"` combines their losses in one optimizer step.
For LRML with contrastive repulsion enabled,
`contrastive_update_mode="alternating"` instead takes three actual model
updates per outer iteration: supervised, neighbor attraction, then anchored
random-pair repulsion. The last two branches are freshly forwarded after the
preceding update. This preserves the cited algorithm's optimizer boundary while
retaining the repository's mini-batched estimators; it does not reproduce the
paper's one-example/one-pair stochastic granularity.

## Configuration

```json
{
  "method_params": {
    "regularizer_params": {
      "graph_batch_size": 128,
      "reduction": "sum",
      "contrastive": true,
      "contrastive_update_mode": "alternating"
    }
  }
}
```

References:

- https://ronan.collobert.com/pub/2012_deeplearning_springer.pdf
- https://arxiv.org/abs/1603.08861
