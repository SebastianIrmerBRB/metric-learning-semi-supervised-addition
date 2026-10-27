# Recall@K

Every run reports two retrieval metrics: `precision_at_1` and
`mean_average_precision_at_r`. `--recall_at_k` adds Recall@K for the K values
you name, and nothing else changes when you leave it off.

```text
--recall_at_k 1 2 4 8            # or "recall_at_k": [1, 2, 4, 8] in a config
```

## What it measures

Recall@K in the retrieval sense is the share of queries with **at least one**
correct neighbour among their K nearest. That is a hit rate, not a share of the
returned list, which is what separates it from precision at K: a query with one
correct neighbour out of four scores 1.0 for Recall@4 and 0.25 for
Precision@4.

Recall@1 is therefore the same quantity as `precision_at_1` and the run reports
both identically. Asking for it is a useful sanity check, not new information.

Queries whose label has no other sample in the reference set are excluded, the
same exclusion upstream applies to `precision_at_1`, so the two stay
comparable. With `--dataset_protocol` splits that use query/gallery evaluation,
Recall@K is measured query-to-gallery like the other metrics.

Measured on CUB with a frozen DINOv2 head, two epochs:

```text
test: Precision@1 = 86.9 , MAP@R = 51.3 , R@1 = 86.9 , R@2 = 92.4 , R@4 = 94.9 , R@8 = 96.5
```

## Where the values land

| Surface | Contents |
| --- | --- |
| `info.log` | appended to each evaluation's `Precision@1 = ... , MAP@R = ...` line |
| `metrics.csv` | one `recall_at_<K>` column per requested K, for every eval row |
| TensorBoard | `<split>/recall_at_<K>` and `epoch/<split>/recall_at_<K>`, grouped with the other validation curves |
| `cv_results.csv` | `best_valid_recall_at_<K>` and `test_recall_at_<K>` per fold, plus the mean row |
| `cv_summary.json` | `best_valid_recall_at_k`, `test_recall_at_k`, `epoch0_test_recall_at_k`, `concatenated_test_recall_at_k` per fold and aggregated |
| `grid_summary.{json,csv}` | per-run columns and their mean/std in `grid_aggregate.csv` |

In JSON the values live in one object per evaluation, keyed by K, so requesting
more K values never changes the record's shape:

```json
"test_recall_at_k": {"1": 0.8693, "2": 0.9240, "4": 0.9487, "8": 0.9649}
```

Validation reports the running maximum per K, exactly as
`best_valid_precision_at_1` and `best_valid_mean_average_precision_at_r` do, so
each K's best epoch is independent of the checkpoint that
`--selection_metric` selected.

## Cost

The metrics reuse the neighbour list the other metrics already need. MAP@R asks
for `max_bin_count` neighbours, so any K below the largest class size is free;
a larger K makes the FAISS search fetch that many neighbours instead. A K above
the number of searchable references is clamped to it, with a warning.

## What it does not do

Recall@K is reported, not optimized. `--selection_metric` still chooses between
`precision_at_1` and `map_at_r` for checkpointing and early stopping, and Optuna
objectives still come from `OBJECTIVE_METRICS` in
[`training/types.py`](../training/types.py), which Recall@K is deliberately not
part of. Making a Recall@K an HPO objective would mean selecting on a metric
whose K is a free parameter of the report rather than of the model.

## Implementation

`utils.RecallAtKAccuracyCalculator` subclasses pytorch-metric-learning's
`AccuracyCalculator` and binds one `calculate_recall_at_<K>` method per
requested K before delegating to the upstream constructor, which discovers
metrics by scanning for `calculate_*`. `utils.evaluate_embeddings` appends the
Recall@K mapping to its result tuple, and `utils.unpack_evaluation_result`
turns any combination of the optional outputs into a named
`EvaluationOutcome` so callers never have to branch on tuple length.
