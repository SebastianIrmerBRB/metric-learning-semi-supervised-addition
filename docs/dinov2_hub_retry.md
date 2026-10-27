# DINOv2 Torch Hub Retry

The DINOv2 model is loaded through Torch Hub when `DinoWrapper` is created:

```python
torch.hub.load("facebookresearch/dinov2:main", model_name)
```

This operation can require network access when the DINOv2 repository code or model
weights are not already present in the local Torch Hub cache.

## Observed Problem

During a long Optuna hyperparameter search, a trial failed while constructing the
DINOv2 model:

```text
urllib.error.HTTPError: HTTP Error 504: Gateway Time-out
```

The failed model load propagated out of the Optuna objective. Optuna marked the
current trial as failed and stopped the complete search because the exception was
not configured as a recoverable trial error.

The failure happened before training started. It was caused by a temporary GitHub
or network gateway problem, not by the model, dataset, or trial parameters.

## Root Cause

The original Hub repository reference did not specify a Git branch:

```python
torch.hub.load("facebookresearch/dinov2", model_name)
```

When no branch is specified, Torch Hub requests the DINOv2 GitHub page to determine
whether the repository uses `main` or `master`. This lookup can happen even when a
cached repository is available.

Torch Hub can fall back to its cache for some connection errors, but an HTTP `504`
response is propagated as an exception. A single temporary gateway timeout could
therefore terminate a long-running experiment.

## Changes

The repository reference is now pinned to the known DINOv2 branch:

```python
DINOV2_REPO = "facebookresearch/dinov2:main"
```

Specifying `:main` avoids the unnecessary request used to discover the default
branch and makes cached repository use more reliable.

The model load is also wrapped by `load_dinov2_with_retry()`. It makes at most four
load attempts and uses exponential delays:

```text
Attempt 1 fails -> wait 5 seconds
Attempt 2 fails -> wait 10 seconds
Attempt 3 fails -> wait 20 seconds
Attempt 4 fails -> raise the exception
```

A warning is emitted before each retry so temporary failures and delays remain
visible in experiment logs.

## Retry Policy

Retries are limited to errors that are likely to be temporary:

- HTTP `408 Request Timeout`
- HTTP `429 Too Many Requests`
- HTTP `5xx` server and gateway errors, including `504 Gateway Time-out`
- URL, connection, socket timeout, and HTTP protocol errors

Non-retryable errors are raised immediately. For example, an HTTP `404` may indicate
an invalid repository, branch, or m odel configuration and is unlikely to improve
after waiting.

After all retry attempts are exhausted, the final error is raised normally. This
prevents the experiment from waiting indefinitely during a persistent outage.

## Configuration

The retry behavior is defined in `models/retrieval_model.py`:

```python
DINOV2_HUB_MAX_ATTEMPTS = 4
DINOV2_HUB_RETRY_DELAY_SECONDS = 5
```

`DINOV2_HUB_MAX_ATTEMPTS` includes the initial load attempt. The total added wait
time with the default configuration is 35 seconds before the fourth and final
failure is raised.

## Operational Notes

The first successful load still requires network access unless both the DINOv2
repository and requested model weights already exist in the Torch Hub cache.

The retry wrapper protects model initialization from short network interruptions.
It does not make persistent network outages recoverable and does not cause Optuna
to continue after all model-loading attempts fail.
