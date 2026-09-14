"""Loss/SSL-aware batch-size thresholds for scheduler-assigned GPUs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


DEFAULT_GPU_BATCH_SIZE_THRESHOLD = 128
RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLD_ENV = "RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLD"
RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLDS_ENV = "RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLDS"

_WILDCARDS = {"all", "*"}


@dataclass(frozen=True)
class GpuBatchSizeThresholdRule:
    """One ``LOSS:SSL_METHOD=THRESHOLD`` device-selection rule."""

    loss: str
    ssl_method: str
    threshold: int

    @property
    def loss_is_wildcard(self) -> bool:
        return _is_wildcard(self.loss)

    @property
    def ssl_method_is_wildcard(self) -> bool:
        return _is_wildcard(self.ssl_method)

    @property
    def specificity(self) -> int:
        return int(not self.loss_is_wildcard) + int(not self.ssl_method_is_wildcard)

    def to_spec(self) -> str:
        loss = "all" if self.loss_is_wildcard else self.loss
        ssl_method = "all" if self.ssl_method_is_wildcard else self.ssl_method
        return f"{loss}:{ssl_method}={self.threshold}"

    def __str__(self) -> str:
        return self.to_spec()


def parse_gpu_batch_size_threshold_rule(
    value: str | GpuBatchSizeThresholdRule,
) -> GpuBatchSizeThresholdRule:
    """Parse one loss/SSL threshold rule with an actionable error message."""

    if isinstance(value, GpuBatchSizeThresholdRule):
        return value
    if not isinstance(value, str):
        raise ValueError(
            "GPU batch-size threshold rules must be strings formatted as "
            "LOSS:SSL_METHOD=THRESHOLD"
        )

    raw = value.strip()
    selectors, equals, raw_threshold = raw.rpartition("=")
    if not equals or "=" in selectors or selectors.count(":") != 1:
        raise ValueError(
            f"GPU batch-size threshold rule {value!r} must be formatted as "
            "LOSS:SSL_METHOD=THRESHOLD"
        )
    loss, ssl_method = (part.strip() for part in selectors.split(":", 1))
    if not loss or not ssl_method or not raw_threshold.strip():
        raise ValueError(
            f"GPU batch-size threshold rule {value!r} must include a loss, "
            "an SSL method, and a threshold"
        )
    try:
        threshold = int(raw_threshold.strip())
    except ValueError as exc:
        raise ValueError(
            f"GPU batch-size threshold in rule {value!r} must be an integer"
        ) from exc
    if threshold < 0:
        raise ValueError(
            f"GPU batch-size threshold in rule {value!r} must be non-negative"
        )
    return GpuBatchSizeThresholdRule(
        loss="all" if _is_wildcard(loss) else loss,
        ssl_method="all" if _is_wildcard(ssl_method) else ssl_method,
        threshold=threshold,
    )


def encode_gpu_batch_size_threshold_rules(
    rules: Sequence[GpuBatchSizeThresholdRule],
) -> str:
    """Serialize validated rules for a scheduler child environment."""

    return json.dumps(
        [parse_gpu_batch_size_threshold_rule(rule).to_spec() for rule in rules],
        separators=(",", ":"),
    )


def decode_gpu_batch_size_threshold_rules(
    raw_rules: str | None,
) -> tuple[GpuBatchSizeThresholdRule, ...]:
    """Read rules from JSON, also accepting a whitespace-separated env value."""

    if raw_rules is None or not raw_rules.strip():
        return ()
    raw_rules = raw_rules.strip()
    try:
        decoded = json.loads(raw_rules)
    except json.JSONDecodeError:
        values: Any = raw_rules.split()
    else:
        values = [decoded] if isinstance(decoded, str) else decoded
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError(
            f"{RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLDS_ENV} must be a JSON "
            "array of LOSS:SSL_METHOD=THRESHOLD strings"
        )
    return tuple(parse_gpu_batch_size_threshold_rule(value) for value in values)


def ssl_method_aliases(ssl_config: Any) -> tuple[str, ...]:
    """Return canonical and user-facing names for a resolved SSL config."""

    if ssl_config is None:
        return ()
    method = _config_value(ssl_config, "method")
    method_params = _config_value(ssl_config, "method_params") or {}
    regularizer = (
        method_params.get("regularizer") if isinstance(method_params, Mapping) else None
    )

    aliases: list[str] = []
    for name in (method, regularizer):
        if name is None:
            continue
        name = str(name).strip()
        if name and name not in aliases:
            aliases.append(name)
        normalized = _normalize_identifier(name)
        for suffix in ("entropy", "labelspreading", "labelpropagation"):
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                short_name = normalized[: -len(suffix)]
                if short_name not in aliases:
                    aliases.append(short_name)

    if (
        method is not None
        and _normalize_identifier(method) == "none"
        and "supervised" not in aliases
    ):
        aliases.append("supervised")
    return tuple(aliases)


def gpu_batch_size_threshold_input(
    batch_size: Any,
    ssl_config: Any = None,
) -> tuple[int, dict[str, int]]:
    """Return the workload size compared with a training-GPU threshold.

    LRML's global-graph step runs both the supervised batch and a sampled
    graph-edge batch.  Treat their configured sizes as one combined workload
    for device selection.  LRML in-batch graphs reuse labeled supervised
    embeddings, so only their additional unlabeled forward is added.
    """

    supervised_batch_size = _positive_batch_component(batch_size, "batch_size")
    components = {"batch_size": supervised_batch_size}
    if not any(
        _normalize_identifier(alias) == "lrml"
        for alias in ssl_method_aliases(ssl_config)
    ):
        return supervised_batch_size, components

    graph_batch_mode = str(
        _config_value(ssl_config, "graph_batch_mode") or "global"
    ).casefold()
    if graph_batch_mode == "in_batch":
        graph_unlabeled_batch_size = _config_value(
            ssl_config,
            "graph_unlabeled_batch_size",
        )
        if graph_unlabeled_batch_size is None:
            return supervised_batch_size, components
        components["lrml_graph_unlabeled_batch_size"] = _positive_batch_component(
            graph_unlabeled_batch_size,
            "LRML graph_unlabeled_batch_size",
        )
    else:
        method_params = _config_value(ssl_config, "method_params") or {}
        regularizer_params = (
            method_params.get("regularizer_params", {})
            if isinstance(method_params, Mapping)
            else {}
        )
        raw_graph_batch_size = (
            regularizer_params.get("graph_batch_size")
            if isinstance(regularizer_params, Mapping)
            else None
        )
        # This mirrors LRMLRegularizer.make_loader: an omitted graph edge batch
        # size defaults to the supervised batch size.
        graph_batch_size = (
            supervised_batch_size
            if raw_graph_batch_size is None
            else _positive_batch_component(
                raw_graph_batch_size,
                "LRML graph_batch_size",
            )
        )
        components["lrml_graph_edge_batch_size"] = graph_batch_size

    return sum(components.values()), components


def resolve_gpu_batch_size_threshold(
    *,
    default_threshold: int,
    rules: Sequence[GpuBatchSizeThresholdRule],
    loss: str | None,
    ssl_config: Any = None,
) -> tuple[int, GpuBatchSizeThresholdRule | None]:
    """Select the most specific rule, using the later rule to break ties."""

    method_names = ssl_method_aliases(ssl_config)
    matched_rule = None
    matched_specificity = -1
    for raw_rule in rules:
        rule = parse_gpu_batch_size_threshold_rule(raw_rule)
        if not _rule_matches(rule, loss, method_names):
            continue
        if rule.specificity >= matched_specificity:
            matched_rule = rule
            matched_specificity = rule.specificity
    if matched_rule is None:
        return int(default_threshold), None
    return matched_rule.threshold, matched_rule


def _rule_matches(
    rule: GpuBatchSizeThresholdRule,
    loss: str | None,
    method_names: Sequence[str],
) -> bool:
    if not rule.loss_is_wildcard:
        if loss is None or _normalize_loss(rule.loss) != _normalize_loss(loss):
            return False
    if not rule.ssl_method_is_wildcard:
        expected_method = _normalize_identifier(rule.ssl_method)
        if expected_method not in {
            _normalize_identifier(method_name) for method_name in method_names
        }:
            return False
    return True


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _positive_batch_component(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _normalize_loss(value: str) -> str:
    normalized = _normalize_identifier(value)
    # Friendly names such as ``ArcFace`` and canonical PML names such as
    # ``ArcFaceLoss`` deliberately identify the same loss.
    return normalized[:-4] if normalized.endswith("loss") else normalized


def _normalize_identifier(value: Any) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _is_wildcard(value: str) -> bool:
    return value.strip().casefold() in _WILDCARDS
