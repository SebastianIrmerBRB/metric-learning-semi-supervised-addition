"""Resolve portable JSON templates shared by experiment, SSL and run configs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested objects; lists/scalars replace, and an empty object resets."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if value and isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, _parents: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load JSON with relative ``extends`` paths, merging parents left to right.

    Every load returns fresh dictionaries. Cycles, missing parents, malformed
    JSON, and invalid parent references fail before any experiment is started.
    HPO retains its specialized spaces/per_loss resolver in training.hpo.
    """
    path = Path(path).resolve()
    if path in _parents:
        chain = " -> ".join(map(str, (*_parents, path)))
        raise ValueError(f"Circular config inheritance: {chain}")
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    if "extends" not in raw:
        return raw
    parents = raw.pop("extends")
    references = parents if isinstance(parents, list) else [parents]
    if not references or any(not isinstance(item, str) or not item for item in references):
        raise ValueError(f"'extends' must name a parent path or a nonempty list of paths: {path}")
    result: dict[str, Any] = {}
    for reference in references:
        parent = Path(reference)
        if not parent.is_absolute():
            parent = path.parent / parent
        result = merge_config(result, load_config(parent, (*_parents, path)))
    return merge_config(result, raw)
