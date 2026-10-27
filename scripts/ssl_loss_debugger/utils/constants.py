"""Shared constants and paths for the SSL loss debugger."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEBUG_DATASETS = (
    "synthetic",
    "Cars196",
    "CUB",
)

REGULARIZER_SCENARIOS = (
    "lrml",
    "slrml",
    "seraph",
    "hoffer_entropy",
)

# Regularizer scenario names are ``semi_supervised.REGULARIZER_REGISTRY`` keys.
# ``seraph_entropy`` was that key before the rename recorded in docs/seraph.md and
# still appears in saved run commands, so it keeps resolving to the current name.
SCENARIO_ALIASES = {"seraph_entropy": "seraph"}

# Scenarios whose name is also a METHOD_REGISTRY entry. Their debug config names
# the method the production config would name instead of the generic
# "regularized" composition, so the two configs stay comparable.
METHOD_NAMED_SCENARIOS = ("seraph",)
LOSS_SCENARIOS = ("proxy_anchor",)
ALL_SCENARIOS = (*LOSS_SCENARIOS, *REGULARIZER_SCENARIOS)

# What the tiny model is fed. Cached backbone features are one deterministic row
# per dataset position, so scenarios that compare two augmentations of the same
# image cannot run on them - they would compare a sample with itself.
BACKBONE_FEATURES_PIXELS = "pixels"
BACKBONE_FEATURES_CACHED = "cached"
BACKBONE_FEATURE_SOURCES = (BACKBONE_FEATURES_PIXELS, BACKBONE_FEATURES_CACHED)

BREAK_PHASES = (
    "none",
    "before-build",
    "after-build",
    "before-loader",
    "after-loader",
    "before-loss",
    "after-loss",
    "before-backward",
)
