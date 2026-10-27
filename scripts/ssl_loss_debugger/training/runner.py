"""Top-level scenario dispatch."""

from __future__ import annotations

import argparse

from scripts.ssl_loss_debugger.losses.scenarios import run_direct_proxy_loss
from scripts.ssl_loss_debugger.training.context import make_context
from scripts.ssl_loss_debugger.training.regularizers import run_regularizer
from scripts.ssl_loss_debugger.utils.constants import REGULARIZER_SCENARIOS


def run_scenario(args: argparse.Namespace, scenario: str) -> None:
    # Give every scenario the same initial data/model state when running "all".
    context = make_context(args)
    if scenario == "proxy_anchor":
        run_direct_proxy_loss(context, scenario)
    elif scenario in REGULARIZER_SCENARIOS:
        run_regularizer(context, scenario)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
