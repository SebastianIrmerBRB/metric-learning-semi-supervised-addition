"""Application entry point for the SSL loss debugger."""

from __future__ import annotations

import sys
from pathlib import Path

# Running this file directly puts ``scripts/ssl_loss_debugger`` ahead of the
# repository root on sys.path.  That makes imports such as ``losses`` and
# ``utils`` resolve to this package's helper directories instead of the
# production packages at the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from loguru import logger

from scripts.ssl_loss_debugger.training.runner import run_scenario
from scripts.ssl_loss_debugger.utils.cli import parse_args
from scripts.ssl_loss_debugger.utils.constants import ALL_SCENARIOS


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, format="<level>{level: <8}</level> | {message}")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Safe when main() is invoked repeatedly from an interactive debugger.
        pass
    torch.autograd.set_detect_anomaly(args.anomaly_detection)
    scenarios = ALL_SCENARIOS if args.scenario == "all" else (args.scenario,)
    print(
        f"device={args.device}, torch_threads={torch.get_num_threads()}, "
        f"epochs={args.epochs}, backbone_tuning={args.backbone_tuning}, "
        f"scenarios={list(scenarios)}"
    )
    for scenario in scenarios:
        run_scenario(args, scenario)
    print("\nAll requested debug scenarios completed with finite backward passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
