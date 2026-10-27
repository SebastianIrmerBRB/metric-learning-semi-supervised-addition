"""Loss construction and direct loss-debugging scenarios."""

from scripts.ssl_loss_debugger.losses.factory import (
    compute_supervised_loss,
    make_supervised_criterion,
)

__all__ = ["compute_supervised_loss", "make_supervised_criterion"]
