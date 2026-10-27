"""Per-epoch size and movement of the trainable projection head.

The head's output is L2-normalized before any loss sees it, so the loss is
invariant to scaling the head's parameters: only their direction matters. An
update of fixed size then turns the direction by an angle proportional to
``1 / ||W||``, and Adam's per-coordinate step is roughly ``lr`` whatever the
gradient scale. So when ``||W||`` grows the effective step shrinks even at a
constant learning rate, and decoupled weight decay (AdamW) counters that by
pulling ``||W||`` back.

``relative_update = ||W_end - W_start|| / ||W_start||`` over one epoch is that
effective step: the quantity that shrinks if the head stalls because it grew.
"""

import torch


PREFIX = "head_dynamics/"


class HeadDynamicsTracker:
    """Snapshot the head at the start of an epoch; report norms and movement at its end."""

    def __init__(self, module):
        self.named_parameters = [
            (name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad
        ]
        self._start = None

    def begin_epoch(self):
        with torch.no_grad():
            self._start = [parameter.detach().clone() for _, parameter in self.named_parameters]

    def end_epoch(self, num_steps):
        """Return the epoch's diagnostics, or ``{}`` if no epoch was begun."""

        if self._start is None or not self.named_parameters:
            return {}
        with torch.no_grad():
            per_parameter = torch.stack(
                [
                    torch.stack(
                        (
                            parameter.detach().float().pow(2).sum(),
                            start.float().pow(2).sum(),
                            (parameter.detach().float() - start.float()).pow(2).sum(),
                        )
                    )
                    for (_, parameter), start in zip(self.named_parameters, self._start)
                ]
            ).to(device="cpu", dtype=torch.float64)
        self._start = None

        diagnostics = {}
        for (name, _), (end_sq, start_sq, update_sq) in zip(self.named_parameters, per_parameter.tolist()):
            diagnostics[f"{PREFIX}{name}/weight_norm"] = end_sq ** 0.5
            if start_sq > 0.0:
                diagnostics[f"{PREFIX}{name}/relative_update"] = (update_sq / start_sq) ** 0.5
        end_sq, start_sq, update_sq = per_parameter.sum(dim=0).tolist()
        diagnostics[f"{PREFIX}weight_norm"] = end_sq ** 0.5
        diagnostics[f"{PREFIX}weight_norm_start"] = start_sq ** 0.5
        diagnostics[f"{PREFIX}update_norm"] = update_sq ** 0.5
        if start_sq > 0.0:
            relative_update = (update_sq / start_sq) ** 0.5
            diagnostics[f"{PREFIX}relative_update"] = relative_update
            if num_steps > 0:
                diagnostics[f"{PREFIX}relative_update_per_step"] = relative_update / num_steps
        diagnostics[f"{PREFIX}steps"] = float(num_steps)
        return diagnostics
