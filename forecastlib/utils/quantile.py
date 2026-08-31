"""Quantile (pinball) loss and the config plumbing that turns it on.

The original TFT (Lim et al. 2021) is trained with a pinball loss and emits one
forecast per requested quantile, which is what makes questions like "how often
does this series exceed threshold tau" answerable. This module holds the pieces
that are shared between the model (which needs to know how wide to make its
output head) and the LightningModule (which needs the loss and the metrics).

Relationship to ``UncertaintyLightningModule``: that module is a *parametric*
alternative - a mean+variance head trained with Gaussian NLL - and stays as it
is. Quantile regression is non-parametric and needs no distributional
assumption, but it only produces the levels it was trained on. The two are kept
independent on purpose; they share the evaluation helpers below
(``coverage_series``/``interval_width_series``) so their intervals can be
compared with the same numbers rather than two parallel implementations.

Output layout convention
------------------------
A quantile model returns ``[..., c_out, n_quantiles]``: the quantile axis is
**last** and its entries are in **ascending** order of quantile level. Putting
it last means the existing ``outputs[:, -pred_len:, target_idx]`` slicing keeps
working untouched (``target_idx`` indexes the channel axis, which has not
moved) and simply carries the extra axis along.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

# Loss names in ``CustomLightningModule.loss_dict`` that consume a quantile axis.
QUANTILE_LOSSES = frozenset({"pinball", "quantile"})

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


def resolve_quantiles(configs) -> list[float] | None:
    """Return the quantile levels a config asks for, or ``None`` for a point model.

    The gate is the *loss*, not the model name: any model that grows a quantile
    head can opt in by setting ``self.n_quantiles``. ``configs.quantiles``
    defaults to :data:`DEFAULT_QUANTILES` and is sorted ascending, matching the
    output layout documented in this module.

    Both the model and the LightningModule call this, so they always agree on
    the head width. Note that it deliberately only *reads* the config: for a
    checkpoint to reload, ``loss`` (and ``quantiles``, when set explicitly) must
    survive into hparams, which happens automatically because reading them
    through ``ConfigTracker`` marks them as accessed.
    """
    loss = getattr(configs, "loss", None)
    quantiles = getattr(configs, "quantiles", None)

    if loss not in QUANTILE_LOSSES:
        if quantiles:
            # Not an error: it keeps sweeps over `loss` with a shared
            # `quantiles` default working. But it is worth saying out loud,
            # because the result is a plain point model.
            print(
                f"Warning: configs.quantiles={list(quantiles)} is ignored because loss={loss!r} "
                f"is not a quantile loss (one of {sorted(QUANTILE_LOSSES)})."
            )
        return None

    quantiles = list(quantiles) if quantiles else list(DEFAULT_QUANTILES)
    if not quantiles:
        raise ValueError(f"loss={loss!r} needs a non-empty configs.quantiles")
    if any(not 0.0 < float(q) < 1.0 for q in quantiles):
        raise ValueError(f"quantiles must all lie strictly between 0 and 1, got {quantiles}")
    quantiles = sorted(float(q) for q in quantiles)
    if len(set(quantiles)) != len(quantiles):
        raise ValueError(f"quantiles must be unique, got {quantiles}")
    return quantiles


def median_index(quantiles: Sequence[float]) -> int:
    """Index of the level to use as the point forecast: 0.5 if present, else nearest."""
    return min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))


def pinball_loss(y_pred: Tensor, y_true: Tensor, quantiles: Sequence[float] | Tensor) -> Tensor:
    """Mean pinball (quantile) loss.

    Args:
        y_pred: ``[..., n_quantiles]`` - predictions, quantile axis last.
        y_true: ``[...]`` - targets, i.e. ``y_pred`` without the quantile axis.
        quantiles: the levels, ascending, matching ``y_pred``'s last axis.

    Returns:
        Scalar loss, averaged over every element and every quantile, so it stays
        a drop-in for ``F.mse_loss`` as far as EarlyStopping/ModelCheckpoint are
        concerned.

    """
    q = torch.as_tensor(quantiles, dtype=y_pred.dtype, device=y_pred.device)
    if y_pred.shape[-1] != q.shape[0]:
        raise ValueError(
            f"pinball_loss expects a trailing quantile axis of size {q.shape[0]}, got y_pred with "
            f"shape {tuple(y_pred.shape)}. Does this model actually have a quantile head?"
        )
    if y_pred.shape[:-1] != y_true.shape:
        raise ValueError(
            f"y_pred {tuple(y_pred.shape)} must equal y_true {tuple(y_true.shape)} plus a trailing quantile axis"
        )

    errors = y_true.unsqueeze(-1) - y_pred
    return torch.maximum(q * errors, (q - 1.0) * errors).mean()


class QuantileLoss(nn.Module):
    """``pinball_loss`` with the levels bound, so it is callable as ``criterion(pred, true)``.

    The levels live in a non-persistent buffer: they follow the module across
    devices but stay out of the checkpoint's ``state_dict``, so a quantile
    checkpoint has exactly the same keys as its model and nothing else.
    """

    def __init__(self, quantiles: Sequence[float]) -> None:
        super().__init__()
        self.quantiles = list(quantiles)
        self.register_buffer("quantile_levels", torch.tensor(self.quantiles), persistent=False)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return pinball_loss(y_pred, y_true, self.quantile_levels)

    def extra_repr(self) -> str:
        return f"quantiles={self.quantiles}"


def pinball_series(y_true: Tensor, y_pred: Tensor, quantiles: Sequence[float]) -> Tensor:
    """Per-horizon-step pinball loss, in the ``*_series`` shape the callbacks expect.

    Args:
        y_true: ``[n_samples, pred_len, ...]``.
        y_pred: ``y_true``'s shape plus a trailing quantile axis.
        quantiles: the levels.

    Returns:
        ``[pred_len, ...]`` - averaged over samples and quantiles.

    """
    q = torch.as_tensor(quantiles, dtype=y_pred.dtype, device=y_pred.device)
    errors = y_true.unsqueeze(-1) - y_pred
    loss = torch.maximum(q * errors, (q - 1.0) * errors).mean(dim=-1)
    return loss.mean(dim=0)


def coverage_series(y_true: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Fraction of observations inside ``[lower, upper]``, per horizon step.

    Args:
        y_true: ``[n_samples, pred_len, ...]``.
        lower: same shape - lower interval bound.
        upper: same shape - upper interval bound.

    Returns:
        ``[pred_len, ...]``. For a well calibrated q10-q90 band this should sit
        near 0.8.

    """
    return ((lower <= y_true) & (y_true <= upper)).float().mean(dim=0)


def interval_width_series(lower: Tensor, upper: Tensor) -> Tensor:
    """Mean width of ``[lower, upper]`` per horizon step, shape ``[pred_len, ...]``."""
    return (upper - lower).mean(dim=0)


def quantile_crossing_series(y_pred: Tensor) -> Tensor:
    """Fraction of adjacent quantile pairs that are out of order, per horizon step.

    Quantile regression does not enforce monotonicity, so q10 <= q50 <= q90 is
    an empirical property of a converged model rather than a guarantee. This
    reports how often it is violated.

    Args:
        y_pred: ``[n_samples, pred_len, ..., n_quantiles]``, levels ascending.

    Returns:
        ``[pred_len, ...]`` - 0.0 when every step is properly ordered.

    """
    if y_pred.shape[-1] < 2:
        return torch.zeros(y_pred.shape[1:-1], dtype=y_pred.dtype, device=y_pred.device)
    crossings = (y_pred.diff(dim=-1) < 0).float().mean(dim=-1)
    return crossings.mean(dim=0)
