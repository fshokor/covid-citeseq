"""RNA -> protein model architectures and their sparsity-penalty variants.

Three architecture classes:
    MLP                - RNA -> hidden -> protein (single hidden layer, ReLU)
    LassoLinear        - RNA -> protein (no hidden layer)
    SkipConnectionMLP  - RNA -> protein via a direct linear term + a hidden
                          (LeakyReLU) term, summed

Sparsity is not baked into the architectures themselves -- it's applied as a
penalty function during training (see train.py). `VARIANTS` enumerates every
(architecture, penalty) combination actually used across the notebooks, so a
notebook can request one by name instead of re-wiring penalty functions by hand.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------

class MLP(nn.Module):
    """RNA -> hidden -> protein, multi-output regression.

    fc1 : gene -> hidden (Linear)
    fc2 : hidden -> protein (Linear)
    Nonlinearity: ReLU.
    """

    def __init__(self, rna_dim: int, hidden_dim: int, protein_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(rna_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, protein_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class LassoLinear(nn.Module):
    """RNA -> protein, single linear layer, no hidden layer, no nonlinearity."""

    def __init__(self, rna_dim: int, protein_dim: int):
        super().__init__()
        self.linear = nn.Linear(rna_dim, protein_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SkipConnectionMLP(nn.Module):
    """RNA -> protein via a direct linear term plus a hidden nonlinear term.

    direct : gene -> protein (Linear), uncontested linear channel per protein
    fc1    : gene -> hidden (Linear)
    fc2    : hidden -> protein (Linear)
    Nonlinearity: ReLU.

    forward() always returns direct + hidden. (nb09_architecture_comparison had
    a version whose forward() silently dropped the hidden term -- fixed here.)
    """

    def __init__(self, rna_dim: int, hidden_dim: int, protein_dim: int):
        super().__init__()
        self.direct = nn.Linear(rna_dim, protein_dim)
        self.fc1 = nn.Linear(rna_dim, hidden_dim)
        # self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, protein_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direct_pred, hidden_pred = self.forward_components(x)
        return direct_pred + hidden_pred

    def forward_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Direct and hidden-path predictions, kept separate for decomposition."""
        direct_pred = self.direct(x)
        hidden_pred = self.fc2(self.activation(self.fc1(x)))
        return direct_pred, hidden_pred


# --------------------------------------------------------------------------
# Penalty building blocks
# --------------------------------------------------------------------------

def l1(weight: torch.Tensor, lam: float) -> torch.Tensor:
    """Plain L1 penalty: lam * sum(|weight|)."""
    return lam * weight.abs().sum()


def group_l1(weight: torch.Tensor, lam: float) -> torch.Tensor:
    """Group-lasso penalty: lam * sum over rows of the L2 norm of each row.

    For an fc1 weight of shape (hidden_dim, rna_dim), this penalizes each
    gene's full weight column (dim=0) as one group, driving whole genes to
    zero together rather than scattering zeros across individual weights.
    """
    return lam * weight.norm(p=2, dim=0).sum()


# --------------------------------------------------------------------------
# Variant registry
# --------------------------------------------------------------------------

DEFAULT_LAMBDAS = {
    "fc1": 1e-4,
    "fc2": 1e-4,
    "direct": 1e-4,
    "fc1_group": 1e-3,
}


@dataclass
class Variant:
    """One (architecture, penalty) configuration."""

    name: str
    model_cls: type
    needs_hidden_dim: bool
    penalty_fn: Callable[[nn.Module, dict], torch.Tensor]
    description: str = ""


def _no_penalty(model: nn.Module, lambdas: dict) -> torch.Tensor:
    return torch.tensor(0.0)


def _mlp_fc1_penalty(model: MLP, lambdas: dict) -> torch.Tensor:
    return l1(model.fc1.weight, lambdas["fc1"])


def _mlp_fc2_penalty(model: MLP, lambdas: dict) -> torch.Tensor:
    return l1(model.fc2.weight, lambdas["fc2"])


def _mlp_fc1_fc2_penalty(model: MLP, lambdas: dict) -> torch.Tensor:
    return l1(model.fc1.weight, lambdas["fc1"]) + l1(model.fc2.weight, lambdas["fc2"])


def _mlp_fc1_group_fc2_penalty(model: MLP, lambdas: dict) -> torch.Tensor:
    return group_l1(model.fc1.weight, lambdas["fc1_group"]) + l1(model.fc2.weight, lambdas["fc2"])


def _lasso_penalty(model: LassoLinear, lambdas: dict) -> torch.Tensor:
    return l1(model.linear.weight, lambdas["direct"])


def _skip_fc1_penalty(model: SkipConnectionMLP, lambdas: dict) -> torch.Tensor:
    return l1(model.fc1.weight, lambdas["fc1"])


def _skip_fc2_penalty(model: SkipConnectionMLP, lambdas: dict) -> torch.Tensor:
    return l1(model.fc2.weight, lambdas["fc2"])


def _skip_fc1_fc2_penalty(model: SkipConnectionMLP, lambdas: dict) -> torch.Tensor:
    return l1(model.fc1.weight, lambdas["fc1"]) + l1(model.fc2.weight, lambdas["fc2"])


def _skip_fc1_fc2_direct_penalty(model: SkipConnectionMLP, lambdas: dict) -> torch.Tensor:
    return (
        l1(model.fc1.weight, lambdas["fc1"])
        + l1(model.fc2.weight, lambdas["fc2"])
        + l1(model.direct.weight, lambdas["direct"])
    )


def _skip_direct_penalty(model: SkipConnectionMLP, lambdas: dict) -> torch.Tensor:
    return l1(model.direct.weight, lambdas["direct"])


VARIANTS: dict[str, Variant] = {
    "mlp_no_l1": Variant(
        "mlp_no_l1", MLP, True, _no_penalty,
        "RNA -> hidden -> protein, no regularization.",
    ),
    "mlp_fc1": Variant(
        "mlp_fc1", MLP, True, _mlp_fc1_penalty,
        "RNA -> hidden -> protein, L1 on fc1 (nb05/nb06 style).",
    ),
    "mlp_fc2": Variant(
        "mlp_fc2", MLP, True, _mlp_fc2_penalty,
        "RNA -> hidden -> protein, L1 on fc2 only.",
    ),
    "mlp_fc1_fc2": Variant(
        "mlp_fc1_fc2", MLP, True, _mlp_fc1_fc2_penalty,
        "RNA -> hidden -> protein, L1 on fc1 and fc2 (nb07 style).",
    ),
    "mlp_fc1_group_fc2": Variant(
        "mlp_fc1_group_fc2", MLP, True, _mlp_fc1_group_fc2_penalty,
        "RNA -> hidden -> protein, group-L1 on fc1 + L1 on fc2 (nb09 style).",
    ),
    "lasso": Variant(
        "lasso", LassoLinear, False, _lasso_penalty,
        "RNA -> protein, linear only, L1 (nb08 style).",
    ),
    "skip_no_l1": Variant(
        "skip_no_l1", SkipConnectionMLP, True, _no_penalty,
        "RNA -> protein, direct + hidden, no regularization.",
    ),
    "skip_fc1": Variant(
        "skip_fc1", SkipConnectionMLP, True, _skip_fc1_penalty,
        "RNA -> protein, direct + hidden, L1 on fc1 only.",
    ),
    "skip_fc2": Variant(
        "skip_fc2", SkipConnectionMLP, True, _skip_fc2_penalty,
        "RNA -> protein, direct + hidden, L1 on fc2 only.",
    ),
    "skip_fc1_fc2": Variant(
        "skip_fc1_fc2", SkipConnectionMLP, True, _skip_fc1_fc2_penalty,
        "RNA -> protein, direct + hidden, L1 on fc1 and fc2.",
    ),
    "skip_fc1_fc2_direct": Variant(
        "skip_fc1_fc2_direct", SkipConnectionMLP, True, _skip_fc1_fc2_direct_penalty,
        "RNA -> protein, direct + hidden, L1 on fc1, fc2, and direct (nb08/nb09 primary).",
    ),
    "skip_direct": Variant(
        "skip_direct", SkipConnectionMLP, True, _skip_direct_penalty,
        "RNA -> protein, direct + hidden, L1 on direct only.",
    ),
}


def build_variant(
    name: str,
    rna_dim: int,
    protein_dim: int,
    hidden_dim: Optional[int] = None,
    lambdas: Optional[dict] = None,
) -> tuple[nn.Module, Callable[[nn.Module], torch.Tensor]]:
    """Instantiate a model and its bound penalty function by variant name.

    hidden_dim is required for every variant except 'lasso'. lambdas overrides
    the defaults in DEFAULT_LAMBDAS (missing keys fall back to the default).

    Returns (model, penalty_fn) where penalty_fn takes only the model, ready
    to pass straight into train.fit_model.
    """
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant {name!r}. Available: {sorted(VARIANTS)}")
    variant = VARIANTS[name]

    if variant.needs_hidden_dim:
        if hidden_dim is None:
            raise ValueError(f"Variant {name!r} requires hidden_dim.")
        model = variant.model_cls(rna_dim=rna_dim, hidden_dim=hidden_dim, protein_dim=protein_dim)
    else:
        model = variant.model_cls(rna_dim=rna_dim, protein_dim=protein_dim)

    merged_lambdas = dict(DEFAULT_LAMBDAS)
    if lambdas:
        merged_lambdas.update(lambdas)

    def penalty_fn(m: nn.Module, _penalty=variant.penalty_fn, _lambdas=merged_lambdas) -> torch.Tensor:
        return _penalty(m, _lambdas)

    return model, penalty_fn
