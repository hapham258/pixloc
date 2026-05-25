import torch
from ..dependencies.pixloc_imports import barron_loss

def loss_identity(residuals, J, H_sigma):
    return residuals, J

def loss_barron(residuals, J, H_sigma):
    res = ((residuals / (H_sigma * 2))**2).sum(dim=-1)
    _, weights, _ = barron_loss(res, alpha=torch.tensor(-2))

    weights[weights < 0.5 / H_sigma] = weights.max()
    weights /= weights.max()
    weights = weights.unsqueeze(-1)

    J = weights.unsqueeze(-1) * J
    residuals = weights * residuals

    return residuals, J
