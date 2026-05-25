import torch

@torch.jit.script
def mean(ps, xs):
    return torch.einsum('ni,bn->bi', xs, ps)

@torch.jit.script
def covariance(ps, xs, ys, xs_mean, ys_mean):
    cov = torch.einsum('bn,ni,nj->bij', ps, xs, ys) 
    return cov - xs_mean.unsqueeze(-1) @ ys_mean.unsqueeze(-2)
