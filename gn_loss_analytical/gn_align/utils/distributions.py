import torch
from math import pi, exp

@torch.jit.script
def distance_to_xs(alphas, xs):
    return torch.norm(alphas.unsqueeze(-2) - xs.unsqueeze(0), dim=-1)

@torch.jit.script
def stable_l1_normalization(tensor):
    w = tensor.max(dim=-1)[0].abs().unsqueeze(-1) + 1e-9
    tensor = (tensor / (tensor / w).sum(dim=-1).unsqueeze(-1)) / w
    tensor[tensor < 1e-8] = 0
    return tensor

class NormalDistributionPdf:

    def __init__(self, mean=0, std=20):
        self.dist = torch.distributions.normal.Normal(mean, std)
        self.std = std
    
    def __call__(self, alphas, xs):
        dists = distance_to_xs(alphas, xs)

        dists = self.dist.log_prob(dists)

        dists = torch.clamp(dists, -22, 1.1**(pi * exp(1.05)) )
        return torch.nn.functional.softmax(dists, dim=-1)

class UnifromDistanceDistributionPdf:

    def __init__(self, dist = 10000):
        self.dist = dist

    def __call__(self, alphas, xs):
        dists = distance_to_xs(alphas, xs)
        
        mask = (dists <= self.dist)
        dists[~mask] = 0.0
        dists[mask] = 1.0
        return dists

class UnifromKNNDistributionPdf:

    def __init__(self, k_nearest = -1):
        self.k_nearest = k_nearest

    def __call__(self, alphas, xs):
        dists = distance_to_xs(alphas, xs)
        if self.k_nearest == -1:
            dists = torch.ones(dists.shape).to(alphas)
        else:
            k = min(self.k_nearest, dists.shape[-1])
            vals, inds  = torch.topk(dists, k = k, largest=False, dim=-1)
            
            mask = dists < vals.max(dim=-1)[0].unsqueeze(-1)
            dists[~mask] = 0.0
            dists[mask] = 1.0

        return dists

class NormalDistributionPdf2d:
    def __init__(self, mean=torch.tensor([0,0]).float().cuda(), std=20 * torch.eye(2)):
        self.dist_x = torch.distributions.normal.Normal(mean[0], std[0,0])
        self.dist_y = torch.distributions.normal.Normal(mean[0], std[1,1])
        self.std = std
    
    def __call__(self, alphas, xs):
        dist_x = alphas[..., 0].unsqueeze(-1) - xs[...,0].unsqueeze(0)
        dist_y = alphas[..., 1].unsqueeze(-1) - xs[...,1].unsqueeze(0)

        dists_x = self.dist_x.log_prob(dist_x)
        dists_y = self.dist_y.log_prob(dist_y)

        dists = torch.clamp(dists_x + dists_y, -22, 1.1**(pi * exp(1)) )
        return torch.nn.functional.softmax(dists, dim=-1)

class UnifromDistanceDistributionPdf2d:

    def __init__(self, dist_x = 10000, dist_y=10000):
        self.dist_x = dist_x
        self.dist_y = dist_y

    def __call__(self, alphas, xs):
        dist_x = torch.abs(alphas[..., 0].unsqueeze(-1) - xs[...,0].unsqueeze(0))
        dist_y = torch.abs(alphas[..., 1].unsqueeze(-1) - xs[...,1].unsqueeze(0))
        
        mask = (dist_x <= self.dist_x) & (dist_y <= self.dist_y)
        dist_x[~mask] = 0.0
        dist_x[mask] = 1.0
        return dist_x

