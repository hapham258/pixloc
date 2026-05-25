from gn_align.utils.distributions import NormalDistributionPdf, NormalDistributionPdf2d, UnifromDistanceDistributionPdf, UnifromDistanceDistributionPdf2d, UnifromKNNDistributionPdf
from gn_align.extractors.spoint_extractor import SuperPointExtractor
from gn_align.covariance_fit_interpolator import CovarianceFitInterpolator1
from pixloc.pixlib.geometry.interpolation import interpolate_tensor_bilinear

import argparse
import torch
import torchvision
import numpy as np
import cv2
import os

def svd_values(tensor, svd=None, min_max=None):
    with torch.no_grad():
        ttt = torchvision.transforms.functional.resize(tensor, [tensor.shape[-2]//8, tensor.shape[-1]//8])
        mi = torch.quantile(ttt, torch.tensor([0.05]).cuda())
        ma = torch.quantile(ttt, torch.tensor([0.95]).cuda())
        tensor = torch.clip(tensor, mi, ma)
        tensor = tensor.detach()
        tensor = tensor.permute(1, 2, 0)
        h, w, c = tensor.shape
        values = tensor.reshape(-1, tensor.shape[2])
        values = values - values.mean(dim=0)
        vals = torch.einsum("ji,jk->ik", [values, values])
        vals /= values.shape[0]
        if svd is None:
            U, S, V = torch.linalg.svd(vals, full_matrices=True)
            svd = V[:3, :]
        tensor = (svd @ tensor.reshape(h, w, c, 1)).squeeze(-1).cpu().numpy()

        if min_max is None:
            min_max = [np.min(tensor)]
            tensor -= np.min(tensor)
            min_max.append(np.max(tensor))
            tensor /= np.max(tensor)
        else:
            tensor -= min_max[0]
            tensor /= min_max[1]
        tensor *= 255
        return tensor.astype(np.uint8), svd, min_max



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    input_image = args.input_image
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    extractor = SuperPointExtractor(3, 0.0005, 4096)
    cfi = CovarianceFitInterpolator1(4096, 2000, 256, 1000)

    xs, image_scale = extractor.extract_keypoints(input_image)
    ys, ys_scale = extractor.extract_descriptors(input_image)
    ys = interpolate_tensor_bilinear(ys, xs * torch.tensor(ys_scale).reshape(1, 2).to('cuda'))[0]
    
    diag = image_scale.norm()

    distributions = []
    distributions.append(UnifromDistanceDistributionPdf(0.2 * diag))
    distributions.append(UnifromDistanceDistributionPdf(0.2 * diag))
    distributions.append(UnifromDistanceDistributionPdf(0.1 * diag))
    
    distributions.append(UnifromDistanceDistributionPdf2d(0.2 * diag, 0.1 * diag))
    distributions.append(UnifromDistanceDistributionPdf2d(0.1 * diag, 0.05 * diag))

    distributions.append(NormalDistributionPdf(0, 0.15 * diag))
    distributions.append(NormalDistributionPdf(0, 0.08 * diag))
    distributions.append(NormalDistributionPdf(0, 0.03 * diag))
    distributions.append(NormalDistributionPdf(0, 0.01 * diag))

    distributions.append(NormalDistributionPdf2d(torch.tensor([0,0]).float().cuda(), torch.tensor([[0.05 * diag, 0],[0, 0.03 * diag]]).cuda()))
    distributions.append(NormalDistributionPdf2d(torch.tensor([0,0]).float().cuda(), torch.tensor([[0.03 * diag, 0],[0, 0.01 * diag]]).cuda()))

    w, h = image_scale.reshape(2).int().cpu().numpy()
    xy = torch.meshgrid(
        torch.linspace(0, w - 1, w).int(),
        torch.linspace(0, h - 1, h).int())
    xys = torch.concat((xy[0].reshape(-1, 1), xy[1].reshape(-1, 1)),
                            dim=1).float().cuda()
    
    batch_step = 2000
    svd, min_max = None, None
    torch.cuda.empty_cache()
    for iteration, dist in enumerate(distributions):
        print("Processing distribution", iteration, f'Out of {len(distributions)}')
        out_t = torch.empty((ys.shape[-1], h, w)).cuda()
        with torch.no_grad():
            for i in range(0, xys.shape[0], batch_step):
                alphas = xys[i: i + batch_step,:]
                values, jacobiands, valid = cfi(alphas, xs, ys, dist, 1e-5)
                out_t[:, alphas[:, 1][valid].long(), alphas[:, 0][valid].long()] = values[valid].T
                del values
                del valid
            new_img, svd, min_max = svd_values(out_t, svd, min_max)
            cv2.imwrite(os.path.join(output_dir, f"{iteration}.png"), new_img)
        del out_t
        torch.cuda.empty_cache()

