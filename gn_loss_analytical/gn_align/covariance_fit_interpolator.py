import torch
from .utils.utils import inverse2x2
from .utils.covariance import mean, covariance
from .utils.distributions import stable_l1_normalization

class CovarianceFitInterpolator1:

    def __init__(self, point_xy_n, xs_n, ys_channels, point_batch=1000, device = 'cuda'):
        """
        point_n:      number of 3D points
        xs_n:         number of keypoints
        ys_channels:  number of channels in keypoint features
        """
        self.point_xy_n = point_xy_n
        self.xs_n = xs_n
        self.ys_channels = ys_channels

        self.device = device
        self.point_batch = point_batch
        self.approximated_version = True

        self.alloc_tensors()
        pass
    
    def alloc_tensors(self):
        self.values = torch.zeros((self.point_xy_n, self.ys_channels), device = self.device)
        self.valids = torch.ones((self.point_xy_n), dtype = torch.bool, device = self.device)
        self.jacobians = torch.zeros((self.point_xy_n, self.ys_channels, 2), device = self.device)
        self.probabilities = torch.zeros((self.point_xy_n, self.xs_n), device = self.device)

        self.Cov_xx_all = torch.zeros((self.point_xy_n, 2, 2), device = self.device)

    def __call__(self, point_xy, xs, ys, point_pdf, inverse_precision):
        """
        point_xy:    (B, 2) 3D points projections onto image
        xs:       (B, N, 2) keypoint positions
        ys:       (B, N, C) keypoint features
        """
        if (point_xy.shape[0] > self.point_xy_n) or \
           (xs.shape[0] > self.xs_n) or (ys.shape[-1] > self.ys_channels):
            self.point_xy_n = point_xy.shape[0]
            self.xs_n = xs.shape[0]
            self.ys_channels = ys.shape[-1]
            self.alloc_tensors()

        point_xy_n = point_xy.shape[0]
        xs_n = xs.shape[0]
        ys_channels = ys.shape[-1]

        for i in range(0, point_xy_n, self.point_batch):
            i_next = min(i+self.point_batch, point_xy_n)
            point_xy_b = point_xy[i:i_next]
            self.probabilities[i:i_next, :xs_n] = point_pdf(point_xy_b, xs)
        
        self.probabilities[:point_xy_n, :xs_n] = stable_l1_normalization(self.probabilities[:point_xy_n, :xs_n])

        self.valids = ~torch.isnan(self.probabilities[:point_xy_n, :xs_n]).any(dim=-1)
        if self.approximated_version:
            for i in range(0, point_xy_n, self.point_batch):
                i_next = min(i+self.point_batch, point_xy_n)
                valid = self.valids[i:i_next]
                ps = self.probabilities[i:i_next, :xs_n][valid]
                
                mean_x = mean(ps, xs)
                self.Cov_xx_all[i:i_next][valid] = covariance(ps, xs, xs, mean_x, mean_x)

            self.Cov_xx_all[:point_xy_n], self.valids[:point_xy_n] = \
                inverse2x2(self.Cov_xx_all[:point_xy_n], self.valids[:point_xy_n], inverse_precision)

            for i in range(0, point_xy_n, self.point_batch):
                i_next = min(i+self.point_batch, point_xy_n)
                valid = self.valids[i:i_next]

                point_xy_b = point_xy[i:i_next][valid]
                ps = self.probabilities[i:i_next, :xs_n][valid]

                mean_x = mean(ps, xs)
                mean_y = mean(ps, ys)
                # for some reason (ys, xs) takes 2x memory
                Cov_yx = covariance(ps, xs, ys, mean_x, mean_y).mT
                J = Cov_yx @ self.Cov_xx_all[i:i_next][valid]

                self.values[i:i_next][valid] = (J @ (point_xy_b - mean_x).unsqueeze(-1)).squeeze(-1) + mean_y
                self.jacobians[i:i_next][valid] = J
        
        else:
            for i in range(0, point_xy_n, self.point_batch):
                i_next = min(i+self.point_batch, point_xy_n)
                valid = self.valids[i:i_next]

                ps = self.probabilities[i:i_next, :xs_n][valid]
                point_xy_b = point_xy[i:i_next][valid]

                mean_x = mean(ps, xs)
                mean_y = mean(ps, ys)

                Cov_yy = covariance(ps, ys, ys, mean_y, mean_y)
                Cov_yx = covariance(ps, xs, ys, mean_x, mean_y).mT

                J_inv = torch.linalg.lstsq(Cov_yy, Cov_yx).solution.mT

                J = J_inv.mT @ torch.linalg.pinv(J_inv @ J_inv.mT)

                self.values[i:i_next][valid] = (J @ (point_xy_b - mean_x).unsqueeze(-1)).squeeze(-1) + mean_y
                self.jacobians[i:i_next][valid] = J


        return self.values[:point_xy_n], self.jacobians[:point_xy_n], self.valids[:point_xy_n]

               

