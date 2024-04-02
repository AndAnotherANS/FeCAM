import unittest

import torch
from models.covariance_optimization import mahal_chol, mahal, normalize_chol, normalize_full
from scipy.spatial import distance
from torch.nn import functional as F

class MyTestCase(unittest.TestCase):
    def test_mahalanobis_equal_to_scipy(self):
        dummy_cov_lower = torch.tril(torch.rand([20, 20]) + 0.5)
        dummy_cov = dummy_cov_lower @ dummy_cov_lower.T
        dummy_cov_inv = torch.linalg.pinv(dummy_cov)
        dummy_mean = torch.zeros([1, 20])
        dummy_x = F.normalize(torch.normal(2, 4, [1, 20]), p=2, dim=-1)

        scipy_mahal = torch.tensor(distance.mahalanobis(dummy_x.squeeze().numpy(), dummy_mean.squeeze().numpy(), dummy_cov_inv.numpy())) ** 2
        full_mahal = mahal(dummy_x, dummy_mean, dummy_cov)
        cholesky_mahal = mahal_chol(dummy_x, dummy_mean, dummy_cov_lower)

        self.assertTrue(torch.allclose(scipy_mahal, full_mahal, atol=1e-3))
        self.assertTrue(torch.allclose(scipy_mahal, cholesky_mahal, atol=1e-3))


    def test_normalization(self):
        dummy_cov_lower = torch.tril(torch.rand([20, 20]) + 0.5)
        dummy_cov = dummy_cov_lower @ dummy_cov_lower.T

        norm_chol = normalize_chol(dummy_cov_lower)
        norm_full = normalize_full(dummy_cov)

        self.assertTrue(torch.allclose(norm_chol @ norm_chol.T, norm_full))
        self.assertTrue(torch.allclose(torch.diag(norm_chol @ norm_chol.T), torch.ones(20)))
        self.assertTrue(torch.allclose(torch.diag(norm_full), torch.ones(20)))


if __name__ == '__main__':
    unittest.main()
