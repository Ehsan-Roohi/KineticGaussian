from __future__ import annotations

import unittest

import torch

from kgfr.conditional_models import ConditionalPhaseGaussianMixture


class ConditionalModelTest(unittest.TestCase):
    def test_shape_finiteness_and_gradient(self) -> None:
        torch.manual_seed(5)
        model = ConditionalPhaseGaussianMixture(num_kernels=12, variant="xvx", mach_degree=2)
        z = torch.rand(17, 4) * 2.0 - 1.0
        logf = model(torch.tensor(-0.6), z)
        self.assertEqual(tuple(logf.shape), (17,))
        self.assertTrue(torch.isfinite(logf).all())
        loss = (logf**2).mean()
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_per_sample_mach(self) -> None:
        model = ConditionalPhaseGaussianMixture(num_kernels=7, variant="diag", mach_degree=1)
        z = torch.zeros(5, 4)
        mach = torch.linspace(-1.0, 1.0, 5)
        prediction = model(mach, z)
        self.assertEqual(tuple(prediction.shape), (5,))


if __name__ == "__main__":
    unittest.main()
