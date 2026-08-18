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

    def test_amplitude_only_conditioning_keeps_kernel_geometry_shared(self) -> None:
        model = ConditionalPhaseGaussianMixture(
            num_kernels=7,
            variant="xvx",
            mach_degree=1,
            conditioning="amplitude",
        )
        with torch.no_grad():
            model.log_amp_coeff[1].fill_(0.5)
        centers, scales, amps, corr = model._conditioned(torch.tensor([-1.0, 1.0]))
        self.assertTrue(torch.allclose(centers[0], centers[1]))
        self.assertTrue(torch.allclose(scales[0], scales[1]))
        self.assertTrue(torch.allclose(corr[0], corr[1]))
        self.assertFalse(torch.allclose(amps[0], amps[1]))
        self.assertEqual(model.parameter_count(), 7 * (4 + 4 + 1 + 2))


if __name__ == "__main__":
    unittest.main()
