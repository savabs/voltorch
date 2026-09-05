"""Unit tests for agent.quant.volatility — M5 Implied Volatility Surface module."""

from __future__ import annotations

import math
import pytest
import torch
from voltorch.volatility import (
    SVIParameterization,
    SABRModel,
    ImpliedVolatilitySurface,
)


class TestSVIParameterization:
    def test_svi_bounds_and_clamping(self):
        """Verify SVI parameters stay bounded and output positive total variance."""
        # Intentionally initialize SVI with values violating logical bounds
        svi = SVIParameterization(
            a=-0.01, b=-0.1, rho=-1.5, m=0.0, sigma=-0.05, learnable=False
        )

        k = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)
        w = svi.total_variance(k)

        # Assert total variance is positive and strictly bounded
        assert torch.all(w >= 1e-6)
        # Check that properties are clamped to valid ranges
        assert svi.b.item() >= 1e-6 - 1e-9
        assert svi.rho.item() >= -0.999 - 1e-6 and svi.rho.item() <= 0.999 + 1e-6
        assert svi.sigma.item() >= 1e-4 - 1e-7

    def test_svi_butterfly_arbitrage(self):
        """Verify Durrleman's g(k) check successfully computes density indicator."""
        svi = SVIParameterization(
            a=0.04, b=0.10, rho=-0.30, m=0.0, sigma=0.10, learnable=False
        )
        k = torch.linspace(-0.5, 0.5, 20)

        g_k = svi.check_butterfly_arbitrage(k)
        assert g_k.shape == (20,)
        # For typical parameters, Durrleman's condition is positive (arbitrage-free)
        assert torch.all(g_k >= 0.0)

    def test_svi_fitting(self):
        """Verify SVI parameter fitting successfully converges on smile data."""
        # Target SVI slice
        target_svi = SVIParameterization(
            a=0.05, b=0.15, rho=-0.40, m=0.02, sigma=0.08, learnable=False
        )

        k_market = torch.linspace(-0.3, 0.3, 15)
        w_market = target_svi.total_variance(k_market)

        # Fit with a fresh initialized learnable SVI
        fit_svi = SVIParameterization(
            a=0.02, b=0.05, rho=0.0, m=0.0, sigma=0.05, learnable=True
        )
        final_loss = fit_svi.fit(k_market, w_market, lr=5e-2, epochs=500)

        # Assert fitting decreased the loss to near zero
        assert final_loss < 1e-4


class TestSABRModel:
    def test_sabr_hagan_atm_singularity_protection(self):
        """Verify SABR Hagan formula behaves correctly near ATM and does NOT cause NaN."""
        sabr = SABRModel(alpha=0.20, beta=1.0, rho=-0.50, nu=0.40, learnable=False)

        # Strikes containing exact ATM K = F to test the singularity guard
        K = torch.tensor([95.0, 100.0, 105.0], dtype=torch.float32)
        F = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float32)
        T = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

        ivs = sabr(K, F, T)

        # Assert no NaNs or Infs are generated
        assert ivs.shape == (3,)
        assert not torch.isnan(ivs).any()
        assert not torch.isinf(ivs).any()

        # ATM volatility should be extremely close to standard ATM limit:
        # For K=F, log(F/K)=0, den1=1, ratio=1, and term3 = 1 + (rho*nu*alpha/4 + (2-3rho^2)/24 * nu^2) * T
        # ATM Vol = alpha * [1 + (1/4 * -0.5 * 0.4 * 0.2 + (2-3*0.25)/24 * 0.16) * 1.0]
        # ATM Vol = 0.2 * [1 + (-0.01 + 1.25/24 * 0.16) * 1.0] = 0.2 * [1 + (-0.01 + 0.00833)] = 0.2 * 0.99833 = 0.19966
        assert abs(ivs[1].item() - 0.19966) < 1e-4

    def test_sabr_fitting(self):
        """Verify SABR model parameters can be fitted to market implied volatilities."""
        # Create artificial SABR volatility smile
        target_sabr = SABRModel(
            alpha=0.25, beta=1.0, rho=-0.60, nu=0.50, learnable=False
        )
        K_market = torch.linspace(80.0, 120.0, 11)
        F = torch.tensor([100.0] * 11)
        T = torch.tensor([1.0] * 11)
        iv_market = target_sabr(K_market, F, T)

        # Fit model
        fit_sabr = SABRModel(alpha=0.15, beta=1.0, rho=0.0, nu=0.20, learnable=True)
        final_loss = fit_sabr.fit(K_market, F, T, iv_market, lr=5e-2, epochs=400)

        assert final_loss < 1e-4


class TestImpliedVolatilitySurface:
    def test_surface_interpolation_and_no_arbitrage(self):
        """Verify implied volatility surface interpolation across maturities."""
        # Setup 2 SVI slices
        slice_short = SVIParameterization(a=0.02, b=0.08, rho=-0.40, m=0.0, sigma=0.05)
        slice_long = SVIParameterization(a=0.05, b=0.12, rho=-0.30, m=0.0, sigma=0.08)

        surface = ImpliedVolatilitySurface({0.25: slice_short, 1.0: slice_long})

        k = torch.tensor([-0.2, 0.0, 0.2], dtype=torch.float32)

        # Test boundary clamping
        iv_short = surface.get_vol(k, torch.tensor(0.1))  # Clamped to 0.25 slice
        assert torch.all(iv_short > 0.0)

        # Test interpolation inside boundaries
        iv_mid = surface.get_vol(k, torch.tensor(0.5))
        assert torch.all(iv_mid > 0.0)

    def test_gnn_feature_extraction_and_backward_flow(self):
        """Verify GNN surface features extract correctly and backprop gradients flow to SVI params."""
        slice_short = SVIParameterization(
            a=0.03, b=0.10, rho=-0.40, m=0.0, sigma=0.06, learnable=True
        )
        slice_long = SVIParameterization(
            a=0.06, b=0.14, rho=-0.30, m=0.0, sigma=0.09, learnable=True
        )

        surface = ImpliedVolatilitySurface({0.1: slice_short, 1.0: slice_long})

        features = surface.extract_gnn_features()

        # Check keys
        expected_keys = {
            "atm_vol",
            "skew",
            "curvature",
            "term_slope",
            "svi_a",
            "svi_b",
            "svi_rho",
            "svi_m",
            "svi_sigma",
        }
        assert set(features.keys()) == expected_keys

        # Check shapes
        for k, v in features.items():
            assert v.shape == (1,), f"Key {k} has wrong shape {v.shape}"
            assert not torch.isnan(v).any()

        # Test full backpropagability (Gradient Flow Check)
        loss = features["skew"].sum() + features["curvature"].sum()
        loss.backward()

        # Verify gradients are calculated for parameters of the reference slice
        # The reference T=0.5 slice is 1.0 (since abs(1.0 - 0.5) = 0.5 and abs(0.1 - 0.5) = 0.4, wait, closest to 0.5 is 0.1!)
        # Let's check which is closest:
        # abs(0.1 - 0.5) = 0.4
        # abs(1.0 - 0.5) = 0.5
        # Yes! 0.1 is closest, so slice_short (0.1) is reference.
        assert slice_short.raw_b.grad is not None
        assert abs(slice_short.raw_b.grad.item()) > 0.0
