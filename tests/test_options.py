"""Tests for agent.quant.options — M2 Options Pricing & Greeks module."""

from __future__ import annotations

import math
import pytest
import torch
from voltorch.options import (
    BlackScholes,
    implied_volatility,
    BaroneAdesiWhaley,
    HestonCOS,
    BatesCOS,
    MertonCOS,
    VarianceGammaCOS,
)


class TestBlackScholes:
    def test_smoke_pricing_call(self):
        """Smoke test validating call pricing matches standard BS values."""
        bs = BlackScholes()
        # Inputs
        S = torch.tensor([100.0], dtype=torch.float64)
        K = torch.tensor([100.0], dtype=torch.float64)
        T = torch.tensor([1.0], dtype=torch.float64)
        r = torch.tensor([0.05], dtype=torch.float64)
        sigma = torch.tensor([0.20], dtype=torch.float64)

        # BS price for S=100, K=100, T=1, r=0.05, sigma=0.20 should be approx 10.45058
        price = bs(S, K, T, r, sigma, is_call=True)
        assert abs(price.item() - 10.45058) < 1e-4

    def test_smoke_pricing_put(self):
        """Smoke test validating put pricing matches standard BS values."""
        bs = BlackScholes()
        S = torch.tensor([100.0], dtype=torch.float64)
        K = torch.tensor([100.0], dtype=torch.float64)
        T = torch.tensor([1.0], dtype=torch.float64)
        r = torch.tensor([0.05], dtype=torch.float64)
        sigma = torch.tensor([0.20], dtype=torch.float64)

        # BS price for put should be approx 5.5735
        price = bs(S, K, T, r, sigma, is_call=False)
        assert abs(price.item() - 5.5735) < 1e-4

    def test_with_dividends(self):
        """Verify that dividend yield reduces call price and increases put price."""
        bs = BlackScholes()
        S = torch.tensor([100.0])
        K = torch.tensor([100.0])
        T = torch.tensor([1.0])
        r = torch.tensor([0.05])
        sigma = torch.tensor([0.20])
        q = torch.tensor([0.03])

        price_call_no_div = bs(S, K, T, r, sigma, is_call=True)
        price_call_with_div = bs(S, K, T, r, sigma, q=q, is_call=True)
        assert price_call_with_div.item() < price_call_no_div.item()

        price_put_no_div = bs(S, K, T, r, sigma, is_call=False)
        price_put_with_div = bs(S, K, T, r, sigma, q=q, is_call=False)
        assert price_put_with_div.item() > price_put_no_div.item()

    def test_vectorized_multidimensional(self):
        """Verify that BlackScholes handles multidimensional and batched tensors."""
        bs = BlackScholes()
        # Batch of 4 pricing inputs
        S = torch.tensor([100.0, 95.0, 105.0, 100.0])
        K = torch.tensor([100.0, 100.0, 100.0, 110.0])
        T = torch.tensor([1.0, 0.5, 0.25, 2.0])
        r = torch.tensor([0.05, 0.05, 0.05, 0.05])
        sigma = torch.tensor([0.20, 0.25, 0.15, 0.30])

        prices = bs(S, K, T, r, sigma, is_call=True)
        assert prices.shape == (4,)
        assert not torch.isnan(prices).any()

    def test_analytical_greeks(self):
        """Verify analytical Greeks match known reference values for ATM call."""
        bs = BlackScholes()
        S = torch.tensor([100.0], dtype=torch.float64)
        K = torch.tensor([100.0], dtype=torch.float64)
        T = torch.tensor([1.0], dtype=torch.float64)
        r = torch.tensor([0.05], dtype=torch.float64)
        sigma = torch.tensor([0.20], dtype=torch.float64)

        greeks = bs.analytical_greeks(S, K, T, r, sigma, is_call=True)

        # Known reference values for these inputs:
        # delta ≈ 0.6368, gamma ≈ 0.0188, vega ≈ 37.52, theta ≈ -6.41, rho ≈ 53.23
        assert abs(greeks["delta"].item() - 0.6368) < 0.01
        assert abs(greeks["gamma"].item() - 0.0188) < 0.001
        assert abs(greeks["vega"].item() - 37.52) < 0.5
        assert abs(greeks["theta"].item() - (-6.41)) < 0.5
        assert abs(greeks["rho"].item() - 53.23) < 0.5

    def test_autograd_vs_analytical_greeks(self):
        """Verify autograd Greeks match analytical Greeks to 1e-5."""
        bs = BlackScholes()
        S = torch.tensor([100.0, 95.0], dtype=torch.float64)
        K = torch.tensor([100.0, 100.0], dtype=torch.float64)
        T = torch.tensor([1.0, 0.5], dtype=torch.float64)
        r = torch.tensor([0.05, 0.05], dtype=torch.float64)
        sigma = torch.tensor([0.20, 0.25], dtype=torch.float64)

        analytical = bs.analytical_greeks(S, K, T, r, sigma, is_call=True)
        autograd = bs.autograd_greeks(S, K, T, r, sigma, is_call=True)

        for greek in ["delta", "gamma", "vega", "theta", "rho"]:
            diff = torch.abs(analytical[greek] - autograd[greek])
            assert torch.all(diff < 1e-5), f"{greek} mismatch: {diff}"


class TestImpliedVolatility:
    def test_iv_roundtrip(self):
        """Verify implied vol recovers original sigma from BS price."""
        S = torch.tensor([100.0], dtype=torch.float64)
        K = torch.tensor([100.0], dtype=torch.float64)
        T = torch.tensor([1.0], dtype=torch.float64)
        r = torch.tensor([0.05], dtype=torch.float64)
        true_sigma = torch.tensor([0.20], dtype=torch.float64)

        bs = BlackScholes()
        market_price = bs(S, K, T, r, true_sigma, is_call=True)

        iv = implied_volatility(S, K, T, r, market_price, is_call=True)
        assert abs(iv.item() - true_sigma.item()) < 1e-4

    def test_iv_differentiability(self):
        """Verify implied volatility is differentiable w.r.t market price."""
        S = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)
        K = torch.tensor([100.0], dtype=torch.float64)
        T = torch.tensor([1.0], dtype=torch.float64)
        r = torch.tensor([0.05], dtype=torch.float64)
        market_price = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)

        iv = implied_volatility(S, K, T, r, market_price, is_call=True)
        grad = torch.autograd.grad(iv.sum(), market_price, create_graph=True)[0]
        assert grad is not None
        assert not torch.isnan(grad).any()


class TestBaroneAdesiWhaley:
    def test_american_price_ge_european(self):
        """Verify American option price >= European price."""
        baw = BaroneAdesiWhaley()
        bs = BlackScholes()
        S = torch.tensor([100.0], dtype=torch.float64)
        K = torch.tensor([100.0], dtype=torch.float64)
        T = torch.tensor([1.0], dtype=torch.float64)
        r = torch.tensor([0.05], dtype=torch.float64)
        sigma = torch.tensor([0.20], dtype=torch.float64)
        q = torch.tensor([0.03], dtype=torch.float64)

        am_call = baw(S, K, T, r, sigma, q=q, is_call=True)
        eur_call = bs(S, K, T, r, sigma, q=q, is_call=True)
        assert am_call.item() >= eur_call.item() - 1e-6

        am_put = baw(S, K, T, r, sigma, q=q, is_call=False)
        eur_put = bs(S, K, T, r, sigma, q=q, is_call=False)
        assert am_put.item() >= eur_put.item() - 1e-6

    def test_american_early_exercise_premium(self):
        """Verify deep ITM American put shows early exercise premium."""
        baw = BaroneAdesiWhaley()
        bs = BlackScholes()
        S = torch.tensor([50.0], dtype=torch.float64)
        K = torch.tensor([100.0], dtype=torch.float64)
        T = torch.tensor([1.0], dtype=torch.float64)
        r = torch.tensor([0.05], dtype=torch.float64)
        sigma = torch.tensor([0.20], dtype=torch.float64)

        am_put = baw(S, K, T, r, sigma, is_call=False)
        eur_put = bs(S, K, T, r, sigma, is_call=False)
        # Deep ITM American put should have a significant early exercise premium
        assert am_put.item() > eur_put.item() + 0.1

    def test_no_dividend_call_equals_european(self):
        """Verify American call without dividends equals European call."""
        baw = BaroneAdesiWhaley()
        bs = BlackScholes()
        S = torch.tensor([100.0], dtype=torch.float64)
        K = torch.tensor([100.0], dtype=torch.float64)
        T = torch.tensor([1.0], dtype=torch.float64)
        r = torch.tensor([0.05], dtype=torch.float64)
        sigma = torch.tensor([0.20], dtype=torch.float64)

        am_call = baw(S, K, T, r, sigma, is_call=True)
        eur_call = bs(S, K, T, r, sigma, is_call=True)
        assert abs(am_call.item() - eur_call.item()) < 1e-4


class TestFourierCOS:
    def test_merton_cos_limiting_case(self):
        """Merton Jump Diffusion with zero jumps should equal Black-Scholes."""
        merton = MertonCOS(sigma=0.20, lambda_j=0.0, mu_j=0.0, sigma_j=0.10)
        bs = BlackScholes()

        S = torch.tensor([100.0], dtype=torch.float32)
        K = torch.tensor([100.0], dtype=torch.float32)
        T = torch.tensor([1.0], dtype=torch.float32)
        r = torch.tensor([0.05], dtype=torch.float32)

        p_merton_call = merton(S, K, T, r, is_call=True)
        p_bs_call = bs(S, K, T, r, torch.tensor([0.20]), is_call=True)

        assert abs(p_merton_call.item() - p_bs_call.item()) < 1e-4

    def test_heston_cos_pricing(self):
        """Verify Heston COS pricing is stable and matches standard analytical reference values."""
        heston = HestonCOS(kappa=2.0, theta=0.04, xi=0.30, rho=-0.70, v0=0.04)

        S = torch.tensor([100.0], dtype=torch.float32)
        K = torch.tensor([100.0], dtype=torch.float32)
        T = torch.tensor([1.0], dtype=torch.float32)
        r = torch.tensor([0.05], dtype=torch.float32)

        p_heston_call = heston(S, K, T, r, is_call=True)
        # Reference analytical Heston price for these parameters is 10.3942
        assert abs(p_heston_call.item() - 10.3942) < 1e-3

    def test_bates_vs_heston(self):
        """Bates with zero jump intensity should equal Heston."""
        bates = BatesCOS(
            kappa=2.0, theta=0.04, xi=0.30, rho=-0.70, v0=0.04, lambda_j=0.0
        )
        heston = HestonCOS(kappa=2.0, theta=0.04, xi=0.30, rho=-0.70, v0=0.04)

        S = torch.tensor([100.0], dtype=torch.float32)
        K = torch.tensor([100.0], dtype=torch.float32)
        T = torch.tensor([1.0], dtype=torch.float32)
        r = torch.tensor([0.05], dtype=torch.float32)

        p_bates = bates(S, K, T, r, is_call=True)
        p_heston = heston(S, K, T, r, is_call=True)

        assert abs(p_bates.item() - p_heston.item()) < 1e-4

    def test_variance_gamma_prices(self):
        """Verify Variance Gamma pricing is stable and produces positive prices."""
        vg = VarianceGammaCOS(sigma=0.20, nu=0.10, theta=-0.10)
        S = torch.tensor([100.0], dtype=torch.float32)
        K = torch.tensor([100.0], dtype=torch.float32)
        T = torch.tensor([1.0], dtype=torch.float32)
        r = torch.tensor([0.05], dtype=torch.float32)

        p_call = vg(S, K, T, r, is_call=True)
        p_put = vg(S, K, T, r, is_call=False)

        assert p_call.item() > 0.0
        assert p_put.item() > 0.0
        # Call price should be greater than put price for r=0.05 at S=K=100
        assert p_call.item() > p_put.item()

    def test_cos_differentiability(self):
        """Verify that Heston COS options pricing is fully differentiable w.r.t underlying Spot price and model parameters."""
        heston = HestonCOS(
            kappa=2.0, theta=0.04, xi=0.30, rho=-0.70, v0=0.04, learnable=True
        )

        S = torch.tensor([100.0], dtype=torch.float32, requires_grad=True)
        K = torch.tensor([100.0], dtype=torch.float32)
        T = torch.tensor([1.0], dtype=torch.float32)
        r = torch.tensor([0.05], dtype=torch.float32)

        price = heston(S, K, T, r, is_call=True)
        price_sum = price.sum()

        # Delta via autograd
        delta = torch.autograd.grad(price_sum, S, create_graph=True)[0]
        assert delta is not None
        assert delta.item() > 0.0

        # Gradient w.r.t Heston parameters (e.g. v0)
        grad_v0 = torch.autograd.grad(price_sum, heston.v0)[0]
        assert grad_v0 is not None
        # Option price should increase with initial variance v0 (positive vega-like derivative)
        assert grad_v0.item() > 0.0
