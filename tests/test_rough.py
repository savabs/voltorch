"""Unit tests for agent.quant.rough — M6 Rough Volatility module."""

from __future__ import annotations

import math
import pytest
import torch
from voltorch.rough import RoughBergomiModel, estimate_hurst_exponent
from voltorch.options import implied_volatility, BlackScholes


class TestRoughVolatility:
    def test_rough_bergomi_paths_and_clamping(self):
        """Verify rBergomi model clamps parameters and generates positive paths."""
        # Intentionally initialize with violating bounds
        rbergomi = RoughBergomiModel(
            H=-0.1, eta=-1.0, rho=1.5, xi_0=-0.05, learnable=False
        )

        assert abs(rbergomi.H.item() - 0.01) < 1e-6
        assert abs(rbergomi.eta.item() - 1e-4) < 1e-9
        assert abs(rbergomi.rho.item() - 0.999) < 1e-6
        assert abs(rbergomi.xi_0.item() - 1e-4) < 1e-9

        # Generate paths with normal bounds
        rbergomi_normal = RoughBergomiModel(
            H=0.07, eta=2.0, rho=-0.90, xi_0=0.04, learnable=False
        )
        S, V = rbergomi_normal.generate_paths(n_paths=100, n_steps=50, T=1.0)

        assert S.shape == (51, 100)
        assert V.shape == (51, 100)

        # Variance must be strictly positive
        assert torch.all(V > 0.0)
        # Spot price should also be strictly positive
        assert torch.all(S > 0.0)

    def test_hurst_exponent_estimation(self):
        """Verify Hurst exponent estimator correctly extracts H from simulated rough paths."""
        rbergomi = RoughBergomiModel(
            H=0.10, eta=1.5, rho=0.0, xi_0=0.04, learnable=False
        )

        # Simulate log-variance paths (which behave like fractional Brownian motion of index H)
        S, V = rbergomi.generate_paths(n_paths=1, n_steps=200, T=1.0)
        log_vol_path = 0.5 * torch.log(V.squeeze())

        H_est = estimate_hurst_exponent(log_vol_path, max_lag=15)

        # The estimated Hurst should be rough (H < 0.5) and within a reasonable range
        assert H_est.item() > 0.0
        assert H_est.item() < 0.5

    def test_rbergomi_exploding_atm_skew_signature(self):
        """Verify the exploding ATM implied volatility skew signature (skew ~ T^{H-1/2})
        by pricing Monte Carlo options across short vs long maturities.
        """
        rbergomi = RoughBergomiModel(
            H=0.07, eta=2.5, rho=-0.90, xi_0=0.04, learnable=False
        )

        S0 = 100.0
        r = 0.05

        # We will price European call options at two maturities: T_short vs T_long
        maturities = [0.1, 0.5]
        skews = []

        for T in maturities:
            # Generate paths
            S, V = rbergomi.generate_paths(n_paths=10000, n_steps=100, T=T)
            S_T = S[-1]  # Spot price at maturity T

            # Compute call prices for strikes around S0: K_down=98, K_up=102
            K_down = torch.tensor([98.0], dtype=torch.float32)
            K_up = torch.tensor([102.0], dtype=torch.float32)

            payoff_down = torch.clamp(S_T - K_down, min=0.0)
            payoff_up = torch.clamp(S_T - K_up, min=0.0)

            # Average discounted payoffs to get Monte Carlo call prices
            disc = math.exp(-r * T)
            price_down = disc * payoff_down.mean()
            price_up = disc * payoff_up.mean()

            # Solve for implied volatility using Newton-Raphson from options.py
            S_tensor = torch.tensor([S0], dtype=torch.float32)
            T_tensor = torch.tensor([T], dtype=torch.float32)
            r_tensor = torch.tensor([r], dtype=torch.float32)

            # Use flat standard init as initial guess
            iv_down = implied_volatility(
                S_tensor, K_down, T_tensor, r_tensor, price_down.unsqueeze(0)
            )
            iv_up = implied_volatility(
                S_tensor, K_up, T_tensor, r_tensor, price_up.unsqueeze(0)
            )

            # Implied volatility slope (skew) near ATM: | d(sig)/dK |
            skew = abs(iv_up.item() - iv_down.item()) / (K_up.item() - K_down.item())
            skews.append(skew)

        # For a rough volatility model (H < 0.5), the ATM skew increases as T -> 0
        # Thus, short maturity skew must be significantly larger than long maturity skew
        print(
            f"Rough Vol ATM Skew: Short T ({maturities[0]}) = {skews[0]:.6f}, Long T ({maturities[1]}) = {skews[1]:.6f}"
        )
        assert skews[0] > skews[1]
