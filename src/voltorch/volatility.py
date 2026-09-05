"""
TirraMind — Implied Volatility Surface Module (Math Stack M5)

Provides SVI parameterization, SABR modeling, and surface feature extraction
for feeding low-dimensional volatility structures into GNN nodes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════
# SVI Parameterization (Gatheral 2014)
# ═══════════════════════════════════════════════════════════════


class SVIParameterization(nn.Module):
    """Raw SVI parameterization of a total implied variance slice:
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    """

    def __init__(
        self,
        a: float = 0.04,
        b: float = 0.10,
        rho: float = -0.50,
        m: float = 0.0,
        sigma: float = 0.10,
        learnable: bool = True,
    ):
        super().__init__()
        raw_a = torch.tensor(a, dtype=torch.float32)
        raw_b = torch.tensor(b, dtype=torch.float32)
        raw_rho = torch.tensor(rho, dtype=torch.float32)
        raw_m = torch.tensor(m, dtype=torch.float32)
        raw_sigma = torch.tensor(sigma, dtype=torch.float32)

        if learnable:
            self.raw_a = nn.Parameter(raw_a)
            self.raw_b = nn.Parameter(raw_b)
            self.raw_rho = nn.Parameter(raw_rho)
            self.raw_m = nn.Parameter(raw_m)
            self.raw_sigma = nn.Parameter(raw_sigma)
        else:
            self.register_buffer("raw_a", raw_a)
            self.register_buffer("raw_b", raw_b)
            self.register_buffer("raw_rho", raw_rho)
            self.register_buffer("raw_m", raw_m)
            self.register_buffer("raw_sigma", raw_sigma)

    @property
    def a(self) -> torch.Tensor:
        return self.raw_a

    @property
    def b(self) -> torch.Tensor:
        return torch.clamp(self.raw_b, min=1e-6)

    @property
    def rho(self) -> torch.Tensor:
        return torch.clamp(self.raw_rho, min=-0.999, max=0.999)

    @property
    def m(self) -> torch.Tensor:
        return self.raw_m

    @property
    def sigma(self) -> torch.Tensor:
        return torch.clamp(self.raw_sigma, min=1e-4)

    def total_variance(self, k: torch.Tensor) -> torch.Tensor:
        """Calculate total implied variance w(k) for log-moneyness k."""
        k_m = k - self.m
        term = self.rho * k_m + torch.sqrt(k_m**2 + self.sigma**2)
        w = self.a + self.b * term
        return torch.clamp(w, min=1e-6)

    def implied_volatility(
        self, k: torch.Tensor, T: float | torch.Tensor
    ) -> torch.Tensor:
        """Calculate implied volatility sigma(k) for log-moneyness k and maturity T."""
        w = self.total_variance(k)
        T_safe = (
            torch.clamp(T, min=1e-5) if isinstance(T, torch.Tensor) else max(T, 1e-5)
        )
        return torch.sqrt(w / T_safe)

    def check_butterfly_arbitrage(self, k: torch.Tensor) -> torch.Tensor:
        """Compute Durrleman's g(k) density check function.
        No butterfly arbitrage requires g(k) >= 0 for all k.
        """
        w = self.total_variance(k)
        k_m = k - self.m
        h = torch.sqrt(k_m**2 + self.sigma**2)

        # First derivative w'(k)
        w_prime = self.b * (self.rho + k_m / h)

        # Second derivative w''(k)
        w_prime_prime = self.b * (self.sigma**2 / (h**3))

        # Durrleman condition g(k)
        g_k = (
            (1.0 - k * w_prime / (2.0 * w)) ** 2
            - (w_prime**2 / 4.0) * (1.0 / w + 0.25)
            + w_prime_prime / 2.0
        )
        return g_k

    def fit(
        self,
        k_market: torch.Tensor,
        w_market: torch.Tensor,
        lr: float = 1e-2,
        epochs: int = 1000,
    ) -> float:
        """Fit SVI parameters to market total variance data via gradient descent."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            w_pred = self.total_variance(k_market)
            loss = loss_fn(w_pred, w_market)
            loss.backward()
            optimizer.step()

        return loss.item()


# ═══════════════════════════════════════════════════════════════
# SABR Stochastic Volatility Model (Hagan 2002)
# ═══════════════════════════════════════════════════════════════


class SABRModel(nn.Module):
    """Vectorized, fully differentiable SABR stochastic volatility model
    using Hagan's asymptotic implied volatility expansion.
    """

    def __init__(
        self,
        alpha: float = 0.20,
        beta: float = 1.0,
        rho: float = -0.50,
        nu: float = 0.40,
        learnable: bool = True,
    ):
        super().__init__()
        raw_alpha = torch.tensor(alpha, dtype=torch.float32)
        raw_rho = torch.tensor(rho, dtype=torch.float32)
        raw_nu = torch.tensor(nu, dtype=torch.float32)

        self.beta = beta

        if learnable:
            self.raw_alpha = nn.Parameter(raw_alpha)
            self.raw_rho = nn.Parameter(raw_rho)
            self.raw_nu = nn.Parameter(raw_nu)
        else:
            self.register_buffer("raw_alpha", raw_alpha)
            self.register_buffer("raw_rho", raw_rho)
            self.register_buffer("raw_nu", raw_nu)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.clamp(self.raw_alpha, min=1e-4)

    @property
    def rho(self) -> torch.Tensor:
        return torch.clamp(self.raw_rho, min=-0.999, max=0.999)

    @property
    def nu(self) -> torch.Tensor:
        return torch.clamp(self.raw_nu, min=1e-4)

    def forward(
        self, K: torch.Tensor, F: torch.Tensor, T: torch.Tensor
    ) -> torch.Tensor:
        """Compute SABR Hagan implied volatility.

        Args:
            K: Strike price tensor.
            F: Forward asset price tensor.
            T: Maturity tensor.
        """
        K = torch.clamp(K, min=1e-6)
        F = torch.clamp(F, min=1e-6)
        T = torch.clamp(T, min=1e-5)

        alpha = self.alpha
        beta = self.beta
        rho = self.rho
        nu = self.nu

        F_mid = torch.sqrt(F * K)
        log_FK = torch.log(F / K)

        # Denominator 1 (expansion of forward/strike geometry)
        one_beta = 1.0 - beta
        log_FK_2 = log_FK**2
        den1 = (F_mid**one_beta) * (
            1.0 + (one_beta**2 / 24.0) * log_FK_2 + (one_beta**4 / 1920.0) * (log_FK**4)
        )

        # Singularity-protected z and chi(z) calculation
        z = (nu / alpha) * (F_mid**one_beta) * log_FK
        sqrt_term = torch.sqrt(1.0 - 2.0 * rho * z + z**2)
        x_z = torch.log((sqrt_term + z - rho) / (1.0 - rho))

        # ATM limit expansion for z/chi(z)
        ratio_num = z / torch.where(
            torch.abs(x_z) < 1e-6, torch.ones_like(x_z) * 1e-6, x_z
        )
        ratio_taylor = 1.0 - 0.5 * rho * z + ((2.0 - 3.0 * rho**2) / 12.0) * (z**2)
        ratio = torch.where(torch.abs(z) < 1e-4, ratio_taylor, ratio_num)

        # Term 3 (time expansion correction)
        term3_num = (one_beta**2 / 24.0) * (
            alpha**2 / torch.clamp(F_mid ** (2.0 * one_beta), min=1e-6)
        )
        term3_mid = 0.25 * rho * beta * nu * alpha / (F_mid**one_beta)
        term3_end = ((2.0 - 3.0 * (rho**2)) / 24.0) * (nu**2)
        term3 = 1.0 + (term3_num + term3_mid + term3_end) * T

        return (alpha / den1) * ratio * term3

    def fit(
        self,
        K_market: torch.Tensor,
        F: torch.Tensor,
        T: torch.Tensor,
        iv_market: torch.Tensor,
        lr: float = 1e-2,
        epochs: int = 1000,
    ) -> float:
        """Fit SABR parameters to market implied volatilities via gradient descent."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            iv_pred = self.forward(K_market, F, T)
            loss = loss_fn(iv_pred, iv_market)
            loss.backward()
            optimizer.step()

        return loss.item()


# ═══════════════════════════════════════════════════════════════
# Implied Volatility Surface
# ═══════════════════════════════════════════════════════════════


class ImpliedVolatilitySurface(nn.Module):
    """Implied Volatility Surface parameterization by aggregating SVI slices
    and supporting robust bilinear total-variance interpolation across strikes and maturities.
    """

    def __init__(self, slices: dict[float, SVIParameterization]):
        super().__init__()
        self.maturities = sorted(list(slices.keys()))
        self.slices = nn.ModuleDict(
            {self._get_key(T): slices[T] for T in self.maturities}
        )

    def _get_key(self, T: float | torch.Tensor) -> str:
        T_val = T.item() if isinstance(T, torch.Tensor) else float(T)
        # Avoid dots inside key names as required by PyTorch nn.ModuleDict
        return f"slice_{str(T_val).replace('.', '_')}"

    def get_vol(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Compute interpolated implied volatility for arbitrary log-moneyness k and maturity T
        using non-arbitrage preserving total variance linear interpolation.
        """
        if len(self.maturities) == 1:
            # Single slice fallback
            slice_svi = self.slices[self._get_key(self.maturities[0])]
            return slice_svi.implied_volatility(k, T)

        # Find closest maturity slices
        T_min = self.maturities[0]
        T_max = self.maturities[-1]

        # Clamp T boundary slices to prevent extrapolations outside domain bounds
        if T <= T_min:
            return self.slices[self._get_key(T_min)].implied_volatility(k, T)
        if T >= T_max:
            return self.slices[self._get_key(T_max)].implied_volatility(k, T)

        # Locate interval index
        prev_idx = 0
        for i, mat in enumerate(self.maturities):
            if mat <= T:
                prev_idx = i
            else:
                break

        T_prev = self.maturities[prev_idx]
        T_next = self.maturities[prev_idx + 1]

        w_prev = self.slices[self._get_key(T_prev)].total_variance(k)
        w_next = self.slices[self._get_key(T_next)].total_variance(k)

        # Linear interpolation in total variance space (w = sigma^2 * T) to exclude calendar arbitrage
        weight = (T - T_prev) / (T_next - T_prev)
        w_interp = (1.0 - weight) * w_prev + weight * w_next

        return torch.sqrt(torch.clamp(w_interp, min=1e-6) / T)

    def extract_gnn_features(self) -> dict[str, torch.Tensor]:
        """Extract low-dimensional implied volatility surface node features for GNN conditioning.

        Features extracted at standard reference T = 0.5 (or closest available):
            - atm_vol: ATM Implied Volatility (k = 0).
            - skew: ATM Volatility Skew d(sigma)/dk at k=0.
            - curvature: ATM Volatility Smile Curvature d2(sigma)/dk^2 at k=0.
            - term_slope: ATM Volatility Term Structure Slope (T_max - T_min).
            - SVI params: SVI parameters [a, b, rho, m, sigma] of the reference slice.
        """
        # Pick reference slice
        ref_T = 0.5
        closest_T = min(self.maturities, key=lambda x: abs(x - ref_T))
        ref_slice = self.slices[self._get_key(closest_T)]

        # ATM point log-moneyness
        k_atm = torch.tensor([0.0], dtype=torch.float32)

        # Extract SVI parameters
        a = ref_slice.a
        b = ref_slice.b
        rho = ref_slice.rho
        m = ref_slice.m
        sigma = ref_slice.sigma

        # Implied Vol and total variance at ATM
        w = ref_slice.total_variance(k_atm)
        sig = ref_slice.implied_volatility(k_atm, closest_T)

        # Calculate exact analytical derivatives of total variance w at ATM
        k_m = -m
        h = torch.sqrt(k_m**2 + sigma**2)
        w_prime = b * (rho + k_m / h)
        w_prime_prime = b * (sigma**2 / (h**3))

        # Convert variance derivatives to implied volatility derivatives at ATM
        # sigma = sqrt(w / T)
        # d(sigma)/dk = w' / (2 * T * sigma)
        skew = w_prime / (2.0 * closest_T * sig)

        # d2(sigma)/dk^2 = (2 * w * w'' - w'^2) / (4 * T^2 * sigma^3)
        curvature = (2.0 * w * w_prime_prime - w_prime**2) / (
            4.0 * (closest_T**2) * (sig**3)
        )

        # Term structure slope
        sig_short = self.slices[self._get_key(self.maturities[0])].implied_volatility(
            k_atm, self.maturities[0]
        )
        sig_long = self.slices[self._get_key(self.maturities[-1])].implied_volatility(
            k_atm, self.maturities[-1]
        )
        term_slope = (
            (sig_long - sig_short) / (self.maturities[-1] - self.maturities[0])
            if len(self.maturities) > 1
            else torch.zeros_like(sig)
        )

        return {
            "atm_vol": sig,
            "skew": skew,
            "curvature": curvature,
            "term_slope": term_slope,
            "svi_a": a.unsqueeze(0),
            "svi_b": b.unsqueeze(0),
            "svi_rho": rho.unsqueeze(0),
            "svi_m": m.unsqueeze(0),
            "svi_sigma": sigma.unsqueeze(0),
        }
