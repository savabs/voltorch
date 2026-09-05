"""
TirraMind — Rough Volatility Module (Math Stack M6)

Provides rough Bergomi (rBergomi) model simulation, Bennedsen-Lunde-Pakkanen (BLP)
hybrid simulation scheme, and empirical Hurst exponent estimation.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class RoughBergomiModel(nn.Module):
    """Rough Bergomi (rBergomi) model (Bayer, Friz, Gatheral 2016)
    simulating variance and spot price paths using the Bennedsen-Lunde-Pakkanen (BLP)
    hybrid simulation scheme.
    """

    def __init__(
        self,
        H: float = 0.07,
        eta: float = 2.0,
        rho: float = -0.90,
        xi_0: float = 0.04,
        learnable: bool = True,
    ):
        super().__init__()
        raw_H = torch.tensor(H, dtype=torch.float32)
        raw_eta = torch.tensor(eta, dtype=torch.float32)
        raw_rho = torch.tensor(rho, dtype=torch.float32)
        raw_xi_0 = torch.tensor(xi_0, dtype=torch.float32)

        if learnable:
            self.raw_H = nn.Parameter(raw_H)
            self.raw_eta = nn.Parameter(raw_eta)
            self.raw_rho = nn.Parameter(raw_rho)
            self.raw_xi_0 = nn.Parameter(raw_xi_0)
        else:
            self.register_buffer("raw_H", raw_H)
            self.register_buffer("raw_eta", raw_eta)
            self.register_buffer("raw_rho", raw_rho)
            self.register_buffer("raw_xi_0", raw_xi_0)

    @property
    def H(self) -> torch.Tensor:
        # H must be strictly between 0.0 and 0.5
        return torch.clamp(self.raw_H, min=0.01, max=0.499)

    @property
    def eta(self) -> torch.Tensor:
        # Vol-of-vol must be positive
        return torch.clamp(self.raw_eta, min=1e-4)

    @property
    def rho(self) -> torch.Tensor:
        # Correlation must be in [-0.999, 0.999]
        return torch.clamp(self.raw_rho, min=-0.999, max=0.999)

    @property
    def xi_0(self) -> torch.Tensor:
        # Forward variance must be positive
        return torch.clamp(self.raw_xi_0, min=1e-4)

    def _simulate_volterra_path(self, Z: torch.Tensor, T: float) -> torch.Tensor:
        """Simulate the Riemann-Liouville fractional Volterra paths Y_t using
        the Bennedsen-Lunde-Pakkanen (BLP) hybrid scheme in PyTorch.

        Args:
            Z: Brownian increments tensor of shape (n_steps, n_paths).
            T: Maturity/horizon of simulation.

        Returns:
            Y_total: Volterra paths of shape (n_steps+1, n_paths).
        """
        n_steps, n_paths = Z.shape
        dt = T / n_steps
        device = Z.device
        H = self.H

        # Construct strictly lower triangular weight matrix M of shape (n_steps, n_steps)
        # rows corresponds to step index i, cols corresponds to increment index p < i.
        rows = torch.arange(n_steps, dtype=torch.float32, device=device).unsqueeze(1)
        cols = torch.arange(n_steps, dtype=torch.float32, device=device).unsqueeze(0)
        diff = rows - cols

        mask = diff > 0
        j = torch.where(mask, diff, torch.ones_like(diff))

        alpha = H + 0.5
        b_j = (dt**H / alpha) * ((j + 1.0) ** alpha - j**alpha)
        M = torch.where(mask, b_j, torch.zeros_like(b_j))

        # 1. History Part: Vectorized matrix multiplication instead of costly loops
        Y_history = torch.matmul(M, Z)

        # 2. Local Part: Exact variance matching integration over the last step [t_{i-1}, t_i]
        Y_local = (dt**H / torch.sqrt(2.0 * H)) * Z

        # Total Volterra process
        Y = Y_history + Y_local

        # Prepend zero boundary condition at t = 0
        zeros = torch.zeros(1, n_paths, device=device, dtype=torch.float32)
        return torch.cat([zeros, Y], dim=0)

    def generate_paths(
        self, n_paths: int, n_steps: int, T: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate spot price and variance paths under the rough Bergomi model.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time discretization steps.
            T: Maturity/horizon of simulation.

        Returns:
            S: Spot price paths of shape (n_steps+1, n_paths).
            V: Variance paths of shape (n_steps+1, n_paths).
        """
        device = self.raw_H.device
        dt = T / n_steps

        # Generate correlated Brownian increments
        Z = torch.randn(n_steps, n_paths, device=device)
        Z_perp = torch.randn(n_steps, n_paths, device=device)

        # 1. Simulate Volterra fractional Brownian paths
        Y = self._simulate_volterra_path(Z, T)

        # 2. Calculate variance paths V_t
        t = torch.linspace(0.0, T, n_steps + 1, device=device).unsqueeze(
            -1
        )  # (n_steps+1, 1)
        exponent = self.eta * torch.sqrt(2.0 * self.H) * Y - 0.5 * (self.eta**2) * (
            t ** (2.0 * self.H)
        )
        V = self.xi_0 * torch.exp(exponent)

        # 3. Simulate spot price process S_t (Euler-Maruyama in log-space)
        x = torch.zeros(n_steps + 1, n_paths, device=device)
        x[0] = math.log(100.0)  # Reference spot S0 = 100.0

        for i in range(n_steps):
            V_prev = V[i]
            dZ = (
                self.rho * Z[i] + torch.sqrt(1.0 - self.rho**2) * Z_perp[i]
            ) * math.sqrt(dt)
            x[i + 1] = x[i] + (-0.5 * V_prev) * dt + torch.sqrt(V_prev) * dZ

        S = torch.exp(x)
        return S, V


def estimate_hurst_exponent(series: torch.Tensor, max_lag: int = 20) -> torch.Tensor:
    """Estimate the Hurst exponent H of a log-volatility sequence
    using linear scaling regression of quadratic increments over multiple lags.

    Args:
        series: 1D PyTorch tensor representing the sequence (e.g. log realized volatility).
        max_lag: Maximum lag to perform scaling analysis over.

    Returns:
        H: Estimated Hurst parameter of shape ().
    """
    series = series.flatten()
    N = series.shape[0]

    lags = torch.arange(1, max_lag + 1, dtype=torch.float32, device=series.device)
    log_lags = torch.log(lags)

    log_variances = []
    for lag in range(1, max_lag + 1):
        # Compute quadratic increments for given lag
        diff = series[lag:] - series[:-lag]
        var = torch.mean(diff**2)
        log_variances.append(torch.log(var))

    log_vars = torch.stack(log_variances)

    # Perform linear regression to estimate slope of log_vars vs log_lags
    # Slope = Cov(log_lags, log_vars) / Var(log_lags)
    mean_u = torch.mean(log_lags)
    mean_v = torch.mean(log_vars)

    u_diff = log_lags - mean_u
    v_diff = log_vars - mean_v

    slope = torch.sum(u_diff * v_diff) / torch.sum(u_diff**2)

    # Since E[(x_{t+dt} - x_t)^2] is proportional to dt^(2H), the slope is 2H
    H = 0.5 * slope
    return torch.clamp(H, min=0.01, max=0.99)
