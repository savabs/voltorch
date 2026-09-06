"""SSVI: one implied-volatility surface across all expiries, arbitrage-free by
construction (Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces",
2014).

Total variance at log-moneyness k and ATM total variance theta:

    w(k, theta) = theta/2 * (1 + rho*phi(theta)*k + sqrt((phi(theta)*k + rho)^2 + 1 - rho^2))
    phi(theta)  = eta / (theta^gamma * (1 + theta)^(1 - gamma)),   0 < gamma <= 1

Theorem 4.2 of the paper: the surface is free of butterfly arbitrage iff

    theta * phi(theta) * (1 + |rho|) < 4        and
    theta * phi(theta)^2 * (1 + |rho|) <= 4

for every theta on the surface, and free of calendar arbitrage iff theta is
non-decreasing in T. Both are enforced structurally here -- theta by a
cumulative softplus over sorted expiries, the two inequalities by projecting
eta after every optimiser step -- so a fitted surface cannot violate them.
Durrleman's g(k) >= 0 is still checked in the tests, as a *test* of the
implementation, never as the constraint.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x))


class SSVI(nn.Module):
    def __init__(
        self,
        expiries: torch.Tensor,
        theta_init: torch.Tensor | None = None,
        rho: float = -0.2,
        eta: float = 1.0,
        gamma: float = 0.5,
    ):
        super().__init__()
        T, order = torch.sort(expiries.to(torch.float64).flatten())
        if (T <= 0).any():
            raise ValueError("expiries must be positive year fractions")
        self.register_buffer("T", T)
        n = len(T)
        if theta_init is None:
            theta_init = 0.04 * T
        theta_init = theta_init.to(torch.float64).flatten()[order]
        incr = torch.diff(theta_init, prepend=torch.zeros(1, dtype=torch.float64)).clamp(min=1e-6)
        self.raw_theta = nn.Parameter(_inv_softplus(incr))
        self.raw_rho = nn.Parameter(torch.atanh(torch.tensor(rho, dtype=torch.float64).clamp(-0.99, 0.99)))
        self.raw_eta = nn.Parameter(_inv_softplus(torch.tensor(eta, dtype=torch.float64)))
        self.raw_gamma = nn.Parameter(torch.logit(torch.tensor(gamma, dtype=torch.float64).clamp(0.01, 0.99)))
        self.project()

    # -- parameters --------------------------------------------------------
    @property
    def theta(self) -> torch.Tensor:          # (n,), non-decreasing: no calendar arbitrage
        return torch.cumsum(nn.functional.softplus(self.raw_theta), 0)

    @property
    def rho(self) -> torch.Tensor:
        return torch.tanh(self.raw_rho)

    @property
    def eta(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_eta)

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gamma)

    def phi(self, theta: torch.Tensor) -> torch.Tensor:
        g = self.gamma
        return self.eta / (theta.clamp(min=1e-12) ** g * (1.0 + theta) ** (1.0 - g))

    # -- the two inequalities -----------------------------------------------
    @torch.no_grad()
    def project(self) -> None:
        """Shrink eta until Theorem 4.2 holds at every expiry. Called after
        each optimiser step, so the constraint is never merely penalised."""
        th = self.theta
        g = self.gamma
        r = 1.0 + self.rho.abs()
        base = th ** g * (1.0 + th) ** (1.0 - g)           # phi = eta / base
        bound1 = (4.0 * base / (th * r)).min() * 0.999      # theta*phi*(1+|rho|) < 4
        bound2 = torch.sqrt(4.0 * base ** 2 / (th * r)).min()  # theta*phi^2*(1+|rho|) <= 4
        eta_max = torch.minimum(bound1, bound2)
        if self.eta > eta_max:
            self.raw_eta.copy_(_inv_softplus(eta_max))

    def satisfies_theorem(self) -> bool:
        th, r = self.theta, 1.0 + self.rho.abs()
        p = self.phi(th)
        return bool(((th * p * r) < 4.0).all() and ((th * p * p * r) <= 4.0 + 1e-12).all())

    # -- surface -----------------------------------------------------------
    @property
    def _dt(self):
        return self.raw_rho.dtype

    def theta_at(self, T: torch.Tensor) -> torch.Tensor:
        """ATM total variance at arbitrary T: linear in T between expiries
        (monotone, so calendar-free), linearly extended outside the range."""
        th = self.theta
        Ts = self.T
        T = T.to(self._dt)
        if len(Ts) == 1:
            return th[0] * T / Ts[0]
        idx = torch.clamp(torch.searchsorted(Ts, T, right=True) - 1, 0, len(Ts) - 2)
        t0, t1 = Ts[idx], Ts[idx + 1]
        w0, w1 = th[idx], th[idx + 1]
        slope = (w1 - w0) / (t1 - t0)
        out = w0 + slope * (T - t0)
        return out.clamp(min=1e-10)

    def total_variance(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        k = k.to(self._dt)
        th = self.theta_at(T)
        p = self.phi(th)
        rho = self.rho
        return 0.5 * th * (1.0 + rho * p * k + torch.sqrt((p * k + rho) ** 2 + 1.0 - rho ** 2))

    def implied_vol(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        T = T.to(self._dt)
        return torch.sqrt(self.total_variance(k, T) / T.clamp(min=1e-8))

    # -- fitting -----------------------------------------------------------
    def fit(
        self,
        k: torch.Tensor,
        T: torch.Tensor,
        iv: torch.Tensor,
        weights: torch.Tensor | None = None,
        adam_steps: int = 400,
        lbfgs_steps: int = 60,
        lr: float = 0.05,
    ) -> dict:
        """Weighted least squares in implied-vol space. Adam to get near,
        L-BFGS to finish; eta projected after every step of both."""
        dev = self.raw_rho.device
        k, T, iv = (x.to(self._dt).to(dev) for x in (k, T, iv))
        w = torch.ones_like(iv) if weights is None else weights.to(self._dt).to(dev)
        w = w / w.mean()

        def loss_fn():
            return (w * (self.implied_vol(k, T) - iv) ** 2).mean()

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(adam_steps):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            opt.step()
            self.project()
        if lbfgs_steps:
            lb = torch.optim.LBFGS(self.parameters(), max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

            def closure():
                lb.zero_grad()
                loss = loss_fn()
                loss.backward()
                return loss
            lb.step(closure)
            self.project()
        with torch.no_grad():
            resid = self.implied_vol(k, T) - iv
            return {
                "rmse_vol_pts": float(torch.sqrt((resid ** 2).mean()) * 100),
                "weighted_loss": float(loss_fn()),
                "theorem_4_2": self.satisfies_theorem(),
                "rho": float(self.rho), "eta": float(self.eta), "gamma": float(self.gamma),
                "theta": [float(x) for x in self.theta],
            }
