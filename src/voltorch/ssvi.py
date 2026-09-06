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


class ESSVI(nn.Module):
    """Extended SSVI (Hendriks & Martini, "The extended SSVI volatility
    surface", 2019): one (theta_i, rho_i, psi_i) per expiry,

        w_i(k) = 1/2 * (theta_i + rho_i*psi_i*k + sqrt((psi_i*k + theta_i*rho_i)^2 + theta_i^2*(1 - rho_i^2)))

    (SSVI is the special case psi_i = theta_i*phi(theta_i), rho_i = rho.)
    Free of butterfly arbitrage per slice iff psi_i(1+|rho_i|) < 4 and
    psi_i^2(1+|rho_i|) <= 4*theta_i; free of calendar arbitrage between
    consecutive slices i < j iff theta_i <= theta_j, psi_i <= psi_j and
    |rho_j*psi_j - rho_i*psi_i| <= psi_j - psi_i. All four are enforced by
    the parameterisation below, so the surface stays arbitrage-free at every
    optimiser step while fitting each expiry's skew separately -- which is
    what SSVI's single rho cannot do on a crypto chain spanning 2 to 300 days.
    """

    def __init__(self, expiries: torch.Tensor, theta_init: torch.Tensor | None = None,
                 psi_init: torch.Tensor | None = None, rho_init: float = -0.1):
        super().__init__()
        T, order = torch.sort(expiries.to(torch.float64).flatten())
        if (T <= 0).any():
            raise ValueError("expiries must be positive year fractions")
        self.register_buffer("T", T)
        n = len(T)
        theta_init = (0.04 * T) if theta_init is None else theta_init.to(torch.float64).flatten()[order]
        psi_init = (0.3 * torch.sqrt(theta_init)) if psi_init is None else psi_init.to(torch.float64).flatten()[order]
        self.raw_theta = nn.Parameter(_inv_softplus(torch.diff(theta_init, prepend=torch.zeros(1, dtype=torch.float64)).clamp(min=1e-8)))
        self.raw_psi = nn.Parameter(_inv_softplus(torch.diff(psi_init, prepend=torch.zeros(1, dtype=torch.float64)).clamp(min=1e-8)))
        # rho_1*psi_1 = psi_1*tanh(raw_1); rho_j*psi_j = rho_i*psi_i + (psi_j - psi_i)*tanh(raw_j)
        self.raw_rho = nn.Parameter(torch.full((n,), math.atanh(max(min(rho_init, 0.99), -0.99)), dtype=torch.float64))
        self.project()

    @property
    def _dt(self):
        return self.raw_rho.dtype

    @property
    def theta(self) -> torch.Tensor:
        return torch.cumsum(nn.functional.softplus(self.raw_theta), 0)

    @property
    def psi(self) -> torch.Tensor:
        return torch.cumsum(nn.functional.softplus(self.raw_psi), 0)

    @property
    def rho_psi(self) -> torch.Tensor:
        psi = self.psi
        t = torch.tanh(self.raw_rho)
        incr = torch.diff(psi, prepend=torch.zeros(1, dtype=psi.dtype, device=psi.device))
        return torch.cumsum(incr * t, 0)          # |rho_j psi_j| <= psi_j by construction

    @property
    def rho(self) -> torch.Tensor:
        return self.rho_psi / self.psi.clamp(min=1e-12)

    @torch.no_grad()
    def project(self) -> None:
        """Butterfly conditions bound psi_i from above by theta_i and rho_i.
        Shrink the psi *increments* so every slice satisfies them while psi
        stays non-decreasing (the bound is applied as a running minimum
        from the last expiry backwards)."""
        th, rho = self.theta, self.rho
        r = 1.0 + rho.abs()
        bound = torch.minimum(4.0 / r * 0.999, torch.sqrt(4.0 * th / r))
        target = torch.minimum(self.psi, bound)
        target = torch.flip(torch.cummin(torch.flip(target, [0]), 0).values, [0])   # non-decreasing after capping
        incr = torch.diff(target, prepend=torch.zeros(1, dtype=target.dtype, device=target.device)).clamp(min=1e-8)
        self.raw_psi.copy_(_inv_softplus(incr))

    def satisfies_conditions(self) -> bool:
        th, psi, rp = self.theta, self.psi, self.rho_psi
        rho = rp / psi
        r = 1.0 + rho.abs()
        bfly = ((psi * r) < 4.0).all() and ((psi * psi * r) <= 4.0 * th + 1e-12).all()
        cal = (torch.diff(th) >= 0).all() and (torch.diff(psi) >= 0).all() and \
              (torch.diff(rp).abs() <= torch.diff(psi) + 1e-12).all()
        return bool(bfly and cal)

    def _interp(self, v: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        Ts = self.T
        T = T.to(self._dt)
        if len(Ts) == 1:
            return v[0] * T / Ts[0]
        idx = torch.clamp(torch.searchsorted(Ts, T, right=True) - 1, 0, len(Ts) - 2)
        t0, t1 = Ts[idx], Ts[idx + 1]
        lam = ((T - t0) / (t1 - t0)).clamp(0.0, 1.0)
        return v[idx] + lam * (v[idx + 1] - v[idx])

    def total_variance(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        k = k.to(self._dt)
        th = self._interp(self.theta, T); psi = self._interp(self.psi, T); rp = self._interp(self.rho_psi, T)
        rho = rp / psi.clamp(min=1e-12)
        return 0.5 * (th + rp * k + torch.sqrt((psi * k + th * rho) ** 2 + th ** 2 * (1.0 - rho ** 2)))

    def implied_vol(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        T = T.to(self._dt)
        return torch.sqrt(self.total_variance(k, T) / T.clamp(min=1e-8))

    def fit(self, k, T, iv, weights=None, adam_steps: int = 400, lbfgs_steps: int = 60, lr: float = 0.05) -> dict:
        dev = self.raw_rho.device
        k, T, iv = (x.to(self._dt).to(dev) for x in (k, T, iv))
        w = torch.ones_like(iv) if weights is None else weights.to(self._dt).to(dev)
        w = w / w.mean()

        def loss_fn():
            return (w * (self.implied_vol(k, T) - iv) ** 2).mean()

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(adam_steps):
            opt.zero_grad(); loss = loss_fn(); loss.backward(); opt.step(); self.project()
        if lbfgs_steps:
            lb = torch.optim.LBFGS(self.parameters(), max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

            def closure():
                lb.zero_grad(); loss = loss_fn(); loss.backward(); return loss
            lb.step(closure); self.project()
        with torch.no_grad():
            resid = self.implied_vol(k, T) - iv
            return {"rmse_vol_pts": float(torch.sqrt((resid ** 2).mean()) * 100), "weighted_loss": float(loss_fn()),
                    "conditions": self.satisfies_conditions(),
                    "rho": [float(x) for x in self.rho], "psi": [float(x) for x in self.psi],
                    "theta": [float(x) for x in self.theta]}
