"""
TirraMind — Differentiable SDE Module (Math Stack M1)

Provides differentiable SDE models as nn.Modules built on torchsde.
Models: GBM, HestonSDE.  Solvers: euler, milstein via torchsde.sdeint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchsde


@dataclass
class SDEConfig:
    method: str = "euler"
    dt: float = 1.0 / 252
    adaptive: bool = False
    rtol: float = 1e-5
    atol: float = 1e-5
    adjoint: bool = False


class _SDEBase(nn.Module):
    noise_type: str = "diagonal"
    sde_type: str = "ito"

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def simulate(
        self,
        y0: torch.Tensor,
        ts: torch.Tensor,
        config: Optional[SDEConfig] = None,
        n_samples: int = 1,
    ) -> torch.Tensor:
        cfg = config or SDEConfig()
        if n_samples > 1:
            y0 = y0.repeat_interleave(n_samples, dim=0)
        sdeint = torchsde.sdeint_adjoint if cfg.adjoint else torchsde.sdeint
        return sdeint(
            self,
            y0,
            ts,
            method=cfg.method,
            dt=cfg.dt,
            adaptive=cfg.adaptive,
            rtol=cfg.rtol,
            atol=cfg.atol,
        )


class GBM(_SDEBase):
    """dS = mu * S * dt + sigma * S * dW"""

    def __init__(self, mu: float = 0.05, sigma: float = 0.20, learnable: bool = True):
        super().__init__()
        raw_mu = torch.tensor(mu, dtype=torch.float32)
        raw_sigma = torch.tensor(sigma, dtype=torch.float32)
        if learnable:
            self.mu = nn.Parameter(raw_mu)
            self.sigma = nn.Parameter(raw_sigma)
        else:
            self.register_buffer("mu", raw_mu)
            self.register_buffer("sigma", raw_sigma)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.mu * y

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.sigma * y


class HestonSDE(_SDEBase):
    """dS = mu*S*dt + sqrt(V)*S*dW1,  dV = kappa*(theta-V)*dt + xi*sqrt(V)*dW2,  corr=rho"""

    noise_type = "general"

    def __init__(
        self,
        mu: float = 0.05,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.30,
        rho: float = -0.70,
        learnable: bool = True,
    ):
        super().__init__()
        for name, val in [
            ("mu", mu),
            ("kappa", kappa),
            ("theta", theta),
            ("xi", xi),
            ("rho", rho),
        ]:
            t = torch.tensor(val, dtype=torch.float32)
            if learnable:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        S, V = y[:, 0:1], y[:, 1:2]
        return torch.cat(
            [self.mu * S, self.kappa * (self.theta - torch.clamp(V, min=0.0))], dim=1
        )

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        S, V = y[:, 0:1], y[:, 1:2]
        sv = torch.sqrt(torch.clamp(V, min=0.0))
        vs, vv = sv * S, self.xi * sv
        rc = torch.clamp(self.rho, -0.999, 0.999)
        L22 = torch.sqrt(1.0 - rc**2)
        return torch.cat([vs, vs * rc, torch.zeros_like(vv), vv * L22], dim=1).reshape(
            -1, 2, 2
        )


def make_time_grid(
    T: float = 1.0, n_steps: int = 252, device: Optional[torch.device] = None
) -> torch.Tensor:
    return torch.linspace(0.0, T, n_steps + 1, device=device)
