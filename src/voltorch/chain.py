"""fit_chain: one call from a raw chain to a publishable report.

The report is the product. It answers, for a chain at one instant:
  * how well an arbitrage-free SSVI surface fits the two-sided OTM quotes
    (RMSE in vol points, share of quotes where the fit lies inside bid-ask);
  * whether the venue's own marks contain arbitrage that exceeds the bid-ask
    spread (butterfly / vertical in price space, calendar in variance space);
  * that the fitted surface contains none (Durrleman on a grid + calendar);
  * that autograd greeks agree with closed-form Black-76.
Everything is computed from the chain passed in; nothing is looked up.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch

from .arbitrage import butterfly_violations, calendar_violations, durrleman_g, vertical_violations
from .options import BlackScholes
from .ssvi import SSVI

_N = torch.distributions.Normal(0.0, 1.0)


def _black76_greeks_closed_form(F, K, T, sigma, is_call):
    """delta, gamma, vega, theta for Black-76 with r = 0 (undiscounted)."""
    sq = sigma * torch.sqrt(T)
    d1 = (torch.log(F / K) + 0.5 * sigma ** 2 * T) / sq
    d2 = d1 - sq
    pdf = torch.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    delta = torch.where(is_call, _N.cdf(d1), _N.cdf(d1) - 1.0)
    gamma = pdf / (F * sq)
    vega = F * pdf * torch.sqrt(T)
    theta = -F * pdf * sigma / (2 * torch.sqrt(T))
    return delta, gamma, vega, theta


def otm(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-the-money, two-sided quotes: calls above the forward, puts
    below. Deep-ITM prices are dominated by intrinsic value and their
    implied vols are numerically meaningless."""
    m = df["two_sided"] & df["bid_iv"].gt(1e-3) & df["ask_iv"].gt(df["bid_iv"])
    m &= (df["is_call"] & (df["strike"] >= df["forward"])) | (~df["is_call"] & (df["strike"] < df["forward"]))
    return df[m].copy()


@dataclass
class FitReport:
    currency: str
    as_of: str
    n_quotes: int
    n_two_sided: int
    n_fit: int
    expiries: int
    fit: dict
    inside_bid_ask_share: float
    venue_violations: dict
    our_violations: dict
    greeks_max_abs_err: dict
    timings_s: dict
    device: str
    slices: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=float, indent=1)


def fit_chain(df: pd.DataFrame, *, currency: str = "", device: str = "cpu", as_of: str = "",
              adam_steps: int = 300, lbfgs_steps: int = 60) -> FitReport:
    t0 = time.time()
    q = otm(df)
    if len(q) < 30:
        raise ValueError(f"only {len(q)} usable OTM quotes")
    q["k"] = np.log(q["strike"] / q["forward"])
    q["mid_iv"] = 0.5 * (q["bid_iv"] + q["ask_iv"])
    q["spread"] = (q["ask_iv"] - q["bid_iv"]).clip(lower=0.002)
    expiries = np.sort(q["T"].unique())

    # -- fit -----------------------------------------------------------------
    theta0 = torch.tensor([float((q[q["T"] == t].assign(a=lambda d: d.k.abs()).nsmallest(3, "a")["mid_iv"] ** 2).mean() * t)
                           for t in expiries], dtype=torch.float64)
    model = SSVI(torch.tensor(expiries, dtype=torch.float64), theta_init=theta0, rho=-0.1, eta=0.8, gamma=0.5)
    if device != "cpu":
        model = model.to(torch.float32).to(device)
    k = torch.tensor(q["k"].values); T = torch.tensor(q["T"].values); iv = torch.tensor(q["mid_iv"].values)
    w = torch.tensor(1.0 / q["spread"].values ** 2)
    t1 = time.time()
    fit = model.fit(k, T, iv, w, adam_steps=adam_steps, lbfgs_steps=lbfgs_steps)
    t_fit = time.time() - t1
    model = model.to("cpu").to(torch.float64)
    with torch.no_grad():
        fitted = model.implied_vol(torch.tensor(q["k"].values), torch.tensor(q["T"].values)).numpy()
    q["fit_iv"] = fitted
    inside = float(((fitted >= q["bid_iv"].values) & (fitted <= q["ask_iv"].values)).mean())

    # -- venue marks: arbitrage beyond the spread ----------------------------
    bs = BlackScholes()
    venue = {"butterfly": [], "vertical": [], "calendar": []}
    cal_slices = []
    slices_out = []
    for t in expiries:
        s = q[q["T"] == t].sort_values("strike")
        F = float(s["forward"].median())
        # put-call parity (r = 0): C = P + F - K, so every quote becomes a call price
        K = torch.tensor(s["strike"].values); Tt = torch.full((len(s),), float(t), dtype=torch.float64)
        Ft = torch.full((len(s),), F, dtype=torch.float64); z = torch.zeros(len(s), dtype=torch.float64)
        mark_c = bs(Ft, K, Tt, z, torch.tensor(s["mark_iv"].values), is_call=True).numpy()
        bid_c = bs(Ft, K, Tt, z, torch.tensor(s["bid_iv"].values), is_call=True).numpy()
        ask_c = bs(Ft, K, Tt, z, torch.tensor(s["ask_iv"].values), is_call=True).numpy()
        tol = np.maximum(ask_c - bid_c, 1e-9)          # a violation must exceed the spread in price
        for v in butterfly_violations(s["strike"].values, mark_c, tol):
            venue["butterfly"].append({"T": float(t), **v})
        for v in vertical_violations(s["strike"].values, mark_c, 1.0, tol):
            venue["vertical"].append({"T": float(t), **v})
        cal_slices.append((float(t), s["k"].values, (s["mark_iv"].values ** 2) * float(t)))
        slices_out.append({"T": float(t), "forward": F, "k": s["k"].round(4).tolist(),
                           "strike": s["strike"].tolist(), "bid_iv": s["bid_iv"].round(4).tolist(),
                           "ask_iv": s["ask_iv"].round(4).tolist(), "mark_iv": s["mark_iv"].round(4).tolist(),
                           "fit_iv": s["fit_iv"].round(4).tolist()})
    # calendar tolerance: spread in total-variance units, per slice
    cal_tol = [float(((s["spread"] * s["mid_iv"] * 2 * s["T"]).median())) for t in expiries for s in [q[q["T"] == t]]]
    venue["calendar"] = calendar_violations(cal_slices, cal_tol)

    # -- our surface: must be clean -----------------------------------------
    grid = torch.linspace(-2.0, 2.0, 400, dtype=torch.float64)
    g_min = min(float(durrleman_g(grid, lambda kk: model.total_variance(kk, torch.full_like(kk, float(t)))).min()) for t in expiries)
    with torch.no_grad():
        ws = torch.stack([model.total_variance(grid, torch.full_like(grid, float(t))) for t in expiries])
        cal_min = float(torch.diff(ws, dim=0).min()) if len(expiries) > 1 else 0.0
    ours = {"durrleman_min_g": g_min, "butterfly_violations": int(g_min < -1e-9),
            "calendar_min_dw": cal_min, "calendar_violations": int(cal_min < -1e-12)}

    # -- greeks: autograd vs closed form -------------------------------------
    F = torch.tensor(q["forward"].values, requires_grad=True); K = torch.tensor(q["strike"].values)
    T = torch.tensor(q["T"].values, requires_grad=True); sig = torch.tensor(q["mid_iv"].values, requires_grad=True)
    ic = torch.tensor(q["is_call"].values)
    price = bs(F, K, T, torch.zeros_like(K), sig, is_call=ic)
    delta = torch.autograd.grad(price.sum(), F, create_graph=True)[0]
    gamma = torch.autograd.grad(delta.sum(), F, retain_graph=True)[0]
    vega = torch.autograd.grad(price.sum(), sig, retain_graph=True)[0]
    theta = -torch.autograd.grad(price.sum(), T)[0]
    d, g, v, th = _black76_greeks_closed_form(F.detach(), K, T.detach(), sig.detach(), ic)
    greeks = {"delta": float((delta - d).abs().max()), "gamma": float((gamma - g).abs().max()),
              "vega": float((vega - v).abs().max()), "theta": float((theta - th).abs().max())}

    return FitReport(
        currency=currency, as_of=as_of, n_quotes=int(len(df)), n_two_sided=int(df["two_sided"].sum()),
        n_fit=int(len(q)), expiries=int(len(expiries)), fit=fit, inside_bid_ask_share=inside,
        venue_violations={k: v for k, v in venue.items()},
        our_violations=ours, greeks_max_abs_err=greeks,
        timings_s={"fit": round(t_fit, 3), "total": round(time.time() - t0, 3)}, device=device,
        slices=slices_out,
    )
