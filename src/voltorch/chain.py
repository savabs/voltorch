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

from .arbitrage import butterfly_violations, calendar_violations, durrleman_g, executable_violations, vertical_violations
from .options import BlackScholes
from .ssvi import ESSVI, SVISlice

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
    refined: dict
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
    model = ESSVI(torch.tensor(expiries, dtype=torch.float64), theta_init=theta0)
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

    # -- refinement: per-slice SVI, arbitrage-CHECKED, backbone fallback ------
    grid = torch.linspace(-2.0, 2.0, 400, dtype=torch.float64)
    refined_w, refined_status, validated = {}, {}, {}
    t_r = time.time()
    for t in expiries:
        s = q[q["T"] == t]
        kk = torch.tensor(s["k"].values); ivt = torch.tensor(s["mid_iv"].values); ww = torch.tensor(1.0 / s["spread"].values ** 2)
        # every grid is scaled to the slice's quoted range: a 2-day expiry has
        # quotes inside |k| < 0.12 and astronomically large wings at |k| = 2,
        # which would dominate any least squares or penalty evaluated there.
        span = float(max(abs(s["k"].min()), abs(s["k"].max()), 0.05))
        warm_grid = torch.linspace(-1.5 * span, 1.5 * span, 80, dtype=torch.float64)
        pen_grid = torch.linspace(-2.0 * span, 2.0 * span, 80, dtype=torch.float64)
        chk_lo, chk_hi = max(-2.0, -3.0 * span), min(2.0, 3.0 * span)
        chk_grid = torch.linspace(chk_lo, chk_hi, 300, dtype=torch.float64)
        sl = SVISlice.from_backbone(model, float(t), warm_grid)
        sl.fit(kk, ivt, ww, steps=600, lr=0.02, g_grid=pen_grid)
        with torch.no_grad():
            wg = sl.total_variance(grid)
        g = durrleman_g(chk_grid, sl.total_variance)
        refined_w[float(t)] = wg
        refined_status[float(t)] = "ok" if float(g.min()) >= -1e-9 else "butterfly_fail"
        validated[float(t)] = (chk_lo, chk_hi)
    # calendar check between accepted neighbours on the range where both were
    # validated (far-wing extrapolations carry no quotes and are not claimed);
    # on failure demote the later slice to the backbone
    prev_w, prev_rng = None, None
    gn = grid.numpy()
    for t in expiries:
        if refined_status[float(t)] != "ok":
            with torch.no_grad():
                refined_w[float(t)] = model.total_variance(grid, torch.full_like(grid, float(t)))
        rng = validated[float(t)]
        if prev_w is not None:
            lo, hi = max(rng[0], prev_rng[0]), min(rng[1], prev_rng[1])
            m = (gn >= lo) & (gn <= hi)
            if m.any() and float((refined_w[float(t)][m] - prev_w[m]).min()) < -1e-12:
                refined_status[float(t)] = "calendar_fail"
                with torch.no_grad():
                    refined_w[float(t)] = model.total_variance(grid, torch.full_like(grid, float(t)))
        prev_w, prev_rng = refined_w[float(t)], rng
    def _ref_iv(kv, t):
        return np.sqrt(np.interp(kv, grid.numpy(), refined_w[float(t)].numpy()) / float(t))
    q["ref_iv"] = np.concatenate([_ref_iv(q[q["T"] == t]["k"].values, t) for t in expiries]) if all((q["T"] == t).any() for t in expiries) else fitted
    q = q.sort_values(["T", "strike"])  # concat above followed expiry order; realign
    q["ref_iv"] = np.concatenate([_ref_iv(q[q["T"] == t]["k"].values, t) for t in expiries])
    ref_err = (q["ref_iv"] - q["mid_iv"]) * 100
    refined = {"rmse_vol_pts": float(np.sqrt((ref_err ** 2).mean())),
               "inside_bid_ask_share": float(((q["ref_iv"] >= q["bid_iv"]) & (q["ref_iv"] <= q["ask_iv"])).mean()),
               "slices_refined": int(sum(v == "ok" for v in refined_status.values())),
               "slices_fallback": {f"{t*365:.1f}d": v for t, v in refined_status.items() if v != "ok"},
               "time_s": round(time.time() - t_r, 3),
               "guarantee": "butterfly (Durrleman g>=0) and calendar checked on a dense grid within each slice's validated log-moneyness range (3x the quoted span, capped at |k|<=2); outside it, and wherever a check failed, the eSSVI backbone is used",
               "validated_k_range": {f"{t*365:.1f}d": [round(a, 3), round(b, 3)] for t, (a, b) in validated.items()}}

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
        err = (s["fit_iv"] - s["mid_iv"]) * 100; rerr = (s["ref_iv"] - s["mid_iv"]) * 100
        slices_out.append({"T": float(t), "forward": F, "n": int(len(s)),
                           "rmse_vol_pts": float(np.sqrt((err ** 2).mean())),
                           "inside_bid_ask": float(((s["fit_iv"] >= s["bid_iv"]) & (s["fit_iv"] <= s["ask_iv"])).mean()),
                           "refined_rmse_vol_pts": float(np.sqrt((rerr ** 2).mean())),
                           "refined_inside_bid_ask": float(((s["ref_iv"] >= s["bid_iv"]) & (s["ref_iv"] <= s["ask_iv"])).mean()),
                           "refined_status": refined_status[float(t)], "ref_iv": s["ref_iv"].round(4).tolist(),
                           "k": s["k"].round(4).tolist(),
                           "strike": s["strike"].tolist(), "bid_iv": s["bid_iv"].round(4).tolist(),
                           "ask_iv": s["ask_iv"].round(4).tolist(), "mark_iv": s["mark_iv"].round(4).tolist(),
                           "fit_iv": s["fit_iv"].round(4).tolist()})
    # calendar tolerance: spread in total-variance units, per slice
    cal_tol = [float(((s["spread"] * s["mid_iv"] * 2 * s["T"]).median())) for t in expiries for s in [q[q["T"] == t]]]
    venue["calendar"] = calendar_violations(cal_slices, cal_tol)
    # the one that matters: arbitrage you could trade against the bids and asks
    venue["executable"] = executable_violations(df[df["two_sided"]])

    # -- our surface: must be clean -----------------------------------------
    grid = torch.linspace(-2.0, 2.0, 400, dtype=torch.float64)
    g_min = min(float(durrleman_g(grid, lambda kk: model.total_variance(kk, torch.full_like(kk, float(t)))).min()) for t in expiries)
    with torch.no_grad():
        ws = torch.stack([model.total_variance(grid, torch.full_like(grid, float(t))) for t in expiries])
        cal_min = float(torch.diff(ws, dim=0).min()) if len(expiries) > 1 else 0.0
    ours = {"durrleman_min_g": g_min, "butterfly_violations": int(g_min < -1e-9),
            "calendar_min_dw": cal_min, "calendar_violations": int(cal_min < -1e-12),
            "conditions_hold": bool(fit["conditions"])}

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
    greeks = {"delta": float((delta - d).abs().max().detach()), "gamma": float((gamma - g).abs().max().detach()),
              "vega": float((vega - v).abs().max().detach()), "theta": float((theta - th).abs().max().detach())}

    return FitReport(
        currency=currency, as_of=as_of, n_quotes=int(len(df)), n_two_sided=int(df["two_sided"].sum()),
        n_fit=int(len(q)), expiries=int(len(expiries)), fit=fit, inside_bid_ask_share=inside, refined=refined,
        venue_violations={k: v for k, v in venue.items()},
        our_violations=ours, greeks_max_abs_err=greeks,
        timings_s={"fit": round(t_fit, 3), "total": round(time.time() - t0, 3)}, device=device,
        slices=slices_out,
    )
