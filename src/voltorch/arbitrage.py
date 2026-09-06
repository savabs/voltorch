"""Arbitrage checks for any surface or any chain -- ours or a venue's marks.

Three static conditions on call prices C(K, T) with discount factor D:

  butterfly   C is convex in K            (Durrleman's g(k) >= 0 in IV space)
  vertical    0 <= -dC/dK <= D
  calendar    total variance w(k, T) non-decreasing in T at fixed k

Every check takes a tolerance and reports only violations that exceed it, so
that noise inside the bid-ask spread is never called arbitrage. The reports
are plain rows -- (expiry, strike, magnitude, tolerance) -- so they can be
published next to the quotes they came from.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def durrleman_g(k: torch.Tensor, w_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """g(k) for an arbitrary total-variance slice w_fn(k), derivatives by
    autograd. No butterfly arbitrage iff g >= 0 everywhere."""
    k = k.detach().to(torch.float64).requires_grad_(True)
    w = w_fn(k)
    w1 = torch.autograd.grad(w.sum(), k, create_graph=True)[0]
    w2 = torch.autograd.grad(w1.sum(), k, create_graph=True)[0]
    g = (1.0 - k * w1 / (2.0 * w)) ** 2 - (w1 ** 2 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0
    return g.detach()


def butterfly_violations(strikes: np.ndarray, calls: np.ndarray, tol: np.ndarray | float = 0.0) -> list[dict]:
    """Discrete convexity with uneven spacing. For interior K with neighbours
    K-, K+: the linear interpolation of C(K-), C(K+) at K must be >= C(K).
    Magnitude is how far below it C(K) sits; reported when > tol[K]."""
    order = np.argsort(strikes)
    K, C = np.asarray(strikes, float)[order], np.asarray(calls, float)[order]
    tol = np.broadcast_to(np.asarray(tol, float), K.shape)[order]
    out = []
    for i in range(1, len(K) - 1):
        lam = (K[i] - K[i - 1]) / (K[i + 1] - K[i - 1])
        interp = (1 - lam) * C[i - 1] + lam * C[i + 1]
        mag = C[i] - interp
        if mag > tol[i]:
            out.append({"strike": float(K[i]), "magnitude": float(mag), "tolerance": float(tol[i]),
                        "neighbours": (float(K[i - 1]), float(K[i + 1]))})
    return out


def vertical_violations(strikes: np.ndarray, calls: np.ndarray, discount: float = 1.0,
                        tol: np.ndarray | float = 0.0) -> list[dict]:
    """-dC/dK must lie in [0, D]. Slope is taken between adjacent strikes."""
    order = np.argsort(strikes)
    K, C = np.asarray(strikes, float)[order], np.asarray(calls, float)[order]
    tol = np.broadcast_to(np.asarray(tol, float), K.shape)[order]
    out = []
    for i in range(len(K) - 1):
        slope = -(C[i + 1] - C[i]) / (K[i + 1] - K[i])
        t = max(tol[i], tol[i + 1]) / (K[i + 1] - K[i])
        if slope < -t:
            out.append({"strike": float(K[i]), "next_strike": float(K[i + 1]), "kind": "call_increasing",
                        "magnitude": float(-slope), "tolerance": float(t)})
        elif slope > discount + t:
            out.append({"strike": float(K[i]), "next_strike": float(K[i + 1]), "kind": "slope_exceeds_discount",
                        "magnitude": float(slope - discount), "tolerance": float(t)})
    return out


def calendar_violations(slices: list[tuple[float, np.ndarray, np.ndarray]],
                        tol: float | list[float] = 0.0) -> list[dict]:
    """slices: [(T, k, w)] sorted by T, each with log-moneyness k and total
    variance w (in the same forward-moneyness convention). For consecutive
    expiries, w at the later T interpolated onto the earlier slice's k must
    be >= the earlier w. Only the overlapping k-range is compared."""
    slices = sorted(slices, key=lambda s: s[0])
    tols = np.broadcast_to(np.asarray(tol, float), (len(slices),))
    out = []
    for i in range(len(slices) - 1):
        T0, k0, w0 = slices[i]
        T1, k1, w1 = slices[i + 1]
        o0, o1 = np.argsort(k0), np.argsort(k1)
        k0, w0, k1, w1 = k0[o0], w0[o0], k1[o1], w1[o1]
        lo, hi = max(k0.min(), k1.min()), min(k0.max(), k1.max())
        m = (k0 >= lo) & (k0 <= hi)
        if not m.any():
            continue
        w1_on_k0 = np.interp(k0[m], k1, w1)
        mag = w0[m] - w1_on_k0
        bad = mag > tols[i]
        for kk, mm in zip(k0[m][bad], mag[bad]):
            out.append({"T_earlier": float(T0), "T_later": float(T1), "k": float(kk),
                        "magnitude": float(mm), "tolerance": float(tols[i])})
    return out
