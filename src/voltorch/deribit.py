"""Deribit public chain loader. No key. Layer 1: fetch and normalise only.

Deribit options are inverse (coin-settled): prices are quoted in BTC/ETH per
contract, strikes in USD, interest_rate 0. We convert bid/ask to USD with the
index price and back out implied vols with our own Black-76 solver on the
per-expiry forward (``underlying_price``), so the bid/ask IVs on the page are
ours, not the venue's -- and ``mark_iv`` is theirs, kept for comparison.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import torch

from .options import implied_volatility

API = "https://www.deribit.com/api/v2/public"
UA = "voltorch (github.com/savabs/voltorch)"
COLUMNS = ["instrument", "expiry", "T", "strike", "is_call", "forward", "index_price",
           "mark_iv", "bid", "ask", "bid_usd", "ask_usd", "bid_iv", "ask_iv", "two_sided",
           "open_interest", "volume"]


def _get(path: str, session: requests.Session, **params) -> list:
    r = session.get(f"{API}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()["result"]


def fetch_chain(currency: str = "BTC", *, session: requests.Session | None = None,
                now: datetime | None = None, min_T_days: float = 1.0) -> pd.DataFrame:
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", UA)
    now = now or datetime.now(timezone.utc)
    inst = {i["instrument_name"]: i for i in _get("get_instruments", s, currency=currency, kind="option", expired="false")}
    time.sleep(0.2)
    book = _get("get_book_summary_by_currency", s, currency=currency, kind="option")
    rows = []
    for b in book:
        i = inst.get(b["instrument_name"])
        if not i:
            continue
        exp = datetime.fromtimestamp(i["expiration_timestamp"] / 1000, tz=timezone.utc)
        T = (exp - now).total_seconds() / (365.0 * 86400)
        if T * 365 < min_T_days:
            continue
        idx = b.get("index_price") or b.get("estimated_delivery_price") or np.nan
        bid, ask = b.get("bid_price"), b.get("ask_price")
        rows.append({
            "instrument": b["instrument_name"], "expiry": exp, "T": T, "strike": float(i["strike"]),
            "is_call": i["option_type"] == "call", "forward": float(b["underlying_price"]),
            "index_price": float(idx) if idx else np.nan,
            "mark_iv": (b.get("mark_iv") or np.nan) / 100.0,
            "bid": bid if bid is not None else np.nan, "ask": ask if ask is not None else np.nan,
            "open_interest": b.get("open_interest") or 0.0, "volume": b.get("volume") or 0.0,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    # inverse contract: price in coin * index = USD price of the option
    df["bid_usd"] = df["bid"] * df["index_price"]
    df["ask_usd"] = df["ask"] * df["index_price"]
    df["two_sided"] = df["bid"].notna() & df["ask"].notna() & (df["bid"] > 0) & (df["ask"] > 0)
    df["bid_iv"], df["ask_iv"] = np.nan, np.nan
    m = df["two_sided"].values
    if m.any():
        F = torch.tensor(df.loc[m, "forward"].values, dtype=torch.float64)
        K = torch.tensor(df.loc[m, "strike"].values, dtype=torch.float64)
        T = torch.tensor(df.loc[m, "T"].values, dtype=torch.float64)
        r = torch.zeros_like(F)
        for col, src in (("bid_iv", "bid_usd"), ("ask_iv", "ask_usd")):
            P = torch.tensor(df.loc[m, src].values, dtype=torch.float64)
            ivs = np.full(m.sum(), np.nan)
            for is_call in (True, False):
                cm = torch.tensor(df.loc[m, "is_call"].values == is_call)
                if cm.any():
                    iv = implied_volatility(F[cm], K[cm], T[cm], r[cm], P[cm], is_call=is_call, max_iters=40)
                    ivs[cm.numpy()] = iv.detach().numpy()
            df.loc[m, col] = ivs
    return df[COLUMNS].sort_values(["expiry", "strike", "is_call"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)), "two_sided": int(df["two_sided"].sum()),
        "expiries": int(df["expiry"].nunique()),
        "forward_range": [float(df["forward"].min()), float(df["forward"].max())] if len(df) else None,
        "T_range_days": [float(df["T"].min() * 365), float(df["T"].max() * 365)] if len(df) else None,
    }
