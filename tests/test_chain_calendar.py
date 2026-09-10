"""The calendar check is claimed where there are quotes, and only there.

Fixture is one real Deribit BTC chain, archived 2026-09-10. It is in the
repository so the claim below is reproducible rather than asserted: on this
chain the refined slices cross in their extrapolated wings and do not cross
anywhere both expiries are quoted. Before 0.2.1 that crossing demoted three
good slices to the backbone and cost about 0.05 vol points of accuracy.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch

from voltorch import fit_chain, otm
from voltorch.ssvi import ESSVI, SVISlice

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "deribit_btc_2026-09-10.csv.gz")


@pytest.fixture(scope="module")
def chain() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


@pytest.fixture(scope="module")
def report(chain):
    return fit_chain(chain, currency="BTC")


def test_the_fixture_is_a_real_chain(chain):
    assert len(chain) > 500
    assert chain["two_sided"].sum() > 400
    assert chain["expiry"].nunique() >= 8


def test_calendar_is_checked_inside_the_quoted_range_of_both_expiries(report):
    quoted = report.refined["quoted_k_range"]
    checked = report.refined["calendar_checked_k_range"]
    keys = list(quoted)
    for prev, cur in zip(keys, keys[1:]):
        rng = checked[cur]
        if rng is None:
            continue
        lo, hi = rng
        assert lo >= max(quoted[prev][0], quoted[cur][0]) - 1e-9
        assert hi <= min(quoted[prev][1], quoted[cur][1]) + 1e-9


def test_the_first_expiry_has_nothing_to_be_checked_against(report):
    """One range per adjacent pair, so one fewer than there are expiries."""
    quoted = list(report.refined["quoted_k_range"])
    checked = list(report.refined["calendar_checked_k_range"])
    assert len(checked) == report.expiries - 1
    assert checked == quoted[1:]


def test_the_guarantee_says_where_calendar_is_claimed(report):
    g = report.refined["guarantee"]
    assert "where BOTH are quoted" in g
    assert "extrapolation" in g


def test_no_slice_is_demoted_on_a_wing_it_does_not_quote(report):
    """The regression. On this chain every refined slice survives."""
    assert report.refined["slices_fallback"] == {}, report.refined["slices_fallback"]
    assert report.refined["slices_refined"] == report.expiries


def test_refinement_actually_beats_the_backbone_here(report):
    assert report.refined["rmse_vol_pts"] < report.fit["rmse_vol_pts"]
    assert report.refined["inside_bid_ask_share"] > report.inside_bid_ask_share
    assert report.refined["rmse_vol_pts"] < 1.0


def test_our_surface_is_still_arbitrage_free(report):
    assert report.our_violations["butterfly_violations"] == 0
    assert report.our_violations["calendar_violations"] == 0


def test_the_wings_really_do_cross_on_this_chain(chain):
    """Without this, the regression test above could pass for the wrong reason.

    Refit the slices the way fit_chain does and compare each adjacent pair twice:
    once over three times the quoted span, which is what 0.2.0 checked, and once
    over the range both expiries quote. The first must find a crossing and the
    second must not. That is the whole finding, pinned.
    """
    q = otm(chain).copy()
    q["k"] = np.log(q["strike"] / q["forward"])
    q["mid_iv"] = 0.5 * (q["bid_iv"] + q["ask_iv"])
    q["spread"] = (q["ask_iv"] - q["bid_iv"]).clip(lower=0.002)
    expiries = np.sort(q["T"].unique())
    theta0 = torch.tensor([
        float((q[q["T"] == t].assign(a=lambda d: d.k.abs()).nsmallest(3, "a")["mid_iv"] ** 2).mean() * t)
        for t in expiries], dtype=torch.float64)
    backbone = ESSVI(torch.tensor(expiries, dtype=torch.float64), theta_init=theta0)
    backbone.fit(torch.tensor(q["k"].values), torch.tensor(q["T"].values),
                 torch.tensor(q["mid_iv"].values),
                 torch.tensor(1.0 / q["spread"].values ** 2), adam_steps=300, lbfgs_steps=60)

    slices = {}
    for t in expiries:
        s = q[q["T"] == t]
        kk = torch.tensor(s["k"].values)
        ivt = torch.tensor(s["mid_iv"].values)
        ww = torch.tensor(1.0 / s["spread"].values ** 2)
        span = float(max(abs(s["k"].min()), abs(s["k"].max()), 0.05))
        warm = torch.linspace(-1.5 * span, 1.5 * span, 80, dtype=torch.float64)
        pen = torch.linspace(-2.0 * span, 2.0 * span, 80, dtype=torch.float64)
        cands = []
        for sl in (SVISlice.from_backbone(backbone, float(t), warm),
                   SVISlice.from_quotes(float(t), kk, ivt)):
            sl.fit(kk, ivt, ww, steps=600, lr=0.02, g_grid=pen)
            with torch.no_grad():
                cands.append((float(((sl.implied_vol(kk) - ivt) ** 2 * ww).mean()), sl))
        slices[float(t)] = {
            "slice": min(cands, key=lambda c: c[0])[1],
            "quoted": (float(s["k"].min()), float(s["k"].max())),
            "wide": (max(-2.0, -3.0 * span), min(2.0, 3.0 * span)),
        }

    def worst_dip(prev, cur, lo, hi):
        kk = torch.linspace(lo, hi, 300, dtype=torch.float64)
        with torch.no_grad():
            return float((cur["slice"].total_variance(kk) - prev["slice"].total_variance(kk)).min())

    wide_crossings, quoted_crossings = 0, 0
    for a, b in zip(expiries, expiries[1:]):
        prev, cur = slices[float(a)], slices[float(b)]
        wlo = max(prev["wide"][0], cur["wide"][0])
        whi = min(prev["wide"][1], cur["wide"][1])
        qlo = max(prev["quoted"][0], cur["quoted"][0])
        qhi = min(prev["quoted"][1], cur["quoted"][1])
        if whi > wlo and worst_dip(prev, cur, wlo, whi) < -1e-12:
            wide_crossings += 1
        if qhi > qlo and worst_dip(prev, cur, qlo, qhi) < -1e-12:
            quoted_crossings += 1

    assert wide_crossings >= 1, "fixture no longer exercises the extrapolated-wing case"
    assert quoted_crossings == 0, "a real calendar crossing appeared where both are quoted"
