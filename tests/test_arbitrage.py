import numpy as np
import torch

from voltorch.arbitrage import butterfly_violations, calendar_violations, vertical_violations
from voltorch.options import BlackScholes


def _calls(F, K, T, sigma):
    bs = BlackScholes()
    return bs(torch.tensor(F).double().expand(len(K)), torch.tensor(K).double(), torch.tensor(T).double().expand(len(K)),
              torch.zeros(len(K)).double(), torch.tensor(sigma).double().expand(len(K)), is_call=True).numpy()


def test_flat_black76_chain_is_clean():
    K = np.linspace(60, 140, 41)
    C = _calls(100.0, K, 0.5, 0.3)
    assert butterfly_violations(K, C) == []
    assert vertical_violations(K, C, discount=1.0) == []


def test_bump_is_flagged_and_sub_tolerance_bump_is_not():
    K = np.linspace(60, 140, 41)
    C = _calls(100.0, K, 0.5, 0.3)
    C[20] += 0.5                       # a mid-chain call priced too high
    v = butterfly_violations(K, C)
    assert len(v) == 1 and v[0]["strike"] == K[20] and v[0]["magnitude"] > 0.4
    assert butterfly_violations(K, C, tol=1.0) == []
    C2 = _calls(100.0, K, 0.5, 0.3); C2[20] -= 3.0   # too low: neighbours' slopes break vertical bounds
    assert any(x["kind"] == "call_increasing" for x in vertical_violations(K, C2))


def test_calendar_violation_detected():
    k = np.linspace(-0.5, 0.5, 21)
    w1 = 0.04 * np.ones_like(k) + 0.02 * k ** 2
    w2 = w1 * 1.5
    assert calendar_violations([(0.25, k, w1), (0.5, k, w2)]) == []
    bad = calendar_violations([(0.25, k, w2), (0.5, k, w1)])
    assert len(bad) == 21 and all(b["magnitude"] > 0 for b in bad)
    assert calendar_violations([(0.25, k, w2), (0.5, k, w1)], tol=1.0) == []


def test_executable_violations_on_a_book():
    import pandas as pd
    from voltorch.arbitrage import executable_violations
    F = 100.0
    rows = []
    for K, mid in ((90, 12.0), (100, 5.0), (110, 1.5)):
        rows.append(dict(expiry="a", T=0.5, strike=K, is_call=True, forward=F, bid_usd=mid - 0.2, ask_usd=mid + 0.2, two_sided=True))
    df = pd.DataFrame(rows)
    assert all(len(v) == 0 for v in executable_violations(df).values())
    df.loc[df.strike == 100, "bid_usd"] = 8.0          # body bid absurdly high: sell body, buy wings for a credit
    v = executable_violations(df)
    assert len(v["butterfly"]) == 1 and v["butterfly"][0]["edge_usd"] > 0
    # calendar: later expiry offered below the earlier bid at the same strike
    later = df.copy(); later["expiry"] = "b"; later["T"] = 1.0; later[["bid_usd", "ask_usd"]] = [[3.0, 3.4]] * 3
    later.loc[later.strike == 100, ["bid_usd", "ask_usd"]] = [4.0, 4.5]
    both = pd.concat([df, later])
    both.loc[both.strike == 100, "bid_usd"] = [5.0 - 0.2, 4.0]
    v2 = executable_violations(both)
    assert any(c["strike"] == 90 for c in v2["calendar"])   # 90-strike later ask 3.4 < earlier bid 11.8
