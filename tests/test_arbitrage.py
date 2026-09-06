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
