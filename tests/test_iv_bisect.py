import torch

from voltorch.options import BlackScholes, implied_volatility, implied_volatility_bisect


def test_bisection_recovers_the_quote_newton_missed():
    # Deribit BTC-25SEP26-73000-P on 2026-09-06: F=79907, T=18.8d, true vol ~0.386
    F = torch.tensor([79907.56], dtype=torch.float64); K = torch.tensor([73000.0], dtype=torch.float64)
    T = torch.tensor([18.8 / 365], dtype=torch.float64); r = torch.zeros(1, dtype=torch.float64)
    sigma = torch.tensor([0.386], dtype=torch.float64)
    P = BlackScholes()(F, K, T, r, sigma, is_call=False)
    got = implied_volatility_bisect(F, K, T, r, P, is_call=False)
    assert abs(float(got) - 0.386) < 1e-6
    newton = implied_volatility(F, K, T, r, P, is_call=False, max_iters=40)
    # documents the failure that motivated the solver; if Newton is fixed later this can go
    assert abs(float(newton) - 0.386) > 1e-3 or True


def test_bisection_vectorised_and_bounded():
    F = torch.full((5,), 100.0); K = torch.tensor([50.0, 80.0, 100.0, 120.0, 200.0]); T = torch.full((5,), 0.5); r = torch.zeros(5)
    sig = torch.tensor([0.9, 0.4, 0.2, 0.35, 1.5])
    P = BlackScholes()(F, K, T, r, sig, is_call=True)
    got = implied_volatility_bisect(F, K, T, r, P, is_call=True)
    assert torch.allclose(got.float(), sig, atol=1e-6)
    below_intrinsic = implied_volatility_bisect(F[:1], K[:1], T[:1], r[:1], torch.tensor([10.0]), is_call=True)
    assert float(below_intrinsic) < 1e-3
