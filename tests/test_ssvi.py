import torch

from voltorch.arbitrage import durrleman_g
from voltorch.ssvi import SSVI


def _synthetic(device="cpu"):
    T = torch.tensor([0.05, 0.1, 0.25, 0.5, 1.0], dtype=torch.float64)
    truth = SSVI(T, theta_init=0.16 * T, rho=-0.35, eta=0.8, gamma=0.45).to(device)
    ks = torch.linspace(-0.6, 0.6, 41, dtype=torch.float64)
    k = ks.repeat(len(T)); Tt = T.repeat_interleave(len(ks))
    with torch.no_grad():
        iv = truth.implied_vol(k.to(device), Tt.to(device)).cpu()
    return T, k, Tt, iv, truth


def test_recovers_synthetic_surface():
    T, k, Tt, iv, truth = _synthetic()
    m = SSVI(T, rho=0.0, eta=0.5, gamma=0.5)
    rep = m.fit(k, Tt, iv, adam_steps=300, lbfgs_steps=80)
    assert rep["rmse_vol_pts"] < 0.05, rep
    assert abs(rep["rho"] - (-0.35)) < 0.02
    assert rep["theorem_4_2"]


def test_theta_monotone_and_theorem_after_projection():
    T = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)
    m = SSVI(T, rho=0.9, eta=50.0, gamma=0.3)   # absurd eta; projection must shrink it
    assert m.satisfies_theorem()
    th = m.theta
    assert (torch.diff(th) > 0).all()


def test_durrleman_nonnegative_on_grid():
    T, k, Tt, iv, truth = _synthetic()
    m = SSVI(T, rho=0.0, eta=0.5, gamma=0.5)
    m.fit(k, Tt, iv, adam_steps=200, lbfgs_steps=40)
    grid = torch.linspace(-1.5, 1.5, 200, dtype=torch.float64)
    for t in T:
        g = durrleman_g(grid, lambda kk: m.total_variance(kk, t.expand_as(kk)))
        assert (g >= -1e-9).all(), (float(t), float(g.min()))


def test_calendar_free_at_fixed_k():
    T, k, Tt, iv, truth = _synthetic()
    m = SSVI(T, rho=0.0, eta=0.5, gamma=0.5)
    m.fit(k, Tt, iv, adam_steps=200, lbfgs_steps=40)
    grid = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64)
    ws = torch.stack([m.total_variance(grid, t.expand_as(grid)) for t in T])
    assert (torch.diff(ws, dim=0) >= -1e-12).all()


def test_runs_on_mps_if_available():
    if not torch.backends.mps.is_available():
        return
    T = torch.tensor([0.1, 0.5], dtype=torch.float64)
    m = SSVI(T)
    # MPS has no float64; the surface must still evaluate in float32 there.
    m32 = m.to(torch.float32).to("mps")
    out = m32.total_variance(torch.linspace(-0.5, 0.5, 5, device="mps"), torch.full((5,), 0.3, device="mps"))
    assert torch.isfinite(out).all()
