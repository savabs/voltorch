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


def test_essvi_recovers_per_slice_skew_and_stays_arbitrage_free():
    from voltorch.ssvi import ESSVI
    T = torch.tensor([0.02, 0.1, 0.3, 0.8], dtype=torch.float64)
    truth = ESSVI(T, theta_init=torch.tensor([0.004, 0.02, 0.06, 0.15], dtype=torch.float64),
                  psi_init=torch.tensor([0.05, 0.10, 0.16, 0.25], dtype=torch.float64))
    with torch.no_grad():
        truth.raw_rho.copy_(torch.tensor([-0.6, -0.2, 0.3, 0.5]))   # skew that changes sign across expiries
    assert truth.satisfies_conditions()
    ks = torch.linspace(-0.5, 0.5, 31, dtype=torch.float64)
    k = ks.repeat(len(T)); Tt = T.repeat_interleave(len(ks))
    with torch.no_grad():
        iv = truth.implied_vol(k, Tt)
    m = ESSVI(T)
    rep = m.fit(k, Tt, iv, adam_steps=400, lbfgs_steps=100)
    assert rep["rmse_vol_pts"] < 0.05, rep
    assert rep["conditions"]
    grid = torch.linspace(-1.5, 1.5, 200, dtype=torch.float64)
    for t in T:
        assert (durrleman_g(grid, lambda kk: m.total_variance(kk, t.expand_as(kk))) >= -1e-9).all()
    ws = torch.stack([m.total_variance(grid, t.expand_as(grid)) for t in T])
    assert (torch.diff(ws, dim=0) >= -1e-12).all()


def test_svi_slice_scale_invariant_and_checked():
    from voltorch.ssvi import ESSVI, SVISlice
    # a 2-day and a 300-day slice with the same annualised smile must fit equally
    # well: the parameterisation is in annualised variance, so T must not matter
    errs = []
    for T in (2 / 365, 300 / 365):
        k = torch.linspace(-0.3, 0.3, 25, dtype=torch.float64)
        iv = 0.5 + 0.3 * k ** 2 - 0.1 * k        # not SVI-shaped: a small residual is expected
        bb = ESSVI(torch.tensor([T], dtype=torch.float64), theta_init=torch.tensor([0.25 * T], dtype=torch.float64))
        best = None
        for sl in (SVISlice.from_backbone(bb, T, torch.linspace(-0.45, 0.45, 80, dtype=torch.float64)), SVISlice.from_quotes(T, k, iv)):
            sl.fit(k, iv, torch.ones_like(iv), steps=600, lr=0.02, g_grid=torch.linspace(-0.6, 0.6, 50, dtype=torch.float64))
            e = float(torch.sqrt(((sl.implied_vol(k) - iv) ** 2).mean()) * 100)
            if best is None or e < best[0]:
                best = (e, sl)
        errs.append(best[0]); sl = best[1]
        g = durrleman_g(torch.linspace(-0.9, 0.9, 200, dtype=torch.float64), sl.total_variance)
        assert (g >= -1e-9).all()
    assert all(e < 0.5 for e in errs), errs
    # residual T-dependence comes through the Durrleman penalty (g has 1/w terms);
    # 0.05 vol pts on a 0.5-vol smile is far inside any spread
    assert abs(errs[0] - errs[1]) < 0.1, errs
