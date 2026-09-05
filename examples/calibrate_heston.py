"""Recover Heston parameters from prices by gradient descent.

The point is not that the fit succeeds. It is that the gradient exists at all:
it flows from the mean-squared error, back through the Fourier-COS expansion,
through a complex-valued characteristic function with a branch cut in it, and
into kappa/theta/xi/rho/v0. No finite differences, no derivative-free optimiser.

It also shows something true about Heston that a fit-quality number hides. On a
single maturity, kappa and xi are only weakly identified -- they trade off
against each other, so the prices fit almost perfectly while the parameters are
wrong. Adding maturities identifies them, because kappa controls how fast
variance mean-reverts and that is a statement about the term structure.
"""
import torch
from voltorch import HestonCOS

torch.manual_seed(0)

TRUE = dict(kappa=2.5, theta=0.05, xi=0.40, rho=-0.65, v0=0.05)


def fit(strikes, maturities, steps, lr=2e-2):
    """Generate prices from a known Heston, then try to recover it."""
    truth = HestonCOS(**TRUE, learnable=False)
    K = strikes.repeat(len(maturities))
    T = maturities.repeat_interleave(len(strikes))
    S = torch.full_like(K, 100.0)
    r = torch.full_like(K, 0.02)
    with torch.no_grad():
        market = truth(S, K, T, r)

    # Start somewhere else entirely and let autograd find its way back.
    model = HestonCOS(kappa=1.0, theta=0.10, xi=0.20, rho=-0.20, v0=0.10,
                      learnable=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(S, K, T, r), market)
        loss.backward()
        opt.step()
    return model, loss.sqrt().item(), len(K)


def report(title, model, rmse, n):
    print(f"\n{title}\n  {n} quotes, rmse {rmse:.5f}")
    print(f"  {'param':8}{'true':>9}{'fitted':>10}")
    for name, true_val in TRUE.items():
        print(f"  {name:8}{true_val:9.4f}{getattr(model, name).item():10.4f}")


strikes = torch.linspace(75.0, 125.0, 16)

model, rmse, n = fit(strikes, torch.tensor([1.0]), steps=600)
report("ONE MATURITY -- prices fit, parameters do not", model, rmse, n)
print("  kappa and xi are weakly identified here: the smile pins down the")
print("  variance level and the skew, but not the speed of mean reversion.")

model, rmse, n = fit(strikes, torch.tensor([0.1, 0.25, 0.5, 1.0, 2.0]), steps=1500)
report("FIVE MATURITIES -- the term structure identifies them", model, rmse, n)
print("  Worse rmse, better parameters. Fit quality and identification are")
print("  different things, and only one of them is what you wanted.")
