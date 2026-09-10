# voltorch

**Differentiable option pricing and volatility surfaces in PyTorch.**

Every model is an `nn.Module`. Every price is differentiable with respect to the
spot, the strike, the maturity **and** the model parameters. So:

- **Greeks come from autograd**, not from bumping inputs and subtracting.
- **Calibration is gradient descent**, not a derivative-free optimiser crawling a
  five-dimensional surface.
- **A pricer can sit inside a network** and receive gradients through it.

```bash
pip install voltorch
```


## 0.2: arbitrage-free surfaces from a live chain

```python
from voltorch import fit_chain
from voltorch.deribit import fetch_chain      # no key; Deribit public API

report = fit_chain(fetch_chain("BTC"), currency="BTC")
report.refined["rmse_vol_pts"], report.refined["inside_bid_ask_share"]
report.venue_violations["executable"]          # butterflies/verticals/calendars you could trade
report.our_violations                           # must be zero; Durrleman g and calendar Δw
report.greeks_max_abs_err                       # autograd vs closed-form Black-76
```

Live, refit every 30 minutes: **https://savabs.github.io/voltorch/**

* `ESSVI` — extended SSVI (Hendriks–Martini 2019): per-expiry (θ, ρ, ψ) with the
  butterfly and calendar conditions enforced by the parameterisation, so the
  surface is arbitrage-free *by construction* at every optimiser step.
* `SVISlice` — per-expiry raw SVI refinement, arbitrage-*checked* on a dense grid
  (Durrleman g ≥ 0, calendar against neighbours) with fallback to the backbone.
  Calendar is checked, and therefore claimed, only where **both** expiries have
  quotes; beyond the quoted range the surface is extrapolation and is labelled
  as such. `report.refined` carries `quoted_k_range` and
  `calendar_checked_k_range` so the reader can see exactly where the claim
  applies.
* `arbitrage` — `durrleman_g`, `butterfly_violations`, `vertical_violations`,
  `calendar_violations`, and `executable_violations` (against bids and asks).
* `implied_volatility_bisect` — bracketed bisection for market quotes; it cannot
  stall where Newton does (a documented Deribit case is in the tests).
* `deribit.fetch_chain` — coin prices convert to USD on the *forward*
  (`price × F = Black-76`), verified against the venue's marks to 1e-4.

Install: `pip install "voltorch[page]"` for the loader and the page renderer.

## Greeks, without finite differences

```python
import torch
from voltorch import BlackScholes

S = torch.tensor(100.0, requires_grad=True)
K, T, r, sigma = (torch.tensor(x) for x in (100.0, 1.0, 0.05, 0.20))

price = BlackScholes()(S, K, T, r, sigma)
delta, = torch.autograd.grad(price, S, create_graph=True)
gamma, = torch.autograd.grad(delta, S)

print(f"price {price.item():.4f}  delta {delta.item():.4f}  gamma {gamma.item():.4f}")
```

Second-order greeks are a second `grad` call. No bump size to choose, and no
cancellation error from choosing it badly.

## Calibrating Heston by gradient descent

`learnable=True` registers the model parameters as `nn.Parameter`, so the whole
of `torch.optim` applies:

```python
import torch
from voltorch import HestonCOS

model = HestonCOS(kappa=1.0, theta=0.10, xi=0.20, rho=-0.20, v0=0.10,
                  learnable=True)
opt = torch.optim.Adam(model.parameters(), lr=2e-2)

S = torch.full((16,), 100.0)
K = torch.linspace(75.0, 125.0, 16)
T = torch.full((16,), 1.0)
r = torch.full((16,), 0.02)

for _ in range(600):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(S, K, T, r), market_prices)
    loss.backward()
    opt.step()
```

The gradient flows from the loss, back through the Fourier-COS expansion,
through a complex-valued characteristic function with a branch cut in it, and
into `kappa`, `theta`, `xi`, `rho` and `v0`.

`examples/calibrate_heston.py` runs this against prices from a known Heston and
shows something a fit-quality number hides. On one maturity the recovered prices
are near-exact (rmse 0.0036) and the parameters are wrong — `kappa` comes back
1.04 against a true 2.5, because on a single smile `kappa` and `xi` trade off
against each other. Across five maturities the rmse is *worse* (0.013) and the
parameters are right (`kappa` 2.07, `rho` −0.649, `v0` 0.0493), because the
speed of mean reversion is a statement about the term structure. Fit quality and
identification are different things.

## What is in it

| | |
|---|---|
| **European** | `BlackScholes` — closed form, vectorised, with greeks by autograd |
| **American** | `BaroneAdesiWhaley` — quadratic approximation for early exercise |
| **Stochastic vol** | `HestonCOS` |
| **Jumps** | `BatesCOS` (Heston + jumps), `MertonCOS` (lognormal jumps) |
| **Pure jump** | `VarianceGammaCOS` |
| **Surfaces** | `SVIParameterization`, `SABRModel`, `ImpliedVolatilitySurface` |
| **Rough vol** | `RoughBergomiModel`, `estimate_hurst_exponent` |
| **Inversion** | `implied_volatility` |
| **Paths** (extra) | `voltorch.sde` — `GBM`, `HestonSDE` via `torchsde` |

All Fourier models share the `FourierCOS` base (Fang & Oosterlee), so a new
model is a characteristic function and its cumulants — the expansion, the
truncation interval and the payoff coefficients are inherited.

## The part that is actually hard

The formulae are in the papers. What is not in the papers is the short list of
numerical failures that silently poison a direct implementation. Each is guarded
here, at its site, with the symptom it prevents:

- **The complex logarithm's branch cut** in Heston and Bates. The principal
  branch flips sign partway along the integration, and the price is wrong in a
  way that looks like a modest calibration error. Uses Albrecher's *Little Trap*
  formulation, which keeps the integrand continuous — without it, gradients are
  NaN rather than merely wrong.
- **The `z / χ(z)` singularity in Hagan's SABR** as `K → F`. It is `0/0`,
  finite in the limit and NaN in floating point, and it lands exactly
  at-the-money — the strike you care about most. Guarded by a Taylor expansion
  below `|z| < 1e-4`.
- **Martingale drift correction.** The `-½σ²T` diffusion term and the `-λκT`
  jump compensator must both appear in the risk-neutral characteristic function.
  Omit either and prices drift away from the forward as maturity grows, which
  reads as a term-structure effect and is a bug.
- **Interpolating total variance, not volatility.** `w(k,T) = σ²T` is what is
  linear in time; interpolating raw implied vol admits calendar arbitrage
  between the pillars you interpolated from.
- **Durrleman's condition** on the SVI slice, so a fitted smile is butterfly
  arbitrage-free rather than merely close to the quotes.

Each of these was found by having the gradients go NaN and working backwards.

## Tested against limits, not against itself

39 tests. Heston, Bates, Merton and Variance Gamma each converge to the
Black-Scholes price as their extra parameters go to zero — a check that fails
loudly if a characteristic function or a cumulant is wrong, unlike a regression
test against a number the same code produced yesterday.

The rough Bergomi model reproduces the empirical power-law explosion of the
at-the-money skew, `ψ(T) ∝ T^(H-1/2)`: at `H = 0.07` the skew at `T = 0.1` is
~11× the skew at `T = 0.5`. That is the signature rough volatility exists to
explain, and it is a real check rather than a smoke test.

```bash
pip install voltorch[dev] && pytest
```

## Honest limitations

- Single-asset, European exercise for the Fourier models. No basket, no
  American under stochastic volatility (`BaroneAdesiWhaley` is Black-Scholes
  dynamics).
- `float32` throughout, matching PyTorch's default. Deep out-of-the-money prices
  over long maturities will show it; cast to `float64` if that matters to you.
- The COS method assumes the characteristic function is known in closed form.
  It is not a general PDE or Monte Carlo engine.
- `RoughBergomiModel` simulates; it does not admit a closed-form
  characteristic function, so it does not price through `FourierCOS`.
- Not a risk system. There is no calendar, no day count convention, no
  settlement, no market data. It prices and it differentiates.
- The no-arbitrage claim on a refined slice covers the quoted range, not the
  wings. Butterfly is checked out to three times the quoted span; calendar only
  where both neighbouring expiries are quoted. Outside that, the SVI wing is an
  extrapolation and nothing is asserted about it. Until 0.2.1 the calendar check
  ran on the wings too, which rejected good slices for crossings at strikes
  nobody quotes — see the changelog.

## Changelog

### 0.2.1

The calendar check now runs on the range where both expiries are quoted, instead
of on three times the quoted span capped at |k| ≤ 2 — which reached a strike at
13% of the forward, where nothing trades.

Measured over eight BTC chains taken ninety seconds apart, the old range rejected
23 refined slices and **not one of them failed the same check where quotes
exist**: the worst dips sat at k = 0.55, −1.26 and −2.00, while inside the quoted
region the slices were ordered correctly by ten to a hundred times the bid-ask
spread expressed in total variance. Two extrapolations crossing in an unquoted
wing is not calendar arbitrage.

On one archived chain, refit from the identical bytes:

| | 0.2.0 | 0.2.1 |
|---|---|---|
| refined RMSE, vol pts | 1.044 | 0.690 |
| inside the bid-ask | 83.6% | 91.8% |
| slices demoted to the backbone | 3 of 10 | 0 of 10 |

Across the eight-chain sample the mean error falls from 0.749 to 0.700 and its
run-to-run spread halves, from 0.260 to 0.132.

The claim narrows with the check: calendar consistency is asserted where both
expiries are quoted and nowhere else. `report.refined` gains `quoted_k_range` and
`calendar_checked_k_range`, and a pair whose quoted ranges do not overlap is
reported as `calendar_unverifiable` and demoted, because an unverified slice
should not be published. Butterfly is unchanged.

The regression is pinned by `tests/test_chain_calendar.py` against a real chain
committed with it, which asserts both halves of the finding: that the wings do
cross on that chain, and that the quoted region does not.

## Licence

Apache-2.0.
