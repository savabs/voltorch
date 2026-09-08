# voltorch — working notes for agents

Differentiable option pricing and arbitrage-free volatility surfaces in PyTorch.
Every model is an `nn.Module`; every price is differentiable with respect to
spot, strike, maturity **and** model parameters. Greeks come from autograd,
calibration is gradient descent, and a pricer can sit inside a network.

Read this before changing anything. Most of what follows is not derivable from
the code, and the parts that are subtle are subtle in ways that fail silently.

---

## What this repo is for

As of September 2026 this is **the second product** behind `../tirramind-universe`:
a crypto options volatility-surface and risk API. The live page is the shop
window; the archive is the asset.

The commercial position, decided after market research:

- **Price band $500–999/month, and do not exceed it while solo.** Every
  self-serve ceiling in this market sits there — Laevitas Enterprise $500,
  Coinglass Professional $699, Glassnode Vector $749, Block Scholes Prime £999,
  Fireblocks Essentials $999. Above roughly $1,000/mo every vendor switches to
  contact-sales, and that is the same line as the procurement line: DDQs,
  SOC 2, and DORA registers that ask for audited financials and "depth of
  resources (including staffing)". A one-person company fails those on the form.
  Below the line, nobody asks who you are.
- **The direct comparable is Block Scholes** — SVI-calibrated surfaces, Greeks,
  funding, sold self-serve at £499 (1-hour granularity, 1,000 req/day) and £999.
  That is a research product, not a feed. They are FCA-authorised and distribute
  on the Bloomberg Terminal, so do not compete on distribution or on
  institutional credibility.
- **The serious competitor is SignalPlus** — ~$500M valuation, Cumberland,
  FalconX and Galaxy as clients, free distribution through Deribit trading
  competitions. Do not try to be them.
- **What is actually unoccupied:** continuously-refit, *arbitrage-free by
  construction*, cross-venue surfaces with the venue's own executable violations
  published beside them. Nobody under $1,000/mo sells that.

Realistic ceiling while solo: **$1–3M ARR**. The buyer set is 300–800 firms
globally and it is cycle-exposed — crypto vol demand contracts in a bear market.
That is why this is second, not first.

### Read Deribit's ToS before shipping

Price **derived analytics**, never raw ticks. Redistribution of venue market
data is a licensing question that has not been resolved in this repo.

---

## The clock that cannot be restarted

`scripts/live_page.py` writes, every 30 minutes, one immutable gzipped file per
currency per run:

```
docs/archive/YYYY/MM/DD/HHMMZ-{BTC,ETH}.json.gz     ~8 KB each, ~280 MB/year
```

It holds the **whole fitted surface and the book it was fitted to** — per-expiry
forwards, strikes, log-moneyness, two-sided bid/ask IV, the venue's mark, and
both fitted curves on the same grid — plus every violation check.

`docs/history.jsonl` holds only fit *diagnostics* (rmse, counts, violation
totals). A diagnostic cannot be refitted, resampled or backtested. The book can.
**The book is gone the moment the market moves**, so the archive is written
first, separately, one file per run so nothing already recorded is ever
rewritten. Started 2026-09-07.

If the workflow breaks, fixing it outranks every feature in the repo. A
competitor starting later can never have these hours.

---

## Architecture

**The product is four modules.** The rest is inherited surface area from
TirraMind's math stack (their docstrings still say "Math Stack M1/M2/M5/M6") and
is not on the critical path.

```
core     ssvi.py       SSVI and ESSVI surfaces, arbitrage-free by construction
         chain.py      fit_chain: raw chain -> publishable FitReport. THE product
         arbitrage.py  butterfly / vertical / calendar / executable checks
         deribit.py    public chain loader, no key. Fetch and normalise only

inherited options.py   Black-Scholes, Barone-Adesi-Whaley, Fourier COS
                       (Heston/Bates/Merton/VG), implied_volatility_bisect
         volatility.py SVI parameterisation, SABR, surface features
         rough.py      rough Bergomi, BLP hybrid scheme, Hurst estimation
         sde.py        GBM / Heston as torchsde nn.Modules

         scripts/live_page.py   30-min refit, archive, render docs/index.html
```

`FitReport` (in `chain.py`) is the unit of value. For a chain at one instant it
carries: fit quality against two-sided OTM quotes (RMSE in vol points, share
inside bid-ask), the venue's own arbitrage violations, **our** violations, and
autograd greeks checked against closed-form Black-76.

---

## Invariants — do not break these

1. **Arbitrage-freeness is structural, never a penalty term.** The Gatheral–
   Jacquier butterfly conditions and the ESSVI calendar conditions are enforced
   *by the parameterisation* — theta by cumulative softplus over sorted
   expiries, eta projected after every optimiser step. The surface is therefore
   arbitrage-free at **every** step, not merely at convergence. Never replace
   this with a soft penalty; a penalty makes it arbitrage-*checked*, which is a
   different and much weaker product.
2. **Durrleman's g(k) ≥ 0 is a test of the implementation, never the
   constraint.** It lives in the test suite. If you find yourself enforcing g
   during fitting, the parameterisation has been broken.
3. **`our_violations` must be zero. Always.** A run where the fitted surface
   contains arbitrage is a bug, not a finding.
4. **`venue_violations["executable"]` is the product.** Those are butterflies,
   verticals and calendars measured against live *bids and asks*, not mids — so
   a non-zero row is something a reader could in principle have traded. A day
   with zero executable arbitrage is the expected state of a market-made venue;
   the interesting rows are the non-zero ones, published with their legs.
5. **Every check takes a tolerance.** Noise inside the bid-ask spread is never
   called arbitrage.
6. **The publish gate is real.** Fewer than 300 two-sided quotes on either chain
   and `live_page.py` exits 1 and publishes nothing. Never weaken it to make CI
   green — a thin chain produces a confident-looking wrong surface.

---

## Why ESSVI and not SSVI

SSVI carries one `rho` for the entire surface. A crypto chain spans roughly 2 to
300 days, and the skew differs enormously across that range, so a single rho
cannot fit both ends. ESSVI (Hendriks & Martini 2019) gives each expiry its own
`(theta, rho, psi)` while keeping all four arbitrage conditions structural.

`chain.py` fits ESSVI as a backbone, then optionally refines each expiry with a
raw `SVISlice` that is arbitrage-*checked* on a dense grid. **A slice that fails
its check falls back to the backbone** and the report records the fallback
reason per expiry (`refined["slices_fallback"]`, e.g. `{"3.8d": "calendar_fail"}`).
Fallbacks are published, not hidden.

---

## The Deribit convention — verified, and easy to get wrong

Deribit options are **inverse** (coin-settled): prices are quoted in BTC/ETH per
contract, strikes in USD, `interest_rate` 0. The relation, verified against the
venue's own marks to 1e-4 relative:

```
price_in_coin * F  ==  Black76(F, K, T, sigma)     with F = underlying_price
```

**F is the forward, not the index spot.** Using spot is off by the basis —
**3.7% at 292 days** — and silently biases every implied vol and every parity
relation. There is a test pinning this (`tests/test_deribit_convention.py`).
`mark_iv` is kept alongside as the venue's own number, for comparison only; our
IVs come from our own bracketed bisection on bid and ask.

---

## Numerical guards

`__init__.py` states the thesis: the hard part is not the formulae, it is the
handful of numerical failures that silently poison a naive implementation — a
branch cut that flips sign mid-integration, a 0/0 that is finite in the limit
but NaN in floating point, a drift term whose omission looks like a small
pricing error and is actually a lost martingale property.

Each guard is documented at its site with the symptom it prevents. Known ones:

- `options.py:58` — stable boundaries against division by zero and NaN gradients
- `options.py:427,436,440` — Barone-Adesi-Whaley only computes the early-exercise
  premium when `b < r`; otherwise `S* = K` and masked `A2`, because `inf*0 = NaN`
- `options.py:641` — Albrecher et al. (2007) "little trap" branch-cut-stable
  formulation for the Heston characteristic function

Do not remove a guard because a test passes without it. The failures these
prevent are silent and appear at specific parameter values.

---

## Commands

```bash
./venv/bin/python -m pytest -q            # 55 passed, 2 skipped
./venv/bin/python scripts/live_page.py    # fetch, fit, archive, render docs/
```

The `surface` GitHub Action runs `live_page.py` every 30 minutes on the free
tier (~1 min per run), commits `docs/`, and therefore picks up the archive with
no workflow change. `docs/index.html` is static HTML with inline SVG and no
JavaScript libraries — keep it that way.

---

## Sibling repos

- `../tirramind-universe` — **the first product**: point-in-time US securities
  reference data with filing-level provenance. Same shape of moat, better legal
  footing (SEC filings are US Government works), weaker incumbents. Read its
  `CLAUDE.md` before deciding what to work on.
- `../pipelie` — data-quality checks on PyPI; worth running over any new output
  table before it is published.
