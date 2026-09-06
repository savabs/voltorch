# An arbitrage-free volatility surface for Deribit, refit every 30 minutes, with its errors published

*Draft 2026-09-06. Live page: https://savabs.github.io/voltorch/ · `pip install "voltorch[chain]"`*

## What it is

Every 30 minutes a GitHub Actions job pulls the full BTC and ETH option books
from Deribit's public API (no key), fits a volatility surface that **cannot
contain butterfly or calendar arbitrage**, and publishes: the fit error per
expiry in vol points, the share of quotes where the fitted vol sits inside the
bid–ask spread, every executable arbitrage in the book (buy at ask, sell at
bid), and the agreement between autograd greeks and closed-form Black-76. All
of it is recomputable from the library in about five seconds on a laptop CPU.

## The surface

Two layers, labelled separately on the page because they carry different
guarantees:

1. **eSSVI backbone** (Hendriks & Martini 2019). One (θ, ρ, ψ) per expiry.
   The butterfly conditions ψ(1+|ρ|) < 4, ψ²(1+|ρ|) ≤ 4θ and the calendar
   conditions θ↑, ψ↑, |ρⱼψⱼ − ρᵢψᵢ| ≤ ψⱼ − ψᵢ are built into the
   parameterisation, so the surface is arbitrage-free at every optimiser
   step, not just at the end. Fit: ~1.8 vol points RMSE, ~62% of quotes
   inside the spread. That is the price of three parameters per expiry on a
   crypto chain that runs from 2 to 300 days.
2. **Per-expiry SVI refinement.** Five parameters per slice, warm-started
   from the backbone and from the quotes, fitted in annualised variance so a
   2-day and a 300-day slice are on the same numeric scale. Nothing
   structural prevents arbitrage here, so each slice is *checked* — Durrleman
   g(k) ≥ 0 on a 300-point grid, calendar against both neighbours on the range
   where both were validated — and a slice that fails falls back to the
   backbone and is labelled on the page. Fit: **BTC 0.64 vol points, 86%
   inside the spread; ETH 0.66, 96%**, with the 200–300-day expiries at
   0.15–0.45 and 96–100% inside.

## Three bugs the live book found in the library

They are the reason this page exists, and each is now a test.

- **The Newton implied-vol solver stalled on ordinary quotes.** A Deribit put
  (BTC-25SEP26-73000-P, 18 days, forward 79,907) priced at 0.0060/0.0070 BTC
  — perfectly consistent with its neighbours — came back with an implied vol
  of 0.06 instead of 0.39, because fixed-iteration Newton from a
  Brenner–Subrahmanyam start sits where vega ≈ 0. Quotes now go through a
  64-step bracketed bisection that cannot fail.
- **Coin prices convert to USD on the forward, not spot.** Deribit's
  `mark_price × underlying_price` equals Black-76 on the forward to 1e-4;
  using the index instead is off by the basis — 3.7% at 292 days — and
  silently biases every implied vol and every parity relation. Before the
  fix the page showed $2,000 "arbitrages" on 190k-strike butterflies; after
  it, zero.
- **The slice fitter's warm start depended on T.** The same annualised
  smile fitted to 0.21 vol points at 2 days and 0.055 at 300 days. A
  T-invariant start from the quotes' own moments, plus an L-BFGS finish,
  brought them within 0.05.

## What the book actually contains

Two things the page says that a vendor's page would not:

- **Deribit's own marks contain no arbitrage.** They are model-generated. A
  claim that "the venue's marks contain N arbitrages a day" — which I had
  planned to make — is false, and the page checks it every half hour rather
  than asserting it.
- **Executable arbitrage in the book was zero** in every run so far. That is
  the expected state of a market-made venue. The count is on the page with
  its legs and edge in USD; when it is non-zero it will be listed, and the
  7-day history will show when.

## Greeks

Autograd delta/gamma/vega/theta against closed-form Black-76 across the whole
OTM chain: max |error| ≤ 1e-9. That number is on the page every run.

## Use it

```python
from voltorch import fit_chain
from voltorch.deribit import fetch_chain
r = fit_chain(fetch_chain("BTC"), currency="BTC")
r.refined, r.venue_violations["executable"], r.our_violations, r.greeks_max_abs_err
```

Apache-2.0. If a number on the page looks wrong, the run that produced it is
in `docs/history.jsonl` and the code that produced it is one `pip install`
away.
