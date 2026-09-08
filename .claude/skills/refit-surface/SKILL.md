---
name: refit-surface
description: Run and verify a live Deribit surface refit — check that our own surface is arbitrage-free, that the archive wrote, and that the venue violations look sane. Use after changing ssvi.py, chain.py, arbitrage.py or deribit.py, or when asked to refit or check the live surface.
---

# Refitting and checking the live surface

## 1. Tests first

```bash
./venv/bin/python -m pytest -q
```

Expect **55 passed, 2 skipped**. `tests/test_deribit_convention.py` pins the
forward-vs-spot relation against the venue's own marks — if it fails, stop.
Using spot instead of the forward is off by the basis (3.7% at 292 days) and
silently biases every implied vol on the page.

## 2. Run a refit

```bash
./venv/bin/python scripts/live_page.py
```

It fetches both chains live, fits, archives, and renders `docs/index.html`.
It **exits 1 and publishes nothing** if either chain returns fewer than 300
two-sided quotes. That is correct behaviour, not a failure to work around — a
thin chain produces a confident-looking wrong surface.

## 3. The check that matters most

```bash
./venv/bin/python - <<'EOF'
import glob, gzip, json
f = sorted(glob.glob("docs/archive/*/*/*/*.json.gz"))[-2:]
for p in f:
    d = json.load(gzip.open(p, "rt"))
    ours = d["our_violations"]
    ex = d["venue_violations"]["executable"]
    print(f"{d['currency']} {d['as_of']}")
    print(f"  our_violations      : {ours}")
    print(f"  venue executable    : { {k: len(v) for k, v in ex.items()} }")
    print(f"  refined rmse (vol pts): {d['refined']['rmse_vol_pts']:.3f}"
          f"  inside bid-ask: {d['refined']['inside_bid_ask_share']:.1%}")
    print(f"  fallback slices     : {d['refined']['slices_fallback']}")
EOF
```

Read it in this order:

- **`our_violations` must be zero.** Not small — zero. The surface is
  arbitrage-free *by construction*, enforced by the parameterisation at every
  optimiser step, so a non-zero value means the parameterisation is broken. It
  is a bug, never a finding.
- **`venue_violations["executable"]` is the product.** These are measured
  against live bids and asks rather than mids, so a non-zero row is something a
  reader could in principle have traded. Zero is the expected state of a
  market-made venue; non-zero rows are the interesting ones and get published
  with their legs.
- **Fallback slices are normal and must stay visible.** A refined SVI slice that
  fails its arbitrage check falls back to the ESSVI backbone and records why
  (`{"3.8d": "calendar_fail"}`). Never suppress these.

## 4. Confirm the archive wrote

```bash
ls -la docs/archive/$(date -u +%Y/%m/%d)/
```

Two files per run, roughly 8 KB each. If `live_page.py` ran but nothing appeared
here, the archive path broke — fix that before anything else. `history.jsonl`
keeps only diagnostics; a diagnostic cannot be refitted or backtested, and the
book it came from is unrecoverable once the market moves.

## What not to do

- Do not replace the structural arbitrage constraints with a soft penalty. That
  turns the product from arbitrage-*free* into arbitrage-*checked*, which is
  what competitors already sell.
- Do not enforce Durrleman's g during fitting. It is a test of the
  implementation, and if you need it as a constraint the parameterisation has
  already been broken.
- Do not lower `MIN_TWO_SIDED` to make a run publish.
- Do not overwrite an archived run. A guard hook blocks it.
