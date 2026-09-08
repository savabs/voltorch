"""B1: fit both Deribit chains, append to the history, render docs/index.html.

Static HTML, inline SVG, no JavaScript libraries. Every number on the page
comes from this run; the history table is the only state and it is
append-only.
"""

from __future__ import annotations

import gzip
import html
import io
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from voltorch import __version__
from voltorch.chain import fit_chain
from voltorch.deribit import fetch_chain

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
HIST = os.path.join(DOCS, "history.jsonl")
ARCHIVE = os.path.join(DOCS, "archive")
MIN_TWO_SIDED = 300
CSS = """body{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:24px 16px;color:#111;background:#fff}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px}h3{font-size:15px;margin:18px 0 4px}
table{border-collapse:collapse;font-size:13px;margin:6px 0 14px}th,td{border-bottom:1px solid #ddd;padding:3px 8px;text-align:right}th:first-child,td:first-child{text-align:left}
th{background:#f4f4f4}.muted{color:#666}.ok{color:#1a7f37}.bad{color:#b42318}code{background:#f4f4f4;padding:1px 4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:8px}svg{max-width:100%;height:auto}"""


def smile_svg(slice_: dict) -> str:
    k = np.array(slice_["k"]); bid = np.array(slice_["bid_iv"]) * 100; ask = np.array(slice_["ask_iv"]) * 100
    mark = np.array(slice_["mark_iv"]) * 100; fit = np.array(slice_["fit_iv"]) * 100; ref = np.array(slice_["ref_iv"]) * 100
    fig, ax = plt.subplots(figsize=(4.2, 2.6), dpi=100)
    ax.fill_between(k, bid, ask, color="#cfe3ff", label="bid–ask")
    ax.plot(k, mark, ".", ms=4, color="#333", label="venue mark")
    ax.plot(k, fit, "-", lw=1, color="#999", label="eSSVI backbone")
    ax.plot(k, ref, "-", lw=1.4, color="#b42318" if slice_["refined_status"] != "ok" else "#1a7f37",
            label="refined" + ("" if slice_["refined_status"] == "ok" else f" ({slice_['refined_status']} → backbone)"))
    ax.set_title(f"{slice_['T']*365:.1f}d  F={slice_['forward']:,.0f}  n={slice_['n']}  rmse={slice_['refined_rmse_vol_pts']:.2f}  inside={slice_['refined_inside_bid_ask']:.0%}", fontsize=8)
    ax.set_xlabel("log-moneyness k = ln(K/F)", fontsize=7); ax.set_ylabel("IV %", fontsize=7); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper center", ncol=2, frameon=False)
    buf = io.StringIO(); fig.tight_layout(); fig.savefig(buf, format="svg"); plt.close(fig)
    return buf.getvalue()


def section(r) -> str:
    v = r.venue_violations; ex = v["executable"]
    n_ex = sum(len(x) for x in ex.values())
    rows = "".join(f"<tr><td>{s['T']*365:.1f}d</td><td>{s['n']}</td><td>{s['rmse_vol_pts']:.2f}</td><td>{s['inside_bid_ask']:.0%}</td>"
                   f"<td>{s['refined_rmse_vol_pts']:.2f}</td><td>{s['refined_inside_bid_ask']:.0%}</td><td class='{'ok' if s['refined_status']=='ok' else 'bad'}'>{s['refined_status']}</td></tr>" for s in r.slices)
    exrows = "".join(f"<tr><td>{kind}</td><td>{html.escape(json.dumps({k: v for k, v in row.items() if k != 'edge_usd'}))}</td><td>{row['edge_usd']:.2f}</td></tr>"
                     for kind in ("butterfly", "vertical", "calendar") for row in sorted(ex[kind], key=lambda x: -x["edge_usd"])[:10])
    g = r.greeks_max_abs_err
    return f"""
<h2>{r.currency} — {r.n_quotes} quotes, {r.n_two_sided} two-sided, {r.n_fit} OTM used, {r.expiries} expiries</h2>
<table><tr><th>surface</th><th>RMSE (vol pts)</th><th>inside bid–ask</th><th>arbitrage-free</th></tr>
<tr><td>eSSVI backbone</td><td>{r.fit['rmse_vol_pts']:.2f}</td><td>{r.inside_bid_ask_share:.0%}</td><td class='ok'>by construction (Hendriks–Martini conditions hold: {r.fit['conditions']})</td></tr>
<tr><td>refined (per-slice SVI)</td><td>{r.refined['rmse_vol_pts']:.2f}</td><td>{r.refined['inside_bid_ask_share']:.0%}</td><td class='ok'>checked: Durrleman g ≥ 0 and calendar, {r.refined['slices_refined']}/{r.expiries} slices; fallback {html.escape(json.dumps(r.refined['slices_fallback']))}</td></tr>
<tr><td>venue marks (Deribit <code>mark_iv</code>)</td><td>—</td><td>—</td><td class='ok'>butterfly {len(v['butterfly'])} · vertical {len(v['vertical'])} · calendar {len(v['calendar'])} beyond the spread</td></tr></table>
<p><b>Executable arbitrage in the book right now: {n_ex}</b> <span class='muted'>(buy at ask / sell at bid; butterfly {len(ex['butterfly'])}, vertical {len(ex['vertical'])}, calendar {len(ex['calendar'])}; put–call parity on the forward, r = 0)</span></p>
{'<table><tr><th>kind</th><th>legs</th><th>edge USD</th></tr>' + exrows + '</table>' if n_ex else ''}
<p class='muted'>Our surface: Durrleman min g = {r.our_violations['durrleman_min_g']:.3f}, calendar min Δw = {r.our_violations['calendar_min_dw']:.1e}. Autograd vs closed-form Black-76 max |err|: delta {g['delta']:.1e}, gamma {g['gamma']:.1e}, vega {g['vega']:.1e}, theta {g['theta']:.1e}. Fit {r.timings_s['fit']}s, refine {r.refined['time_s']}s, total {r.timings_s['total']}s on {r.device}.</p>
<h3>Per expiry</h3><table><tr><th>expiry</th><th>n</th><th>backbone rmse</th><th>inside</th><th>refined rmse</th><th>inside</th><th>status</th></tr>{rows}</table>
<div class='grid'>{''.join(smile_svg(s) for s in r.slices)}</div>"""


def archive(r, now: datetime) -> str:
    """Write the whole fitted surface and the quotes it was fitted to.

    ``history.jsonl`` keeps fit diagnostics — rmse, counts, violations — which
    say how good a fit was but do not contain the surface. A diagnostic cannot
    be refitted, resampled or backtested; the quotes can. This is the only part
    of the run that is unrecoverable once the book moves, so it is written
    first and separately, one immutable file per run rather than an appended
    file, so that nothing already recorded is ever rewritten.

    Roughly 10 KB gzipped per currency per run (~1 MB/day for both chains).
    """
    path = os.path.join(ARCHIVE, f"{now:%Y/%m/%d}")
    os.makedirs(path, exist_ok=True)
    name = os.path.join(path, f"{now:%H%M}Z-{r.currency}.json.gz")
    payload = {
        "as_of": r.as_of,
        "currency": r.currency,
        "voltorch_version": __version__,
        "n_quotes": r.n_quotes,
        "n_two_sided": r.n_two_sided,
        "n_fit": r.n_fit,
        "fit": r.fit,
        "refined": r.refined,
        "venue_violations": r.venue_violations,
        "our_violations": r.our_violations,
        "greeks_max_abs_err": r.greeks_max_abs_err,
        # the irreplaceable part: per-expiry forwards, the two-sided book in
        # vol space, and both fitted curves evaluated on the same grid
        "slices": [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in s_.items()} for s_ in r.slices],
    }
    with gzip.open(name, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return name


def main() -> int:
    now = datetime.now(timezone.utc)
    reports = []
    for cur in ("BTC", "ETH"):
        df = fetch_chain(cur, now=now)
        if int(df["two_sided"].sum()) < MIN_TWO_SIDED:
            print(f"{cur}: only {int(df['two_sided'].sum())} two-sided quotes — refusing to publish")
            return 1
        r = fit_chain(df, currency=cur, as_of=now.isoformat(timespec="seconds"))
        reports.append(r)
        print(f"{cur}: quotes={r.n_quotes} refined rmse={r.refined['rmse_vol_pts']:.2f} inside={r.refined['inside_bid_ask_share']:.0%} executable={sum(len(x) for x in r.venue_violations['executable'].values())}")
    os.makedirs(DOCS, exist_ok=True)
    for r in reports:
        print(f"archived {archive(r, now)}")
    with open(HIST, "a", encoding="utf-8") as fh:
        for r in reports:
            ex = r.venue_violations["executable"]
            fh.write(json.dumps({"as_of": r.as_of, "currency": r.currency, "n_quotes": r.n_quotes, "n_fit": r.n_fit,
                                 "backbone_rmse": round(r.fit["rmse_vol_pts"], 3), "refined_rmse": round(r.refined["rmse_vol_pts"], 3),
                                 "refined_inside": round(r.refined["inside_bid_ask_share"], 3),
                                 "executable": {k: len(v) for k, v in ex.items()},
                                 "executable_max_edge_usd": round(max([x["edge_usd"] for k in ex.values() for x in k] or [0.0]), 2),
                                 "fallback": r.refined["slices_fallback"]}) + "\n")
    hist = [json.loads(l) for l in open(HIST, encoding="utf-8") if l.strip()]
    cutoff = (now.timestamp() - 7 * 86400)
    recent = [h for h in hist if datetime.fromisoformat(h["as_of"]).timestamp() >= cutoff]
    hrows = "".join(f"<tr><td>{h['as_of'][:16]}Z</td><td>{h['currency']}</td><td>{h['n_fit']}</td><td>{h['backbone_rmse']:.2f}</td><td>{h['refined_rmse']:.2f}</td><td>{h['refined_inside']:.0%}</td>"
                    f"<td>{sum(h['executable'].values())}</td><td>{h['executable_max_edge_usd']:.0f}</td></tr>" for h in reversed(recent[-96:]))
    page = f"""<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>voltorch — live arbitrage-free Deribit surface</title><style>{CSS}</style>
<h1>Live arbitrage-free volatility surface — Deribit BTC &amp; ETH</h1>
<p class='muted'>Generated {now:%Y-%m-%d %H:%M} UTC by <a href='https://github.com/savabs/voltorch'>voltorch</a> {__version__}, refit every 30 minutes from the public book. No key, no vendor data.
Backbone: eSSVI (Hendriks–Martini 2019), arbitrage-free <em>by construction</em>. Refinement: per-expiry SVI, arbitrage-<em>checked</em> on a dense grid, falling back to the backbone where a check fails.
Bid/ask implied vols are ours (bisection, Black-76 on the forward: coin price × F = USD price, verified to 1e-4 against the venue's marks). Every claim on this page is recomputable from <code>pip install voltorch[page]</code>.</p>
{''.join(section(r) for r in reports)}
<h2>History (last 7 days, newest first)</h2>
<table><tr><th>as of</th><th>ccy</th><th>n fit</th><th>backbone rmse</th><th>refined rmse</th><th>inside</th><th>executable arb</th><th>max edge USD</th></tr>{hrows}</table>
<p class='muted'>Raw history: <a href='history.jsonl'>history.jsonl</a> (append-only). A day with zero executable arbitrage is the expected state of a market-made venue; the interesting rows are the non-zero ones, and they are listed with their legs above when they occur.</p>"""
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    print("page written", os.path.join(DOCS, "index.html"), f"{len(page)/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
