"""Run every HDC analysis and write output/hdc/results.json for the report.

    python -m hdc.run            # uses whatever real option data has been downloaded so far
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from hdc import synthetic
from hdc.backtest import build, calibrate, tbill, yahoo_series
from hdc.data import cache_dir
from hdc.portfolio import CAPITAL, metrics, simulate, spy_curve
from hdc.real_options import REAL_FROM
from hdc.trades import Costs
from hdc.universe import load_yahoo
from tfs.config import load_config, resolve

START, END = "2010-01-04", "2026-11-24"
P_SWEEP = (0.0, 0.15, 0.30, 0.50, 0.75, 1.0)


def trade_table(trades: list[dict]) -> pd.DataFrame:
    rows = []
    for t in trades:
        k, a, u, h = (t["paths"][x] for x in ("kept", "assigned", "unhedged", "hold61"))
        rows.append({"ticker": t["ticker"], "ex": t["ex"], "year": t["ex"][:4], "s0": t["s0"], "div": t["div"] / t["s0"],
                     "e0": t["e0"] / t["s0"], "itm": t["itm_strikes"], "rational": t["rational_exercise"],
                     "stock": k["stock"] / t["s0"], "option": k["option"] / t["s0"], "cost": k["cost"] / t["s0"],
                     "net": k["net"] / t["s0"], "kind": k["kind"], "assigned_net": a["net"] / t["s0"] if a else np.nan,
                     "itm_at_exm1": a is not None, "unhedged_net": u["net"] / t["s0"],
                     "hold61_net": h["net"] / t["s0"] if h else np.nan, "filled_days": t["filled_days"]})
    return pd.DataFrame(rows)


def per_trade_stats(df: pd.DataFrame, col: str = "net") -> dict:
    x = df[col].dropna()
    return {"n": int(len(x)), "avg": float(x.mean()), "median": float(x.median()), "win": float((x > 0).mean()),
            "p05": float(x.quantile(0.05)), "p01": float(x.quantile(0.01)), "worst": float(x.min()),
            "t": float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 else np.nan}


def run_sims(trades, cal, rate, costs, path, tax, p, n_sims, seed=0, **kw):
    """Monte Carlo over which calls get exercised early (probability p for each call in the money
    the evening before the ex-date). Returns the median run's metrics and the spread."""
    if p in (0.0, 1.0) or n_sims <= 1:
        a = np.full(len(trades), p == 1.0)
        res = simulate(trades, cal, rate, costs, path, a, tax, **kw)
        m = metrics(res["equity"], rate)
        m.update(win=float((res["log"].net > 0).mean()), trades=int(len(res["log"])), taxes=float(res["taxes"].sum()))
        return {"median": m, "runs": [m], "result": res}
    rng = np.random.default_rng(seed)
    runs, keep = [], None
    for _ in range(n_sims):
        a = rng.random(len(trades)) < p
        res = simulate(trades, cal, rate, costs, path, a, tax, **kw)
        m = metrics(res["equity"], rate)
        m.update(win=float((res["log"].net > 0).mean()), trades=int(len(res["log"])), taxes=float(res["taxes"].sum()))
        runs.append(m)
        keep = keep or res
    med = {k: float(np.median([r[k] for r in runs])) for k in runs[0] if isinstance(runs[0][k], (int, float))}
    med["cagr_p05"] = float(np.percentile([r["cagr"] for r in runs], 5))
    med["cagr_p95"] = float(np.percentile([r["cagr"] for r in runs], 95))
    return {"median": med, "runs": runs, "result": keep}


def rational_assign(trades) -> np.ndarray:
    return np.array([t["rational_exercise"] for t in trades])


def high_yield_basket(cal: list[str]) -> pd.Series:
    """Equal-weight buy-and-hold of the qualifying stocks (large cap, yield > 3%), rebalanced monthly,
    dividends reinvested. Members: a qualifying ex-date in the previous 120 days. Survivorship caveat:
    only stocks Yahoo still has."""
    ev = pd.read_parquet(cache_dir() / "hdc" / "events.parquet")
    rets = {}
    for t in ev.ticker.unique():
        y = load_yahoo(t)
        rets[t] = y.adj_close.pct_change()
    R = pd.DataFrame(rets).reindex(cal)
    months = sorted({d[:7] for d in cal})
    val, out = 1.0, {}
    for mo in months:
        days = [d for d in cal if d[:7] == mo]
        lo = (pd.Timestamp(days[0]) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
        mem = sorted(set(ev[(ev.ex >= lo) & (ev.ex < days[0])].ticker))
        w = np.ones(len(mem)) / len(mem) if mem else None
        sub = R.loc[days, mem].fillna(0.0) if mem else None
        grow = np.ones(len(mem)) if mem else None
        for d in days:
            if mem:
                grow = grow * (1 + sub.loc[d].to_numpy())
                out[d] = val * float(w @ grow)
            else:
                out[d] = val
        val = out[days[-1]]
    return pd.Series(out)


def strip(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("runs", "result")}


def main(n_sims: int = 40):
    t0 = time.time()
    spy = yahoo_series("SPY")
    cal = [d for d in spy.index if START <= d <= END]
    rate = tbill()
    out = {"period": [cal[0], cal[-1]], "capital": CAPITAL}

    cb = calibrate()
    m = cb["vol_mult"] if cb.get("n", 0) >= 30 else synthetic.VOL_MULT
    out["calibration"] = {k: v for k, v in cb.items() if k != "grid"} | {"used": m, "grid": cb.get("grid")}
    print(f"vol multiplier {m} from {cb.get('n', 0)} real prices ({time.time() - t0:.0f}s)")

    costs = Costs()
    tr, skips = build("model", costs, m)
    tr_above, _ = build("model", costs, m, rule="above")
    out["skips"] = {k: int(v) for k, v in skips.items()}
    tt = trade_table(tr)
    out["per_trade"] = {"spec_rule": per_trade_stats(tt), "assigned_path": per_trade_stats(tt, "assigned_net"),
                        "unhedged": per_trade_stats(tt, "unhedged_net"), "hold61": per_trade_stats(tt, "hold61_net"),
                        "rule_flipped": per_trade_stats(trade_table(tr_above))}
    out["components"] = {c: float(tt[c].mean()) for c in ("div", "e0", "stock", "option", "cost", "net")}
    out["kinds"] = tt.kind.value_counts().to_dict()
    out["rational_share"] = float(tt.rational.mean())
    out["itm_at_exm1_share"] = float(tt.itm_at_exm1.mean())
    out["by_year"] = tt.groupby("year").agg(n=("net", "size"), avg=("net", "mean"), win=("net", lambda s: float((s > 0).mean())),
                                            unhedged=("unhedged_net", "mean")).reset_index().to_dict("records")
    print(f"trades {len(tr)} ({time.time() - t0:.0f}s)")

    # account level: scenarios
    sc = {}
    curves = {}
    for name, path, tax in (("A_pre_tax", "kept", "pre_tax"), ("A_short_term", "kept", "A_short_term"),
                            ("B_pre_tax", "hold61", "pre_tax"), ("B_as_specified", "hold61", "B_as_specified"),
                            ("B_actual_law", "hold61", "B_actual_law"), ("unhedged_pre_tax", "unhedged", "pre_tax"),
                            ("unhedged_A", "unhedged", "A_short_term"), ("A_short_term_offset", "kept", "A_short_term_offset"),
                            ("B_as_specified_offset", "hold61", "B_as_specified_offset"),
                            ("B_actual_law_offset", "hold61", "B_actual_law_offset"),
                            ("unhedged_A_offset", "unhedged", "A_short_term_offset")):
        r = run_sims(tr, cal, rate, costs, path, tax, 0.0, 1)
        sc[name] = strip(r["median"])
        curves[name] = r["result"]["equity"]
    # early assignment
    sweep = {}
    for p in P_SWEEP:
        sweep[str(p)] = {tx: strip(run_sims(tr, cal, rate, costs, "kept", tx, p, n_sims, seed=1)["median"])
                         for tx in ("pre_tax", "A_short_term", "A_short_term_offset")}
        print(f"  assignment {p}: {sweep[str(p)]['pre_tax']['cagr']:+.2%} ({time.time() - t0:.0f}s)")
    ra = rational_assign(tr)
    for tax in ("pre_tax", "A_short_term", "A_short_term_offset"):
        res = simulate(tr, cal, rate, costs, "kept", ra, tax)
        mm = metrics(res["equity"], rate)
        mm.update(win=float((res["log"].net > 0).mean()), trades=int(len(res["log"])))
        sweep.setdefault("rational", {})[tax] = mm
    curves["A_p15"] = run_sims(tr, cal, rate, costs, "kept", "pre_tax", 0.15, 1, seed=1)["result"]["equity"]
    out["scenarios"], out["assignment"] = sc, sweep

    # option bid-ask sensitivity (not in the spec's costs)
    sens = {}
    for hs in (0.0, 0.02, 0.05):
        c2 = Costs(opt_half_spread=hs)
        t2 = tr if hs == 0 else build("model", c2, m)[0]
        r2 = run_sims(t2, cal, rate, c2, "kept", "A_short_term", 0.0, 1)
        sens[str(hs)] = {"per_trade": per_trade_stats(trade_table(t2)), "A_short_term": strip(r2["median"]),
                         "A_short_term_offset": strip(run_sims(t2, cal, rate, c2, "kept", "A_short_term_offset", 0.0, 1)["median"]),
                         "pre_tax": strip(run_sims(t2, cal, rate, c2, "kept", "pre_tax", 0.0, 1)["median"])}
    out["option_spread"] = sens
    print(f"spread sensitivity done ({time.time() - t0:.0f}s)")

    # hybrid: 80% SPY core + 20% HDC sleeve; the sleeve's after-tax gain each year moves into SPY
    hyb = {}
    for label, tax, dtax in (("pre_tax", "pre_tax", 0.0), ("after_tax", "A_short_term", 0.20),
                             ("after_tax_offset", "A_short_term_offset", 0.20)):
        sl = simulate(tr, cal, rate, costs, "kept", None, tax, capital=0.2 * CAPITAL, sweep=True)
        core = spy_curve(spy, cal, dtax, 0.8 * CAPITAL, buys=sl["sweeps"])
        h = core + sl["equity"]
        spy_only = spy_curve(spy, cal, dtax, CAPITAL)
        hyb[label] = {"hybrid": metrics(h, rate), "spy": metrics(spy_only, rate),
                      "swept_total": float(sl["sweeps"].sum())}
        curves[f"hybrid_{label}"], curves[f"spy_{label}"] = h, spy_only
    out["hybrid"] = hyb
    hy = high_yield_basket(cal)
    out["benchmarks"] = {"spy_pre_tax": metrics(spy_curve(spy, cal), rate),
                         "spy_after_div_tax": metrics(spy_curve(spy, cal, 0.20), rate),
                         "high_yield_basket": metrics(hy, rate)}
    curves["high_yield_basket"] = hy * CAPITAL

    # real option prices (Dec 2024 on): same events priced by the model, for comparison
    rtr, rsk = build("real", costs, m)
    real_events = {(t["ticker"], t["ex"]) for t in rtr}
    ev = pd.read_parquet(cache_dir() / "hdc" / "events.parquet")
    sub = ev[[(a, b) in real_events for a, b in zip(ev.ticker, ev.ex)]]
    mtr_same, _ = build("model", costs, m, events=sub)
    out["real"] = {"skips": {k: int(v) for k, v in rsk.items()}, "n": len(rtr),
                   "per_trade": per_trade_stats(trade_table(rtr)) if rtr else None,
                   "assigned_path": per_trade_stats(trade_table(rtr), "assigned_net") if rtr else None,
                   "unhedged": per_trade_stats(trade_table(rtr), "unhedged_net") if rtr else None,
                   "model_same_events": per_trade_stats(trade_table(mtr_same)) if mtr_same else None,
                   "filled_share": float(np.mean([t["filled_days"] > 0 for t in rtr])) if rtr else None,
                   "rational_share": float(np.mean([t["rational_exercise"] for t in rtr])) if rtr else None,
                   "components": {c: float(trade_table(rtr)[c].mean()) for c in ("div", "e0", "stock", "option", "cost", "net")} if rtr else None}
    out["n_events"] = int(len(ev)), int((ev.earnings == "clear").sum())
    out["n_tickers"] = int(ev.ticker.nunique())

    od = resolve(load_config(), "output") / "hdc"
    od.mkdir(parents=True, exist_ok=True)
    tt.to_csv(od / "trades_model.csv", index=False)
    if rtr:
        trade_table(rtr).to_csv(od / "trades_real.csv", index=False)
    pd.DataFrame({k: v for k, v in curves.items()}).to_csv(od / "curves.csv")
    (od / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"HDC RUN DONE ({time.time() - t0:.0f}s)")
    return out


if __name__ == "__main__":
    main()
