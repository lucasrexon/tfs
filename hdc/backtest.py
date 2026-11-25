"""Turn events into trades, with model-priced calls (2010-2026) or real calls (Dec 2024 on)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hdc import synthetic
from hdc.data import cache_dir
from hdc.real_options import pick, window_events
from hdc.trades import Costs, outcomes, select_call
from hdc.universe import load_yahoo


def yahoo_series(sym: str, start: str = "2009-01-01") -> pd.DataFrame:
    """SPY, ^IRX and other index series, cached like the stocks."""
    f = cache_dir() / "hdc" / "yahoo" / f"_{sym.replace('^', '')}.parquet"
    if not f.exists():
        import yfinance as yf
        d = yf.Ticker(sym).history(start=start, end="2026-11-25", auto_adjust=False, actions=True)
        d.index = pd.to_datetime(d.index).strftime("%Y-%m-%d")
        d = d.rename(columns=str.lower).rename(columns={"adj close": "adj_close", "stock splits": "splits"})
        d.to_parquet(f)
    return pd.read_parquet(f)


def tbill() -> pd.Series:
    """13-week T-bill yield as a decimal, indexed by Timestamp (forward-filled)."""
    s = yahoo_series("^IRX").close / 100.0
    s.index = pd.to_datetime(s.index)
    return s.ffill()


class Stock:
    def __init__(self, t: str):
        y = load_yahoo(t)
        self.closes, self.adj = y.close, y.adj_close
        self.divs = y.dividends[y.dividends > 0]

    def window(self, start: str, n: int = 60) -> pd.Series:
        return self.closes.loc[start:].iloc[:n]


# ---------------------------------------------------------------- real option prices from the cache
def _bars(opt: str, start: str, end: str) -> pd.DataFrame | None:
    f = cache_dir() / "hdc" / "optbars" / f"{opt.replace(':', '_')}_{start}_{end}.json"
    if not f.exists():
        return None
    rows = json.loads(f.read_text())
    if not rows:
        return pd.DataFrame(columns=["c", "v"])
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    return df[["c", "v"]]


def real_quotes(ev: dict):
    """Real entry prices (adjusted to Yahoo's split basis) for the event's 1- and 2-strike ITM calls,
    or None if the event's option data has not been downloaded."""
    f = cache_dir() / "hdc" / "contracts" / f"{ev['ticker']}.json"
    if not f.exists():
        return None
    p = pick(pd.DataFrame(json.loads(f.read_text())), ev["entry"], ev["s0"] * ev["real_factor"])
    if p is None:
        return {"expiry": None, "quotes": [], "bars": {}}
    exp, strikes = p
    fac = ev["real_factor"]
    qs, bars = [], {}
    for i, (opt, k) in enumerate(strikes):
        b = _bars(opt, ev["entry"], exp)
        if b is None:
            if i == 0:
                return None                                   # not downloaded yet
            break                                             # second strike not needed (or not fetched)
        k_adj = float(k) / fac
        c = float(b.loc[ev["entry"], "c"]) / fac if ev["entry"] in b.index else None
        qs.append((k_adj, c))
        bars[k_adj] = b["c"] / fac
    return {"expiry": exp, "quotes": qs, "bars": bars}


# ---------------------------------------------------------------- trades
def make_trade(ev: dict, st: Stock, rate: pd.Series, costs: Costs, mode: str, vol_mult: float,
               rule: str = "below") -> dict | None:
    closes = st.window(ev["entry"], 60)
    if ev["exit"] not in closes.index:
        return None
    sigma = synthetic.realized_vol(st.adj, ev["entry"]) * vol_mult
    if mode == "model":
        q = synthetic.quotes(ev, closes, st.adj, rate, vol_mult)
        if q is None:
            return {"skip": "no 7-14 day expiry"}
        expiry, qs, real_bars = q["expiry"], q["quotes"], {}
    else:
        rq = real_quotes(ev)
        if rq is None:
            return None
        if rq["expiry"] is None:
            return {"skip": "no 7-14 day expiry"}
        expiry, qs, real_bars = rq["expiry"], rq["quotes"], rq["bars"]
        expiry = max([d for d in closes.index if d <= expiry], default=expiry)
    if not qs or all(c is None for _, c in qs):
        return {"skip": "no option trade at entry"}
    sel = select_call(ev["s0"], ev["div"], qs, rule)
    if sel is None:
        return {"skip": "rule not met"}
    k, c0, e0 = sel
    dates = [d for d in closes.index if d <= expiry]
    model = synthetic.call_series(closes, dates, k, expiry, ev["ex"], ev["div"], sigma, rate)
    filled = 0
    if mode == "real":
        rb = real_bars.get(k, pd.Series(dtype=float))
        calls = pd.Series({d: (float(rb[d]) if d in rb.index else np.nan) for d in dates})
        filled = int(calls.isna().sum())
        calls = calls.fillna(model)
        calls[ev["entry"]] = c0
        calls = np.maximum(calls, closes.reindex(calls.index) - k)      # never below intrinsic
    else:
        calls = model
    o = outcomes(ev, k, c0, closes, calls, expiry, costs, st.divs)
    exm1 = list(closes.index)[list(closes.index).index(ev["ex"]) - 1]
    e_exm1 = float(calls.get(exm1, np.nan)) - (float(closes[exm1]) - k)
    return {"ticker": ev["ticker"], "ex": ev["ex"], "entry": ev["entry"], "exit": ev["exit"], "expiry": expiry,
            "div": ev["div"], "s0": ev["s0"], "k": k, "c0": c0, "e0": e0, "sigma": sigma, "yield": ev["yield"],
            "itm_strikes": 1 + [x for x, _ in qs].index(k), "e_exm1": e_exm1,
            "rational_exercise": bool(float(closes[exm1]) > k and e_exm1 < ev["div"]),
            "filled_days": filled, "mode": mode, "paths": o}


def build(mode: str, costs: Costs, vol_mult: float, rule: str = "below", events: pd.DataFrame | None = None,
          progress=None) -> tuple[list[dict], pd.Series]:
    if events is None:
        ev = pd.read_parquet(cache_dir() / "hdc" / "events.parquet")
        ev = ev[ev.earnings == "clear"]
        if mode == "real":
            ev = window_events()
    else:
        ev = events
    rate = tbill()
    trades, skips, cache = [], {}, {}
    for n, r in enumerate(ev.sort_values("entry").to_dict("records")):
        st = cache.get(r["ticker"]) or cache.setdefault(r["ticker"], Stock(r["ticker"]))
        t = make_trade(r, st, rate, costs, mode, vol_mult, rule)
        if t is None:
            skips["not downloaded"] = skips.get("not downloaded", 0) + 1
        elif "skip" in t:
            skips[t["skip"]] = skips.get(t["skip"], 0) + 1
        else:
            trades.append(t)
        if progress and n % 500 == 0:
            progress(f"  {mode}: {n}/{len(ev)} events, {len(trades)} trades")
    return trades, pd.Series(skips, dtype=int)


def calibrate(progress=print) -> dict:
    """Volatility multiplier that makes model time values match real ones on the entry day
    (all real calls with a trade that day, both strikes)."""
    rate = tbill()
    pts = []
    for r in window_events().to_dict("records"):
        rq = real_quotes(r)
        if not rq or not rq["expiry"]:
            continue
        st = Stock(r["ticker"])
        rv = synthetic.realized_vol(st.adj, r["entry"])
        for k, c in rq["quotes"]:
            if c is None or not np.isfinite(rv):
                continue
            pts.append((r, st, k, c, rv, rq["expiry"]))
    if not pts:
        return {"vol_mult": synthetic.VOL_MULT, "n": 0}
    grid = np.round(np.arange(0.6, 2.01, 0.05), 2)
    err = []
    for m in grid:
        e = 0.0
        for r, st, k, c, rv, exp in pts:
            closes = st.window(r["entry"], 30)
            mod = synthetic.call_series(closes, [r["entry"]], k, exp, r["ex"], r["div"], rv * m, rate).iloc[0]
            e += abs(mod - c)
        err.append(e / len(pts))
    best = float(grid[int(np.argmin(err))])
    real_e = np.array([max(c - (r["s0"] - k), 0) for r, st, k, c, rv, exp in pts])
    return {"vol_mult": best, "n": len(pts), "mean_abs_err": float(min(err)), "grid": dict(zip(map(float, grid), err)),
            "real_time_value_median": float(np.median(real_e))}
