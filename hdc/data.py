"""Data for the HDC backtest, all cached under data_cache/hdc/.

Real data (what this repo's environment can reach):
  - stock daily bars: Massive grouped daily files already cached by tfs (split-adjusted, NOT
    dividend-adjusted, so the ex-date drop is in the prices)
  - dividends: Massive /v3/reference/dividends (history back to 2010 on the free plan)
  - option contracts: Massive /v3/reference/options/contracts (listed strikes and expiries)
  - option daily bars: Massive /v2/aggs (last 2 years on the free plan; real trade prices, not quotes)
  - company size and earnings dates: SEC (shares outstanding; 8-K item 2.02 = results of operations)

hdc/yf_source.py has the yfinance equivalents for running locally over 2010-2026.
"""
from __future__ import annotations

import json
from datetime import date as Date, timedelta
from pathlib import Path

import pandas as pd

from tfs.config import load_config, resolve
from tfs.data.massive import MassiveClient


def cache_dir() -> Path:
    cfg = load_config()
    return resolve(cfg, cfg["data"]["cache_dir"])


def massive() -> MassiveClient:
    cfg = load_config()
    return MassiveClient.from_config(cfg, cache_dir())


def _paged(cl: MassiveClient, url: str, params: dict) -> list[dict]:
    out = []
    while url:
        j = cl._get(url, params)
        out += j.get("results") or []
        url, params = j.get("next_url"), None
    return out


def months(start: str, end: str) -> list[tuple[str, str]]:
    d, e, out = Date.fromisoformat(start).replace(day=1), Date.fromisoformat(end), []
    while d <= e:
        nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
        out.append((d.isoformat(), (nxt - timedelta(days=1)).isoformat()))
        d = nxt
    return out


def dividends(cl: MassiveClient, start: str, end: str, progress=print) -> pd.DataFrame:
    """Every US cash dividend with an ex-date in [start, end], one cached file per month."""
    root = cl.cache / "hdc" / "dividends"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for a, b in months(start, end):
        f = root / f"{a[:7]}.json"
        if not f.exists():
            r = _paged(cl, "/v3/reference/dividends", {"ex_dividend_date.gte": a, "ex_dividend_date.lte": b, "limit": 1000})
            f.write_text(json.dumps(r))
            progress(f"dividends {a[:7]}: {len(r)}")
        rows += json.loads(f.read_text())
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[(df.ex_dividend_date >= start) & (df.ex_dividend_date <= end)]
    return df.drop_duplicates(subset=["ticker", "ex_dividend_date", "cash_amount", "dividend_type"]).reset_index(drop=True)


def option_contracts(cl: MassiveClient, underlying: str, exp_from: str, exp_to: str,
                     k_lo: float, k_hi: float) -> pd.DataFrame:
    """Every listed call on `underlying` expiring in [exp_from, exp_to] with strike in [k_lo, k_hi],
    expired and still active."""
    f = cl.cache / "hdc" / "contracts" / f"{underlying}.json"
    if f.exists():
        return pd.DataFrame(json.loads(f.read_text()))
    rows = []
    for expired in ("true", "false"):
        rows += _paged(cl, "/v3/reference/options/contracts", {
            "underlying_ticker": underlying, "contract_type": "call", "expired": expired,
            "expiration_date.gte": exp_from, "expiration_date.lte": exp_to,
            "strike_price.gte": round(k_lo, 2), "strike_price.lte": round(k_hi, 2), "limit": 1000})
    keep = ["ticker", "underlying_ticker", "expiration_date", "strike_price", "shares_per_contract", "exercise_style"]
    rows = [{k: r.get(k) for k in keep} for r in rows]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rows))
    return pd.DataFrame(rows)


def option_bars(cl: MassiveClient, opt: str, start: str, end: str) -> pd.DataFrame:
    """Daily bars of one option contract (date-indexed; close = last trade of the day)."""
    f = cl.cache / "hdc" / "optbars" / f"{opt.replace(':', '_')}_{start}_{end}.json"
    if f.exists():
        rows = json.loads(f.read_text())
    else:
        rows = cl._get(f"/v2/aggs/ticker/{opt}/range/1/day/{start}/{end}", {"adjusted": "false", "limit": 5000}).get("results") or []
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rows))
    if not rows:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v", "vw"])
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    return df.reindex(columns=["o", "h", "l", "c", "v", "vw"])


def price_panel(tickers: set[str] | None = None) -> pd.DataFrame:
    """Long frame date, ticker, o, h, l, c, v from the cached Massive grouped daily files."""
    g = sorted((cache_dir() / "grouped").glob("*.parquet"))
    out = []
    for p in g:
        df = pd.read_parquet(p, columns=["T", "o", "h", "l", "c", "v"])
        if tickers is not None:
            df = df[df["T"].isin(tickers)]
        df.insert(0, "date", p.stem)
        out.append(df)
    return pd.concat(out, ignore_index=True).rename(columns={"T": "ticker"})


if __name__ == "__main__":
    import sys
    cl = massive()
    if sys.argv[1] == "dividends":
        d = dividends(cl, sys.argv[2], sys.argv[3])
        print("DIVIDENDS DONE", len(d), "calls", cl.calls)
