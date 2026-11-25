"""Point-in-time large-cap dividend universe, 2010-2026.

No free source has point-in-time S&P 500 membership, so "large cap" is rebuilt from data:
  market cap on each date = real (unadjusted-for-splits) share price x shares outstanding from the
  latest SEC cover-page share count (dei:EntityCommonStockSharesOutstanding, via the XBRL frames API).
Candidates are NYSE/Nasdaq common stocks that file with the SEC. Prices and dividends come from
Yahoo Finance, which mostly lacks delisted stocks (a known survivorship gap; see the report).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hdc.data import cache_dir


def shares_frames() -> pd.DataFrame:
    """cik, period end, shares: every quarterly SEC frame cached in data_cache/sec/frames."""
    rows = []
    for f in sorted((cache_dir() / "sec" / "frames").glob("shares_CY*.json")):
        for d in json.loads(f.read_text()).get("data", []):
            rows.append({"cik": int(d["cik"]), "end": d["end"], "shares": float(d["val"]), "loc": d.get("loc") or ""})
    df = pd.DataFrame(rows).drop_duplicates(["cik", "end"]).sort_values(["cik", "end"])
    return df[df.shares > 0]


def candidates(min_cap_now: float = 3e9, min_shares: float = 150e6) -> pd.DataFrame:
    """Tickers that could have been worth $10B+ at some point: worth min_cap_now at the latest
    cached price, or with at least min_shares shares at some point (catches big companies that
    later shrank)."""
    dom = pd.DataFrame(json.loads((cache_dir() / "hdc_sec_tickers.json").read_text()))
    sh = shares_frames()
    us = sh.groupby("cik")["loc"].last().str.startswith("US-").rename("us")    # S&P 500: US companies only
    mx = sh.groupby("cik").shares.max().rename("max_shares")
    last = sh.groupby("cik").shares.last().rename("last_shares")
    dom = dom.join(mx, on="cik").join(last, on="cik").join(us, on="cik")
    dom = dom[dom.us.fillna(False).astype(bool)]
    g = sorted((cache_dir() / "grouped").glob("*.parquet"))[-1]
    px = pd.read_parquet(g, columns=["T", "c"]).set_index("T")["c"]
    dom["cap_now"] = dom.ticker.map(px) * dom.last_shares
    keep = (dom.cap_now >= min_cap_now) | (dom.max_shares >= min_shares)
    return dom[keep].reset_index(drop=True)


def yahoo_history(tickers: list[str], start: str = "2009-06-01", end: str = "2026-11-25",
                  batch: int = 80, progress=print) -> None:
    """Daily OHLCV (split-adjusted, not dividend-adjusted), dividends and splits per ticker, cached
    as data_cache/hdc/yahoo/<T>.parquet. Already-cached tickers are skipped, so a stopped run resumes."""
    import yfinance as yf
    root = cache_dir() / "hdc" / "yahoo"
    root.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if not (root / f"{t}.parquet").exists() and not (root / f"{t}.none").exists()]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        ysym = [t.replace(".", "-") for t in chunk]                 # Yahoo writes BRK.B as BRK-B
        for attempt in range(4):
            try:
                df = yf.download(ysym, start=start, end=end, actions=True, auto_adjust=False,
                                 group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:                                   # throttled: wait and retry
                progress(f"  yahoo retry {attempt + 1}: {str(e)[:80]}")
                time.sleep(30 * (attempt + 1))
        else:
            raise RuntimeError("Yahoo kept refusing")
        for t, y in zip(chunk, ysym):
            try:
                d = df[y].dropna(subset=["Close"]) if len(ysym) > 1 else df.dropna(subset=["Close"])
            except KeyError:
                d = pd.DataFrame()
            if d.empty:
                (root / f"{t}.none").write_text("")
                continue
            d = d.rename(columns=str.lower).rename(columns={"adj close": "adj_close", "stock splits": "splits"})
            d.index = pd.to_datetime(d.index).strftime("%Y-%m-%d")
            d.index.name = "date"
            d[["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]].to_parquet(root / f"{t}.parquet")
        progress(f"  yahoo {min(i + batch, len(todo))}/{len(todo)}")
        time.sleep(2)


def load_yahoo(t: str) -> pd.DataFrame | None:
    f = cache_dir() / "hdc" / "yahoo" / f"{t}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def real_price_factor(splits: pd.Series) -> pd.Series:
    """Multiply Yahoo's split-adjusted price on each date by this to get the price that actually
    traded that day (product of all split ratios AFTER the date)."""
    s = splits.replace(0, np.nan).fillna(1.0)
    after = s[::-1].cumprod()[::-1].shift(-1).fillna(1.0)          # splits strictly after each date
    return after


if __name__ == "__main__":
    c = candidates()
    print(f"candidates: {len(c)}")
    (cache_dir() / "hdc").mkdir(exist_ok=True)
    c.to_csv(cache_dir() / "hdc" / "candidates.csv", index=False)
    yahoo_history(sorted(c.ticker.unique()))
    print("YAHOO DONE")
