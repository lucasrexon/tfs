"""Every ex-dividend date that the strategy could trade, with the facts known at entry.

Entry E = 2 trading days before the ex-date, exit X = 1 trading day after. An event qualifies if, at E:
  - company value (real price x latest SEC share count) is at least $10B ("large cap")
  - indicated yield (typical recent dividend x payments per year / price) is above 3%
  - no earnings report within 2 trading days either side of the ex-date (5-day window)
Prices are Yahoo's split-adjusted closes (dividends not taken out), so the ex-date drop is real.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hdc.data import cache_dir
from hdc.universe import load_yahoo, real_price_factor, shares_frames

MIN_CAP, MIN_YIELD, EARN_WINDOW = 10e9, 0.03, 2
MIN_DOLLAR_VOLUME = 25e6     # 60-day median $ traded per day; every real large cap clears this easily
BDC = {"ARCC"}               # business development companies: not eligible for the S&P 500


def shares_asof(sh: pd.DataFrame, dates: pd.Index) -> pd.Series:
    """Latest SEC share count dated on or before each date (forward filled)."""
    s = sh.set_index("end").shares.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


def frequency(ex_dates: list[str], i: int) -> int:
    """Payments per year, from the ex-dates in the 365 days up to and including this one."""
    d = pd.Timestamp(ex_dates[i])
    n = sum(1 for x in ex_dates[: i + 1] if (d - pd.Timestamp(x)).days < 365)
    if n >= 10:
        return 12
    return min((4, 2, 1), key=lambda f: abs(f - n)) if i >= 3 else 4


def earnings_dates(tickers: list[str], progress=print) -> dict[str, list[str]]:
    """Yahoo earnings dates (New York date of the release), cached per ticker."""
    import yfinance as yf
    root = cache_dir() / "hdc" / "earnings"
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, t in enumerate(tickers):
        f = root / f"{t}.json"
        if not f.exists():
            try:
                e = yf.Ticker(t.replace(".", "-")).get_earnings_dates(limit=100)
                days = sorted({ts.tz_convert("America/New_York").strftime("%Y-%m-%d") for ts in e.index}) if e is not None else []
            except Exception:
                days = None                                  # unknown: the event is not traded
            f.write_text(json.dumps(days))
            if i % 25 == 0:
                progress(f"  earnings {i}/{len(tickers)}")
        out[t] = json.loads(f.read_text())
    return out


def ticker_events(t: str, cik: int, sh: pd.DataFrame) -> list[dict]:
    y = load_yahoo(t)
    if y is None or y.dividends.sum() == 0:
        return []
    dates = list(y.index)
    pos = {d: i for i, d in enumerate(dates)}
    fac = real_price_factor(y.splits)
    shares = shares_asof(sh[sh.cik == cik], y.index)
    divs = y[y.dividends > 0]
    ex_dates = list(divs.index)
    out = []
    for i, (ex, D) in enumerate(divs.dividends.items()):
        j = pos[ex]
        if j < 2 or j + 1 >= len(dates):
            continue
        e_day, x_day = dates[j - 2], dates[j + 1]
        s0 = float(y.close.iloc[j - 2])
        recent = divs.dividends.iloc[max(0, i - 3): i + 1]
        freq = frequency(ex_dates, i)
        yld = float(np.median(recent)) * freq / s0
        cap = s0 * float(fac.iloc[j - 2]) * float(shares.iloc[j - 2]) if pd.notna(shares.iloc[j - 2]) else np.nan
        dvol = float((y.close * y.volume).iloc[max(0, j - 62): j - 1].median())
        out.append({"ticker": t, "cik": cik, "ex": ex, "entry": e_day, "exit": x_day, "div": float(D), "freq": freq,
                    "yield": yld, "s0": s0, "real_factor": float(fac.iloc[j - 2]), "cap": cap, "dollar_volume": dvol,
                    "special": bool(len(recent) >= 3 and D > 1.8 * float(np.median(recent.iloc[:-1])))})
    return out


PARTNERSHIP = r"\bL\.?P\.?$|\bL\.?P\.?\b|PARTNERS\b|Partners\b"


def excluded(cand: pd.DataFrame) -> set[str]:
    """Not S&P 500-style common stock: partnerships (MLPs pay distributions, not dividends), and the
    preferred shares or notes that share a company's SEC record (keep the most-traded ticker)."""
    out = set(cand[cand.name.str.contains(PARTNERSHIP, regex=True, na=False)].ticker)
    for cik, g in cand.groupby("cik"):
        if len(g) > 1:
            dv = {t: float((y.close * y.volume).tail(250).median()) if (y := load_yahoo(t)) is not None else 0.0
                  for t in g.ticker}
            keep = max(dv, key=dv.get)
            out |= set(g.ticker) - {keep}
    return out


def build(progress=print) -> pd.DataFrame:
    cand = pd.read_csv(cache_dir() / "hdc" / "candidates.csv")
    sh = shares_frames()
    rows = []
    for k, r in enumerate(cand.itertuples()):
        rows += ticker_events(r.ticker, int(r.cik), sh)
        if k % 300 == 0:
            progress(f"  events: {k}/{len(cand)} tickers")
    ev = pd.DataFrame(rows)
    ev = ev[(ev.ex >= "2010-01-01") & (ev.cap >= MIN_CAP) & (ev["yield"] > MIN_YIELD)].copy()
    flags_f = cache_dir() / "hdc" / "sec_flags.csv"             # foreign filers (20-F / 40-F only)
    foreign = set(pd.read_csv(flags_f).query("foreign").ticker) if flags_f.exists() else set()
    ev = ev[~ev.ticker.isin(excluded(cand) | foreign | BDC) & (ev.dollar_volume >= MIN_DOLLAR_VOLUME)].copy()
    earn = earnings_dates(sorted(ev.ticker.unique()), progress)
    flags = []
    for r in ev.itertuples():
        days = earn.get(r.ticker)
        if days is None:
            flags.append("unknown")
            continue
        y = load_yahoo(r.ticker)
        idx = list(y.index)
        j = idx.index(r.ex)
        lo, hi = idx[max(0, j - EARN_WINDOW)], idx[min(len(idx) - 1, j + EARN_WINDOW)]
        flags.append("near" if any(lo <= d <= hi for d in days) else "clear")
    ev["earnings"] = flags
    ev.to_parquet(cache_dir() / "hdc" / "events.parquet", index=False)
    return ev


if __name__ == "__main__":
    ev = build()
    print(ev.groupby(ev.ex.str[:4]).size().to_string())
    print(ev.earnings.value_counts().to_string())
    print("EVENTS DONE", len(ev), "tickers", ev.ticker.nunique())
