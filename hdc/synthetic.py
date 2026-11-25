"""Model-priced calls for events with no real option prices (2010 to Nov 2024).

For each event: the expiry is the first listed Friday 7-14 calendar days after entry (weekly
expiries from 2013; before that only the monthly third-Friday expiry, so many events have no
7-14 day option), strikes are the 1st and 2nd listed strikes below the entry price, and every
price comes from the American-call tree in hdc/pricing.py with
    volatility = 30-day realized volatility x VOL_MULT   (calibrated on real 2024-2026 prices)
    rate       = 13-week T-bill yield (Yahoo ^IRX)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hdc.pricing import american_call, itm_strikes

VOL_MULT = 1.15          # replaced by the calibrated value in hdc/backtest.py
WEEKLY_FROM = "2013-01-01"


def third_friday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(year=y, month=m, day=15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def listed_expiry(entry: str, days: list[str]) -> str | None:
    """First listed expiry 7-14 calendar days after entry, moved to the previous trading day if
    that Friday is a holiday."""
    e = pd.Timestamp(entry)
    if entry >= WEEKLY_FROM:
        fr = [e + pd.Timedelta(days=k) for k in range(7, 15) if (e + pd.Timedelta(days=k)).weekday() == 4]
    else:
        fr = [f for f in (third_friday(e.year, e.month), third_friday((e + pd.offsets.MonthBegin(1)).year,
                                                                    (e + pd.offsets.MonthBegin(1)).month))
              if 7 <= (f - e).days <= 14]
    if not fr:
        return None
    f = fr[0].strftime("%Y-%m-%d")
    prior = [d for d in days if d <= f]
    return prior[-1] if prior and prior[-1] > entry else None


def realized_vol(adj_close: pd.Series, day: str, n: int = 30) -> float:
    """Annualized close-to-close volatility of total-return prices over the n days to `day`."""
    s = adj_close.loc[:day].tail(n + 1)
    r = np.log(s).diff().dropna()
    return float(r.std(ddof=1) * np.sqrt(252)) if len(r) >= 10 else np.nan


def call_series(closes: pd.Series, dates: list[str], k: float, expiry: str, ex: str, div: float,
                sigma: float, rate: pd.Series) -> pd.Series:
    """Model price of the call on each date (the dividend is in the tree until the ex-date)."""
    out = {}
    T_exp = pd.Timestamp(expiry)
    for d in dates:
        if d > expiry:
            break
        s = float(closes[d])
        T = max((T_exp - pd.Timestamp(d)).days, 0) / 365.0
        r = float(np.log1p(rate.asof(pd.Timestamp(d)) if not rate.empty else 0.0))
        t_div = (pd.Timestamp(ex) - pd.Timestamp(d)).days / 365.0 if d < ex else None
        out[d] = american_call(s, k, T, r, sigma, div if t_div is not None else 0.0, t_div)
    return pd.Series(out)


def quotes(ev: dict, closes: pd.Series, adj: pd.Series, rate: pd.Series, vol_mult: float = VOL_MULT):
    """Entry prices of the 1- and 2-strike ITM calls, plus what is needed to price the chosen one later."""
    days = list(closes.index)
    expiry = listed_expiry(ev["entry"], days)
    if expiry is None:
        return None
    sig = realized_vol(adj, ev["entry"]) * vol_mult
    if not np.isfinite(sig):
        return None
    real = ev["s0"] * ev["real_factor"]
    ks = [k / ev["real_factor"] for k in itm_strikes(real, 2)]       # listed on real prices
    qs = []
    for k in ks:
        c = call_series(closes, [ev["entry"]], k, expiry, ev["ex"], ev["div"], sig, rate)
        qs.append((k, float(c.iloc[0]) if len(c) else None))
    return {"expiry": expiry, "sigma": sig, "quotes": qs}
