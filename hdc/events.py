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


# Companies that re-registered under a new SEC number (holding-company reorganizations, mergers): their
# earlier share counts sit under the old number. Found by matching names in the SEC frames.
PREDECESSOR = {2115436: (34088,),      # ExxonMobil Holdings (2026) <- Exxon Mobil Corp
               2012383: (1364742,),    # BlackRock (2024) <- BlackRock Inc.
               1701605: (808362,),     # Baker Hughes Co <- Baker Hughes Inc / Holdings LLC
               1666700: (30554,),      # DuPont de Nemours <- E. I. du Pont de Nemours
               2041610: (813828,)}     # Paramount Skydance <- CBS / ViacomCBS / Paramount Global
BACKFILL_DAYS = 3 * 365      # SEC share data that starts late: use the first count for up to 3 years before it
NO_BACKFILL = {1711269}      # Evergy: formed by a 2018 merger that doubled the share count


def company_shares(sh: pd.DataFrame, cik: int) -> pd.DataFrame:
    """A company's SEC share counts, with any predecessor's counts before its own begin."""
    own = sh[sh.cik == cik]
    old = sh[sh.cik.isin(PREDECESSOR.get(cik, ()))]
    if len(own):
        old = old[old.end < own.end.min()]
    return pd.concat([old, own])


def shares_asof(sh: pd.DataFrame, dates: pd.Index, backfill: bool = True) -> pd.Series:
    """Latest SEC share count dated on or before each date (forward filled); dates up to BACKFILL_DAYS
    before the first count get the first count."""
    s = sh.set_index("end").shares.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    out = s.reindex(s.index.union(dates)).ffill().reindex(dates)
    if backfill and len(s):
        first = pd.Timestamp(s.index[0])
        early = (pd.to_datetime(dates) < first) & (pd.to_datetime(dates) >= first - pd.Timedelta(days=BACKFILL_DAYS))
        out[early] = float(s.iloc[0])
    return out


SPECIAL_MULT = 1.6           # a payment this many times the typical one is a special dividend
FREQ_BANDS = ((20, 45, 12), (60, 120, 4), (150, 220, 2), (300, 430, 1))   # median days between payments


def regular_dividend(dates: list, amounts: list[float], i: int) -> tuple[int | None, float | None, bool]:
    """(payments a year, regular payment, is payment i a special), using only the payments up to i.

    Specials, in date order: a payment over SPECIAL_MULT x the median of the last 4 regular ones (two alike
    in a row are a raise instead). Payments off the schedule (under half the usual gap after the previous
    one) count as a quarter paid in two parts if they add up to about one regular payment, else as specials.
    The schedule must be steady (every recent gap 0.5-1.6x the median, at least 3 regular payments in the
    last 850 days) or the result is None. The indicated dividend (latest regular x payments a year) is
    capped at 1.35x the regular payments of the past year, which catches schedules misread from the gaps."""
    special, reg_amts = [], []
    for k in range(i + 1):
        a = float(amounts[k])
        if len(reg_amts) >= 2 and a > SPECIAL_MULT * float(np.median(reg_amts[-4:])):
            if special and special[-1] and abs(a - float(amounts[k - 1])) <= 0.2 * a:
                special[-1] = False                               # a raise, not two specials
                reg_amts += [float(amounts[k - 1]), a]
                special.append(False)
            else:
                special.append(True)
            continue
        reg_amts.append(a)
        special.append(False)
    d_i = pd.Timestamp(dates[i])
    day = {k: pd.Timestamp(dates[k]).value / 864e11 for k in range(i + 1)}
    w = [k for k in range(i + 1) if (d_i - pd.Timestamp(dates[k])).days <= 850 and not special[k]]
    if len(w) < 3:
        return None, None, special[i]
    gm = float(np.median(np.diff([day[k] for k in w])))
    level = float(np.median([amounts[k] for k in w[-4:]]))
    groups, off = [], set()
    for k in w:
        if groups and day[k] - groups[-1][0] < 0.5 * gm:
            if groups[-1][1] + amounts[k] <= 1.3 * level:         # a quarter paid in two parts
                groups[-1] = [day[k], groups[-1][1] + float(amounts[k])]
            else:
                off.add(k)                                       # an extra payment: a special
            continue
        groups.append([day[k], float(amounts[k])])
    is_special = bool(special[i] or i in off)
    if len(groups) < 3:
        return None, None, is_special
    tail = groups[-5:]
    days = [x for x, _ in tail]
    gaps = np.diff(days)
    # Yahoo merges a special paid with a regular dividend into one payment, which then looks like a gap of
    # two periods: count such a gap as the periods it spans when a special falls inside it
    sp_days = [day[k] for k in range(i + 1) if special[k] or k in off]
    g_min = float(gaps.min())
    gaps = np.array([gp / max(1, round(gp / g_min)) if any(a < x < b for x in sp_days) else gp
                     for gp, a, b in zip(gaps, days[:-1], days[1:])])
    g = float(np.median(gaps))
    freq = next((f for lo, hi, f in FREQ_BANDS if lo <= g < hi), None)
    if freq is None or (gaps < 0.5 * g).any() or (gaps > 1.6 * g).any():
        return None, None, is_special
    ttm = sum(float(amounts[k]) for k in w if k not in off and (d_i - pd.Timestamp(dates[k])).days < 365)
    annual = min(tail[-1][1] * freq, 1.35 * ttm)
    return freq, annual / freq, is_special


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
    shares = shares_asof(company_shares(sh, cik), y.index, backfill=cik not in NO_BACKFILL)
    divs = y[y.dividends > 0]
    ex_dates = list(divs.index)
    amounts = [float(v) for v in divs.dividends]
    out = []
    for i, (ex, D) in enumerate(divs.dividends.items()):
        j = pos[ex]
        if j < 2 or j + 1 >= len(dates):
            continue
        e_day, x_day = dates[j - 2], dates[j + 1]
        s0 = float(y.close.iloc[j - 2])
        freq, regular, special = regular_dividend(ex_dates, amounts, i)
        if freq is None:
            continue                                             # no steady regular dividend to speak of
        yld = regular * freq / s0
        cap = s0 * float(fac.iloc[j - 2]) * float(shares.iloc[j - 2]) if pd.notna(shares.iloc[j - 2]) else np.nan
        dvol = float((y.close * y.volume).iloc[max(0, j - 62): j - 1].median())
        out.append({"ticker": t, "cik": cik, "ex": ex, "entry": e_day, "exit": x_day, "div": float(D), "freq": freq,
                    "yield": yld, "s0": s0, "real_factor": float(fac.iloc[j - 2]), "cap": cap, "dollar_volume": dvol,
                    "special": special})
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
    ev = ev[~ev.special].copy()      # listed options are adjusted for special dividends: nothing to capture
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
