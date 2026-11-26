"""No options: own the N highest-yielding qualifying stocks, equal weight, rebalanced once a year.

Selection on the first trading day of each year uses only what was known the day before: stocks with a
qualifying ex-date (large cap, yield over 3%, see events.py) in the previous 120 days, ranked by the
indicated dividend (median of the last 4 payments x payments a year) over the previous close, still worth
$10B+ at that close. Dividends are reinvested in the same stock on the ex-date. Costs: 0.05% of every
stock trade. After tax: qualified dividends and long-term gains at 20%, REIT dividends (ordinary income)
at 37%, realized at the yearly rebalance; capital losses carry forward against later gains.

Survivorship: stocks delisted before 2026 (Walgreens, Pioneer, Reynolds American, Lorillard, ...) have
no Yahoo history and cannot be picked.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hdc.backtest import tbill, yahoo_series
from hdc.data import cache_dir
from hdc.portfolio import metrics, spy_curve
from hdc.universe import load_yahoo
from tfs.config import load_config, resolve

START, END = "2010-01-04", "2026-11-24"
CAPITAL = 1e6
STOCK_COST = 0.0005
MIN_CAP = 10e9
QUALIFIED, ORDINARY, LONG_TERM = 0.20, 0.37, 0.20
REIT_SIC = 6798


def panels(tickers: list[str], cal: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes, divs = {}, {}
    for t in tickers:
        y = load_yahoo(t)
        closes[t], divs[t] = y.close, y.dividends
    c = pd.DataFrame(closes).reindex(cal).ffill()
    d = pd.DataFrame(divs).reindex(cal).fillna(0.0)
    return c, d


def rebalance_days(cal: list[str], every: str = "year") -> list[str]:
    key = (lambda d: d[:4]) if every == "year" else (lambda d: (d[:4], (int(d[5:7]) - 1) // 3))
    out, seen = [], set()
    for d in cal:
        if key(d) not in seen:
            seen.add(key(d))
            out.append(d)
    return out


def pick(ev: pd.DataFrame, closes: pd.DataFrame, day: str, prev: str, n: int, exclude: set[str] = frozenset()):
    lo = (pd.Timestamp(day) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    if day <= ev.ex.min():           # the event list starts in Jan 2010: for the first picks use Q1 2010
        hi = (pd.Timestamp(day) + pd.Timedelta(days=90)).strftime("%Y-%m-%d")   # dividend rates (mostly declared by then)
        recent = ev[(ev.ex >= day) & (ev.ex < hi)].sort_values("ex").groupby("ticker").first()
    else:
        recent = ev[(ev.ex >= lo) & (ev.ex < day)].sort_values("ex").groupby("ticker").last()
    recent = recent[~recent.index.isin(exclude)]
    px = closes.loc[prev, recent.index]
    yld = recent["yield"] * recent.s0 / px
    cap = recent.cap * px / recent.s0
    yld = yld[(cap >= MIN_CAP) & px.notna()].sort_values(ascending=False)
    return yld.head(n)


def simulate(closes: pd.DataFrame, divs: pd.DataFrame, cal: list[str], schedule: dict[str, list[str]],
             taxed: bool, reits: set[str]) -> dict:
    shares: dict[str, float] = {}
    basis: dict[str, float] = {}
    cash, carry = CAPITAL, 0.0
    div_q = div_o = realized = 0.0
    eq, taxes = {}, {}
    for i, d in enumerate(cal):
        for t in list(shares):
            dv = float(divs.at[d, t])
            if dv > 0:
                amt = shares[t] * dv
                if t in reits:
                    div_o += amt
                else:
                    div_q += amt
                net = amt
                shares[t] += net / float(closes.at[d, t])
                basis[t] += net
        if d in schedule:
            new = schedule[d]
            value = {t: shares[t] * float(closes.at[d, t]) for t in shares}
            equity = cash + sum(value.values())
            target = equity * (1 - STOCK_COST) / len(new)
            for t in list(shares):                                   # sell first
                want = target if t in new else 0.0
                if value[t] > want:
                    frac = (value[t] - want) / value[t]
                    proceeds = value[t] - want
                    realized += proceeds - basis[t] * frac
                    cash += proceeds * (1 - STOCK_COST)
                    shares[t] *= 1 - frac
                    basis[t] *= 1 - frac
                    if t not in new:
                        del shares[t], basis[t]
            for t in new:
                have = shares.get(t, 0.0) * float(closes.at[d, t])
                if target > have:
                    buy = target - have
                    shares[t] = shares.get(t, 0.0) + buy / float(closes.at[d, t])
                    basis[t] = basis.get(t, 0.0) + buy
                    cash -= buy * (1 + STOCK_COST)
        last_of_year = i == len(cal) - 1 or cal[i + 1][:4] != d[:4]
        if last_of_year:
            gains = realized + carry
            tax = div_q * QUALIFIED + div_o * ORDINARY + max(gains, 0.0) * LONG_TERM
            carry = min(gains, 0.0)
            if taxed:
                cash -= tax
            taxes[d[:4]] = tax
            div_q = div_o = realized = 0.0
        eq[d] = cash + sum(shares[t] * float(closes.at[d, t]) for t in shares)
    return {"equity": pd.Series(eq), "taxes": taxes}


def calendar_years(eq: pd.Series) -> pd.Series:
    e = eq.copy()
    e.index = pd.to_datetime(e.index)
    ye = e.resample("YE").last()
    prev = pd.concat([pd.Series([e.iloc[0]]), ye.iloc[:-1]]).to_numpy()
    out = ye / prev - 1
    out.index = out.index.year
    return out


def main():
    cfg = load_config()
    od = resolve(cfg, "output") / "hdc"
    spy = yahoo_series("SPY")
    cal = [d for d in spy.index if START <= d <= END]
    rate = tbill()
    ev = pd.read_parquet(cache_dir() / "hdc" / "events.parquet")
    flags = pd.read_csv(cache_dir() / "hdc" / "sec_flags.csv")
    reits = set(flags[flags.sic == REIT_SIC].ticker)
    closes, divs = panels(sorted(ev.ticker.unique()), cal)

    variants = {"top5": (5, "year", set()), "top10": (10, "year", set()),
                "top5_no_reits": (5, "year", reits), "top5_quarterly": (5, "quarter", set())}
    out, curves, picks_by_year = {}, {}, {}
    for name, (n, every, excl) in variants.items():
        sched = {}
        for d in rebalance_days(cal, every):
            prev = cal[max(cal.index(d) - 1, 0)]
            p = pick(ev, closes, d, prev, n, excl)
            sched[d] = list(p.index)
            if name in ("top5", "top5_no_reits") and every == "year":
                picks_by_year.setdefault(name, {})[d[:4]] = [(t, round(float(y), 4)) for t, y in p.items()]
        for taxed in (False, True):
            if every == "quarter" and taxed:
                continue                     # quarterly turnover makes gains short-term; pre-tax only
            r = simulate(closes, divs, cal, sched, taxed, reits)
            key = f"{name}_{'after_tax' if taxed else 'pre_tax'}"
            m = metrics(r["equity"], rate)
            yrs = calendar_years(r["equity"])
            m.update(avg_year=float(yrs.mean()), median_year=float(yrs.median()), worst_year=float(yrs.min()),
                     worst_year_when=int(yrs.idxmin()), best_year=float(yrs.max()), losing_years=int((yrs < 0).sum()),
                     years={str(k): float(v) for k, v in yrs.items()})
            out[key] = m
            curves[key] = r["equity"]
    for name, tax in (("spy_pre_tax", 0.0), ("spy_after_tax", 0.20)):
        e = spy_curve(spy, cal, tax, CAPITAL)
        yrs = calendar_years(e)
        m = metrics(e, rate)
        m.update(avg_year=float(yrs.mean()), median_year=float(yrs.median()), worst_year=float(yrs.min()),
                 worst_year_when=int(yrs.idxmin()), best_year=float(yrs.max()), losing_years=int((yrs < 0).sum()),
                 years={str(k): float(v) for k, v in yrs.items()})
        out[name] = m
        curves[name] = e
    res = {"metrics": out, "picks": picks_by_year, "reits": sorted(reits & set(ev.ticker))}
    (od / "top_yield.json").write_text(json.dumps(res, indent=1, default=float))
    pd.DataFrame(curves).to_csv(od / "top_yield_curves.csv")
    return res


if __name__ == "__main__":
    r = main()
    for k, m in r["metrics"].items():
        print(f"{k:24s} cagr {m['cagr']:+.2%} avg yr {m['avg_year']:+.2%} sharpe {m['sharpe']:.2f} "
              f"maxdd {m['max_dd']:.1%} vol {m['vol']:.1%} worst {m['worst_year']:+.1%} ({m['worst_year_when']}) "
              f"losing {m['losing_years']} final {m['final'] / 1e6:.2f}M")
