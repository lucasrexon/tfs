"""Account simulation, taxes and metrics.

Sizing: each trade buys stock worth FRACTION of equity (whole contracts of 100 shares), at most
MAX_OPEN trades at once; when more events start on the same day than there is room for, the ones with
the biggest dividend relative to price go first. Idle cash earns the T-bill rate (taxed as interest).

Taxes are paid from the account on the last trading day of each year:
  A  short-term   dividends ordinary 37%, capital gains 37% (the spec)
  B  as specified dividends 20%, capital gains 20% (the spec's assumption)
  B  actual law   each dividend tested with the 61-day rule (hedged days do not count); capital
                  gains short-term (held under a year; written options are always short-term)
Capital losses offset capital gains, then up to $3,000 a year of ordinary income; the rest carries
forward. Interest on cash is ordinary income in every scenario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hdc.trades import Costs, dividend_qualifies

FRACTION, MAX_OPEN, CAPITAL = 0.10, 10, 1_000_000.0
RATES = {"pre_tax": None,
         "A_short_term": {"ord": 0.37, "qual": 0.37, "cap": 0.37},
         "B_as_specified": {"ord": 0.37, "qual": 0.20, "cap": 0.20, "all_div_qualified": True},
         "B_actual_law": {"ord": 0.37, "qual": 0.20, "cap": 0.37}}
# "_offset" versions: the investor has enough other capital gains that this account's capital losses
# save tax at the same rate in the year they happen (the best case). The plain versions are a
# stand-alone account: losses offset gains, then $3,000 of ordinary income, then carry forward.
RATES |= {k + "_offset": v | {"offset": True} for k, v in list(RATES.items()) if v}


def year_tax(r: dict, ordinary: float, qualified: float, capital: float, carry: float) -> tuple[float, float]:
    if r.get("offset"):
        return max(ordinary, 0) * r["ord"] + max(qualified, 0) * r["qual"] + capital * r["cap"], 0.0
    cap = capital + carry
    if cap < 0:
        off = min(3000.0, -cap, max(ordinary, 0.0))
        ordinary -= off
        return max(ordinary, 0) * r["ord"] + max(qualified, 0) * r["qual"], cap + off
    return max(ordinary, 0) * r["ord"] + max(qualified, 0) * r["qual"] + cap * r["cap"], 0.0


def simulate(trades: list[dict], calendar: list[str], rate: pd.Series, costs: Costs, path: str = "kept",
             assign: np.ndarray | None = None, tax: str = "pre_tax", capital: float = CAPITAL,
             fraction: float = FRACTION, max_open: int = MAX_OPEN, sweep: bool = False) -> dict:
    """path 'kept' (Scenario A: exit the day after the ex-date) or 'hold61' (Scenario B).
    assign[i] True = trade i's call is exercised the evening before the ex-date (if in the money)."""
    r = RATES[tax]
    by_entry: dict[str, list[int]] = {}
    for i, t in enumerate(trades):
        by_entry.setdefault(t["entry"], []).append(i)
    cash, open_pos, log, eq = capital, [], [], {}
    year_inc = {"ordinary": 0.0, "qualified": 0.0, "capital": 0.0}
    carry, taxes, sweeps, base = 0.0, {}, {}, capital
    prev_eq = capital
    rf = rate.reindex(pd.to_datetime(calendar), method="ffill").fillna(0.0).to_numpy()
    for di, d in enumerate(calendar):
        interest = cash * rf[di] / 252 if cash > 0 else 0.0
        cash += interest
        year_inc["ordinary"] += interest
        # exits
        still = []
        for p in open_pos:
            if p["exit_day"] <= d:
                cash += p["n"] * (p["s0"] - p["leg"]["premium"] + p["leg"]["net"] + p["leg"]["open_cost"])
                t, leg = trades[p["i"]], p["leg"]
                divs = [(t["ex"], t["div"])] if leg["dividend"] else []
                divs += leg.get("extra_divs", [])
                for exd, amt in divs:
                    q = r is not None and (r.get("all_div_qualified") or dividend_qualifies(
                        exd, t["entry"], t["entry"], t["exit"], leg["exit_day"]))
                    year_inc["qualified" if q else "ordinary"] += p["n"] * amt
                year_inc["capital"] += p["n"] * (leg["stock"] + leg["option"] - leg["cost"])
                log.append({"i": p["i"], "entry": t["entry"], "exit": leg["exit_day"], "n": p["n"], "kind": leg["kind"],
                            "net": p["n"] * leg["net"], "ret": leg["net"] / t["s0"]})
            else:
                still.append(p)
        open_pos = still
        # entries, biggest dividend relative to price first
        for i in sorted(by_entry.get(d, []), key=lambda i: -trades[i]["div"] / trades[i]["s0"]):
            if len(open_pos) >= max_open:
                break
            t = trades[i]
            use_assigned = assign is not None and assign[i] and t["paths"]["assigned"] is not None
            leg = t["paths"]["assigned"] if use_assigned else t["paths"][path]
            if leg is None:
                continue
            n = int(prev_eq * fraction // (100 * t["s0"])) * 100
            need = n * (t["s0"] - leg["premium"] + leg["open_cost"])
            if n <= 0 or need > cash:
                continue
            cash -= need
            open_pos.append({"i": i, "n": n, "s0": t["s0"], "c0": leg["premium"], "leg": leg,
                             "exit_day": leg["exit_day"], "ex": t["ex"]})
        # mark to market
        val = cash
        for p in open_pos:
            leg, t = p["leg"], trades[p["i"]]
            mk = leg["marks"]
            m = mk.get(d, mk[max(k for k in mk if k <= d)] if any(k <= d for k in mk) else p["s0"] - p["c0"])
            extra = (t["div"] if d >= t["ex"] and leg["dividend"] else 0.0) + sum(a for x, a in leg.get("extra_divs", []) if x <= d)
            val += p["n"] * (m + extra)
        # year end: taxes, then (hybrid) sweep the year's gain into the long-term portfolio
        last_of_year = di == len(calendar) - 1 or calendar[di + 1][:4] != d[:4]
        if last_of_year:
            if r is not None:
                tx, carry = year_tax(r, year_inc["ordinary"], year_inc["qualified"], year_inc["capital"], carry)
                cash -= tx
                val -= tx
                taxes[d[:4]] = tx
            year_inc = {"ordinary": 0.0, "qualified": 0.0, "capital": 0.0}
            if sweep and val > base:
                out = min(val - base, max(cash, 0.0))
                cash -= out
                val -= out
                sweeps[d] = out
        eq[d] = val
        prev_eq = val
    return {"equity": pd.Series(eq), "log": pd.DataFrame(log), "taxes": pd.Series(taxes, dtype=float),
            "sweeps": pd.Series(sweeps, dtype=float)}


def spy_curve(spy: pd.DataFrame, calendar: list[str], div_tax: float = 0.0, capital: float = 1.0,
              buys: pd.Series | None = None) -> pd.Series:
    """Value of a SPY holding with dividends reinvested after `div_tax`; `buys` adds cash on dates."""
    s = spy.reindex(calendar)
    units, out = capital / float(s.close.iloc[0]), {}
    for d, row in s.iterrows():
        if row.dividends and row.dividends > 0 and d != calendar[0]:
            units += units * row.dividends * (1 - div_tax) / row.close
        if buys is not None and d in buys.index:
            units += float(buys[d]) / row.close
        out[d] = units * row.close
    return pd.Series(out)


def metrics(eq: pd.Series, rate: pd.Series) -> dict:
    eq = eq.dropna()
    yrs = (pd.Timestamp(eq.index[-1]) - pd.Timestamp(eq.index[0])).days / 365.25
    r = eq.pct_change().dropna()
    rf = rate.reindex(pd.to_datetime(r.index), method="ffill").fillna(0.0).to_numpy() / 252
    ex = r.to_numpy() - rf
    dd = eq / eq.cummax() - 1
    return {"cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1), "vol": float(r.std() * np.sqrt(252)),
            "sharpe": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else np.nan,
            "max_dd": float(dd.min()), "max_dd_date": str(dd.idxmin()), "final": float(eq.iloc[-1])}
