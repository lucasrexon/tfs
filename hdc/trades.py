"""Net profit of one hedged dividend capture trade, per share, split into its parts.

    net = stock change + call premium kept - call buy-back + dividend - costs

Paths (all per share; the account trades 100 shares per contract):
  kept      not assigned: sell the stock and buy back the call at the close the day after the ex-date
  expiry    not assigned and the call is deep in the money the day after the ex-date (time value
            <= DEEP_ITM): leave it; at expiry the stock is called away at the strike (or, if it fell
            below the strike, the call expires and the stock is sold)
  assigned  the call holder exercises the evening before the ex-date: the stock is delivered at the
            strike, no dividend; profit is the call's time value at entry minus costs
  hold61    Scenario B: buy back the call the day after the ex-date but keep the stock until 62
            calendar days after entry (collecting any further dividends), then sell
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEEP_ITM = 0.03          # $/share of time value left at exit below which the call is left to expire
HOLD_DAYS = 62           # Scenario B: calendar days from entry (more than 61)


@dataclass
class Costs:
    stock_pct: float = 0.0005      # bid-ask cost per stock trade (each buy and each sell), share of price
    opt_comm: float = 0.65         # $ per option contract per trade
    opt_half_spread: float = 0.0   # $ per share lost to the option bid-ask on each option trade

    def per_share_comm(self) -> float:
        return self.opt_comm / 100.0


def select_call(s0: float, div: float, quotes: list[tuple[float, float | None]], rule: str = "below"):
    """quotes: [(strike, call price at entry)] ordered 1 strike ITM, then 2 strikes ITM.
    rule 'below': time value strictly less than the dividend (the spec); 'above': at least the dividend.
    A price below intrinsic (a stale last trade) is treated as selling at intrinsic."""
    for k, c in quotes:
        if c is None or not (k < s0) or c <= 0:
            continue
        c = max(c, s0 - k)
        e = c - (s0 - k)
        if (rule == "below" and e < div) or (rule == "above" and e >= div):
            return k, c, e
    return None


def leg(stock: float, option: float, dividend: float, cost: float, exit_day: str, kind: str,
        marks: dict, extra_divs: list | None = None, premium: float = 0.0, open_cost: float = 0.0) -> dict:
    """premium: cash received for the call at entry; open_cost: costs paid at entry (per share)."""
    return {"stock": stock, "option": option, "dividend": dividend, "cost": cost, "kind": kind,
            "net": stock + option + dividend - cost, "exit_day": exit_day, "marks": marks,
            "extra_divs": extra_divs or [], "premium": premium, "open_cost": open_cost}


def outcomes(ev: dict, k: float, c0: float, closes: pd.Series, calls: pd.Series, expiry: str,
             costs: Costs, divs: pd.Series | None = None) -> dict:
    """Every path for one trade. closes: stock closes by date from entry onward; calls: call price
    by date (entry .. expiry); divs: the stock's dividends by ex-date (for Scenario B)."""
    E, ex, X = ev["entry"], ev["ex"], ev["exit"]
    D, s0 = ev["div"], float(closes[E])
    days = list(closes.index)
    i_ex = days.index(ex)
    exm1 = days[i_ex - 1]
    hs, cm, sp = costs.opt_half_spread, costs.per_share_comm(), costs.stock_pct
    open_cost = s0 * sp + cm + hs                               # buy stock, sell call

    def mark(d):                                                # position value per share at a close
        return float(closes[d]) - float(calls.get(d, max(float(closes[d]) - k, 0.0)))

    out = {}
    # assigned the evening before the ex-date (only if the call is in the money then)
    out["assigned"] = leg(k - s0, c0, 0.0, open_cost, ex, "assigned",
                          {E: mark(E), exm1: mark(exm1)}, premium=c0, open_cost=open_cost) if float(closes[exm1]) > k else None
    # not assigned: close the day after the ex-date ...
    s_x, c_x = float(closes[X]), float(calls.get(X, max(float(closes[X]) - k, 0.0)))
    m = {d: mark(d) for d in days[days.index(E): days.index(X) + 1]}
    kept = leg(s_x - s0, c0 - c_x, D, open_cost + s_x * sp + cm + hs, X, "kept", m, premium=c0, open_cost=open_cost)
    # ... or, when deep in the money, leave it to be assigned at expiry
    exp_day = max([d for d in days if d <= expiry], default=None)
    if c_x - (s_x - k) <= DEEP_ITM and exp_day and exp_day > X:
        s_t = float(closes[exp_day])
        m2 = {d: mark(d) for d in days[days.index(E): days.index(exp_day) + 1]}
        m2[exp_day] = (k if s_t >= k else s_t)                  # cash left at expiry, per share
        if s_t >= k:
            kept = leg(k - s0, c0, D, open_cost, exp_day, "expiry_assigned", m2, premium=c0, open_cost=open_cost)
        else:
            kept = leg(s_t - s0, c0, D, open_cost + s_t * sp, exp_day, "expiry_worthless", m2, premium=c0, open_cost=open_cost)
    out["kept"] = kept
    # the same dates with no call: plain dividend capture, to show what the hedge does
    out["unhedged"] = leg(s_x - s0, 0.0, D, s0 * sp + s_x * sp, X, "unhedged",
                          {d: float(closes[d]) for d in days[days.index(E): days.index(X) + 1]}, premium=0.0, open_cost=s0 * sp)
    # Scenario B: buy back the call the day after the ex-date, keep the stock HOLD_DAYS from entry
    end = (pd.Timestamp(E) + pd.Timedelta(days=HOLD_DAYS)).strftime("%Y-%m-%d")
    later = [d for d in days if d >= end]
    if later:
        s_end = later[0]
        s_e = float(closes[s_end])
        extra = [] if divs is None else [(d, float(v)) for d, v in divs.items() if ex < d <= s_end]
        m3 = {d: (mark(d) if d <= X else float(closes[d])) for d in days[days.index(E): days.index(s_end) + 1]}
        out["hold61"] = leg(s_e - s0, c0 - c_x, D + sum(v for _, v in extra),
                            open_cost + cm + hs + s_e * sp, s_end, "hold61", m3, extra, premium=c0, open_cost=open_cost)
    else:
        out["hold61"] = None
    return out


def dividend_qualifies(ex: str, buy: str, hedge_start: str, hedge_end: str, sell: str) -> bool:
    """US qualified-dividend test (IRC 1(h)(11), 246(c)): the stock must be held more than 60 days
    in the 121-day window starting 60 days before the ex-date, NOT counting days on which an
    in-the-money (or non-qualified) call was written against it (hedge_start..hedge_end).
    The day of purchase is not counted; the day of sale is."""
    lo = pd.Timestamp(ex) - pd.Timedelta(days=60)
    hi = pd.Timestamp(ex) + pd.Timedelta(days=60)
    held = pd.date_range(max(pd.Timestamp(buy) + pd.Timedelta(days=1), lo), min(pd.Timestamp(sell), hi))
    hedged = (held >= pd.Timestamp(hedge_start)) & (held <= pd.Timestamp(hedge_end))
    return int((~hedged).sum()) > 60
