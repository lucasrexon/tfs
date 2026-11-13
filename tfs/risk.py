"""Position sizing and margin, shared by the portfolio simulation and the live bot."""
from __future__ import annotations

import math


def margin_requirement(price: float, shares: int, rule: str = "finra") -> float:
    """Maintenance margin for a SHORT (FINRA Rule 4210, or a strict $5/share house rule)."""
    value = price * shares
    if rule == "full":                      # brokers often demand 100% on a squeezing small cap, at any price
        return max(2.5 * shares, value)
    if rule == "house5" and price < 5:
        return max(5.0 * shares, value)
    if price < 5:
        return max(2.5 * shares, value)
    return max(5.0 * shares, 0.30 * value)


def size_short(equity: float, entry: float, stop: float | None, risk: dict,
               open_exposure: float = 0.0, open_margin: float = 0.0) -> tuple[int, str]:
    """Shares to short, and the binding constraint. 0 shares means no trade."""
    if entry <= 0:
        return 0, "bad_price"
    if stop is not None:
        dist = stop - entry
        if dist <= 0:
            return 0, "entry_above_stop"
        shares = math.floor(equity * risk["risk_per_trade"] / dist)
        why = "risk"
    else:
        shares = math.floor(equity * risk["max_position_pct_no_stop"] / entry)
        why = "no_stop_cap"
    room = equity * risk["max_short_exposure"] - open_exposure
    if shares * entry > room:
        shares, why = max(0, math.floor(room / entry)), "exposure_cap"
    rule = risk.get("margin_rule", "finra")
    while shares > 0 and open_margin + margin_requirement(entry, shares, rule) > equity:
        shares = math.floor(shares * 0.9)
        why = "margin"
    return int(shares), why
