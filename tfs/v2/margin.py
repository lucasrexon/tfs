"""How big can a no-stop short be before a run-up liquidates the account?

Account E, short S shares at P0, so the position is f = S*P0/E of the account.
If the price reaches R*P0:  equity = E*(1 - f*(R - 1)).
The broker liquidates when equity < the maintenance requirement at that price
(FINRA 4210: under $5 the larger of $2.50/share or 100% of value; from $5 the
larger of $5/share or 30% of value; 'house5' is a stricter $5/share under $5).
"""
from __future__ import annotations

from tfs.risk import margin_requirement


def survives(p0: float, f: float, r: float, rule: str = "finra") -> bool:
    shares = f / p0                                   # per $1 of account
    equity = 1 - f * (r - 1)
    return equity >= margin_requirement(r * p0, shares, rule)


def f_max(p0: float, r_star: float, rule: str = "finra") -> float:
    """Largest position (fraction of the account) that survives a run-up to r_star x entry."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if survives(p0, mid, r_star, rule) else (lo, mid)
    return lo


def max_survivable_r(p0: float, f: float, rule: str = "finra") -> float:
    lo, hi = 1.0, 1000.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if survives(p0, f, mid, rule) else (lo, mid)
    return lo
