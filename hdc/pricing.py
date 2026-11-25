"""American call prices with a discrete cash dividend (Cox-Ross-Rubinstein tree, escrowed dividend).

Used for years without real option prices (2010-2024). The tree lets the holder exercise at any
node, so a call whose remaining time value is smaller than the dividend is worth exercising the
evening before the ex-date, which is the risk this strategy takes on. Volatility comes from
realized volatility scaled by a multiplier calibrated on real 2024-2026 option prices
(hdc/real_options.py).
"""
from __future__ import annotations

import numpy as np


def american_call(S: float, K: float, T: float, r: float, sigma: float, div: float = 0.0,
                  t_div: float | None = None, steps: int = 120) -> float:
    """Price of an American call. S spot, K strike, T years to expiry, r continuous rate,
    sigma volatility, div cash dividend paid at t_div years (ex-date) if 0 < t_div < T."""
    if T <= 0:
        return max(S - K, 0.0)
    sigma = max(sigma, 1e-4)
    has_div = div > 0 and t_div is not None and 0 <= t_div < T
    pv_div = div * np.exp(-r * t_div) if has_div else 0.0
    s_star = S - pv_div                                   # the risky part of the stock price
    if s_star <= 0:
        return max(S - K, 0.0)
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)
    j = np.arange(steps + 1)
    s_end = s_star * u ** (steps - 2 * j)                 # after the ex-date nothing is escrowed
    v = np.maximum(s_end - K, 0.0)
    for i in range(steps - 1, -1, -1):
        t = i * dt
        jj = np.arange(i + 1)
        s_node = s_star * u ** (i - 2 * jj)
        if has_div and t < t_div:                          # dividend still to come: add its value back
            s_node = s_node + div * np.exp(-r * (t_div - t))
        v = disc * (p * v[:-1] + (1 - p) * v[1:])
        v = np.maximum(v, s_node - K)                       # early exercise
    return float(v[0])


def strike_grid(S: float) -> float:
    """Typical listed strike spacing for large caps."""
    return 0.5 if S < 25 else 1.0 if S < 200 else 5.0


def itm_strikes(S: float, n: int = 2, step: float | None = None) -> list[float]:
    """The n listed strikes just below S (1 strike ITM, 2 strikes ITM, ...)."""
    st = step or strike_grid(S)
    k1 = np.floor((S - 1e-9) / st) * st
    return [round(k1 - i * st, 2) for i in range(n)]
