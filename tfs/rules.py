"""Entry triggers, shared by the backtest AND the live bot.

Every function looks only at bars up to the bar it is judging, so it cannot
see the future. `find_entry` returns the index of the bar whose CLOSE fired the
signal; the trade fills at the OPEN of the next bar (the 'open' rule fills at
the first regular-session bar's open). The live bot calls the same function on
the bars it has so far and acts only when the signal lands on the newest bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

INTRADAY_ENTRIES = ("open", "first_red_5m", "vwap_loss", "pmh_fail")
ALL_ENTRIES = INTRADAY_ENTRIES + ("next_open",)


@dataclass
class Signal:
    rule: str
    signal_idx: int        # bar that fired (-1 for 'open')
    fill_idx: int          # bar whose open is the fill


def session(bars: pd.DataFrame, start="09:30", end="16:00") -> pd.DataFrame:
    if bars.empty:
        return bars
    day = bars.index[0].normalize()
    return bars[(bars.index >= day + pd.Timedelta(start + ":00")) &
                (bars.index < day + pd.Timedelta(end + ":00"))]


def running_vwap(reg: pd.DataFrame) -> np.ndarray:
    pv = (reg["vw"] * reg["v"]).cumsum().to_numpy()
    v = reg["v"].cumsum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(v > 0, pv / np.where(v > 0, v, 1), reg["c"].to_numpy())
    return out


def find_entry(rule: str, reg: pd.DataFrame, pmh: float, cfg_rules: dict) -> Signal | None:
    """reg = regular-session 1-minute bars (09:30 onward) known so far."""
    if reg.empty:
        return None
    day = reg.index[0].normalize()
    deadline = day + pd.Timedelta(cfg_rules["entry_deadline"] + ":00")
    t = reg.index
    if rule == "open":
        return Signal(rule, -1, 0) if t[0] < deadline else None

    o, h, low, c = (reg[k].to_numpy() for k in ("o", "h", "l", "c"))
    n = len(reg)

    if rule == "vwap_loss":
        vw = running_vwap(reg)
        for i in range(n):
            if t[i] >= deadline:
                return None
            if c[i] < vw[i] * (1 - 1e-9):
                return Signal(rule, i, i + 1)
        return None

    if rule == "first_red_5m":
        buckets = t.floor("5min")
        i = 0
        while i < n:
            b = buckets[i]
            j = i
            while j + 1 < n and buckets[j + 1] == b:
                j += 1
            # bucket complete only once a later bar exists (or its last minute is present)
            complete = (j + 1 < n) or (t[j] >= b + pd.Timedelta("4min"))
            if not complete:
                return None
            if t[j] >= deadline:
                return None
            if c[j] < o[i]:
                return Signal(rule, j, j + 1)
            i = j + 1
        return None

    if rule == "pmh_fail":
        touch = 1 - cfg_rules["pmh_touch_pct"]
        touch_low = None
        for i in range(n):
            if t[i] >= deadline:
                return None
            if h[i] >= pmh * touch:
                touch_low = low[i]            # latest touch bar's low
                continue
            if touch_low is not None and c[i] < touch_low:
                return Signal(rule, i, i + 1)
        return None

    raise ValueError(f"unknown intraday entry rule {rule}")


def stop_price(stop_rule: str, ref_high: float, cfg_rules: dict) -> float | None:
    if stop_rule == "none":
        return None
    if stop_rule == "pmh_plus_5":
        return ref_high * (1 + cfg_rules["pmh_stop_buffer"])
    raise ValueError(stop_rule)


def count_halts(reg: pd.DataFrame, upto_idx: int | None = None, gap_min: int = 5,
                active_bars: int = 8) -> int:
    """Suspected LULD halts: a gap of >= 5 minutes with no bars after a stretch of
    near-continuous trading (8+ bars in the prior 10 minutes). Only bars up to
    `upto_idx` are used, so it is safe to call at entry time."""
    if reg.empty:
        return 0
    t = reg.index if upto_idx is None else reg.index[: upto_idx + 1]
    if len(t) < 2:
        return 0
    mins = ((t - t[0]).total_seconds() // 60).astype(int).to_numpy()
    halts = 0
    for k in range(1, len(mins)):
        if mins[k] - mins[k - 1] >= gap_min + 1:
            recent = mins[(mins > mins[k - 1] - 10) & (mins <= mins[k - 1])]
            if len(recent) >= active_bars:
                halts += 1
    return halts
