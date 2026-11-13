"""No-stop short simulation: every entry x profit target x time limit, per gapper.

A trade ends one of two ways: the price falls to the profit target (cover there),
or the time limit runs out (cover at that close). There is no stop-loss, so for
every trade we also record R, the highest price reached before the exit divided
by the entry price. R is what decides whether the account survives.

Conservative choices:
  * in the minute (or day) the target is hit, that bar's high still counts toward R
  * a sell limit fills at the limit, or at the bar's open if it opened above it
  * slippage, commission, locate fee, and a borrow fee for every night held
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfs.backtest import nights_between
from tfs.rules import find_entry, session

TARGETS = (0.10, 0.15, 0.20, 0.25, 0.30)
HORIZONS = (0, 1, 3, 5, 10)                       # 0 = same day; k = close of the k-th day after entry
PMH_OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.20)
LIMIT_UNTIL = "15:00"


def _entries(g: dict, bars: pd.DataFrame, cfg: dict) -> list[dict]:
    """Each entry: name, fill price (adjusted), index into `seq` where the trade starts."""
    reg = session(bars)
    if reg.empty:
        return []
    day = reg.index[0].normalize()
    out = []
    o, h = reg["o"].to_numpy(), reg["h"].to_numpy()
    # short at the scan time, before the open
    pre = bars[(bars.index >= day + pd.Timedelta("9h25m")) & (bars.index < day + pd.Timedelta("16h"))]
    if not pre.empty and pre.index[0] < day + pd.Timedelta("9h30m"):
        out.append({"entry": "pm_0925", "px": float(pre["o"].iloc[0]), "frame": pre, "i0": 0})
    out.append({"entry": "open", "px": float(o[0]), "frame": reg, "i0": 0})
    until = int(np.searchsorted(reg.index, day + pd.Timedelta(LIMIT_UNTIL + ":00")))
    for x in PMH_OFFSETS:
        lim = g["pm_high"] * (1 + x)
        hit = np.flatnonzero(h[:until] >= lim)
        if len(hit):
            i = int(hit[0])
            out.append({"entry": f"pmh_{x:+.0%}", "px": float(max(lim, o[i])), "frame": reg, "i0": i})
    for rule in ("first_red_5m", "vwap_loss"):
        s = find_entry(rule, reg, g["pm_high"], cfg["rules"])
        if s is not None and s.fill_idx < len(reg):
            out.append({"entry": rule, "px": float(o[s.fill_idx]), "frame": reg, "i0": s.fill_idx})
    return out


def simulate(g: dict, bars: pd.DataFrame, cfg: dict) -> list[dict]:
    cc = cfg["costs"]
    nxt = [dict(d) for d in (g.get("next10") if g.get("next10") is not None else [])]
    factor = g["factor"]
    rows = []
    reg = session(bars)
    if reg.empty:
        return rows
    day = reg.index[0].normalize()
    eod_cut = day + pd.Timedelta(cfg["rules"]["eod_exit"] + ":00")
    dO = np.array([d["o"] for d in nxt]); dH = np.array([d["h"] for d in nxt])
    dL = np.array([d["l"] for d in nxt]); dC = np.array([d["c"] for d in nxt])
    dates = [d["date"] for d in nxt]

    cases = _entries(g, bars, cfg)
    if nxt:
        cases.append({"entry": "next_open", "px": float(dO[0]), "frame": None, "i0": 0})

    for c in cases:
        px = c["px"]
        if c["frame"] is not None:
            f = c["frame"].iloc[c["i0"]:]
            mo, mh, ml, mc = (f[k].to_numpy() for k in ("o", "h", "l", "c"))
            mo = mo.copy(); mo[0] = px                         # the fill bar starts at our fill
            n_min = len(f)
            eod_i = int(np.searchsorted(f.index, eod_cut))     # first bar at/after 15:55
            O = np.concatenate([mo, dO]); H = np.concatenate([mh, dH]); L = np.concatenate([ml, dL])
            C = np.concatenate([mc, dC])
            entry_date, entry_time = g["date"], f.index[0].strftime("%H:%M")
            step_date = [g["date"]] * n_min + dates
        else:
            n_min, eod_i = 0, None
            O, H, L, C = dO, dH, dL, dC
            entry_date, entry_time = dates[0], "09:30"
            step_date = dates
        cmax = np.maximum.accumulate(H)
        for hz in HORIZONS:
            if c["frame"] is not None:
                if hz == 0:
                    last = min(eod_i, n_min) - 1               # last full bar before 15:55
                    end_px = float(O[eod_i]) if eod_i < n_min else float(C[n_min - 1])
                    end_i = max(last, 0)
                else:
                    end_i = n_min + hz - 1
                    end_px = None
            else:
                end_i, end_px = hz, None                       # next_open: horizon counts from the entry day
            partial = False
            if end_i >= len(O):
                if len(O) == 0:
                    continue
                end_i, partial = len(O) - 1, True
            if end_px is None:
                end_px = float(C[end_i])
            for tg in TARGETS:
                tpx = px * (1 - tg)
                hit = np.flatnonzero(L[: end_i + 1] <= tpx)
                if len(hit):
                    i = int(hit[0])
                    exit_px = float(O[i]) if (O[i] <= tpx and i > 0) else tpx
                    r_idx, is_hit = i, True
                else:
                    i, exit_px, r_idx, is_hit = end_i, end_px, end_i, False
                R = float(max(cmax[r_idx], px) / px)
                E, X = px * factor, exit_px * factor
                nights = nights_between(entry_date, step_date[i])
                se = max(cc["min_slippage"], cc["slippage_pct"] * E)
                sx = max(cc["min_slippage"], cc["slippage_pct"] * X)
                fees = 2 * cc["commission_per_share"] + cc["locate_fee_per_share"] + E * cc["borrow_fee_annual"] * nights / 360
                rows.append({"date": g["date"], "ticker": g["ticker"], "entry": c["entry"], "target": tg, "horizon": hz,
                             "hit": is_hit, "ret": ((E - se) - (X + sx) - fees) / E, "R": R, "entry_px": E,
                             "entry_time": entry_time, "exit_date": step_date[i], "nights": nights,
                             "partial": partial and not is_hit})
    return rows


def squeeze_multiple(g: dict, bars: pd.DataFrame, days: int = 5) -> float | None:
    """Highest price from the 09:25 scan through `days` trading days later, over the 09:25 price."""
    if bars.empty or not g.get("pm_last"):
        return None
    day = bars.index[0].normalize()
    post = bars[bars.index >= day + pd.Timedelta("9h25m")]
    hi = float(post["h"].max()) if not post.empty else g["pm_last"]
    nxt = g.get("next10") if g.get("next10") is not None else []
    for d in list(nxt)[:days]:
        hi = max(hi, d["h"])
    return hi / g["pm_last"]
