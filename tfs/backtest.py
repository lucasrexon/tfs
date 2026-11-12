"""Simulate every rule on every gapper, with the honesty layer built in.

Honesty layer:
  * fills at the NEXT bar's open after a signal, never at the signal bar's price
  * stops fill at max(stop, bar open) plus 3% extra slippage (gaps and halts jump stops)
  * multi-day exits check the overnight gap: an open above the stop fills at the open
  * slippage, commission, locate fee and borrow fee per calendar day held overnight
  * short-sale restriction (Rule 201) active at entry -> worse entry fill
  * Do Not Touch filters use only what was known at the entry minute
  * the 'likely hard-to-borrow' flag lets the report show results with those excluded
"""
from __future__ import annotations

from datetime import date as Date

import numpy as np
import pandas as pd

from tfs.rules import INTRADAY_ENTRIES, count_halts, find_entry, running_vwap, session, stop_price


# ----------------------------------------------------------------- features
def gapper_features(g: dict, bars: pd.DataFrame, cfg: dict) -> dict:
    reg = session(bars)
    f = {"gap_bucket": "50-100%" if g["pm_gap"] < 1 else ("100-200%" if g["pm_gap"] < 2 else "200%+")}
    pmh = g["pm_high"]
    if reg.empty:
        f["has_session"] = False
        return f
    f["has_session"] = True
    vw = running_vwap(reg)
    o0 = float(reg["o"].iloc[0])
    f["open"] = o0
    f["open_below_pmh"] = o0 < pmh
    f["open_gap"] = o0 / g["prev_close"] - 1
    first5 = reg[reg.index < reg.index[0].normalize() + pd.Timedelta("9h35m")]
    if not first5.empty:
        f["first5_red"] = float(first5["c"].iloc[-1]) < float(first5["o"].iloc[0])
    day = reg.index[0].normalize()
    for label, hhmm in (("0935", "9h35m"), ("1000", "10h"), ("1030", "10h30m")):
        m = reg.index <= day + pd.Timedelta(hhmm)
        if m.any():
            k = int(np.flatnonzero(m)[-1])
            f[f"above_vwap_{label}"] = bool(reg["c"].iloc[k] >= vw[k])
    hi = int(np.argmax(reg["h"].to_numpy()))
    f["hod"] = float(reg["h"].iloc[hi])
    f["hod_time"] = reg.index[hi].strftime("%H:%M")
    f["hod_above_pmh"] = f["hod"] > pmh
    f["close"] = float(reg["c"].iloc[-1])
    f["day_return_from_open"] = f["close"] / o0 - 1
    f["halts"] = count_halts(reg)
    f["ssr_carry"] = bool(g.get("prev_prev_close") and g["prev_low"] <= 0.9 * g["prev_prev_close"])
    f["ssr_today"] = bool((reg["l"] <= 0.9 * g["prev_close"]).any())
    gap, rot = g["pm_gap"], g.get("float_rotation")
    f["likely_htb"] = bool(gap >= cfg["dnt"]["likely_htb_gap"] or
                           (rot is not None and rot >= cfg["dnt"]["likely_htb_rotation"]))
    return f


# ------------------------------------------------------------------ costs
def trade_costs(entry: float, exit_: float, nights: int, ssr: bool, stopped: bool, cc: dict):
    """Per-share effective prices for a short, in REAL dollars."""
    slip_e = max(cc["min_slippage"], cc["slippage_pct"] * entry) + (cc["ssr_extra_slippage_pct"] * entry if ssr else 0)
    slip_x = max(cc["min_slippage"], cc["slippage_pct"] * exit_)
    entry_eff = entry - slip_e
    exit_eff = exit_ + slip_x          # stop extra slippage is already in exit_ for stopped trades
    fees = 2 * cc["commission_per_share"] + cc["locate_fee_per_share"] + entry * cc["borrow_fee_annual"] * nights / 360
    return entry_eff, exit_eff, fees


def nights_between(d1: str, d2: str) -> int:
    return (Date.fromisoformat(d2) - Date.fromisoformat(d1)).days


# --------------------------------------------------------------- simulate
def simulate_gapper(g: dict, bars: pd.DataFrame, cfg: dict) -> list[dict]:
    rc, cc, dnt = cfg["rules"], cfg["costs"], cfg["dnt"]
    reg = session(bars)
    if reg.empty:
        return []
    day = reg.index[0].normalize()
    factor = g["factor"]
    nxt = g.get("next")
    nxt = [] if nxt is None else [dict(d) for d in nxt]
    pmh = g["pm_high"]
    o, h, low, c = (reg[k].to_numpy() for k in ("o", "h", "l", "c"))
    t = reg.index
    eod_cut = day + pd.Timedelta(rc["eod_exit"] + ":00")
    eod_idx = int(np.searchsorted(t, eod_cut))            # first bar at/after 15:55
    tstop_idx = int(np.searchsorted(t, day + pd.Timedelta(rc["time_stop"] + ":00")))
    ipo_ok = g.get("days_since_ipo") is None or g["days_since_ipo"] >= dnt["min_days_since_ipo"]
    trades = []

    entries = []
    for e in rc["entries"]:
        if e in INTRADAY_ENTRIES:
            s = find_entry(e, reg, pmh, rc)
            if s is None or s.fill_idx >= len(reg):
                continue
            entries.append((e, s.fill_idx, float(o[s.fill_idx]), t[s.fill_idx], pmh))
        elif e == "next_open" and nxt:
            entries.append((e, None, float(nxt[0]["o"]), None, float(h.max())))

    for e, fi, px, ts, ref_high in entries:
        halts_before = count_halts(reg, fi) if fi is not None else count_halts(reg)
        ssr = bool(g.get("prev_prev_close") and g["prev_low"] <= 0.9 * g["prev_prev_close"])
        if fi is not None:
            ssr = ssr or bool((low[: fi] <= 0.9 * g["prev_close"]).any())
        else:
            ssr = bool(low.min() <= 0.9 * g["prev_close"]) or (nxt[0]["o"] <= 0.9 * float(c[-1]))
        dnt_reason = None
        if not ipo_ok:
            dnt_reason = "recent_ipo"
        elif halts_before >= dnt["max_halts_before_entry"]:
            dnt_reason = "halts"

        for x in rc["exits"]:
            if e == "next_open" and x in ("eod", "time_stop_1100", "next_open"):
                continue     # next_open entries are measured with the day-2/day-3 exits only
            for st in rc["stops"]:
                stop = stop_price(st, ref_high, rc)
                base = {"date": g["date"], "ticker": g["ticker"], "entry_rule": e, "exit_rule": x,
                        "stop_rule": st, "rule": f"{e}|{x}|{st}", "pm_gap": g["pm_gap"],
                        "likely_htb": g.get("likely_htb", False), "dnt": dnt_reason, "ssr_at_entry": ssr,
                        "halts_before_entry": halts_before}
                if dnt_reason:
                    trades.append({**base, "status": "skipped_dnt"})
                    continue
                if stop is not None and px >= stop:
                    trades.append({**base, "status": "skipped_above_stop"})
                    continue
                res = _walk(e, x, fi, px, stop, reg, o, h, low, c, t, eod_idx, tstop_idx, nxt, cc)
                if res is None:
                    trades.append({**base, "status": "no_exit_data"})
                    continue
                exit_px, exit_date, exit_time, stopped, mae_px, mfe_px, partial = res
                entry_date = g["date"] if fi is not None else nxt[0]["date"]
                nights = nights_between(entry_date, exit_date)
                E, X = px * factor, exit_px * factor
                ee, xe, fees = trade_costs(E, X, nights, ssr, stopped, cc)
                pnl = ee - xe - fees
                trades.append({**base, "status": "filled" if not partial else "exit_data_missing",
                               "entry_date": entry_date,
                               "entry_time": ts.strftime("%H:%M") if ts is not None else "09:30",
                               "exit_date": exit_date, "exit_time": exit_time,
                               "entry_px": E, "exit_px": X, "stop_px": stop * factor if stop else None,
                               "stopped": stopped, "nights": nights,
                               "ret": pnl / E, "pnl_per_share": pnl,
                               "r_mult": pnl / (stop * factor - E) if stop else None,
                               "mae": mae_px / px - 1, "mfe": 1 - mfe_px / px,
                               "gross_ret": (E - X) / E})
    return trades


def _walk(e, x, fi, px, stop, reg, o, h, low, c, t, eod_idx, tstop_idx, nxt, cc):
    """Return (exit_px, exit_date, exit_time, stopped, max_high, min_low, partial) in adjusted prices."""
    slip = 1 + cc["stop_extra_slippage_pct"]
    hi_seen, lo_seen = px, px
    day_str = t[0].strftime("%Y-%m-%d")

    def stop_hit_intraday(i):
        return stop is not None and h[i] >= stop

    if fi is not None:   # ---- intraday part of the trade
        last = min(eod_idx, len(reg) - 1) if x in ("eod", "time_stop_1100") else len(reg) - 1
        for i in range(fi, len(reg)):
            if x in ("eod", "time_stop_1100") and i >= eod_idx:
                return float(o[i]), day_str, t[i].strftime("%H:%M"), False, hi_seen, lo_seen, False
            if x == "time_stop_1100" and i == tstop_idx and o[i] >= px:
                return float(o[i]), day_str, t[i].strftime("%H:%M"), False, hi_seen, lo_seen, False
            if stop_hit_intraday(i):
                fill = max(stop, float(o[i])) * slip
                hi_seen = max(hi_seen, float(h[i]))
                return fill, day_str, t[i].strftime("%H:%M"), True, hi_seen, lo_seen, False
            hi_seen, lo_seen = max(hi_seen, float(h[i])), min(lo_seen, float(low[i]))
        if x in ("eod", "time_stop_1100"):
            return float(c[last]), day_str, t[last].strftime("%H:%M"), False, hi_seen, lo_seen, False
        days = list(nxt)
    else:                # ---- next_open entry: day 2 is the entry day
        if not nxt:
            return None
        d0 = nxt[0]
        if stop is not None and d0["h"] >= stop:
            hi_seen = max(hi_seen, d0["h"])
            return stop * slip, d0["date"], "intraday", True, hi_seen, lo_seen, False
        hi_seen, lo_seen = max(hi_seen, d0["h"]), min(lo_seen, d0["l"])
        days = list(nxt[1:])
        x = {"close_d2": "close_d1", "close_d3": "close_d2"}.get(x, x)

    # ---- overnight part, daily bars
    need = {"next_open": 1, "close_d1": 0, "close_d2": 1, "close_d3": 2}[x]
    if x == "close_d1":
        d = nxt[0]
        return d["c"], d["date"], "16:00", False, hi_seen, lo_seen, False
    for k in range(need):
        if k >= len(days):
            return None if not days else (days[-1]["c"], days[-1]["date"], "16:00", False, hi_seen, lo_seen, True)
        d = days[k]
        if stop is not None and d["o"] >= stop:
            return d["o"] * slip, d["date"], "09:30", True, max(hi_seen, d["h"]), lo_seen, False
        if x == "next_open":
            return d["o"], d["date"], "09:30", False, max(hi_seen, d["o"]), min(lo_seen, d["o"]), False
        if stop is not None and d["h"] >= stop:
            return stop * slip, d["date"], "intraday", True, max(hi_seen, d["h"]), lo_seen, False
        hi_seen, lo_seen = max(hi_seen, d["h"]), min(lo_seen, d["l"])
        if k == need - 1:
            return d["c"], d["date"], "16:00", False, hi_seen, lo_seen, False
    return None


def run_backtest(gappers: pd.DataFrame, load_bars, cfg: dict, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats, trades = [], []
    for i, g in enumerate(gappers.to_dict("records")):
        if progress:
            progress(i, len(gappers))
        bars = load_bars(g["ticker"], g["date"])
        f = gapper_features(g, bars, cfg)
        g["likely_htb"] = f.get("likely_htb", False)
        feats.append({**{k: g[k] for k in ("date", "ticker", "pm_gap", "pm_high", "pm_last", "prev_close",
                                            "pm_volume_real", "pm_price_real", "shares_out", "float_rotation",
                                            "runner_days", "days_since_ipo") if k in g}, **f})
        trades += simulate_gapper(g, bars, cfg)
    return pd.DataFrame(feats), pd.DataFrame(trades)
