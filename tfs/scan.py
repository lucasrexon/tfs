"""Find every pre-market gapper in the date range.

Stage 1 (one call per trading day): walk the whole market's daily bars in date
order and keep stocks whose open or high ran far above the prior close.
Stage 2 (a few calls per candidate): pull the day's 1-minute bars and keep only
stocks that were really +50% at 09:25 with real volume and price, then attach
float, listing date and split factor. Only what was knowable at 09:25 is used
to decide whether a stock is a gapper.
"""
from __future__ import annotations

from datetime import date as Date, timedelta

import pandas as pd

from tfs.data.massive import split_factor

NEXT_DAYS = 3


def trading_days(client, start: str, end: str):
    d, e = Date.fromisoformat(start), Date.fromisoformat(end)
    while d <= e:
        if d.weekday() < 5:
            df = client.grouped_daily(d.isoformat())
            if not df.empty:
                yield d.isoformat(), df
        d += timedelta(days=1)


def stage1(client, cfg: dict, universe: set[str] | None, progress=None) -> pd.DataFrame:
    sc = cfg["scan"]
    prev: dict[str, tuple] = {}     # ticker -> (close, low, prev_close)
    streak: dict[str, int] = {}     # consecutive up-closes up to yesterday
    pending: list[dict] = []
    done: list[dict] = []
    for i, (day, df) in enumerate(trading_days(client, cfg["data"]["start"], cfg["data"]["end"])):
        bars = {r.T: r for r in df.itertuples(index=False)}
        still = []
        for c in pending:   # collect day+1..day+3 daily bars for exits
            r = bars.get(c["ticker"])
            if r is not None:
                c["next"].append({"date": day, "o": r.o, "h": r.h, "l": r.l, "c": r.c, "v": r.v})
            c["_left"] -= 1
            (still if c["_left"] > 0 else done).append(c)
        pending = still
        for t, r in bars.items():
            p = prev.get(t)
            if p is None or not p[0] or p[0] <= 0:
                continue
            pc = p[0]
            if (r.o >= sc["stage1_open_ratio"] * pc or r.h >= sc["stage1_high_ratio"] * pc) \
                    and (universe is None or t in universe):
                pending.append({
                    "date": day, "ticker": t, "prev_close": pc, "prev_low": p[1],
                    "prev_prev_close": p[2], "runner_days": streak.get(t, 0),
                    "d_open": r.o, "d_high": r.h, "d_low": r.l, "d_close": r.c, "d_volume": r.v,
                    "next": [], "_left": NEXT_DAYS,
                })
        for t, r in bars.items():
            p = prev.get(t)
            streak[t] = (streak.get(t, 0) + 1) if (p and r.c > p[0]) else 0
            prev[t] = (r.c, r.l, p[0] if p else None)
        if progress:
            progress(i, day, len(pending) + len(done))
    done += pending
    for c in done:
        c.pop("_left", None)
    return pd.DataFrame(done)


def premarket_stats(bars: pd.DataFrame, cutoff: str) -> dict:
    """Pre-market numbers as of the cutoff (bars that START before the cutoff)."""
    if bars.empty:
        return {"pm_last": None, "pm_high": None, "pm_volume": 0.0, "pm_bars": 0}
    t = bars.index
    day = t[0].normalize()
    pm = bars[(t >= day + pd.Timedelta("4h")) & (t < day + pd.Timedelta(cutoff + ":00"))]
    if pm.empty:
        return {"pm_last": None, "pm_high": None, "pm_volume": 0.0, "pm_bars": 0}
    return {"pm_last": float(pm["c"].iloc[-1]), "pm_high": float(pm["h"].max()),
            "pm_volume": float(pm["v"].sum()), "pm_bars": int(len(pm))}


def stage2(client, cfg: dict, cands: pd.DataFrame, progress=None) -> pd.DataFrame:
    sc = cfg["scan"]
    out = []
    for i, c in enumerate(cands.to_dict("records")):
        if progress:
            progress(i, len(cands), c["ticker"], c["date"])
        bars = client.minute_bars(c["ticker"], c["date"])
        pm = premarket_stats(bars, sc["premarket_cutoff"])
        if pm["pm_last"] is None:
            continue
        gap = pm["pm_last"] / c["prev_close"] - 1
        if gap < sc["min_premarket_gap"]:
            continue
        factor = split_factor(client.splits(c["ticker"]), c["date"])
        pm_price_real = pm["pm_last"] * factor
        pm_vol_real = pm["pm_volume"] / factor
        if pm_price_real < sc["min_price"] or pm_vol_real < sc["min_premarket_volume"]:
            continue
        det = client.ticker_details(c["ticker"], c["date"])
        shares = det.get("share_class_shares_outstanding") or det.get("weighted_shares_outstanding")
        list_date = det.get("list_date")
        c.update({
            "pm_gap": gap, "pm_last": pm["pm_last"], "pm_high": pm["pm_high"],
            "pm_volume_real": pm_vol_real, "pm_price_real": pm_price_real, "factor": factor,
            "shares_out": float(shares) if shares else None,
            "float_rotation": (pm_vol_real / float(shares)) if shares else None,
            "list_date": list_date,
            "days_since_ipo": (Date.fromisoformat(c["date"]) - Date.fromisoformat(list_date)).days
            if list_date else None,
            "type": det.get("type"), "name": det.get("name"),
        })
        out.append(c)
    return pd.DataFrame(out)
