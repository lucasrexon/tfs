"""No-stop short simulation: every entry x profit target x time limit, per gapper.

A trade ends one of two ways: the price falls to the profit target (cover there),
or the time limit runs out (cover at that close). There is no stop-loss, so for
every trade we also record R, the highest price reached before the exit divided
by the entry price. R is what decides whether the account survives.

Entry families
  at a time        pm_0925, open, t_1000, t_1030, t_1100, t_1400, close_1555, next_open
  into strength    pmh_+x% (limit near the pre-market high), ext_+x% (limit above the open)
  on weakness      first_red_5m, vwap_loss, orb5_break, orb15_break, climax_break, red_1030
  backside         vwap_reject (lost VWAP, retest fails), lower_high (first lower high)
  day 2            next_open, d2_red (day 2 trades below day 1's close)
  scaling in       ladder_pm, ladder_open (a third now, a third 25% higher, a third 50% higher)

Conservative choices:
  * in the minute (or day) the target is hit, that bar's high still counts toward R
  * a sell limit fills at the limit, or at the bar's open if it opened above it
  * a signal on a bar's close fills at the next bar's open
  * slippage, commission, locate fee, and a borrow fee for every night held
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfs.backtest import nights_between
from tfs.rules import find_entry, running_vwap, session

TARGETS = (0.10, 0.15, 0.20, 0.25, 0.30)
HORIZONS = (0, 1, 3, 5, 10)                       # 0 = same day; k = close of the k-th day after entry
PMH_OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.20)
EXT_OFFSETS = (0.20, 0.30, 0.50)
TIMES = ("10:00", "10:30", "11:00", "14:00")
LADDER = (1.0, 1.25, 1.50)
LIMIT_UNTIL = "15:00"


def lower_high(o, h, low, c, start: int, until: int, pull: float = 0.10, bounce: float = 0.05) -> int | None:
    """Index of the bar that confirms the first lower high: the high of day is in, price pulled
    back `pull`, bounced `bounce` without making a new high, then a bar closes under the bar before."""
    hod, pl, bh, state = h[0], None, None, 0
    for i in range(1, until):
        if h[i] > hod:
            hod, pl, bh, state = h[i], None, None, 0
            continue
        if state == 0:
            if low[i] <= hod * (1 - pull):
                pl, state = low[i], 1
        elif state == 1:
            pl = min(pl, low[i])
            if h[i] >= pl * (1 + bounce):
                bh, state = h[i], 2
        else:
            bh = max(bh, h[i])
            if i >= start and c[i] < low[i - 1]:
                return i
    return None


def _entries(g: dict, bars: pd.DataFrame, cfg: dict) -> list[dict]:
    """Each entry: name, fill price (adjusted), frame of minute bars, index where the trade starts."""
    reg = session(bars)
    if reg.empty:
        return []
    day = reg.index[0].normalize()
    at = lambda hhmm: int(np.searchsorted(reg.index, day + pd.Timedelta(hhmm + ":00")))
    out = []
    o, h, low, c, v = (reg[k].to_numpy() for k in ("o", "h", "l", "c", "v"))
    n = len(reg)

    def add(name, i, px=None, **kw):
        if i is not None and 0 <= i < n:
            out.append({"entry": name, "px": float(o[i] if px is None else px), "frame": reg, "i0": int(i), **kw})

    # ---- at a time
    pre = bars[(bars.index >= day + pd.Timedelta("9h25m")) & (bars.index < day + pd.Timedelta("16h"))]
    if not pre.empty and pre.index[0] < day + pd.Timedelta("9h30m"):
        out.append({"entry": "pm_0925", "px": float(pre["o"].iloc[0]), "frame": pre, "i0": 0})
    add("open", 0)
    for hhmm in TIMES:
        i = at(hhmm)
        if i < n and reg.index[i] < day + pd.Timedelta(hhmm + ":00") + pd.Timedelta("15min"):
            add(f"t_{hhmm.replace(':', '')}", i)
    i = at(cfg["rules"]["eod_exit"])
    add("close_1555", i, no_same_day=True)

    # ---- into strength: resting sell limits
    until = at(LIMIT_UNTIL)
    for name, base, offs in (("pmh", g["pm_high"], PMH_OFFSETS), ("ext", o[0], EXT_OFFSETS)):
        for x in offs:
            lim = base * (1 + x)
            hit = np.flatnonzero(h[:until] >= lim)
            if len(hit):
                i = int(hit[0])
                add(f"{name}_{x:+.0%}", i, px=max(lim, o[i]), mid_bar=bool(o[i] < lim))   # low may have come first

    # ---- on weakness (signal on a close, fill at the next open)
    for rule in ("first_red_5m", "vwap_loss"):
        s = find_entry(rule, reg, g["pm_high"], cfg["rules"])
        if s is not None:
            add(rule, s.fill_idx)
    for k in (5, 15):
        if n > k:
            brk = np.flatnonzero(c[k:until] < low[:k].min())
            if len(brk):
                add(f"orb{k}_break", k + int(brk[0]) + 1)
    if n > 5:                                           # close under the low of the busiest minute so far
        vmax_i = np.maximum.accumulate(np.where(v == np.maximum.accumulate(v), np.arange(n), 0))
        prev_busy = np.r_[0, vmax_i[:-1]]
        brk = np.flatnonzero(c[5:until] < low[prev_busy[5:until]])
        if len(brk):
            add("climax_break", 5 + int(brk[0]) + 1)
    vw = running_vwap(reg)
    i = at("10:30")
    if 0 < i < n and c[i - 1] < o[0] and c[i - 1] < vw[i - 1]:
        add("red_1030", i)

    # ---- backside
    lost = np.flatnonzero(c[:until] < vw[:until] * (1 - 1e-9))
    if len(lost):
        s = int(lost[0]) + 1
        rej = np.flatnonzero((h[s:until] >= vw[s:until]) & (c[s:until] < vw[s:until]) & (c[s:until] < o[s:until]))
        if len(rej):
            add("vwap_reject", s + int(rej[0]) + 1)
    i = lower_high(o, h, low, c, 5, until)
    if i is not None:
        add("lower_high", i + 1)
    return out


def _series(c: dict, g_date: str, eod_cut, dO, dH, dL, dC, dates):
    """Minute bars from the entry onward, then one bar per later day."""
    px = c["px"]
    if c["frame"] is None:
        return dict(O=dO, H=dH, L=dL, C=dC, n_min=0, eod_i=None, entry_date=dates[0], entry_time="09:30", step_date=dates)
    f = c["frame"].iloc[c["i0"]:]
    mo, mh, ml, mc = (f[k].to_numpy() for k in ("o", "h", "l", "c"))
    mo = mo.copy(); mo[0] = px                         # the fill bar starts at our fill
    if c.get("mid_bar"):
        ml = ml.copy(); ml[0] = px                     # ignore the fill bar's low; keep its high
    n_min = len(f)
    return dict(O=np.concatenate([mo, dO]), H=np.concatenate([mh, dH]), L=np.concatenate([ml, dL]),
                C=np.concatenate([mc, dC]), n_min=n_min, eod_i=int(np.searchsorted(f.index, eod_cut)),
                entry_date=g_date, entry_time=f.index[0].strftime("%H:%M"), step_date=[g_date] * n_min + dates)


def _end(s: dict, hz: int, intraday: bool):
    """(index of the last bar, price at the time limit, partial?) or None if there is no data."""
    O, C, n_min, eod_i = s["O"], s["C"], s["n_min"], s["eod_i"]
    end_px = None
    if intraday:
        if hz == 0:
            end_i = max(min(eod_i, n_min) - 1, 0)      # last full bar before 15:55
            end_px = float(O[eod_i]) if eod_i < n_min else float(C[n_min - 1])
        else:
            end_i = n_min + hz - 1
    else:
        end_i = hz                                     # day entries: horizon counts from the entry day
    partial = False
    if end_i >= len(O):
        if len(O) == 0:
            return None
        end_i, partial = len(O) - 1, True
    return end_i, (float(C[end_i]) if end_px is None else end_px), partial


def _costs(cc: dict, E: float, X: float, nights: int) -> float:
    """Net return of one short from E to X (real dollars per share), as a fraction of E."""
    se = max(cc["min_slippage"], cc["slippage_pct"] * E)
    sx = max(cc["min_slippage"], cc["slippage_pct"] * X)
    fees = 2 * cc["commission_per_share"] + cc["locate_fee_per_share"] + E * cc["borrow_fee_annual"] * nights / 360
    return ((E - se) - (X + sx) - fees) / E


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
        d1c = float(reg["c"].iloc[-1])                 # day 2 trades below day 1's close
        if dO[0] < d1c:
            cases.append({"entry": "d2_red", "px": float(dO[0]), "frame": None, "i0": 0})
        elif dL[0] <= d1c:
            cases.append({"entry": "d2_red", "px": d1c, "frame": None, "i0": 0})

    for c in cases:
        px = c["px"]
        s = _series(c, g["date"], eod_cut, dO, dH, dL, dC, dates)
        O, H, L = s["O"], s["H"], s["L"]
        cmax = np.maximum.accumulate(H)
        for hz in HORIZONS:
            if hz == 0 and c.get("no_same_day"):
                continue
            e = _end(s, hz, c["frame"] is not None)
            if e is None:
                continue
            end_i, end_px, partial = e
            for tg in TARGETS:
                tpx = px * (1 - tg)
                hit = np.flatnonzero(L[: end_i + 1] <= tpx)
                if len(hit):
                    i = int(hit[0])
                    exit_px = float(O[i]) if (O[i] <= tpx and i > 0) else tpx
                    is_hit = True
                else:
                    i, exit_px, is_hit = end_i, end_px, False
                nights = nights_between(s["entry_date"], s["step_date"][i])
                rows.append({"date": g["date"], "ticker": g["ticker"], "entry": c["entry"], "target": tg, "horizon": hz,
                             "hit": is_hit, "ret": _costs(cc, px * factor, exit_px * factor, nights),
                             "R": float(max(cmax[i], px) / px), "entry_px": px * factor,
                             "entry_time": s["entry_time"], "exit_date": s["step_date"][i], "nights": nights,
                             "partial": partial and not is_hit})
        if c["entry"] in ("pm_0925", "open"):
            rows += _ladder(g, c, s, cc)
    return rows


def _ladder(g: dict, c: dict, s: dict, cc: dict) -> list[dict]:
    """Scale in with equal dollars: a third at the entry, a third 25% higher, a third 50% higher.

    The target is measured from the average entry of what has filled so far. `ret` is per dollar
    of the FULL planned position, and R is the single-position run-up that would stress the
    account equally (with 100% margin a position of size f survives while 1 >= f*(2R - 1)).
    """
    O, H, L = s["O"], s["H"], s["L"]
    p0, factor = c["px"], g["factor"]
    w = 1.0 / len(LADDER)
    fills = [(0, p0)]
    for m in LADDER[1:]:
        hit = np.flatnonzero(H >= p0 * m)
        if len(hit):
            i = int(hit[0])
            fills.append((i, max(p0 * m, O[i]) if i > 0 else p0 * m))
    rows = []
    for hz in HORIZONS:
        e = _end(s, hz, True)
        if e is None:
            continue
        end_i, end_px, partial = e
        for tg in TARGETS:
            held, r_eff, exit_i, exit_px, is_hit = [], 0.0, end_i, end_px, False
            for k, (fi, fp) in enumerate(fills):
                if fi > end_i:
                    break
                held.append(fp)
                nxt_fill = fills[k + 1][0] if k + 1 < len(fills) else len(O)
                seg_end = min(nxt_fill, end_i + 1)                      # bars fi .. seg_end-1 (+ the next fill bar's high)
                hi = H[fi:min(seg_end + 1, end_i + 1)].max()
                avg = len(held) / sum(1 / p for p in held)
                tpx = avg * (1 - tg)
                lo_from = fi if k == 0 else fi + 1                       # a later fill bar's low may have come first
                hit = np.flatnonzero(L[lo_from:seg_end] <= tpx)
                if len(hit):
                    j = lo_from + int(hit[0])
                    exit_i, exit_px, is_hit = j, (float(O[j]) if (O[j] <= tpx and j > lo_from) else tpx), True
                    hi = H[fi:j + 1].max()
                a, m_ = sum(w * hi / p for p in held), w * len(held)
                r_eff = max(r_eff, a + (1 - m_) / 2)
                if is_hit:
                    break
            nights = nights_between(s["entry_date"], s["step_date"][exit_i])
            ret = sum(w * _costs(cc, p * factor, exit_px * factor, nights) for p in held)
            rows.append({"date": g["date"], "ticker": g["ticker"], "entry": "ladder_" + c["entry"].split("_")[0],
                         "target": tg, "horizon": hz, "hit": is_hit, "ret": ret, "R": float(max(r_eff, 1.0)),
                         "entry_px": len(held) / sum(1 / p for p in held) * factor, "entry_time": s["entry_time"],
                         "exit_date": s["step_date"][exit_i], "nights": nights, "partial": partial and not is_hit})
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
