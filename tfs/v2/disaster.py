"""Two honest looks at the worst case of a no-stop short.

1. tail_table: how often a gapper runs R times above its 09:25 price within 5 days, measured
   and extrapolated with a power-law tail (Hill estimator):
       runs per year at or above R  =  gappers per year * (k/n) * (R / x_k) ** -alpha
2. with_stop: the same target / time-limit trade plus a far-away "disaster stop" at k times the
   entry price. The stop fills at the stop price, or at the bar's open when the price jumped
   over it (a halt or an overnight gap), plus the stop slippage from config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfs.backtest import nights_between
from tfs.rules import session
from tfs.v2.paths import _costs, _end, _entries, _series


def hill(x: np.ndarray, k: int = 40) -> tuple[float, float]:
    """Tail index alpha and the threshold x_k from the k largest values."""
    s = np.sort(np.asarray(x, float))[::-1]
    k = min(k, len(s) - 1)
    return float(1 / np.mean(np.log(s[:k] / s[k]))), float(s[k])


def tail_table(sq: np.ndarray, per_year: float, levels=(3, 5, 10, 20, 33, 50, 100), k: int = 40) -> pd.DataFrame:
    sq = np.asarray(sq, float); sq = sq[~np.isnan(sq)]
    alpha, xk = hill(sq, k)
    rows = []
    for r in levels:
        fit = per_year * (k / len(sq)) * (r / xk) ** -alpha
        rows.append({"run_up": r, "seen": int((sq >= r).sum()), "fit_per_year": fit, "once_every_years": 1 / fit})
    out = pd.DataFrame(rows); out.attrs["alpha"] = alpha
    return out


def with_stop(g: dict, bars: pd.DataFrame, cfg: dict, entry: str, target: float, horizon: int, k: float) -> dict | None:
    cc = cfg["costs"]
    reg = session(bars)
    if reg.empty:
        return None
    nxt = [dict(d) for d in (g.get("next10") if g.get("next10") is not None else [])]
    day = reg.index[0].normalize()
    eod_cut = day + pd.Timedelta(cfg["rules"]["eod_exit"] + ":00")
    arr = lambda key: np.array([d[key] for d in nxt])
    c = next((c for c in _entries(g, bars, cfg) if c["entry"] == entry), None)
    if c is None:
        return None
    s = _series(c, g["date"], eod_cut, arr("o"), arr("h"), arr("l"), arr("c"), [d["date"] for d in nxt])
    e = _end(s, horizon, True)
    if e is None:
        return None
    end_i, end_px, _ = e
    O, H, L = s["O"], s["H"], s["L"]
    px = c["px"]
    tpx, spx = px * (1 - target), px * k
    t_hit = np.flatnonzero(L[: end_i + 1] <= tpx)
    s_hit = np.flatnonzero(H[: end_i + 1] >= spx)
    ti = int(t_hit[0]) if len(t_hit) else None
    si = int(s_hit[0]) if len(s_hit) else None
    if si is not None and (ti is None or si <= ti):         # same bar: assume the stop came first
        i, stopped, hit = si, True, False
        exit_px = max(spx, O[i] if i > 0 else spx) * (1 + cc.get("stop_extra_slippage_pct", 0.03))
        r_max = exit_px / px
    elif ti is not None:
        i, stopped, hit = ti, False, True
        exit_px = float(O[i]) if (O[i] <= tpx and i > 0) else tpx
        r_max = max(H[: i + 1].max(), px) / px
    else:
        i, stopped, hit, exit_px = end_i, False, False, end_px
        r_max = max(H[: i + 1].max(), px) / px
    nights = nights_between(s["entry_date"], s["step_date"][i])
    return {"date": g["date"], "ticker": g["ticker"], "entry": entry, "target": target, "horizon": horizon, "stop_x": k,
            "hit": hit, "stopped": stopped, "ret": _costs(cc, px * g["factor"], exit_px * g["factor"], nights),
            "R": float(r_max), "entry_px": px * g["factor"], "exit_date": s["step_date"][i], "nights": nights}
