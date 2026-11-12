"""Run one rule's trades through a small account with all the risk rules on."""
from __future__ import annotations

import pandas as pd

from tfs.risk import margin_requirement, size_short


def _ts(date: str, hhmm: str) -> pd.Timestamp:
    hhmm = {"intraday": "12:00"}.get(hhmm, hhmm)
    return pd.Timestamp(f"{date} {hhmm}")


def simulate_account(trades: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    risk, cc = cfg["risk"], cfg["costs"]
    t = trades[trades["status"].isin(["filled", "exit_data_missing"])].copy()
    if t.empty:
        return pd.DataFrame(), {"trades": 0}
    t["t_in"] = [_ts(d, h) for d, h in zip(t["entry_date"], t["entry_time"])]
    t["t_out"] = [_ts(d, h) for d, h in zip(t["exit_date"], t["exit_time"])]
    t = t.sort_values("t_in")
    equity = float(risk["starting_equity"])
    open_pos: list[dict] = []
    rows, per_day, day_pnl = [], {}, {}
    skipped = {"daily_loss_limit": 0, "max_trades_per_day": 0, "size_zero": 0}

    def close_until(ts):
        nonlocal equity
        for p in sorted([p for p in open_pos if p["t_out"] <= ts], key=lambda p: p["t_out"]):
            equity += p["pnl"]
            d = p["exit_date"]
            day_pnl[d] = day_pnl.get(d, 0.0) + p["pnl"]
            open_pos.remove(p)
            rows.append({"time": p["t_out"], "equity": equity, "ticker": p["ticker"], "pnl": p["pnl"],
                         "shares": p["shares"], "ret": p["ret"]})

    for r in t.itertuples(index=False):
        close_until(r.t_in)
        d = r.entry_date
        if day_pnl.get(d, 0.0) <= -risk["daily_loss_limit"] * equity:
            skipped["daily_loss_limit"] += 1
            continue
        if per_day.get(d, 0) >= risk["max_trades_per_day"]:
            skipped["max_trades_per_day"] += 1
            continue
        if any(p["ticker"] == r.ticker for p in open_pos):
            continue                      # never add to an existing short
        exposure = sum(p["entry_px"] * p["shares"] for p in open_pos)
        margin = sum(margin_requirement(p["entry_px"], p["shares"], risk.get("margin_rule", "finra"))
                     for p in open_pos)
        stop = r.stop_px if pd.notna(r.stop_px) else None
        shares, _ = size_short(equity, r.entry_px, stop, risk, exposure, margin)
        if shares <= 0:
            skipped["size_zero"] += 1
            continue
        commission = max(cc["commission_min"], cc["commission_per_share"] * shares) * 2
        per_share_comm = 2 * cc["commission_per_share"]
        pnl = (r.pnl_per_share + per_share_comm) * shares - commission
        per_day[d] = per_day.get(d, 0) + 1
        open_pos.append({"ticker": r.ticker, "shares": shares, "entry_px": r.entry_px, "t_out": r.t_out,
                         "exit_date": r.exit_date, "pnl": pnl, "ret": r.ret})
    close_until(pd.Timestamp.max)
    curve = pd.DataFrame(rows)
    start = float(risk["starting_equity"])
    if curve.empty:
        return curve, {"trades": 0, **skipped}
    peak = curve["equity"].cummax()
    dd = (curve["equity"] / peak - 1).min()
    summary = {"trades": len(curve), "start": start, "end": float(curve["equity"].iloc[-1]),
               "total_return": float(curve["equity"].iloc[-1] / start - 1),
               "max_drawdown": float(min(dd, curve["equity"].min() / start - 1, 0)),
               "worst_trade": float(curve["pnl"].min()), "best_trade": float(curve["pnl"].max()), **skipped}
    return curve, summary
