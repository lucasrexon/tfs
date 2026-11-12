"""Per-rule statistics, all after costs."""
from __future__ import annotations

import numpy as np
import pandas as pd

FILLED = ("filled", "exit_data_missing")


def summarize(tr: pd.DataFrame) -> dict:
    f = tr[tr["status"].isin(FILLED)]
    n = len(f)
    out = {"candidates": len(tr), "trades": n,
           "skipped_dnt": int((tr["status"] == "skipped_dnt").sum()),
           "skipped_above_stop": int((tr["status"] == "skipped_above_stop").sum()),
           "no_exit_data": int((tr["status"] == "no_exit_data").sum())}
    if n == 0:
        return out
    r = f["ret"].to_numpy()
    wins, losses = r[r > 0], r[r <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    out.update({
        "win_rate": len(wins) / n,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(r.mean()),
        "median": float(np.median(r)),
        "gross_expectancy": float(f["gross_ret"].mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "worst": float(r.min()),
        "worst5": ", ".join(f"{t} {d} {x:+.0%}" for t, d, x in
                            f.nsmallest(5, "ret")[["ticker", "date", "ret"]].itertuples(index=False)),
        "mae_p95": float(np.percentile(f["mae"], 95)),
        "stopped_pct": float(f["stopped"].mean()),
        "avg_r": float(f["r_mult"].mean()) if f["r_mult"].notna().any() else None,
        # t-stat of the mean: is the expectancy distinguishable from zero?
        "t_stat": float(r.mean() / (r.std(ddof=1) / np.sqrt(n))) if n > 2 and r.std(ddof=1) > 0 else None,
    })
    return out


def rule_table(trades: pd.DataFrame, exclude_htb: bool = False) -> pd.DataFrame:
    tr = trades[~trades["likely_htb"]] if exclude_htb else trades
    rows = [{"rule": rule, **summarize(g)} for rule, g in tr.groupby("rule")]
    return pd.DataFrame(rows).sort_values("expectancy", ascending=False, na_position="last") \
        if rows else pd.DataFrame()


def by_group(trades: pd.DataFrame, rule: str, col: str) -> pd.DataFrame:
    t = trades[(trades["rule"] == rule)]
    rows = [{col: k, **summarize(g)} for k, g in t.groupby(col)]
    return pd.DataFrame(rows)
