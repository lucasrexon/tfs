"""Export build-period backtest results as JSON files for the interactive report.

Only the build period is exported; the sealed test period never leaves the
machine until `python -m tfs holdout` is run. Each file is {rows, meta}, and
meta carries this script's own source so readers can see how every number was
made.

    python -m tfs.export_dashboard            -> dashboard/*.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tfs.config import load_config, resolve
from tfs.portfolio import simulate_account
from tfs.stats import summarize

FILLED = ("filled", "exit_data_missing")


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _rows(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def main():
    cfg = load_config()
    out = resolve(cfg, "output")
    dest = resolve(cfg, "dashboard")
    dest.mkdir(exist_ok=True)
    be = cfg["split"]["build_end"]
    code = Path(__file__).read_text()
    g = pd.read_parquet(out / "gappers.parquet")
    f = pd.read_parquet(out / "features.parquet")
    t = pd.read_parquet(out / "trades.parquet")
    b = t[t["date"] <= be].copy()
    fb = f[f["date"] <= be].copy()
    bucket = fb.set_index(["date", "ticker"])["gap_bucket"]
    b = b.join(bucket, on=["date", "ticker"])

    meta_common = [{"label": "export_dashboard.py", "value": code, "type": "code", "lang": "python"},
                   {"label": "Period", "value": f"{cfg['data']['start']} to {be} (build period only)"},
                   {"label": "Input", "value": "output/trades.parquet, features.parquet, gappers.parquet from `python -m tfs run`"}]

    # ---- rules: one row per rule x universe, with the $10k account result
    rule_rows, curve_rows = [], []
    for universe, part in (("all", b), ("borrowable", b[~b["likely_htb"]])):
        for rule, gr in part.groupby("rule"):
            e, x, s = rule.split("|")
            st = summarize(gr)
            st.pop("worst5", None)
            curve, acct = simulate_account(gr, cfg)
            rule_rows.append({"rule": rule, "entry": e, "exit": x, "stop": s, "universe": universe,
                              "same_day": x in ("eod", "time_stop_1100"), **st,
                              "acct_end": acct.get("end"), "acct_return": acct.get("total_return"),
                              "acct_max_dd": acct.get("max_drawdown"), "acct_trades": acct.get("trades", 0)})
            if not curve.empty:
                c = curve.reset_index(drop=True)
                idx = np.unique(np.linspace(0, len(c) - 1, min(len(c), 160)).round().astype(int))
                curve_rows.append({"rule": rule, "universe": universe, "n": 0, "date": None,
                                   "equity": float(cfg["risk"]["starting_equity"])})
                for i in idx:
                    curve_rows.append({"rule": rule, "universe": universe, "n": int(i + 1),
                                       "date": str(c.loc[i, "time"])[:10], "equity": round(float(c.loc[i, "equity"]), 2)})
    rules = pd.DataFrame(rule_rows)

    # ---- trades: every filled build-period trade
    tr = b[b["status"].isin(FILLED)][["rule", "date", "ticker", "ret", "mae", "gap_bucket", "likely_htb",
                                      "stopped"]].copy()
    tr["ret"] = tr["ret"].round(4)
    tr["mae"] = tr["mae"].round(4)

    # ---- gappers: what the stocks themselves did
    gp = fb[["date", "ticker", "pm_gap", "gap_bucket", "open_gap", "day_return_from_open", "hod_time",
             "open_below_pmh", "likely_htb", "float_rotation", "pm_price_real", "halts"]].copy()

    summary = pd.DataFrame([{
        "gappers_total": int(len(g)), "build_gappers": int((g["date"] <= be).sum()),
        "test_gappers": int((g["date"] > be).sum()), "rule_variants": int(rules["rule"].nunique()),
        "build_start": cfg["data"]["start"], "build_end": be, "test_end": cfg["data"]["end"],
        "risk_per_trade": cfg["risk"]["risk_per_trade"], "start_equity": cfg["risk"]["starting_equity"],
        "daily_loss_limit": cfg["risk"]["daily_loss_limit"], "max_trades_per_day": cfg["risk"]["max_trades_per_day"],
    }])
    files = {"rules": rules, "trades": tr, "curves": pd.DataFrame(curve_rows), "gappers": gp, "summary": summary}
    for name, df in files.items():
        p = dest / f"{name}.json"
        p.write_text(json.dumps({"rows": _rows(df), "meta": meta_common}, separators=(",", ":")))
        print(f"{p}  {len(df):,} rows  {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
