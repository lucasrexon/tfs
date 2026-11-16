"""The one-time test of the final strategy on the sealed year (never used to choose anything).

    python -m tfs.v2.sealed

Reads the rules from config.yaml (v2.strategy), which were fixed and pushed before this ran.
Refuses to run twice: the result is saved to output/v2/sealed_result.json and shown again
on every later call.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from tfs.config import load_config, resolve
from tfs.data.massive import MassiveClient
from tfs.v2.combine import LIKELY_BORROWABLE
from tfs.v2.ideas import account, ev, stop_grid, COLS


def main():
    cfg = load_config()
    od = resolve(cfg, "output") / "v2"
    out_f = od / "sealed_result.json"
    if out_f.exists():
        print("The sealed test has already been run. Saved result:\n")
        print(out_f.read_text())
        return
    st = cfg["v2"]["strategy"]
    be = cfg["split"]["build_end"]
    cache = resolve(cfg, cfg["data"]["cache_dir"])
    g = pd.read_parquet(od / "gappers40.parquet")
    g = g[(g.date > be) & (g.pm_gap >= st["min_gap"])].reset_index(drop=True)
    has_next = np.array([x is not None and len(x) >= st["limit_days"] for x in g.next10])
    excluded = int((~has_next).sum())
    g = g[has_next].reset_index(drop=True)
    feat = pd.read_parquet(od / "features.parquet").merge(pd.read_parquet(od / "features2.parquet"), on=["date", "ticker"])
    cl = MassiveClient(cfg["data"]["base_url"], cache)

    rows = []
    for i, r in enumerate(g.to_dict("records")):
        if i % 100 == 0:
            print(f"  sealed {i:,}/{len(g):,}", flush=True)
        rr, _ = stop_grid(r, cl.minute_bars(r["ticker"], r["date"]), cfg)
        rows += [x for x in rr if x[2] == st["entry"] and np.isclose(x[3], st["target"]) and x[4] == st["limit_days"]
                 and np.isclose(x[5], st["disaster_stop_x"])]                  # everything else is thrown away unseen
    t = pd.DataFrame(rows, columns=COLS).merge(
        feat[["date", "ticker", "pm_gap", "going_concern", "float_rotation", "industry", "catalyst"]], on=["date", "ticker"])
    t = t.sort_values(["date", "ticker"]).reset_index(drop=True)
    days = sorted(t.date.unique())
    years = (pd.Timestamp(days[-1]) - pd.Timestamp(days[0])).days / 365.25
    f = st["position_pct"]
    main_t = t[t.going_concern == 1]
    t.to_csv(od / "sealed_trades.csv", index=False)

    def summary(x, max_open):
        m, ts = ev(x)
        a = account(x, f, years, max_pos=max_open, n_sims=500)
        return {"trades_available": len(x), "trades_taken": a["taken"], "hit_rate": float(x.hit.mean()), "stopped": float(x.stopped.mean()),
                "avg_trade": m, "t_stat": ts, "worst_trade_of_position": float(x.ret.min()),
                "typical_year": a["typical"], "bad_luck_year": a["bad"], "typical_worst_drop": a["dd"],
                "liquidated_runs": a["liq"], "closest_to_liquidation": a["stress"]}

    res = {"run_at": datetime.now().isoformat(timespec="seconds"), "period": [days[0], days[-1]], "years": years,
           "rules": st, "excluded_no_next_day": excluded,
           "primary": summary(main_t, st["max_open"]),
           "secondary_1_open": summary(main_t, 1),
           "secondary_no_going_concern_rule_1_open": summary(t, 1),
           "secondary_no_going_concern_rule_3_open": summary(t, st["max_open"])}
    for extra in (0.01, 0.03):
        a = account(main_t.assign(ret=main_t.ret - extra), f, years, max_pos=st["max_open"], n_sims=500)
        res[f"primary_extra_cost_{int(extra * 100)}pct"] = {"typical_year": a["typical"], "bad_luck_year": a["bad"]}
    e = main_t[LIKELY_BORROWABLE(main_t)]
    res["primary_likely_borrowable_only"] = summary(e, st["max_open"]) if len(e) >= 10 else {"trades_available": len(e)}
    q = pd.to_datetime(main_t.date).dt.to_period("Q").astype(str)
    res["avg_trade_by_quarter"] = {k: [int(len(v)), float(v.mean())] for k, v in main_t.ret.groupby(q)}
    res["worst_trades"] = main_t.nsmallest(5, "ret")[["date", "ticker", "ret", "R", "stopped", "industry"]].to_dict("records")
    p = res["primary"]
    res["passed"] = bool(p["avg_trade"] > 0 and p["t_stat"] >= 2 and p["typical_year"] > 0 and p["liquidated_runs"] == 0)
    out_f.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
