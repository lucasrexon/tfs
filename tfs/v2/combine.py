"""Combinations of the ideas that worked, on the build year only.

    python -m tfs.v2.combine

  C1  09:25 short, cover at -20% or next close, stop 3x, 2.4% each; only companies with a
      going-concern warning; up to 1/2/3 positions; random pick or biggest pullback first
  C2  15:55 short, cover at -10% or next close, stop 2x; position sized on the FIRST half so
      the worst stop-out seen there costs about 7% of the account; up to 1/2/3 positions
  C3  C1 and C2 together, sharing one limit on open positions
Every number is shown for the first half, the second half (unseen when choosing), and the year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfs.config import load_config, resolve
from tfs.v2.ideas import LOSS_BUDGET, account, ev, pick

LIKELY_BORROWABLE = lambda x: (x.pm_gap < 2.0) & ~(x.float_rotation >= 3.0)


def main():
    cfg = load_config()
    od = resolve(cfg, "output") / "v2"
    be = cfg["split"]["build_end"]
    t = pd.read_parquet(od / "ideas_trades.parquet")
    feat = pd.read_parquet(od / "features.parquet").merge(pd.read_parquet(od / "features2.parquet"), on=["date", "ticker"])
    feat = feat[feat.date <= be]
    t = t.merge(feat[["date", "ticker", "pm_gap", "going_concern", "pullback", "float_rotation"]], on=["date", "ticker"])
    days = sorted(feat.date.unique())
    mid = days[len(days) // 2]
    years = (pd.Timestamp(days[-1]) - pd.Timestamp(days[0])).days / 365.25
    periods = (("first half", lambda x: x[x.date < mid], years / 2), ("second half", lambda x: x[x.date >= mid], years / 2),
               ("whole year", lambda x: x, years))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)

    a = pick(t, "pm_0925", 0.20, 1, stop=3.0)
    c1 = a[a.going_concern == 1].copy()
    c1["f"] = 0.024
    d = pick(t, "close_1555", 0.10, 1, stop=2.0).copy()
    h1 = d[d.date < mid]
    worst = h1.loc[h1.stopped, "R"].max()
    f_d = min(0.30, LOSS_BUDGET / (worst - 1))
    d["f"] = f_d
    print(f"C2 position from the first half: worst 2x stop-out filled at {worst:.2f}x, so {f_d:.1%} per trade")
    both = pd.concat([c1.assign(m="C1"), d.assign(m="C2")], ignore_index=True)

    rows = []
    for name, x in (("C1", c1), ("C2", d), ("C3 = C1 + C2", both)):
        for mp in (1, 2, 3):
            for order in ("random", "pullback"):
                if order == "pullback" and name != "C1":
                    continue
                for pn, cut, yrs in periods:
                    y = cut(x)
                    sc = y.pullback.fillna(0).to_numpy() if order == "pullback" else None
                    r = account(y, y.f.to_numpy(), yrs, max_pos=mp, score=sc)
                    m, ts = ev(y)
                    rows.append({"combo": name, "max_open": mp, "pick": order, "period": pn, "trades": r["taken"], "ev": m, "t": ts,
                                 "typical_yr": r["typical"], "bad_luck_yr": r["bad"], "worst_drop": r["dd"],
                                 "closest_to_liq": r["stress"], "liq": r["liq"]})
    res = pd.DataFrame(rows)
    res.to_csv(od / "combine.csv", index=False)
    print(res.round(3).to_string(index=False))

    print("\nChecks on the whole year (random pick, up to 3 open):")
    for name, x in (("C1", c1), ("C2", d), ("C3", both)):
        for extra in (0.0, 0.01, 0.03):
            r = account(x.assign(ret=x.ret - extra), x.f.to_numpy(), years, max_pos=3)
            print(f"  {name} extra cost {extra:.0%}/trade: typical {r['typical']:+.1%}, bad luck {r['bad']:+.1%}")
        e = x[LIKELY_BORROWABLE(x)]
        r = account(e, e.f.to_numpy(), years, max_pos=3)
        print(f"  {name} likely-borrowable only ({len(e)} of {len(x)}): EV {e.ret.mean():+.2%}, typical {r['typical']:+.1%}, bad luck {r['bad']:+.1%}")
        print(f"  {name} worst single trade {x.ret.min():+.0%} of the position = {(x.ret * x.f).min():+.1%} of the account; "
              f"stopped out {x.stopped.mean():.1%}")


if __name__ == "__main__":
    main()
