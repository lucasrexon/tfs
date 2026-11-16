"""News and industry patterns in the 09:25 short, labelled by significance, and filters built from them.

    python -m tfs.v2.patterns

Trades: the 09:25 short (cover at -20% or the next day's close, stop 3x) on every gapper up 50%+,
in both years (build Nov 2024-Nov 2025, test Nov 2025-Nov 2026). Gappers with and without a
going-concern warning are both included, so the warning itself can be compared too.

Significance labels (each group against all other trades):
  strong      false-discovery-adjusted q < 0.05 across every group tested, same direction in both years
  moderate    p < 0.05, same direction in both years
  weak        same direction in both years, p >= 0.05
  unreliable  the direction flips between the years
  too few     fewer than 15 trades, or fewer than 5 in one of the years

Filters are learned on one year and scored on the other (both ways round), so the reported gain is
what the rule did on trades it was not built from.
"""
from __future__ import annotations

import json
from math import erf, sqrt

import numpy as np
import pandas as pd

from tfs.config import load_config, resolve
from tfs.v2.ideas import account, pick

F, MAX_OPEN = 0.024, 3
MIN_N, MIN_YEAR = 15, 5


def bh_q(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1)
    return out


def p_two_sided(z: float) -> float:
    return float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))


def load(cfg) -> pd.DataFrame:
    od = resolve(cfg, "output") / "v2"
    be = cfg["split"]["build_end"]
    t = pd.read_parquet(od / "ideas_trades.parquet")
    t = t.merge(pd.read_parquet(od / "features.parquet")[["date", "ticker", "pm_gap"]], on=["date", "ticker"])
    b = pick(t, "pm_0925", 0.20, 1, stop=3.0)
    b = b[b.date <= be]
    s = pd.read_csv(od / "sealed_trades.csv")
    cols = ["date", "ticker", "entry", "target", "horizon", "stop_x", "hit", "stopped", "ret", "R", "entry_px", "exit_date"]
    x = pd.concat([b[cols].assign(year="build"), s[cols].assign(year="test")], ignore_index=True)
    f = pd.read_parquet(od / "features.parquet")[["date", "ticker", "industry", "going_concern", "pm_gap", "price", "float_rotation"]]
    n = pd.read_parquet(od / "news2.parquet")
    g = pd.read_parquet(od / "gappers40.parquet")[["date", "ticker", "name"]]
    x = x.merge(f, on=["date", "ticker"], how="left").merge(n, on=["date", "ticker"], how="left").merge(g, on=["date", "ticker"], how="left")
    x["going_concern"] = x.going_concern.fillna(0).astype(int).map({1: "warning", 0: "no warning"})
    x["news_0925"] = x.type_before_0925.fillna("none")
    x["news_known_later"] = x.type_any.fillna("none")
    x["news_timing"] = x.timing_before_0925.fillna("none")
    x["news_source"] = x.source_before_0925.fillna("none").map(
        lambda v: "none" if v == "none" else "SEC filing only" if v == "sec" else "news wire or media")
    x["older_news"] = np.where(x.type_prior_days.fillna("none").isin(["none"]), "no news in prior days", "news in prior days")
    x["any_news_0925"] = np.where(x.news_0925 == "none", "no news by 09:25", "news by 09:25")
    x["win"] = (x.ret > 0).astype(int)
    x["big_loss"] = (x.ret <= -0.5).astype(int)
    x["industry_x_news"] = x.industry.fillna("other") + " + " + np.where(x.news_0925 == "none", "no news", x.news_0925)
    return x.sort_values(["date", "ticker"]).reset_index(drop=True)


DIMS = {"industry": "industry", "news type (known by 09:25)": "news_0925", "any news by 09:25": "any_news_0925",
        "news timing": "news_timing", "news source": "news_source", "older news": "older_news",
        "going-concern warning": "going_concern", "industry + news type": "industry_x_news",
        "news type (incl. later that day, not tradable)": "news_known_later"}


def group_table(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim, col in DIMS.items():
        for val, gx in x.groupby(col):
            rest = x[x[col] != val]
            nb, nt = int((gx.year == "build").sum()), int((gx.year == "test").sum())
            r = {"dimension": dim, "group": val, "n": len(gx), "n_build": nb, "n_test": nt,
                 "avg_trade": gx.ret.mean(), "avg_rest": rest.ret.mean(), "win_rate": gx.win.mean(), "target_hit": gx.hit.mean(),
                 "big_loss_rate": gx.big_loss.mean(), "stopped": gx.stopped.mean(), "worst": gx.ret.min(),
                 "avg_build": gx[gx.year == "build"].ret.mean() if nb else np.nan,
                 "avg_test": gx[gx.year == "test"].ret.mean() if nt else np.nan}
            for yr in ("build", "test"):
                a, b = gx[gx.year == yr].ret, rest[rest.year == yr].ret
                r[f"diff_{yr}"] = a.mean() - b.mean() if len(a) else np.nan
            if len(gx) >= 2 and len(rest) >= 2:
                se = sqrt(gx.ret.var(ddof=1) / len(gx) + rest.ret.var(ddof=1) / len(rest))
                r["t"] = (gx.ret.mean() - rest.ret.mean()) / se if se > 0 else 0.0
                r["p"] = p_two_sided(r["t"])
            else:
                r["t"], r["p"] = np.nan, np.nan
            rows.append(r)
    t = pd.DataFrame(rows)
    ok = (t.n >= MIN_N) & (t.n_build >= MIN_YEAR) & (t.n_test >= MIN_YEAR) & ~t.dimension.str.contains("not tradable")
    t["q"] = np.nan
    t.loc[ok, "q"] = bh_q(t.loc[ok, "p"].fillna(1).to_numpy())
    same = np.sign(t.diff_build) == np.sign(t.diff_test)

    def lab(r, s):
        if r.n < MIN_N or r.n_build < MIN_YEAR or r.n_test < MIN_YEAR:
            return "too few"
        if not s:
            return "unreliable"
        if pd.notna(r.q) and r.q < 0.05:
            return "strong"
        if r.p < 0.05:
            return "moderate"
        return "weak"
    t["significance"] = [lab(r, s) for r, s in zip(t.itertuples(), same)]
    t["direction"] = np.where(t.avg_trade > t.avg_rest, "better than the rest", "worse than the rest")
    return t


def learn_skip(train: pd.DataFrame, cols: list[str], rule: str) -> dict[str, set]:
    """Groups to skip, learned on `train` only."""
    out = {}
    for col in cols:
        bad = set()
        for val, gx in train.groupby(col):
            if len(gx) < 10:
                continue
            rest = train[train[col] != val]
            se = sqrt(gx.ret.var(ddof=1) / len(gx) + rest.ret.var(ddof=1) / len(rest))
            z = (gx.ret.mean() - rest.ret.mean()) / se if se > 0 else 0
            if (rule == "avg_below_zero" and gx.ret.mean() < 0) or (rule == "clearly_worse" and z < -1.65):
                bad.add(val)
        out[col] = bad
    return out


def apply_skip(x: pd.DataFrame, skip: dict[str, set]) -> pd.DataFrame:
    m = np.ones(len(x), bool)
    for col, vals in skip.items():
        m &= ~x[col].isin(vals).to_numpy()
    return x[m]


def acct(x: pd.DataFrame) -> dict:
    years = max(0.25, (pd.Timestamp(x.date.max()) - pd.Timestamp(x.date.min())).days / 365.25) if len(x) else 1
    a = account(x, F, years, max_pos=MAX_OPEN, n_sims=300)
    return {"trades": a["taken"], "avg_trade": float(x.ret.mean()) if len(x) else np.nan, "typical_yr": a["typical"],
            "bad_luck_yr": a["bad"], "worst_drop": a["dd"], "closest_to_liq": a["stress"], "liq": a["liq"]}


def main():
    cfg = load_config()
    od = resolve(cfg, "output") / "v2"
    x = load(cfg)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 60)
    print(f"Trades: {len(x)} (build {int((x.year == 'build').sum())}, test {int((x.year == 'test').sum())})")

    gt = group_table(x)
    gt.to_csv(od / "patterns_groups.csv", index=False)
    show = ["dimension", "group", "n", "n_build", "n_test", "avg_trade", "avg_rest", "avg_build", "avg_test", "win_rate",
            "big_loss_rate", "t", "p", "q", "significance"]
    order = {"strong": 0, "moderate": 1, "weak": 2, "unreliable": 3, "too few": 4}
    for dim in DIMS:
        d = gt[gt.dimension == dim].copy()
        d["o"] = d.significance.map(order)
        d = d.sort_values(["o", "p"])
        if dim == "industry + news type":
            d = d[d.significance != "too few"]
        print(f"\n=== {dim} ===")
        print(d[show].round(3).to_string(index=False))

    # ---------------- filters: learn on one year, score on the other
    base_rules = {"current strategy (going-concern only)": lambda y: y[y.going_concern == "warning"],
                  "all 50%+ gappers": lambda y: y}
    learnable = {"industry": ["industry"], "news type": ["news_0925"], "industry and news type": ["industry", "news_0925"],
                 "news timing and source": ["news_timing", "news_source"],
                 "industry, news type, timing": ["industry", "news_0925", "news_timing"]}
    rows, learned = [], {}
    for bname, base in base_rules.items():
        for train_y, test_y in (("build", "test"), ("test", "build")):
            tr, te = base(x[x.year == train_y]), base(x[x.year == test_y])
            r0 = acct(te)
            rows.append({"start from": bname, "learned on": train_y, "scored on": test_y, "filter": "none", "skipped": "-", **r0})
            for lname, cols in learnable.items():
                for rule in ("avg_below_zero", "clearly_worse"):
                    sk = learn_skip(tr, cols, rule)
                    r1 = acct(apply_skip(te, sk))
                    rows.append({"start from": bname, "learned on": train_y, "scored on": test_y,
                                 "filter": f"skip {lname} groups: {rule.replace('_', ' ')}",
                                 "skipped": "; ".join(f"{c}: {sorted(v)}" for c, v in sk.items() if v) or "nothing", **r1})
    ft = pd.DataFrame(rows)
    ft.to_csv(od / "patterns_filters.csv", index=False)
    print("\n=== Filters learned on one year, scored on the other (2.4% per short, up to 3 open) ===")
    print(ft.drop(columns=["skipped"]).round(3).to_string(index=False))

    # which filters beat their own baseline in BOTH directions, on return AND risk
    base_r = ft[ft["filter"] == "none"].set_index(["start from", "scored on"])
    keep = []
    for (bname, filt), d in ft[ft["filter"] != "none"].groupby(["start from", "filter"]):
        ok_ret = ok_risk = True
        gains = []
        for r in d.itertuples():
            b0 = base_r.loc[(bname, r._3)]
            ok_ret &= r.typical_yr > b0.typical_yr
            ok_risk &= (r.bad_luck_yr >= b0.bad_luck_yr) and (r.worst_drop >= b0.worst_drop)
            gains.append((r.typical_yr - b0.typical_yr, r.bad_luck_yr - b0.bad_luck_yr, r.worst_drop - b0.worst_drop))
        keep.append({"start from": bname, "filter": filt, "higher return both ways": ok_ret, "less risk both ways": ok_risk,
                     "avg gain typical_yr": np.mean([g[0] for g in gains]), "avg gain bad_luck_yr": np.mean([g[1] for g in gains]),
                     "avg change worst_drop": np.mean([g[2] for g in gains])})
    kt = pd.DataFrame(keep).sort_values(["higher return both ways", "less risk both ways", "avg gain typical_yr"], ascending=False)
    kt.to_csv(od / "patterns_filter_verdict.csv", index=False)
    print("\n=== Verdict: does the filter raise return AND lower risk on the year it did not learn from, both ways round? ===")
    print(kt.round(3).to_string(index=False))

    # trade list for the report
    tl = x[["year", "date", "ticker", "name", "industry", "going_concern", "news_0925", "news_timing", "news_source",
            "headline_before_0925", "news_known_later", "headline_later_today", "pm_gap", "price", "hit", "stopped", "ret", "R"]]
    tl.to_csv(od / "patterns_trades.csv", index=False)
    print("\nPATTERNS DONE")


if __name__ == "__main__":
    main()
