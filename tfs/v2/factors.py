"""Test each chosen factor as a skip rule on top of the current bot rules, on a year it did not see.

    python -m tfs.v2.factors

Base: the current strategy (09:25 short, gap >= 50%, going-concern warning, not consumer / other
services, 2.4% per short, up to 3 open, cover at -20% or next close, stop 3x), in both years.

For each factor:
  * yes/no factors split trades into yes / no / unknown; number factors into thirds, with the
    cut points taken from the year the rule learns from
  * significance label per group (pooled over both years, same rules as tfs.v2.patterns)
  * skip rule learned on one year, scored on the other, both ways round:
      "avg below zero"  skip groups whose trades lost money on average (10+ trades)
      "clearly worse"   skip groups clearly worse than the rest (z < -1.65)
    PASS = higher typical year AND bad luck no worse AND worst drop no worse, in both directions.
With about 30 factors and 2 rule types, one or two may pass by luck; a pass is a lead, not proof.
"""
from __future__ import annotations

import json
from math import sqrt

import numpy as np
import pandas as pd

from tfs.config import load_config, resolve
from tfs.v2.patterns import acct, bh_q, load, p_two_sided

SKIP_IND = ("consumer", "other")
FACTORS = {  # name -> (column, kind, plain label, group)
    "warrants_itm": ("warrants_itm", "bin", "Warrants in the money at 9:25", "Dilution"),
    "warrants_itm_by_gap": ("warrants_itm_by_gap", "bin", "The gap put warrants in the money", "Dilution"),
    "warrant_overhang": ("warrant_overhang", "num", "Warrants outstanding vs shares", "Dilution"),
    "equity_line_12m": ("equity_line_12m", "bin", "Equity line in the last 12 months", "Dilution"),
    "conv_note_12m": ("conv_note_12m", "bin", "Convertible notes in the last 12 months", "Dilution"),
    "baby_shelf": ("baby_shelf", "bin", "Baby-shelf limited (shelf, under $75M)", "Dilution"),
    "gap_lifts_past_75m": ("gap_lifts_past_75m", "bin", "The gap lifts it past $75M (shelf limit off)", "Dilution"),
    "lockup_window": ("lockup_window", "bin", "IPO lockup ending (150-200 days after IPO)", "Dilution"),
    "neg_equity": ("neg_equity", "bin", "Negative stockholders' equity", "Balance sheet"),
    "equity_lt_2_5m": ("equity_lt_2_5m", "bin", "Equity under $2.5M (Nasdaq minimum)", "Balance sheet"),
    "liab_to_assets": ("liab_to_assets", "num", "Liabilities vs assets", "Balance sheet"),
    "cash_to_mcap": ("cash_to_mcap", "num", "Cash vs company value", "Balance sheet"),
    "rev_growth": ("rev_growth", "num", "Revenue growth vs a year earlier", "Balance sheet"),
    "gross_margin": ("gross_margin", "num", "Gross margin", "Balance sheet"),
    "rsplit_30d": ("rsplit_30d", "bin", "Reverse split in the last 30 days", "Listing"),
    "days_since_rsplit": ("days_since_rsplit", "num", "Days since the last reverse split", "Listing"),
    "exchange": ("exchange", "cat", "Exchange", "Listing"),
    "below_climax_low": ("below_climax_low", "bin", "Price below the busiest pre-market minute's low", "Technical"),
    "climax_low_dist": ("climax_low_dist", "num", "Distance from that low", "Technical"),
    "just_under_round": ("just_under_round", "bin", "Pre-market high just under a round number", "Technical"),
    "double_top": ("double_top", "bin", "Pre-market high tested twice and failed", "Technical"),
    "vs_sma20": ("vs_sma20", "num", "Previous close vs 20-day average", "Technical"),
    "vs_sma50": ("vs_sma50", "num", "Previous close vs 50-day average", "Technical"),
    "vs_sma200": ("vs_sma200", "num", "Previous close vs 200-day average", "Technical"),
    "rsi14": ("rsi14", "num", "RSI(14) at the previous close", "Technical"),
    "ah_share": ("ah_share", "num", "Share of the gap made after hours", "Technical"),
    "ah_move": ("ah_move", "num", "After-hours move", "Technical"),
    "same_ind_today": ("same_ind_today", "num", "Other gappers in the same industry today", "Sector"),
    "same_ind_prev5": ("same_ind_prev5", "num", "Same-industry gappers in the 5 days before", "Sector"),
    "sentiment": ("sentiment", "cat", "Massive news sentiment by 9:25", "News"),
}


def bins(train: pd.Series, s: pd.Series, kind: str) -> pd.Series:
    if kind == "num":
        v = train.dropna()
        if v.nunique() < 3:
            return s.map(lambda z: "unknown" if pd.isna(z) else ("high" if z > v.median() else "low"))
        lo, hi = v.quantile(1 / 3), v.quantile(2 / 3)
        return s.map(lambda z: "unknown" if pd.isna(z) else "low third" if z <= lo else "high third" if z > hi else "middle third")
    if kind == "bin":
        return s.map(lambda z: "unknown" if pd.isna(z) else ("yes" if z >= 0.5 else "no"))
    return s.fillna("unknown").astype(str)


def learn(train: pd.DataFrame, col: str, rule: str) -> set:
    bad = set()
    for val, gx in train.groupby(col):
        if len(gx) < 10:
            continue
        rest = train[train[col] != val]
        se = sqrt(gx.ret.var(ddof=1) / len(gx) + rest.ret.var(ddof=1) / len(rest)) if len(rest) > 1 else 0
        z = (gx.ret.mean() - rest.ret.mean()) / se if se > 0 else 0
        if (rule == "avg below zero" and gx.ret.mean() < 0) or (rule == "clearly worse" and z < -1.65):
            bad.add(val)
    return bad


def main():
    cfg = load_config()
    od = resolve(cfg, "output") / "v2"
    x = load(cfg)
    x = x[(x.going_concern == "warning") & ~x.industry.isin(SKIP_IND)].copy()
    f3 = pd.read_parquet(od / "features3.parquet")
    x = x.merge(f3, on=["date", "ticker"], how="left", suffixes=("", "_f3"))
    extra = od / "features4.parquet"                     # after-hours and sentiment, when downloaded
    if extra.exists():
        x = x.merge(pd.read_parquet(extra), on=["date", "ticker"], how="left")
    print(f"Base trades (current bot rules): {len(x)} (build {int((x.year == 'build').sum())}, test {int((x.year == 'test').sum())})")
    base = {y: acct(x[x.year == y]) for y in ("build", "test")}
    for y, a in base.items():
        print(f"  {y}: typical {a['typical_yr']:+.1%}, bad luck {a['bad_luck_yr']:+.1%}, worst drop {a['worst_drop']:.1%}, avg trade {a['avg_trade']:+.2%}")

    groups, tests = [], []
    for name, (col, kind, label, fam) in FACTORS.items():
        if col not in x or x[col].notna().sum() < 30:
            print(f"  {name}: not enough data yet")
            continue
        cover = float(x[col].notna().mean())
        g_all = bins(x[col], x[col], kind)                                 # pooled bins for the labels
        for val in sorted(g_all.unique()):
            m = g_all == val
            gx, rest = x[m], x[~m]
            nb, nt = int((gx.year == "build").sum()), int((gx.year == "test").sum())
            d = {y: (gx[gx.year == y].ret.mean() - rest[rest.year == y].ret.mean()) if (gx.year == y).any() else np.nan
                 for y in ("build", "test")}
            se = sqrt(gx.ret.var(ddof=1) / len(gx) + rest.ret.var(ddof=1) / len(rest)) if len(gx) > 1 and len(rest) > 1 else np.nan
            tt = (gx.ret.mean() - rest.ret.mean()) / se if se and se > 0 else np.nan
            groups.append({"factor": name, "label": label, "family": fam, "coverage": cover, "group": val, "n": len(gx), "n_build": nb,
                           "n_test": nt, "avg_trade": gx.ret.mean(), "avg_rest": rest.ret.mean(),
                           "avg_build": gx[gx.year == "build"].ret.mean(), "avg_test": gx[gx.year == "test"].ret.mean(),
                           "win_rate": (gx.ret > 0).mean(), "big_loss_rate": (gx.ret <= -0.5).mean(), "t": tt,
                           "p": p_two_sided(tt) if not np.isnan(tt) else np.nan, "same_dir": bool(np.sign(d["build"]) == np.sign(d["test"]))})
        for rule in ("avg below zero", "clearly worse"):
            res = {"factor": name, "label": label, "family": fam, "rule": rule}
            ok_ret = ok_risk = True
            for tr_y, te_y in (("build", "test"), ("test", "build")):
                tr, te = x[x.year == tr_y].copy(), x[x.year == te_y].copy()
                tr["_g"], te["_g"] = bins(tr[col], tr[col], kind), bins(tr[col], te[col], kind)
                sk = learn(tr, "_g", rule)
                a = acct(te[~te["_g"].isin(sk)]) if sk else base[te_y]
                b0 = base[te_y]
                res[f"skip_learned_on_{tr_y}"] = ", ".join(sorted(sk)) or "nothing"
                res[f"{te_y}_typical"], res[f"{te_y}_bad"], res[f"{te_y}_dd"] = a["typical_yr"], a["bad_luck_yr"], a["worst_drop"]
                res[f"{te_y}_trades"] = a["trades"]
                ok_ret &= bool(sk) and a["typical_yr"] > b0["typical_yr"]
                ok_risk &= bool(sk) and a["bad_luck_yr"] >= b0["bad_luck_yr"] and a["worst_drop"] >= b0["worst_drop"]
            res["passed"] = bool(ok_ret and ok_risk)
            res["higher_return_both"], res["less_risk_both"] = bool(ok_ret), bool(ok_risk)
            tests.append(res)
    gt = pd.DataFrame(groups)
    ok = (gt.n >= 15) & (gt.n_build >= 5) & (gt.n_test >= 5)
    gt["q"] = np.nan
    gt.loc[ok, "q"] = bh_q(gt.loc[ok, "p"].fillna(1).to_numpy())

    def lab(r):
        if not (r.n >= 15 and r.n_build >= 5 and r.n_test >= 5):
            return "too few"
        if not r.same_dir:
            return "unreliable"
        if pd.notna(r.q) and r.q < 0.05:
            return "strong"
        return "moderate" if r.p < 0.05 else "weak"
    gt["significance"] = gt.apply(lab, axis=1)
    tt = pd.DataFrame(tests)
    gt.to_csv(od / "factors_groups.csv", index=False)
    tt.to_csv(od / "factors_tests.csv", index=False)
    json.dump({"base": base, "n": len(x)}, open(od / "factors_base.json", "w"), default=float)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 40)
    print("\n=== Groups (pooled over both years) ===")
    print(gt[["factor", "group", "coverage", "n", "n_build", "n_test", "avg_trade", "avg_rest", "avg_build", "avg_test", "win_rate",
              "big_loss_rate", "p", "q", "significance"]].round(3).to_string(index=False))
    print("\n=== Skip rules on the unseen year ===")
    print(tt[["factor", "rule", "skip_learned_on_build", "test_typical", "test_bad", "test_dd", "skip_learned_on_test", "build_typical",
              "build_bad", "build_dd", "higher_return_both", "less_risk_both", "passed"]].round(3).to_string(index=False))
    print("\nPassed:", tt[tt.passed][["factor", "rule"]].values.tolist())
    print("FACTORS TEST DONE")


if __name__ == "__main__":
    main()
