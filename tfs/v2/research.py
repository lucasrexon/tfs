"""TFS v2 research: find the best no-stop short method and what predicts extreme squeezes.

    python -m tfs.v2.research            (build period only; the sealed year is never read)

Steps
  1. simulate every entry x target x time limit on every gapper (paths.py)
  2. score each method:  account return = EV per trade x safe position size x trades
     where the safe size is the largest position no historical run-up would have liquidated
  3. squeeze study: which fundamentals, news and industries go with extreme squeezes
  4. train a squeeze-risk model on the first half, test it on the second half
  5. success study: what goes with hitting the profit target
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tfs.config import load_config, resolve
from tfs.v2.features import load_gappers
from tfs.v2.margin import f_max, survives
from tfs.v2.paths import simulate, squeeze_multiple
from tfs.v2.prep import next_days
from tfs.v2.stats2 import Logit, auc, correlate, cvar, lift_table, wilson

BINARY = ["runway_lt6", "shelf_3y", "s1_12m", "offering_90d", "deficiency_12m", "charter_amend_12m", "going_concern",
          "atm_24m", "reverse_split_12m", "morning_filing", "foreign", "china_hk", "foreign_filer", "has_revenue",
          "ssr_carry", "sub_dollar_before", "recent_ipo"]
NUMERIC = ["pm_gap", "pmh_ext", "pullback", "price", "pm_volume", "pm_dollar_vol", "shares_out", "float_rotation",
           "mcap_before", "runner_days", "days_since_ipo", "cash", "burn_per_month", "runway_months", "dilution_12m",
           "offerings_12m", "filings_30d", "dilution_pressure", "revenue_annual"]
GAP_LEVELS = (0.40, 0.50, 0.75, 1.00)
SQUEEZE_X = 2.0            # "extreme squeeze": price doubles from the 09:25 price within 5 days
MARGIN_RULE = "full"       # 100% requirement at any price: what brokers do to a squeezing small cap
SAFETY = 1.5               # size for a run-up 1.5x the worst one seen


def build_trades(cfg, g: pd.DataFrame, cache: Path, od: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tf, sf = od / "trades.parquet", od / "squeeze.parquet"
    if tf.exists() and sf.exists() and len(pd.read_parquet(sf)) == len(g):
        return pd.read_parquet(tf), pd.read_parquet(sf)
    from tfs.data.massive import MassiveClient
    cl = MassiveClient(cfg["data"]["base_url"], cache)
    if "next10" not in g:
        nd, _ = next_days(g, cache / "grouped")
        g = g.assign(next10=[nd.get((t, d), []) for t, d in zip(g.ticker, g.date)])
    rows, sq = [], []
    for i, r in enumerate(g.to_dict("records")):
        if i % 300 == 0:
            print(f"  paths {i:,}/{len(g):,}", flush=True)
        bars = cl.minute_bars(r["ticker"], r["date"])
        rows += simulate(r, bars, cfg)
        sq.append({"date": r["date"], "ticker": r["ticker"], "sq5": squeeze_multiple(r, bars, 5)})
    t, s = pd.DataFrame(rows), pd.DataFrame(sq)
    t.to_parquet(tf); s.to_parquet(sf)
    return t, s


def method_table(t: pd.DataFrame, years: float) -> pd.DataFrame:
    """One row per entry x target x horizon."""
    rows = []
    for (e, tg, hz), x in t.groupby(["entry", "target", "horizon"]):
        if len(x) < 40:
            continue
        r = x["ret"].to_numpy()
        fi = np.array([f_max(p, R * SAFETY, MARGIN_RULE) for p, R in zip(x["entry_px"], x["R"])])
        f_safe = float(fi.min())
        k = int(x["hit"].sum())
        rows.append({"entry": e, "target": tg, "horizon": hz, "n": len(x), "hit_rate": k / len(x),
                     "ev": float(r.mean()), "median": float(np.median(r)), "cvar5": cvar(r), "worst": float(r.min()),
                     "t_stat": float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if r.std() > 0 else np.nan,
                     "R_p95": float(x["R"].quantile(.95)), "R_p99": float(x["R"].quantile(.99)), "R_max": float(x["R"].max()),
                     "f_safe": f_safe, "reserve": 1 - f_safe,
                     "acct_per_trade": float(r.mean()) * f_safe,
                     "acct_per_year": float(r.mean()) * f_safe * len(x) / years})
    return pd.DataFrame(rows)


def account_sim(x: pd.DataFrame, f: float, start: float = 10000.0) -> dict:
    """One position at a time, fixed fraction f of current equity, no stop."""
    x = x.sort_values(["date", "entry_time"])
    eq, busy_until, n, liq, peak, dd = start, "", 0, None, start, 0.0
    for r in x.itertuples(index=False):
        if r.date <= busy_until:
            continue
        if not survives(r.entry_px, f, r.R, MARGIN_RULE):
            liq = f"{r.ticker} {r.date} (ran to {r.R:.1f}x)"
            break
        eq *= 1 + f * r.ret
        busy_until = r.exit_date
        n += 1
        peak = max(peak, eq); dd = min(dd, eq / peak - 1)
    return {"trades": n, "end": eq, "ret": eq / start - 1, "max_dd": dd, "liquidated": liq}


def main():
    cfg = load_config()
    cache = resolve(cfg, cfg["data"]["cache_dir"])
    od = resolve(cfg, "output") / "v2"
    be = cfg["split"]["build_end"]
    g = load_gappers(cfg)
    g = g[g["date"] <= be].reset_index(drop=True)          # the sealed year stops here
    print(f"Build-period gappers: {len(g):,} (lowest gap in file {g.pm_gap.min():.0%})")
    t, sq = build_trades(cfg, g, cache, od)
    feat = pd.read_parquet(od / "features.parquet")
    feat = feat[feat["date"] <= be].merge(sq, on=["date", "ticker"], how="left")
    feat["recent_ipo"] = (feat["days_since_ipo"] < 180).astype(float).where(feat["days_since_ipo"].notna())
    feat["squeeze"] = (feat["sq5"] >= SQUEEZE_X).astype(int)
    t = t.merge(feat[["date", "ticker", "pm_gap", "squeeze", "sq5"]], on=["date", "ticker"], how="left")
    days = sorted(g["date"].unique())
    years = max(0.25, (pd.Timestamp(days[-1]) - pd.Timestamp(days[0])).days / 365.25)
    mid = days[len(days) // 2]
    res: dict = {"build_start": days[0], "build_end": days[-1], "mid": mid, "gappers": len(g), "years": years,
                 "margin_rule": MARGIN_RULE, "safety": SAFETY, "squeeze_x": SQUEEZE_X}
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 80)

    # ---------- 1. methods, per gap level
    best_rows = []
    for lvl in GAP_LEVELS:
        tl = t[t["pm_gap"] >= lvl]
        if tl[["date", "ticker"]].drop_duplicates().shape[0] < 60:
            continue
        mt = method_table(tl, years)
        mt["gap_level"] = lvl
        best_rows.append(mt)
    methods = pd.concat(best_rows, ignore_index=True)
    methods.to_parquet(od / "methods.parquet")
    cols = ["gap_level", "entry", "target", "horizon", "n", "hit_rate", "ev", "cvar5", "t_stat", "R_max", "f_safe", "acct_per_year"]
    print("\n=== Top 15 methods by account return per year (safe size, all trades taken) ===")
    print(methods.sort_values("acct_per_year", ascending=False).head(15)[cols].round(4).to_string(index=False))
    print("\n=== Top 10 by EV per trade (before sizing) ===")
    print(methods.sort_values("ev", ascending=False).head(10)[cols].round(4).to_string(index=False))
    t20 = methods[(methods.target == 0.20)]
    print("\n=== Your 20% target: best per entry (gap >= 50%) ===")
    print(t20[t20.gap_level == 0.5].sort_values("acct_per_year", ascending=False).groupby("entry").head(1)[cols].round(4).to_string(index=False))

    # ---------- 2. walk-forward: choose on H1, confirm on H2
    h1, h2 = t[t["date"] < mid], t[t["date"] >= mid]
    m1, m2 = method_table(h1[h1.pm_gap >= .5], years / 2), method_table(h2[h2.pm_gap >= .5], years / 2)
    wf = m1.merge(m2, on=["entry", "target", "horizon"], suffixes=("_h1", "_h2"))
    top1 = wf.sort_values("acct_per_year_h1", ascending=False).head(10)
    print("\n=== Walk-forward (gap >= 50%): top 10 picked on first half, how they did on second half ===")
    print(top1[["entry", "target", "horizon", "n_h1", "ev_h1", "f_safe_h1", "acct_per_year_h1", "n_h2", "ev_h2", "R_max_h2", "acct_per_year_h2"]].round(4).to_string(index=False))
    rank_corr = wf[["ev_h1", "ev_h2"]].corr(method="spearman").iloc[0, 1]
    print(f"Rank correlation of EV between halves across all methods: {rank_corr:.2f}")
    res["wf_rank_corr"] = float(rank_corr)

    # ---------- 3. squeeze study
    fb = feat.copy()
    for c in ("catalyst", "industry"):
        if c in fb:
            for k, n in fb[c].value_counts().items():
                if n >= 15:
                    fb[f"{c}={k}"] = (fb[c] == k).astype(int)
    dummies = [c for c in fb.columns if "=" in c]
    feats = [c for c in BINARY + NUMERIC + dummies if c in fb.columns]
    binary = set(BINARY + dummies)
    print(f"\n=== Extreme squeezes: {fb.squeeze.sum()} of {len(fb)} gappers ({fb.squeeze.mean():.1%}) at least doubled from the 09:25 price within 5 days ===")
    cs = correlate(fb, feats, "squeeze", binary)
    cs.to_csv(od / "corr_squeeze.csv", index=False)
    print(cs.head(22).round(4).to_string(index=False))
    for c in ("catalyst", "industry"):
        if c in fb:
            lt = lift_table(fb, c, "squeeze")
            lt.to_csv(od / f"lift_squeeze_{c}.csv", index=False)
            print(f"\nSqueeze rate by {c}:"); print(lt.round(3).to_string(index=False))

    # ---------- 4. squeeze-risk model: fit on H1, test on H2
    X = fb[[c for c in feats if c in fb.columns]].copy()
    for c in ("pm_volume", "pm_dollar_vol", "shares_out", "mcap_before", "cash", "burn_per_month", "price", "revenue_annual"):
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    a, b = fb["date"] < mid, fb["date"] >= mid
    model = Logit(l2=5.0).fit(X[a], fb.loc[a, "squeeze"])
    fb["sq_risk"] = model.predict(X)
    a1, a2 = auc(fb.loc[a, "squeeze"], fb.loc[a, "sq_risk"]), auc(fb.loc[b, "squeeze"], fb.loc[b, "sq_risk"])
    print(f"\n=== Squeeze-risk model: AUC {a1:.2f} on the half it learned from, {a2:.2f} on the half it never saw (0.50 = useless, 1.00 = perfect) ===")
    print(model.weights().head(12).round(3).to_string())
    res.update(auc_h1=float(a1), auc_h2=float(a2), squeeze_rate=float(fb.squeeze.mean()), squeezes=int(fb.squeeze.sum()))
    oos = fb[b].copy()
    print("\nSecond half (unseen), skipping the riskiest X% by the model:")
    rows = []
    for skip in (0, .1, .2, .3, .5):
        cut = oos["sq_risk"].quantile(1 - skip) if skip else np.inf
        keep = oos[oos["sq_risk"] <= cut]
        rows.append({"skip": skip, "kept": len(keep), "squeezes_left": int(keep.squeeze.sum()),
                     "squeeze_rate": keep.squeeze.mean(), "worst_sq5": keep.sq5.max(), "p99_sq5": keep.sq5.quantile(.99)})
    sk = pd.DataFrame(rows); print(sk.round(3).to_string(index=False))
    sk.to_csv(od / "skip_oos.csv", index=False)
    fb[["date", "ticker", "sq5", "squeeze", "sq_risk"]].to_parquet(od / "sq_risk.parquet")

    # ---------- 5. success study for the 20% target
    pick = methods[(methods.gap_level == .5) & (methods.target == .20)].sort_values("acct_per_year", ascending=False).iloc[0]
    sel = t[(t.entry == pick.entry) & (t.target == .20) & (t.horizon == pick.horizon) & (t.pm_gap >= .5)]
    fs = fb.merge(sel[["date", "ticker", "hit", "ret", "R"]], on=["date", "ticker"])
    fs["success"] = fs["hit"].astype(int)
    print(f"\n=== What goes with hitting the 20% target (entry {pick.entry}, limit {int(pick.horizon)} days; {fs.success.mean():.0%} hit) ===")
    cx = correlate(fs, feats, "success", binary)
    cx.to_csv(od / "corr_success.csv", index=False)
    print(cx.head(15).round(4).to_string(index=False))
    for c in ("catalyst", "industry"):
        if c in fs:
            lt = lift_table(fs, c, "success"); lt.to_csv(od / f"lift_success_{c}.csv", index=False)
            print(f"\n20%-target hit rate by {c}:"); print(lt.round(3).to_string(index=False))

    # ---------- 6. account simulation of the pick, with and without the squeeze filter (second half only for the filter)
    print(f"\n=== $10,000 account, one position at a time, method: {pick.entry} / 20% target / {int(pick.horizon)}-day limit ===")
    for label, x, f in (("whole build year, safe size", sel, pick.f_safe),
                        ("whole build year, 2x safe size", sel, min(1, pick.f_safe * 2)),
                        ("whole build year, 5x safe size", sel, min(1, pick.f_safe * 5))):
        print(f"  {label} (position {f:.1%}): {account_sim(x, f)}")
    res["pick"] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in pick.to_dict().items()}
    (od / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print("\nRESEARCH DONE")


if __name__ == "__main__":
    main()
