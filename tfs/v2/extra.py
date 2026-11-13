"""Second batch of features per gapper, all known by 09:25 on the gap morning.

    python -m tfs.v2.extra        (offline: uses the cache; run tfs.v2.shorts first)

  price history   relative volume, prior run, distance to the 1-year high, overhead supply,
                  earlier spikes in the same stock and how the last one ended
  pre-market      where price sits against pre-market VWAP, when the high was made,
                  whether it is still climbing into the open
  market mood     how many gappers there were lately and how they ended
  short side      short interest, days to cover, daily short volume, fails-to-deliver
  SEC forms       resale registrations going effective, prospectuses, insider forms, late filings
"""
from __future__ import annotations

import json
from datetime import date as Date, timedelta

import numpy as np
import pandas as pd

from tfs.config import load_config, resolve
from tfs.v2.features import load_gappers, trading_days
from tfs.v2.shorts import Shorts

SPIKE = 1.40          # a day whose high is 40%+ above the previous close


def load_history(cache, days: list[str], tickers: set[str]) -> dict[str, pd.DataFrame]:
    parts = []
    for d in days:
        x = pd.read_parquet(cache / "grouped" / f"{d}.parquet", columns=["T", "o", "h", "l", "c", "v"])
        x = x[x["T"].isin(tickers)]
        parts.append(x.assign(date=d))
    allx = pd.concat(parts)
    return {t: x.sort_values("date").reset_index(drop=True) for t, x in allx.groupby("T")}


def history_features(h: pd.DataFrame | None, g: dict) -> dict:
    out = dict.fromkeys(["hist_days", "rvol_pm", "dollar_adv20", "ret_5d", "ret_20d", "pm_vs_hi250", "new_high_250",
                         "overhead_vol", "drawdown_60", "gap_in_atr", "prior_spikes_6m", "days_since_spike",
                         "last_spike_fade", "prev_vol_ratio", "green_streak", "tug_20"])
    if h is None:
        return out
    h = h[h["date"] < g["date"]]
    n = len(h)
    out["hist_days"] = n
    if n < 21:
        return out
    o, hi, lo, c, v = (h[k].to_numpy() for k in ("o", "h", "l", "c", "v"))
    adv = v[-21:-1].mean()
    out["rvol_pm"] = float(g["pm_volume_real"] * g["factor"] / adv) if adv > 0 else None
    out["dollar_adv20"] = float((v[-20:] * c[-20:]).mean())
    out["prev_vol_ratio"] = float(v[-1] / adv) if adv > 0 else None
    out["ret_5d"] = float(c[-1] / c[-6] - 1)
    out["ret_20d"] = float(c[-1] / c[-21] - 1)
    out["drawdown_60"] = float(c[-1] / c[-60:].max() - 1)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(hi - lo, np.maximum(abs(hi - pc), abs(lo - pc)))
    atr = tr[-14:].mean() / c[-1]
    out["gap_in_atr"] = float(g["pm_gap"] / atr) if atr > 0 else None
    k = 0
    while k < n - 1 and c[-1 - k] > c[-2 - k]:
        k += 1
    out["green_streak"] = k
    up_night, down_day = o[-20:] > pc[-20:], c[-20:] < o[-20:]
    out["tug_20"] = float((up_night & down_day).mean())
    spikes = np.flatnonzero(hi[1:] >= SPIKE * c[:-1]) + 1
    out["prior_spikes_6m"] = int((spikes >= n - 126).sum())
    if len(spikes):
        s = int(spikes[-1])
        out["days_since_spike"] = n - s
        if s + 3 < n:
            out["last_spike_fade"] = float(c[s + 3] / hi[s] - 1)
    if n >= 120:
        w = slice(-250, None)
        out["pm_vs_hi250"] = float(g["pm_last"] / hi[w].max() - 1)
        out["new_high_250"] = int(g["pm_high"] > hi[w].max())
        tot = v[w].sum()
        out["overhead_vol"] = float(v[w][c[w] > g["pm_last"]].sum() / tot) if tot > 0 else None
    return out


def premarket_features(bars: pd.DataFrame, g: dict, cutoff: str = "09:25") -> dict:
    out = dict.fromkeys(["pm_vwap_dist", "mins_since_pmh", "pmh_late", "pm_ret_30", "pm_vol_last30", "gap_age_min",
                         "pm_loc", "pm_bar_range", "pm_bars"])
    if bars.empty:
        return out
    day = bars.index[0].normalize()
    cut = day + pd.Timedelta(cutoff + ":00")
    pm = bars[bars.index < cut]
    if len(pm) < 5 or pm["v"].sum() <= 0:
        return out
    out["pm_bars"] = len(pm)
    vwap = float((pm["vw"] * pm["v"]).sum() / pm["v"].sum())
    last = float(pm["c"].iloc[-1])
    out["pm_vwap_dist"] = last / vwap - 1
    t_high = pm["h"].idxmax()
    out["mins_since_pmh"] = float((cut - t_high).total_seconds() / 60)
    out["pmh_late"] = int(t_high >= day + pd.Timedelta("9h"))
    before = pm[pm.index < cut - pd.Timedelta("30min")]
    if len(before):
        out["pm_ret_30"] = last / float(before["c"].iloc[-1]) - 1
    out["pm_vol_last30"] = float(pm["v"][pm.index >= cut - pd.Timedelta("30min")].sum() / pm["v"].sum())
    started = pm[pm["h"] >= 1.2 * g["prev_close"]]
    if len(started):
        out["gap_age_min"] = float((cut - started.index[0]).total_seconds() / 60)
        run = pm[pm.index >= started.index[0]]
        rng = run["h"].max() - run["l"].min()
        out["pm_loc"] = float((last - run["l"].min()) / rng) if rng > 0 else None
    tail = pm.tail(30)
    out["pm_bar_range"] = float(((tail["h"] - tail["l"]) / tail["c"]).median())
    return out


def sec_form_features(sub: dict, gap_date: str) -> dict:
    D = Date.fromisoformat(gap_date)
    fl = [f for f in sub.get("filings", []) if f.get("filingDate", "") < gap_date]

    def n(days, test):
        lo = (D - timedelta(days=days)).isoformat()
        return sum(1 for f in fl if f["filingDate"] >= lo and test(f["form"] or ""))
    sell = [f["filingDate"] for f in fl if (f["form"] or "").startswith(("424B", "S-1", "S-3", "F-1", "F-3"))]
    return {"effect_30d": int(n(30, lambda x: x == "EFFECT") > 0),
            "prospectus_30d": n(30, lambda x: x.startswith("424B")),
            "form4_90d": n(90, lambda x: x in ("4", "4/A")),
            "holder_13_12m": n(365, lambda x: x.startswith(("SC 13", "SCHEDULE 13"))),
            "late_filing_12m": int(n(365, lambda x: x.startswith("NT ")) > 0),
            "days_since_offer_doc": (D - Date.fromisoformat(max(sell))).days if sell else None}


def main():
    cfg = load_config()
    cache = resolve(cfg, cfg["data"]["cache_dir"])
    od = resolve(cfg, "output") / "v2"
    g = load_gappers(cfg)
    days = trading_days(cfg)
    hist = load_history(cache, days, set(g["ticker"]) | {"IWM"})
    feat = pd.read_parquet(od / "features.parquet")[["date", "ticker", "cik", "shares_out"]]
    cik = {(r.date, r.ticker): r.cik for r in feat.itertuples()}
    sh = Shorts(cache)
    splits_f = next(iter(cache.glob("splits_since_*.json")), None)
    splits = {}
    for r in (json.loads(splits_f.read_text()) if splits_f else []):
        splits.setdefault(r["ticker"], []).append(r)

    # market mood: gappers on earlier days only
    per_day = g.assign(oc=g["d_close"] / g["d_open"] - 1).groupby("date").agg(n=("ticker", "size"), oc=("oc", "mean"))
    idx = {d: i for i, d in enumerate(days)}
    iwm = hist.get("IWM")
    iwm_c = dict(zip(iwm["date"], iwm["c"])) if iwm is not None else {}

    rows = []
    for i, r in enumerate(g.to_dict("records")):
        if i % 400 == 0:
            print(f"  extra {i:,}/{len(g):,}", flush=True)
        row = {"date": r["date"], "ticker": r["ticker"]}
        row.update(history_features(hist.get(r["ticker"]), r))
        mf = cache / "minute" / r["date"] / f"{r['ticker']}.parquet"
        row.update(premarket_features(pd.read_parquet(mf) if mf.exists() else pd.DataFrame(), r,
                                      cfg["scan"]["premarket_cutoff"]))
        k = idx.get(r["date"], 0)
        prev5 = [d for d in days[max(0, k - 5):k]]
        p = per_day.reindex(prev5).dropna()
        row["n_gap_today"] = int(per_day.loc[r["date"], "n"])
        row["n_gap_prev5"] = int(p["n"].sum()) if len(p) else 0
        row["fade_prev5"] = float((p["oc"] * p["n"]).sum() / p["n"].sum()) if len(p) else None
        if k >= 6 and days[k - 1] in iwm_c and days[k - 6] in iwm_c:
            row["iwm_ret_5d"] = iwm_c[days[k - 1]] / iwm_c[days[k - 6]] - 1
        wd = Date.fromisoformat(r["date"]).weekday()
        row["monday"], row["friday"] = int(wd == 0), int(wd == 4)
        row.update(sh.features(r["ticker"], r["date"], r.get("shares_out"), splits.get(r["ticker"])))
        c = cik.get((r["date"], r["ticker"]))
        sf = cache / "sec" / "sub" / f"{c}.json" if c else None
        if sf is not None and sf.exists():
            row.update(sec_form_features(json.loads(sf.read_text()), r["date"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(od / "features2.parquet")
    print(f"EXTRA DONE {len(out):,} rows")
    print(out.drop(columns=["date", "ticker"]).describe().T[["count", "mean", "50%", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
