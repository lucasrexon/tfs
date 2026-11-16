"""Third batch of factors per gapper (everything known by 09:25). Builds output/v2/features3.parquet.

    python -m tfs.v2.factors_data        (SEC calls are cached; first run takes about 20 minutes)

  dilution     warrants in the money at the 09:25 price and warrant overhang (XBRL); equity lines and
               convertible notes in filings (EDGAR full-text search); baby-shelf limit; IPO lockup window
  balance      negative / tiny equity, liabilities vs assets, cash vs market value, revenue growth, gross margin
  listing      days since an actual reverse split; exchange (Nasdaq / NYSE American / NYSE)
  pre-market   price below the low of the busiest pre-market minute; pre-market high just under a round
               number; a failed second test of the pre-market high (double top)
  daily chart  previous close vs its 20/50/200-day averages; RSI(14) at the previous close
  sector       other 40%+ gappers in the same industry that morning and in the 5 days before
"""
from __future__ import annotations

import json
from datetime import date as Date, timedelta

import numpy as np
import pandas as pd

from tfs.config import load_config, resolve
from tfs.sec import Edgar, FTS_QUERIES
from tfs.v2.extra import load_history
from tfs.v2.features import trading_days

FTS_QUERIES.update({
    "eloc": ('"equity purchase agreement"', "8-K,10-Q,10-K,S-1,424B3,6-K,20-F"),
    "standby_equity": ('"standby equity"', "8-K,10-Q,10-K,S-1,424B3,6-K,20-F"),
    "conv_note": ('"convertible promissory note"', "8-K,10-Q,10-K,6-K,20-F"),
})
CONCEPTS = {
    "w_price": ["ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1", "ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights"],
    "w_out": ["ClassOfWarrantOrRightOutstanding", "ClassOfWarrantOrRightNumberOfSecuritiesCalledByWarrantsOrRights"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liab": ["Liabilities"], "assets": ["Assets"],
    "gross": ["GrossProfit"],
    "rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenue", "SalesRevenueNet"],
    "pfloat": ["EntityPublicFloat"],
}
ROUND = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100])


def facts3(ed: Edgar, cik: str) -> dict:
    def fetch():
        j = ed._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json") or {}
        out = {}
        for tax, concepts in j.get("facts", {}).items():
            for name, body in concepts.items():
                for key, names in CONCEPTS.items():
                    if name in names:
                        for unit, vals in body.get("units", {}).items():
                            out.setdefault(key, []).extend({"val": v.get("val"), "start": v.get("start"), "end": v.get("end"),
                                                           "filed": v.get("filed"), "form": v.get("form"), "unit": unit}
                                                          for v in vals if (v.get("filed") or "") >= "2019-01-01")
        return out
    return ed._cached(f"facts3/{cik}.json", fetch)


def latest(vals, cutoff, duration=None):
    v = [x for x in vals or [] if x.get("filed") and x["filed"] < cutoff and x.get("val") is not None
         and (duration is None or bool(x.get("start")) == duration)]
    if not v:
        return None, []
    end = max(x["end"] for x in v)
    same = [x for x in v if x["end"] == end]
    return end, same


def split_adj(splits, after: str, until: str) -> float:
    """Multiply an old per-share price by this to put it on the gap day's share basis."""
    f = 1.0
    for s in splits or []:
        if after < s.get("execution_date", "") <= until:
            fr, to = float(s.get("split_from") or 1), float(s.get("split_to") or 1)
            if fr > 0 and to > 0:
                f *= fr / to
    return f


def months(x):
    return max(1.0, (Date.fromisoformat(x["end"]) - Date.fromisoformat(x["start"])).days / 30.44)


def sec_factors(ed, cik, r, splits) -> dict:
    out = {}
    fx = facts3(ed, cik)
    cut = r["date"]
    price = r["price"]
    end, w = latest(fx.get("w_price"), cut, duration=False)
    if w:
        adj = split_adj(splits, end, cut)
        px = min(float(x["val"]) for x in w if float(x["val"]) > 0) * adj if any(float(x["val"]) > 0 for x in w) else None
        if px:
            out["warrant_price_ratio"] = px / price                         # below 1: in the money at 09:25
            out["warrants_itm"] = int(px <= price)
            out["warrants_itm_by_gap"] = int(px <= price and px > r["prev_price"])  # the gap itself put them in the money
    end, w = latest(fx.get("w_out"), cut, duration=False)
    if w and r.get("shares_out"):
        out["warrant_overhang"] = max(float(x["val"]) for x in w) / split_adj(splits, end, cut) / r["shares_out"]
    _, e = latest(fx.get("equity"), cut, duration=False)
    if e:
        eq = float(e[0]["val"])
        out["equity"] = eq
        out["neg_equity"] = int(eq < 0)
        out["equity_lt_2_5m"] = int(eq < 2.5e6)
    _, l = latest(fx.get("liab"), cut, duration=False)
    _, a = latest(fx.get("assets"), cut, duration=False)
    if l and a and float(a[0]["val"]) > 0:
        out["liab_to_assets"] = float(l[0]["val"]) / float(a[0]["val"])
    revs = [x for x in fx.get("rev") or [] if x.get("filed") and x["filed"] < cut and x.get("start") and x.get("val") is not None]
    if revs:
        last = max(revs, key=lambda x: (x["end"], x["filed"]))
        m = months(last)
        prior = [x for x in revs if abs((Date.fromisoformat(x["end"]) - Date.fromisoformat(last["end"])).days + 365) < 25
                 and abs(months(x) - m) < 1.5]
        if prior and float(prior[0]["val"]) > 0:
            out["rev_growth"] = float(last["val"]) / float(prior[0]["val"]) - 1
        g = [x for x in fx.get("gross") or [] if x.get("end") == last["end"] and x.get("start") == last["start"]]
        if g and float(last["val"]) > 0:
            out["gross_margin"] = float(g[0]["val"]) / float(last["val"])
    _, pf = latest(fx.get("pfloat"), cut)
    if pf:
        out["public_float_sec"] = float(pf[0]["val"])
    for k in ("eloc", "standby_equity", "conv_note"):
        lo = (Date.fromisoformat(cut) - timedelta(days=365)).isoformat()
        out[f"{k}_12m"] = int(any(lo <= (h.get("file_date") or "") < cut for h in ed.fts(cik, k)))
    out["equity_line_12m"] = int(out["eloc_12m"] or out["standby_equity_12m"])
    return out


def premarket_shape(bars: pd.DataFrame, r: dict) -> dict:
    out = {}
    if bars.empty:
        return out
    day = bars.index[0].normalize()
    pm = bars[bars.index < day + pd.Timedelta("9h25m")]
    if len(pm) < 5:
        return out
    i = int(pm["v"].to_numpy().argmax())
    busy_low = float(pm["l"].iloc[i])
    out["below_climax_low"] = int(r["pm_last"] < busy_low)
    out["climax_low_dist"] = r["pm_last"] / busy_low - 1
    out["climax_minutes_ago"] = float((day + pd.Timedelta("9h25m") - pm.index[i]).total_seconds() / 60)
    real_high = r["pm_high"] * r["factor"]
    above = ROUND[ROUND >= real_high]
    if len(above):
        out["round_gap"] = above[0] / real_high - 1                    # how far below the next round number
        out["just_under_round"] = int(out["round_gap"] <= 0.03)
    h, lo = pm["h"].to_numpy(), pm["l"].to_numpy()
    pmh = r["pm_high"]
    touch = np.flatnonzero(h >= 0.98 * pmh)
    dt = 0
    if len(touch) >= 2:
        first = touch[0]
        later = touch[(touch - first) >= 10]
        if len(later) and lo[first:later[0]].min() <= 0.92 * pmh:
            dt = 1
    out["double_top"] = dt
    return out


def daily_trend(h: pd.DataFrame | None, date: str) -> dict:
    out = {}
    if h is None:
        return out
    c = h[h["date"] < date]["c"].to_numpy()
    if len(c) < 15:
        return out
    for n in (20, 50, 200):
        if len(c) >= n:
            out[f"vs_sma{n}"] = c[-1] / c[-n:].mean() - 1
    d = np.diff(c[-15:])
    up, dn = d[d > 0].sum() / 14, -d[d < 0].sum() / 14
    out["rsi14"] = 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)
    return out


def main():
    cfg = load_config()
    cache = resolve(cfg, cfg["data"]["cache_dir"])
    od = resolve(cfg, "output") / "v2"
    g = pd.read_parquet(od / "gappers40.parquet")
    f = pd.read_parquet(od / "features.parquet")
    g = g.merge(f[["date", "ticker", "cik", "price", "prev_price", "shares_out", "mcap_before", "shelf_3y", "days_since_ipo",
                   "cash", "industry"]], on=["date", "ticker"], suffixes=("", "_f"))
    if "shares_out_f" in g:
        g["shares_out"] = g["shares_out"].fillna(g["shares_out_f"])
    splits = {}
    for s in json.loads(next(iter(cache.glob("splits_since_*.json"))).read_text()):
        splits.setdefault(s["ticker"], []).append(s)
    days = trading_days(cfg)
    hist = load_history(cache, days, set(g["ticker"]))
    ed = Edgar(cache)
    per_day = g.groupby(["date", "industry"]).size()
    idx = {d: i for i, d in enumerate(days)}
    rows = []
    for i, r in enumerate(g.to_dict("records")):
        if i % 200 == 0:
            print(f"  factors {i:,}/{len(g):,} (SEC calls {ed.calls})", flush=True)
        row = {"date": r["date"], "ticker": r["ticker"]}
        if r.get("cik"):
            try:
                row.update(sec_factors(ed, r["cik"], r, splits.get(r["ticker"])))
            except Exception as e:
                row["sec3_error"] = str(e)[:80]
        mc = r.get("mcap_before")
        if mc:
            row["baby_shelf"] = int(bool(r.get("shelf_3y")) and mc < 75e6)
            row["gap_lifts_past_75m"] = int(mc < 75e6 and r["shares_out"] and r["shares_out"] * r["price"] >= 75e6)
            if r.get("cash") is not None and not pd.isna(r.get("cash")):
                row["cash_to_mcap"] = r["cash"] / mc
        dsi = r.get("days_since_ipo")
        row["lockup_window"] = int(dsi is not None and not pd.isna(dsi) and 150 <= dsi <= 200)
        rs = [s for s in splits.get(r["ticker"], []) if float(s.get("split_from") or 1) > float(s.get("split_to") or 1)
              and s.get("execution_date", "") <= r["date"]]
        if rs:
            last = max(s["execution_date"] for s in rs)
            row["days_since_rsplit"] = (Date.fromisoformat(r["date"]) - Date.fromisoformat(last)).days
        row["rsplit_30d"] = int(rs != [] and row["days_since_rsplit"] <= 30)
        det_f = cache / "details" / f"{r['ticker']}_{r['date']}.json"
        ex = json.loads(det_f.read_text()).get("primary_exchange") if det_f.exists() else None
        row["exchange"] = {"XNAS": "nasdaq", "XASE": "nyse_american", "XNYS": "nyse", "ARCX": "nyse_arca"}.get(ex, "other" if ex else None)
        mf = cache / "minute" / r["date"] / f"{r['ticker']}.parquet"
        if mf.exists():
            row.update(premarket_shape(pd.read_parquet(mf), r))
        row.update(daily_trend(hist.get(r["ticker"]), r["date"]))
        row["same_ind_today"] = int(per_day.get((r["date"], r["industry"]), 1) - 1)
        k = idx.get(r["date"], 0)
        row["same_ind_prev5"] = int(sum(per_day.get((d, r["industry"]), 0) for d in days[max(0, k - 5):k]))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(od / "features3.parquet")
    print("FACTORS DONE", len(out))
    print(out.drop(columns=["date", "ticker"]).describe(include="all").T[["count", "mean"]].to_string())


if __name__ == "__main__":
    main()
