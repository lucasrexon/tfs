"""Ideas tested on the build year, ordered by how much each one changes the strategy.

    python -m tfs.v2.ideas            (offline; the sealed year is never read)

Baseline (chosen 2026-11-13): each short is 2.4% of the account, disaster stop at 3x the
entry price, one position at a time, gap >= 50%.
    A  short at 09:25, cover at -20% or the next day's close
    D  short at 15:55, cover at -10% or the next day's close

    1 filters     skip some gappers
    2 settings    stop distance, target, time limit, gap size
    3 selection   which gapper to take when there are several
    4 sizing      smaller positions on risky gappers
    5 portfolio   several positions at once; both methods together
    6 entry       a different entry under the same risk rules
    7 other       a normal (tight) stop; buying instead of shorting

Every choice is made on the first half of the year and checked on the second half.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfs.backtest import nights_between
from tfs.config import load_config, resolve
from tfs.data.massive import MassiveClient
from tfs.rules import session
from tfs.v2.features import load_gappers
from tfs.v2.paths import _costs, _end, _entries, _series
from tfs.v2.stats2 import Logit

ENTRIES = ("pm_0925", "close_1555", "open", "ext_+20%", "red_1030", "t_1400")
STOPS = (1.15, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 0.0)          # 0 = no stop
TARGETS = (0.10, 0.15, 0.20, 0.25, 0.30)
HORIZONS = (0, 1, 3)
F = 0.024
STOP = 3.0
LOSS_BUDGET = F * 2.9            # what the worst 3x stop-out seen costs the account (fill at 3.9x)


# ------------------------------------------------------------------ simulation
def stop_grid(g: dict, bars: pd.DataFrame, cfg: dict) -> tuple[list[dict], dict]:
    """Every entry x target x time limit x stop distance for one gapper, plus what is known at 15:55."""
    cc = cfg["costs"]
    reg = session(bars)
    if reg.empty:
        return [], {}
    nxt = [dict(d) for d in (g.get("next10") if g.get("next10") is not None else [])]
    day = reg.index[0].normalize()
    eod_cut = day + pd.Timedelta(cfg["rules"]["eod_exit"] + ":00")
    arr = lambda k: np.array([d[k] for d in nxt])
    dates = [d["date"] for d in nxt]
    rows, d1 = [], {}
    for c in _entries(g, bars, cfg):
        if c["entry"] not in ENTRIES:
            continue
        s = _series(c, g["date"], eod_cut, arr("o"), arr("h"), arr("l"), arr("c"), dates)
        O, H, L = s["O"], s["H"], s["L"]
        px = c["px"]
        cmax, cmin = np.maximum.accumulate(H), np.minimum.accumulate(L)
        if c["entry"] == "close_1555":
            before = reg.iloc[:c["i0"]]
            if len(before):
                hi, lo, op = before["h"].max(), before["l"].min(), before["o"].iloc[0]
                d1 = {"d1_ret_open": px / op - 1, "d1_vs_pm": px / g["pm_last"] - 1,
                      "d1_loc": (px - lo) / (hi - lo) if hi > lo else None, "d1_high_x": hi / op}
        for hz in HORIZONS:
            if hz == 0 and c.get("no_same_day"):
                continue
            e = _end(s, hz, True)
            if e is None:
                continue
            end_i, end_px, _ = e
            n = end_i + 1
            for tg in TARGETS:
                tpx = px * (1 - tg)
                ti = int(np.searchsorted(-cmin[:n], -tpx))            # first bar whose low reaches the target
                for k in STOPS:
                    si = int(np.searchsorted(cmax[:n], px * k)) if k else n
                    if si < n and si <= ti:                            # same bar: assume the stop came first
                        i, hit, stopped = si, False, True
                        x = max(px * k, O[i] if i > 0 else px * k) * (1 + cc["stop_extra_slippage_pct"])
                        r = x / px
                    elif ti < n:
                        i, hit, stopped = ti, True, False
                        x = float(O[i]) if (O[i] <= tpx and i > 0) else tpx
                        r = max(cmax[i], px) / px
                    else:
                        i, hit, stopped, x = end_i, False, False, end_px
                        r = max(cmax[i], px) / px
                    nights = nights_between(s["entry_date"], s["step_date"][i])
                    rows.append((g["date"], g["ticker"], c["entry"], tg, hz, k, hit, stopped,
                                 _costs(cc, px * g["factor"], x * g["factor"], nights), float(r), px * g["factor"],
                                 s["step_date"][i]))
    return rows, d1


COLS = ["date", "ticker", "entry", "target", "horizon", "stop_x", "hit", "stopped", "ret", "R", "entry_px", "exit_date"]


def long_breakout(g: dict, bars: pd.DataFrame, cfg: dict, window_end: str = "10:00") -> list[dict]:
    """Buy the break of the pre-market high in the first half hour; stop below, target above, out by 15:55."""
    cc = cfg["costs"]
    reg = session(bars)
    if reg.empty:
        return []
    day = reg.index[0].normalize()
    o, h, low, c = (reg[k].to_numpy() for k in ("o", "h", "l", "c"))
    until = int(np.searchsorted(reg.index, day + pd.Timedelta(window_end + ":00")))
    brk = np.flatnonzero(h[:until] >= g["pm_high"])
    if not len(brk):
        return []
    i = int(brk[0])
    px = max(g["pm_high"], o[i])
    eod = min(int(np.searchsorted(reg.index, day + pd.Timedelta(cfg["rules"]["eod_exit"] + ":00"))), len(reg) - 1)
    rows = []
    for tg in (0.20, 0.50, 0.0):
        for st in (0.10, 0.20):
            s_hit = np.flatnonzero(low[i:eod + 1] <= px * (1 - st))          # the fill bar's low counts against us
            t_hit = np.flatnonzero(h[i + 1:eod + 1] >= px * (1 + tg)) if tg else np.array([], int)
            si = i + int(s_hit[0]) if len(s_hit) else None
            ti = i + 1 + int(t_hit[0]) if len(t_hit) else None
            if si is not None and (ti is None or si <= ti):
                x = min(px * (1 - st), o[si] if si > i else px) * (1 - cc["stop_extra_slippage_pct"])
            elif ti is not None:
                x = px * (1 + tg)
            else:
                x = o[eod]
            E, X = px * g["factor"], x * g["factor"]
            se = max(cc["min_slippage"], cc["slippage_pct"] * E); sx = max(cc["min_slippage"], cc["slippage_pct"] * X)
            rows.append({"date": g["date"], "ticker": g["ticker"], "target": tg, "stop": st,
                         "ret": ((X - sx) - (E + se) - 2 * cc["commission_per_share"]) / E})
    return rows


# ------------------------------------------------------------------ account
def f_liq(p0, r):
    """Largest position that survives a run-up to r x entry at 100% margin (and $2.50/share under $2.50)."""
    return 1 / (r - 1 + np.maximum(r, 2.5 / p0))


def account(x: pd.DataFrame, f, years: float, max_pos: int = 1, score=None, n_sims: int = 200, seed: int = 11) -> dict:
    """Fixed fraction per trade, up to max_pos positions at once. Which gapper gets taken when
    there are too many is random (many simulations) unless `score` ranks them (higher first).
    Liquidation check is the harsh one: every open position hits its worst price at the same time."""
    if x.empty:
        return {"taken": 0, "typical": 0.0, "bad": 0.0, "liq": 0.0, "dd": 0.0, "stress": 0.0}
    d = pd.to_datetime(x["date"]).to_numpy().astype("datetime64[D]").astype(int)
    e = pd.to_datetime(x["exit_date"]).to_numpy().astype("datetime64[D]").astype(int)
    ret = x["ret"].to_numpy()
    fi = np.broadcast_to(np.asarray(f, float), ret.shape)
    stress = fi / f_liq(x["entry_px"].to_numpy(), x["R"].to_numpy())
    rng = np.random.default_rng(seed)
    if score is not None:
        n_sims = 1
    outs, liqs, taken, dds, peak_stress = [], 0, [], [], 0.0
    for _ in range(n_sims):
        key = rng.random(len(d)) if score is None else -np.asarray(score, float)
        eq, n, peak, dd, dead, open_ = 1.0, 0, 1.0, 0.0, False, []
        for i in np.lexsort((key, d)):
            open_ = [p for p in open_ if p[0] >= d[i]]
            if len(open_) >= max_pos:
                continue
            tot = stress[i] + sum(p[1] for p in open_)
            peak_stress = max(peak_stress, tot)
            if tot >= 1:
                dead = True
                break
            eq *= 1 + fi[i] * ret[i]
            open_.append((e[i], stress[i]))
            n += 1
            peak = max(peak, eq)
            dd = min(dd, eq / peak - 1)
        liqs += dead
        outs.append(-1.0 if dead else eq - 1)
        taken.append(n); dds.append(dd)
    ann = lambda r: (1 + r) ** (1 / years) - 1 if r > -1 else -1.0
    return {"taken": int(np.median(taken)), "typical": ann(float(np.median(outs))), "bad": ann(float(np.percentile(outs, 5))),
            "liq": liqs / n_sims, "dd": float(np.median(dds)), "stress": float(peak_stress)}


def ev(x: pd.DataFrame) -> tuple[float, float]:
    r = x["ret"].to_numpy()
    if len(r) < 2 or r.std() == 0:
        return (float(r.mean()) if len(r) else np.nan), np.nan
    return float(r.mean()), float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


def pick(t: pd.DataFrame, entry: str, target: float, horizon: int, stop: float = STOP, gap: float = 0.5) -> pd.DataFrame:
    return t[(t.entry == entry) & np.isclose(t.target, target) & (t.horizon == horizon) & np.isclose(t.stop_x, stop)
             & (t.pm_gap >= gap)]


def line(label: str, x: pd.DataFrame, years: float, f=F, **kw) -> dict:
    m, ts = ev(x)
    a = account(x, f, years, **kw)
    return {"what": label, "n": len(x), "hit": x.hit.mean() if len(x) else np.nan, "stopped": x.stopped.mean() if len(x) else np.nan,
            "ev": m, "t": ts, "taken": a["taken"], "typical_yr": a["typical"], "bad_luck_yr": a["bad"], "worst_drop": a["dd"],
            "liq": a["liq"]}


def show(title: str, rows: list[dict], od=None, name: str | None = None):
    df = pd.DataFrame(rows)
    print(f"\n=== {title} ===")
    print(df.round(3).to_string(index=False))
    if od is not None and name:
        df.to_csv(od / f"ideas_{name}.csv", index=False)
    return df


# ------------------------------------------------------------------ main
def main():
    cfg = load_config()
    cache = resolve(cfg, cfg["data"]["cache_dir"])
    od = resolve(cfg, "output") / "v2"
    be = cfg["split"]["build_end"]
    g = load_gappers(cfg)
    g = g[g["date"] <= be].reset_index(drop=True)              # the sealed year stops here
    days = sorted(g["date"].unique())
    mid = days[len(days) // 2]
    years = (pd.Timestamp(days[-1]) - pd.Timestamp(days[0])).days / 365.25
    hy = years / 2
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)

    tf, lf, df_ = od / "ideas_trades.parquet", od / "ideas_long.parquet", od / "ideas_d1.parquet"
    if tf.exists() and lf.exists() and df_.exists():
        t, lg, d1 = pd.read_parquet(tf), pd.read_parquet(lf), pd.read_parquet(df_)
    else:
        cl = MassiveClient(cfg["data"]["base_url"], cache)
        rows, lrows, d1rows = [], [], []
        for i, r in enumerate(g.to_dict("records")):
            if i % 300 == 0:
                print(f"  grid {i:,}/{len(g):,}", flush=True)
            bars = cl.minute_bars(r["ticker"], r["date"])
            rr, dd = stop_grid(r, bars, cfg)
            rows += rr
            d1rows.append({"date": r["date"], "ticker": r["ticker"], **dd})
            lrows += long_breakout(r, bars, cfg)
        t, lg, d1 = pd.DataFrame(rows, columns=COLS), pd.DataFrame(lrows), pd.DataFrame(d1rows)
        t.to_parquet(tf); lg.to_parquet(lf); d1.to_parquet(df_)

    feat = pd.read_parquet(od / "features.parquet").merge(pd.read_parquet(od / "features2.parquet"), on=["date", "ticker"])
    feat = feat[feat["date"] <= be].merge(d1, on=["date", "ticker"], how="left")
    t = t.merge(feat[["date", "ticker", "pm_gap"]], on=["date", "ticker"])
    lg = lg.merge(feat[["date", "ticker", "pm_gap"]], on=["date", "ticker"])
    base = {"A": pick(t, "pm_0925", 0.20, 1), "D": pick(t, "close_1555", 0.10, 1)}
    name = {"A": "A: short 09:25, -20% or next close", "D": "D: short 15:55, -10% or next close"}
    H1 = lambda x: x[x["date"] < mid]
    H2 = lambda x: x[x["date"] >= mid]

    print(f"Build year {days[0]} to {days[-1]}, split at {mid}. Position {F:.1%}, disaster stop {STOP:.0f}x, one position at a time.")
    show("0. Baseline", [line(name[k], x, years) for k, x in base.items()]
         + [line(name[k] + " [first half]", H1(x), hy) for k, x in base.items()]
         + [line(name[k] + " [second half]", H2(x), hy) for k, x in base.items()], od, "0_baseline")

    # ---------------------------------------------------------------- 1. filters
    q = lambda col, p: H1(feat)[col].quantile(p)
    fl = {
        "pre-market high after 09:00": feat.pmh_late == 1,
        "still rising 5%+ in last 30 min": feat.pm_ret_30 >= 0.05,
        "merger or buyout news": feat.catalyst == "merger_buyout",
        "any news found": feat.catalyst != "no_news_found",
        "SEC filing that morning": feat.morning_filing == 1,
        "crypto company": feat.industry == "crypto",
        "days to cover 3+": feat.days_to_cover >= 3,
        "calm pre-market (bottom third bar range)": feat.pm_bar_range <= q("pm_bar_range", 1 / 3),
        "bigger company (top third market cap)": feat.mcap_before >= q("mcap_before", 2 / 3),
        "low float rotation (bottom third)": feat.float_rotation <= q("float_rotation", 1 / 3),
        "low relative volume (bottom third)": feat.rvol_pm <= q("rvol_pm", 1 / 3),
        "no going-concern warning": feat.going_concern == 0,
        "within 20% of 1-year high": feat.pm_vs_hi250 >= -0.20,
        "gap under 75%": feat.pm_gap < 0.75,
    }
    fl_d = {
        "15:55 price above the open (green day)": feat.d1_ret_open > 0,
        "15:55 price in top 30% of day's range": feat.d1_loc >= 0.7,
        "15:55 price 20%+ above 09:25 price": feat.d1_vs_pm >= 0.20,
        "day high 50%+ above the open": feat.d1_high_x >= 1.5,
        "15:55 price 30%+ below the open (already faded)": feat.d1_ret_open <= -0.30,
    }
    keyed = feat.set_index(["date", "ticker"])
    consistent = {}
    for k, x in base.items():
        rows = []
        flags = {**fl, **(fl_d if k == "D" else {})}
        ix = pd.MultiIndex.from_frame(x[["date", "ticker"]])
        for nm, m in flags.items():
            mk = pd.Series(m.fillna(False).to_numpy(), index=keyed.index).reindex(ix).to_numpy()
            keep, skip = x[~mk], x[mk]
            if len(skip) < 25:
                continue
            (k1, _), (s1, _), (k2, _), (s2, _) = ev(H1(keep)), ev(H1(skip)), ev(H2(keep)), ev(H2(skip))
            a, b = keep.ret.to_numpy(), skip.ret.to_numpy()
            welch = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            ok = bool(k1 > s1 and k2 > s2 and welch >= 2)
            acc = account(keep, F, years)
            rows.append({"skip if": nm, "n_skip": len(skip), "ev_skip": b.mean(), "ev_keep": a.mean(), "t_diff": welch,
                         "h1_keep": k1, "h1_skip": s1, "h2_keep": k2, "h2_skip": s2, "both_halves": ok,
                         "typical_yr": acc["typical"], "bad_luck_yr": acc["bad"]})
            if ok:
                consistent.setdefault(k, {})[nm] = mk
        show(f"1. Filters on {name[k]} (baseline typical {account(x, F, years)['typical']:+.1%}/yr)",
             sorted(rows, key=lambda r: -r["t_diff"]), od, f"1_filters_{k}")
    filt = {}
    rows = []
    for k, x in base.items():
        ms = list(consistent.get(k, {}).values())
        mk = np.any(ms, axis=0) if ms else np.zeros(len(x), bool)
        filt[k] = mk
        print(f"  {k}: filters that helped in both halves: {list(consistent.get(k, {})) or 'none'}")
        for lab, y, yrs in ((" all year", x, years), (" first half", H1(x), hy), (" second half", H2(x), hy)):
            sel = mk[np.isin(x.index, y.index)]
            rows.append(line(f"{k} unfiltered{lab}", y, yrs))
            rows.append(line(f"{k} filtered{lab}", y[~sel], yrs))
    show("1b. All consistent filters together (chosen with both halves, so the full-year gain is flattering)", rows, od, "1b_combined")

    # ---------------------------------------------------------------- 2. settings
    spec = {"A": ("pm_0925", 0.20, 1), "D": ("close_1555", 0.10, 1)}
    for k, (en, tg, hz) in spec.items():
        rows = []
        var = [("stop", s, pick(t, en, tg, hz, stop=s)) for s in STOPS] + \
              [("target", x_, pick(t, en, x_, hz)) for x_ in TARGETS] + \
              [("limit_days", h, pick(t, en, tg, h)) for h in HORIZONS if not (h == 0 and en == "close_1555")] + \
              [("min_gap", gp, pick(t, en, tg, hz, gap=gp)) for gp in (0.40, 0.50, 0.75, 1.00)]
        for what, val, x in var:
            m1, m2 = ev(H1(x))[0], ev(H2(x))[0]
            a = account(x, F, years)
            rows.append({"setting": what, "value": ("none" if (what == "stop" and val == 0) else val), "n": len(x), "hit": x.hit.mean(),
                         "stopped": x.stopped.mean(), "worst_trade": x.ret.min(), "ev": x.ret.mean(), "ev_h1": m1, "ev_h2": m2,
                         "typical_yr": a["typical"], "bad_luck_yr": a["bad"], "worst_drop": a["dd"],
                         "chosen": bool((what, val) in (("stop", STOP), ("target", tg), ("limit_days", hz), ("min_gap", 0.5)))})
        show(f"2. One setting at a time, {name[k]} (position {F:.1%} throughout)", rows, od, f"2_settings_{k}")
    rows = []
    for k, (en, tg0, hz0) in spec.items():                                 # tune all settings on H1, check on H2
        best = None
        for tg in TARGETS:
            for hz in HORIZONS:
                for s in (2.0, 2.5, 3.0, 4.0, 5.0):
                    x = pick(t, en, tg, hz, stop=s)
                    if len(x) < 100:
                        continue
                    a1 = account(H1(x), F, hy, n_sims=60)["typical"]
                    if best is None or a1 > best[0]:
                        best = (a1, tg, hz, s)
        _, tg, hz, s = best
        rows.append(line(f"{k} chosen settings, second half", H2(base[k]), hy))
        rows.append(line(f"{k} tuned on first half (target {tg:.0%}, limit {hz}d, stop {s}x), second half", H2(pick(t, en, tg, hz, stop=s)), hy))
    show("2b. Tuning every setting on the first half, then checking the second half", rows, od, "2b_tuned")

    # ---------------------------------------------------------------- 3. selection
    from tfs.v2.research import BINARY, NUMERIC
    fcols = [c for c in BINARY + NUMERIC if c in feat.columns]
    for k, x in base.items():
        xf = x.merge(feat, on=["date", "ticker"], suffixes=("", "_f"))
        scores = {"biggest gap": xf.pm_gap, "highest float rotation": xf.float_rotation, "biggest pullback from pre-market high": xf.pullback,
                  "choppiest pre-market": xf.pm_bar_range, "smallest company": -xf.mcap_before, "no news first": (xf.catalyst == "no_news_found").astype(float),
                  "most fails-to-deliver": xf.ftd_pct_so, "fewest days to cover": -xf.days_to_cover,
                  "pre-market high made earliest": xf.mins_since_pmh, "highest relative volume": xf.rvol_pm}
        if k == "D":
            scores.update({"faded most by 15:55": -xf.d1_ret_open, "closing nearest the day's low": -xf.d1_loc})
        cols = fcols + ([c for c in ("d1_ret_open", "d1_vs_pm", "d1_loc", "d1_high_x") if k == "D"])
        X = xf[cols].astype(float)
        a = (xf["date"] < mid).to_numpy()
        X = X.clip(lower=X[a].quantile(.01), upper=X[a].quantile(.99), axis=1)
        model = Logit(l2=10.0).fit(X[a], (xf.ret[a] > 0).astype(int))
        scores["model trained on first half"] = pd.Series(model.predict(X), index=xf.index)
        r1, r2 = account(H1(xf), F, hy), account(H2(xf), F, hy)
        rows = [{"take first": "random (baseline)", "h1_typical": r1["typical"], "h2_typical": r2["typical"], "h2_bad_luck": r2["bad"]}]
        for nm, sc in scores.items():
            sc = sc.fillna(sc.median())
            s1 = account(H1(xf), F, hy, score=sc[xf["date"] < mid])
            s2 = account(H2(xf), F, hy, score=sc[xf["date"] >= mid])
            rows.append({"take first": nm, "h1_typical": np.nan if nm.startswith("model") else s1["typical"], "h2_typical": s2["typical"],
                         "h2_bad_luck": np.nan})
        show(f"3. Which gapper to take when there are several, {name[k]} (typical account growth per year)", rows, od, f"3_selection_{k}")

    # ---------------------------------------------------------------- 4. sizing
    rows = []
    for k, x in base.items():
        mk = filt[k]
        for lab, fr, fs in (("flat 2.4%", F, F), ("half size on flagged", F / 2, F), ("1.2% flagged / 3.0% others", 0.012, 0.030),
                            ("skip flagged", 0.0, F)):
            for hl, sel, yrs in (("all year", np.ones(len(x), bool), years), ("second half", (x["date"] >= mid).to_numpy(), hy)):
                y, m = x[sel], mk[sel]
                if fr == 0:
                    y, m = y[~m], m[~m]
                rows.append(line(f"{k} {lab}, {hl}", y, yrs, f=np.where(m, fr, fs)))
    show("4. Sizing by risk (flagged = any filter that helped in both halves)", rows, od, "4_sizing")

    # ---------------------------------------------------------------- 5. portfolio
    rows = []
    both = pd.concat([base["A"], base["D"]])
    for lab, x in (("A", base["A"]), ("D", base["D"]), ("A and D together", both)):
        for mp in (1, 2, 3, 5, 99):
            for hl, y, yrs in (("all year", x, years), ("second half", H2(x), hy)):
                a = account(y, F, yrs, max_pos=mp)
                rows.append({"what": lab, "max_positions": "no limit" if mp == 99 else mp, "period": hl, "taken": a["taken"],
                             "typical_yr": a["typical"], "bad_luck_yr": a["bad"], "worst_drop": a["dd"], "liq": a["liq"],
                             "closest_to_liquidation": a["stress"]})
    show(f"5. Several positions at once, {F:.1%} each (closest_to_liquidation: 1.0 = liquidated if every open short hit its worst price together)",
         rows, od, "5_portfolio")

    # ---------------------------------------------------------------- 6. entry
    rows = []
    for en in ENTRIES:
        best = None
        for tg in TARGETS:
            for hz in HORIZONS:
                x = pick(t, en, tg, hz)
                if len(H1(x)) < 40:
                    continue
                a1 = account(H1(x), F, hy, n_sims=60)["typical"]
                if best is None or a1 > best[0]:
                    best = (a1, tg, hz)
        if best is None:
            continue
        a1, tg, hz = best
        x2 = H2(pick(t, en, tg, hz))
        m2, t2 = ev(x2)
        a2 = account(x2, F, hy)
        rows.append({"entry": en, "target": tg, "limit_days": hz, "h1_typical": a1, "n_h2": len(x2), "hit_h2": x2.hit.mean(),
                     "stopped_h2": x2.stopped.mean(), "ev_h2": m2, "t_h2": t2, "h2_typical": a2["typical"], "h2_bad_luck": a2["bad"],
                     "h2_worst_drop": a2["dd"]})
    show(f"6. Other entries under the same risk rules ({F:.1%}, stop {STOP:.0f}x): target and limit chosen on first half, shown on second half",
         sorted(rows, key=lambda r: -r["h2_typical"]), od, "6_entries")

    # ---------------------------------------------------------------- 7. other strategies
    rows = []
    for k, (en, tg, hz) in spec.items():
        for s in (1.15, 1.25, 1.5, 2.0, 3.0):
            x = pick(t, en, tg, hz, stop=s)
            worst = x.loc[x.stopped, "R"].max() if x.stopped.any() else s
            f = min(0.30, LOSS_BUDGET / (worst - 1))                   # the worst stop-out seen costs the same ~7% of the account
            a, a2 = account(x, f, years), account(H2(x), f, hy)
            rows.append({"method": k, "stop": s, "position": f, "worst_fill": worst, "hit": x.hit.mean(), "stopped": x.stopped.mean(),
                         "ev": x.ret.mean(), "ev_h1": ev(H1(x))[0], "ev_h2": ev(H2(x))[0], "typical_yr": a["typical"], "bad_luck_yr": a["bad"],
                         "worst_drop": a["dd"], "h2_typical": a2["typical"]})
    show("7a. A normal stop instead of a far one (position sized so the worst stop-out seen costs about 7% of the account)", rows, od, "7a_tight_stop")
    rows = []
    lg = lg[lg.pm_gap >= 0.5]
    for (tg, st), x in lg.groupby(["target", "stop"]):
        m, ts = ev(x)
        rows.append({"buy": "break of pre-market high, 09:30-10:00", "target": "none (hold to 15:55)" if tg == 0 else f"+{tg:.0%}", "stop": f"-{st:.0%}",
                     "n": len(x), "win_rate": (x.ret > 0).mean(), "ev": m, "t": ts, "ev_h1": ev(H1(x))[0], "ev_h2": ev(H2(x))[0]})
    show("7b. The opposite trade: buying the breakout", rows, od, "7b_long")
    print("\nIDEAS DONE")


if __name__ == "__main__":
    main()
