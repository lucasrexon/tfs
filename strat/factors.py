"""Fundamental and technical stock-selection strategies, monthly, US stocks 2010-2026.

Universe each month: the Yahoo-covered US SEC filers in data_cache/hdc/candidates.csv (partnerships and
duplicate share classes removed) worth $2B+ (price x latest SEC share count, 45 days after the period
end), trading $10M+ a day (60-day median), price $5+.

Fundamentals: SEC XBRL frames for calendar year y (annual facts, latest filing), used from May 1 of y+1.
Strategy: each month-end, rank the universe on one signal, hold the top fifth equally weighted for the
next month. Costs: 0.1% of the value traded (0.05% each side).

Survivorship: the universe is today's listed companies, so firms that died before 2026 are missing.
That flatters every strategy's raw return; the column to read is the return over the equal-weight
universe, which has the same bias.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from hdc.backtest import tbill, yahoo_series
from hdc.data import cache_dir
from hdc.events import PARTNERSHIP, company_shares, shares_asof
from hdc.universe import load_yahoo, real_price_factor, shares_frames

START, END = "2010-01-31", "2026-10-31"
MIN_CAP, MIN_DV, MIN_PX = 2e9, 10e6, 5.0
COST = 0.001
YEARS = range(2009, 2026)
DURATION = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "GrossProfit",
            "CostOfRevenue", "CostOfGoodsAndServicesSold", "OperatingIncomeLoss", "NetIncomeLoss",
            "NetCashProvidedByUsedInOperatingActivities", "PaymentsOfDividends", "PaymentsOfDividendsCommonStock",
            "PaymentsForRepurchaseOfCommonStock"]
INSTANT = ["Assets", "StockholdersEquity"]


# ---------------------------------------------------------------- SEC frames
def fetch_frames(progress=print) -> None:
    from tfs.sec import Edgar
    ed = Edgar(cache_dir(), rps=5.0)
    root = cache_dir() / "sec" / "frames"
    for c in DURATION + INSTANT:
        for y in YEARS:
            period = f"CY{y}Q4I" if c in INSTANT else f"CY{y}"
            f = root / f"{c}_{period}.json"
            if f.exists():
                continue
            d = ed._get(f"https://data.sec.gov/api/xbrl/frames/us-gaap/{c}/USD/{period}.json")
            f.write_text(json.dumps(d or {}))
        progress(f"  frames {c} done ({ed.calls} calls)")


def fundamentals() -> pd.DataFrame:
    """One row per (cik, year): the annual facts, with fallbacks across tags."""
    root = cache_dir() / "sec" / "frames"
    raw = {}
    for c in DURATION + INSTANT:
        rows = []
        for y in YEARS:
            period = f"CY{y}Q4I" if c in INSTANT else f"CY{y}"
            f = root / f"{c}_{period}.json"
            if not f.exists():
                continue
            for d in (json.loads(f.read_text()) or {}).get("data", []):
                rows.append((int(d["cik"]), y, float(d["val"])))
        raw[c] = pd.DataFrame(rows, columns=["cik", "year", c]).drop_duplicates(["cik", "year"]).set_index(["cik", "year"])[c]
    df = pd.concat(raw.values(), axis=1)
    rev = df["Revenues"].fillna(df["RevenueFromContractWithCustomerExcludingAssessedTax"]).fillna(df["SalesRevenueNet"])
    cost = df["CostOfRevenue"].fillna(df["CostOfGoodsAndServicesSold"])
    out = pd.DataFrame({
        "revenue": rev,
        "gross_profit": df["GrossProfit"].fillna(rev - cost),
        "op_income": df["OperatingIncomeLoss"],
        "net_income": df["NetIncomeLoss"],
        "ocf": df["NetCashProvidedByUsedInOperatingActivities"],
        "dividends": df["PaymentsOfDividends"].fillna(df["PaymentsOfDividendsCommonStock"]),
        "buybacks": df["PaymentsForRepurchaseOfCommonStock"],
        "assets": df["Assets"],
        "equity": df["StockholdersEquity"],
    })
    return out


# ---------------------------------------------------------------- monthly panel
def universe() -> pd.DataFrame:
    cand = pd.read_csv(cache_dir() / "hdc" / "candidates.csv")
    cand = cand[~cand.name.str.contains(PARTNERSHIP, regex=True, na=False)]
    keep = []
    for cik, g in cand.groupby("cik"):                       # one share class per company: the most traded
        if len(g) == 1:
            keep.append(g.iloc[0])
            continue
        dv = {}
        for _, r in g.iterrows():
            y = load_yahoo(r.ticker)
            dv[r.ticker] = float((y.close * y.volume).tail(250).median()) if y is not None and len(y) else 0.0
        keep.append(g[g.ticker == max(dv, key=dv.get)].iloc[0])
    return pd.DataFrame(keep)


def panel(progress=print) -> dict[str, pd.DataFrame]:
    """Month-end panels (rows month-ends, columns tickers): ret_next, cap, dv, px and technical signals."""
    cand = universe()
    sh = shares_frames()
    months = pd.date_range("2009-01-31", "2026-11-30", freq="ME")
    cols = {k: {} for k in ("ret", "cap", "dv", "px", "mom", "high52", "vol", "rev1")}
    cik_of = {}
    for k, r in enumerate(cand.itertuples()):
        y = load_yahoo(r.ticker)
        if y is None or len(y) < 260:
            continue
        y = y.copy()
        y.index = pd.to_datetime(y.index)
        fac = real_price_factor(y.splits)
        fac.index = pd.to_datetime(fac.index)
        s = company_shares(sh, int(r.cik)).copy()
        s["end"] = (pd.to_datetime(s.end) + pd.Timedelta(days=45)).dt.strftime("%Y-%m-%d")   # known 45 days later
        shares = shares_asof(s, y.index.strftime("%Y-%m-%d"), backfill=False)
        shares.index = y.index
        cap_d = y.close * fac * shares
        dv_d = (y.close * y.volume).rolling(60).median()
        dr = y.adj_close.pct_change()
        m_adj = y.adj_close.resample("ME").last()
        cols["ret"][r.ticker] = m_adj.pct_change()
        cols["cap"][r.ticker] = cap_d.resample("ME").last()
        cols["dv"][r.ticker] = dv_d.resample("ME").last()
        cols["px"][r.ticker] = (y.close * fac).resample("ME").last()
        cols["mom"][r.ticker] = m_adj.shift(1) / m_adj.shift(12) - 1
        cols["high52"][r.ticker] = (y.close / y.close.rolling(252).max()).resample("ME").last()
        cols["vol"][r.ticker] = (dr.rolling(252).std() * np.sqrt(252)).resample("ME").last()
        cols["rev1"][r.ticker] = m_adj.pct_change()
        cik_of[r.ticker] = int(r.cik)
        if k % 300 == 0:
            progress(f"  prices {k}/{len(cand)}")
    P = {k: pd.DataFrame(v).reindex(months) for k, v in cols.items()}
    P["ret_next"] = P["ret"].shift(-1)
    P["cik"] = pd.Series(cik_of)
    return P


def add_fundamentals(P: dict, F: pd.DataFrame) -> None:
    """Fundamental signals, using calendar year y's facts from May of y+1."""
    months = P["cap"].index
    fy = np.where(months.month >= 5, months.year - 1, months.year - 2)
    tick = list(P["cap"].columns)
    cik = P["cik"].reindex(tick)
    def field(name):
        s = F[name]
        out = np.full((len(months), len(tick)), np.nan)
        for j, t in enumerate(tick):
            c = cik.get(t)
            if c is None or pd.isna(c):
                continue
            try:
                ser = s.xs(int(c), level="cik")
            except KeyError:
                continue
            out[:, j] = ser.reindex(fy).to_numpy()
        return pd.DataFrame(out, index=months, columns=tick)
    f = {n: field(n) for n in ("revenue", "gross_profit", "op_income", "net_income", "ocf", "dividends", "buybacks",
                                "assets", "equity")}
    cap = P["cap"]
    pos = lambda x: x.where(x > 0)
    P["ep"] = f["net_income"] / cap
    P["bm"] = pos(f["equity"]) / cap
    P["cfp"] = f["ocf"] / cap
    P["gpa"] = f["gross_profit"] / pos(f["assets"])
    P["roe"] = f["net_income"] / pos(f["equity"])
    P["shy"] = (f["dividends"].fillna(0) + f["buybacks"].fillna(0)).where(f["ocf"].notna()) / cap
    P["ey_op"] = f["op_income"] / cap
    P["roa_op"] = f["op_income"] / pos(f["assets"])


# ---------------------------------------------------------------- strategies
def rank(x: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    return x.where(mask).rank(axis=1, pct=True)


SIGNALS = {
    # name: (description, function of P and the eligible mask -> score, higher = buy)
    "momentum": ("Momentum: return over the past 12 months, skipping the latest month", lambda P, m: P["mom"]),
    "high52": ("52-week high: price closest to its 52-week high", lambda P, m: P["high52"]),
    "low_vol": ("Low volatility: lowest 1-year daily volatility", lambda P, m: -P["vol"]),
    "reversal": ("Short-term reversal: biggest losers of the past month", lambda P, m: -P["rev1"]),
    "value": ("Value: average rank of earnings, book and cash flow to price", lambda P, m:
              (rank(P["ep"], m) + rank(P["bm"], m) + rank(P["cfp"], m)) / 3),
    "quality": ("Quality: gross profit / assets (Novy-Marx)", lambda P, m: P["gpa"]),
    "shareholder_yield": ("Shareholder yield: dividends + buybacks / market value", lambda P, m: P["shy"]),
    "magic_formula": ("Magic formula: operating earnings yield + operating return on assets", lambda P, m:
                      rank(P["ey_op"], m) + rank(P["roa_op"], m)),
    "quality_value": ("Quality + value", lambda P, m: rank(P["gpa"], m) + rank(P["ep"], m) + rank(P["cfp"], m)),
    "momentum_quality": ("Momentum + quality", lambda P, m: rank(P["mom"], m) + rank(P["gpa"], m)),
    "mom_qual_value": ("Momentum + quality + value", lambda P, m:
                       rank(P["mom"], m) + rank(P["gpa"], m) + (rank(P["ep"], m) + rank(P["cfp"], m)) / 2),
}


def eligible(P: dict) -> pd.DataFrame:
    return (P["cap"] >= MIN_CAP) & (P["dv"] >= MIN_DV) & (P["px"] >= MIN_PX) & P["ret_next"].notna()


def portfolio(score: pd.DataFrame, mask: pd.DataFrame, ret_next: pd.DataFrame, top: float | int = 0.2):
    s = score.where(mask & score.notna())
    if isinstance(top, float):
        sel = s.rank(axis=1, pct=True, ascending=False) <= top
    else:
        sel = s.rank(axis=1, ascending=False, method="first") <= top
    w = sel.div(sel.sum(axis=1), axis=0).fillna(0.0)
    gross = (w * ret_next.fillna(0.0)).sum(axis=1)
    # weights drift during the month; turnover against the drifted weights
    drift = w * (1 + ret_next.fillna(0.0))
    drift = drift.div(drift.sum(axis=1).replace(0, np.nan), axis=0).shift(1).fillna(0.0)
    turn = (w - drift).abs().sum(axis=1)
    net = gross - turn * COST / 2              # turnover counts buys and sells; each side costs 0.05%
    n = sel.sum(axis=1)
    return net.where(n > 0), n, turn


def realized(x: pd.Series) -> pd.Series:
    """Re-date returns from the formation month-end to the month they are earned in."""
    x = x.copy()
    x.index = x.index + pd.offsets.MonthEnd(1)
    return x


def stats(r: pd.Series, rf: pd.Series, bench: pd.Series | None = None) -> dict:
    """r and bench are indexed by formation month-end (as portfolio() returns them); rf by calendar month."""
    r = realized(r.dropna())
    bench = realized(bench) if bench is not None else None
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    out = {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(12),
           "sharpe": (r - rf.reindex(r.index).fillna(0)).mean() / r.std() * np.sqrt(12),
           "max_dd": float((eq / eq.cummax() - 1).min()), "worst_year": float(((1 + r).groupby(r.index.year).prod() - 1).min()),
           "best_year": float(((1 + r).groupby(r.index.year).prod() - 1).max())}
    for a, b in (("2010", "2017"), ("2018", "2026")):
        x = r.loc[a:b]
        out[f"cagr_{a}_{b[2:]}"] = (1 + x).prod() ** (12 / len(x)) - 1 if len(x) else np.nan
    if bench is not None:
        ex = (r - bench.reindex(r.index)).dropna()
        out["excess"] = (1 + r).prod() ** (1 / yrs) - (1 + bench.reindex(r.index)).prod() ** (1 / yrs)
        out["excess_t"] = ex.mean() / ex.std() * np.sqrt(len(ex))
        out["beat_months"] = float((ex > 0).mean())
        for a, b in (("2010", "2017"), ("2018", "2026")):
            x, y = r.loc[a:b], bench.reindex(r.index).loc[a:b]
            out[f"excess_{a}_{b[2:]}"] = (1 + x).prod() ** (12 / len(x)) - (1 + y).prod() ** (12 / len(y))
    return out


def main(progress=print):
    fetch_frames(progress)
    F = fundamentals()
    P = panel(progress)
    add_fundamentals(P, F)
    m = eligible(P)
    rng = slice(START, END)
    rf = (tbill().resample("ME").mean() / 12)
    spy = yahoo_series("SPY").adj_close
    spy.index = pd.to_datetime(spy.index)
    spy_m = spy.resample("ME").last().pct_change().shift(-1)
    ew = P["ret_next"].where(m).mean(axis=1)
    res = {"universe_names": float(m.sum(axis=1).loc[rng].median()),
           "equal_weight_universe": stats(ew.loc[rng], rf), "spy": stats(spy_m.loc[rng], rf)}
    curves = {"equal_weight_universe": ew.loc[rng], "spy": spy_m.loc[rng]}
    for name, (desc, fn) in SIGNALS.items():
        score = fn(P, m)
        for top in (0.2, 30):
            r, n, turn = portfolio(score, m, P["ret_next"], top)
            key = name if top == 0.2 else f"{name}_top30"
            st = stats(r.loc[rng], rf, ew.loc[rng])
            st.update(desc=desc, names=float(n.loc[rng].median()), turnover_yr=float(turn.loc[rng].mean() * 12))
            res[key] = st
            curves[key] = r.loc[rng]
    from tfs.config import load_config, resolve
    od = resolve(load_config(), "output") / "strat"
    od.mkdir(parents=True, exist_ok=True)
    (od / "factors.json").write_text(json.dumps(res, indent=1, default=float))
    pd.DataFrame(curves).to_csv(od / "factor_curves.csv")
    return res


if __name__ == "__main__":
    res = main()
    rows = []
    for k, v in res.items():
        if isinstance(v, dict):
            rows.append({"strategy": k, **{c: v.get(c) for c in ("cagr", "excess", "excess_t", "max_dd", "sharpe", "vol",
                                                                 "cagr_2010_17", "cagr_2018_26", "excess_2010_17",
                                                                 "excess_2018_26", "names", "turnover_yr")}})
    df = pd.DataFrame(rows).set_index("strategy")
    pd.set_option("display.width", 250)
    print(df.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("universe names per month:", res["universe_names"])
