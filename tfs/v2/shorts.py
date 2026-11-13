"""Short-side data: short interest (twice a month), daily short volume, fails-to-deliver.

    MASSIVE_API_KEY=... python -m tfs.v2.shorts        (about 60 API calls + 55 small SEC files)

Everything is used point-in-time: short interest is published about 8 business days after
its settlement date, and the SEC publishes fails-to-deliver about two weeks after each
half-month, so a figure only counts once it was public before the gap morning.
"""
from __future__ import annotations

import io
import zipfile
from datetime import date as Date, timedelta
from pathlib import Path

import pandas as pd
import requests

from tfs.config import load_config, resolve
from tfs.data.massive import MassiveClient, split_factor
from tfs.sec import UA

SI_LAG_DAYS = 14          # calendar days from settlement to public
FTD_LAG_DAYS = 31         # calendar days until a fail record is surely public
FTD_WINDOW = 45


def _paged(cl: MassiveClient, path: str, params: dict) -> list[dict]:
    rows, url = [], path
    while url:
        j = cl._get(url, params)
        rows += j.get("results") or []
        url, params = j.get("next_url"), None
    return rows


def download(cfg, tickers: list[str]) -> None:
    cache = resolve(cfg, cfg["data"]["cache_dir"])
    cl = MassiveClient.from_config(cfg, cache)
    start = (Date.fromisoformat(cfg["data"]["start"]) - timedelta(days=75)).isoformat()
    for name, path, key, size in (("short_interest", "/stocks/v1/short-interest", "settlement_date", 50),
                                  ("short_volume", "/stocks/v1/short-volume", "date", 40)):
        f = cache / f"{name}.parquet"
        if f.exists():
            continue
        rows = []
        for i in range(0, len(tickers), size):
            rows += _paged(cl, path, {"ticker.any_of": ",".join(tickers[i:i + size]), f"{key}.gte": start, "limit": 50000})
            print(f"  {name}: {min(i + size, len(tickers))}/{len(tickers)} tickers, {len(rows):,} rows", flush=True)
        cols = ["ticker", key, "short_interest", "avg_daily_volume", "days_to_cover"] if name == "short_interest" \
            else ["ticker", key, "total_volume", "short_volume"]
        pd.DataFrame(rows).reindex(columns=cols).rename(columns={key: "date"}).to_parquet(f, index=False)
    f = cache / "ftd.parquet"
    if not f.exists():
        want, parts = set(tickers), []
        s = requests.Session(); s.headers["User-Agent"] = UA
        m = Date.fromisoformat(start).replace(day=1)
        end = Date.fromisoformat(cfg["data"]["end"])
        while m <= end:
            for half in "ab":
                r = s.get(f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{m:%Y%m}{half}.zip", timeout=60)
                if r.status_code != 200:
                    print(f"  ftd {m:%Y%m}{half}: not published ({r.status_code})"); continue
                z = zipfile.ZipFile(io.BytesIO(r.content))
                d = pd.read_csv(z.open(z.namelist()[0]), sep="|", dtype=str, encoding="latin-1", on_bad_lines="skip")
                d.columns = ["date", "cusip", "ticker", "qty", "desc", "price"][:len(d.columns)]
                d = d[d["ticker"].isin(want) & d["date"].str.fullmatch(r"\d{8}", na=False)]
                parts.append(pd.DataFrame({"ticker": d["ticker"], "date": pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"),
                                           "qty": pd.to_numeric(d["qty"], errors="coerce")}))
            m = (m.replace(day=28) + timedelta(days=5)).replace(day=1)
        pd.concat(parts).dropna().to_parquet(f, index=False)
        print(f"  ftd: {sum(len(p) for p in parts):,} rows")


class Shorts:
    def __init__(self, cache: Path):
        rd = lambda n: pd.read_parquet(cache / n) if (cache / n).exists() else pd.DataFrame(columns=["ticker", "date"])
        self.si = {t: x.sort_values("date") for t, x in rd("short_interest.parquet").groupby("ticker")}
        self.sv = {t: x.sort_values("date") for t, x in rd("short_volume.parquet").groupby("ticker")}
        self.ftd = {t: x.sort_values("date") for t, x in rd("ftd.parquet").groupby("ticker")}

    def features(self, ticker: str, gap_date: str, shares_out: float | None, splits: list[dict] | None = None) -> dict:
        D = Date.fromisoformat(gap_date)
        out = {"si_pct_so": None, "days_to_cover": None, "si_chg": None, "sv_ratio_prev": None, "sv_ratio_5d": None,
               "ftd_pct_so": None, "ftd_days": None}
        now = split_factor(splits or [], gap_date)
        basis = lambda d: split_factor(splits or [], d) / now          # shares on `d` -> shares on the gap date
        si = self.si.get(ticker)
        if si is not None:
            s = si[si["date"] <= (D - timedelta(days=SI_LAG_DAYS)).isoformat()]
            if len(s):
                last = s.iloc[-1]
                if shares_out:
                    x = float(last["short_interest"]) * basis(last["date"]) / shares_out
                    out["si_pct_so"] = x if x <= 1 else None           # above 100%: share counts on different bases
                out["days_to_cover"] = float(last["days_to_cover"]) if pd.notna(last["days_to_cover"]) else None
                if len(s) > 1 and s.iloc[-2]["short_interest"] > 0:
                    prev = s.iloc[-2]
                    out["si_chg"] = float(last["short_interest"] * basis(last["date"])) / float(prev["short_interest"] * basis(prev["date"])) - 1
        sv = self.sv.get(ticker)
        if sv is not None:
            v = sv[sv["date"] < gap_date].tail(5)
            if len(v) and v["total_volume"].iloc[-1] > 0:
                out["sv_ratio_prev"] = float(v["short_volume"].iloc[-1] / v["total_volume"].iloc[-1])
            if len(v) and v["total_volume"].sum() > 0:
                out["sv_ratio_5d"] = float(v["short_volume"].sum() / v["total_volume"].sum())
        fd = self.ftd.get(ticker)
        if shares_out:
            out["ftd_pct_so"], out["ftd_days"] = 0.0, 0
            if fd is not None:
                lo, hi = (D - timedelta(days=FTD_LAG_DAYS + FTD_WINDOW)).isoformat(), (D - timedelta(days=FTD_LAG_DAYS)).isoformat()
                w = fd[(fd["date"] >= lo) & (fd["date"] <= hi)]
                if len(w):
                    q = w["qty"].to_numpy() * [basis(d) for d in w["date"]] / shares_out
                    out["ftd_pct_so"], out["ftd_days"] = float(min(q.max(), 1.0)), int((q >= 0.005).sum())
        return out


if __name__ == "__main__":
    cfg = load_config()
    g = pd.read_parquet(resolve(cfg, "output") / "v2" / "gappers40.parquet")
    download(cfg, sorted(g["ticker"].unique()))
    print("SHORTS DONE")
