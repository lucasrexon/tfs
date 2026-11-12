"""Massive (formerly Polygon.io) REST client with an on-disk cache.

Everything downloaded is cached, so a 5-year scan is paid for once and every
later backtest runs offline. Prices are split-adjusted (adjusted=true) so a
reverse split never looks like a gap; real-dollar prices are rebuilt from the
split history where a rule needs them (price floor, per-share costs, margin).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

NY = ZoneInfo("America/New_York")
BAR_COLS = ["o", "h", "l", "c", "v", "vw"]


class MassiveError(RuntimeError):
    pass


class MassiveClient:
    def __init__(self, base_url: str, cache_dir: str | Path, api_key: str | None = None,
                 max_rpm: int = 0, timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "TFS-research-bot/0.1"
        if api_key:
            self.s.headers["Authorization"] = f"Bearer {api_key}"
        self.min_interval = 60.0 / max_rpm if max_rpm else 0.0
        self._last = 0.0
        self.timeout = timeout
        self.calls = 0
        self._bulk_splits: dict[str, list] | None = None

    @classmethod
    def from_config(cls, cfg: dict, cache_dir: Path) -> "MassiveClient":
        d = cfg["data"]
        return cls(d["base_url"], cache_dir, os.environ.get(d.get("api_key_env", "MASSIVE_API_KEY")),
                   d.get("max_requests_per_minute", 0))

    # ------------------------------------------------------------------ http
    def _get(self, url: str, params: dict | None = None) -> dict:
        if not url.startswith("http"):
            url = self.base + url
        for attempt in range(6):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            self.calls += 1
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt * (5 if r.status_code == 429 else 1)))
                continue
            if r.status_code in (401, 403):
                raise MassiveError(
                    f"{r.status_code} from Massive: no API key, or your plan does not include this data "
                    f"({url.split('?')[0]}). Response: {r.text[:200]}")
            if r.status_code == 404:
                return {}
            raise MassiveError(f"HTTP {r.status_code} for {url}: {r.text[:200]}")
        raise MassiveError(f"Gave up after retries: {url}")

    # ---------------------------------------------------------- endpoints
    def grouped_daily(self, date: str) -> pd.DataFrame:
        """All US listed stocks for one day (adjusted). Empty frame on market holidays."""
        f = self.cache / "grouped" / f"{date}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        j = self._get(f"/v2/aggs/grouped/locale/us/market/stocks/{date}",
                      {"adjusted": "true", "include_otc": "false"})
        rows = j.get("results") or []
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["T", *BAR_COLS])
        df = df.reindex(columns=["T", *BAR_COLS])
        df[BAR_COLS] = df[BAR_COLS].astype("float64")
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(f, index=False)
        return df

    def minute_bars(self, ticker: str, date: str) -> pd.DataFrame:
        """1-minute bars 04:00-20:00 ET for one day, adjusted, indexed by NY time."""
        f = self.cache / "minute" / date / f"{ticker}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        j = self._get(f"/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}",
                      {"adjusted": "true", "sort": "asc", "limit": 50000})
        df = bars_from_results(j.get("results") or [])
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(f)
        return df

    def ticker_details(self, ticker: str, date: str) -> dict:
        f = self.cache / "details" / f"{ticker}_{date}.json"
        if f.exists():
            return json.loads(f.read_text())
        j = self._get(f"/v3/reference/tickers/{ticker}", {"date": date}).get("results") or {}
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(j))
        return j

    def preload_splits(self, since: str) -> int:
        """Every split executed since `since`, in a few paged calls instead of one call per
        stock. Only splits AFTER a gap day change its real price, so this is complete."""
        f = self.cache / f"splits_since_{since}.json"
        if f.exists():
            rows = json.loads(f.read_text())
        else:
            rows, url, params = [], "/v3/reference/splits", {"execution_date.gte": since, "limit": 1000}
            while url:
                j = self._get(url, params)
                rows += j.get("results") or []
                url, params = j.get("next_url"), None
            f.write_text(json.dumps(rows))
        self._bulk_splits = {}
        for r in rows:
            self._bulk_splits.setdefault(r["ticker"], []).append(r)
        return len(rows)

    def splits(self, ticker: str) -> list[dict]:
        if self._bulk_splits is not None:
            return self._bulk_splits.get(ticker, [])
        f = self.cache / "splits" / f"{ticker}.json"
        if f.exists():
            return json.loads(f.read_text())
        out, url, params = [], "/v3/reference/splits", {"ticker": ticker, "limit": 1000}
        while url:
            j = self._get(url, params)
            out += j.get("results") or []
            url, params = j.get("next_url"), None
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out))
        return out

    def tickers(self, types: list[str]) -> set[str]:
        """Every ticker of the given types, listed AND delisted (fights survivorship bias)."""
        f = self.cache / "universe.json"
        if f.exists():
            return set(json.loads(f.read_text()))
        found: set[str] = set()
        for t in types:
            for active in ("true", "false"):
                url, params = "/v3/reference/tickers", {"type": t, "market": "stocks",
                                                        "active": active, "limit": 1000}
                while url:
                    j = self._get(url, params)
                    found.update(r["ticker"] for r in j.get("results") or [])
                    url, params = j.get("next_url"), None
        f.write_text(json.dumps(sorted(found)))
        return found


def bars_from_results(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=BAR_COLS, index=pd.DatetimeIndex([], tz=NY, name="ts"))
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(NY)
    df = df.reindex(columns=BAR_COLS).astype("float64")
    df.index = pd.DatetimeIndex(idx, name="ts")
    df["vw"] = df["vw"].fillna((df["h"] + df["l"] + df["c"]) / 3)
    return df


def split_factor(splits: list[dict], date: str) -> float:
    """Multiply an adjusted price by this to get the real price traded on `date`.

    A 1-for-10 reverse split later (split_from=10, split_to=1) makes adjusted
    history 10x the real price, so real = adjusted * to/from for every split
    executed after `date`. Volume goes the other way: real = adjusted / factor.
    """
    f = 1.0
    for s in splits:
        if s.get("execution_date", "") > date:
            fr, to = float(s.get("split_from", 1) or 1), float(s.get("split_to", 1) or 1)
            if fr > 0 and to > 0:
                f *= to / fr
    return f
