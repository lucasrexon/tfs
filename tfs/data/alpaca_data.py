"""Alpaca market data: daily and 1-minute stock bars for years the Massive free plan cannot reach.

Prices come from the SIP feed (all exchanges), split-adjusted like Massive's
adjusted=true bars, and are saved in the same file format under
data_cache/alpaca/, so the scan, backtest and research code reads them unchanged.
Alpaca's free plan serves SIP history older than 15 minutes at about 200
requests a minute. In the cloud the network proxy adds the account's keys
(headers APCA-API-KEY-ID and APCA-API-SECRET-KEY); locally set those two
environment variables.

HybridClient routes price calls for dates before the Massive window to Alpaca
and everything else (news, splits, company details) to Massive.
"""
from __future__ import annotations

import os
import time
from datetime import date as Date, timedelta
from pathlib import Path

import pandas as pd
import requests

from tfs.data.massive import BAR_COLS, NY

BASE = "https://data.alpaca.markets"


class AlpacaError(RuntimeError):
    pass


def frame_from_bars(rows: list[dict]) -> pd.DataFrame:
    """Alpaca bar dicts (t as RFC 3339) -> the Massive minute-bar frame: NY-time index, o h l c v vw."""
    if not rows:
        return pd.DataFrame(columns=BAR_COLS, index=pd.DatetimeIndex([], tz=NY, name="ts"))
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
    df = df.reindex(columns=BAR_COLS).astype("float64")
    df.index = pd.DatetimeIndex(idx, name="ts")
    df["vw"] = df["vw"].fillna((df["h"] + df["l"] + df["c"]) / 3)
    return df.sort_index()


def day_window_utc(day: str) -> tuple[str, str]:
    """UTC start/end that cover 04:00-20:00 New York time on `day`, whether EST or EDT."""
    d = Date.fromisoformat(day)
    return f"{day}T08:00:00Z", f"{(d + timedelta(days=1)).isoformat()}T01:00:00Z"


class AlpacaPrices:
    def __init__(self, cache_dir: str | Path, max_rpm: int = 190, timeout: int = 60,
                 feed: str = "sip", adjustment: str = "split"):
        self.cache = Path(cache_dir) / "alpaca"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "TFS-research-bot/0.1"
        k, sec = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if k and sec:                                   # locally; in the cloud the proxy adds them
            self.s.headers.update({"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": sec})
        self.min_interval = 60.0 / max_rpm if max_rpm else 0.0
        self._last = 0.0
        self.timeout, self.feed, self.adjustment = timeout, feed, adjustment
        self.calls = 0

    @classmethod
    def from_config(cls, cfg: dict, cache_dir: Path) -> "AlpacaPrices":
        a = cfg["data"].get("alpaca", {})
        return cls(cache_dir, a.get("max_requests_per_minute", 190), feed=a.get("feed", "sip"),
                   adjustment=a.get("adjustment", "split"))

    # ------------------------------------------------------------------ http
    def _get(self, path: str, params: dict) -> dict:
        for attempt in range(7):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.s.get(BASE + path, params=params, timeout=self.timeout)
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
                raise AlpacaError(f"{r.status_code} from Alpaca: the key is missing or wrong, or the plan "
                                  f"lacks this data ({r.headers.get('X-Proxy-Error', '')} {r.text[:160]})")
            raise AlpacaError(f"HTTP {r.status_code} for {path}: {r.text[:200]}")
        raise AlpacaError(f"Gave up after retries: {path}")

    def bars(self, symbols: list[str], timeframe: str, start: str, end: str) -> dict[str, list[dict]]:
        """Every bar for these symbols, following page tokens. asof='-' turns off Alpaca's
        renamed-symbol mapping, so a symbol means what it meant on the day it traded."""
        out: dict[str, list[dict]] = {}
        params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": start, "end": end,
                  "feed": self.feed, "adjustment": self.adjustment, "asof": "-", "limit": 10000, "sort": "asc"}
        while True:
            j = self._get("/v2/stocks/bars", params)
            for sym, rows in (j.get("bars") or {}).items():
                out.setdefault(sym, []).extend(rows)
            tok = j.get("next_page_token")
            if not tok:
                return out
            params["page_token"] = tok

    # ---------------------------------------------------------- endpoints
    def minute_bars(self, ticker: str, date: str) -> pd.DataFrame:
        """1-minute bars 04:00-20:00 ET for one day, split-adjusted, indexed by NY time."""
        f = self.cache / "minute" / date / f"{ticker}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        start, end = day_window_utc(date)
        df = frame_from_bars(self.bars([ticker], "1Min", start, end).get(ticker, []))
        df = df[df.index.strftime("%Y-%m-%d") == date]
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(f)
        return df

    def grouped_daily(self, date: str) -> pd.DataFrame:
        """All stocks for one day in Massive's grouped format (T o h l c v vw), built by build_daily."""
        f = self.cache / "grouped" / f"{date}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        return pd.DataFrame(columns=["T", *BAR_COLS])

    def build_daily(self, symbols: list[str], days: list[str], batch: int = 100, progress=None) -> int:
        """Download daily bars for every symbol over `days` (sorted trading days), in batches of
        symbols, then write one grouped file per day. Each finished batch is kept, so a stopped
        run picks up where it left off."""
        start, end = days[0], days[-1]
        bdir = self.cache / "daily_batches" / f"{start}_{end}"
        bdir.mkdir(parents=True, exist_ok=True)
        syms = sorted(set(symbols))
        chunks = [syms[i:i + batch] for i in range(0, len(syms), batch)]
        for i, ch in enumerate(chunks):
            f = bdir / f"{i:04d}_{ch[0]}_{ch[-1]}.parquet".replace("/", "_")
            if f.exists():
                continue
            rows = []
            for sym, bars in self.bars(ch, "1Day", start, f"{end}T23:59:59Z").items():
                for b in bars:
                    rows.append({"T": sym, "t": b["t"], **{k: b.get(k) for k in BAR_COLS}})
            pd.DataFrame(rows, columns=["T", "t", *BAR_COLS]).to_parquet(f, index=False)
            if progress:
                progress(i + 1, len(chunks))
        allb = pd.concat([pd.read_parquet(p) for p in sorted(bdir.glob("*.parquet"))], ignore_index=True)
        allb["date"] = pd.to_datetime(allb["t"], utc=True).dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
        allb[BAR_COLS] = allb[BAR_COLS].astype("float64")
        gdir = self.cache / "grouped"
        gdir.mkdir(parents=True, exist_ok=True)
        by_day = dict(tuple(allb.groupby("date")))
        for d in days:
            g = by_day.get(d)
            g = pd.DataFrame(columns=["T", *BAR_COLS]) if g is None else g[["T", *BAR_COLS]]
            g.reset_index(drop=True).to_parquet(gdir / f"{d}.parquet", index=False)
        return len(allb)


class HybridClient:
    """Massive for everything, except prices on dates before `alpaca_before` (or all prices
    when force_alpaca is set, for comparing the two vendors on the same days)."""

    def __init__(self, massive, alpaca: AlpacaPrices, alpaca_before: str, force_alpaca: bool = False):
        self.m, self.a, self.before, self.force = massive, alpaca, alpaca_before, force_alpaca

    def _alp(self, date: str) -> bool:
        return self.force or date < self.before

    def grouped_daily(self, date: str) -> pd.DataFrame:
        return self.a.grouped_daily(date) if self._alp(date) else self.m.grouped_daily(date)

    def minute_bars(self, ticker: str, date: str) -> pd.DataFrame:
        return self.a.minute_bars(ticker, date) if self._alp(date) else self.m.minute_bars(ticker, date)

    def __getattr__(self, name):                         # news, splits, details, tickers ...
        return getattr(self.m, name)
