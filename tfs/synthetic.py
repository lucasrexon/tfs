"""A fake market with the same interface as MassiveClient.

Used by `python -m tfs demo` and the tests, so the whole pipeline can be
checked before any real data is bought. The numbers it produces mean NOTHING
about the real market.
"""
from __future__ import annotations

import hashlib
from datetime import date as Date, timedelta

import numpy as np
import pandas as pd

from tfs.data.massive import BAR_COLS, NY


def _rng(*key) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


class FakeClient:
    def __init__(self, start="2024-01-02", end="2024-12-31", n_tickers=150, events_per_day=1.5, seed=7):
        self.seed = seed
        self.days = [d.isoformat() for d in pd.bdate_range(start, end).date]
        self.symbols = [f"F{i:03d}" for i in range(n_tickers)] + ["WRNTW"]
        self.calls = 0
        rng = _rng(seed, "events")
        # daily closes: random walk
        
        px = {t: float(rng.uniform(1.2, 12)) for t in self.symbols}
        self.events: dict[tuple, dict] = {}
        self.minute: dict[tuple, pd.DataFrame] = {}
        self.daily: dict[str, dict[str, dict]] = {d: {} for d in self.days}
        for di, d in enumerate(self.days):
            todays = set()
            if di > 0:
                for _ in range(rng.poisson(events_per_day)):
                    todays.add(self.symbols[int(rng.integers(0, len(self.symbols)))])
            for t in self.symbols:
                p = px[t]
                if t in todays:
                    kind = rng.choice(["fade", "fade", "squeeze", "weak_pm", "thin"])
                    bars = self._gap_day(d, t, p, kind, rng)
                    self.minute[(t, d)] = bars
                    self.events[(t, d)] = {"kind": kind}
                    reg = bars[(bars.index.hour * 60 + bars.index.minute >= 570) &
                               (bars.index.hour * 60 + bars.index.minute < 960)]
                    bar = {"o": reg["o"].iloc[0], "h": bars["h"].max(), "l": bars["l"].min(),
                           "c": reg["c"].iloc[-1], "v": bars["v"].sum()}
                else:
                    r = rng.normal(0, 0.03)
                    o = p * (1 + rng.normal(0, 0.01))
                    c = max(0.2, p * np.exp(r))
                    bar = {"o": o, "h": max(o, c) * (1 + abs(rng.normal(0, 0.01))),
                           "l": min(o, c) * (1 - abs(rng.normal(0, 0.01))), "c": c,
                           "v": float(rng.integers(5e4, 2e6))}
                bar["vw"] = (bar["h"] + bar["l"] + bar["c"]) / 3
                self.daily[d][t] = bar
                px[t] = bar["c"]

    def _gap_day(self, d, t, p, kind, rng) -> pd.DataFrame:
        day = pd.Timestamp(d, tz=NY)
        idx = day + pd.to_timedelta(np.arange(240, 1200), unit="min")   # 04:00..19:59
        n_pm = 330                                                     # 04:00..09:29
        gap = float(rng.uniform(0.55, 2.5)) if kind != "weak_pm" else float(rng.uniform(0.1, 0.3))
        pm_path = p * (1 + gap * np.clip(np.linspace(0, 1, n_pm) ** 0.6 + rng.normal(0, 0.03, n_pm), 0, 1.15))
        o_reg = pm_path[-1] * (1 + rng.normal(0, 0.03))
        if kind == "weak_pm":
            o_reg = p * 1.6
        n_reg = 390
        if kind == "squeeze":
            target = np.log(1 + rng.uniform(0.3, 1.5))
        else:
            target = np.log(1 - rng.uniform(0.15, 0.45))
        pop = np.concatenate([np.linspace(0, np.log(rng.uniform(1.0, 1.15)), 12),
                              np.full(n_reg - 12, np.log(rng.uniform(1.0, 1.15)))])
        drift = np.linspace(0, target, n_reg) + pop * (np.linspace(1, 0, n_reg) ** 2)
        noise = np.cumsum(rng.normal(0, 0.006, n_reg))
        reg_path = o_reg * np.exp(drift + noise - noise[0])
        ah = np.full(1200 - 960, reg_path[-1]) * (1 + np.cumsum(rng.normal(0, 0.002, 240)))
        close = np.concatenate([pm_path, reg_path, ah])
        opn = np.concatenate([[p * 1.02], close[:-1]])
        opn[n_pm] = o_reg
        hi = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.004, len(close))))
        lo = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.004, len(close))))
        pm_vol = 3e6 if kind != "thin" else 1.5e5
        vol = np.concatenate([np.full(n_pm, pm_vol / n_pm), np.full(n_reg, 4e7 / n_reg), np.full(240, 2e3)])
        df = pd.DataFrame({"o": opn, "h": hi, "l": lo, "c": close, "v": vol}, index=pd.DatetimeIndex(idx, name="ts"))
        df["vw"] = (df["h"] + df["l"] + df["c"]) / 3
        if kind == "squeeze":                     # two suspected halts
            df = df.drop(df.index[n_pm + 20: n_pm + 26]).drop(df.index[n_pm + 45: n_pm + 51])
        return df[BAR_COLS]

    # ---- MassiveClient interface
    def grouped_daily(self, date):
        self.calls += 1
        if date not in self.daily:
            return pd.DataFrame(columns=["T", *BAR_COLS])
        return pd.DataFrame([{"T": t, **b} for t, b in self.daily[date].items()])

    def minute_bars(self, ticker, date):
        self.calls += 1
        if (ticker, date) in self.minute:
            return self.minute[(ticker, date)]
        b = self.daily.get(date, {}).get(ticker)
        if b is None:
            return pd.DataFrame(columns=BAR_COLS, index=pd.DatetimeIndex([], tz=NY, name="ts"))
        day = pd.Timestamp(date, tz=NY)
        idx = day + pd.to_timedelta(np.arange(570, 960), unit="min")
        path = np.linspace(b["o"], b["c"], len(idx))
        df = pd.DataFrame({"o": path, "h": path * 1.002, "l": path * 0.998, "c": path, "v": b["v"] / len(idx)},
                          index=pd.DatetimeIndex(idx, name="ts"))
        df["vw"] = df["c"]
        return df

    def ticker_details(self, ticker, date):
        r = _rng(self.seed, "det", ticker)
        list_date = (Date.fromisoformat(self.days[0]) - timedelta(days=int(r.integers(30, 3000)))).isoformat()
        return {"ticker": ticker, "type": "CS", "name": f"Fake {ticker} Corp",
                "share_class_shares_outstanding": int(r.integers(2e6, 4e7)), "list_date": list_date}

    def splits(self, ticker):
        return [{"execution_date": "2099-01-01", "split_from": 10, "split_to": 1}] if ticker == "F000" else []

    def tickers(self, types):
        return {t for t in self.symbols if not t.endswith("W")}
