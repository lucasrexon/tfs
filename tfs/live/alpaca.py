"""Alpaca broker + market data (REST). Paper trading unless told otherwise.

Keys come from environment variables on the ACCOUNT OWNER's computer:
  APCA_API_KEY_ID, APCA_API_SECRET_KEY
Alpaca only lets you short stocks it marks easy-to-borrow, which by itself keeps
the bot away from most hard-to-borrow squeezes.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

from tfs.data.massive import BAR_COLS, NY


class Alpaca:
    def __init__(self, key: str, secret: str, paper: bool = True, feed: str = "sip"):
        self.trade_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        self.paper, self.feed = paper, feed
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    @classmethod
    def from_env(cls, paper: bool, feed: str) -> "Alpaca":
        k, s = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not k or not s:
            raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (the account owner's Alpaca keys).")
        return cls(k, s, paper, feed)

    def _req(self, method, url, **kw):
        for attempt in range(4):
            r = self.s.request(method, url, timeout=20, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Alpaca {r.status_code}: {r.text[:300]}")
            return r.json() if r.text else {}
        raise RuntimeError(f"Alpaca: gave up on {url}")

    # ---- trading
    def account(self): return self._req("GET", f"{self.trade_url}/v2/account")
    def clock(self): return self._req("GET", f"{self.trade_url}/v2/clock")
    def asset(self, sym): return self._req("GET", f"{self.trade_url}/v2/assets/{sym}")
    def positions(self): return self._req("GET", f"{self.trade_url}/v2/positions")

    def assets(self):
        return self._req("GET", f"{self.trade_url}/v2/assets", params={"status": "active", "asset_class": "us_equity"})

    def short_with_stop(self, sym: str, qty: int, stop: float, client_id: str):
        """Market short with a protective buy-stop attached (OTO): the stop lives at the broker."""
        return self._req("POST", f"{self.trade_url}/v2/orders", json={
            "symbol": sym, "qty": str(qty), "side": "sell", "type": "market", "time_in_force": "day",
            "order_class": "oto", "stop_loss": {"stop_price": f"{stop:.2f}"}, "client_order_id": client_id})

    def close_position(self, sym: str):
        return self._req("DELETE", f"{self.trade_url}/v2/positions/{sym}", params={"cancel_orders": "true"})

    # ---- data
    def snapshots(self, syms: list[str]) -> dict:
        out = {}
        for i in range(0, len(syms), 500):
            out.update(self._req("GET", f"{self.data_url}/v2/stocks/snapshots",
                                 params={"symbols": ",".join(syms[i:i + 500]), "feed": self.feed}))
        return out

    def bars(self, sym: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        rows, token = [], None
        while True:
            p = {"timeframe": "1Min", "start": start.isoformat(), "end": end.isoformat(), "feed": self.feed,
                 "adjustment": "raw", "limit": 10000}
            if token:
                p["page_token"] = token
            j = self._req("GET", f"{self.data_url}/v2/stocks/{sym}/bars", params=p)
            rows += j.get("bars") or []
            token = j.get("next_page_token")
            if not token:
                break
        if not rows:
            return pd.DataFrame(columns=BAR_COLS, index=pd.DatetimeIndex([], tz=NY, name="ts"))
        df = pd.DataFrame(rows)
        idx = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
        df = df.reindex(columns=BAR_COLS).astype("float64")
        df.index = pd.DatetimeIndex(idx, name="ts")
        df["vw"] = df["vw"].fillna((df["h"] + df["l"] + df["c"]) / 3)
        return df

    def split_today(self, sym: str, day: str) -> bool:
        try:
            j = self._req("GET", f"{self.data_url}/v1/corporate-actions",
                          params={"symbols": sym, "types": "forward_split,reverse_split", "start": day, "end": day})
            ca = j.get("corporate_actions") or {}
            return any(ca.get(k) for k in ("forward_splits", "reverse_splits"))
        except RuntimeError:
            return False
