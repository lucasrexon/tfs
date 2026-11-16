"""Downloads for the after-hours and news-sentiment factors (resumable; about 7 hours at 5 requests a minute).

    MASSIVE_API_KEY=... python -m tfs.v2.dl_extra

  1. news with Massive's per-ticker sentiment ("insights"), one call per ticker  -> data_cache/news_ins/
  2. 1-minute bars for the trading day BEFORE each gap (for the after-hours move) -> data_cache/minute/
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from tfs.config import load_config, resolve
from tfs.data.massive import MassiveClient


def news_with_insights(cl: MassiveClient, ticker: str, start: str, end: str) -> int:
    f = cl.cache / "news_ins" / f"{ticker}.json"
    if f.exists():
        return 0
    rows, url, params, calls = [], "/v2/reference/news", {"ticker": ticker, "published_utc.gte": start,
                                                          "published_utc.lte": end, "limit": 1000, "order": "asc",
                                                          "sort": "published_utc"}, 0
    while url:
        j = cl._get(url, params); calls += 1
        for r in j.get("results") or []:
            ins = [i for i in (r.get("insights") or []) if i.get("ticker") == ticker]
            rows.append({"published_utc": r.get("published_utc"), "title": r.get("title", ""),
                         "publisher": (r.get("publisher") or {}).get("name"), "tickers": r.get("tickers") or [],
                         "sentiment": ins[0].get("sentiment") if ins else None,
                         "reason": (ins[0].get("sentiment_reasoning") or "")[:200] if ins else None})
        url, params = j.get("next_url"), None
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rows))
    return calls


def main():
    cfg = load_config()
    cache = resolve(cfg, cfg["data"]["cache_dir"])
    cl = MassiveClient.from_config(cfg, cache)
    t = pd.read_csv(resolve(cfg, "output") / "v2" / "patterns_trades.csv")
    g = pd.read_parquet(resolve(cfg, "output") / "v2" / "gappers40.parquet")[["date", "ticker", "prev_day"]]
    t = t.merge(g, on=["date", "ticker"])
    tickers = sorted(t.ticker.unique())
    for i, tk in enumerate(tickers):
        if i % 50 == 0:
            print(f"[{datetime.now():%H:%M}] news {i}/{len(tickers)}", flush=True)
        news_with_insights(cl, tk, cfg["data"]["start"], cfg["data"]["end"])
    pairs = sorted(set(zip(t.ticker, t.prev_day)))
    for i, (tk, d) in enumerate(pairs):
        if i % 100 == 0:
            print(f"[{datetime.now():%H:%M}] prev-day bars {i}/{len(pairs)}", flush=True)
        cl.minute_bars(tk, d)
    print("DL_EXTRA DONE", flush=True)


if __name__ == "__main__":
    main()
