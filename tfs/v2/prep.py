"""Download what v2 needs. Resumable: everything is cached, rerun to continue.

  1. gappers at +40% (the lowest threshold tested), from cached minute bars
  2. 10 trading days of daily bars after each gap (offline, from cached daily data)
  3. Massive news per ticker (1 call per ticker, free plan: 5/min)
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from tfs.config import load_config, resolve
from tfs.data.massive import MassiveClient
from tfs.news import ticker_news
from tfs.scan import stage2

V2_MIN_GAP = 0.40
NEXT_N = 10


def out_dir(cfg) -> Path:
    p = resolve(cfg, "output") / "v2"
    p.mkdir(parents=True, exist_ok=True)
    return p


def next_days(gappers: pd.DataFrame, grouped_dir: Path, n: int = NEXT_N) -> dict:
    days = sorted(p.stem for p in grouped_dir.glob("*.parquet")
                  if not pd.read_parquet(p, columns=["T"]).empty)
    pos = {d: i for i, d in enumerate(days)}
    need: dict[str, list] = {}
    for g in gappers[["ticker", "date"]].itertuples(index=False):
        i = pos.get(g.date)
        if i is None:
            continue
        for d in days[i + 1: i + 1 + n]:
            need.setdefault(d, []).append(g)
    out: dict = {}
    for d, gs in need.items():
        df = pd.read_parquet(grouped_dir / f"{d}.parquet").set_index("T")
        for g in gs:
            if g.ticker in df.index:
                r = df.loc[g.ticker]
                out.setdefault((g.ticker, g.date), []).append(
                    {"date": d, "o": float(r.o), "h": float(r.h), "l": float(r.l), "c": float(r.c), "v": float(r.v)})
    prev = {}
    for g in gappers[["ticker", "date"]].itertuples(index=False):
        i = pos.get(g.date)
        prev[(g.ticker, g.date)] = days[i - 1] if i else None
    return out, prev


def main():
    cfg = load_config()
    od = out_dir(cfg)
    cl = MassiveClient.from_config(cfg, resolve(cfg, cfg["data"]["cache_dir"]))
    cl.preload_splits(cfg["data"]["start"])
    t0 = time.time()
    log = lambda m: print(f"[{datetime.now():%H:%M}] {m}", flush=True)

    gf = od / "gappers40.parquet"
    if gf.exists():
        g = pd.read_parquet(gf)
    else:
        cands = pd.read_parquet(resolve(cfg, "output") / "candidates.parquet")
        c2 = json.loads(json.dumps(cfg))
        c2["scan"]["min_premarket_gap"] = V2_MIN_GAP
        log(f"Checking {len(cands):,} candidates for +40% gappers (new ones need a details call) ...")
        g = stage2(cl, c2, cands, progress=lambda i, n, t, d: i % 250 == 0 and log(f"  {i:,}/{n:,}"))
        nd, prev = next_days(g, resolve(cfg, cfg["data"]["cache_dir"]) / "grouped")
        g["next10"] = [nd.get((t, d), []) for t, d in zip(g.ticker, g.date)]
        g["prev_day"] = [prev.get((t, d)) for t, d in zip(g.ticker, g.date)]
        g.to_parquet(gf)
        log(f"  {len(g):,} gappers at +40% or more ({cl.calls} API calls)")

    tickers = sorted(set(g.ticker))
    log(f"News: {len(tickers):,} tickers ...")
    for i, t in enumerate(tickers):
        ticker_news(cl, t, cfg["data"]["start"], cfg["data"]["end"] + "T23:59:59Z")
        if i % 50 == 0:
            log(f"  news {i:,}/{len(tickers):,}  ({cl.calls} API calls, {time.time() - t0:.0f}s)")
    log("PREP DONE")


if __name__ == "__main__":
    main()
