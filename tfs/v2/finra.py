"""FINRA daily short-volume files: every exchange-listed stock that traded each day.

Used as the stock list for years the Massive free plan cannot reach (its grouped
daily bars only go back 2 years). The files include stocks that were later
delisted, so the older years are not limited to survivors. Each day is saved
as data_cache/finra/<YYYY-MM-DD>.parquet with symbol, total and short volume.

    python -m tfs.v2.finra 2020-10-01 2024-11-08
"""
from __future__ import annotations

import io
import sys
import time
from datetime import date as Date, timedelta

import pandas as pd
import requests

from tfs.config import load_config, resolve

URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"


def parse(text: str) -> pd.DataFrame:
    """Parse one file; the last line is a record count, not a stock."""
    df = pd.read_csv(io.StringIO(text), sep="|", dtype={"Symbol": str})
    df = df[df["Symbol"].notna() & df["TotalVolume"].notna()]
    df = df.rename(columns={"Symbol": "symbol", "ShortVolume": "short_volume",
                            "ShortExemptVolume": "short_exempt_volume", "TotalVolume": "total_volume"})
    return df[["symbol", "short_volume", "short_exempt_volume", "total_volume"]].reset_index(drop=True)


def day_file(cache, day: str, session: requests.Session) -> pd.DataFrame | None:
    """One day's file, from the cache or FINRA. None on weekends and market holidays."""
    f = cache / "finra" / f"{day}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    missing = cache / "finra" / "missing.txt"
    if missing.exists() and day in missing.read_text().split():
        return None
    for attempt in range(4):
        r = session.get(URL.format(d=day.replace("-", "")), timeout=60)
        if r.status_code in (403, 404):        # no file: market closed that day
            missing.parent.mkdir(parents=True, exist_ok=True)
            with open(missing, "a") as fh:
                fh.write(day + "\n")
            return None
        if r.status_code == 200:
            df = parse(r.text)
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(f)
            return df
        time.sleep(2 ** attempt)
    raise RuntimeError(f"FINRA {day}: HTTP {r.status_code}")


def main(start: str, end: str):
    cfg = load_config()
    cache = resolve(cfg, cfg["data"]["cache_dir"])
    s = requests.Session()
    d, e, n = Date.fromisoformat(start), Date.fromisoformat(end), 0
    while d <= e:
        if d.weekday() < 5:
            df = day_file(cache, d.isoformat(), s)
            n += df is not None
            if d.day == 1 or (d.weekday() == 0 and d.day <= 3):
                print(f"{d}: {n} trading days so far", flush=True)
        d += timedelta(days=1)
    print(f"FINRA DONE {n} trading days", flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("2020-10-01", "2024-11-08")))
