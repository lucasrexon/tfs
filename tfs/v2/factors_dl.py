"""After-hours and news-sentiment factors (needs tfs.v2.dl_extra first). Builds output/v2/features4.parquet.

    python -m tfs.v2.factors_dl

  ah_move    last after-hours price on the day before (16:00-20:00) vs that day's close
  ah_share   how much of the 09:25 gap was already made after hours (ah_move / pm_gap)
  sentiment  Massive's sentiment for this ticker in articles from 16:00 the day before to 09:25:
             positive / negative / neutral / mixed, or "no article"
"""
from __future__ import annotations

import json

import pandas as pd

from tfs.config import load_config, resolve
from tfs.data.massive import NY


def main():
    cfg = load_config()
    cache = resolve(cfg, cfg["data"]["cache_dir"])
    od = resolve(cfg, "output") / "v2"
    g = pd.read_parquet(od / "gappers40.parquet")[["date", "ticker", "prev_day", "prev_close", "pm_gap"]]
    rows = []
    for r in g.itertuples():
        row = {"date": r.date, "ticker": r.ticker}
        mf = cache / "minute" / str(r.prev_day) / f"{r.ticker}.parquet"
        if mf.exists():
            b = pd.read_parquet(mf)
            day = pd.Timestamp(r.prev_day, tz=NY)
            ah = b[(b.index >= day + pd.Timedelta("16h")) & (b.index < day + pd.Timedelta("20h"))]
            if len(ah) and r.prev_close:
                row["ah_move"] = float(ah["c"].iloc[-1]) / r.prev_close - 1
                row["ah_share"] = row["ah_move"] / r.pm_gap if r.pm_gap else None
                row["ah_volume"] = float(ah["v"].sum())
            else:
                row["ah_move"], row["ah_share"] = 0.0, 0.0              # no after-hours trades: nothing moved
        nf = cache / "news_ins" / f"{r.ticker}.json"
        if nf.exists():
            lo, hi = pd.Timestamp(f"{r.prev_day} 16:00", tz=NY), pd.Timestamp(f"{r.date} 09:25", tz=NY)
            s = [a["sentiment"] for a in json.loads(nf.read_text()) if a.get("sentiment")
                 and lo <= pd.Timestamp(a["published_utc"]).tz_convert(NY) < hi]
            row["sentiment"] = ("no article" if not s else s[0] if len(set(s)) == 1 else "mixed")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(od / "features4.parquet")
    print("FEATURES4 DONE", len(out), out[["ah_move", "ah_share"]].notna().sum().to_dict() if "ah_move" in out else {},
          out["sentiment"].value_counts().to_dict() if "sentiment" in out else {})


if __name__ == "__main__":
    main()
