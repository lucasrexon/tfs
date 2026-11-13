"""One row of features per gapper: technical (known at 09:25), SEC filings, news category.

Resumable: SEC and news answers are cached, so rerunning continues where it stopped.
    python -m tfs.v2.features
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tfs.config import load_config, resolve
from tfs.news import categorize, pregap_headlines
from tfs.sec import Edgar, company_features, dilution_pressure

ITEM_HINT = {"2.02": "earnings", "3.02": "offering_financing", "1.01": "contract_partnership", "2.01": "merger_buyout"}


def load_gappers(cfg) -> pd.DataFrame:
    v2 = resolve(cfg, "output") / "v2" / "gappers40.parquet"
    return pd.read_parquet(v2 if v2.exists() else resolve(cfg, "output") / "gappers.parquet")


def trading_days(cfg) -> list[str]:
    return sorted(p.stem for p in (resolve(cfg, cfg["data"]["cache_dir"]) / "grouped").glob("*.parquet")
                  if p.stat().st_size > 20000)


def main():
    cfg = load_config()
    cache = resolve(cfg, cfg["data"]["cache_dir"])
    od = resolve(cfg, "output") / "v2"
    od.mkdir(parents=True, exist_ok=True)
    g = load_gappers(cfg)
    days = trading_days(cfg)
    prev_of = {d: (days[i - 1] if i else None) for i, d in enumerate(days)}
    ed = Edgar(cache)
    splits_f = next(iter(cache.glob("splits_since_*.json")), None)
    splits = {}
    for r in (json.loads(splits_f.read_text()) if splits_f else []):
        splits.setdefault(r["ticker"], []).append(r)
    rows = []
    for i, r in enumerate(g.to_dict("records")):
        if i % 100 == 0:
            print(f"[{datetime.now():%H:%M}] features {i:,}/{len(g):,}  (SEC calls {ed.calls})", flush=True)
        det_f = cache / "details" / f"{r['ticker']}_{r['date']}.json"
        det = json.loads(det_f.read_text()) if det_f.exists() else {}
        prev_day = prev_of.get(r["date"])
        real_prev = r["prev_close"] * r["factor"]
        row = {"date": r["date"], "ticker": r["ticker"], "pm_gap": r["pm_gap"],
               "pmh_ext": r["pm_high"] / r["prev_close"] - 1, "pullback": 1 - r["pm_last"] / r["pm_high"],
               "price": r["pm_price_real"], "prev_price": real_prev,
               "pm_volume": r["pm_volume_real"], "pm_dollar_vol": r["pm_volume_real"] * r["pm_price_real"],
               "shares_out": r.get("shares_out"), "float_rotation": r.get("float_rotation"),
               "mcap_before": (real_prev * r["shares_out"]) if r.get("shares_out") else None,
               "runner_days": r.get("runner_days"), "days_since_ipo": r.get("days_since_ipo"),
               "ssr_carry": int(bool(r.get("prev_prev_close") and r["prev_low"] <= 0.9 * r["prev_prev_close"])),
               "sub_dollar_before": int(real_prev < 1)}
        texts, source = [], "none"
        cik = det.get("cik")
        if cik:
            try:
                f = company_features(ed, cik, r["date"], prev_day, det.get("description", ""), splits.get(r["ticker"]))
                for acc, form, primary in f.pop("_morning")[:2]:
                    t = ed.press_release_text(cik, acc, primary)
                    if t:
                        texts.append(t[:1500]); source = "sec"
                row.update(f)
                row["dilution_pressure"] = dilution_pressure(f)
            except Exception as e:                       # one bad company must not stop the run
                row["sec_error"] = str(e)[:80]
        news_f = cache / "news" / f"{r['ticker']}.json"
        if news_f.exists() and prev_day:
            hl = pregap_headlines(json.loads(news_f.read_text()), r["date"], prev_day, cfg["scan"]["premarket_cutoff"])
            if hl:
                texts = [h["title"] for h in hl] + texts
                source = "news" if source == "none" else "news+sec"
            row["headline"] = hl[0]["title"][:160] if hl else None
            row["n_headlines"] = len(hl)
        cat = categorize(texts)
        if cat in ("no_news_found", "other_news"):
            for it in (row.get("morning_items") or "").split(","):
                if it in ITEM_HINT:
                    cat = ITEM_HINT[it]
                    break
        row["catalyst"], row["news_source"] = cat, source
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(od / "features.parquet")
    print(f"[{datetime.now():%H:%M}] FEATURES DONE {len(out):,} rows, SEC calls {ed.calls}", flush=True)
    print(out["catalyst"].value_counts().to_string())
    print(out["industry"].value_counts().to_string() if "industry" in out else "no SEC data")


if __name__ == "__main__":
    main()
