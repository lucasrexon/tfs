"""Command line.

  python -m tfs demo        whole pipeline on a fake market (no key needed)
  python -m tfs scan        find 5 years of gappers (downloads + caches data)
  python -m tfs backtest    simulate every rule on every gapper
  python -m tfs report      write reports/report.html (build period only)
  python -m tfs run         scan + backtest + report
  python -m tfs holdout     open the sealed test period for config.holdout_rules
  python -m tfs card        live: print this morning's gapper cards
  python -m tfs bot         live: run the trading bot (paper by default)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from tfs.config import load_config, resolve


def _client(cfg, synthetic):
    if synthetic:
        from tfs.synthetic import FakeClient
        return FakeClient(cfg["data"]["start"], cfg["data"]["end"])
    from tfs.data.massive import MassiveClient
    return MassiveClient.from_config(cfg, resolve(cfg, cfg["data"]["cache_dir"]))


def _out(cfg, synthetic) -> Path:
    p = resolve(cfg, "output_demo" if synthetic else "output")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tick(every=50):
    t0 = time.time()

    def f(i, *info):
        if i % every == 0:
            print(f"  [{time.time() - t0:6.0f}s] {i} {' '.join(map(str, info))}", flush=True)
    return f


def cmd_scan(cfg, synthetic=False, then=None):
    """Stage 1, then stage 2 in two passes. `then` runs after each pass (backtest + report)."""
    from tfs.data.massive import split_factor
    from tfs.scan import stage1, stage2
    c, out = _client(cfg, synthetic), _out(cfg, synthetic)
    sc = cfg["scan"]
    print("Universe: listed + delisted common stocks ...")
    uni = c.tickers(cfg["data"]["stock_types"])
    print(f"  {len(uni):,} tickers")
    if hasattr(c, "preload_splits"):
        print(f"  {c.preload_splits(cfg['data']['start'])} splits loaded in bulk")
    print("Stage 1: daily bars for the whole market ...")
    cands = stage1(c, cfg, uni, progress=_tick(100))
    # Necessary conditions, no false negatives: pre-market volume >= X needs day volume >= X,
    # and a $1 pre-market price needs a day high >= $1 (both in real, split-unadjusted terms).
    f = [split_factor(c.splits(t), d) for t, d in zip(cands["ticker"], cands["date"])]
    cands = cands[(cands["d_volume"] / pd.Series(f, index=cands.index) >= sc["min_premarket_volume"]) &
                  (cands["d_high"] * pd.Series(f, index=cands.index) >= sc["min_price"])]
    cands.to_parquet(out / "candidates.parquet")
    tier1 = cands[cands["d_open"] >= sc["pass1_open_ratio"] * cands["prev_close"]]
    tier2 = cands.drop(tier1.index)
    print(f"  {len(cands):,} candidates: pass 1 {len(tier1):,} (opened >= {sc['pass1_open_ratio']}x), pass 2 {len(tier2):,}")
    found = []
    for name, part in (("pass 1", tier1), ("pass 2", tier2)):
        print(f"Stage 2 {name}: minute bars, +50% at 09:25 check ...")
        g = stage2(c, cfg, part, progress=_tick(100))
        found.append(g.assign(scan_pass=name))
        allg = pd.concat(found, ignore_index=True)
        allg.to_parquet(out / "gappers.parquet")
        print(f"  {name}: {len(g):,} gappers ({len(allg):,} total, {getattr(c, 'calls', '?')} API calls)")
        if then:
            then()
    return allg


def cmd_backtest(cfg, synthetic=False):
    from tfs.backtest import run_backtest
    c, out = _client(cfg, synthetic), _out(cfg, synthetic)
    g = pd.read_parquet(out / "gappers.parquet")
    print(f"Backtesting {len(g):,} gappers ...")
    feats, trades = run_backtest(g, c.minute_bars, cfg, progress=_tick(200))
    feats.to_parquet(out / "features.parquet")
    trades.to_parquet(out / "trades.parquet")
    feats.to_csv(out / "features.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)
    print(f"  {len(trades):,} rule-trades written to {out}")
    return feats, trades


def _load(out):
    return (pd.read_parquet(out / "gappers.parquet"), pd.read_parquet(out / "features.parquet"),
            pd.read_parquet(out / "trades.parquet"))


def cmd_report(cfg, synthetic=False, holdout=None, runs=0):
    from tfs.report import build_report
    out = _out(cfg, synthetic)
    g, f, t = _load(out)
    if holdout is None and (out / "holdout_results.parquet").exists():
        holdout = pd.read_parquet(out / "holdout_results.parquet")
        runs = len(json.loads((out / "holdout_log.json").read_text()))
    html = build_report(cfg, g, f, t, synthetic, holdout, runs)
    rp = resolve(cfg, "reports")
    rp.mkdir(exist_ok=True)
    path = rp / ("demo_report.html" if synthetic else "report.html")
    path.write_text(html)
    print(f"Report: {path}")
    return path


def cmd_holdout(cfg, synthetic=False):
    from tfs.stats import rule_table
    rules = cfg.get("holdout_rules") or []
    if not rules:
        sys.exit("Write 1-3 rules under holdout_rules in config.yaml first (e.g. vwap_loss|eod|pmh_plus_5).")
    if len(rules) > 5:
        sys.exit("At most 5 holdout rules. Testing many rules on the holdout turns it into another build set.")
    out = _out(cfg, synthetic)
    _, _, t = _load(out)
    test = t[(t["date"] > cfg["split"]["build_end"]) & t["rule"].isin(rules)]
    log_f = out / "holdout_log.json"
    log = json.loads(log_f.read_text()) if log_f.exists() else []
    log.append({"time": datetime.now().isoformat(timespec="seconds"), "rules": rules})
    log_f.write_text(json.dumps(log, indent=1))
    if len(log) > 1:
        print(f"WARNING: this is look #{len(log)} at the test period. Results are now less trustworthy.")
    res = rule_table(test)
    res.to_parquet(out / "holdout_results.parquet")
    print(res.to_string(index=False))
    cmd_report(cfg, synthetic, res, len(log))


def demo_config(cfg):
    cfg = json.loads(json.dumps(cfg))
    cfg["data"].update(start="2023-01-02", end="2024-12-31")
    cfg["split"]["build_end"] = "2024-06-30"
    cfg["report"]["min_trades"] = 20
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tfs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["demo", "download", "scan", "backtest", "report", "run", "holdout", "card", "bot"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="use the fake market")
    ap.add_argument("--start", help="override data.start (YYYY-MM-DD)")
    ap.add_argument("--end", help="override data.end")
    ap.add_argument("--rpm", type=int, help="max requests per minute (free Massive plan: 5)")
    ap.add_argument("--build-end", help="override split.build_end")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    for k, v in (("start", a.start), ("end", a.end), ("max_requests_per_minute", a.rpm)):
        if v is not None:
            cfg["data"][k] = v
    if a.build_end:
        cfg["split"]["build_end"] = a.build_end
    syn = a.synthetic or a.command == "demo"
    if syn:
        cfg = demo_config(cfg)
    if a.command == "demo":
        cmd_scan(cfg, syn); cmd_backtest(cfg, syn); cmd_report(cfg, syn)
    elif a.command == "download":
        from tfs.scan import trading_days
        c = _client(cfg, syn)
        for i, (d, df) in enumerate(trading_days(c, cfg["data"]["start"], cfg["data"]["end"])):
            if i % 20 == 0:
                print(f"  {d}: {len(df):,} stocks", flush=True)
        print("Daily data cached.")
    elif a.command == "scan":
        cmd_scan(cfg, syn)
    elif a.command == "backtest":
        cmd_backtest(cfg, syn)
    elif a.command == "report":
        cmd_report(cfg, syn)
    elif a.command == "run":
        cmd_scan(cfg, syn, then=lambda: (cmd_backtest(cfg, syn), cmd_report(cfg, syn)))
    elif a.command == "holdout":
        cmd_holdout(cfg, syn)
    elif a.command in ("card", "bot"):
        from tfs.live.bot import run_bot, morning_cards
        (morning_cards if a.command == "card" else run_bot)(cfg)
