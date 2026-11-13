"""Self-contained HTML report (one file, no internet needed to open it)."""
from __future__ import annotations

import html
import math

import pandas as pd

from tfs.portfolio import simulate_account
from tfs.stats import by_group, rule_table

CSS = """
:root{--bg:#f5f6f8;--surface:#fff;--surface-2:#eef0f4;--ink:#151a28;--ink-2:#4a5164;--muted:#767e93;
--line:#dde1e9;--accent:#2546c7;--good:#0c7a3a;--bad:#b3261e;--warn-bg:#fff4dc;--warn:#8a5c00;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1320;--surface:#171c2c;--surface-2:#1f2537;
--ink:#eef0f6;--ink-2:#b8bed0;--muted:#8a92a8;--line:#2b3247;--accent:#7f99ff;--good:#4cc07a;--bad:#f07a73;
--warn-bg:#2d2512;--warn:#e2a63c;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f1320;--surface:#171c2c;--surface-2:#1f2537;--ink:#eef0f6;--ink-2:#b8bed0;
--muted:#8a92a8;--line:#2b3247;--accent:#7f99ff;--good:#4cc07a;--bad:#f07a73;--warn-bg:#2d2512;--warn:#e2a63c;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15.5px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:28px 16px 64px}h1{font-size:30px;margin:0 0 6px;line-height:1.2}
h2{font-size:22px;margin:36px 0 8px}h3{font-size:17px;margin:22px 0 6px}p{max-width:72ch;margin:0 0 10px}
.muted{color:var(--muted)}.banner{background:var(--warn-bg);color:var(--warn);border-radius:10px;padding:12px 14px;font-weight:600;margin:14px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:14px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px}
.card .k{font-size:12.5px;color:var(--muted)}.card .v{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface);margin:8px 0 16px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:var(--surface-2);color:var(--ink-2);font-weight:600;position:sticky;top:0}
td:first-child,th:first-child{text-align:left}tr:last-child td{border-bottom:none}
.pos{color:var(--good)}.neg{color:var(--bad)}code{font-size:13px;background:var(--surface-2);padding:1px 5px;border-radius:4px}
svg text{fill:var(--muted);font-size:11px}.chart{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:10px}
"""

PCT_COLS = {"win_rate", "avg_win", "avg_loss", "expectancy", "median", "gross_expectancy", "worst",
            "mae_p95", "stopped_pct", "pm_gap", "total_return", "max_drawdown", "ret", "mae"}


def _fmt(col, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if col in PCT_COLS and isinstance(v, (int, float)):
        signed = ("expectancy", "median", "gross_expectancy", "ret")
        cls = "pos" if (col in signed + ("avg_win",) and v > 0) else ("neg" if col in signed and v < 0 else "")
        return f'<span class="{cls}">{v:+.1%}</span>' if col not in ("win_rate", "stopped_pct", "mae_p95") \
            else f"{v:.0%}"
    if col in ("entry_px", "exit_px") and isinstance(v, (int, float)):
        return f"${v:,.2f}"
    if isinstance(v, float):
        return "∞" if math.isinf(v) else f"{v:,.2f}"
    return html.escape(str(v))


def table(df: pd.DataFrame, cols: list[str], labels: dict | None = None, limit: int | None = None) -> str:
    if df is None or df.empty:
        return '<p class="muted">No data.</p>'
    labels = labels or {}
    d = df.head(limit) if limit else df
    head = "".join(f"<th>{html.escape(labels.get(c, c))}</th>" for c in cols if c in d.columns)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(c, r.get(c))}</td>" for c in cols if c in d.columns) + "</tr>"
                   for r in d.to_dict("records"))
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def equity_svg(curve: pd.DataFrame, start: float) -> str:
    if curve.empty:
        return '<p class="muted">No trades.</p>'
    W, H, P = 960, 260, 44
    y = [start] + curve["equity"].tolist()
    lo, hi = min(y + [start]) * 0.98, max(y + [start]) * 1.02
    def X(i): return P + (W - 2 * P) * i / max(1, len(y) - 1)
    def Y(v): return H - P + (-(H - 2 * P)) * (v - lo) / (hi - lo or 1)
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(y))
    ticks = "".join(f'<line x1="{P}" x2="{W-P}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--line)"/>'
                    f'<text x="{P-6}" y="{Y(v)+4:.1f}" text-anchor="end">${v:,.0f}</text>'
                    for v in [lo + (hi - lo) * k / 4 for k in range(5)])
    base = f'<line x1="{P}" x2="{W-P}" y1="{Y(start):.1f}" y2="{Y(start):.1f}" stroke="var(--muted)" stroke-dasharray="4 4"/>'
    return (f'<div class="chart"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Account equity after each trade">'
            f'{ticks}{base}<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="{pts}"/>'
            f'<text x="{W-P}" y="{H-P+18}" text-anchor="end">trade #{len(y)-1}</text>'
            f'<text x="{P}" y="{H-P+18}">start</text></svg></div>')


RULE_COLS = ["rule", "trades", "win_rate", "avg_win", "avg_loss", "expectancy", "gross_expectancy", "median",
             "profit_factor", "t_stat", "mae_p95", "stopped_pct", "skipped_dnt", "skipped_above_stop"]
LABELS = {"trades": "trades", "win_rate": "win %", "avg_win": "avg win", "avg_loss": "avg loss",
          "expectancy": "avg / trade (net)", "gross_expectancy": "before costs", "profit_factor": "profit factor",
          "t_stat": "t-stat", "mae_p95": "MAE p95", "stopped_pct": "stopped", "skipped_dnt": "DNT skips",
          "skipped_above_stop": "above stop"}


def build_report(cfg, gappers, feats, trades, synthetic: bool, holdout: pd.DataFrame | None = None,
                 holdout_runs: int = 0, scan_complete: bool = True) -> str:
    be = cfg["split"]["build_end"]
    build = trades[trades["date"] <= be]
    n_build_g = int((gappers["date"] <= be).sum())
    rt = rule_table(build)
    ranked = rt[rt["trades"] >= cfg["report"]["min_trades"]] if not rt.empty else rt
    rt_nohtb = rule_table(build, exclude_htb=True)
    rt_nohtb = rt_nohtb[rt_nohtb["trades"] >= cfg["report"]["min_trades"]] if not rt_nohtb.empty else rt_nohtb
    best = ranked.iloc[0]["rule"] if not ranked.empty else None
    fb = feats[feats["date"] <= be]

    parts = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width,initial-scale=1'><title>TFS Backtest Report</title><style>{CSS}</style></head>"
             f"<body><div class='wrap'>"]
    parts.append("<p class='muted'>TFS · research bot · Plan T (technical)</p><h1>Pre-market gapper fade: 5-year backtest</h1>")
    if synthetic:
        parts.append("<div class='banner'>SYNTHETIC DEMO DATA. These numbers come from a fake market used to test the "
                     "software. They say nothing about real stocks.</div>")
    if not scan_complete:
        parts.append("<div class='banner'>Partial scan: only stocks that opened 30%+ above the prior close have been "
                     "checked so far (pass 1). Pass 2 adds gappers that fell back before the open; until it finishes, "
                     "results may be slightly too kind to shorting.</div>")
    parts.append(f"<p>Every US-listed common stock up {cfg['scan']['min_premarket_gap']:.0%}+ at "
                 f"{cfg['scan']['premarket_cutoff']} ET with ≥ ${cfg['scan']['min_price']:.2f} price and ≥ "
                 f"{cfg['scan']['min_premarket_volume']:,} pre-market shares, {cfg['data']['start']} to {cfg['data']['end']}. "
                 f"Rules are tuned on the <b>build period</b> (to {be}) only. The test period stays sealed until you "
                 f"pick rules and run <code>python -m tfs holdout</code> once.</p>")
    cards = [("Gappers found", f"{len(gappers):,}"), ("In build period", f"{n_build_g:,}"),
             ("Rule variants", f"{rt['rule'].nunique() if not rt.empty else 0}"),
             ("Opened below PMH", f"{fb['open_below_pmh'].mean():.0%}" if "open_below_pmh" in fb and len(fb) else "–"),
             ("Closed below open", f"{(fb['day_return_from_open'] < 0).mean():.0%}" if "day_return_from_open" in fb and len(fb) else "–"),
             ("Likely hard to borrow", f"{fb['likely_htb'].mean():.0%}" if "likely_htb" in fb and len(fb) else "–")]
    parts.append("<div class='cards'>" + "".join(f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div></div>"
                                                 for k, v in cards) + "</div>")

    parts.append("<h2>Every rule, ranked (build period, after all costs)</h2>"
                 f"<p>“avg / trade” is the average return per trade on the money shorted, after slippage, commission, "
                 f"locate and borrow fees. Rules with fewer than {cfg['report']['min_trades']} trades are not ranked. A t-stat "
                 f"under about 2 means the result could easily be luck. With {rt['rule'].nunique() if not rt.empty else 0} "
                 f"variants tested, expect one or two to look good by chance alone.</p>")
    parts.append(table(ranked, RULE_COLS, LABELS))
    parts.append("<h3>Same table without the likely hard-to-borrow names</h3><p>Gappers above +200% or with float "
                 "rotation above 3x are usually impossible to borrow. If a rule only works on those, it does not work.</p>")
    parts.append(table(rt_nohtb, RULE_COLS, LABELS, limit=15))

    if best:
        parts.append(f"<h2>Closest look: <code>{html.escape(best)}</code></h2><p>The top-ranked rule above. Being on "
                     f"top of {len(ranked)} variants is not proof; check that it holds up across years and gap sizes.</p>")
        bt = build.assign(year=build["date"].str[:4])
        parts.append("<h3>By year</h3>" + table(by_group(bt, best, "year"),
                     ["year", "trades", "win_rate", "expectancy", "median", "worst", "mae_p95"], LABELS))
        fm = feats.set_index(["date", "ticker"])["gap_bucket"] if "gap_bucket" in feats else None
        if fm is not None:
            bb = build.join(fm, on=["date", "ticker"])
            parts.append("<h3>By gap size</h3>" + table(by_group(bb, best, "gap_bucket"),
                         ["gap_bucket", "trades", "win_rate", "expectancy", "median", "worst", "mae_p95"], LABELS))
        curve, summ = simulate_account(build[build["rule"] == best], cfg)
        parts.append(f"<h3>${cfg['risk']['starting_equity']:,} account, all risk rules on</h3>"
                     f"<p>{cfg['risk']['risk_per_trade']:.2%} risk per trade, {cfg['risk']['daily_loss_limit']:.0%} daily "
                     f"loss limit, {cfg['risk']['max_short_exposure']:.0%} exposure cap, max "
                     f"{cfg['risk']['max_trades_per_day']} trades a day, {cfg['risk']['margin_rule']} margin.</p>")
        if summ.get("trades"):
            parts.append("<div class='cards'>" + "".join(
                f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in [
                    ("Ending equity", f"${summ['end']:,.0f}"), ("Total return", f"{summ['total_return']:+.1%}"),
                    ("Max drawdown", f"{summ['max_drawdown']:.1%}"), ("Worst trade", f"{'−' if summ['worst_trade'] < 0 else ''}${abs(summ['worst_trade']):,.0f}"),
                    ("Trades taken", f"{summ['trades']}"), ("Skipped by limits",
                                                             f"{summ['daily_loss_limit'] + summ['max_trades_per_day']}")]) + "</div>")
        parts.append(equity_svg(curve, cfg["risk"]["starting_equity"]))
        w = build[(build["rule"] == best) & build["status"].isin(["filled", "exit_data_missing"])].nsmallest(10, "ret")
        parts.append("<h3>The 10 worst trades</h3>" + table(w, ["date", "ticker", "pm_gap", "entry_time", "exit_date",
                                                              "exit_time", "entry_px", "exit_px", "ret", "mae"],
                                                             {"pm_gap": "gap", "ret": "return", "mae": "MAE"}))

    if "hod_time" in fb and len(fb):
        h = fb["hod_time"].dropna().str[:2].value_counts().sort_index()
        hd = pd.DataFrame({"hour": [f"{k}:00" for k in h.index], "gappers": h.values,
                           "share": (h.values / h.sum())})
        parts.append("<h2>When does the day's high print?</h2><p>Time of the regular-session high, build period.</p>"
                     + table(hd.rename(columns={"share": "win_rate"}), ["hour", "gappers", "win_rate"],
                             {"win_rate": "share"}))

    parts.append("<h2>Sealed test period</h2>")
    if holdout is not None and not holdout.empty:
        warn = (f"<div class='banner'>The test period has been opened {holdout_runs} times. Every extra look makes "
                f"these numbers less trustworthy.</div>") if holdout_runs > 1 else ""
        parts.append(warn + "<p>Results on data the rules were never tuned on. This is the number that counts.</p>"
                     + table(holdout, RULE_COLS, LABELS))
    else:
        parts.append("<p class='muted'>Not opened yet. Pick 1 to 3 rules from the build tables, write them under "
                     "<code>holdout_rules</code> in config.yaml, then run <code>python -m tfs holdout</code>.</p>")

    parts.append("<h2>What this backtest cannot see</h2><ul>"
                 "<li><b>Borrow availability.</b> Free history does not say whether shares could be borrowed. The second "
                 "table drops the names that usually cannot be. Live trading will skip anything the broker cannot lend.</li>"
                 "<li><b>Float.</b> Shares outstanding is used as the float. Real float is smaller, so rotation is understated.</li>"
                 "<li><b>Halts</b> are detected from gaps in the 1-minute bars. Close, not exact.</li>"
                 "<li><b>Fills.</b> Market-order fills at the next bar's open plus slippage. Thin stocks can be worse.</li>"
                 "<li><b>Fewer setups ahead.</b> Nasdaq's 2025–2026 listing rules cut micro-cap IPOs, so future gappers may be rarer.</li>"
                 "</ul><p class='muted'>Educational research. Not financial advice. Short selling can lose more than the account.</p>")
    parts.append("</div></body></html>")
    return "".join(parts)
