"""Build reports/patterns.html from the tfs.v2.patterns outputs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tfs.config import load_config, resolve

NICE = {
    "medical_health": "Medical devices & health services", "biotech_pharma": "Biotech & pharma", "tech": "Tech & telecom",
    "finance_shell": "Finance & shell companies", "industrial_ev": "Industrial & EV", "consumer": "Consumer (food, retail, leisure)",
    "crypto": "Crypto", "energy_mining": "Energy & mining", "other": "Other services (business, education, consulting)",
    "none": "No news found", "fda_regulatory": "FDA / regulatory", "financing": "Financing deal", "being_acquired": "Being bought out",
    "other_news": "Other news", "clinical_data": "Clinical trial data", "earnings": "Earnings", "merger_combination": "Merger / combination",
    "partnership_license": "Partnership / licensing", "company_acquires": "Company buys something", "hot_theme": "Hot theme (AI, quantum, drones, defense)",
    "filing_no_text": "SEC filing, no text", "crypto_treasury": "Crypto treasury", "movers_article_only": "Only a 'stock soars' article",
    "contract_order": "Contract / order", "compliance_split": "Reverse split / compliance",
    "overnight": "Overnight (before 4:00)", "early_premarket": "Early pre-market (4:00-8:00)", "late_premarket": "Late pre-market (8:00-9:25)",
    "SEC filing only": "SEC filing only", "news wire or media": "News wire or media", "warning": "Has a going-concern warning",
    "no warning": "No going-concern warning", "news by 09:25": "Some news by 9:25", "no news by 09:25": "No news by 9:25",
    "news in prior days": "News in the 4 days before", "no news in prior days": "No news in the 4 days before",
}


def nice(v: str) -> str:
    if " + " in v:
        a, b = v.split(" + ", 1)
        return f"{NICE.get(a, a)} + {('no news' if b == 'no news' else NICE.get(b, b))}"
    return NICE.get(v, v)


def main():
    cfg = load_config()
    od = resolve(cfg, "output") / "v2"
    g = pd.read_csv(od / "patterns_groups.csv")
    ft = pd.read_csv(od / "patterns_filters.csv")
    fin = json.loads((od / "patterns_final.json").read_text())
    tr = pd.read_csv(od / "patterns_trades.csv")

    def r(v, k=4):
        return None if pd.isna(v) else round(float(v), k)
    groups = [{"dim": x.dimension, "key": x.group, "name": nice(x.group), "n": int(x.n), "nb": int(x.n_build), "nt": int(x.n_test),
               "avg": r(x.avg_trade), "rest": r(x.avg_rest), "b": r(x.avg_build), "t": r(x.avg_test), "win": r(x.win_rate),
               "big": r(x.big_loss_rate), "p": r(x.p), "q": r(x.q), "sig": x.significance} for x in g.itertuples()]
    filt = []
    for x in ft.itertuples():
        filt.append({"base": x._1, "learn": x._2, "score": x._3, "filter": x.filter, "skipped": x.skipped, "trades": int(x.trades),
                     "avg": r(x.avg_trade), "typ": r(x.typical_yr), "bad": r(x.bad_luck_yr), "dd": r(x.worst_drop)})
    trades = []
    for x in tr.itertuples():
        trades.append([x.year, x.date, x.ticker, (x.name or "")[:48] if isinstance(x.name, str) else "", nice(x.industry) if isinstance(x.industry, str) else "",
                       1 if x.going_concern == "warning" else 0, nice(x.news_0925), x.news_timing,
                       x.headline_before_0925 if isinstance(x.headline_before_0925, str) else "",
                       nice(x.news_known_later), x.headline_later_today if isinstance(x.headline_later_today, str) else "",
                       r(x.pm_gap, 3), r(x.ret, 4), int(bool(x.hit)), int(bool(x.stopped))])
    data = {"groups": groups, "filters": filt, "final": fin, "trades": trades,
            "coverage": {"trades": len(tr), "news_0925": int((tr.news_0925 != "none").sum()),
                         "news_any": int((tr.news_known_later != "none").sum()),
                         "build": int((tr.year == "build").sum()), "test": int((tr.year == "test").sum())}}
    html = (resolve(cfg, "dashboard_page") / "patterns_template.html").read_text()
    out = resolve(cfg, "reports") / "patterns.html"
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    print(out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
