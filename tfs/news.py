"""News headlines per gapper (Massive) and keyword categories.

The bot may only use news published before the scan time, so a headline counts
for a gap day if it came out after 16:00 ET on the previous trading day and
before the scan (09:25 ET by default).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from tfs.data.massive import NY

# Checked in this order; the first category that matches wins.
CATEGORIES: list[tuple[str, list[str]]] = [
    ("merger_buyout", [r"\bmerger\b", r"\bto be acquired\b", r"\bacquire[sd]?\b", r"\bacquisition\b", r"\bbuyout\b",
                       r"\btender offer\b", r"\btakeover\b", r"business combination", r"definitive agreement",
                       r"\bmajority stake\b", r"\bgoing private\b"]),
    ("offering_financing", [r"\boffering\b", r"\bpriced\b", r"private placement", r"registered direct", r"\bwarrants?\b",
                            r"\bfinancing\b", r"\bconvertible\b", r"at[- ]the[- ]market", r"equity line", r"\braises?\b",
                            r"\bpipe\b", r"\bproceeds\b"]),
    ("fda_clinical", [r"\bfda\b", r"\bclearance\b", r"\bphase [1-3i]+\b", r"\btrial\b", r"\btopline\b", r"orphan drug",
                      r"fast track", r"breakthrough", r"\bind\b", r"\bnda\b", r"\bbla\b", r"\bpdufa\b", r"510\(k\)",
                      r"\bce mark\b", r"\bpatients?\b", r"\bclinical\b", r"\btherapy\b", r"\bdrug\b"]),
    ("contract_partnership", [r"\bcontract\b", r"\bpartnership\b", r"\bcollaboration\b", r"purchase order", r"\border\b",
                              r"\bawarded\b", r"\bselected\b", r"\bmou\b", r"letter of intent", r"\bdistribution\b",
                              r"\blicens", r"\bagreement\b", r"\bdeal\b", r"\bpartner"]),
    ("crypto_ai_pivot", [r"\bbitcoin\b", r"\bcrypto", r"\bblockchain\b", r"\btoken", r"\btreasury\b", r"digital asset",
                         r"\bethereum\b", r"\bsolana\b", r"\bai\b", r"artificial intelligence", r"\bquantum\b", r"\bdrone"]),
    ("earnings", [r"\bresults\b", r"\brevenue\b", r"\bquarter\b", r"\bfiscal\b", r"\bearnings\b", r"\bguidance\b"]),
    ("compliance_split", [r"reverse (stock )?split", r"nasdaq notice", r"regain(s|ed)? compliance", r"minimum bid",
                          r"\bdelist", r"\bhearing\b"]),
]
# Headlines that are about the stock, not by the company (law-firm ads, movers lists)
NOISE = [r"shareholder alert", r"investigat", r"class action", r"\bmovers\b", r"stocks? moving", r"premarket",
         r"pre-market", r"why .* (shares|stock) (are|is)", r"trading higher", r"top gainers", r"mid-day", r"halted"]


def categorize(texts: list[str]) -> str:
    t = " | ".join(texts).lower()
    if not t.strip():
        return "no_news_found"
    for name, pats in CATEGORIES:
        if any(re.search(p, t) for p in pats):
            return name
    return "other_news"


def ticker_news(client, ticker: str, start: str, end: str) -> list[dict]:
    """All Massive headlines for one ticker over the whole period, cached."""
    f = Path(client.cache) / "news" / f"{ticker}.json"
    if f.exists():
        return json.loads(f.read_text())
    rows, url = [], "/v2/reference/news"
    params = {"ticker": ticker, "published_utc.gte": start, "published_utc.lte": end, "limit": 1000,
              "order": "asc", "sort": "published_utc"}
    while url:
        j = client._get(url, params)
        rows += [{"published_utc": r.get("published_utc"), "title": r.get("title", ""),
                  "description": (r.get("description") or "")[:500], "publisher": (r.get("publisher") or {}).get("name"),
                  "tickers": r.get("tickers") or []} for r in j.get("results") or []]
        url, params = j.get("next_url"), None
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rows))
    return rows


def pregap_headlines(rows: list[dict], gap_date: str, prev_date: str, scan: str = "09:25") -> list[dict]:
    lo = pd.Timestamp(f"{prev_date} 16:00", tz=NY)
    hi = pd.Timestamp(f"{gap_date} {scan}", tz=NY)
    out = []
    for r in rows:
        try:
            ts = pd.Timestamp(r["published_utc"]).tz_convert(NY)
        except Exception:
            continue
        if lo <= ts < hi and not any(re.search(p, (r["title"] or "").lower()) for p in NOISE):
            out.append(r)
    return out
