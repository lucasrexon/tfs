"""Finer news labels for every gapper, with timing and source.

    python -m tfs.v2.news2          (uses the cache; fetches SEC press releases it does not have yet)

Sources: Massive headlines (Benzinga, GlobeNewswire) and company press releases attached to
8-K / 6-K filings on SEC EDGAR. Each item is tagged by when it became public:
  before_0925   after 16:00 the previous trading day and before 09:25 on the gap day: the bot can use it
  prior_days    the 4 calendar days before that: older news the gap may be reacting to
  later_today   09:25 to midnight on the gap day: explains the gap but came too late for this bot
                (a paid real-time feed might have had it before 09:25)
"""
from __future__ import annotations

import json
import re

import pandas as pd

from tfs.config import load_config, resolve
from tfs.data.massive import NY
from tfs.sec import Edgar

# first match wins
TYPES: list[tuple[str, list[str]]] = [
    ("being_acquired", [r"to be acquired", r"\bacquired by\b", r"tender offer", r"go(ing)?[- ]private", r"take[- ]private",
                        r"\bbuyout\b", r"per share in cash", r"all[- ]cash (deal|transaction)"]),
    ("merger_combination", [r"business combination", r"reverse merger", r"merger agreement", r"\bmerger\b", r"\bspac\b",
                            r"\bmerge\b"]),
    ("company_acquires", [r"\bto acquire\b", r"\bacquisition of\b", r"\bacquires\b", r"\bacquire\b", r"letter of intent",
                          r"\bloi\b", r"majority stake", r"\bacquisition\b"]),
    ("crypto_treasury", [r"bitcoin", r"\bcrypto", r"digital asset", r"ethereum", r"\bsolana\b", r"\btokens?\b", r"blockchain",
                         r"\bxrp\b", r"dogecoin", r"\bbnb\b", r"stablecoin", r"\bweb3\b"]),
    ("hot_theme", [r"\bai\b", r"artificial intelligence", r"quantum", r"data cent(er|re)", r"\bgpu", r"\bdrones?\b", r"humanoid",
                   r"robot", r"nuclear", r"rare earth", r"\bspace\b", r"satellite", r"defen[cs]e"]),
    ("fda_regulatory", [r"\bfda\b", r"\bapproval\b", r"\bapproved\b", r"clearance", r"510\(k\)", r"ce mark", r"\bind\b", r"\bnda\b",
                        r"\bbla\b", r"pdufa", r"orphan drug", r"fast track", r"breakthrough"]),
    ("clinical_data", [r"\bphase\b", r"topline", r"\btrial\b", r"\bpatients?\b", r"\bclinical\b", r"\bstudy\b", r"\bdata\b"]),
    ("financing", [r"\boffering\b", r"\bpriced\b", r"private placement", r"registered direct", r"\bwarrants?\b", r"\bfinancing\b",
                   r"\bconvertible\b", r"at[- ]the[- ]market", r"equity line", r"\bproceeds\b", r"\braises?\b", r"\bpipe\b",
                   r"securities purchase agreement", r"\bfunding\b", r"\binvestment of \$"]),
    ("contract_order", [r"\bcontract\b", r"purchase order", r"\border\b", r"\bawarded\b", r"\bselected\b", r"\bgovernment\b",
                        r"department of"]),
    ("partnership_license", [r"partnership", r"collaborat", r"\blicens", r"\bmou\b", r"\bagreement\b", r"distribution",
                             r"\bpartner", r"strategic", r"\blaunch"]),
    ("earnings", [r"\bresults\b", r"\brevenue\b", r"\bquarter\b", r"\bearnings\b", r"\bguidance\b", r"\bfiscal\b"]),
    ("compliance_split", [r"reverse (stock )?split", r"nasdaq (notice|letter|deficiency|hearing)", r"regain(s|ed)? compliance",
                          r"minimum bid", r"\bdelist", r"\bhearing\b", r"stockholders? (approve|meeting)", r"annual meeting"]),
]
MOVERS = [r"\bmovers\b", r"stocks? moving", r"premarket", r"pre-market", r"why .* (shares|stock) (are|is)", r"trading higher",
          r"top gainers", r"mid-day", r"halted", r"\bsoar", r"\bjumps?\b", r"\bsurg", r"\bskyrocket", r"\brall(y|ies)\b", r"shoots? up"]
JUNK = [r"shareholder alert", r"investigat", r"class action", r"\blawsuit\b", r"law firm", r"us stocks", r"stock market today"]
WIRES = ("globenewswire", "pr newswire", "business wire", "accesswire", "newsfile")
ITEM_TYPE = {"2.01": "company_acquires", "1.01": "partnership_license", "3.02": "financing", "2.02": "earnings",
             "3.01": "compliance_split", "5.07": "compliance_split"}


def _match(t: str) -> str | None:
    for name, pats in TYPES:
        if any(re.search(p, t) for p in pats):
            return name
    return None


def headline_of(text: str) -> str:
    """The press release's own headline: what follows 'Exhibit 99.1' (and 'for immediate release')."""
    t = re.sub(r"\s+", " ", text or "")
    m = re.search(r"exhibit 99\.?\d?\s*(.*)", t, re.I)
    if m:
        t = m.group(1)
    t = re.sub(r"^(for immediate release|press release|news release)[:\s-]*", "", t, flags=re.I)
    return t[:220]


def label(texts: list[str]) -> str:
    """Headlines first; the body only if no headline says anything."""
    if not any(t.strip() for t in texts):
        return "none"
    for part in ([headline_of(t) for t in texts], [t[:1000] for t in texts]):
        hit = _match(" | ".join(part).lower())
        if hit:
            return hit
    return "other_news"


STOP_WORDS = {"the", "inc", "corp", "corporation", "holdings", "group", "limited", "ltd", "company", "co", "international",
              "global", "technologies", "technology", "therapeutics", "pharmaceuticals", "common", "stock", "class"}


def is_target(name: str, texts: list[str]) -> bool:
    """'X to acquire <this company>': the gapper is the one being bought."""
    words = [w for w in re.findall(r"[a-z0-9]+", (name or "").lower()) if len(w) >= 3 and w not in STOP_WORDS]
    if not words:
        return False
    t = " | ".join(headline_of(x) for x in texts).lower()
    return bool(re.search(r"(acquire[sd]?|acquisition of|to buy|purchase of)\W+(\w+\W+){0,3}" + re.escape(words[0]), t))


def is_cover_page(text: str) -> bool:
    head = (text or "")[:400].lower()
    return head.startswith("false") or "securities and exchange commission" in head or "xbrli" in head


def gather(r: dict, news: list[dict], sub: dict, ed: Edgar | None, cik: str | None) -> dict:
    prev = pd.Timestamp(f"{r['prev_day']} 16:00", tz=NY)
    scan = pd.Timestamp(f"{r['date']} 09:25", tz=NY)
    end = pd.Timestamp(r["date"], tz=NY) + pd.Timedelta(days=1)
    lo = prev - pd.Timedelta(days=4)

    def when(ts):
        if prev <= ts < scan:
            return "before_0925"
        if lo <= ts < prev:
            return "prior_days"
        if scan <= ts < end:
            return "later_today"
        return None

    items = []
    for x in news:
        try:
            ts = pd.Timestamp(x["published_utc"]).tz_convert(NY)
        except Exception:
            continue
        w = when(ts)
        title = (x.get("title") or "").lower()
        if w is None or len(x.get("tickers") or []) > 5 or any(re.search(p, title) for p in JUNK):
            continue
        pub = (x.get("publisher") or "").lower()
        items.append({"when": w, "ts": ts, "src": "wire" if any(k in pub for k in WIRES) else "media",
                      "movers": any(re.search(p, title) for p in MOVERS),
                      "text": f"{x.get('title') or ''}. {x.get('description') or ''}"})
    for f in (sub or {}).get("filings", []):
        if f.get("form") not in ("8-K", "6-K", "8-K/A"):
            continue
        try:
            ts = pd.Timestamp(f["acceptanceDateTime"]).tz_convert(NY)
        except Exception:
            continue
        w = when(ts)
        if w is None:
            continue
        text = ""
        if ed is not None and w in ("before_0925", "later_today"):
            try:
                text = ed.press_release_text(cik, f["accessionNumber"], f.get("primaryDocument"))[:1500]
            except Exception:
                text = ""
            if is_cover_page(text):
                text = ""                                  # the 8-K form itself, not a press release: use its item numbers
        items.append({"when": w, "ts": ts, "src": "sec", "movers": False, "text": text,
                      "items": f.get("items") or ""})

    out = {"date": r["date"], "ticker": r["ticker"]}
    for w in ("before_0925", "prior_days", "later_today"):
        its = [i for i in items if i["when"] == w]
        real = [i["text"] for i in its if i["text"] and not i["movers"]]
        movers = [i["text"] for i in its if i["movers"]]
        typ = label(real)
        if typ in ("none", "other_news"):                      # 8-K item numbers when there is no text to read
            for i in its:
                hit = next((ITEM_TYPE[k] for k in (i.get("items") or "").split(",") if k in ITEM_TYPE), None)
                if hit:
                    typ = hit
                    break
        if typ in ("company_acquires", "merger_combination") and is_target(r.get("name"), real):
            typ = "being_acquired"
        if typ == "none" and movers:
            typ = label(movers)                                # "stock soars on acquisition news" still says why
            typ = "movers_article_only" if typ in ("none", "other_news") else typ
        if typ == "none" and its:
            typ = "filing_no_text"
        out[f"type_{w}"] = typ
        out[f"n_{w}"] = len(its)
        if w == "before_0925":
            out["source_before_0925"] = "+".join(sorted({i["src"] for i in its})) or "none"
            first = min((i["ts"] for i in its), default=None)
            out["timing_before_0925"] = ("none" if first is None else "overnight" if first < pd.Timestamp(f"{r['date']} 04:00", tz=NY)
                                         else "early_premarket" if first < pd.Timestamp(f"{r['date']} 08:00", tz=NY)
                                         else "late_premarket")
            heads = [headline_of(i["text"])[:160] for i in sorted(its, key=lambda i: i["ts"]) if i["text"]]
            out["headline_before_0925"] = heads[0] if heads else None
        if w == "later_today":
            heads = [headline_of(i["text"])[:160] for i in sorted(its, key=lambda i: i["ts"]) if i["text"]]
            out["headline_later_today"] = heads[0] if heads else None
    # the best explanation of the gap, using everything up to midnight (not all of it was known at 09:25)
    out["type_any"] = next((out[f"type_{w}"] for w in ("before_0925", "later_today", "prior_days")
                            if out[f"type_{w}"] not in ("none", "filing_no_text", "movers_article_only", "other_news")),
                           out["type_before_0925"])
    return out


def main():
    cfg = load_config()
    cache = resolve(cfg, cfg["data"]["cache_dir"])
    od = resolve(cfg, "output") / "v2"
    g = pd.read_parquet(od / "gappers40.parquet")
    cik = pd.read_parquet(od / "features.parquet").set_index(["date", "ticker"])["cik"].to_dict()
    ed = Edgar(cache)
    rows = []
    for i, r in enumerate(g.to_dict("records")):
        if i % 200 == 0:
            print(f"  news {i:,}/{len(g):,} (SEC calls {ed.calls})", flush=True)
        nf = cache / "news" / f"{r['ticker']}.json"
        news = json.loads(nf.read_text()) if nf.exists() else []
        c = cik.get((r["date"], r["ticker"]))
        sf = cache / "sec" / "sub" / f"{c}.json"
        sub = json.loads(sf.read_text()) if (c and sf.exists()) else {}
        rows.append(gather(r, news, sub, ed if c else None, c))
    out = pd.DataFrame(rows)
    out.to_parquet(od / "news2.parquet")
    print("NEWS2 DONE")
    for c in ("type_before_0925", "type_any", "timing_before_0925", "source_before_0925"):
        print(out[c].value_counts().to_string(), "\n")


if __name__ == "__main__":
    main()
