"""SEC EDGAR features, point in time: only what was filed before the scan on the gap day.

Sources (all free, no key; SEC asks for <= 10 requests/second):
  submissions  data.sec.gov/submissions/CIK##########.json   list of filings, SIC code, address
  companyfacts data.sec.gov/api/xbrl/companyfacts/CIK...json   cash, cash burn, shares outstanding
  full text    efts.sec.gov/LATEST/search-index               "substantial doubt", "at-the-market", ...
  archives     www.sec.gov/Archives/...                       press release text of morning 8-K / 6-K
"""
from __future__ import annotations

import json
import re
import time
from datetime import date as Date, timedelta
from html import unescape
from pathlib import Path

import pandas as pd
import requests

from tfs.data.massive import NY

UA = "TFS-research-bot/0.2"
US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND "
                "OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR VI GU".split())
CHINA_HK = {"F4", "K3", "F5"}          # EDGAR country codes: China, Hong Kong, (F5 = Macau)
OFFERING_FORMS = {"424B1", "424B2", "424B3", "424B4", "424B5", "424B7", "424B8"}
SHELF_FORMS = {"S-3", "S-3/A", "S-3ASR", "F-3", "F-3/A", "F-3ASR"}
S1_FORMS = {"S-1", "S-1/A", "F-1", "F-1/A"}
FTS_QUERIES = {
    "going_concern": ('"substantial doubt"', "10-K,10-Q,20-F,10-KT"),
    "atm": ('"at-the-market"', "8-K,424B5,S-3,F-3,10-Q,10-K,6-K,20-F"),
    "reverse_split": ('"reverse stock split"', "8-K,6-K,DEF 14A,PRE 14A,DEF 14C,PRE 14C"),
}

CASH = ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash", "CashAndCashEquivalents"]
OPCF = ["NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations", "CashFlowsFromUsedInOperatingActivities"]
REVENUE = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenue", "SalesRevenueNet"]


def industry(sic: str | int | None, text: str = "") -> str:
    t = (text or "").lower()
    if re.search(r"bitcoin|blockchain|crypto|digital asset|ethereum|solana|token", t):
        return "crypto"
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "other"
    if 2833 <= s <= 2836 or s == 8731:
        return "biotech_pharma"
    if 3841 <= s <= 3851 or s == 5047 or 8000 <= s <= 8099:
        return "medical_health"
    if 3570 <= s <= 3579 or 3600 <= s <= 3699 or 7370 <= s <= 7379 or 4800 <= s <= 4899:
        return "tech"
    if 1000 <= s <= 1499 or s == 2911 or 4900 <= s <= 4991:
        return "energy_mining"
    if 2000 <= s <= 2399 or 5000 <= s <= 5999 or 7000 <= s <= 7299 or 7800 <= s <= 7999:
        return "consumer"
    if 1500 <= s <= 1799 or 3400 <= s <= 3569 or 3580 <= s <= 3599 or 3700 <= s <= 3799 or 4000 <= s <= 4799:
        return "industrial_ev"
    if 6000 <= s <= 6799:
        return "finance_shell"
    return "other"


class Edgar:
    def __init__(self, cache_dir: str | Path, rps: float = 5.0):
        self.cache = Path(cache_dir) / "sec"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.min_iv, self._last, self.calls = 1.0 / rps, 0.0, 0

    def _get(self, url: str, params: dict | None = None, as_text=False):
        for attempt in range(5):
            w = self.min_iv - (time.monotonic() - self._last)
            if w > 0:
                time.sleep(w)
            self._last = time.monotonic()
            try:
                r = self.s.get(url, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            self.calls += 1
            if r.status_code == 200:
                return r.text if as_text else r.json()
            if r.status_code == 404:
                return "" if as_text else {}
            if r.status_code in (403, 429, 500, 502, 503):
                time.sleep(min(30, 2 ** attempt * 2))
                continue
            raise RuntimeError(f"SEC {r.status_code} {url}")
        raise RuntimeError(f"SEC gave up: {url}")

    def _cached(self, key: str, fetch):
        f = self.cache / key
        if f.exists():
            return json.loads(f.read_text())
        v = fetch()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(v))
        return v

    def submissions(self, cik: str) -> dict:
        def fetch():
            j = self._get(f"https://data.sec.gov/submissions/CIK{cik}.json")
            if not j:
                return {}
            rec = j.get("filings", {}).get("recent", {})
            rows = [dict(zip(rec.keys(), v)) for v in zip(*rec.values())] if rec else []
            for extra in j.get("filings", {}).get("files", []):       # older filings pages
                if extra.get("filingTo", "") >= "2022-01-01":
                    e = self._get(f"https://data.sec.gov/submissions/{extra['name']}")
                    rows += [dict(zip(e.keys(), v)) for v in zip(*e.values())] if e else []
            keep = ("form", "filingDate", "acceptanceDateTime", "items", "accessionNumber", "primaryDocument")
            return {"sic": j.get("sic"), "sicDescription": j.get("sicDescription"), "name": j.get("name"),
                    "stateOfIncorporation": j.get("stateOfIncorporation"),
                    "country": (j.get("addresses", {}).get("business", {}) or {}).get("stateOrCountry"),
                    "filings": [{k: r.get(k) for k in keep} for r in rows if r.get("filingDate", "") >= "2021-01-01"]}
        return self._cached(f"sub/{cik}.json", fetch)

    def facts(self, cik: str) -> dict:
        """Only the concepts we use, with value, period and filed date."""
        def fetch():
            j = self._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
            out = {}
            for tax, concepts in (j or {}).get("facts", {}).items():
                for name, body in concepts.items():
                    if name in CASH + OPCF + REVENUE + ["EntityCommonStockSharesOutstanding", "NetIncomeLoss",
                                                         "ProfitLoss"]:
                        for unit, vals in body.get("units", {}).items():
                            out.setdefault(name, []).extend(
                                {"val": v.get("val"), "start": v.get("start"), "end": v.get("end"),
                                 "filed": v.get("filed"), "form": v.get("form"), "unit": unit}
                                for v in vals if v.get("filed", "") >= "2020-01-01")
            return out
        return self._cached(f"facts/{cik}.json", fetch)

    def fts(self, cik: str, key: str) -> list[dict]:
        q, forms = FTS_QUERIES[key]

        def fetch():
            hits, frm = [], 0
            while True:
                j = self._get("https://efts.sec.gov/LATEST/search-index",
                              {"q": q, "ciks": cik, "forms": forms, "dateRange": "custom",
                               "startdt": "2021-01-01", "enddt": "2026-12-31", "from": frm})
                hs = (j or {}).get("hits", {}).get("hits", [])
                hits += [{"file_date": h["_source"].get("file_date"), "form": h["_source"].get("form")} for h in hs]
                total = (j or {}).get("hits", {}).get("total", {}).get("value", 0)
                frm += len(hs)
                if not hs or frm >= total or frm >= 500:
                    return hits
        return self._cached(f"fts/{key}/{cik}.json", fetch)

    def press_release_text(self, cik: str, accession: str, primary: str | None = None) -> str:
        """Text of EX-99 exhibits (press releases) of one filing, first 3,000 characters."""
        acc = accession.replace("-", "")

        def fetch():
            idx = self._get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/index.json")
            docs = [d["name"] for d in (idx or {}).get("directory", {}).get("item", [])
                    if re.search(r"ex-?99|ex99|press", d["name"].lower()) and d["name"].lower().endswith((".htm", ".html", ".txt"))]
            text = ""
            if not docs and primary:
                docs = [primary]
            for name in docs[:2]:
                h = self._get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{name}", as_text=True)
                text += " " + re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", h or "")))
            return {"text": text.strip()[:3000]}
        return self._cached(f"pr/{cik}_{acc}.json", fetch)["text"]


# ---------------------------------------------------------------- features
def _latest(facts: dict, names: list[str], cutoff: str, duration: bool | None = None):
    best = None
    for n in names:
        for v in facts.get(n, []):
            if not v.get("filed") or v["filed"] >= cutoff or v.get("val") is None:
                continue
            is_dur = bool(v.get("start"))
            if duration is not None and is_dur != duration:
                continue
            if best is None or (v["end"], v["filed"]) > (best["end"], best["filed"]):
                best = v
        if best:
            return best
    return None


def _months(v) -> float:
    if not v or not v.get("start"):
        return 0
    return max(1.0, (Date.fromisoformat(v["end"]) - Date.fromisoformat(v["start"])).days / 30.44)


def company_features(ed: Edgar, cik: str, gap_date: str, prev_day: str | None, desc: str = "",
                     splits: list[dict] | None = None) -> dict:
    """Everything known before 09:25 ET on gap_date."""
    sub = ed.submissions(cik)
    facts = ed.facts(cik)
    D = Date.fromisoformat(gap_date)
    cutoff = gap_date                      # filed strictly before the gap day ...
    morning_lo = pd.Timestamp(f"{prev_day or gap_date} 16:00", tz=NY)
    morning_hi = pd.Timestamp(f"{gap_date} 09:25", tz=NY)

    def within(days, forms=None, items=None):
        lo = (D - timedelta(days=days)).isoformat()
        n = 0
        for f in sub.get("filings", []):
            if lo <= f["filingDate"] < cutoff and (forms is None or f["form"] in forms) and \
                    (items is None or any(i in (f.get("items") or "").split(",") for i in items)):
                n += 1
        return n

    morning = []
    for f in sub.get("filings", []):
        try:
            acc_t = pd.Timestamp(f["acceptanceDateTime"]).tz_convert(NY)   # ... or accepted that morning
        except Exception:
            continue
        if morning_lo <= acc_t < morning_hi and f["form"] in ("8-K", "6-K", "8-K/A"):
            morning.append(f)

    cash = _latest(facts, CASH, cutoff, duration=False)
    opcf = _latest(facts, OPCF, cutoff, duration=True)
    burn = (-opcf["val"] / _months(opcf)) if opcf and opcf["val"] < 0 else 0.0
    runway = (cash["val"] / burn) if (cash and burn > 0) else (99.0 if cash else None)
    sh = [v for v in facts.get("EntityCommonStockSharesOutstanding", []) if v.get("filed") and v["filed"] < cutoff]
    sh.sort(key=lambda v: v["end"])
    sh_now = sh[-1]["val"] if sh else None
    year_ago = (D - timedelta(days=365)).isoformat()
    old = next((v for v in reversed(sh) if v["end"] <= year_ago), None)
    sh_old = old["val"] if old else None
    if sh_old and splits:                  # put the old count on today's split basis
        for sp in splits:
            if old["end"] < sp.get("execution_date", "") <= gap_date:
                sh_old *= float(sp.get("split_to", 1) or 1) / float(sp.get("split_from", 1) or 1)
    dil = (sh_now / sh_old - 1) if (sh_now and sh_old) else None
    if dil is not None and dil < -0.5:     # a share count that fell by half is an unadjusted reverse split
        dil = None
    rev = _latest(facts, REVENUE, cutoff, duration=True)
    rev_annual = (rev["val"] / _months(rev) * 12) if rev else None

    fts = {k: ed.fts(cik, k) for k in FTS_QUERIES}

    def fts_within(k, days):
        lo = (D - timedelta(days=days)).isoformat()
        return int(any(lo <= (h["file_date"] or "") < cutoff for h in fts[k]))

    def after(days, forms):
        hi = (D + timedelta(days=days)).isoformat()
        return int(any(gap_date <= f["filingDate"] <= hi and f["form"] in forms for f in sub.get("filings", [])))

    country = sub.get("country") or ""
    forms_all = {f["form"] for f in sub.get("filings", [])}
    return {
        "cik": cik, "sic": sub.get("sic"), "industry": industry(sub.get("sic"), f"{sub.get('name', '')} {desc}"),
        "foreign": int(bool(country) and country not in US_STATES),
        "china_hk": int(country in CHINA_HK),
        "foreign_filer": int(bool(forms_all & {"20-F", "6-K", "40-F"})),
        "cash": cash["val"] if cash else None,
        "burn_per_month": burn,
        "runway_months": min(runway, 99.0) if runway is not None else None,
        "runway_lt6": int(runway is not None and runway < 6),
        "shares_out_sec": sh_now,
        "dilution_12m": dil,
        "revenue_annual": rev_annual,
        "has_revenue": int(bool(rev_annual and rev_annual > 1e6)),
        "shelf_3y": int(within(3 * 365, SHELF_FORMS) > 0),
        "s1_12m": int(within(365, S1_FORMS) > 0),
        "offering_90d": int(within(90, OFFERING_FORMS) > 0),
        "offerings_12m": within(365, OFFERING_FORMS),
        "deficiency_12m": int(within(365, {"8-K"}, ["3.01"]) > 0),
        "charter_amend_12m": int(within(365, {"8-K"}, ["5.03"]) > 0),
        "filings_30d": within(30),
        "going_concern": fts_within("going_concern", 456),
        "atm_24m": fts_within("atm", 730),
        "reverse_split_12m": fts_within("reverse_split", 365),
        "morning_filing": int(bool(morning)),
        "morning_items": ",".join(sorted({i for f in morning for i in (f.get("items") or "").split(",") if i})),
        "_morning": [(f["accessionNumber"], f["form"], f.get("primaryDocument")) for f in morning],
        # outcome only (known after the gap): did the company sell stock within 5 days?
        "offering_after_5d": after(5, OFFERING_FORMS),
    }


def dilution_pressure(f: dict) -> int:
    """Starting weights; the research replaces them with data-driven ones."""
    return (2 * f.get("shelf_3y", 0) + 2 * f.get("atm_24m", 0) + 2 * f.get("runway_lt6", 0)
            + f.get("going_concern", 0) + f.get("offering_90d", 0) + f.get("reverse_split_12m", 0)
            + f.get("deficiency_12m", 0))
