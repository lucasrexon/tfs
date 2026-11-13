import numpy as np
import pandas as pd
import pytest

from tfs.config import load_config
from tfs.data.massive import NY
from tfs.news import categorize, pregap_headlines
from tfs.sec import dilution_pressure, industry
from tfs.v2.margin import f_max, max_survivable_r, survives
from tfs.v2.paths import simulate
from tfs.v2.stats2 import Logit, auc, bh, correlate, cvar, wilson

CFG = load_config()


def day_bars(closes, start_min=570, day="2025-03-03"):
    idx = pd.Timestamp(day, tz=NY) + pd.to_timedelta(np.arange(start_min, start_min + len(closes)), unit="min")
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"o": o, "h": np.maximum(o, c), "l": np.minimum(o, c), "c": c, "v": 1e5},
                      index=pd.DatetimeIndex(idx, name="ts"))
    df["vw"] = (df.h + df.l + df.c) / 3
    return df


def gapper(nxt=None):
    return {"date": "2025-03-03", "ticker": "T", "prev_close": 1.0, "pm_high": 2.2, "pm_last": 2.0, "factor": 1.0,
            "next10": nxt or []}


def pick(rows, entry, target, horizon):
    return next(r for r in rows if r["entry"] == entry and r["target"] == target and r["horizon"] == horizon)


def test_margin_closed_form_and_floor():
    assert f_max(3.0, 2.0, "full") == pytest.approx(1 / 3, abs=1e-6)       # 1/(2R-1)
    assert f_max(3.0, 5.0, "full") == pytest.approx(1 / 9, abs=1e-6)
    assert f_max(1.0, 1.0, "full") == pytest.approx(1 / 2.5, abs=1e-6)     # $2.50/share floor on a $1 stock
    assert survives(3.0, 0.10, max_survivable_r(3.0, 0.10, "full") - 0.01, "full")
    assert not survives(3.0, 0.10, max_survivable_r(3.0, 0.10, "full") + 0.01, "full")


def test_target_hit_same_day_and_runup_counted():
    closes = np.r_[np.linspace(2.0, 2.6, 30), np.linspace(2.6, 1.5, 360)]    # runs up 30%, then fades
    rows = simulate(gapper(), day_bars(closes), CFG)
    r = pick(rows, "open", 0.20, 0)
    assert r["hit"] and r["R"] == pytest.approx(1.3, abs=0.01)
    assert 0.15 < r["ret"] < 0.20                                            # 20% less costs (a $0.03 locate fee is 1.5% of a $2 stock)
    assert r["nights"] == 0


def test_no_stop_holds_through_squeeze_until_time_limit():
    closes = np.linspace(2.0, 6.0, 390)                                       # triples on day 1
    nxt = [{"date": "2025-03-04", "o": 6.5, "h": 9.0, "l": 6.0, "c": 8.0, "v": 1e6},
           {"date": "2025-03-05", "o": 8.0, "h": 8.5, "l": 7.0, "c": 7.5, "v": 1e6},
           {"date": "2025-03-06", "o": 7.5, "h": 7.6, "l": 5.0, "c": 5.5, "v": 1e6}]
    rows = simulate(gapper(nxt), day_bars(closes), CFG)
    r = pick(rows, "open", 0.20, 3)
    assert not r["hit"] and r["exit_date"] == "2025-03-06" and r["nights"] == 3
    assert r["R"] == pytest.approx(4.5, abs=0.01)                             # 9.0 / 2.0
    assert r["ret"] < -1.7                                                    # lost more than the position


def test_overnight_gap_down_fills_at_open():
    closes = np.full(390, 2.0)
    nxt = [{"date": "2025-03-04", "o": 1.2, "h": 1.3, "l": 1.1, "c": 1.2, "v": 1e6}]
    r = pick(simulate(gapper(nxt), day_bars(closes), CFG), "open", 0.20, 1)
    assert r["hit"] and r["ret"] > 0.35                                       # covered at 1.20, not 1.60


def test_pmh_limit_entry_needs_the_price_to_get_there():
    closes = np.r_[np.linspace(2.0, 2.3, 20), np.linspace(2.3, 1.6, 370)]     # tops at 2.30; PMH 2.20
    rows = simulate(gapper(), day_bars(closes), CFG)
    entries = {r["entry"] for r in rows}
    assert "pmh_+0%" in entries and "pmh_+10%" not in entries                 # 2.42 never traded
    assert pick(rows, "pmh_+0%", 0.20, 0)["entry_px"] == pytest.approx(2.2, abs=0.01)


def test_news_categories_and_window():
    assert categorize(["Acme Announces $5 Million Registered Direct Offering"]) == "offering_financing"
    assert categorize(["Acme Receives FDA Clearance for Device"]) == "fda_clinical"
    assert categorize(["Acme to Be Acquired by BigCo for $3.00 per Share"]) == "merger_buyout"
    assert categorize(["Acme Adopts Bitcoin Treasury Strategy"]) == "crypto_ai_pivot"
    assert categorize([]) == "no_news_found"
    rows = [{"published_utc": "2025-03-03T12:00:00Z", "title": "Acme Signs Contract"},          # 07:00 ET: in
            {"published_utc": "2025-03-03T15:00:00Z", "title": "Acme Stock Soars"},             # 10:00 ET: too late
            {"published_utc": "2025-03-03T11:00:00Z", "title": "SHAREHOLDER ALERT: Investigation"},
            {"published_utc": "2025-02-28T15:00:00Z", "title": "Old news"}]
    hl = pregap_headlines(rows, "2025-03-03", "2025-02-28")
    assert [h["title"] for h in hl] == ["Acme Signs Contract"]


def test_industry_and_dilution_score():
    assert industry("2834") == "biotech_pharma" and industry("7372") == "tech" and industry("6770") == "finance_shell"
    assert industry("7372", "bitcoin miner") == "crypto"
    assert dilution_pressure({"shelf_3y": 1, "atm_24m": 1, "runway_lt6": 1, "going_concern": 1}) == 7


def test_stats_tools():
    rng = np.random.default_rng(1)
    n = 600
    x = rng.integers(0, 2, n)
    y = (rng.random(n) < np.where(x == 1, 0.5, 0.2)).astype(int)
    noise = rng.normal(size=n)
    df = pd.DataFrame({"flag": x, "noise": noise, "y": y})
    c = correlate(df, ["flag", "noise"], "y", {"flag"}).set_index("feature")
    assert c.loc["flag", "r"] > 0.2 and c.loc["flag", "significant"] and not c.loc["noise", "significant"]
    assert c.loc["flag", "rate_with"] > c.loc["flag", "rate_without"]
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert cvar([-1.0, 0.1, 0.1, 0.1] * 5, 0.25) == pytest.approx(-1.0)
    assert list(bh(pd.Series([0.001, 0.5, 0.9]))) == [True, False, False]
    m = Logit(1.0).fit(df[["flag", "noise"]], y)
    assert auc(y, m.predict(df[["flag", "noise"]])) > 0.6 and abs(m.weights()["flag"]) > abs(m.weights()["noise"])
