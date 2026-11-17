import pandas as pd

from tfs.data.alpaca_data import AlpacaPrices, HybridClient, day_window_utc, frame_from_bars
from tfs.v2.finra import parse


class FakeResp:
    def __init__(self, j, code=200):
        self.status_code, self._j, self.headers, self.text = code, j, {}, ""

    def json(self):
        return self._j


class FakeSession:
    """Serves canned pages; records every request's params."""
    def __init__(self, pages):
        self.pages, self.seen, self.headers = list(pages), [], {}

    def get(self, url, params=None, timeout=None):
        self.seen.append(dict(params))
        return FakeResp(self.pages.pop(0))


def bar(t, o, v=100.0):
    return {"t": t, "o": o, "h": o + 1, "l": o - 1, "c": o + 0.5, "v": v, "vw": o, "n": 3}


def test_day_window_covers_premarket_and_after_hours_in_both_seasons():
    assert day_window_utc("2021-01-15") == ("2021-01-15T08:00:00Z", "2021-01-16T01:00:00Z")   # EST: 03:00-20:00
    assert day_window_utc("2021-07-15") == ("2021-07-15T08:00:00Z", "2021-07-16T01:00:00Z")   # EDT: 04:00-21:00


def test_frame_matches_massive_format():
    df = frame_from_bars([bar("2021-08-27T12:01:00Z", 5.0), bar("2021-08-27T08:00:00Z", 4.0)])
    assert list(df.columns) == ["o", "h", "l", "c", "v", "vw"]
    assert str(df.index.tz) == "America/New_York" and df.index.name == "ts"
    assert df.index[0].strftime("%H:%M") == "04:00" and df["o"].iloc[0] == 4.0      # sorted by time
    empty = frame_from_bars([])
    assert empty.empty and list(empty.columns) == ["o", "h", "l", "c", "v", "vw"]


def test_minute_bars_follow_pages_keep_only_that_day_and_cache(tmp_path):
    a = AlpacaPrices(tmp_path, max_rpm=0)
    a.s = FakeSession([
        {"bars": {"SPRT": [bar("2021-08-27T08:00:00Z", 4.0)]}, "next_page_token": "p2"},
        {"bars": {"SPRT": [bar("2021-08-27T23:59:00Z", 6.0), bar("2021-08-28T00:30:00Z", 7.0)]}, "next_page_token": None},
    ])
    df = a.minute_bars("SPRT", "2021-08-27")
    assert len(df) == 3 and df.index.strftime("%Y-%m-%d").unique().tolist() == ["2021-08-27"]
    assert a.s.seen[0]["asof"] == "-" and a.s.seen[0]["feed"] == "sip" and a.s.seen[0]["adjustment"] == "split"
    assert a.s.seen[1]["page_token"] == "p2"
    a.s = FakeSession([])                                    # second call comes from the cache
    assert len(a.minute_bars("SPRT", "2021-08-27")) == 3


def test_build_daily_writes_grouped_days_and_resumes(tmp_path):
    a = AlpacaPrices(tmp_path, max_rpm=0)
    days = ["2021-08-26", "2021-08-27", "2021-08-30"]
    a.s = FakeSession([
        {"bars": {"AAA": [bar("2021-08-26T04:00:00Z", 1.0), bar("2021-08-27T04:00:00Z", 2.0)]}, "next_page_token": None},
        {"bars": {"CCC": [bar("2021-08-27T04:00:00Z", 3.0)]}, "next_page_token": None},
    ])
    n = a.build_daily(["CCC", "AAA", "BBB"], days, batch=2)
    assert n == 3
    g = a.grouped_daily("2021-08-27")
    assert sorted(g["T"]) == ["AAA", "CCC"] and list(g.columns) == ["T", "o", "h", "l", "c", "v", "vw"]
    assert a.grouped_daily("2021-08-30").empty                # a trading day with no bars still gets a file
    a.s = FakeSession([])                                      # both batches cached: no new requests
    assert a.build_daily(["CCC", "AAA", "BBB"], days, batch=2) == 3


def test_hybrid_routes_old_prices_to_alpaca_and_the_rest_to_massive():
    class M:
        def grouped_daily(self, d): return "massive"
        def minute_bars(self, t, d): return "massive"
        def news(self): return "news from massive"

    class A:
        def grouped_daily(self, d): return "alpaca"
        def minute_bars(self, t, d): return "alpaca"

    h = HybridClient(M(), A(), alpaca_before="2024-11-11")
    assert h.minute_bars("X", "2021-06-01") == "alpaca" and h.grouped_daily("2024-11-08") == "alpaca"
    assert h.minute_bars("X", "2025-06-02") == "massive" and h.news() == "news from massive"
    assert HybridClient(M(), A(), "2024-11-11", force_alpaca=True).minute_bars("X", "2025-06-02") == "alpaca"


def test_finra_parse_drops_the_trailer_line():
    txt = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
           "20210827|AMC|16677649|244268|44523576|B,Q,N\n20210827|SPRT|37979725|551311|76802421|B,Q,N\n9766\n")
    df = parse(txt)
    assert df["symbol"].tolist() == ["AMC", "SPRT"] and df["total_volume"].iloc[1] == 76802421
