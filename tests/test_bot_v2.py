import copy
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tfs.config import load_config
from tfs.data.massive import NY
from tfs.live.bot_v2 import BotV2, GoingConcern, _make
from tfs.live.guard import Journal

BASE = load_config()


def pm_bars(day, prev, gap, vol=1_000_000):
    """04:00-09:24 bars rising from prev to prev*(1+gap)."""
    idx = pd.Timestamp(day, tz=NY) + pd.to_timedelta(np.arange(240, 565), unit="min")
    c = np.linspace(prev * 1.1, prev * (1 + gap), len(idx))
    df = pd.DataFrame({"o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": vol / len(idx)}, index=pd.DatetimeIndex(idx, name="ts"))
    df["vw"] = df.c
    return df


class Fake:
    paper = True

    def __init__(self, clock, stocks):
        self.clock, self.stocks = clock, stocks           # sym -> dict(prev, gap, etb)
        self.orders, self.cancelled, self.closed, self.held, self.order_state = [], [], [], {}, {}
        self.filled = {}
        self.last = {}

    def assets(self): return [{"symbol": s, "tradable": True, "exchange": "NASDAQ"} for s in self.stocks]

    def snapshots(self, syms):
        return {s: {"dailyBar": {"t": "2025-03-02T05:00:00Z", "c": self.stocks[s]["prev"]},
                    "latestTrade": {"p": self.stocks[s]["prev"] * (1 + self.stocks[s]["gap"])}} for s in syms}

    def bars(self, sym, start, end):
        b = pm_bars(self.clock[0].strftime("%Y-%m-%d"), self.stocks[sym]["prev"], self.stocks[sym]["gap"])
        return b[(b.index >= start) & (b.index <= end)]

    def account(self): return {"equity": "10000", "shorting_enabled": True}
    def asset(self, sym): return {"shortable": True, "easy_to_borrow": self.stocks[sym].get("etb", True)}

    def limit_order(self, sym, qty, side, limit, cid, extended):
        oid = f"o{len(self.orders)}"
        self.orders.append({"id": oid, "symbol": sym, "qty": qty, "side": side, "limit": limit, "extended": extended, "type": "limit"})
        self.order_state[oid] = {"status": "new", "filled_qty": "0"}
        return {"id": oid}

    def exit_oco(self, sym, qty, target, stop, cid):
        oid = f"x{len(self.orders)}"
        self.orders.append({"id": oid, "symbol": sym, "qty": qty, "type": "oco", "target": target, "stop": stop})
        return {"id": oid}

    def get_order(self, oid): return self.order_state[oid]
    def cancel_order(self, oid): self.cancelled.append(oid)
    def positions(self): return [{"symbol": s, "qty": str(-q)} for s, q in self.held.items()]
    def filled_orders(self, sym, after): return self.filled.get(sym, [])
    def next_trading_day(self, day): return (pd.Timestamp(day) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    def last_price(self, sym): return self.last.get(sym)
    def close_position(self, sym): self.closed.append(sym); self.held.pop(sym, None)


def make(tmp_path, stocks, gc, when="2025-03-03 09:20", state=None, auto=True):
    cfg = copy.deepcopy(BASE)
    cfg["live"].update(auto_approve=auto, kill_switch_file=str(tmp_path / "STOP"))
    clock = [pd.Timestamp(when, tz=NY).to_pydatetime()]
    b = Fake(clock, stocks)
    sf = tmp_path / "state.json"
    if state:
        sf.write_text(json.dumps(state))
    class GC:
        def __call__(self, s, d): return gc.get(s)
        def industry(self, s): return INDUSTRY.get(s, "tech")
    bot = BotV2(cfg, b, Journal(tmp_path / "j.csv"), GC(), sf, now_fn=lambda: clock[0], rng=random.Random(1))
    return bot, b, clock


def events(tmp_path, kind):
    j = pd.read_csv(tmp_path / "j.csv")
    return [(r.symbol, json.loads(r.details)) for r in j[j.event == kind].itertuples()]


STOCKS = {"AAA": {"prev": 2.0, "gap": 0.8}, "BBB": {"prev": 3.0, "gap": 0.6}, "CCC": {"prev": 1.5, "gap": 0.7},
          "NOGC": {"prev": 2.0, "gap": 0.9}, "SMALL": {"prev": 2.0, "gap": 0.45}, "HTB": {"prev": 2.0, "gap": 0.9, "etb": False},
          "FLAT": {"prev": 2.0, "gap": 0.1}}
INDUSTRY = {"SHOP": "consumer"}
GC = {"AAA": True, "BBB": True, "CCC": True, "NOGC": False, "SMALL": True, "HTB": True, "FLAT": True}
OPEN_OLD = {"OLD": {"status": "open", "qty": 10, "entry_px": 5.0, "target": 4.0, "stop": 15.0, "entry_date": "2025-02-28",
                    "exit_date": "2025-03-03", "exit_order": "xold"}}


def test_entries_follow_the_rules_and_the_open_limit(tmp_path):
    bot, b, clock = make(tmp_path, STOCKS, GC, state=OPEN_OLD)
    bot.prescan()
    assert set(bot.cands) == {"AAA", "BBB", "CCC", "NOGC", "SMALL", "HTB"}       # FLAT never got to +40%
    clock[0] = pd.Timestamp("2025-03-03 09:25:20", tz=NY).to_pydatetime()
    b.held["OLD"] = 10
    bot.enter()
    shorts = [o for o in b.orders if o["side"] == "sell"]
    assert len(shorts) == 2                                                     # 3 allowed, 1 already open
    assert all(o["extended"] and o["symbol"] in {"AAA", "BBB", "CCC"} for o in shorts)
    for o in shorts:
        assert o["qty"] == int(0.024 * 10000 / o["limit"])
    why = {s: d["reason"] for s, d in events(tmp_path, "v2_skip") + events(tmp_path, "v2_reject")}
    assert why["NOGC"] == "no_going_concern" and why["SMALL"] == "gap_below_min"
    assert "max_open_reached" in why.values()
    assert why.get("HTB") in ("not_easy_to_borrow", "max_open_reached")


def test_fill_places_target_and_stop_then_broker_exit_is_recorded(tmp_path):
    bot, b, clock = make(tmp_path, {"AAA": STOCKS["AAA"]}, GC)
    bot.prescan()
    clock[0] = pd.Timestamp("2025-03-03 09:25:10", tz=NY).to_pydatetime()
    bot.enter()
    oid = bot.pos["AAA"]["entry_order"]
    b.order_state[oid] = {"status": "filled", "filled_qty": str(bot.pos["AAA"]["qty"]), "filled_avg_price": "3.50"}
    b.held["AAA"] = bot.pos["AAA"]["qty"]
    bot.manage()
    oco = [o for o in b.orders if o["type"] == "oco"][0]
    assert oco["target"] == pytest.approx(2.80) and oco["stop"] == pytest.approx(10.50)
    assert bot.pos["AAA"]["exit_date"] == "2025-03-04"
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["AAA"]["status"] == "open"                                      # survives a restart
    clock[0] = pd.Timestamp("2025-03-03 11:00", tz=NY).to_pydatetime()
    del b.held["AAA"]
    b.filled["AAA"] = [{"side": "buy", "status": "filled", "filled_avg_price": "2.80"}]
    bot.manage()
    ex = events(tmp_path, "v2_exit")
    assert ex[0][1]["reason"] == "target" and ex[0][1]["ret"] == pytest.approx(0.2)
    assert "AAA" not in bot.pos


def test_unfilled_entry_is_cancelled(tmp_path):
    bot, b, clock = make(tmp_path, {"AAA": STOCKS["AAA"]}, GC)
    bot.prescan()
    clock[0] = pd.Timestamp("2025-03-03 09:25:10", tz=NY).to_pydatetime()
    bot.enter()
    clock[0] = pd.Timestamp("2025-03-03 09:29:05", tz=NY).to_pydatetime()
    bot.manage()
    assert b.cancelled and "AAA" not in bot.pos and events(tmp_path, "v2_no_fill")


def test_time_limit_covers_at_1555_on_the_next_day(tmp_path):
    bot, b, clock = make(tmp_path, {}, {}, when="2025-03-03 15:50", state=OPEN_OLD)
    b.held["OLD"] = 10
    bot.manage()
    assert not b.closed                                                         # not yet
    clock[0] = pd.Timestamp("2025-03-03 15:55:30", tz=NY).to_pydatetime()
    b.last["OLD"] = 4.5
    bot.manage()
    assert b.closed == ["OLD"] and "xold" in b.cancelled
    assert events(tmp_path, "v2_exit")[0][1]["reason"] == "time_limit"


def test_stop_is_watched_outside_regular_hours(tmp_path):
    bot, b, clock = make(tmp_path, {}, {}, when="2025-03-03 08:10", state=OPEN_OLD)
    b.held["OLD"] = 10
    b.last["OLD"] = 16.0                                                        # above the 15.00 stop, pre-market
    bot.manage()
    cover = [o for o in b.orders if o["side"] == "buy"][0]
    assert cover["extended"] and cover["limit"] == pytest.approx(16.48) and "xold" in b.cancelled
    bot.manage()
    assert len([o for o in b.orders if o["side"] == "buy"]) == 1                # only once


def test_kill_switch_and_approval(tmp_path):
    bot, b, clock = make(tmp_path, {"AAA": STOCKS["AAA"]}, GC)
    (tmp_path / "STOP").write_text("")
    bot.prescan()
    clock[0] = pd.Timestamp("2025-03-03 09:25:10", tz=NY).to_pydatetime()
    bot.enter()
    assert not b.orders and events(tmp_path, "v2_reject")[0][1]["reason"] == "kill_switch"
    (tmp_path / "x").mkdir()
    bot2, b2, clock2 = make(tmp_path / "x", {"AAA": STOCKS["AAA"]}, GC, auto=False)
    bot2.input = lambda prompt: "no"
    bot2.prescan()
    clock2[0] = pd.Timestamp("2025-03-03 09:25:10", tz=NY).to_pydatetime()
    bot2.enter()
    assert not b2.orders


def test_live_mode_needs_the_owner_to_allow_overnight():
    cfg = copy.deepcopy(BASE)
    cfg["live"].update(mode="live", allow_overnight=False)
    with pytest.raises(SystemExit, match="overnight"):
        _make(cfg, input_fn=lambda p: "I AM THE ACCOUNT OWNER")


def test_going_concern_lookup():
    class Ed:
        def _get(self, url, params=None):
            if url.endswith("company_tickers.json"):
                return {"0": {"ticker": "AAA", "cik_str": 123}, "1": {"ticker": "BBB", "cik_str": 456}}
            assert params["ciks"] in ("0000000123", "0000000456") and params["enddt"] == "2025-03-02"
            return {"hits": {"total": {"value": 2 if params["ciks"] == "0000000123" else 0}}}
    gc = GoingConcern(Ed())
    assert gc("AAA", "2025-03-03") is True and gc("BBB", "2025-03-03") is False and gc("ZZZ", "2025-03-03") is None


def test_skipped_industry_is_not_traded(tmp_path):
    stocks = {"AAA": STOCKS["AAA"], "SHOP": {"prev": 2.0, "gap": 0.9}}
    bot, b, clock = make(tmp_path, stocks, {"AAA": True, "SHOP": True})
    bot.prescan()
    clock[0] = pd.Timestamp("2025-03-03 09:25:10", tz=NY).to_pydatetime()
    bot.enter()
    assert [o["symbol"] for o in b.orders] == ["AAA"]
    assert dict(events(tmp_path, "v2_skip"))["SHOP"]["reason"] == "skipped_industry"


def test_industry_lookup_uses_the_research_groups():
    class Ed:
        def _get(self, url, params=None):
            if url.endswith("company_tickers.json"):
                return {"0": {"ticker": "EDU", "cik_str": 7}}
            assert url.endswith("CIK0000000007.json")
            return {"sic": "8200", "name": "Some Education Group"}
    assert GoingConcern(Ed()).industry("EDU") == "other"
