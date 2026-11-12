from datetime import datetime

import pandas as pd
import pytest

from tfs.backtest import simulate_gapper
from tfs.config import load_config
from tfs.data.massive import NY
from tfs.live.bot import Bot
from tfs.live.guard import Guard, Journal
from tfs.synthetic import FakeClient


class FakeBroker:
    paper = True

    def __init__(self, client, day, clock):
        self.c, self.day, self.clock_ref = client, day, clock
        i = client.days.index(day)
        self.prev_day = client.days[i - 1]
        self.orders, self.pos = [], {}

    def now(self):
        return self.clock_ref[0]

    def assets(self):
        return [{"symbol": s, "tradable": True, "exchange": "NASDAQ"} for s in self.c.symbols]

    def snapshots(self, syms):
        out = {}
        for s in syms:
            prev = self.c.daily[self.prev_day][s]["c"]
            b = self.c.minute_bars(s, self.day)
            b = b[b.index + pd.Timedelta("1min") <= pd.Timestamp(self.now())]
            last = float(b["c"].iloc[-1]) if len(b) else prev
            out[s] = {"dailyBar": {"t": self.prev_day + "T04:00:00Z", "c": prev}, "latestTrade": {"p": last}}
        return out

    def bars(self, sym, start, end):
        b = self.c.minute_bars(sym, self.day)
        return b[(b.index >= start) & (b.index <= end)]

    def split_today(self, sym, day): return False
    def asset(self, sym): return {"shortable": True, "easy_to_borrow": True}
    def account(self): return {"equity": "10000", "last_equity": "10000", "shorting_enabled": True}

    def positions(self):
        return [{"symbol": s, "qty": str(-q), "market_value": str(-q * 2.0), "unrealized_pl": "0"}
                for s, q in self.pos.items()]

    def short_with_stop(self, sym, qty, stop, cid):
        self.orders.append({"symbol": sym, "qty": qty, "stop": stop, "time": self.now()})
        self.pos[sym] = qty
        return {"id": cid}

    def close_position(self, sym):
        self.pos.pop(sym, None)


def _pick_event(c, kind="fade"):
    for (t, d), ev in c.events.items():
        if ev["kind"] == kind and d != c.days[0]:
            return t, d


@pytest.mark.parametrize("rule", ["vwap_loss|eod|pmh_plus_5", "first_red_5m|eod|pmh_plus_5"])
def test_live_bot_enters_on_same_minute_as_backtest(tmp_path, rule):
    c = FakeClient("2024-01-02", "2024-02-29", events_per_day=0.3)
    sym, day = _pick_event(c)
    cfg = load_config()
    cfg["live"].update(rule=rule, auto_approve=True)
    clock = [datetime.fromisoformat(day + "T09:25:05").replace(tzinfo=NY)]
    broker = FakeBroker(c, day, clock)
    bot = Bot(cfg, broker, Journal(tmp_path / "j.csv"), Guard(cfg, tmp_path / "STOP"), now_fn=lambda: clock[0])
    watch = bot.premarket_scan()
    assert sym in watch
    t = pd.Timestamp(clock[0])
    while t.strftime("%H:%M") < "16:00":
        clock[0] = t.to_pydatetime()
        if t.strftime("%H:%M") >= "09:30":
            bot.step()
            bot.manage()
        t += pd.Timedelta("1min")
    live = [o for o in broker.orders if o["symbol"] == sym]
    cfg2 = load_config()
    e, x, s = rule.split("|")
    cfg2["rules"].update(entries=[e], exits=[x], stops=[s])
    g = {"date": day, "ticker": sym, "prev_close": c.daily[broker.prev_day][sym]["c"], "prev_low": 0,
         "prev_prev_close": None, "pm_gap": watch[sym]["pm_gap"], "pm_high": watch[sym]["pm_high"],
         "factor": 1.0, "days_since_ipo": 1000, "next": []}
    bt = [r for r in simulate_gapper(g, c.minute_bars(sym, day), cfg2) if r["status"] == "filled"]
    if not bt:
        assert not live
        return
    assert live, "backtest traded but live bot did not"
    assert live[0]["time"].strftime("%H:%M") == bt[0]["entry_time"]
    assert not broker.pos, "bot must be flat after 15:55"


def test_guard_blocks(tmp_path):
    cfg = load_config()
    g = Guard(cfg, tmp_path / "STOP")
    ok = dict(sym="X", qty=100, entry=2.0, stop=2.2, account={"equity": "10000", "last_equity": "10000"},
              positions=[], asset={"shortable": True, "easy_to_borrow": True}, trades_today=0, now_hhmm="09:45")
    assert g.check(**ok) is None
    assert g.check(**{**ok, "asset": {"shortable": True, "easy_to_borrow": False}}) == "not_borrowable"
    assert g.check(**{**ok, "account": {"equity": "9700", "last_equity": "10000"}}) == "daily_loss_limit"
    assert g.check(**{**ok, "positions": [{"symbol": "X", "qty": "-50", "market_value": "-100"}]}) == "already_in_position"
    assert g.check(**{**ok, "qty": 5000}) == "exposure_cap"
    assert g.check(**{**ok, "stop": None}) == "no_stop"
    assert g.check(**{**ok, "now_hhmm": "11:31"}) == "past_entry_deadline"
    (tmp_path / "STOP").write_text("")
    assert g.check(**ok) == "kill_switch"


def test_guard_needs_typed_yes(tmp_path):
    cfg = load_config()
    assert Guard(cfg, tmp_path / "S", input_fn=lambda _: "yes").approve("x") is False
    assert Guard(cfg, tmp_path / "S", input_fn=lambda _: "YES").approve("x") is True


def test_bot_refuses_overnight_and_stopless_rules(tmp_path):
    cfg = load_config()
    for rule in ("open|close_d2|pmh_plus_5", "next_open|close_d3|pmh_plus_5", "open|eod|none"):
        cfg["live"]["rule"] = rule
        with pytest.raises(SystemExit):
            Bot(cfg, None, Journal(tmp_path / "j.csv"), Guard(cfg, tmp_path / "S"))
