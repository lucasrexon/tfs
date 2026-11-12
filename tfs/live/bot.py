"""The trading bot. Runs on the ACCOUNT OWNER's computer, during market hours.

  09:25  scan the whole market for +50% pre-market gappers, print a card for each
  09:30  watch each gapper; when the chosen rule fires on a just-finished bar, size
         the short with the same code the backtest uses, pass the guard, ask the
         owner (unless auto_approve), send a market short with a broker-side stop
  11:00  time stop (if the rule uses one)
  15:55  close every position the bot opened today

Everything is written to the journal. Paper trading unless config live.mode is
'live' AND the owner confirms at startup.
"""
from __future__ import annotations

import time
from datetime import datetime

import pandas as pd

from tfs.config import parse_rule_id, resolve
from tfs.data.massive import NY
from tfs.live.guard import Guard, Journal
from tfs.risk import margin_requirement, size_short
from tfs.rules import INTRADAY_ENTRIES, count_halts, find_entry, session, stop_price
from tfs.scan import premarket_stats

LISTED = {"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS", "NYSEARCA"}


class Bot:
    def __init__(self, cfg: dict, broker, journal: Journal, guard: Guard, now_fn=lambda: datetime.now(NY)):
        self.cfg, self.b, self.j, self.g, self.now = cfg, broker, journal, guard, now_fn
        self.entry_rule, self.exit_rule, self.stop_rule = parse_rule_id(cfg["live"]["rule"])
        if self.entry_rule not in INTRADAY_ENTRIES or self.exit_rule not in ("eod", "time_stop_1100"):
            raise SystemExit(f"Live bot trades same-day rules only (entries {INTRADAY_ENTRIES}, exits eod or "
                             f"time_stop_1100). '{cfg['live']['rule']}' holds overnight, which TFS does not allow live.")
        if self.stop_rule == "none":
            raise SystemExit("Live rules must have a stop. Use ...|pmh_plus_5.")
        self.watch: dict[str, dict] = {}
        self.done: set[str] = set()       # symbols already traded or rejected today
        self.opened: set[str] = set()
        self.trades_today = 0
        self.time_stop_done = False
        self.flattened = False

    # ------------------------------------------------------------ helpers
    def _day(self):
        return self.now().replace(hour=0, minute=0, second=0, microsecond=0)

    def _hhmm(self):
        return self.now().strftime("%H:%M")

    def _completed_bars(self, sym, start="04:00"):
        day = self._day()
        now_min = pd.Timestamp(self.now()).floor("min")
        bars = self.b.bars(sym, pd.Timestamp(day) + pd.Timedelta(start + ":00"), now_min)
        return bars[bars.index + pd.Timedelta("1min") <= pd.Timestamp(self.now())]

    # -------------------------------------------------------------- scan
    def premarket_scan(self) -> dict[str, dict]:
        sc, today = self.cfg["scan"], self._day().strftime("%Y-%m-%d")
        assets = {a["symbol"]: a for a in self.b.assets()
                  if a.get("tradable") and a.get("exchange") in LISTED}
        snaps = self.b.snapshots(sorted(assets))
        cands = []
        for sym, s in snaps.items():
            try:
                db, pdb, lt = s.get("dailyBar") or {}, s.get("prevDailyBar") or {}, s.get("latestTrade") or {}
                prev = db["c"] if db and str(db.get("t", ""))[:10] < today else pdb.get("c")
                last = lt.get("p")
                if prev and last and last / prev - 1 >= sc["min_premarket_gap"] and last >= sc["min_price"]:
                    cands.append((sym, float(prev)))
            except (KeyError, TypeError, ZeroDivisionError):
                continue
        self.watch = {}
        for sym, prev in cands:
            bars = self._completed_bars(sym)
            pm = premarket_stats(bars, sc["premarket_cutoff"])
            if pm["pm_last"] is None:
                continue
            gap = pm["pm_last"] / prev - 1
            info = {"symbol": sym, "prev_close": prev, "pm_gap": gap, **pm, "asset": assets[sym]}
            if gap < sc["min_premarket_gap"] or pm["pm_volume"] < sc["min_premarket_volume"]:
                continue
            if self.b.split_today(sym, today):
                self.j.log("skip", sym, reason="split_today")
                continue
            info["likely_htb"] = gap >= self.cfg["dnt"]["likely_htb_gap"]
            self.watch[sym] = info
            self.j.log("card", sym, card=card_text(info, self.cfg))
        self.j.log("scan_done", gappers=len(self.watch), candidates=len(cands))
        return self.watch

    # -------------------------------------------------------------- trade
    def step(self):
        rc = self.cfg["rules"]
        for sym, info in self.watch.items():
            if sym in self.done:
                continue
            bars = self._completed_bars(sym)
            reg = session(bars)
            if reg.empty:
                continue
            sig = find_entry(self.entry_rule, reg, info["pm_high"], rc)
            if sig is None:
                continue
            if sig.fill_idx < len(reg) - 1:
                self.done.add(sym)
                self.j.log("stale_signal", sym, signal_time=str(reg.index[max(sig.signal_idx, 0)]))
                continue
            self.done.add(sym)
            self._try_enter(sym, info, reg, sig)

    def _try_enter(self, sym, info, reg, sig):
        entry = float(reg["c"].iloc[-1])
        stop = stop_price(self.stop_rule, info["pm_high"], self.cfg["rules"])
        halts = count_halts(reg)
        if halts >= self.cfg["dnt"]["max_halts_before_entry"]:
            return self.j.log("reject", sym, reason="halts", halts=halts)
        if entry >= stop:
            return self.j.log("reject", sym, reason="price_above_stop", entry=entry, stop=stop)
        acct = self.b.account()
        positions = self.b.positions()
        shorts = [p for p in positions if float(p.get("qty", 0)) < 0]
        exposure = sum(abs(float(p["market_value"])) for p in shorts)
        margin = sum(margin_requirement(abs(float(p["market_value"])) / abs(float(p["qty"])), abs(int(float(p["qty"]))),
                                        self.cfg["risk"].get("margin_rule", "finra")) for p in shorts)
        qty, why = size_short(float(acct["equity"]), entry, stop, self.cfg["risk"], exposure, margin)
        asset = self.b.asset(sym)
        no = self.g.check(sym=sym, qty=qty, entry=entry, stop=stop, account=acct, positions=positions,
                          asset=asset, trades_today=self.trades_today, now_hhmm=self._hhmm())
        details = dict(rule=self.cfg["live"]["rule"], entry_est=round(entry, 4), stop=round(stop, 2), qty=qty,
                       size_limit=why, risk_dollars=round(qty * (stop - entry), 2))
        if no:
            return self.j.log("reject", sym, reason=no, **details)
        text = (f"SHORT {qty} {sym} at market (~${entry:.2f}), stop ${stop:.2f}. "
                f"Max planned loss ${qty * (stop - entry):,.0f} before slippage. "
                f"[{'LIVE MONEY' if not getattr(self.b, 'paper', True) else 'paper'}]")
        if not self.g.approve(text):
            return self.j.log("declined", sym, **details)
        cid = f"tfs-{self._day():%Y%m%d}-{sym}"
        order = self.b.short_with_stop(sym, qty, stop, cid)
        self.trades_today += 1
        self.opened.add(sym)
        self.j.log("order", sym, order_id=order.get("id"), **details)

    def manage(self):
        hhmm = self._hhmm()
        if self.exit_rule == "time_stop_1100" and not self.time_stop_done and hhmm >= self.cfg["rules"]["time_stop"]:
            self.time_stop_done = True
            for p in self.b.positions():
                if p["symbol"] in self.opened and float(p.get("unrealized_pl", 0)) <= 0:
                    self.b.close_position(p["symbol"])
                    self.j.log("exit", p["symbol"], reason="time_stop", unrealized_pl=p.get("unrealized_pl"))
        if not self.flattened and hhmm >= self.cfg["rules"]["eod_exit"]:
            self.flattened = True
            for p in self.b.positions():
                if p["symbol"] in self.opened:
                    self.b.close_position(p["symbol"])
                    self.j.log("exit", p["symbol"], reason="eod", unrealized_pl=p.get("unrealized_pl"))
            acct = self.b.account()
            self.j.log("day_end", equity=acct.get("equity"), last_equity=acct.get("last_equity"),
                       trades=self.trades_today)


def card_text(i: dict, cfg: dict) -> str:
    a = i.get("asset", {})
    borrow = "easy to borrow" if a.get("easy_to_borrow") else ("shortable, HTB" if a.get("shortable") else "NOT shortable")
    flags = []
    if not a.get("easy_to_borrow"):
        flags.append("no easy borrow")
    if i.get("likely_htb"):
        flags.append("gap 200%+ (squeeze risk)")
    stop = i["pm_high"] * (1 + cfg["rules"]["pmh_stop_buffer"])
    return (f"{i['symbol']} +{i['pm_gap']:.0%} pre-market | last ${i['pm_last']:.2f} | prev close ${i['prev_close']:.2f} | "
            f"PMH ${i['pm_high']:.2f} -> stop ${stop:.2f} | PM vol {i['pm_volume']:,.0f} | {borrow}"
            + (f" | DO NOT TOUCH: {', '.join(flags)}" if flags else ""))


def _make(cfg, input_fn=input):
    from tfs.live.alpaca import Alpaca
    live = cfg["live"]
    paper = live["mode"] != "live"
    if not paper:
        print("\n  LIVE MODE: this bot will short stocks with REAL money in the account whose keys are set.\n"
              "  Shorts can lose more than the account. Only the account owner may run this.")
        if input_fn("  Type I AM THE ACCOUNT OWNER to continue: ").strip() != "I AM THE ACCOUNT OWNER":
            raise SystemExit("Not confirmed. Nothing was sent.")
    broker = Alpaca.from_env(paper=paper, feed=live["data_feed"])
    journal = Journal(resolve(cfg, live["journal"]))
    guard = Guard(cfg, resolve(cfg, live["kill_switch_file"]), input_fn)
    return Bot(cfg, broker, journal, guard)


def morning_cards(cfg):
    bot = _make({**cfg, "live": {**cfg["live"], "mode": "paper"}})
    for sym, i in bot.premarket_scan().items():
        print(card_text(i, cfg))


def run_bot(cfg):
    bot = _make(cfg)
    clock = bot.b.clock()
    if not clock.get("is_open") and str(clock.get("next_open", ""))[:10] != bot.now().strftime("%Y-%m-%d"):
        raise SystemExit(f"Market is closed today. Next open: {clock.get('next_open')}")
    bot.j.log("start", mode=cfg["live"]["mode"], rule=cfg["live"]["rule"], auto_approve=cfg["live"]["auto_approve"])
    scanned = False
    while True:
        hhmm = bot._hhmm()
        if hhmm >= "16:00":
            break
        if not scanned and hhmm >= cfg["scan"]["premarket_cutoff"]:
            bot.premarket_scan()
            scanned = True
        if scanned and "09:30" <= hhmm < cfg["rules"]["entry_deadline"]:
            bot.step()
        if scanned:
            bot.manage()
        time.sleep(cfg["live"]["poll_seconds"] if hhmm >= "09:25" else 30)
    bot.j.log("stop")
