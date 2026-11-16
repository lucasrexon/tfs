"""The v2 trading bot: the final strategy in config.yaml (v2.strategy). Paper trading by default.

Runs on the ACCOUNT OWNER's computer every trading day, started before 09:20 ET:

  09:20  scan for +40% pre-market gappers; look up each one's going-concern warning and industry
         (SIC code) on SEC EDGAR
  09:25  for gappers now +50% or more (price >= $1, pre-market volume >= 500K) that have the
         warning, are not in a skipped industry (v2.strategy.skip_industries) and are easy to borrow: short 2.4% of the account with a pre-market limit order,
         at most 3 shorts open at once (random pick if more qualify). The owner types YES per order
         unless auto_approve is on.
         As soon as a short fills: one order that covers at -20% OR at the 3x stop (broker-side).
  all day  outside regular hours the broker's stop cannot fire, so the bot watches the price and
         covers with a limit order if it reaches the stop
  15:55  on the day after entry, anything still open is covered at market (the time limit)

State (open shorts, their target, stop and exit day) is kept in journal/v2_state.json, so the bot
can be stopped and restarted. Every decision goes to the journal CSV.

This strategy FAILED its one-time sealed-year test (reports/sealed_result.json): average trade
+1.7% against +7.6% in the build year, not distinguishable from luck. Paper trading is the test
that is left. See `python -m tfs paper` for how paper results compare.
"""
from __future__ import annotations

import json
import math
import random
import time
from datetime import date as Date, datetime, timedelta
from pathlib import Path

import pandas as pd

from tfs.config import resolve
from tfs.data.massive import NY
from tfs.live.guard import Journal
from tfs.scan import premarket_stats

LISTED = {"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS", "NYSEARCA"}
PRESCAN_GAP = 0.40
GC_LOOKBACK_DAYS = 456


# ---------------------------------------------------------------- SEC lookups (fresh every day)
class GoingConcern:
    """Was a 10-K/10-Q/20-F containing "substantial doubt" filed in the 456 days before today?"""

    def __init__(self, edgar):
        self.ed, self._ciks, self._memo = edgar, None, {}

    def cik(self, sym: str) -> str | None:
        if self._ciks is None:
            j = self.ed._get("https://www.sec.gov/files/company_tickers.json") or {}
            self._ciks = {v["ticker"].upper(): f"{int(v['cik_str']):010d}" for v in j.values()}
        return self._ciks.get(sym.upper())

    def __call__(self, sym: str, day: str) -> bool | None:
        """True / False, or None if it could not be checked (the bot then does not trade it)."""
        if (sym, day) in self._memo:
            return self._memo[(sym, day)]
        try:
            cik = self.cik(sym)
            if cik is None:
                out = None
            else:
                lo = (Date.fromisoformat(day) - timedelta(days=GC_LOOKBACK_DAYS)).isoformat()
                hi = (Date.fromisoformat(day) - timedelta(days=1)).isoformat()
                j = self.ed._get("https://efts.sec.gov/LATEST/search-index",
                                 {"q": '"substantial doubt"', "ciks": cik, "forms": "10-K,10-Q,20-F,10-KT",
                                  "dateRange": "custom", "startdt": lo, "enddt": hi})
                out = (j or {}).get("hits", {}).get("total", {}).get("value", 0) > 0
        except Exception:
            out = None
        self._memo[(sym, day)] = out
        return out

    def industry(self, sym: str) -> str | None:
        """The same 9 groups as the research (tfs.sec.industry), from the SEC's SIC code and company name."""
        from tfs.sec import industry
        if sym in self._memo:
            return self._memo[sym]
        try:
            cik = self.cik(sym)
            j = self.ed._get(f"https://data.sec.gov/submissions/CIK{cik}.json") if cik else None
            out = industry(j.get("sic"), j.get("name", "")) if j else None
        except Exception:
            out = None
        self._memo[sym] = out
        return out


# ---------------------------------------------------------------- the bot
class BotV2:
    def __init__(self, cfg: dict, broker, journal: Journal, going_concern, state_file: Path,
                 now_fn=lambda: datetime.now(NY), input_fn=input, rng=None):
        self.cfg, self.st, self.live = cfg, cfg["v2"]["strategy"], cfg["live"]
        self.b, self.j, self.gc, self.now, self.input = broker, journal, going_concern, now_fn, input_fn
        self.state_file = Path(state_file)
        self.kill = resolve(cfg, self.live["kill_switch_file"]) if "_root" in cfg else Path(self.live["kill_switch_file"])
        self.rng = rng or random.Random()
        self.pos: dict[str, dict] = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        self.cands: dict[str, dict] = {}
        self.entered_today = False

    # ------------------------------------------------------------ helpers
    def today(self) -> str: return self.now().strftime("%Y-%m-%d")
    def hhmm(self) -> str: return self.now().strftime("%H:%M")
    def regular_hours(self) -> bool: return "09:30" <= self.hhmm() < "16:00"

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.pos, indent=1))

    def _bars(self, sym):
        day = pd.Timestamp(self.today(), tz=NY)
        now = pd.Timestamp(self.now())
        b = self.b.bars(sym, day + pd.Timedelta("4h"), now.floor("min"))
        return b[b.index + pd.Timedelta("1min") <= now]

    # ------------------------------------------------------------ 09:20 scan
    def prescan(self):
        sc, today = self.cfg["scan"], self.today()
        assets = {a["symbol"]: a for a in self.b.assets() if a.get("tradable") and a.get("exchange") in LISTED}
        snaps = self.b.snapshots(sorted(assets))
        self.cands = {}
        for sym, s in snaps.items():
            try:
                db, pdb, lt = s.get("dailyBar") or {}, s.get("prevDailyBar") or {}, s.get("latestTrade") or {}
                prev = db["c"] if db and str(db.get("t", ""))[:10] < today else pdb.get("c")
                last = lt.get("p")
                if prev and last and last / prev - 1 >= PRESCAN_GAP and last >= self.st["min_price"]:
                    self.cands[sym] = {"prev_close": float(prev), "asset": assets[sym]}
            except (KeyError, TypeError, ZeroDivisionError):
                continue
        skip = set(self.st.get("skip_industries") or [])
        for sym, c in self.cands.items():
            c["going_concern"] = self.gc(sym, today)
            c["industry"] = self.gc.industry(sym) if (skip and hasattr(self.gc, "industry")) else None
            self.j.log("v2_candidate", sym, prev_close=c["prev_close"], going_concern=c["going_concern"], industry=c["industry"])
        self.j.log("v2_prescan_done", candidates=len(self.cands))

    # ------------------------------------------------------------ 09:25 entries
    def enter(self):
        self.entered_today = True
        sc = self.cfg["scan"]
        ok = []
        for sym, c in self.cands.items():
            pm = premarket_stats(self._bars(sym), "09:25")
            if pm["pm_last"] is None:
                continue
            gap = pm["pm_last"] / c["prev_close"] - 1
            why = None
            if gap < self.st["min_gap"]:
                why = "gap_below_min"
            elif pm["pm_last"] < self.st["min_price"]:
                why = "price_below_min"
            elif pm["pm_volume"] < sc["min_premarket_volume"]:
                why = "premarket_volume_low"
            elif self.st["require_going_concern"] and c["going_concern"] is not True:
                why = "no_going_concern" if c["going_concern"] is False else "going_concern_unknown"
            elif self.st.get("skip_industries") and c.get("industry") in self.st["skip_industries"]:
                why = "skipped_industry"
            elif self.st.get("skip_industries") and c.get("industry") is None:
                why = "industry_unknown"
            elif sym in self.pos:
                why = "already_in_position"
            if why:
                self.j.log("v2_skip", sym, reason=why, gap=round(gap, 3))
                continue
            ok.append((sym, pm["pm_last"], gap))
        slots = self.st["max_open"] - len(self.pos)
        self.rng.shuffle(ok)
        for sym, last, gap in ok:
            if slots <= 0:
                self.j.log("v2_skip", sym, reason="max_open_reached", gap=round(gap, 3))
                continue
            if self._short(sym, last, gap):
                slots -= 1
        self.save()

    def _guard(self, sym, qty, acct, asset) -> str | None:
        if self.kill.exists():
            return "kill_switch"
        if acct.get("trading_blocked") or acct.get("account_blocked"):
            return "account_blocked"
        if not acct.get("shorting_enabled", True):
            return "shorting_not_enabled"
        if not asset.get("shortable") or not asset.get("easy_to_borrow"):
            return "not_easy_to_borrow"
        if qty <= 0:
            return "size_zero"
        if not ("09:25" <= self.hhmm() < "09:29"):
            return "outside_entry_window"
        return None

    def _short(self, sym, last, gap) -> bool:
        acct = self.b.account()
        limit = math.floor(last * (1 - self.cfg["costs"]["slippage_pct"]) * 100) / 100
        qty = int(self.st["position_pct"] * float(acct["equity"]) / limit) if limit > 0 else 0
        no = self._guard(sym, qty, acct, self.b.asset(sym))
        info = dict(gap=round(gap, 3), limit=limit, qty=qty, position=round(qty * limit, 2))
        if no:
            self.j.log("v2_reject", sym, reason=no, **info)
            return False
        text = (f"SHORT {qty} {sym} (+{gap:.0%} pre-market) with a limit at ${limit:.2f} = ${qty * limit:,.0f}, "
                f"{self.st['position_pct']:.1%} of the account. Cover at -{self.st['target']:.0%} or at the "
                f"{self.st['disaster_stop_x']:.0f}x stop, else at tomorrow's close. [{'paper' if getattr(self.b, 'paper', True) else 'LIVE MONEY'}]")
        if not (self.live["auto_approve"] or self.input(f"\n  {text}\n  Account owner: type YES to send: ").strip() == "YES"):
            self.j.log("v2_declined", sym, **info)
            return False
        o = self.b.limit_order(sym, qty, "sell", limit, f"tfs2-{self.today()}-{sym}", extended=True)
        self.pos[sym] = {"status": "pending", "entry_order": o.get("id"), "qty": qty, "limit": limit,
                         "entry_date": self.today(), "exit_date": self.b.next_trading_day(self.today())}
        self.j.log("v2_order", sym, order_id=o.get("id"), **info)
        return True

    # ------------------------------------------------------------ fills, exits, watching
    def manage(self):
        for sym in list(self.pos):
            p = self.pos[sym]
            if p["status"] == "pending":
                self._check_fill(sym, p)
            elif p["status"] == "open":
                self._check_open(sym, p)
        self.save()

    def _check_fill(self, sym, p):
        o = self.b.get_order(p["entry_order"])
        filled = int(float(o.get("filled_qty") or 0))
        if o.get("status") == "filled" or (filled and self.hhmm() >= "09:29"):
            if o.get("status") != "filled":
                self.b.cancel_order(p["entry_order"])            # keep the part that filled
            px = float(o["filled_avg_price"])
            p.update(status="open", qty=filled, entry_px=px, target=round(px * (1 - self.st["target"]), 2),
                     stop=round(px * self.st["disaster_stop_x"], 2))
            x = self.b.exit_oco(sym, filled, p["target"], p["stop"], f"tfs2x-{self.today()}-{sym}")
            p["exit_order"] = x.get("id")
            self.j.log("v2_filled", sym, qty=filled, entry_px=px, target=p["target"], stop=p["stop"], exit_day=p["exit_date"])
        elif self.hhmm() >= "09:29":
            self.b.cancel_order(p["entry_order"])
            self.j.log("v2_no_fill", sym, limit=p["limit"])
            del self.pos[sym]

    def _held(self, sym) -> bool:
        return any(q["symbol"] == sym and float(q.get("qty", 0)) < 0 for q in self.b.positions())

    def _check_open(self, sym, p):
        if not self._held(sym):                                   # the broker covered it: target or stop
            fills = [o for o in self.b.filled_orders(sym, p["entry_date"]) if o.get("side") == "buy" and o.get("status") == "filled"]
            legs = fills + [l for o in self.b.filled_orders(sym, p["entry_date"]) for l in (o.get("legs") or [])
                            if l.get("status") == "filled" and l.get("side") == "buy"]
            px = float(legs[0]["filled_avg_price"]) if legs else None
            reason = "unknown" if px is None else ("target" if px <= p["target"] * 1.001 else "stop")
            return self._closed(sym, p, px, reason)
        if self.today() >= p["exit_date"] and self.hhmm() >= self.cfg["rules"]["eod_exit"] and self.regular_hours():
            if p.get("exit_order"):
                self.b.cancel_order(p["exit_order"])
            self.b.close_position(sym)
            px = self.b.last_price(sym)
            return self._closed(sym, p, px, "time_limit" if self.today() == p["exit_date"] else "late_time_limit")
        if not self.regular_hours() and not p.get("ext_cover"):
            last = self.b.last_price(sym)
            if last and last >= p["stop"]:
                if p.get("exit_order"):
                    self.b.cancel_order(p["exit_order"])
                lim = round(last * 1.03, 2)
                o = self.b.limit_order(sym, p["qty"], "buy", lim, f"tfs2s-{self.today()}-{sym}", extended=True)
                p["ext_cover"] = o.get("id")
                self.j.log("v2_ext_stop", sym, last=last, stop=p["stop"], limit=lim)

    def _closed(self, sym, p, px, reason):
        ret = (p["entry_px"] - px) / p["entry_px"] if px else None
        self.j.log("v2_exit", sym, reason=reason, entry_date=p["entry_date"], entry_px=p["entry_px"], exit_px=px,
                   qty=p["qty"], ret=None if ret is None else round(ret, 4),
                   pnl=None if px is None else round((p["entry_px"] - px) * p["qty"], 2))
        del self.pos[sym]


# ---------------------------------------------------------------- running it
def _make(cfg, input_fn=input):
    from tfs.live.alpaca import Alpaca
    from tfs.sec import Edgar
    live = cfg["live"]
    paper = live["mode"] != "live"
    if not paper:
        if not live.get("allow_overnight"):
            raise SystemExit("This strategy holds shorts overnight. The account owner must set live.allow_overnight: true.")
        print("\n  LIVE MODE: this bot will short stocks with REAL money and hold them overnight.\n"
              "  Shorts can lose more than the position. This strategy failed its sealed-year test.\n"
              "  Only the account owner may run this.")
        if input_fn("  Type I AM THE ACCOUNT OWNER to continue: ").strip() != "I AM THE ACCOUNT OWNER":
            raise SystemExit("Not confirmed. Nothing was sent.")
    broker = Alpaca.from_env(paper=paper, feed=live["data_feed"])
    journal = Journal(resolve(cfg, live["journal"]))
    gc = GoingConcern(Edgar(resolve(cfg, cfg["data"]["cache_dir"]) / "live"))
    return BotV2(cfg, broker, journal, gc, resolve(cfg, "journal") / "v2_state.json", input_fn=input_fn)


def run_bot_v2(cfg):
    bot = _make(cfg)
    clock = bot.b.clock()
    if not clock.get("is_open") and str(clock.get("next_open", ""))[:10] != bot.today():
        raise SystemExit(f"Market is closed today. Next open: {clock.get('next_open')}")
    bot.j.log("v2_start", mode=cfg["live"]["mode"], strategy=cfg["v2"]["strategy"]["name"], open_shorts=list(bot.pos))
    scanned = False
    while bot.hhmm() < "16:05":
        h = bot.hhmm()
        if not scanned and h >= "09:20":
            bot.prescan(); scanned = True
        if scanned and not bot.entered_today and "09:25" <= h < "09:29":
            bot.enter()
        bot.manage()
        time.sleep(5 if "09:24" <= h < "09:31" else cfg["live"]["poll_seconds"])
    acct = bot.b.account()
    bot.j.log("v2_day_end", equity=acct.get("equity"), open_shorts=list(bot.pos))


def paper_report(cfg):
    """Compare closed paper trades with what the research expected."""
    f = resolve(cfg, cfg["live"]["journal"])
    if not f.exists():
        raise SystemExit("No journal yet: run the bot first.")
    j = pd.read_csv(f)
    ex = pd.DataFrame([json.loads(d) for d in j.loc[j.event == "v2_exit", "details"]])
    if ex.empty:
        print("No closed v2 trades yet.")
        return
    r = ex["ret"].dropna()
    print(f"Closed trades: {len(ex)} | covered at target {(ex.reason == 'target').mean():.0%} | stopped {(ex.reason == 'stop').mean():.0%}")
    print(f"Average trade (before fees): {r.mean():+.1%}   total P&L ${ex.pnl.sum():,.2f}")
    print("Expected from the research: build year +7.6% per trade, sealed year +1.7% per trade (after costs).")
    if len(r) < 50:
        print(f"Only {len(r)} trades: too few to judge. About 50 or more are needed.")
