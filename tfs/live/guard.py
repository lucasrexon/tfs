"""The last check before any order. Every 'no' is written to the journal.

These are the report's hard-coded rules; the bot cannot turn them off.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from tfs.data.massive import NY


class Journal:
    """Append-only CSV: every scan, card, signal, refusal, order and exit, timestamped."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(["time_et", "event", "symbol", "details"])

    def log(self, event: str, symbol: str = "", **details):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now(NY).isoformat(timespec="seconds"), event, symbol,
                                    json.dumps(details, default=str)])
        print(f"[{datetime.now(NY):%H:%M:%S}] {event:<12} {symbol:<6} {json.dumps(details, default=str)}")


class Guard:
    def __init__(self, cfg: dict, kill_switch: Path, input_fn=input):
        self.cfg, self.kill_switch, self.input = cfg, Path(kill_switch), input_fn
        self.risk, self.live = cfg["risk"], cfg["live"]

    def check(self, *, sym: str, qty: int, entry: float, stop: float | None, account: dict,
              positions: list[dict], asset: dict, trades_today: int, now_hhmm: str) -> str | None:
        """Return None if the order may go, else the reason it may not."""
        if self.kill_switch.exists():
            return "kill_switch"
        if account.get("trading_blocked") or account.get("account_blocked"):
            return "account_blocked"
        if not account.get("shorting_enabled", True):
            return "shorting_not_enabled"
        eq, last = float(account["equity"]), float(account.get("last_equity") or account["equity"])
        if eq - last <= -self.risk["daily_loss_limit"] * last:
            return "daily_loss_limit"
        if trades_today >= self.risk["max_trades_per_day"]:
            return "max_trades_per_day"
        if any(p["symbol"] == sym for p in positions):
            return "already_in_position"          # never add to a short
        if not asset.get("shortable") or not asset.get("easy_to_borrow"):
            return "not_borrowable"
        if stop is None:
            return "no_stop"                      # live trades always carry a stop
        if qty <= 0:
            return "size_zero"
        exposure = sum(abs(float(p.get("market_value", 0))) for p in positions if float(p.get("qty", 0)) < 0)
        if exposure + qty * entry > self.risk["max_short_exposure"] * eq:
            return "exposure_cap"
        if now_hhmm >= self.cfg["rules"]["entry_deadline"]:
            return "past_entry_deadline"
        return None

    def approve(self, text: str) -> bool:
        if self.live["auto_approve"]:
            return True
        return self.input(f"\n  {text}\n  Account owner: type YES to send this order: ").strip() == "YES"
