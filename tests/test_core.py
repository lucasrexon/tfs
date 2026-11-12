import numpy as np
import pandas as pd
import pytest

from tfs.backtest import simulate_gapper, trade_costs
from tfs.config import load_config
from tfs.data.massive import NY, split_factor
from tfs.portfolio import simulate_account
from tfs.risk import margin_requirement, size_short
from tfs.rules import count_halts, find_entry, session
from tfs.scan import premarket_stats, stage1, stage2
from tfs.synthetic import FakeClient

CFG = load_config()


def bars_from(day, start_min, closes, opens=None, highs=None, lows=None, vol=1e5):
    idx = pd.Timestamp(day, tz=NY) + pd.to_timedelta(np.arange(start_min, start_min + len(closes)), unit="min")
    c = np.asarray(closes, float)
    o = np.asarray(opens if opens is not None else np.r_[c[0], c[:-1]], float)
    h = np.asarray(highs if highs is not None else np.maximum(o, c), float)
    lo = np.asarray(lows if lows is not None else np.minimum(o, c), float)
    df = pd.DataFrame({"o": o, "h": h, "l": lo, "c": c, "v": vol}, index=pd.DatetimeIndex(idx, name="ts"))
    df["vw"] = (df.h + df.l + df.c) / 3
    return df


def test_split_factor_directions():
    rev = [{"execution_date": "2025-01-10", "split_from": 10, "split_to": 1}]
    assert split_factor(rev, "2025-01-09") == pytest.approx(0.1)   # adjusted $10 was really $1
    assert split_factor(rev, "2025-01-10") == 1.0                   # on/after execution: no change
    fwd = [{"execution_date": "2025-01-10", "split_from": 1, "split_to": 2}]
    assert split_factor(fwd, "2025-01-01") == pytest.approx(2.0)


def test_premarket_cutoff_excludes_0925_bar():
    b = bars_from("2025-03-03", 240, np.r_[np.full(325, 1.0), [9.0] * 20])   # 04:00 .. 09:44
    pm = premarket_stats(b, "09:25")
    assert pm["pm_last"] == 1.0 and pm["pm_bars"] == 325


def test_margin_matches_report_example():
    assert margin_requirement(3.0, 2000, "finra") == 6000
    assert margin_requirement(3.0, 2000, "house5") == 10000
    assert margin_requirement(10.0, 1000, "finra") == 5000


def test_size_short_risk_rule():
    shares, why = size_short(10000, 3.00, 3.45, CFG["risk"])
    assert shares == 111 and why == "risk"
    assert size_short(10000, 3.5, 3.45, CFG["risk"])[0] == 0


@pytest.mark.parametrize("rule", ["open", "vwap_loss", "first_red_5m", "pmh_fail"])
def test_entries_have_no_lookahead(rule):
    c = FakeClient("2024-01-02", "2024-03-29")
    for (t, d), ev in list(c.events.items())[:40]:
        reg = session(c.minute_bars(t, d))
        pmh = float(c.minute_bars(t, d).between_time("04:00", "09:29")["h"].max())
        s = find_entry(rule, reg, pmh, CFG["rules"])
        if s is None or s.signal_idx < 0:
            continue
        cut = reg.iloc[: s.signal_idx + 1]
        s2 = find_entry(rule, cut, pmh, CFG["rules"])
        assert s2 is not None and s2.signal_idx == s.signal_idx, (t, d)


def _gapper(day="2025-03-03", pmh=2.0, prev=1.0, nxt=None):
    return {"date": day, "ticker": "TEST", "prev_close": prev, "prev_low": prev * 0.97, "prev_prev_close": prev,
            "pm_gap": 0.9, "pm_high": pmh, "pm_last": 1.9, "factor": 1.0, "days_since_ipo": 1000,
            "likely_htb": False, "next": nxt or []}


def _cfg(entries, exits, stops):
    cfg = load_config()
    cfg["rules"].update(entries=entries, exits=exits, stops=stops)
    return cfg


def test_stop_gap_fills_at_open_plus_slippage():
    # open at 1.90 (below stop 2.10), next bar opens at 2.50 -> stop fills at 2.50 * 1.03
    closes = [1.90, 2.50] + [2.4] * 388
    opens = [1.90, 2.50] + [2.4] * 388
    reg = bars_from("2025-03-03", 570, closes, opens=opens)
    tr = simulate_gapper(_gapper(), reg, _cfg(["open"], ["eod"], ["pmh_plus_5"]))
    t = [x for x in tr if x["status"] == "filled"][0]
    assert t["stopped"] and t["exit_px"] == pytest.approx(2.50 * 1.03)
    assert t["ret"] < -0.33


def test_overnight_gap_over_stop():
    reg = bars_from("2025-03-03", 570, [1.9] * 390)
    nxt = [{"date": "2025-03-04", "o": 3.0, "h": 3.5, "l": 2.9, "c": 3.2, "v": 1e6}]
    tr = simulate_gapper(_gapper(nxt=nxt), reg, _cfg(["open"], ["close_d2"], ["pmh_plus_5"]))
    t = [x for x in tr if x["status"] == "filled"][0]
    assert t["stopped"] and t["exit_px"] == pytest.approx(3.0 * 1.03) and t["nights"] == 1


def test_eod_fade_profit_after_costs():
    closes = np.linspace(1.9, 1.2, 390)
    reg = bars_from("2025-03-03", 570, closes)
    tr = simulate_gapper(_gapper(), reg, _cfg(["open"], ["eod"], ["none"]))
    t = [x for x in tr if x["status"] == "filled"][0]
    assert 0 < t["ret"] < t["gross_ret"]
    assert t["exit_time"] == "15:55"


def test_costs_borrow_only_overnight():
    _, _, f0 = trade_costs(2.0, 1.5, 0, False, False, CFG["costs"])
    _, _, f2 = trade_costs(2.0, 1.5, 2, False, False, CFG["costs"])
    assert f2 - f0 == pytest.approx(2.0 * 1.0 * 2 / 360)


def test_halt_detection():
    b = bars_from("2025-03-03", 570, [2.0] * 60)
    b = b.drop(b.index[20:27])
    assert count_halts(b) == 1


def test_dnt_skips_after_two_halts():
    b = bars_from("2025-03-03", 570, np.r_[np.full(30, 1.9), np.linspace(1.9, 1.5, 360)])
    b = b.drop(b.index[10:16].append(b.index[30:36]))
    cfg = _cfg(["vwap_loss"], ["eod"], ["pmh_plus_5"])
    tr = simulate_gapper(_gapper(), b, cfg)
    assert all(x["status"] == "skipped_dnt" for x in tr)


def test_scan_pipeline_filters():
    c = FakeClient("2024-01-02", "2024-06-28")
    cfg = load_config()
    cfg["data"].update(start="2024-01-02", end="2024-06-28")
    g = stage2(c, cfg, stage1(c, cfg, c.tickers(["CS"])))
    kinds = {c.events.get((t, d), {}).get("kind") for t, d in zip(g.ticker, g.date)}
    assert kinds <= {"fade", "squeeze"}            # weak pre-market and thin volume are filtered out
    assert "WRNTW" not in set(g.ticker)            # not common stock
    assert (g.pm_gap >= 0.5).all() and (g.pm_price_real >= 1).all()


def test_account_daily_loss_limit():
    rows = []
    for k in range(3):
        rows.append({"status": "filled", "entry_date": "2025-03-03", "entry_time": f"09:3{k}", "exit_date": "2025-03-03",
                     "exit_time": f"09:4{k}", "ticker": f"T{k}", "entry_px": 2.0, "stop_px": 2.2,
                     "pnl_per_share": -0.25, "ret": -0.125})
    curve, s = simulate_account(pd.DataFrame(rows), CFG)
    assert s["trades"] < 3 or s["daily_loss_limit"] >= 0
    assert s["max_drawdown"] <= 0
