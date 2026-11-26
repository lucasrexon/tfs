import numpy as np
import pandas as pd
import pytest

from hdc.portfolio import RATES, simulate, year_tax
from hdc.pricing import american_call, itm_strikes
from hdc.synthetic import listed_expiry
from hdc.trades import Costs, dividend_qualifies, outcomes, select_call


def test_tree_matches_black_scholes_without_dividend_and_prices_early_exercise():
    from math import erf, exp, log, sqrt
    N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    S, K, T, r, s = 50, 48, 10 / 365, 0.04, 0.25
    d1 = (log(S / K) + (r + s * s / 2) * T) / (s * sqrt(T))
    bs = S * N(d1) - K * exp(-r * T) * N(d1 - s * sqrt(T))
    assert american_call(S, K, T, r, s) == pytest.approx(bs, abs=0.01)
    # a dividend bigger than the time value: worth about intrinsic (exercise the day before)
    c = american_call(50, 47, 10 / 365, 0.04, 0.25, div=0.5, t_div=2 / 365)
    assert 3.0 <= c < 3.05


def test_itm_strikes_use_listed_spacing():
    assert itm_strikes(27.93) == [27.0, 26.0] and itm_strikes(592.7) == [590.0, 585.0] and itm_strikes(24.3) == [24.0, 23.5]


def test_select_call_follows_the_rule():
    q = [(49.0, 1.30), (48.0, 2.05)]            # time values 0.30 and 0.05
    assert select_call(50, 0.25, q, "below")[0] == 48.0       # 1-strike call has 0.30 >= 0.25: take the 2-strike
    assert select_call(50, 0.40, q, "below")[0] == 49.0
    assert select_call(50, 0.25, q, "above")[0] == 49.0
    assert select_call(50, 0.01, q, "below") is None
    assert select_call(50, 0.25, [(49.0, 0.90)], "below")[2] == 0.0   # stale print below intrinsic: sold at intrinsic


def _closes():
    days = ["2025-03-06", "2025-03-07", "2025-03-10", "2025-03-11", "2025-03-12", "2025-03-13", "2025-03-14"]
    # entry 03-06 (ex-2), ex 03-10, exit 03-11, expiry 03-14
    return pd.Series([50.0, 50.2, 49.7, 49.8, 50.0, 50.1, 50.3], index=days)


EV = {"entry": "2025-03-06", "ex": "2025-03-10", "exit": "2025-03-11", "div": 0.5}


def test_paths_add_up():
    closes = _closes()
    calls = pd.Series([2.05, 2.21, 1.72, 1.85, 2.0, 2.1, 2.3], index=closes.index)   # K = 48
    c = Costs()
    o = outcomes(EV, 48.0, 2.05, closes, calls, "2025-03-14", c)
    k = o["kept"]
    assert k["kind"] == "kept"
    assert k["stock"] == pytest.approx(49.8 - 50.0) and k["option"] == pytest.approx(2.05 - 1.85) and k["dividend"] == 0.5
    assert k["cost"] == pytest.approx(50 * 0.0005 + 49.8 * 0.0005 + 2 * 0.0065)
    a = o["assigned"]                                     # delivered at 48 the evening before the ex-date
    assert a["net"] == pytest.approx((48 - 50) + 2.05 - (50 * 0.0005 + 0.0065)) and a["dividend"] == 0
    u = o["unhedged"]
    assert u["option"] == 0 and u["net"] == pytest.approx(-0.2 + 0.5 - (50 + 49.8) * 0.0005)
    # deep in the money at exit (time value 0.05 - 0.02 <= 0.03): left to be assigned at expiry
    calls2 = calls.copy()
    calls2["2025-03-11"] = 1.82
    e = outcomes(EV, 48.0, 2.05, closes, calls2, "2025-03-14", c)["kept"]
    assert e["kind"] == "expiry_assigned" and e["exit_day"] == "2025-03-14"
    assert e["net"] == pytest.approx((48 - 50) + 2.05 + 0.5 - (50 * 0.0005 + 0.0065))


def test_spec_scenario_b_dividend_does_not_qualify():
    # bought 2 days before the ex-date with an in-the-money call until the day after, sold 62 days later
    assert not dividend_qualifies("2025-03-10", "2025-03-06", "2025-03-06", "2025-03-11", "2025-05-07")
    # owned well before and hedged only around the ex-date: qualifies
    assert dividend_qualifies("2025-03-10", "2024-12-30", "2025-03-06", "2025-03-11", "2025-05-07")


def test_capital_losses_cannot_shelter_dividends_in_a_standalone_account():
    r = RATES["A_short_term"]
    tax, carry = year_tax(r, ordinary=10_000, qualified=0, capital=-8_000, carry=0)
    assert tax == pytest.approx((10_000 - 3_000) * 0.37) and carry == pytest.approx(-5_000)
    tax2, _ = year_tax(RATES["A_short_term_offset"], 10_000, 0, -8_000, 0)
    assert tax2 == pytest.approx(2_000 * 0.37)


def test_simulation_cash_adds_up_for_one_trade():
    closes = _closes()
    calls = pd.Series([2.05, 2.21, 1.72, 1.85, 2.0, 2.1, 2.3], index=closes.index)
    c = Costs()
    o = outcomes(EV, 48.0, 2.05, closes, calls, "2025-03-14", c)
    t = {"entry": EV["entry"], "ex": EV["ex"], "exit": EV["exit"], "div": 0.5, "s0": 50.0, "c0": 2.05, "paths": o,
         "rational_exercise": False}
    cal = list(closes.index)
    rate = pd.Series(0.0, index=pd.to_datetime(cal))
    res = simulate([t], cal, rate, c, "kept", None, "pre_tax", capital=100_000, fraction=0.5)
    n = int(100_000 * 0.5 // (100 * 50)) * 100
    assert res["equity"].iloc[-1] == pytest.approx(100_000 + n * o["kept"]["net"])
    assert len(res["log"]) == 1


def test_expiry_calendar():
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2011-01-01", "2025-12-31")]
    assert listed_expiry("2011-03-04", days) == "2011-03-18" and listed_expiry("2011-03-14", days) is None
    assert listed_expiry("2025-04-10", days) == "2025-04-18"


def test_share_history_joins_predecessor_and_backfills_three_years():
    from hdc.events import company_shares, shares_asof
    sh = pd.DataFrame({"cik": [34088, 34088, 2115436], "end": ["2024-12-31", "2026-03-31", "2026-06-30"],
                       "shares": [4.3e9, 4.2e9, 4.1e9], "loc": "US-TX"})
    s = company_shares(sh, 2115436)
    assert list(s.end) == ["2024-12-31", "2026-03-31", "2026-06-30"]
    one = sh[sh.cik == 2115436]
    got = shares_asof(one, pd.Index(["2022-01-03", "2024-01-02", "2026-07-01"]))
    assert pd.isna(got.iloc[0]) and got.iloc[1] == 4.1e9 and got.iloc[2] == 4.1e9


def test_regular_dividend_ignores_specials_and_irregular_payers():
    from hdc.events import regular_dividend
    q = ["2023-03-16", "2023-06-15", "2023-09-14", "2023-11-10", "2023-12-14"]     # Corebridge: two specials
    f, reg, sp = regular_dividend(q, [0.23, 0.85, 0.23, 1.16, 0.23], 4)
    assert (f, reg, sp) == (4, 0.23, False)
    assert regular_dividend(q, [0.23, 0.85, 0.23, 1.16, 0.23], 3)[2] is True
    hca = ["2012-02-14", "2012-10-31", "2012-12-13"]                                  # specials only
    assert regular_dividend(hca, [2.0, 2.5, 2.0], 2)[0] is None
    met = ["2009-11-04", "2010-11-04", "2011-11-07"]                                  # annual payer
    assert regular_dividend(met, [0.74, 0.74, 0.74], 2)[:2] == (1, 0.74)
    kdp = ["2017-06-14", "2017-09-14", "2017-12-14", "2018-03-20", "2018-07-10", "2018-10-04"]
    f, reg, sp = regular_dividend(kdp, [0.58, 0.58, 0.58, 0.58, 103.75, 0.15], 5)     # merger special, then a cut
    assert f == 4 and reg == 0.15 and not sp


def test_regular_dividend_merges_a_quarter_paid_in_two_parts():
    from hdc.events import regular_dividend
    d = ["2015-12-29", "2016-03-29", "2016-06-28", "2016-07-07", "2016-09-28", "2016-12-28"]
    assert regular_dividend(d, [1.2, 1.2, 1.2, 0.144, 1.056, 1.2], 5)[:2] == (4, 1.2)
    assert regular_dividend(d, [1.2, 1.2, 1.2, 0.144, 1.056, 1.2], 4)[:2] == (4, 1.056)


def test_regular_dividend_frequent_specials_and_extra_payments():
    from hdc.events import regular_dividend
    afg = ["2020-10-14", "2021-01-14", "2021-04-14", "2021-07-14", "2021-10-14", "2021-11-12", "2021-12-22", "2022-01-13", "2022-03-14", "2022-04-13", "2022-05-19",
           "2022-07-14", "2022-10-13", "2022-11-14"]
    amt = [0.5, 0.5, 0.5, 0.5, 0.56, 4.0, 2.0, 0.56, 2.0, 0.56, 8.0, 0.56, 0.63, 2.0]
    assert regular_dividend(afg, amt, 13)[2] is True                # the Nov 2022 special
    f, reg, _ = regular_dividend(afg, amt, 12)
    assert f == 4 and abs(reg - 0.63) < 1e-9
    vno = ["2011-11-09", "2012-01-26", "2012-05-10", "2012-08-08", "2012-11-07", "2012-12-06", "2013-01-24"]
    va = [0.505, 0.505, 0.505, 0.505, 0.505, 0.732, 0.534]
    assert regular_dividend(vno, va, 5)[2] is True                  # off-schedule extra payment
    assert regular_dividend(vno, va, 6)[:2] == (4, 0.534)
