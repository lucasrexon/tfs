import numpy as np
import pandas as pd

from strat.factors import portfolio, stats


def test_portfolio_picks_top_names_and_charges_turnover():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    score = pd.DataFrame({"A": [3, 1, 1], "B": [2, 3, 3], "C": [1, 2, 2]}, index=idx, dtype=float)
    ret = pd.DataFrame({"A": [0.10, 0.0, 0.0], "B": [0.0, 0.02, 0.0], "C": [0.0, 0.0, 0.0]}, index=idx)
    mask = pd.DataFrame(True, index=idx, columns=score.columns)
    r, n, turn = portfolio(score, mask, ret, top=1)
    assert list(n) == [1, 1, 1]
    assert r.iloc[0] == 0.10 - 1.0 * 0.0005            # buy A: 100% traded
    assert r.iloc[1] == 0.02 - 2.0 * 0.0005            # sell A, buy B
    assert r.iloc[2] == 0.0                             # hold B


def test_stats_dates_returns_in_the_month_they_are_earned():
    idx = pd.date_range("2021-11-30", periods=2, freq="ME")      # formed Nov and Dec 2021
    r = pd.Series([0.10, -0.20], index=idx)                       # earned Dec 2021 and Jan 2022
    st = stats(r, pd.Series(dtype=float))
    assert np.isclose(st["worst_year"], -0.20) and np.isclose(st["best_year"], 0.10)
