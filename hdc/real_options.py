"""Real option prices (Massive daily bars) for events from Dec 2024 on (the free plan's 2-year window).

The prices are each day's LAST TRADE, not bid/ask quotes, and thin in-the-money strikes may not trade
every day. A strike with no trade on the entry day cannot be priced and is skipped; a later day with
no trade is filled from the calibrated model (counted in the report).

    python -m hdc.real_options download [max_calls]   # resumable; stops after max_calls requests
"""
from __future__ import annotations

import sys

import pandas as pd

from hdc.data import massive, option_bars, option_contracts

REAL_FROM = "2024-12-02"


def underlying_contracts(cl, ticker: str, ev: pd.DataFrame) -> pd.DataFrame:
    real = ev.s0 * ev.real_factor
    exp_to = (pd.Timestamp(ev.entry.max()) + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    return option_contracts(cl, ticker, REAL_FROM, exp_to, 0.85 * real.min(), real.max())


def pick(contracts: pd.DataFrame, entry: str, s_real: float):
    """Expiry 7-14 calendar days out (the nearest) and the two listed strikes just below the price."""
    if contracts.empty:
        return None
    e = pd.Timestamp(entry)
    c = contracts.copy()
    c["dte"] = (pd.to_datetime(c.expiration_date) - e).dt.days
    c = c[(c.dte >= 7) & (c.dte <= 14) & (c.strike_price < s_real)]
    if c.empty:
        return None
    exp = c.expiration_date.min()
    c = c[c.expiration_date == exp].sort_values("strike_price", ascending=False)
    return exp, list(c.head(2)[["ticker", "strike_price"]].itertuples(index=False, name=None))


def event_bars(cl, ev: dict, contracts: pd.DataFrame, need_second) -> dict | None:
    """Bars for the 1-strike ITM call, and the 2-strike one only if need_second(first bars) says so."""
    p = pick(contracts, ev["entry"], ev["s0"] * ev["real_factor"])
    if p is None:
        return None
    exp, strikes = p
    out = {"expiry": exp, "strikes": []}
    for i, (opt, k) in enumerate(strikes):
        if i == 1 and not need_second(out["strikes"][0]):
            break
        b = option_bars(cl, opt, ev["entry"], exp)
        out["strikes"].append({"option": opt, "k_real": float(k), "bars": b})
    return out


def window_events() -> pd.DataFrame:
    from hdc.data import cache_dir
    ev = pd.read_parquet(cache_dir() / "hdc" / "events.parquet")
    return ev[(ev.entry >= REAL_FROM) & (ev.earnings == "clear")].reset_index(drop=True)


def need_second_for(r: dict):
    s0, d = r["s0"] * r["real_factor"], r["div"] * r["real_factor"]

    def need(first):
        b = first["bars"]
        if r["entry"] not in b.index:
            return True                                       # first strike did not trade: try the next
        return not (float(b.loc[r["entry"], "c"]) - (s0 - first["k_real"]) < d)
    return need


def download(max_calls: int = 450, progress=print):
    """Events in a fixed random order, so a partial download is a fair random sample."""
    ev = window_events().sample(frac=1.0, random_state=7)
    cl = massive()
    progress(f"{len(ev)} events from {REAL_FROM}, {ev.ticker.nunique()} underlyings")
    for n, r in enumerate(ev.to_dict("records")):
        if cl.calls >= max_calls:
            progress(f"stopping at {cl.calls} calls; rerun to continue ({n}/{len(ev)} events)")
            return
        con = underlying_contracts(cl, r["ticker"], ev[ev.ticker == r["ticker"]])
        event_bars(cl, r, con, need_second_for(r))
        if n % 25 == 0:
            progress(f"  events {n}/{len(ev)}, calls {cl.calls}")
    progress(f"REAL OPTIONS DONE, calls {cl.calls}")


if __name__ == "__main__":
    if sys.argv[1] == "download":
        download(int(sys.argv[2]) if len(sys.argv) > 2 else 450)
