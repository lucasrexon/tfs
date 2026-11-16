# TFS: pre-market gapper fade

Two parts that share one set of rules:

1. **Research bot.** Finds every US stock that was up 50%+ at 09:25 ET over the last 5 years, then tests 44 short-selling rules on them after all costs.
2. **Trading bot.** Trades the ONE rule that survives the research, in the account owner's Alpaca account. It uses the same entry code, sizing code and risk rules as the backtest. Paper trading is the default.

```
python -m tfs demo        # whole pipeline on a fake market, no key needed
python -m tfs run         # real data: scan + backtest + report
python -m tfs holdout     # open the sealed test period (once)
python -m tfs card        # live: this morning's gapper cards
python -m tfs bot         # live: the trading bot
python -m pytest          # 21 tests, incl. a live-vs-backtest replay
```

## Run it on your own computer (no Claude needed)

1. **Install Python 3.10 or newer** from python.org. On Windows, tick "Add Python to PATH" in the installer.
2. **Open a terminal** (Mac: Terminal app; Windows: PowerShell) and run these one at a time:
   ```bash
   git clone https://github.com/lucasrexon/tfs
   cd tfs
   pip3 install -r requirements.txt
   python3 -m tfs demo
   ```
   The demo opens a fake-data report in your browser. If it does, everything works.
   No `git`? Download the ZIP from the GitHub page (green "Code" button) and unzip it instead.
3. **Optional head start:** unzip `tfs_head_start.zip` *inside* the `tfs` folder. It skips the
   downloads already done in the cloud session.
4. **Add your Massive key** (massive.com, Dashboard, API Keys) and start:
   - Mac: `export MASSIVE_API_KEY=your_key` then `caffeinate -i python3 -m tfs run` (caffeinate stops the Mac sleeping)
   - Windows: `$env:MASSIVE_API_KEY="your_key"` then `python -m tfs run` (set Sleep to "Never" while it runs)

**Progress:** every 25 stocks it prints a line like `[23:40] 525/2,552 (21%) about 9.8 h left`.
**Done:** it beeps, prints `DONE`, and opens `reports/report.html` in your browser.
**Interrupted?** (closed the window, computer restarted) Run the same command again. Everything
downloaded is saved in `data_cache/`, so it continues where it stopped.

## Part 1: research (5 years)

**Needs:** a Massive Stocks Starter plan ($29/month, massive.com; an adult opens it). The free plan only has 2 years of history at 5 calls/minute.

```bash
pip install -r requirements.txt
export MASSIVE_API_KEY=...        # the key from the Massive dashboard
python -m tfs run                 # first run downloads ~1,260 market days + minute bars; later runs use the cache
open reports/report.html
```

How the scan works:
- **Stage 1:** one call per trading day returns the whole market. It keeps stocks that opened at least 1.25x the prior close or traded up to 1.5x it, listed and delisted names included. OTC stocks, ETFs and warrants are excluded.
- **Stage 2:** 1-minute bars for each candidate. A stock counts as a gapper only if it was +50% at 09:25 with a price of $1 or more and 500,000+ real pre-market shares. Prices are split-adjusted, so a reverse split never looks like a gap.

The rules tested (5 entries x 5 exits x 2 stops, set in `config.yaml`):

| Entries | Exits | Stops |
|---|---|---|
| `open` 09:30 open | `eod` 15:55 | `none` |
| `first_red_5m` first red 5-min candle | `time_stop_1100` cover at 11:00 if losing | `pmh_plus_5` pre-market high + 5% |
| `vwap_loss` first 1-min close under VWAP | `next_open` | |
| `pmh_fail` failed retest of pre-market high | `close_d2`, `close_d3` | |
| `next_open` next day's open | | |

**Honesty layer:**
- Fills happen at the next bar's open, never at the signal price.
- Stops fill at the bar open if price gapped through them, plus 3% extra.
- Overnight gaps through a stop are applied.
- Costs: slippage, commission, locate fee, and borrow fee per night held.
- Short-sale restriction (SSR) penalty.
- Do Not Touch filters: 2+ halts before entry, or an IPO less than 6 months old.
- Every table is also shown with likely hard-to-borrow names removed.

### The rule for picking rules

1. Read `reports/report.html`. It only shows the build period (`split.build_end` in config.yaml).
2. Pick 1 to 3 rules that have a positive net average and a t-stat above about 2, and that hold up across years and gap sizes.
3. Write them under `holdout_rules:` in `config.yaml`.
4. Run `python -m tfs holdout` **once**. It tests those rules on the period after `build_end`, data they were never tuned on. Every run is logged, and the report warns if the test period has been opened more than once.

If nothing survives the holdout, the answer is "don't trade it". That is a real result, and it is cheaper to learn here than in an account.

## Part 2: the trading bot (for the account owner)

The bot trades **the account owner's own Alpaca account**, on their computer, under their keys. A minor cannot own the account or place the trades.

**Setup (account owner):**
1. Open an Alpaca account (alpaca.markets). Margin and shorting need a margin account; Alpaca sets the minimum deposit.
2. Create **paper** API keys first.
3. For live data, get the SIP feed (Alpaca's paid "Algo Trader Plus" plan). The free IEX feed shows about 2.5% of volume, so the 500,000-share pre-market filter would miss almost everything. With IEX, set `min_premarket_volume` much lower and expect noise.
4. Set your keys, then start the bot before 09:20 ET:
   ```bash
   export APCA_API_KEY_ID=...
   ```
   ```bash
   export APCA_API_SECRET_KEY=...
   ```
   ```bash
   python -m tfs bot
   ```

**What it does each day:** at 09:25 it scans the whole market and prints a card per gapper. From 09:30 to 11:30 it watches them. When the chosen rule fires on a bar that just finished, it:
- sizes the short (0.5% of equity at risk to the stop);
- runs the guard;
- asks the owner to type `YES` (unless `auto_approve: true`);
- sends a market short with a **broker-side stop**.

At 11:00 it applies the time stop, if the rule uses one. At 15:55 it closes every position it opened. Every scan, card, refusal, order and exit goes into `journal/live_journal.csv`, which is append-only.

**Hard rules the bot will not break:**
- Same-day rules only. It refuses to start with an overnight rule or a rule without a stop.
- Only stocks Alpaca marks shortable **and** easy-to-borrow. Most hot squeezers are hard to borrow, so they are skipped automatically.
- No adding to an existing short.
- Stops after a 2% daily loss.
- Max 3 trades a day and a 50% short-exposure cap.
- Skips stocks with 2+ halts, and stocks with a split today.
- Kill switch: create a file named `STOP` in this folder and no new orders go out.
- Live mode needs `live.mode: live` in config **and** the owner typing `I AM THE ACCOUNT OWNER` at startup.

**Suggested order:**
1. Backtest and holdout.
2. At least 30 trading days of paper trading. Compare the journal with what the backtest says those days should have done.
3. Only then, if the owner decides to go live, start small.

## Part 3: the v2 bot (the final strategy, paper first)

Research in `tfs/v2/` (no-stop paths, survival sizing, SEC, news, short data, 28 entry types, filters,
combinations) picked one strategy on the build year. Its rules are in `config.yaml` under `v2.strategy`:

- the 09:25 pre-market price is at least 50% above the previous close, price at least $1, pre-market volume at least 500,000;
- the company filed a 10-K, 10-Q or 20-F saying "substantial doubt" (a going-concern warning) in the last 456 days;
- short 2.4% of the account at 09:25 with a pre-market limit order, at most 3 shorts open at once;
- cover at 20% below the entry, or at a stop at 3x the entry, or else at 15:55 on the next trading day.

**Sealed-year result (run once, 2026-11-16): FAILED the pass rule.** Average trade +1.7% (t-stat 0.74) against
+7.6% in the build year; typical year +13%, bad luck +2%, never liquidated, worst trade 5.9% of the account.
With 1% more cost per trade than assumed, the typical year falls to +5%; with 3% more it loses 9%.
Full numbers: `reports/sealed_result.json`.

**Run it in paper mode** (same setup as Part 2), started before 09:20 ET every trading day:
```bash
python -m tfs bot2
```
Compare its closed trades with the research:
```bash
python -m tfs paper
```

**What is different from the old bot:**
- It holds shorts overnight on purpose. Live mode needs the owner to set `live.allow_overnight: true` as well as `live.mode: live` and the typed confirmation.
- The exit order (target and stop together) sits at the broker, but a broker stop only works 09:30-16:00. Outside those hours the bot watches the price itself, so it must be running.
- Open shorts are saved in `journal/v2_state.json`, so the bot can be stopped and restarted.
- Alpaca only shorts easy-to-borrow stocks. Most going-concern gappers are hard to borrow, so expect few trades.
- Not yet checked against Alpaca itself: whether Alpaca accepts a short before 09:30. Paper mode is where that shows up (a `v2_reject` or order error in the journal).

**Suggested rule for the account owner:** paper trade until at least 50 trades have closed. Go live only if the paper average trade is clearly positive after real fees.

## What is not built yet

- **Plan F** (dilution score from SEC filings) and **Plan S** (Reddit hype). Plan T comes first because nothing else can be measured without it.
- **Historical borrow availability.** There is no free source for it, so the report shows results with and without likely hard-to-borrow names.
- **Real float.** Shares outstanding stands in for float.

*Educational research software. Not financial advice. Short selling can lose more than the money in the account.*
