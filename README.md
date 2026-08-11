# 🎰 Lottery Lab

An honest statistical sandbox for lottery numbers.

> **The premise:** nobody can predict a fair lottery draw — every combination is
> equally likely, and past draws carry *zero* information about the next one. So
> this project does the two things that are actually real instead:
>
> 1. **Proves** — empirically, on real draw history — that no "strategy" beats
>    chance. (Watching clever ideas faceplant onto the chance line is the fun part.)
> 2. **Engineers** the only genuine levers that exist: **covering designs**
>    (wheeling) that *guarantee* a small-prize match at a known cost, and an
>    **expected-value** model for avoiding jackpot splits.

This replaced an earlier set of LSTM "predictors." They couldn't work, and not for
lack of tuning — see [Why the old approach couldn't work](#why-the-old-approach-couldnt-work).

## The odds (exact)

| Game | Matrix | Match 3 main | Jackpot |
|---|---|---|---|
| Powerball (US) | 5/69 + 1/26 | **1 in 557** | 1 in 292,201,338 |
| Mega Millions (US) | 5/70 + 1/24 | 1 in 582 | 1 in 290,472,336 |
| EuroMillions (EU) | 5/50 + 2/12 | 1 in 214 | 1 in 139,838,160 |
| **EuroDreams (EU)** | 6/40 + 1/5 | **1 in 32** | 1 in 19,191,900 |

**EuroDreams is the realistic target.** Its match-3 is an order of magnitude
friendlier than Powerball's — a single ticket is expected to match 3 about every 32
draws (median wait ~3 months at two draws a week).

## Quick start

```bash
pip install -r requirements.txt        # pandas, numpy, scipy, requests, streamlit
streamlit run app/main.py              # ← the web UI (everything below, pointable & clickable)
python -m lotterylab odds              # the table above
python -m lotterylab prove eurodreams  # run every strategy vs the chance line
python -m lotterylab wheel eurodreams -n 8   # a covering design that guarantees 3-matches
python -m lotterylab wait              # how long until a ticket matches 3?
```

## The web UI

`streamlit run app/main.py` serves a clean dark-gold dashboard over the same
library the CLI uses — same numbers, same honest framing, organised the way the
project thinks:

| Section | Page | What you can do |
|---|---|---|
| Understand | **The Odds** | the exact odds per game, the premise, data status |
| Understand | **Frequency** | draw-frequency chart + live chi-square uniformity test |
| Prove | **Strategies vs Chance** | run every strategy walk-forward; see them hug the baseline (±2 SE band), per-strategy tier breakdown; synthetic-data toggle |
| Engineer | **Wheeling** | build a covering design from a spread or your own numbers; guarantee, cost, and the full ticket block |
| Engineer | **Expected Value** | build a ticket and compare its jackpot-share EV against all-birthday / all-high references |
| Feel It | **Time & Variance** | per-game wait until a 3-match; Monte-Carlo season simulator with net-result histogram |
| Admin | **Data** | snapshots on disk + one-click fetch of fresh official history |

Everything is cached, so the heavy bits (backtests, covering designs, simulations)
run once and then respond instantly.

## What's in the box

| Command | What it does |
|---|---|
| `odds` | Exact match-3 and jackpot odds for every game. |
| `prove [game]` | Runs all strategies through the backtest; shows them all hugging chance. Add `--synth` for thousands of provably-fair draws. |
| `backtest <game> <strategy>` | One strategy, walked forward over real history, scored vs the exact baseline (with a z-score). |
| `wheel <game> -n N` | Builds a covering design over N numbers and reports the **guarantee**, ticket count, and honest cost. |
| `ev <game>` | Compares a "birthday" ticket vs a high-number ticket by expected payout (same win odds, different jackpot share). |
| `freq <game>` | Frequency dashboard + a chi-square uniformity test — labelled *zero predictive power*. |
| `wait` | Expected / median time until a single ticket matches 3, per game. |
| `variance <game>` | Monte-Carlo the net-result distribution of a one-ticket-per-draw player. |

### The three real ideas

- **You can't predict the draw.** The `prove` command demonstrates it: `random`,
  `hot`, `cold`, `last_echo`, and `order_stat_mean` (the constant vector the old
  LSTM converged to) all land within noise of the hypergeometric baseline.
- **Wheeling guarantees coverage, not profit.** A covering design `C(K, k, 3)`
  guarantees a 3-match *if* ≥3 of your K chosen numbers are drawn. It never changes
  the draw odds or per-ticket EV — it buys determinism at a fixed, honest cost
  (which exceeds what a 3-match pays). It's the only thing that literally makes
  "consistently match 3" true.
- **Unpopular numbers raise expected payout, not win probability.** Jackpots are
  split among matching tickets; avoiding birthdays (≤31) and patterns means sharing
  with fewer people *if* you win. Real, but only matters for pari-mutuel/jackpot
  tiers — not fixed lower tiers. (Ziemba; Henze & Riedwyl.)

## Why the old approach couldn't work

The original scripts fed a sliding window of past draws into an LSTM and regressed
the next draw with mean-squared error. Three nested reasons it was doomed:

1. **No information.** Fair draws are i.i.d.; the mutual information between history
   and the next draw is exactly zero. No architecture can extract a signal that
   isn't there.
2. **MSE → the mean.** The error-minimizing output is the *conditional mean*, which
   under independence is just the *unconditional mean* — a fixed vector,
   independent of whatever recent draws you feed in.
3. **Sorting made it look plausible.** Because draws are stored ascending, that mean
   is the vector of **order-statistic means** (Powerball → `[12, 23, 35, 47, 58]`).
   The model was learning the average *sorted* ticket — an artifact of storage
   order — and printing it back every run. Try `backtest powerball order_stat_mean`:
   it's the old model, and it's no better than `random`.

(It was also training EuroMillions on a single example, scaling the Powerball ball
against the white-ball range, and emitting tickets with duplicate balls — but those
are bugs on top of a foundation that couldn't work regardless.)

## Layout

```
lotterylab/
  games.py          GameSpec registry — pools, prices, prize tables (one source of truth)
  combinatorics.py  exact hypergeometric odds + order-statistic means
  validate.py       ticket validity (range / uniqueness / count) — enforced everywhere
  schema.py         canonical Draw + tidy DataFrame
  adapters.py       per-game CSV -> Draw
  store.py          immutable raw snapshots; current-matrix filtering
  strategy.py       random / hot / cold / last_echo / order_stat_mean / biased_high
  baseline.py       the exact chance line
  backtest.py       walk-forward harness with z-vs-baseline  ← the heart
  wheeling.py       covering designs + brute-forced guarantee
  ev.py             jackpot-share-adjusted expected value
  analytics.py      frequency dashboard + chi-square uniformity
  simulate.py       variance simulator + time-to-3-match
  synth.py          provably-fair synthetic draws (offline tests / demos)
  cli.py            python -m lotterylab <command>
app/
  main.py           streamlit run app/main.py — the web UI entry point
  shared.py         cached data access + formatting shared by every page
  views/            one file per page (overview, prove, wheel, ev, …)
data/raw/<game>/    immutable source snapshots (never overwritten)
```

## Data

Source CSVs live under `data/raw/<game>/` as immutable, timestamped snapshots and
are never overwritten (the old scripts clobbered them in place). The loader applies
a project-wide **2018-to-present floor** (`store.MIN_DATE`) combined with each game's
matrix-change date — whichever is later — so odds and backtests are always computed
against one consistent set of rules (e.g. Powerball's pre-2018 draws and any
old-matrix Mega Millions rows are dropped automatically). It tries snapshots
newest-first and skips any that fail to parse, so one bad download never breaks the
pipeline.

To refresh (writes a NEW snapshot, never clobbers; rejects a download that doesn't
parse to any draws):

```python
from lotterylab.store import fetch_raw
fetch_raw("powerball")      # data.ny.gov
fetch_raw("megamillions")  # data.ny.gov
fetch_raw("euromillions")  # FDJ — merges multiple era-files into one snapshot
fetch_raw("eurodreams")    # FDJ — single live file
```

**Current data sources** — all four auto-fetch and run through June 2026:

| Game | Source | Auto-fetch | Latest |
|---|---|---|---|
| Powerball | NY Open Data (`data.ny.gov`) | ✅ `fetch_raw` | through 2026-06 (1,128 draws) |
| Mega Millions | NY Open Data (`data.ny.gov`) | ✅ `fetch_raw` | current matrix since 2025-04 (123 draws) |
| EuroMillions | FDJ official history (draw-info API) | ✅ `fetch_raw` | through 2026-06 (881 draws) |
| EuroDreams | FDJ official history (draw-info API) | ✅ `fetch_raw` | through 2026-06 (272 draws) |

The two European games come from **FDJ** (the French operator). FDJ publishes history
in semicolon-delimited, era-split files (the EuroMillions star pool changed over the
years, so it's chunked; EuroDreams is one file). `fetch_raw` downloads the relevant
files from FDJ's draw-info API, parses the French format, de-dupes by draw date, and
writes one combined snapshot in the adapter's layout. (The UK national-lottery CSV
endpoint was retired — it now serves only the latest draw as XML.)

Mega Millions changed its complete matrix on 2025-04-08: the Mega Ball pool fell
from 25 to 24 and the $5 ticket gained an embedded random multiplier. Its canonical
history therefore starts on that date; mixing older 1/25 special-ball draws into a
1/24 probability baseline would bias the analysis. Approximate non-jackpot payouts
use the multiplier's 3× expected value.

---

*Lottery Lab is for fun and learning. It does not improve anyone's odds of winning,
because nothing can. Please gamble responsibly — the expected value of every ticket
here is well under what you pay for it.*
