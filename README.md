# GTA VI Trophy Rarity Forecast — v2 (tier-pool elasticity)

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mumun110795/gta6-trophy-rarity-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mumun110795/gta6-trophy-rarity-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, the
earn rate Exophase will show at the case's target instant — five
quantiles per trophy, in percent — from the recorded reads of the
game's Exophase trophy page before the checkpoint.

## The model

Exophase refreshes its counts in jumps, so the read history is first
collapsed to **refresh points**: the reads where the tracked player
count or any rate actually changed.

- **Players at the target** — an ensemble of a saturating-exponential
  fit (the tracked pool levels off) and a log-linear fit of
  ln(players) against ln(1 + hours), combined by geometric mean and
  never below the last observed count.
- **Earners per trophy** — projected with an **elasticity model
  estimated per tier pool** (platinum, gold, silver, bronze
  separately): between refresh points, regress Δln(earners) on
  Δln(players), recency-weighted. Each pool is shrunk toward the
  global elasticity by its sample size, and each trophy toward its
  pool. A trophy whose earners lag the pool's growth dilutes as new
  players arrive; the median rate is projected earners over projected
  players.
- **Spread** — per trophy, from the RMS residual of its logit rate
  around its own drift (pooled fallback for sparse trophies), scaled
  by √(horizon in days), floored and capped, and widened ×1.5 when the
  trophy's rate jumped sharply at the last refresh.
- **One frozen page** (the launch-night situation) falls back to
  persistence: median = last rate, spread = the house random-walk
  scale widened ×1.5 for the refresh jump that is likely still unseen.

`main.py` is Python 3 standard library only, deterministic, and runs
in well under the 8-second per-case limit.

## Run it

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` case in the input directory and writes one
`<id>.out` forecast per case to the output directory, in the exact
format the bounty scores.

## License

MIT — see `LICENSE`.

<sub>Entered on [make.fun](https://make.fun/?ref=vdsf53gd&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
