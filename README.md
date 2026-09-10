# Options Pricer: Black-Scholes, Monte Carlo, Implied Vol, SVI Surface

A derivatives-pricing toolkit covering the four things a quant desk
actually does with options pricing day to day: price them closed-form,
validate that pricer against simulation, invert market prices back into
implied vol, and fit a smile/surface to a set of quotes.

## What's in here

| File | What it does |
|---|---|
| `black_scholes.py` | Closed-form BSM price + full Greeks (delta, gamma, vega, theta, rho) for calls and puts |
| `monte_carlo.py` | GBM Monte Carlo pricer with antithetic variates, used to cross-check Black-Scholes |
| `implied_vol.py` | Inverts a market price into implied vol via Newton-Raphson (fast) with a Brent fallback (robust) |
| `vol_surface.py` | Fits the SVI (Gatheral 2004) parametric smile to quotes, per maturity |
| `run_demo.py` | Runs all four in sequence, saves `output/mc_convergence.png` and `output/vol_surface.png` |

## Running it

```bash
pip install -r requirements.txt
python run_demo.py
```

No external data needed — this is pure math, fully self-contained, no
network dependency anywhere.

## The four pieces, and why each matters

**1. Black-Scholes + Greeks.** The baseline closed-form price under
constant-vol GBM. The Greeks are what a trading desk actually risk-manages
against — delta for directional exposure, gamma for how fast delta
changes, vega for vol exposure, theta for time decay, rho for rate
sensitivity. Note the units: vega is quoted here per 100% vol move (divide
by 100 for "per 1 vol point"), theta per year (divide by 365 for "per
day") — get the units right, this is a classic place candidates trip up
in interviews.

**2. Monte Carlo cross-check.** Simulates the risk-neutral terminal price
distribution directly and averages discounted payoffs. The point isn't
that MC is a better call-option pricer (it isn't — BS is exact and instant
for European vanillas); it's the *validation discipline*: before you trust
a pricer for something without a closed form (path-dependent payoffs,
American options, multi-asset baskets), you prove it agrees with the
closed-form case you *can* check. `output/mc_convergence.png` shows the MC
estimate's confidence interval visibly tightening onto the BS price as
path count grows — standard error should scale like `1/sqrt(n_paths)`.

**3. Implied volatility.** Market participants quote options in vol, not
price — IV is "the number you'd need to plug into Black-Scholes to get the
price the market is actually trading at." Solving for it means inverting a
function with no closed form, hence Newton-Raphson (using vega as the
derivative, fast when it converges) with a bisection-style Brent solver as
a fallback for the edge cases where vega is too small (deep ITM/OTM, very
short-dated) for Newton to behave.

**4. SVI vol surface.** Real markets don't trade at a single flat vol —
downside puts trade richer than upside calls (the "skew," compensation for
crash risk) and the whole smile flattens at longer maturities. SVI is the
industry-standard 5-parameter closed form for a single-maturity smile that
is easy to fit, easy to check for static arbitrage, and easy to
interpolate. `vol_surface.py` generates a synthetic-but-realistic market
(built with genuine negative skew and term-structure decay, plus quote
noise) and fits SVI back to it per maturity — the RMSE printed for each
maturity is the fit quality, in vol points.

## Using real market data instead of the synthetic surface

`vol_surface.py`'s `synthetic_market_quotes()` exists because this
environment has no internet access to pull a live options chain — swap it
for a loader that reads a CSV of `(strike, expiry, bid_iv, ask_iv)` from
your broker or a vendor (CBOE, OptionMetrics, a broker API) and everything
downstream — the SVI fit, the plots — works unchanged. That's the one
component of this repo worth upgrading to real data before you put it on
your resume; the pricing math itself doesn't need live data to prove it's
correct.

## What to be ready to explain in an interview

- **Why Black-Scholes assumes constant volatility, and why that's wrong**
  in practice (that's precisely *why* a vol surface exists — if BS's
  constant-vol assumption held, every strike/maturity would imply the same
  vol, and the surface would be flat).
- **Put-call parity** as a free sanity check: `C - P = S*e^(-qT) - K*e^(-rT)`
  should hold between your call and put prices to numerical precision —
  worth adding as an automated test if you extend this.
- **SVI's static-arbitrage conditions** — not enforced here — require
  `b*(1+|rho|) <= 4/T` (roughly, "the wings can't be too steep relative to
  maturity") to avoid butterfly arbitrage. Worth mentioning you know this
  constraint exists even if you haven't wired up the check.

## Extensions worth doing if you have more time

- American option pricing (binomial tree or Longstaff-Schwartz LSM for
  early exercise) to compare against the European price here.
- Local volatility (Dupire's formula) derived directly from the fitted SVI
  surface.
- Put-call parity as an automated consistency test across the whole strike
  grid.
- A real options-chain loader (see above) plus staleness/outlier filtering
  on the quotes before fitting.
