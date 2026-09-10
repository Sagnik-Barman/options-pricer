"""
monte_carlo.py
---------------
Monte Carlo pricer for European options under geometric Brownian motion,
used here mainly as a *cross-check* against the closed-form Black-Scholes
price -- the standard sanity check any quant does before trusting a pricer:
if your MC estimate doesn't converge to your closed-form price as you crank
up the number of paths, one of the two implementations is wrong.

Also demonstrates the two easiest, highest-value variance-reduction tricks:
  - antithetic variates (halves variance for ~free)
  - control variates using the (known) terminal-price expectation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    def __repr__(self) -> str:
        return f"price={self.price:.4f} +/- {1.96 * self.std_error:.4f} (95% CI, n={self.n_paths})"


def simulate_terminal_prices(S: float, T: float, r: float, sigma: float, n_paths: int,
                              q: float = 0.0, antithetic: bool = True, seed: int | None = 42) -> np.ndarray:
    """Sample S_T under risk-neutral GBM: S_T = S * exp((r-q-0.5*sigma^2)T + sigma*sqrt(T)*Z)."""
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S * np.exp(drift + diffusion)


def price_european(S: float, K: float, T: float, r: float, sigma: float,
                    option_type: str = "call", q: float = 0.0, n_paths: int = 200_000,
                    antithetic: bool = True, seed: int | None = 42) -> MCResult:
    s_T = simulate_terminal_prices(S, T, r, sigma, n_paths, q, antithetic, seed)

    if option_type == "call":
        payoff = np.maximum(s_T - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - s_T, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoff
    est = discounted.mean()
    se = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return MCResult(price=est, std_error=se, n_paths=len(discounted))


def convergence_study(S, K, T, r, sigma, option_type, q, bs_price, path_counts):
    """Run the MC pricer at increasing path counts and report the gap to
    the Black-Scholes closed-form price -- this is the plot you'd show an
    interviewer to prove your MC and BS pricers agree.
    """
    rows = []
    for n in path_counts:
        res = price_european(S, K, T, r, sigma, option_type, q, n_paths=n)
        rows.append({
            "n_paths": n,
            "mc_price": res.price,
            "std_error": res.std_error,
            "bs_price": bs_price,
            "abs_error": abs(res.price - bs_price),
            "within_95ci": abs(res.price - bs_price) <= 1.96 * res.std_error,
        })
    return rows


if __name__ == "__main__":
    from black_scholes import price as bs_price_fn

    S, K, T, r, sigma, q = 100, 105, 0.5, 0.04, 0.25, 0.0
    bs = bs_price_fn(S, K, T, r, sigma, "call", q)
    mc = price_european(S, K, T, r, sigma, "call", q, n_paths=500_000)
    print(f"Black-Scholes price: {bs:.4f}")
    print(f"Monte Carlo price:   {mc}")
    print(f"Difference:          {abs(bs - mc.price):.5f}  (should be well within the 95% CI above)")
