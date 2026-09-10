"""
implied_vol.py
---------------
Given an observed option market price, solve for the volatility that makes
Black-Scholes reproduce it -- the "implied volatility" (IV). This is the
inverse problem: price(sigma) is easy (closed form), sigma(price) has no
closed form, so we root-find.

Two solvers, both standard:
  - Newton-Raphson using vega as the derivative (fast, can fail to converge
    for deep ITM/OTM or near-expiry options where vega is tiny)
  - Brentq bisection-style solver as a robust fallback (slower, always
    converges if a sign change exists in the bracket)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from black_scholes import greeks, price


def implied_vol_newton(market_price: float, S: float, K: float, T: float, r: float,
                        option_type: str = "call", q: float = 0.0,
                        initial_guess: float = 0.3, tol: float = 1e-8, max_iter: int = 100) -> float:
    sigma = initial_guess
    for _ in range(max_iter):
        g = greeks(S, K, T, r, sigma, option_type, q)
        diff = g.price - market_price
        if abs(diff) < tol:
            return sigma
        vega = g.vega
        if vega < 1e-10:
            break  # vega too small, Newton step is unreliable -> fall back to Brent
        sigma -= diff / vega
        sigma = max(sigma, 1e-6)  # keep it in a sane, positive region
    return implied_vol_brent(market_price, S, K, T, r, option_type, q)


def implied_vol_brent(market_price: float, S: float, K: float, T: float, r: float,
                       option_type: str = "call", q: float = 0.0,
                       lo: float = 1e-4, hi: float = 5.0) -> float:
    def f(sigma):
        return price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        return brentq(f, lo, hi, xtol=1e-10, maxiter=200)
    except ValueError as e:
        raise ValueError(
            f"No implied vol found in [{lo}, {hi}] for market_price={market_price} "
            f"(check the price isn't below intrinsic value or above the max possible price)"
        ) from e


if __name__ == "__main__":
    S, K, T, r, sigma_true, q = 100, 105, 0.5, 0.04, 0.28, 0.0
    true_price = price(S, K, T, r, sigma_true, "call", q)
    print(f"Priced a call at sigma={sigma_true} -> price={true_price:.4f}")

    iv_newton = implied_vol_newton(true_price, S, K, T, r, "call", q)
    iv_brent = implied_vol_brent(true_price, S, K, T, r, "call", q)
    print(f"Recovered sigma (Newton): {iv_newton:.6f}")
    print(f"Recovered sigma (Brent):  {iv_brent:.6f}")
    print(f"Both should match {sigma_true} to ~6 decimal places.")
