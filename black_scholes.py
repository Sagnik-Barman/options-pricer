"""
black_scholes.py
-----------------
Closed-form Black-Scholes-Merton pricing and Greeks for European options
on a non-dividend- or dividend-paying underlying.

Notation:
    S  spot price
    K  strike
    T  time to expiry, in years
    r  risk-free rate (continuously compounded)
    q  continuous dividend yield (0 if none)
    sigma  volatility (annualized)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float     # per 1.00 (100%) change in vol -- divide by 100 for "per 1 vol point"
    theta: float     # per YEAR -- divide by 365 for "per calendar day"
    rho: float       # per 1.00 (100%) change in rate -- divide by 100 for "per 1% rate move"

    def __repr__(self) -> str:
        return (f"price={self.price:.4f}  delta={self.delta:.4f}  gamma={self.gamma:.5f}  "
                f"vega={self.vega/100:.4f}/vol-pt  theta={self.theta/365:.4f}/day  "
                f"rho={self.rho/100:.4f}/1%rate")


def _d1_d2(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive for Black-Scholes.")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def price(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call", q: float = 0.0) -> float:
    """Black-Scholes-Merton price of a European option."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def greeks(S: float, K: float, T: float, r: float, sigma: float,
           option_type: str = "call", q: float = 0.0) -> Greeks:
    """Full closed-form Greeks. See docstrings inline for sign conventions."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    px = price(S, K, T, r, sigma, option_type, q)

    # gamma and vega are the same formula for calls and puts
    gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf_d1 * np.sqrt(T)  # per 100% vol move

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * disc_r * norm.cdf(d2)
                 + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)  # per 100% rate move
    elif option_type == "put":
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return Greeks(price=px, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 105, 0.5, 0.04, 0.25
    for opt_type in ("call", "put"):
        g = greeks(S, K, T, r, sigma, opt_type)
        print(f"{opt_type.upper():5s} S={S} K={K} T={T} r={r} sigma={sigma}  ->  {g}")
