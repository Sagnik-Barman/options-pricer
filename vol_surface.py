"""
vol_surface.py
---------------
Fits an implied volatility smile/surface using the SVI ("Stochastic
Volatility Inspired") raw parametrization -- the industry-standard,
arbitrage-checkable parametric form for a single-maturity smile, due to
Gatheral (2004).

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

where:
    k       = log-moneyness = ln(K / F)   (F = forward price)
    w(k)    = total implied variance = sigma_impl(k)^2 * T
    a       = overall variance level
    b       = angle/steepness of the wings (b >= 0)
    rho     = smile skew/rotation, in [-1, 1]
    m       = horizontal shift of the smile's minimum
    sigma   = curvature at the smile's minimum (sigma >= 0)

You fit these 5 params per maturity slice to observed (strike, IV) market
quotes by least squares. Stitching several maturities' fitted `a,b,rho,m,sigma`
together (or just plotting their IV curves on one 3D axis) gives the "vol
surface".

No live options-chain feed here (this environment has no internet access,
and a resume project should use real data where the argument being tested
is data-plumbing rather than the model itself) -- so this module generates
a synthetic-but-realistic "true" surface with a normal equity skew (higher
IV for downside puts, term structure flattening with maturity), *fits* SVI
back to it as if it were market data, and reports fit error. Swap
`synthetic_market_quotes()` for a real broker/options-chain CSV loader and
everything downstream is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        w = self.total_variance(k)
        return np.sqrt(np.maximum(w, 1e-10) / T)


def fit_svi(log_moneyness: np.ndarray, market_ivs: np.ndarray, T: float,
            initial_guess: SVIParams | None = None) -> tuple[SVIParams, float]:
    """Least-squares fit of SVI params to (log-moneyness, IV) market quotes
    for a single maturity slice. Returns the fitted params and the RMSE (in
    vol points) of the fit.
    """
    w_market = market_ivs ** 2 * T

    if initial_guess is None:
        initial_guess = SVIParams(a=w_market.min() * 0.8, b=0.1, rho=-0.3, m=0.0, sigma=0.1)

    x0 = np.array([initial_guess.a, initial_guess.b, initial_guess.rho,
                    initial_guess.m, initial_guess.sigma])
    lo = [-1.0, 1e-6, -0.999, -2.0, 1e-4]
    hi = [5.0, 5.0, 0.999, 2.0, 3.0]

    def residuals(x):
        p = SVIParams(*x)
        return p.total_variance(log_moneyness) - w_market

    result = least_squares(residuals, x0, bounds=(lo, hi))
    fitted = SVIParams(*result.x)
    fitted_ivs = fitted.implied_vol(log_moneyness, T)
    rmse = float(np.sqrt(np.mean((fitted_ivs - market_ivs) ** 2)))
    return fitted, rmse


def synthetic_market_quotes(forward: float, maturities: list[float],
                             strikes_per_maturity: int = 15, seed: int = 3):
    """Simulate a realistic equity-index-like vol surface: negative skew
    (downside puts trade richer than upside calls), smile flattens as
    maturity grows, with a little bit of quote noise on top -- mimicking
    real bid/ask-driven IV noise you'd see stitching an actual chain.
    """
    rng = np.random.default_rng(seed)
    surface = {}
    for T in maturities:
        # term structure: skew and curvature both decay with sqrt(T),
        # roughly matching how real equity smiles flatten at longer tenors
        true_params = SVIParams(
            a=0.02 * T,
            b=0.35 / np.sqrt(T / maturities[0]),
            rho=-0.6,
            m=-0.02,
            sigma=0.12 * np.sqrt(maturities[0] / T),
        )
        k = np.linspace(-0.5, 0.4, strikes_per_maturity)  # log-moneyness grid
        strikes = forward * np.exp(k)
        true_iv = true_params.implied_vol(k, T)
        noisy_iv = true_iv + rng.normal(0, 0.003, size=len(k))  # ~30bp quote noise
        surface[T] = {"log_moneyness": k, "strikes": strikes, "market_iv": noisy_iv,
                       "true_params": true_params}
    return surface


if __name__ == "__main__":
    forward = 100.0
    maturities = [30 / 365, 90 / 365, 180 / 365, 365 / 365]
    surface = synthetic_market_quotes(forward, maturities)

    print(f"{'T (yrs)':>8} {'a':>8} {'b':>8} {'rho':>8} {'m':>8} {'sigma':>8} {'fit RMSE (vol pts)':>20}")
    for T, quotes in surface.items():
        fitted, rmse = fit_svi(quotes["log_moneyness"], quotes["market_iv"], T)
        print(f"{T:8.3f} {fitted.a:8.4f} {fitted.b:8.4f} {fitted.rho:8.4f} "
              f"{fitted.m:8.4f} {fitted.sigma:8.4f} {rmse*100:19.3f}%")
