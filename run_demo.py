"""
run_demo.py
-----------
End-to-end demo:
  1. Price a call/put with closed-form Black-Scholes and print full Greeks.
  2. Cross-check against Monte Carlo, show convergence as path count grows.
  3. Solve implied vol from a price (round-trip sanity check).
  4. Fit an SVI smile per maturity to synthetic market quotes, plot the
     smile curves and a 3D vol surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from black_scholes import greeks, price
from implied_vol import implied_vol_newton
from monte_carlo import convergence_study, price_european
from vol_surface import fit_svi, synthetic_market_quotes

OUT_DIR = Path(__file__).parent / "output"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def part1_black_scholes():
    section("1. Black-Scholes price + Greeks")
    S, K, T, r, sigma = 100, 105, 0.5, 0.04, 0.25
    for opt_type in ("call", "put"):
        g = greeks(S, K, T, r, sigma, opt_type)
        print(f"{opt_type.upper():5s}  S={S} K={K} T={T}y r={r} sigma={sigma}")
        print(f"   {g}")
    return S, K, T, r, sigma


def part2_monte_carlo(S, K, T, r, sigma):
    section("2. Monte Carlo cross-check + convergence")
    bs_price = price(S, K, T, r, sigma, "call")
    mc = price_european(S, K, T, r, sigma, "call", n_paths=500_000)
    print(f"Black-Scholes: {bs_price:.4f}")
    print(f"Monte Carlo:   {mc}")

    path_counts = [1_000, 5_000, 25_000, 100_000, 500_000, 2_000_000]
    rows = convergence_study(S, K, T, r, sigma, "call", 0.0, bs_price, path_counts)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df["n_paths"], df["mc_price"], yerr=1.96 * df["std_error"],
                fmt="o-", capsize=3, label="Monte Carlo (95% CI)")
    ax.axhline(bs_price, color="red", ls="--", label=f"Black-Scholes = {bs_price:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths (log scale)")
    ax.set_ylabel("Option price")
    ax.set_title("Monte Carlo Convergence to Black-Scholes")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mc_convergence.png", dpi=140)
    print(f"\nSaved -> {OUT_DIR / 'mc_convergence.png'}")


def part3_implied_vol(S, K, T, r, sigma):
    section("3. Implied volatility round-trip")
    true_price = price(S, K, T, r, sigma, "call")
    recovered = implied_vol_newton(true_price, S, K, T, r, "call")
    print(f"Priced at sigma={sigma} -> price={true_price:.4f}")
    print(f"Solved implied vol from that price: {recovered:.6f}  "
          f"(error vs truth: {abs(recovered - sigma):.2e})")


def part4_vol_surface():
    section("4. SVI vol surface fit")
    forward = 100.0
    maturities = [30 / 365, 90 / 365, 180 / 365, 365 / 365]
    surface = synthetic_market_quotes(forward, maturities)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), subplot_kw={"projection": None})
    ax_smile = axes[0]
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    all_k, all_T, all_iv = [], [], []
    print(f"{'T (yrs)':>8} {'fit RMSE':>12}")
    for T, quotes in surface.items():
        fitted, rmse = fit_svi(quotes["log_moneyness"], quotes["market_iv"], T)
        print(f"{T:8.3f} {rmse*100:11.3f}%")

        k_fine = np.linspace(quotes["log_moneyness"].min(), quotes["log_moneyness"].max(), 200)
        iv_fine = fitted.implied_vol(k_fine, T)

        ax_smile.scatter(quotes["log_moneyness"], quotes["market_iv"] * 100, s=18,
                          label=f"T={T*365:.0f}d market quotes")
        ax_smile.plot(k_fine, iv_fine * 100, lw=1.2)

        all_k.extend(quotes["log_moneyness"])
        all_T.extend([T] * len(quotes["log_moneyness"]))
        all_iv.extend(quotes["market_iv"] * 100)

    ax_smile.set_xlabel("Log-moneyness  ln(K/F)")
    ax_smile.set_ylabel("Implied vol (%)")
    ax_smile.set_title("SVI Fit per Maturity")
    ax_smile.legend(fontsize=8)
    ax_smile.grid(alpha=0.3)

    ax3d.scatter(all_k, np.array(all_T) * 365, all_iv, c=all_iv, cmap="viridis", s=25)
    ax3d.set_xlabel("Log-moneyness")
    ax3d.set_ylabel("Days to expiry")
    ax3d.set_zlabel("Implied vol (%)")
    ax3d.set_title("Vol Surface")

    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUT_DIR / "vol_surface.png", dpi=140)
    print(f"\nSaved -> {OUT_DIR / 'vol_surface.png'}")


if __name__ == "__main__":
    S, K, T, r, sigma = part1_black_scholes()
    part2_monte_carlo(S, K, T, r, sigma)
    part3_implied_vol(S, K, T, r, sigma)
    part4_vol_surface()
