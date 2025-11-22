import numpy as np
import pandas as pd


# --------------------------------------------------------
# 1. Approximate Entropy (ApEn)
# --------------------------------------------------------
def ApEn(U, m=2, r=0.2):
    print(f"Calculating ApEn for series U: {U}")
    """Approximate entropy of series U"""
    N = len(U)

    def _phi(m):
        X = np.array([U[i : i + m] for i in range(N - m + 1)])
        C = []
        for x_i in X:
            dist = np.max(np.abs(X - x_i), axis=1)
            C.append(np.mean(dist <= r))
        return np.mean(np.log(C))

    return _phi(m) - _phi(m + 1)


# --------------------------------------------------------
# 2. Find MaxApEn by sweeping r
# --------------------------------------------------------
def max_apen(U, m=2):
    print(f"Calculating max ApEn for series U: {U}")
    r_values = np.linspace(0.01, 1.0, 50)  # 50 values from 0.01 to 1.0
    apens = []

    for rv in r_values:
        r = rv * np.std(U)
        apens.append(ApEn(U, m=m, r=r))

    idx = np.argmax(apens)
    return apens[idx], r_values[idx]


# --------------------------------------------------------
# 3. Generate bootstrapped shuffled series (Monte Carlo)
# --------------------------------------------------------
def bootstrap_max_apen(U, B=100, m=2):
    print(f"Calculating bootstrap max ApEn for series U: {U}")
    results = []

    for _ in range(B):
        shuffled = np.random.permutation(U)
        max_apen_shuff, _ = max_apen(shuffled, m=m)
        results.append(max_apen_shuff)

    return np.array(results)


# --------------------------------------------------------
# 4. Compute Pincus Index
# --------------------------------------------------------
def pincus_index(U, m=2, B=100):
    print(f"Calculating Pincus Index for series U: {U}")
    # Original max entropy
    max_apen_original, r_used = max_apen(U, m=m)

    # Monte Carlo simulations
    boot = bootstrap_max_apen(U, B=B, m=m)

    median_boot = np.median(boot)
    pi = max_apen_original / median_boot

    return {
        "PI": pi,
        "MaxApEn_original": max_apen_original,
        "MaxApEn_boot_median": median_boot,
        "r_used_fraction_std": r_used,
        "boot_distribution": boot,
    }


# --------------------------------------------------------
# 5. Example usage with stock closing prices
# --------------------------------------------------------
if __name__ == "__main__":
    import yfinance as yf

    # Download NIFTY 50 data for example
    data = yf.download("^NIFTY50", start="2025-01-01", end="2025-11-01")
    prices = data["Close"].dropna()

    # Convert to log returns
    log_returns = np.log(prices / prices.shift(1)).dropna().values

    result = pincus_index(log_returns, m=2, B=50)

    print("\n----- Pincus Index Result -----")
    print(f"Pincus Index        : {result['PI']:.4f}")
    print(f"MaxApEn (original)  : {result['MaxApEn_original']:.4f}")
    print(f"MaxApEn (shuffled)  : {result['MaxApEn_boot_median']:.4f}")
    print(f"r used (in std units): {result['r_used_fraction_std']:.4f}")
