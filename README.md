# The Shape of Volatility Memory

**Replication code for Yatawara (2026)**

This repository provides the complete code to replicate all empirical results in "The Shape of Volatility Memory," submitted to the *Journal of Business and Economic Statistics*.

## Overview

The paper estimates the ARCH(infinity) memory kernel nonparametrically for 100 financial assets across nine asset classes (2000-2026) and finds that 93% reject GARCH's exponential decay. The kernel is sub-exponential, well-described by a stretched exponential with shape parameter alpha that varies systematically across asset classes. A parametric model embedding this kernel (SEARCH) achieves the lowest BIC for 70 of 100 assets and produces significant out-of-sample forecasting gains at short horizons.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   └── shape_of_memory_100.csv      # User must provide (see Data section)
├── src/
│   ├── core_final.py                # Core estimation module
│   ├── parametric_final.py          # Parametric model estimators (5 models)
│   ├── run_discovery.py             # Section 5: Nonparametric discovery + Monte Carlo
│   ├── run_horse_race.py            # Section 7: In-sample BIC comparison
│   ├── run_oos.py                   # Section 8: Multi-horizon OOS forecasting
│   └── generate_tables.py           # All LaTeX tables and publication figures
└── results/
    ├── discovery/                   # Kernel estimation results (K=6, K=8, K=12)
    ├── montecarlo/                  # Monte Carlo validation (1000 reps x 3 DGPs)
    ├── horse_race/                  # Parametric model BIC comparison
    └── oos/                         # Multi-horizon OOS forecasting results
```

## Data

The analysis requires a CSV file `shape_of_memory_100.csv` with the following columns:

| Column   | Description                                      |
|----------|--------------------------------------------------|
| Date     | Trading date (YYYY-MM-DD)                        |
| Ticker   | Asset ticker (e.g., SPY, AAPL, BTC-USD)          |
| LogRet   | Daily log return in percentage (100 x log(P_t/P_{t-1})) |
| Class    | Asset class (US Stock, US Index, Sector ETF, International, Currency, Commodity, Fixed Income, Crypto, Volatility) |
| Name     | Full asset name                                  |
| Sector   | Sub-sector classification                        |

The dataset contains 100 assets:
- 53 US individual stocks
- 5 US equity indices (SPY, QQQ, DIA, IWM, MDY)
- 9 US sector ETFs (XLB through XLY)
- 10 international equity ETFs
- 6 currency ETFs
- 7 commodity ETFs
- 5 fixed income ETFs
- 4 cryptocurrencies (BTC, ETH, LTC, XRP)
- 1 volatility index (VIX)

Sample period: January 2000 to March 2026 (6,599 observations for assets with full histories).

Place the file in `data/shape_of_memory_100.csv`.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later is required. All code uses NumPy, SciPy, Pandas, and Matplotlib only.

## Replication Steps

All commands are run from the `src/` directory. Results are saved to `results/`.

### Step 1: Nonparametric Discovery and Monte Carlo (Section 5)

Estimates the ARCH(infinity) kernel via penalized spline QMLE for all 100 assets at K=6, K=8, and K=12 knot specifications, then runs Monte Carlo validation with 1000 replications under three DGPs (GARCH, SEARCH, FIGARCH).

```bash
cd src/
python run_discovery.py ../data/shape_of_memory_100.csv --workers 32
```

**Expected runtime:** 30-60 minutes with 32 cores.

**Output:**
```
results/discovery/
  discovery_K6.csv          # Full results for K=6
  discovery_K8.csv          # Full results for K=8 (primary)
  discovery_K12.csv         # Full results for K=12
  REPORT_DISCOVERY.txt      # Summary tables

results/montecarlo/
  montecarlo_garch.csv      # 1000 reps under GARCH null
  montecarlo_search.csv     # 1000 reps under SEARCH (alpha=0.55)
  montecarlo_figarch.csv    # 1000 reps under FIGARCH (d=0.40)
  REPORT_MONTECARLO.txt     # Summary statistics
```

**Key results:**
- LR rejects GARCH: 93/100 (identical at K=6, K=8, K=12)
- Median alpha: 0.272 (K=8)
- MC GARCH rejection rate: 2.2% (correct size at 5% nominal)

### Step 2: Parametric Horse Race (Section 7)

Estimates five parametric models (GARCH, SEARCH, GMARCH, FIGARCH, HYGARCH) on all 100 assets and compares by BIC.

```bash
python run_horse_race.py ../data/shape_of_memory_100.csv --workers 32
```

**Expected runtime:** 5-15 minutes with 32 cores.

**Output:**
```
results/horse_race/
  horse_race_results.csv          # BIC values for all 5 models x 100 assets
  REPORT_HORSE_RACE.txt           # Winner counts, shape parameters
  fig_bic_winner_by_class.pdf     # Figure: BIC winners by asset class
```

**Key results:**
- BIC winners: SEARCH 70, FIGARCH 17, GARCH 9, GMARCH 3, HYGARCH 1
- Median BIC improvement over GARCH: SEARCH -26.8

### Step 3: Multi-Horizon Out-of-Sample Forecasting (Section 8)

Fixed-parameter design: estimate on first 70%, forecast last 30%. Evaluates all five models at 11 horizons (h = 1, 5, 10, 22, 44, 66, 88, 110, 132, 154, 176 days).

```bash
python run_oos.py ../data/shape_of_memory_100.csv --workers 32
```

**Expected runtime:** 5-15 minutes with 32 cores.

**Output:**
```
results/oos/
  oos_multihorizon.csv                # QLIKE values for all models x horizons x assets
  REPORT_OOS_MULTIHORIZON.txt         # Winner counts, DM tests, class-level results
  fig_qlike_by_class.pdf              # Figure: QLIKE improvement by class
```

**Key results:**
- h=1: SEARCH wins 45/100, FIGARCH 24/100, GARCH 6/100
- h=10: crossover; GARCH wins 45/100
- h=176: GARCH wins 65/100
- DM tests at h=1: SEARCH significantly beats GARCH for 23 assets (3 reversals)

### Step 4: Tables and Figures (All Sections)

Generates all LaTeX tables and publication-quality figures from the results files.

```bash
python generate_tables.py
```

**Output:**
```
tables/
  tab_descriptive.tex       # Table: Descriptive statistics (Section 4)
  tab_full_K8.tex           # Table: Full discovery results (Appendix A)
  tab_kernel_values.tex     # Table: Kernel values at selected lags (Appendix B)

figures/
  fig_kernel_overlay.pdf    # Figure 1: All 100 kernels (Section 5.2)
  fig_alpha_by_class.pdf    # Figure 2: Alpha by asset class (Section 5.4)
  fig_cross_K.pdf           # Figure 3: Cross-K stability (Section 5.5)
  fig_vix_kernel.pdf        # Figure 4: VIX kernel (Section 5.7)
  fig_mc_garch.pdf          # Figure 5a: MC GARCH histogram (Section 5.8)
  fig_mc_search.pdf         # Figure 5b: MC SEARCH histogram (Section 5.8)
  fig_mc_figarch.pdf        # Figure 5c: MC FIGARCH histogram (Section 5.8)
```

Note: `generate_tables.py` requires `shape_of_memory_100.csv` for the descriptive statistics table and the discovery/montecarlo CSV files for the remaining tables and figures. Run Steps 1-3 first.

## Module Documentation

### core_final.py

Core estimation module. Contains:

| Function | Description |
|----------|-------------|
| `archinf_filter(r, omega, a, gamma, g)` | Vectorized ARCH(infinity) filter via numpy convolution |
| `gaussian_nll(r, sigma2)` | Gaussian quasi-negative-log-likelihood with 2J burn-in |
| `garch_estimate(r, n_restarts=5)` | GJR-GARCH(1,1) via ARCH(infinity) filter |
| `kernel_estimate(r, knot_lags, lam_mono=10)` | Nonparametric spline kernel estimation |
| `stretched_exponential_fit(g)` | Post-hoc NLS fit of stretched exponential |
| `lr_test(llf_spline, llf_garch, K)` | Likelihood ratio test (df = K-1) |
| `simulate_gjr_garch(T, omega, a, gamma, beta)` | Simulate from GJR-GARCH(1,1) |
| `simulate_search(T, omega, a, gamma, c, alpha)` | Simulate from SEARCH model |
| `simulate_figarch(T, omega, a, gamma, phi1, beta1, d)` | Simulate from FIGARCH(1,d,1) |

**Critical design choice:** GARCH is estimated via the ARCH(infinity) filter (not the standard GARCH recursion) so that its log-likelihood is directly comparable to the spline. The GARCH kernel is g(j) = beta^{j-1} for j = 1, ..., 252. This ensures the LR test is valid.

### parametric_final.py

Parametric model estimators. Imports `archinf_filter`, `gaussian_nll`, and `garch_estimate` from `core_final`. All models share the same filter and burn-in, ensuring BIC values are directly comparable.

| Function | Model | Kernel | Parameters |
|----------|-------|--------|------------|
| `search_estimate()` | SEARCH | exp[-c(j^alpha - 1)] | 5 (omega, a, gamma, c, alpha) |
| `gmarch_estimate()` | GMARCH | j^delta exp[-lambda(j-1)] | 5 (omega, a, gamma, delta, lambda) |
| `figarch_estimate()` | FIGARCH | BBM recursion | 6 (omega, a, gamma, d, phi, beta1) |
| `hygarch_estimate()` | HYGARCH | (1-tau)GARCH + tau FIGARCH | 7 (omega, a, gamma, d, phi, beta1, tau) |
| `estimate_all_models()` | All 5 | - | Runs GARCH + 4 alternatives |

Each estimator seeds from the GARCH optimum and uses multiple random restarts. The nesting constraint guarantees LLF(alternative) >= LLF(GARCH).

## Technical Notes

### Burn-in

All estimation uses a burn-in of 2J = 504 observations (two full years of trading days). The first J = 252 observations after the truncation lag are used to initialize the conditional variance; the next J observations are discarded to mitigate initialization effects. The effective sample size is T_eff = T - 2J.

### BIC Computation

BIC = log(T_eff) x k - 2 x LLF, where k is the number of parameters and T_eff = T - 504.

### Knot Positions

| K | Knot lags |
|---|-----------|
| 6 | 1, 5, 21, 63, 126, 252 |
| 8 | 1, 2, 5, 10, 21, 63, 126, 252 |
| 12 | 1, 2, 5, 10, 15, 21, 42, 63, 84, 126, 189, 252 |

### Monte Carlo DGP Parameters

| DGP | Parameters | Persistence |
|-----|------------|-------------|
| GARCH | omega=0.05, a=0.03, gamma=0.07, beta=0.93 | P = 0.93 |
| SEARCH | omega=0.05, a=0.006, gamma=0.008, c=1.5, alpha=0.55 | P = 0.93 |
| FIGARCH | omega=0.05, a=0.005, gamma=0.005, phi1=0.30, beta1=0.20, d=0.40 | P = 0.93 |

### Multi-Step Forecasting

The h-step-ahead forecast uses the ARCH(infinity) recursion:

E_t[sigma^2_{t+h}] = omega + sum_{j=h}^{J} g(j) w_{t+h-j} + (a + gamma/2) sum_{j=1}^{h-1} g(j) E_t[sigma^2_{t+h-j}]

The first sum uses observed news impacts; the second iterates on previous forecasts. The realized proxy is the squared return r^2_{t+h}. Loss function: QLIKE, which is robust to proxy noise (Patton, 2011).

## Citation

```bibtex
@unpublished{Yatawara2026,
  author = {Anjana Yatawara},
  title  = {The Shape of Volatility Memory},
  note   = {Department of Mathematics, California State University, Bakersfield},
  year   = {2026}
}
```

## License

MIT License. See LICENSE file.
