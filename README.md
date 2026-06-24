# The Shape of Volatility Memory

**Replication code for Yatawara (2026)**
Submitted to the *Journal of Applied Econometrics*.

## Overview

This paper estimates the ARCH(infinity) memory kernel nonparametrically for 500
financial assets across multiple asset classes (2000-2026). The estimator (SEARCH:
Stretched-Exponential ARCH) fits a parametric stretched-exponential kernel
`g(j) = exp(-c(j^alpha - 1))` to a sieve-estimated spline kernel. The core
findings are:

- **500 assets** spanning US equities, international equities, sector ETFs, fixed
  income, commodities, currencies, and cryptocurrencies (VIX/VXN excluded).
- **352 identified assets** survive an identification quarantine that removes
  assets with non-identified alpha (floor/cap hits, bimodal optimizer, non-stationary
  fits, degenerate standard errors).
- **300 / 352 = 85.2%** of identified assets reject geometric (exponential) decay at
  the 5% level using a correctly-sized parametric bootstrap (B = 999). The bootstrap
  controls size; the analytic likelihood-ratio test is oversized and is not the
  primary inference tool.
- **Median alpha = 0.25** (identified subset), indicating sub-exponential memory
  ordering across asset classes: equity indices and US stocks cluster near 0.25-0.35;
  fixed income and currencies near 0.53-0.59.
- **Summability (identified subset, n=352):** 121 summable (34%) / 10 long-memory (3%) /
  221 indeterminate (63%) (~12:1 summable-to-long-memory ratio; true long memory is rare, ~3% of assets).
- **Near-geometric classes:** UNRESOLVED -- bootstrap power is low (0.10-0.40 for
  alpha in 0.7-0.8 at median sample size T=3000-6654) so assets near the geometric
  boundary cannot be classified.
- **No out-of-sample edge:** SEARCH does not systematically beat GARCH at any
  horizon in the 500-asset run. The in-sample kernel shape finding does not translate
  to OOS forecast gains.
- **Bootstrap size/power (T = 6654, t5 innovations):** size 3-5%; power 0.40 / 0.26
  / 0.10 at alpha = 0.6 / 0.7 / 0.8.

## Repository Structure

```
.
├── README.md
├── CITATION.cff
├── LICENSE
├── data/
│   └── (see Data section -- raw returns panel not in repo)
├── src/                              # CURRENT 500-asset pipeline
│   ├── kernel_engine.py              # ARCH(inf) kernel estimator (SEARCH + sieve)
│   ├── run_battery_500.py            # Main 500-asset estimation driver
│   ├── boot_exp_500.py               # Exponential-null bootstrap (500 assets)
│   ├── boot_b999.py                  # B=999 bootstrap driver
│   ├── boot_exp_B999_driver.py       # B=999 identified-subset bootstrap
│   ├── ident_robustness.py           # Identification quarantine criteria
│   ├── run_identified_subset.py      # Tabulates identified-subset headline numbers
│   ├── run_size_power.py             # Monte Carlo size/power of bootstrap test
│   ├── mc_validate.py                # MC validation (small pilot)
│   ├── mc_full.py                    # MC full run (N_MC=500 reps)
│   └── verify_paper_numbers.py       # Cross-checks all headline numbers vs results
├── results/
│   ├── identified_subset/
│   │   ├── identified_subset_results.csv   # Alpha, SE, boot p-value, ident flags (n=352)
│   │   ├── boot_exp_B999.csv               # B=999 bootstrap per asset (500 assets)
│   │   ├── size_power_curve.csv            # Bootstrap size/power by (alpha_true, T)
│   │   ├── disclosure_table.csv            # Identification quarantine waterfall
│   │   ├── disclosure_table.md             # Same, formatted
│   │   └── HEADLINE_NUMBERS.md             # Canonical headline numbers (source of truth)
│   ├── summability/
│   │   ├── asset_results.csv               # Full estimation output (500 assets)
│   │   └── summability_verdict.csv         # Summability lens: summable/long-mem/indet
│   └── montecarlo/
│       └── mc_full.csv                     # MC full run results (N_MC=500)
├── figures/                          # Publication figures (see figures/ note below)
└── functions/                        # LEGACY 100-ASSET PIPELINE (SUPERSEDED -- see note)
    ├── core_final.py
    ├── parametric_final.py
    ├── generate_tables.py
    ├── run_discovery.py              # LEGACY: 100-asset; does not reproduce current paper
    ├── run_horse_race.py             # LEGACY: 100-asset BIC race; not current paper
    └── run_oos.py                    # LEGACY: 100-asset OOS; current paper has no OOS edge
```

**Note on `functions/`:** These scripts are from the original 100-asset pipeline that
was submitted to JBES. They contain old numbers (93% reject rate, SEARCH wins 70/100
BIC, OOS gains) that are NOT from the current paper. They are retained for provenance
only. Each file carries a `LEGACY PIPELINE` notice at the top. Do not use them to
reproduce current results.

**Note on `figures/`:** Professional publication figures are maintained separately
by the figure-generation pipeline. If present, they include kernel-overlay plots,
alpha-by-class boxplots, and size/power curves aligned with the current paper.

## Data

The 500-asset return panel (`returns_500.csv`) was assembled from Yahoo Finance using
`src/download_assets.py` and `src/build_returns.py`. The file is approximately 120 MB
and is not included in this repository.

**To obtain the data:**
- Run `src/download_assets.py` followed by `src/build_returns.py` (requires `yfinance`).
- The universe is defined in `src/universe_500.csv` (checked in with `run_battery_500.py`).
- Sample period: 3 January 2000 through 17 June 2026. An asset is retained only if it
  has at least 1,500 trading days and trades through at least May 2026.
- VIX and VXN are excluded from all analysis (volatility indices, not return series).

The results CSVs in `results/` are the full analysis outputs and can be inspected
without the raw data.

## Installation

```bash
pip install numpy scipy pandas matplotlib statsmodels joblib yfinance
```

Python 3.10 or later required. The estimation pipeline uses `joblib` for
parallelization (up to 32 workers).

## How to Reproduce

All paths below assume you have `returns_500.csv` in a location accessible to
the scripts. By default the scripts look for it relative to the `_redo/results/`
path structure; edit the `ROOT` constant at the top of each script if your layout
differs.

### Step 1: Kernel estimation -- 500-asset battery

```bash
cd src/
python run_battery_500.py --workers 32
```

Estimates the sieve kernel and fits SEARCH for all 500 assets.
Output: `asset_results.csv` (500 rows, alpha, SE, LR stat, etc.).
Runtime: approximately 2-4 hours with 32 workers.

### Step 2: Bootstrap test (B = 999)

```bash
python boot_exp_B999_driver.py --workers 32
```

Runs the correctly-sized parametric bootstrap for the geometric-null hypothesis.
Output: `boot_exp_B999.csv` (500 rows, reject_exp_boot, p_boot, n_boot_ok).
Runtime: approximately 12-24 hours with 32 workers.

### Step 3: Identified-subset headline numbers

```bash
python run_identified_subset.py
```

Pure tabulation -- no stochastic steps. Applies the identification quarantine
(Proposition 2 criteria a-e) and computes the 85.2% headline.
Output: `identified_subset_results.csv`, `disclosure_table.csv`, `HEADLINE_NUMBERS.md`.

### Step 4: Bootstrap size and power

```bash
python run_size_power.py --nmc 500 --B 199 --njobs 32
```

Monte Carlo size and power across T in {3000, 6654, 10000} and alpha_true in
{1.0, 0.9, 0.8, 0.7, 0.6, 0.5}, t5 innovations.
Output: `size_power_curve.csv`.
Runtime: approximately 6-12 hours with 32 workers.

### Step 5: Verify all headline numbers

```bash
python verify_paper_numbers.py
```

Cross-checks all reported numbers against the results CSVs. Should produce zero
discrepancies if Steps 1-4 have run successfully.

## Key Results (Pre-computed in `results/`)

| Metric | Value |
|--------|-------|
| Total assets | 500 |
| Identified subset (after quarantine) | 352 |
| Reject geometric, bootstrap B=999 (identified) | 300 / 352 = 85.2% |
| Median alpha (identified subset) | 0.25 |
| Summable kernels (identified subset, n=352) | 121 / 352 (34%) |
| Long-memory kernels (identified subset, n=352) | 10 / 352 (3%) |
| Indeterminate (identified subset, n=352) | 221 / 352 (63%) |
| Bootstrap size (T=6654, alpha=1.0, t5) | 3.0% |
| Bootstrap power (T=6654, alpha=0.6, t5) | 39.8% |
| Bootstrap power (T=6654, alpha=0.7, t5) | 25.6% |
| OOS edge (SEARCH vs GARCH) | None |

## Identification Quarantine (Proposition 2)

Assets excluded from the 352-asset identified subset:

| Criterion | N excluded |
|-----------|-----------|
| (a) alpha < 0.05 (floor, non-identified) | 79 |
| (b) alpha at upper cap (3.0) | 10 |
| (c) Sign-flipped / bimodal optimizer | 2 (unique) |
| (d) Persistence P >= 1 (non-stationary) | 3 (unique) |
| (e) Degenerate SE (SE = 0 or SE > alpha) | 54 (unique) |
| **Total excluded** | **148** |

Full details in `results/identified_subset/disclosure_table.csv`.

## Citation

```bibtex
@article{Yatawara2026,
  author  = {Anjana Yatawara},
  title   = {The Shape of Volatility Memory},
  journal = {Journal of Applied Econometrics},
  year    = {2026},
  note    = {Submitted. Department of Mathematics and Statistics, California State University, Bakersfield}
}
```

## License

MIT License. See LICENSE file.
