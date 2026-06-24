# Data Access

## Current paper (500-asset panel, JAE 2026)

The current paper uses a 500-asset daily return panel (`returns_500.csv`, ~120 MB)
that is too large to include in this repository.

**To reconstruct the data:**

1. Inspect `src/universe_500.csv` for the full list of tickers and asset classes.
2. Run the download and build pipeline:

```bash
python src/download_assets.py    # downloads from Yahoo Finance via yfinance
python src/build_returns.py      # assembles returns_500.csv
```

Requires Python package `yfinance`. Sample period: January 2000 to March 2026.
VIX and VXN are excluded (volatility indices, not return series).

The analysis outputs (estimated parameters, bootstrap results, summability verdicts)
are fully available in `results/` and can be inspected without the raw data.

---

## Legacy data (100-asset panel, original JBES submission -- SUPERSEDED)

The three CSV files in this folder (`shape_of_memory_100_part01.csv`,
`shape_of_memory_100_part02.csv`, `shape_of_memory_100_part03.csv`) are from the
original 100-asset pipeline. They correspond to the `functions/` legacy scripts
and the old headline numbers (93% rejection rate, JBES target). They do NOT
correspond to the current paper. Retained for provenance.
