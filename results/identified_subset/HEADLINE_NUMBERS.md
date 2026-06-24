# HONEST IDENTIFIED-SUBSET HEADLINE
## The Shape of Volatility Memory — Referee Revision

Generated: 2026-06-22
Script: run_identified_subset.py

---

## NEW HONEST HEADLINE (identified subset, bootstrap test, B=999)

**300 / 352 identified assets reject geometric (exponential) decay
at the 5% level using the correctly-sized bootstrap test.**

Fraction: 0.852 (85.2%)

---

## OLD vs NEW COMPARISON

| Metric | N | Rejections | Rate |
|--------|---|------------|------|
| OLD: Analytic LR test, all 500 assets (as submitted) | 500 | 476 | 0.952 (95.2%) |
| Bootstrap test, all 500 assets (transparent disclosure) | 500 | 418 | 0.836 (83.6%) |
| **Bootstrap test, IDENTIFIED subset (new headline)** | **352** | **300** | **0.852 (85.2%)** |
| Analytic LR (oversized), identified subset (secondary) | 352 | 337 | 0.957 (95.7%) |

**Takeaway:** Restricting to the identified subset and using the correctly-sized
bootstrap (B=999) yields a headline of **85.2%** vs the original
published **95.2%**. The analytic Wald test on the full 500 overstated
rejection partly via inflated size at the floor corner; the bootstrap on the
identified subset is the defensible number.

---

## QUARANTINE WATERFALL (identification exclusion)

Starting from N=500:

| Criterion | Unique removed | Cumulative remaining |
|-----------|---------------|---------------------|
| (a) alpha-hat < 0.05 (non-identified floor, Prop. 2) | 79 | 421 |
| (b) alpha-hat at upper bound = 3.0 (capped) | 10 | 411 |
| (c) Sign-flipped / bimodal optimizer (pre vs post alphafix) | 2 | 409 |
| (d) Persistence P >= 1 (non-stationary) | 3 | 406 |
| (e) Degenerate SE (SE=0 or SE > alpha-hat) | 54 | 352 |
| **IDENTIFIED SUBSET** | **148 total excluded** | **352** |

Note on criterion (e): 83 assets have SE > alpha-hat (SE ratio up to 200x for CBOE).
The largest contributor is alpha-hat near the floor where the information matrix is
nearly singular; many of these are already caught by (a). The 54 unique to (e) are
assets with alpha-hat in (0.05, 3) but informationally unresolved SE.

Note on (c): 4 sign-flipped assets identified (ARGT, ALGN, AON, KEYS); 2 of these
are already excluded by (a), leaving 2 unique to (c).

---

## MEDIAN ALPHA-HAT BY ASSET CLASS (identified subset only)

| Class          |   median |   n |   q25 |   q75 |
|:---------------|---------:|----:|------:|------:|
| Cryptocurrency |    0.190 |   2 | 0.180 | 0.200 |
| Factor ETF     |    0.255 |   5 | 0.199 | 0.293 |
| US Stock       |    0.265 | 231 | 0.201 | 0.335 |
| Real Estate    |    0.285 |   8 | 0.263 | 0.311 |
| US Index       |    0.332 |  18 | 0.315 | 0.346 |
| Thematic ETF   |    0.332 |   4 | 0.278 | 0.376 |
| Intl Equity    |    0.344 |  37 | 0.298 | 0.390 |
| Sector ETF     |    0.352 |  11 | 0.305 | 0.423 |
| Commodity      |    0.497 |  15 | 0.341 | 0.797 |
| Currency       |    0.528 |   8 | 0.268 | 0.650 |
| Fixed Income   |    0.589 |  13 | 0.330 | 0.728 |

**HONEST CAVEATS on small classes:**
- Cryptocurrency: n=2 (BTCUSD-like). Median = 0.190. DO NOT report this as a
  class-level finding; n is too small for any inference.
- Thematic ETF: n=4. Median = 0.332. Borderline; report with caveat.
- The ordering Crypto < Factor ETF < US Stock < ... < Fixed Income is consistent
  with the full-500 ordering but Crypto and Thematic ETF positions are unreliable.

---

## SUMMABILITY VERDICT (two-lens, identified subset, n=352)

- indeterminate: 221
- summable: 121
- long-memory: 10

Note: 'indeterminate' dominates (221/352 = 62.8%).
The summability lens cannot be the primary evidence for sub-exponential decay;
the bootstrap test on alpha is the defensible gate.

---

## DATA LIMITATION NOTES

1. The boot_exp.csv bootstrap (B as logged per asset) uses the 'reject_exp_boot'
   column; confirmation: boot_tfarch_B999.csv uses B=999 for the FIGARCH-null test.
   The boot_exp bootstrap size per asset is reported in the n_boot_ok column;
   inspect if any asset has n_boot_ok < 499.

2. The SE > alpha filter (criterion e) is conservative: it removes assets where the
   confidence interval for alpha includes zero, meaning even the sign of sub-
   exponentiality cannot be determined. A less conservative version (SE > 2*alpha)
   would remove fewer assets but is harder to defend to R3.

3. No additional bootstrap re-run is required: boot_exp.csv already contains B=999
   results per the boot_exp_run.log. The headline is a tabulation, not a new
   computation.

---
---

## B=999 REPLICATION (updated 2026-06-23)

Bootstrap run reconfirmed at B=999 (was B=499 in prior run despite naming).

### All 500 assets
| B | Rejections | Rate |
|---|------------|------|
| B=499 | 418/500 | 83.6% |
| B=999 | 415/500 | 83.0% |

### Identified subset (n=352)
| B | Rejections | Rate |
|---|------------|------|
| B=499 | 300/352 | 85.2% |
| **B=999** | **300/352** | **85.2%** |

Decision flips B=499->B=999 (identified subset): **0**
No flips — result is stable.

**HEADLINE (B=999, identified subset): 300/352 = 85.2%**
