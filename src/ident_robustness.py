"""
Identification-robustness facts for the SEARCH shape parameter (referee-driven).
SEARCH g(j)=exp(-c(j^a-1)) interpolates geometric (a=1) and power-law (a->0, c->inf, c*a=delta
fixed, g(j)~j^{-delta}). For strongly sub-exponential assets a sits near its identified floor and
only the tail exponent delta=c*a is separately identified; the REJECTION of a=1 is nonetheless
robust. Also flags non-stationary in-sample fits (P>=1). Saves results/ident_robustness.csv.
"""
import os
import pandas as pd, numpy as np
# Results dir: defaults to <repo>/results so the script runs from a clean clone;
# set SVM_RESULTS to point at an external _redo/results tree if those CSVs are not vendored.
ROOT = os.environ.get("SVM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("SVM_RESULTS", os.path.join(ROOT, "results"))
ar = pd.read_csv(os.path.join(OUT, "asset_results.csv"))
ar['pl_exponent'] = ar.c_search * ar.alpha          # delta: g(j) ~ j^{-delta} in the a->0 limit
ar['at_floor']    = ar.alpha <= 0.0201              # at the identified lower bound (power-law limit)
ar['capped']      = ar.alpha >= 2.999
ar['nonstationary'] = ar.P_search >= 1.0
ar['identified']  = (~ar.at_floor) & (~ar.capped) & (ar.alpha_se > 1e-5) & (~ar.nonstationary)
one = (ar.alpha+1.96*ar.alpha_se) < 1

print("=== SEARCH shape identification ===")
print(f"at alpha floor (<=0.0201, power-law limit): {ar.at_floor.sum()}")
print(f"capped at 3 (super-exp, unidentified):       {ar.capped.sum()}")
print(f"non-stationary in-sample (P>=1):             {ar.nonstationary.sum()}  -> {ar[ar.nonstationary].Ticker.tolist()}")
print(f"se<1e-5 (degenerate se):                     {(ar.alpha_se<1e-5).sum()} -> {ar[ar.alpha_se<1e-5].Ticker.tolist()}")
print(f"cleanly interior-identified:                 {ar.identified.sum()}")
print(f"\ntail exponent delta=c*alpha (all assets): median={ar.pl_exponent.median():.2f}, "
      f"IQR=[{ar.pl_exponent.quantile(.25):.2f},{ar.pl_exponent.quantile(.75):.2f}]")
print(f"  delta<1 (in-sample non-summable kernel):   {(ar.pl_exponent<1).sum()}")

print("\n=== geometric-rejection robustness to dropping non-identified estimates ===")
print(f"all 500: sub-exp = {one.sum()}/500 = {100*one.mean():.1f}%")
sub = ar[ar.identified]
one_id = (sub.alpha+1.96*sub.alpha_se)<1
print(f"identified subset (N={len(sub)}): sub-exp = {one_id.sum()}/{len(sub)} = {100*one_id.mean():.1f}%")
for thr in [0.001,0.005,0.01]:
    s=ar[ar.alpha_se>thr]; print(f"  se>{thr}: {((s.alpha+1.96*s.alpha_se)<1).mean()*100:.1f}% of {len(s)}")

ar[['Ticker','Class','alpha','alpha_se','c_search','pl_exponent','P_search',
    'at_floor','capped','nonstationary','identified']].to_csv(os.path.join(OUT, "ident_robustness.csv"), index=False)
print("\n[written] ident_robustness.csv")
