"""
Trace EVERY number in main.tex to a result file. Regenerates each table and inline figure
from the CSVs and prints them side by side with the paper's claimed value. Run after all
re-computations to confirm the manuscript matches the data.
"""
import os
import numpy as np, pandas as pd
# Results dir: defaults to <repo>/results so the script runs from a clean clone;
# set SVM_RESULTS to point at an external _redo/results tree if those CSVs are not vendored.
ROOT = os.environ.get("SVM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("SVM_RESULTS", os.path.join(ROOT, "results"))

CLS_LABEL = {  # data Class -> paper Table-1 label
 'US Stock':'US individual equities','Intl Equity':'International equity',
 'US Index':'US equity indices','Sector ETF':'Sector ETFs','Factor ETF':'Factor ETFs',
 'Thematic ETF':'Thematic ETFs','Real Estate':'Real estate (REITs)','Commodity':'Commodities',
 'Fixed Income':'Fixed income','Currency':'Currencies','Cryptocurrency':'Cryptocurrency',
 'Volatility':'Volatility (VIX, VXN)'}
ORDER = ['US Stock','Intl Equity','US Index','Sector ETF','Factor ETF','Thematic ETF',
         'Real Estate','Commodity','Fixed Income','Currency','Cryptocurrency','Volatility']

def hdr(s): print("\n"+"="*72+"\n"+s+"\n"+"="*72)

def main():
    ar = pd.read_csv(os.path.join(OUT,"asset_results.csv"))
    one = (ar.alpha+1.96*ar.alpha_se)<1

    hdr("HEADLINE (exp-decay rejection)  [paper abstract/intro/sec6]")
    print(f"sub-exponential one-sided : {one.sum()}/500 = {100*one.mean():.1f}%   [paper: 455 / 91.0%]")
    two=((ar.alpha-1).abs()/ar.alpha_se>1.96)
    print(f"two-sided reject a=1       : {two.sum()}/500 = {100*two.mean():.1f}%   [paper: 458 / 91.6%]")
    print(f"median alpha               : {ar.alpha.median():.3f}            [paper: 0.25]")
    print(f"capped @ alpha=3           : {(ar.alpha>=2.999).sum()}                [paper: ten]")
    print(f"not-sub-exp set            : {(~one).sum()}  ->  {dict(ar[~one].Class.value_counts())}")

    hdr("TABLE 1  (descriptive: N, T, sd, kurt)  [needs summary_500.csv]")
    try:
        S = pd.read_csv(os.path.join(OUT,"summary_500.csv"))
        g = S.groupby('Class').agg(N=('Ticker','size'),T=('T','median'),
                                   sd=('sd','median'),kurt=('exkurt','median'))
        for c in ORDER:
            if c in g.index:
                r=g.loc[c]
                print(f"  {CLS_LABEL[c]:24s} N={int(r['N']):3d}  T={int(r['T']):5d}  sd={r['sd']:.2f}  kurt={r['kurt']:.1f}")
        print(f"  {'All assets':24s} N={len(S):3d}  T={int(S['T'].median()):5d}  sd={S['sd'].median():.2f}  kurt={S['exkurt'].median():.1f}")
    except Exception as e:
        print("  summary_500.csv:",e)

    hdr("TABLE 1  (alpha by class)  [paper Table: Class N alpha se Rej HL]")
    print(f"  {'Class':24s}{'N':>4}{'alpha':>7}{'se':>6}{'Rej%':>6}{'HL':>5}")
    for c in ORDER:
        d=ar[ar.Class==c]
        if len(d):
            rej=100*((d.alpha+1.96*d.alpha_se)<1).mean()
            print(f"  {CLS_LABEL[c]:24s}{len(d):>4}{d.alpha.median():>7.2f}{d.alpha_se.median():>6.2f}{rej:>6.0f}{d.hl_search.median():>5.1f}")
    print(f"  {'All assets':24s}{len(ar):>4}{ar.alpha.median():>7.2f}{ar.alpha_se.median():>6.2f}"
          f"{100*one.mean():>6.0f}{ar.hl_search.median():>5.1f}")

    hdr("SEC 4  Monte-Carlo size/power  [mc_full.csv]")
    try:
        mc=pd.read_csv(os.path.join(OUT,"mc_full.csv"))
        sub=mc[(mc.dgp=='garch')&(mc.innov=='t5')]
        print(f"  garch/t5 size: raw-LR={sub.rej_lr.mean():.3f} [paper 0.29], "
              f"wald={sub.rej_wald.mean():.3f}, scaled={sub.rej_scaled.mean():.3f}")
        sub=mc[(mc.dgp=='garch')&(mc.innov=='gauss')]
        print(f"  garch/gauss : raw-LR={sub.rej_lr.mean():.3f}, wald={sub.rej_wald.mean():.3f}")
    except Exception as e: print("  mc_full.csv:",e)

    hdr("TABLE 2  MCS  [mcs_results.csv]")
    try:
        m=pd.read_csv(os.path.join(OUT,"mcs_results.csv"))
        for h in [1,5,21,63]:
            sub=m[m.h==h]; lo=sub.loc[sub.lowest_mean,'model'].iloc[0]
            mq=sub.loc[sub.model==lo,'mean_qlike'].iloc[0]
            print(f"  h={h:2d}: lowest={lo}({mq})  MCS75={list(sub[sub.in_mcs75].model)}  MCS90={list(sub[sub.in_mcs90].model)}")
    except Exception as e: print("  mcs_results.csv:",e)

    hdr("SEC 6  Vuong (SEARCH vs FIGARCH indistinguishable)  [battery_results.csv]")
    try:
        b=pd.read_csv(os.path.join(OUT,"battery_results.csv"))
        print(f"  indistinguishable (p>0.05): {(b.vuong_p>0.05).sum()}/{len(b)} = {100*(b.vuong_p>0.05).mean():.1f}%  [paper 95%]")
    except Exception as e: print("  battery_results.csv:",e)

    hdr("SEC 6  lag-63 quarterly spike  [lag63_spike.csv]")
    try:
        L=pd.read_csv(os.path.join(OUT,"lag63_spike.csv"))
        print(f"  spike count (|r| ACF, z>2): {int(L.spike.sum())}/{len(L)}  [paper: 79 of 500]")
        print(f"  in US equities: {100*L[L.Class=='US Stock'].spike.mean():.0f}%  [paper 18%]")
        print(f"  in REITs: {100*L[L.Class=='Real Estate'].spike.mean():.0f}%  [paper 33%]")
        for c in ['US Index','Factor ETF','Fixed Income']:
            print(f"  in {c}: {100*L[L.Class==c].spike.mean():.0f}%  [paper 0%]")
    except Exception as e: print("  lag63_spike.csv:",e)

    # summability (only if new verdict present)
    hdr("SEC 6  summability two-lens verdict  [summability_verdict.csv]")
    try:
        sv=pd.read_csv(os.path.join(OUT,"summability_verdict.csv"))
        B=int(sv.n_boot_ok.median())
        print(f"  bootstrap B = {B}")
        print(f"  kernel_summable (p_boot<.05): {sv.kernel_summable.sum()} ({100*sv.kernel_summable.mean():.1f}%)")
        if 'lensB_LM' in sv: print(f"  lensB_LM (spectral d>0): {int(sv.lensB_LM.sum())} ({100*sv.lensB_LM.mean():.1f}%)")
        print("  VERDICT:",dict(sv.verdict.value_counts()))
        for v in ['summable','long-memory','indeterminate']:
            print(f"    {v:14s}: {(sv.verdict==v).sum()} ({100*(sv.verdict==v).mean():.1f}%)")
    except Exception as e: print("  summability_verdict.csv:",e)

if __name__=="__main__":
    main()
