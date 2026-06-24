"""
Comprehensive referee-proofing battery for the 500-asset universe. Adds, per asset:
  - TFARCH bootstrap tempering p (primary summability; B_TF)
  - tempered-FIGARCH bootstrap tempering p (FIGARCH-short-lag-rich corroboration; B_HY)
  - Lens B: local Whittle + GPH d on |r|, break-robust d, bandwidth drift
  - Vuong SEARCH-vs-FIGARCH (V, p, sd_m, degeneracy flag)
  - lag-resolved BIC: searche/figarch BIC at J=30 vs J=252 (is FIGARCH's edge short-lag?)
Merges onto asset_results.csv -> asset_results_battery.csv.
Usage: python run_battery_500.py [n_jobs] [B_TF] [B_HY] [max_assets]
"""
import os, sys, time, zlib
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
import kernel_engine as ke
from boot_tests import bootstrap_tempering
from longmemory import lens_b

# Results dir: defaults to <repo>/results so the script runs from a clean clone;
# set SVM_RESULTS to point at an external _redo/results tree if those CSVs are not vendored.
ROOT = os.environ.get("SVM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("SVM_RESULTS", os.path.join(ROOT, "results"))
def seed_of(tk): return zlib.crc32(tk.encode()) % 100000

def per_obs_ll(fit, r, burn):
    s2 = fit['sigma2'][burn:]
    return -0.5 * (np.log(s2) + r[burn:] ** 2 / s2)

def vuong(r, fse, ffi, burn):
    m = per_obs_ll(fse, r, burn) - per_obs_ll(ffi, r, burn)
    sd = m.std(ddof=1); n = len(m)
    if sd < 1e-8:
        return dict(vuong_V=0.0, vuong_p=1.0, vuong_sd=float(sd), vuong_degenerate=True)
    V = np.sqrt(n) * m.mean() / sd
    return dict(vuong_V=float(V), vuong_p=float(2 * stats.norm.sf(abs(V))),
                vuong_sd=float(sd), vuong_degenerate=False)

def work(tk, r, B_TF, B_HY, J=252):
    rng_seed = seed_of(tk); burn = J
    out = {'Ticker': tk}
    try:
        # full fits for Vuong + lag-resolved
        fse = ke.fit_model(r, 'searche', J=J, n_restarts=3, seed=rng_seed)
        ffi = ke.fit_model(r, 'figarch', J=J, n_restarts=3, seed=rng_seed)
        if fse and ffi:
            out.update(vuong(r, fse, ffi, burn))
            out['bic_searche_J252'] = fse['bic']; out['bic_figarch_J252'] = ffi['bic']
        # lag-resolved: refit at J=30
        fse30 = ke.fit_model(r, 'searche', J=30, n_restarts=2, seed=rng_seed)
        ffi30 = ke.fit_model(r, 'figarch', J=30, n_restarts=2, seed=rng_seed)
        if fse30 and ffi30:
            out['dBIC_se_minus_fi_J252'] = (fse['bic'] - ffi['bic']) if (fse and ffi) else np.nan
            out['dBIC_se_minus_fi_J30'] = fse30['bic'] - ffi30['bic']
        # bootstrap summability tests (skipped when B<=0; analytic still computed below)
        if B_TF > 0:
            tf = bootstrap_tempering(r, family='tfarch', J=J, B=B_TF, seed=rng_seed)
            out['tf_lr'] = tf.get('lr_obs', np.nan); out['tf_p_boot'] = tf.get('p_boot', np.nan)
            out['tf_p_analytic'] = tf.get('p_analytic', np.nan); out['tf_lam'] = tf.get('lam_hat', np.nan)
            out['tf_reject_LM'] = tf.get('reject_longmem_boot', np.nan)
        if B_HY > 0:
            hy = bootstrap_tempering(r, family='tempfigarch', J=J, B=B_HY, seed=rng_seed + 7)
            out['hy_lr'] = hy.get('lr_obs', np.nan); out['hy_p_boot'] = hy.get('p_boot', np.nan)
            out['hy_lam'] = hy.get('lam_hat', np.nan); out['hy_degenerate'] = hy.get('degenerate', np.nan)
            out['hy_reject_LM'] = hy.get('reject_longmem_boot', np.nan)
        # Lens B
        lb = lens_b(r)
        out.update({f'lb_{k}': v for k, v in lb.items()})
    except Exception as e:
        out['battery_error'] = str(e)[:150]
    return out

def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    B_TF = int(sys.argv[2]) if len(sys.argv) > 2 else 199
    B_HY = int(sys.argv[3]) if len(sys.argv) > 3 else 99
    R = pd.read_csv(os.path.join(OUT, "returns_500.csv"), parse_dates=["Date"])
    S = pd.read_csv(os.path.join(OUT, "summary_500.csv"))
    tickers = S.Ticker.tolist()
    if len(sys.argv) > 4: tickers = tickers[:int(sys.argv[4])]
    series = {tk: g.sort_values("Date").r.values for tk, g in R.groupby("Ticker")}
    print(f"[battery] {len(tickers)} assets, n_jobs={n_jobs}, B_TF={B_TF}, B_HY={B_HY}", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(work)(tk, series[tk], B_TF, B_HY) for tk in tickers if tk in series)
    bat = pd.DataFrame(res)
    bat.to_csv(os.path.join(OUT, "battery_results.csv"), index=False)
    core = pd.read_csv(os.path.join(OUT, "asset_results.csv"))
    merged = core.merge(bat, on='Ticker', how='left')
    merged.to_csv(os.path.join(OUT, "asset_results_battery.csv"), index=False)
    print(f"[done] {len(bat)} assets, {time.time()-t0:.0f}s", flush=True)
    report(merged)

def report(df):
    print("\n==== BATTERY SUMMARY ====")
    if 'tf_reject_LM' in df:
        print(f"TFARCH bootstrap reject long-memory (=>summable): {df.tf_reject_LM.mean()*100:.1f}%")
    if 'hy_reject_LM' in df:
        print(f"tempFIGARCH bootstrap reject LM: {df.hy_reject_LM.mean()*100:.1f}%  "
              f"(degenerate: {df.hy_degenerate.mean()*100:.0f}%)")
    if 'lb_d_lw' in df:
        print(f"Lens B local-Whittle d on |r|: median={df.lb_d_lw.median():.3f}")
    if 'vuong_degenerate' in df:
        print(f"Vuong SEARCH-vs-FIGARCH degenerate: {df.vuong_degenerate.mean()*100:.0f}%; "
              f"|V|>1.96: {(df.vuong_p<0.05).mean()*100:.0f}%")

if __name__ == "__main__":
    main()
