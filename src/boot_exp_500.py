"""
Parametric-bootstrap calibration of the exponential-decay test (H0: SEARCH shape alpha=1 =
geometric/GARCH kernel), correctly sized under heavy tails because it resamples each asset's
OWN standardized residuals (so the bootstrap DGP inherits the true mu4). This is the
robustness check for the robust-Wald headline: under H0 the LR null law is ((mu4-1)/2)chi2(1),
not chi2(1); the bootstrap reproduces it without assuming Gaussianity.

Per asset: fit geometric + SEARCH -> LR_obs; simulate B paths under the fitted geometric null
with eps* drawn from the symmetrized standardized residuals; refit both; p_boot.
Usage: python boot_exp_500.py [n_jobs] [B] [subset_n]   (resumable; appends to boot_exp.csv)
"""
import os, sys, time, zlib
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import kernel_engine as ke

# Results dir: defaults to <repo>/results so the script runs from a clean clone;
# set SVM_RESULTS to point at an external _redo/results tree if those CSVs are not vendored.
ROOT = os.environ.get("SVM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("SVM_RESULTS", os.path.join(ROOT, "results"))

def bootstrap_exp(r, J=252, B=499, seed=0, burn=600):
    rng = np.random.default_rng(seed)
    fg = ke.fit_model(r, 'geometric', J=J, n_restarts=3, seed=seed)
    fs = ke.fit_model(r, 'searche',   J=J, n_restarts=3, seed=seed)
    if fg is None or fs is None:
        return dict(error='fit_failed')
    lr_obs = max(2.0 * (fs['llf'] - fg['llf']), 0.0)
    s2 = fg['sigma2']; z = r / np.sqrt(s2)
    z = z - z.mean(); sd = z.std()
    if not (sd > 0):
        return dict(error='degenerate_resid')
    z = z / sd
    z = np.concatenate([z, -z])                         # symmetrize (matches symmetric-eps assumption)
    g0, om, cp, cn = fg['g'], fg['omega'], fg['c_pos'], fg['c_neg']
    pg0, ps0 = fg['params'], fs['params']
    T = len(r); lr_star = []
    for b in range(B):
        eps = rng.choice(z, size=T + burn, replace=True)
        rb = ke.simulate_archinf(g0, om, cp, cn, T=T, rng=rng, burn=burn, eps=eps)
        rb = rb - rb.mean()
        frg = ke.fit_model(rb, 'geometric', J=J, n_restarts=1, seed=b, p0_override=pg0)
        frs = ke.fit_model(rb, 'searche',   J=J, n_restarts=1, seed=b, p0_override=ps0)
        if frg is None or frs is None:
            continue
        lr_star.append(max(2.0 * (frs['llf'] - frg['llf']), 0.0))
    lr_star = np.array(lr_star); n = len(lr_star)
    p_boot = (1 + np.sum(lr_star >= lr_obs)) / (n + 1) if n else np.nan
    return dict(lr_obs=lr_obs, p_boot=p_boot, n_boot_ok=n, alpha=fs['alpha'],
                lr_star_mean=float(lr_star.mean()) if n else np.nan,
                reject_exp_boot=bool(n and p_boot < 0.05))

def work(tk, r, B):
    seed = zlib.crc32(('exp' + tk).encode()) % 100000
    a = bootstrap_exp(r, J=252, B=B, seed=seed); a['Ticker'] = tk
    return a

def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 499
    subset_n = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    outfile = os.path.join(OUT, "boot_exp.csv")
    R = pd.read_csv(os.path.join(OUT, "returns_500.csv"), parse_dates=["Date"])
    S = pd.read_csv(os.path.join(OUT, "summary_500.csv"))
    tickers = S.Ticker.tolist()
    if subset_n > 0: tickers = tickers[:subset_n]
    done = set(pd.read_csv(outfile).Ticker) if os.path.exists(outfile) else set()
    todo = [t for t in tickers if t not in done]
    series = {tk: g.sort_values("Date").r.values for tk, g in R[R.Ticker.isin(todo)].groupby("Ticker")}
    print(f"[boot_exp] total={len(tickers)} done={len(done)} todo={len(todo)} B={B}", flush=True)
    t0 = time.time(); CH = n_jobs
    for i in range(0, len(todo), CH):
        chunk = todo[i:i + CH]
        res = Parallel(n_jobs=n_jobs)(delayed(work)(tk, series[tk], B) for tk in chunk)
        df = pd.DataFrame(res)
        df.to_csv(outfile, mode='a', header=not os.path.exists(outfile), index=False)
        print(f"  {min(i+CH,len(todo))}/{len(todo)}  elapsed={time.time()-t0:.0f}s", flush=True)
    d = pd.read_csv(outfile)
    if 'reject_exp_boot' in d:
        print(f"[done] reject_exp(bootstrap, =>sub-exponential): {d.reject_exp_boot.mean()*100:.1f}%  "
              f"({int(d.reject_exp_boot.sum())}/{len(d)})  {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
