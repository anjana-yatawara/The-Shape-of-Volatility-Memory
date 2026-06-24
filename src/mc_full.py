"""
Full size/power Monte Carlo for the alpha=1 (geometric-decay) test, comparing THREE
calibrations of the same SEARCH nesting under Gaussian AND heavy-tailed (t5) innovations:
  rej_lr     : raw Gaussian-QMLE LR = 2(llf_search - llf_geom)  vs chi2(1)        [as shipped]
  rej_wald   : robust Wald  (alphahat-1)^2 / se(alphahat)^2     vs chi2(1)        [sandwich SE]
  rej_scaled : LR / ((mu4hat-1)/2)                              vs chi2(1)        [QMLE-scaled]
Point: under non-Gaussian eps the raw LR -> ((mu4-1)/2) chi2(1), so rej_lr is oversized;
the robust Wald and the mu4-scaled LR restore size. Daily |returns| are heavy-tailed
(t5 has mu4=9), so this is the empirically relevant regime.
Usage: python mc_full.py [N] [n_jobs] [T]
"""
import numpy as np, time, sys, os
from joblib import Parallel, delayed
import kernel_engine as ke

J = 252
CHI2 = 3.8414588          # chi2(1) 95th pctile
# Results dir: defaults to <repo>/results so the script runs from a clean clone;
# set SVM_RESULTS to point at an external _redo/results tree if those CSVs are not vendored.
ROOT = os.environ.get("SVM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("SVM_RESULTS", os.path.join(ROOT, "results"))

def calib(g, target_P=0.93, lev=1.6):
    s = target_P / g.sum()
    cp = (2.0 / (1.0 + lev)) * s
    return cp, lev * cp

def one(dgp, innov, seed, T):
    rng = np.random.default_rng(seed)
    if   dgp == 'garch':    g = ke.kernel_geometric(0.94, J)
    elif dgp == 'search05': g = ke.kernel_searche(0.5, 0.5, J)
    elif dgp == 'search07': g = ke.kernel_searche(0.4, 0.7, J)
    elif dgp == 'figarch':  g = ke.kernel_figarch(0.4, 0.3, 0.4, J)
    cp, cn = calib(g)
    df = 5 if innov == 't5' else None
    r = ke.simulate_archinf(g, 0.05, cp, cn, T=T, rng=rng, df=df); r = r - r.mean()
    og = ke.fit_model(r, 'geometric', J=J, n_restarts=2, seed=seed)
    os_ = ke.fit_model(r, 'searche', J=J, n_restarts=2, seed=seed)
    if og is None or os_ is None:
        return None
    lr = max(2.0 * (os_['llf'] - og['llf']), 0.0)
    p = os_['params']; alpha = os_['alpha']
    # robust Wald via sandwich SE on alpha (delta-method through softplus)
    wald2 = np.nan
    try:
        V = ke.sandwich_vcov(r, 'searche', p, J=J)
        if V is not None and np.isfinite(V[4, 4]) and V[4, 4] > 0:
            jac = 1.0 / (1.0 + np.exp(-p[4]))             # d softplus/dp = sigmoid
            se = np.sqrt(V[4, 4]) * jac
            if se > 0: wald2 = ((alpha - 1.0) / se) ** 2
    except Exception:
        pass
    # mu4 of standardized residuals under the SEARCH fit
    s2 = os_['sigma2']; z = r[J:] / np.sqrt(s2[J:]); mu4 = float(np.mean(z ** 4))
    scaled = lr / max((mu4 - 1.0) / 2.0, 1e-6)
    return dict(dgp=dgp, innov=innov, alpha=alpha, lr=lr, wald2=wald2, scaled=scaled, mu4=mu4,
                rej_lr=int(lr > CHI2),
                rej_wald=(int(wald2 > CHI2) if np.isfinite(wald2) else np.nan),
                rej_scaled=int(scaled > CHI2))

def run(dgp, innov, N, T, n_jobs):
    t0 = time.time()
    res = [x for x in Parallel(n_jobs=n_jobs)(
        delayed(one)(dgp, innov, 2000 + i, T) for i in range(N)) if x]
    a = np.array([x['alpha'] for x in res])
    def sh(k):
        v = [x[k] for x in res if isinstance(x[k], (int, float)) and np.isfinite(x[k])]
        return np.mean(v) if v else np.nan
    print(f"[{dgp:8s}/{innov:5s}] N={len(res):3d} amed={np.median(a):5.3f} "
          f"mu4={np.median([x['mu4'] for x in res]):4.1f} | "
          f"rej  LR={sh('rej_lr'):.3f}  Wald={sh('rej_wald'):.3f}  scaledLR={sh('rej_scaled'):.3f}"
          f"  | {time.time()-t0:.0f}s", flush=True)
    return res

if __name__ == "__main__":
    N  = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    nj = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    T  = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    print(f"MC size/power N={N} T={T}: raw-LR vs robust-Wald vs scaled-LR; Gaussian vs t5\n" + "=" * 78)
    print("(garch row = SIZE, target 0.05; others = POWER)")
    rows = []
    for innov in ['gauss', 't5']:
        for dgp in ['garch', 'search05', 'search07', 'figarch']:
            rows += run(dgp, innov, N, T, nj)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mc_full.csv"), index=False)
    print("DONE -> mc_full.csv")
