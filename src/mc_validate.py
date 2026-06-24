"""
Monte Carlo validation of the kernel estimator + the exponential-vs-subexponential test.

Key design: SEARCH g(j)=exp[-c(j^a-1)] nests GARCH/geometric EXACTLY at a=1, and a=1 is
INTERIOR to (0,inf). So LR_a1 = 2(llf_search - llf_geom) ~ chi^2(1) under H0 -- a regular
test, free of the boundary/approximate-nesting pathology that made the old spline-vs-GARCH
LR undersized (2.2%@5%). We verify size (GARCH DGP) and power (SEARCH/FIGARCH DGPs).
"""
import numpy as np, time, sys
from joblib import Parallel, delayed
import kernel_engine as ke

J = 252
CHI2_1_95 = 3.8414588

def calib(g, target_P=0.90, lev=1.6):
    """c_pos,c_neg with mean s s.t. P = s*sum(g) = target_P; c_neg/c_pos ~ lev/(2-lev)."""
    s = target_P / g.sum()
    c_pos = (2.0 / (1.0 + lev)) * s          # so mean(c_pos,c_neg)=s
    c_neg = lev * c_pos
    return c_pos, c_neg

def one_rep(dgp, seed, T):
    rng = np.random.default_rng(seed)
    if dgp == 'garch':
        g = ke.kernel_geometric(0.94, J)
    elif dgp == 'search05':
        g = ke.kernel_searche(c=0.5, alpha=0.5, J=J)
    elif dgp == 'search07':
        g = ke.kernel_searche(c=0.4, alpha=0.7, J=J)
    elif dgp == 'figarch':
        g = ke.kernel_figarch(d=0.4, phi=0.3, beta1=0.4, J=J)
    cp, cn = calib(g, 0.93)
    r = ke.simulate_archinf(g, 0.05, cp, cn, T=T, rng=rng); r = r - r.mean()
    og = ke.fit_model(r, 'geometric', J=J, n_restarts=2, seed=seed)
    os_ = ke.fit_model(r, 'searche', J=J, n_restarts=3, seed=seed)
    if og is None or os_ is None:
        return None
    lr = 2.0 * (os_['llf'] - og['llf'])
    return dict(dgp=dgp, alpha=os_['alpha'], lr=max(lr, 0.0),
                reject=1 if lr > CHI2_1_95 else 0, sum_g=os_['sum_g'], P=os_['P'])

def run(dgp, N=200, T=4000, n_jobs=8):
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs)(delayed(one_rep)(dgp, 1000 + i, T) for i in range(N))
    res = [x for x in res if x]
    a = np.array([x['alpha'] for x in res]); rej = np.mean([x['reject'] for x in res])
    print(f"[{dgp:9s}] N={len(res):3d} T={T} | alpha_hat med={np.median(a):.3f} "
          f"mean={a.mean():.3f} sd={a.std():.3f} | reject(a!=1)@5%={rej:.3f} "
          f"| {time.time()-t0:.0f}s", flush=True)
    return res

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(f"MC validation, N={N} reps each\n" + "="*60)
    run('garch',    N=N)   # size: reject should be ~0.05
    run('search05', N=N)   # power + alpha recovery (true 0.5)
    run('search07', N=N)   # power + alpha recovery (true 0.7)
    run('figarch',  N=N)   # power vs non-summable; alpha should be small
    print("DONE")
