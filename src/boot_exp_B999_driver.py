"""
Geometric-decay (H0: alpha=1) residual-resampling bootstrap at B=999 for all 500 assets.
Outputs: boot_exp_B999.csv  (per-asset lr_obs, p_boot, n_boot_ok, alpha, reject_exp_boot)
Resumable: appends after each chunk of 32; skips tickers already in output file.
Usage: python boot_exp_B999_driver.py [n_jobs] [B]
"""
import os, sys, time, zlib, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')

# ---- paths ---------------------------------------------------------------
# Repo root: defaults to the parent of this src/ dir so the script runs from a clean
# clone; override with SVM_ROOT to point at an external working tree. CODE_DIR holds the
# engine modules (kernel_engine), DATA_DIR the intermediate CSVs (set SVM_RESULTS if those
# inputs live in an external _redo/results tree rather than the vendored results/).
ROOT       = os.environ.get("SVM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_DIR   = os.path.join(ROOT, "src")
DATA_DIR   = os.environ.get("SVM_RESULTS", os.path.join(ROOT, "results"))
OUT_DIR    = os.path.join(ROOT, "results", "identified_subset")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE   = os.path.join(OUT_DIR, "boot_exp_B999_progress.log")
OUT_FILE   = os.path.join(OUT_DIR, "boot_exp_B999.csv")

sys.path.insert(0, CODE_DIR)
import kernel_engine as ke
from joblib import Parallel, delayed


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + "\n")


def bootstrap_exp_b999(r, J=252, B=999, seed=0, burn=600):
    rng = np.random.default_rng(seed)
    fg = ke.fit_model(r, 'geometric', J=J, n_restarts=3, seed=seed)
    fs = ke.fit_model(r, 'searche',   J=J, n_restarts=3, seed=seed)
    if fg is None or fs is None:
        return dict(error='fit_failed')
    lr_obs = max(2.0 * (fs['llf'] - fg['llf']), 0.0)
    s2 = fg['sigma2']
    z  = r / np.sqrt(s2)
    z  = z - z.mean()
    sd = z.std()
    if not (sd > 0):
        return dict(error='degenerate_resid')
    z = z / sd
    z = np.concatenate([z, -z])          # symmetrize
    g0, om, cp, cn = fg['g'], fg['omega'], fg['c_pos'], fg['c_neg']
    pg0, ps0 = fg['params'], fs['params']
    T = len(r)
    lr_star = []
    for b in range(B):
        eps = rng.choice(z, size=T + burn, replace=True)
        rb  = ke.simulate_archinf(g0, om, cp, cn, T=T, rng=rng, burn=burn, eps=eps)
        rb  = rb - rb.mean()
        frg = ke.fit_model(rb, 'geometric', J=J, n_restarts=1, seed=b, p0_override=pg0)
        frs = ke.fit_model(rb, 'searche',   J=J, n_restarts=1, seed=b, p0_override=ps0)
        if frg is None or frs is None:
            continue
        lr_star.append(max(2.0 * (frs['llf'] - frg['llf']), 0.0))
    lr_star = np.array(lr_star)
    n = len(lr_star)
    p_boot = (1 + np.sum(lr_star >= lr_obs)) / (n + 1) if n else np.nan
    return dict(
        lr_obs        = float(lr_obs),
        p_boot        = float(p_boot),
        n_boot_ok     = int(n),
        alpha         = float(fs['alpha']),
        lr_star_mean  = float(lr_star.mean()) if n else float('nan'),
        reject_exp_boot = bool(n and p_boot < 0.05),
    )


def work(tk, r, B):
    seed = zlib.crc32(('exp' + tk).encode()) % 100000
    a = bootstrap_exp_b999(r, J=252, B=B, seed=seed)
    a['Ticker'] = tk
    return a


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    B      = int(sys.argv[2]) if len(sys.argv) > 2 else 999

    log(f"boot_exp B={B} n_jobs={n_jobs}  OUT={OUT_FILE}")

    R = pd.read_csv(os.path.join(DATA_DIR, "returns_500.csv"), parse_dates=["Date"])
    S = pd.read_csv(os.path.join(DATA_DIR, "summary_500.csv"))
    tickers = S.Ticker.tolist()

    done = set()
    if os.path.exists(OUT_FILE):
        try:
            done = set(pd.read_csv(OUT_FILE).Ticker)
        except Exception:
            done = set()

    todo = [t for t in tickers if t not in done]
    series = {
        tk: g.sort_values("Date").r.values
        for tk, g in R[R.Ticker.isin(todo)].groupby("Ticker")
    }
    log(f"total={len(tickers)}  done={len(done)}  todo={len(todo)}")

    t0 = time.time()
    CH = n_jobs   # chunk = one full parallel batch

    for i in range(0, len(todo), CH):
        chunk = todo[i:i + CH]
        res = Parallel(n_jobs=n_jobs)(
            delayed(work)(tk, series[tk], B) for tk in chunk
        )
        df = pd.DataFrame(res)
        hdr = not os.path.exists(OUT_FILE)
        df.to_csv(OUT_FILE, mode='a', header=hdr, index=False)
        done_n = min(i + CH, len(todo))
        el     = time.time() - t0
        rate   = el / done_n
        eta    = rate * (len(todo) - done_n)
        log(f"  chunk {done_n}/{len(todo)}  elapsed={el:.0f}s  eta={eta:.0f}s ({eta/3600:.2f}h)")

    # --- final summary -------------------------------------------------------
    df_all = pd.read_csv(OUT_FILE)
    reject_all  = int(df_all['reject_exp_boot'].sum())
    n_all       = len(df_all)
    log(f"[DONE] ALL 500: reject={reject_all}/{n_all} ({100*reject_all/n_all:.1f}%)  "
        f"wall={time.time()-t0:.0f}s")

    # identified subset summary
    ID_FILE = os.path.join(OUT_DIR, "identified_subset_results.csv")
    if os.path.exists(ID_FILE):
        ids = pd.read_csv(ID_FILE)[['Ticker','identified_flag']]
        merged = df_all.merge(ids, on='Ticker', how='left')
        sub = merged[merged['identified_flag'] == True]
        rej_sub = int(sub['reject_exp_boot'].sum())
        n_sub   = len(sub)
        log(f"[DONE] IDENTIFIED SUBSET: reject={rej_sub}/{n_sub} ({100*rej_sub/n_sub:.1f}%)")

    log(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
