"""
Referee-grade summability bootstrap: TFARCH tempering test (H0: lam=0 non-summable power-law
vs H1: lam>0 summable) at B=999 for all 500 assets. The published run used B=99, which floors
p_boot at 0.01 for 66% of assets; B=999 gives a 0.001 floor and resolves the boundary.

Resumable: appends to boot_tfarch_B999.csv after each chunk; skips tickers already done.
Usage: python boot_b999.py [n_jobs] [B]
"""
import os, sys, time, zlib, warnings
import numpy as np, pandas as pd
np.seterr(over='ignore', invalid='ignore', divide='ignore')
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from boot_tests import bootstrap_tempering

# Results dir: defaults to <repo>/results so the script runs from a clean clone;
# set SVM_RESULTS to point at an external _redo/results tree if those CSVs are not vendored.
ROOT = os.environ.get("SVM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("SVM_RESULTS", os.path.join(ROOT, "results"))

def work(tk, r, B):
    seed = zlib.crc32((tk + 'tfarch').encode()) % 100000
    a = bootstrap_tempering(r, family='tfarch', J=252, B=B, seed=seed)
    a['Ticker'] = tk
    return a

def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 999
    outfile = os.path.join(OUT, "boot_tfarch_B999.csv")
    R = pd.read_csv(os.path.join(OUT, "returns_500.csv"))
    S = pd.read_csv(os.path.join(OUT, "summary_500.csv"))
    tickers = S.Ticker.tolist()
    done = set()
    if os.path.exists(outfile):
        try: done = set(pd.read_csv(outfile).Ticker)
        except Exception: done = set()
    todo = [t for t in tickers if t not in done]
    series = {tk: g.r.values for tk, g in R[R.Ticker.isin(todo)].groupby("Ticker")}
    print(f"[boot B={B}] total={len(tickers)} done={len(done)} todo={len(todo)} n_jobs={n_jobs}", flush=True)
    t0 = time.time()
    CH = n_jobs * 2
    for i in range(0, len(todo), CH):
        chunk = todo[i:i + CH]
        res = Parallel(n_jobs=n_jobs)(delayed(work)(tk, series[tk], B) for tk in chunk)
        df = pd.DataFrame(res)
        hdr = not os.path.exists(outfile)
        df.to_csv(outfile, mode='a', header=hdr, index=False)
        el = time.time() - t0
        done_n = min(i + CH, len(todo))
        rate = el / done_n
        print(f"  {done_n}/{len(todo)}  elapsed={el:.0f}s  eta={rate*(len(todo)-done_n):.0f}s", flush=True)
    print(f"[done] {time.time()-t0:.0f}s -> {outfile}", flush=True)

if __name__ == "__main__":
    main()
