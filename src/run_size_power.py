"""
run_size_power.py  --  Bootstrap size and power of the SEARCH exponential-decay test.

H0: alpha = 1  (geometric/GARCH kernel)
H1: alpha < 1  (sub-exponential / slower decay)

Test: parametric bootstrap LR — resample standardised residuals from the fitted
geometric null DGP, refit both geometric and SEARCH, build LR* null distribution,
p-value = (1 + #{LR* >= LR_obs}) / (B + 1).

DGP: SEARCH g(j) = exp(-c(j^alpha - 1))
  c chosen to match half-life ~4 days at each alpha.
  c_pos, c_neg set so persistence P = 0.93, leverage ratio 1.6.

Sample sizes: T in {3000, 6654, 10000}
              (p25, median, p90 of identified asset T distribution)
Innovations:  t5 (primary)  +  Gaussian (reference)
True alpha:   {1.0 (SIZE), 0.9, 0.8, 0.7, 0.6, 0.5}
MC reps:      N_MC  (default 500; --nmc)
Bootstrap B:  B_boot (default 199; --B)

Speed notes:
  - Outer loop parallelized across N_MC reps (32 workers).
  - Inner bootstrap uses n_restarts=1 with warm-start from the observed-data fits.
  - n_restarts=3 for the outer (observed-data) fit only.

Usage:
  cd to output dir, then:
  "C:/Python314/python.exe" run_size_power.py [--nmc 500] [--B 199] [--njobs 32]

Outputs:
  size_power_curve.csv
  size_power.md
  progress.log
"""
import sys, os, time, argparse, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
np.seterr(all='ignore')

# ── paths ──────────────────────────────────────────────────────────────────────
# Repo root: defaults to the parent of this src/ dir so the script runs from a clean
# clone; override with SVM_ROOT to point at an external working tree. CODE_DIR holds the
# engine modules (kernel_engine); OUTDIR receives the size/power tables.
ROOT     = os.environ.get("SVM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_DIR = os.path.join(ROOT, "src")
OUTDIR   = os.path.join(ROOT, "results", "identified_subset")
os.makedirs(OUTDIR, exist_ok=True)

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import kernel_engine as ke

# ── constants ──────────────────────────────────────────────────────────────────
J           = 252
SEED_MC     = 20240622
B_BOOT_DEF  = 49
N_MC_DEF    = 100
N_JOBS_DEF  = 32

T_GRID     = [3000, 6654, 10000]
ALPHA_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
INNOV_GRID = ['t5', 'gauss']

TARGET_P = 0.93
LEV      = 1.6

LOGFILE = os.path.join(OUTDIR, "progress.log")

# ── DGP calibration ────────────────────────────────────────────────────────────
def get_c_for_halflife(alpha, target_hl=4):
    """c so g(hl) = 0.5 for SEARCH kernel."""
    if alpha >= 0.99:
        # For alpha~1: g(j)=beta^{j-1}, beta=exp(-c) -> hl-1 = ln2/c
        c = np.log(2.0) / max(target_hl - 1.0, 0.5)
    else:
        c = np.log(2.0) / max(target_hl**alpha - 1.0, 1e-4)
    return float(np.clip(c, 1e-5, 10.0))


def calib(g):
    s     = TARGET_P / max(g.sum(), 1e-6)
    c_pos = (2.0 / (1.0 + LEV)) * s
    c_neg = LEV * c_pos
    return c_pos, c_neg


def build_dgp_kernel(alpha, J=252):
    c = get_c_for_halflife(alpha)
    g = ke.kernel_searche(c, alpha, J)
    return g


# ── single MC trial ────────────────────────────────────────────────────────────
def one_trial(alpha_true, T, innov, seed, B):
    """
    One MC trial: simulate -> fit -> bootstrap -> reject or not.
    Returns dict or None on failure.
    """
    rng = np.random.default_rng(seed)

    g_true = build_dgp_kernel(alpha_true)
    c_pos, c_neg = calib(g_true)
    omega = 0.05

    df_t = 5 if innov == 't5' else None
    r = ke.simulate_archinf(g_true, omega, c_pos, c_neg, T=T, rng=rng, df=df_t)
    r = r - r.mean()

    # Outer fit (2 restarts for the observed data)
    fg = ke.fit_model(r, 'geometric', J=J, n_restarts=2, seed=seed)
    fs = ke.fit_model(r, 'searche',   J=J, n_restarts=2, seed=seed)
    if fg is None or fs is None:
        return None

    lr_obs    = max(2.0 * (fs['llf'] - fg['llf']), 0.0)
    alpha_hat = fs['alpha']

    # Standardised residuals from geometric null fit
    s2_g = fg['sigma2']
    z    = r / np.sqrt(np.maximum(s2_g, 1e-10))
    z    = z - z.mean()
    sd   = z.std()
    if not (sd > 0):
        return None
    z  /= sd
    z   = np.concatenate([z, -z])   # symmetrize

    g0  = fg['g']
    om0 = fg['omega']
    cp0 = fg['c_pos']
    cn0 = fg['c_neg']
    pg0 = fg['params']
    ps0 = fs['params']

    BURN_BOOT = 300
    rng_b     = np.random.default_rng(seed + 999999)
    lr_star   = []

    for b in range(B):
        eps_b = rng_b.choice(z, size=T + BURN_BOOT, replace=True)
        rb    = ke.simulate_archinf(g0, om0, cp0, cn0, T=T, rng=rng_b,
                                    burn=BURN_BOOT, eps=eps_b)
        rb    = rb - rb.mean()
        # warm-start restarts=1 for speed
        frg = ke.fit_model(rb, 'geometric', J=J, n_restarts=1, seed=b,
                           p0_override=pg0)
        frs = ke.fit_model(rb, 'searche',   J=J, n_restarts=1, seed=b,
                           p0_override=ps0)
        if frg is None or frs is None:
            continue
        lr_star.append(max(2.0 * (frs['llf'] - frg['llf']), 0.0))

    n_ok = len(lr_star)
    if n_ok == 0:
        return None

    lr_star = np.array(lr_star)
    p_boot  = (1 + np.sum(lr_star >= lr_obs)) / (n_ok + 1)
    reject  = int(p_boot < 0.05)

    return dict(
        alpha_true=alpha_true, T=T, innov=innov, seed=seed,
        alpha_hat=alpha_hat, lr_obs=lr_obs, p_boot=p_boot,
        reject=reject, n_boot_ok=n_ok
    )


# ── cell runner ────────────────────────────────────────────────────────────────
def run_cell(alpha_true, T, innov, N, B, n_jobs, base_seed, log_fh):
    t0    = time.time()
    seeds = [base_seed + i for i in range(N)]

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(one_trial)(alpha_true, T, innov, int(s), B)
        for s in seeds
    )
    results = [x for x in results if x is not None]
    n_ok    = len(results)

    rej_rate = np.mean([x['reject'] for x in results]) if n_ok > 0 else np.nan
    elapsed  = time.time() - t0

    msg = (f"alpha={alpha_true:.1f} T={T:6d} innov={innov:5s} | "
           f"N_ok={n_ok}/{N}  rej_rate={rej_rate:.3f}  [{elapsed:.0f}s]")
    print(msg, flush=True)
    log_fh.write(msg + "\n")
    log_fh.flush()

    return dict(
        alpha_true=alpha_true, T=T, innov=innov,
        rejection_rate=rej_rate, n_reps=n_ok,
        n_requested=N, B_boot=B, elapsed_s=elapsed
    )


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nmc',   type=int, default=N_MC_DEF)
    parser.add_argument('--B',     type=int, default=B_BOOT_DEF)
    parser.add_argument('--njobs', type=int, default=N_JOBS_DEF)
    args = parser.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    t_global = time.time()

    CSV_PATH = os.path.join(OUTDIR, "size_power_curve.csv")

    # ── RESUME: load already-completed cells ────────────────────────────────────
    done_keys = set()
    rows = []
    if os.path.exists(CSV_PATH):
        try:
            df_existing = pd.read_csv(CSV_PATH)
            for _, r in df_existing.iterrows():
                key = (float(r['alpha_true']), int(r['T']), str(r['innov']))
                done_keys.add(key)
                rows.append(r.to_dict())
            print(f"[RESUME] Found {len(done_keys)} completed cells in {CSV_PATH}",
                  flush=True)
        except Exception as e:
            print(f"[RESUME] Could not read existing CSV ({e}); starting fresh.",
                  flush=True)

    # append to log (not overwrite) so prior history is preserved
    log_mode = 'a' if os.path.exists(LOGFILE) else 'w'
    with open(LOGFILE, log_mode) as log_fh:
        log_fh.write(
            f"\nrun_size_power.py  RESUMED {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"N_MC={args.nmc}  B={args.B}  njobs={args.njobs}  seed={SEED_MC}\n"
            f"Skipping {len(done_keys)} already-done cells.\n\n"
        )
        log_fh.flush()

        cell_idx   = 0
        total_cells = len(INNOV_GRID) * len(T_GRID) * len(ALPHA_GRID)

        for innov in INNOV_GRID:
            for T in T_GRID:
                for alpha_true in ALPHA_GRID:
                    cell_idx += 1
                    key = (float(alpha_true), int(T), str(innov))

                    if key in done_keys:
                        msg = (f"[{cell_idx}/{total_cells}] SKIP (done): "
                               f"alpha={alpha_true}  T={T}  innov={innov}")
                        print(msg, flush=True)
                        log_fh.write(msg + "\n")
                        log_fh.flush()
                        continue

                    header = (f"\n[{cell_idx}/{total_cells}] "
                              f"alpha={alpha_true}  T={T}  innov={innov}")
                    print(header, flush=True)
                    log_fh.write(header + "\n")
                    log_fh.flush()

                    cell_seed = (SEED_MC
                                 + int(alpha_true * 1000)
                                 + T
                                 + (0 if innov == 't5' else 50000))

                    row = run_cell(
                        alpha_true=alpha_true, T=T, innov=innov,
                        N=args.nmc, B=args.B,
                        n_jobs=args.njobs,
                        base_seed=cell_seed,
                        log_fh=log_fh
                    )
                    rows.append(row)
                    done_keys.add(key)

                    # checkpoint: rewrite full CSV (done + new)
                    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

        total_elapsed = time.time() - t_global
        log_fh.write(f"\n[DONE]  total={total_elapsed:.0f}s\n")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved size_power_curve.csv", flush=True)

    write_md(df, args, total_elapsed)
    print(f"Total wall time: {total_elapsed/60:.1f} min")


def write_md(df, args, total_elapsed):
    lines = []
    lines.append("# Bootstrap Size and Power — SEARCH Exponential-Decay Test")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"N_MC={args.nmc}  B_boot={args.B}  n_jobs={args.njobs}  "
                 f"seed={SEED_MC}  wall={total_elapsed/60:.1f}min")
    lines.append("\n## Design")
    lines.append("- H0: alpha=1 (geometric/GARCH)  H1: alpha<1 (sub-exponential)")
    lines.append("- DGP: g(j)=exp(-c(j^alpha-1)); c for half-life~4 days; P=0.93, lev=1.6")
    lines.append("- Test: parametric bootstrap LR, 5% level; B bootstrap draws per trial")
    lines.append(f"- T in {T_GRID}")
    lines.append(f"- alpha in {ALPHA_GRID}")
    lines.append(f"- Innovations: {INNOV_GRID}")

    for innov in INNOV_GRID:
        sub = df[df['innov'] == innov].copy()
        lines.append(f"\n## Innovation: {innov}")
        T_vals = sorted(sub['T'].unique())
        header = "| alpha_true | " + " | ".join(f"T={T}" for T in T_vals) + " |"
        sep    = "| --- | " + " | ".join(["---"] * len(T_vals)) + " |"
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for a in sorted(sub['alpha_true'].unique(), reverse=True):
            row_vals = []
            for T in T_vals:
                cell = sub[(sub['alpha_true'] == a) & (sub['T'] == T)]
                if len(cell):
                    rr = cell['rejection_rate'].values[0]
                    nr = cell['n_reps'].values[0]
                    row_vals.append(f"{rr:.3f} (N={nr})")
                else:
                    row_vals.append("—")
            label = f"**{a:.1f} (SIZE)**" if a == 1.0 else f"{a:.1f}"
            lines.append(f"| {label} | " + " | ".join(row_vals) + " |")

    # Key summary
    sub_t5 = df[df['innov'] == 't5']
    size_rows = sub_t5[sub_t5['alpha_true'] == 1.0]
    pwr_rows  = sub_t5[sub_t5['alpha_true'] != 1.0]

    lines.append("\n## Key Numbers (t5 innovations)")
    lines.append("\n### SIZE (alpha=1.0)")
    for _, r in size_rows.sort_values('T').iterrows():
        flag = ""
        if not np.isnan(r.rejection_rate):
            flag = ("  [OK: near nominal]" if abs(r.rejection_rate - 0.05) < 0.025
                    else "  [DISTORTED]")
        lines.append(f"- T={int(r.T)}: size = {r.rejection_rate:.3f}  "
                     f"(N={int(r.n_reps)}){flag}")

    lines.append("\n### POWER at median T=6654")
    med_T = min(T_GRID, key=lambda x: abs(x - 6654))
    for a in sorted(pwr_rows['alpha_true'].unique(), reverse=True):
        cell = pwr_rows[(pwr_rows['alpha_true'] == a) & (pwr_rows['T'] == med_T)]
        if len(cell):
            rr = cell['rejection_rate'].values[0]
            nr = cell['n_reps'].values[0]
            lines.append(f"- alpha={a:.1f}: power = {rr:.3f}  (N={int(nr)})")

    lines.append("\n## Verdict")
    try:
        pwr_06_cell = pwr_rows[(pwr_rows['alpha_true'] == 0.6)
                               & (pwr_rows['T'] == med_T)]
        pwr_07_cell = pwr_rows[(pwr_rows['alpha_true'] == 0.7)
                               & (pwr_rows['T'] == med_T)]
        sz_cell     = size_rows[size_rows['T'] == med_T]

        pwr_06 = pwr_06_cell['rejection_rate'].values[0] if len(pwr_06_cell) else np.nan
        pwr_07 = pwr_07_cell['rejection_rate'].values[0] if len(pwr_07_cell) else np.nan
        sz_val = sz_cell['rejection_rate'].values[0]     if len(sz_cell)     else np.nan

        lines.append(f"- Bootstrap size at alpha=1, T={med_T}, t5: {sz_val:.3f}")
        lines.append(f"- Bootstrap power at alpha=0.7, T={med_T}, t5: {pwr_07:.3f}")
        lines.append(f"- Bootstrap power at alpha=0.6, T={med_T}, t5: {pwr_06:.3f}")

        if not np.isnan(pwr_06):
            if pwr_06 < 0.50:
                verdict = (
                    "UNRESOLVED (low power): at the median sample size (T=6654), "
                    "the bootstrap test has insufficient power to reliably distinguish "
                    "alpha=0.6 from alpha=1. The near-geometric non-rejections (alpha "
                    "in [0.5,1.0)) are UNINFORMATIVE — they may simply reflect the test "
                    "failing to detect genuine sub-exponential decay."
                )
            else:
                verdict = (
                    "GENUINELY NEAR-GEOMETRIC: at the median sample size (T=6654), "
                    "the bootstrap test has adequate power to detect alpha=0.6. "
                    "Non-rejections in the near-geometric region are INFORMATIVE — "
                    "those assets appear to have kernels consistent with geometric decay."
                )
            lines.append(f"\n**VERDICT:** {verdict}")
    except Exception as e:
        lines.append(f"(Verdict computation failed: {e})")

    md_path = os.path.join(OUTDIR, "size_power.md")
    with open(md_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved size_power.md", flush=True)


if __name__ == "__main__":
    main()
