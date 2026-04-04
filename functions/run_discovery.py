"""
THE SHAPE OF VOLATILITY MEMORY — Production Runner (Final)
============================================================
Runs everything, saves to organized results/ folder.

Usage (36-core machine):
  python run_production_final.py shape_of_memory_100.csv
  python run_production_final.py shape_of_memory_100.csv --mc-only
  python run_production_final.py shape_of_memory_100.csv --discovery-only
  python run_production_final.py shape_of_memory_100.csv --workers 32 --mc-reps 1000

Output structure:
  results/
    discovery/
      discovery_K6.csv
      discovery_K8.csv
      discovery_K12.csv
      REPORT_DISCOVERY.txt
    montecarlo/
      montecarlo_garch.csv
      montecarlo_search.csv
      montecarlo_figarch.csv
      REPORT_MONTECARLO.txt
    figures/
      fig_alpha_by_class.pdf
      fig_cross_K_alpha.pdf
      fig_mc_garch_alpha_hist.pdf
      fig_mc_search_alpha_hist.pdf
      fig_mc_figarch_alpha_hist.pdf
"""

import sys, os, time, io, argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from scipy.stats import chi2

from core_final import (
    garch_estimate, kernel_estimate, kernel_diagnostics,
    stretched_exponential_fit, lr_test, estimate_one_asset,
    simulate_gjr_garch, simulate_search, simulate_figarch,
    KNOT_LAGS_6, KNOT_LAGS_8, KNOT_LAGS_12,
    J_MAX, REPORT_LAGS,
)

# ==============================================================
#  CONFIG
# ==============================================================
LAM_MONO     = 10.0
N_RESTARTS   = 5
MIN_OBS      = 2500
MIN_OBS_CRY  = 2000
MC_REPS      = 1000
MC_T         = 6600

# -- Monte Carlo DGP parameters --
# DGP 1: GARCH(1,1) — should recover alpha ~ 1.0
MC_GARCH = dict(omega=0.05, a=0.03, gamma=0.07, beta=0.93)

# DGP 2: SEARCH — should recover alpha ~ 0.55
MC_SEARCH = dict(omega=0.05, a=0.01, gamma=0.01, c=1.5, alpha=0.55)

# DGP 3: FIGARCH(1,d,1) — should find alpha < 1 (sub-exponential)
MC_FIGARCH = dict(omega=0.05, a=0.005, gamma=0.005, phi1=0.30, beta1=0.20, d=0.40)

OUTDIR = 'results'


# ==============================================================
#  UTILITIES
# ==============================================================

def ensure_dirs():
    for sub in ['discovery', 'montecarlo', 'figures']:
        os.makedirs(os.path.join(OUTDIR, sub), exist_ok=True)


class Report:
    def __init__(self, path):
        self.path = path
        self.buf = io.StringIO()
    def p(self, text="", end="\n"):
        print(text, end=end)
        self.buf.write(text + end)
    def save(self):
        with open(self.path, 'w') as f:
            f.write(self.buf.getvalue())
        print(f"  Saved: {self.path}")


def load_data(fp):
    df = pd.read_csv(fp)
    if 'Ticker' in df.columns and 'LogRet' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        assets = {}
        for tk in df['Ticker'].unique():
            sub = df[df['Ticker'] == tk].sort_values('Date')
            r = sub['LogRet'].values
            r = r - np.mean(r)
            cls = sub['Class'].iloc[0] if 'Class' in sub.columns else 'Unknown'
            min_obs = MIN_OBS_CRY if cls == 'Crypto' else MIN_OBS
            if len(r) >= min_obs:
                assets[tk] = {
                    'r': r, 'T': len(r),
                    'name': sub['Name'].iloc[0] if 'Name' in sub.columns else tk,
                    'class': cls,
                    'sector': sub['Sector'].iloc[0] if 'Sector' in sub.columns else '',
                }
        return assets
    else:
        date_col = [c for c in df.columns if c.lower() in ('date', 'dates')][0]
        tickers = [c for c in df.columns if c != date_col]
        assets = {}
        for tk in tickers:
            prices = df[tk].dropna().values
            if len(prices) < MIN_OBS + 1:
                continue
            r = 100 * np.diff(np.log(prices))
            r = r - np.mean(r)
            if len(r) >= MIN_OBS:
                assets[tk] = {
                    'r': r, 'T': len(r),
                    'name': tk, 'class': 'Unknown', 'sector': '',
                }
        return assets


# ==============================================================
#  PART 1: DISCOVERY
# ==============================================================

def _worker_discovery(args):
    tk, r, knot_lags, meta = args
    try:
        res = estimate_one_asset(r, knot_lags, lam_mono=LAM_MONO,
                                 n_restarts=N_RESTARTS)
        res['meta'] = meta
        return tk, res
    except Exception as e:
        return tk, None


def run_discovery_one_K(assets, knot_lags, n_workers):
    K = len(knot_lags)
    print(f"\n{'='*60}")
    print(f"  DISCOVERY: K={K}  knots={knot_lags.tolist()}")
    print(f"  {len(assets)} assets, {n_workers} workers")
    print(f"{'='*60}")

    args_list = [
        (tk, data['r'], knot_lags,
         {'name': data['name'], 'class': data['class'], 'sector': data['sector']})
        for tk, data in assets.items()
    ]

    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        out = pool.map(_worker_discovery, args_list)
    elapsed = time.time() - t0

    results = {tk: res for tk, res in out if res is not None}
    n_fail = len(assets) - len(results)
    print(f"  K={K}: {len(results)} done, {n_fail} failed, {elapsed/60:.1f} min")
    return results, elapsed


def results_to_df(results, K):
    rows = []
    for tk, res in results.items():
        d = res['lkv_diag']
        se = res['se_fit']
        lr = res['lr']
        row = {
            'Ticker': tk, 'Class': res['meta']['class'],
            'Sector': res['meta']['sector'], 'K': K, 'T': res['T'],
            'GARCH_LLF': res['garch']['llf'], 'GARCH_BIC': res['garch']['bic'],
            'GARCH_beta': res['garch']['beta'],
            'LKV_LLF': res['lkv']['llf'], 'LKV_BIC': res['lkv']['bic'],
            'LR_stat': lr['lr_stat'], 'LR_df': lr['df'],
            'LR_pval': lr['p_value'], 'LR_reject05': lr['reject_05'],
            'BIC_favors_spline': res['lkv']['bic'] < res['garch']['bic'],
            'Half_Life': d['half_life'],
            'Tenth_Life': d['tenth_life'],
            'Hundredth_Life': d['hundredth_life'],
            'SE_alpha': se['alpha'], 'SE_c': se['c'],
            'SE_R2': se['R2'], 'SE_n_lags': se['n_lags'],
            'Slope_1_5d': d['slopes'].get('short_1_5', np.nan),
            'Slope_5_21d': d['slopes'].get('weekly_5_21', np.nan),
            'Slope_21_63d': d['slopes'].get('monthly_21_63', np.nan),
            'Slope_63_252d': d['slopes'].get('quarterly_63_252', np.nan),
        }
        for j in REPORT_LAGS:
            row[f'g_{j}'] = d['kernel_at_lags'].get(j, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def print_discovery_report(rw, all_dfs):
    rw.p("=" * 70)
    rw.p("  THE SHAPE OF VOLATILITY MEMORY — DISCOVERY REPORT")
    rw.p("  core_final: ARCH(inf) GARCH, NLS SE fit, df=K-1, burn-in=2J")
    rw.p("=" * 70)

    for label, df in all_dfs.items():
        K = df['K'].iloc[0]
        n = len(df)
        rw.p(f"\n{'-'*70}")
        rw.p(f"  {label}  (df={K-1}, chi2 crit={chi2.ppf(0.95, K-1):.2f})")
        rw.p(f"{'-'*70}")

        n_rej = df['LR_reject05'].sum()
        n_bic = df['BIC_favors_spline'].sum()
        a = df['SE_alpha'].dropna()
        r2 = df['SE_R2'].dropna()
        hl = df.loc[df['Half_Life'] < J_MAX, 'Half_Life']

        rw.p(f"  LR rejects GARCH 5%:  {n_rej}/{n} ({100*n_rej/n:.0f}%)")
        rw.p(f"  BIC favors spline:    {n_bic}/{n} ({100*n_bic/n:.0f}%)")
        rw.p(f"  alpha: mean={a.mean():.3f}  med={a.median():.3f}  sd={a.std():.3f}")
        rw.p(f"     range=[{a.min():.3f}, {a.max():.3f}]")
        rw.p(f"     alpha < 1.0: {(a < 1.0).sum()}/{len(a)}")
        rw.p(f"  R2: mean={r2.mean():.3f}  med={r2.median():.3f}")
        rw.p(f"  Half-life: med={hl.median():.0f}  range=[{hl.min():.0f}, {hl.max():.0f}]")

        rw.p(f"\n  {'Class':<16s} {'N':>3s} {'a_med':>7s} {'R2_med':>7s} {'HL_med':>6s} {'Rej':>6s}")
        rw.p(f"  {'-'*48}")
        for cls in sorted(df['Class'].unique()):
            sub = df[df['Class'] == cls]
            rw.p(f"  {cls[:15]:<16s} {len(sub):>3d} "
                 f"{sub['SE_alpha'].median():>7.3f} "
                 f"{sub['SE_R2'].median():>7.3f} "
                 f"{sub.loc[sub['Half_Life']<J_MAX,'Half_Life'].median():>6.0f} "
                 f"{sub['LR_reject05'].sum():>3d}/{len(sub)}")

    # Cross-K stability
    rw.p(f"\n{'='*70}")
    rw.p(f"  alpha MEDIAN BY CLASS ACROSS K SPECIFICATIONS")
    rw.p(f"{'='*70}")
    rw.p(f"  {'Class':<16s}", end="")
    for label in all_dfs:
        rw.p(f"  {label:>8s}", end="")
    rw.p("")
    rw.p(f"  {'-'*50}")
    all_classes = sorted(list(all_dfs.values())[0]['Class'].unique())
    for cls in all_classes:
        rw.p(f"  {cls[:15]:<16s}", end="")
        for label, df in all_dfs.items():
            med = df.loc[df['Class'] == cls, 'SE_alpha'].median()
            rw.p(f"  {med:>8.3f}", end="")
        rw.p("")

    # Full asset table for K=8
    if 'K=8' in all_dfs:
        df8 = all_dfs['K=8']
        rw.p(f"\n{'='*110}")
        rw.p(f"  FULL RESULTS TABLE (K=8)")
        rw.p(f"{'='*110}")
        rw.p(f"  {'Ticker':<8s} {'Class':<14s} {'T':>5s} {'GARCH':>8s} {'LKV':>8s} "
             f"{'LR':>7s} {'p':>6s} {'HL':>4s} {'a':>6s} {'R2':>6s} {'beta':>6s}")
        rw.p(f"  {'-'*108}")
        for _, row in df8.sort_values(['Class', 'Ticker']).iterrows():
            hl_s = str(int(row['Half_Life'])) if row['Half_Life'] < J_MAX else '>252'
            rw.p(f"  {row['Ticker']:<8s} {str(row['Class'])[:13]:<14s} "
                 f"{int(row['T']):>5d} {row['GARCH_LLF']:>8.1f} {row['LKV_LLF']:>8.1f} "
                 f"{row['LR_stat']:>7.1f} {row['LR_pval']:>6.3f} {hl_s:>4s} "
                 f"{row['SE_alpha']:>6.3f} {row['SE_R2']:>6.3f} {row['GARCH_beta']:>6.3f}")


# ==============================================================
#  PART 2: MONTE CARLO
# ==============================================================

def _worker_mc(args):
    rep, seed, knot_lags, dgp_name, dgp_func, dgp_params = args
    try:
        if dgp_name == 'GARCH':
            r, _ = dgp_func(MC_T, seed=seed, **dgp_params)
        else:
            r, _, _ = dgp_func(MC_T, seed=seed, **dgp_params)
        res = estimate_one_asset(r, knot_lags, lam_mono=LAM_MONO, n_restarts=3)
        return {
            'rep': rep, 'dgp': dgp_name,
            'alpha': res['se_fit']['alpha'],
            'R2': res['se_fit']['R2'],
            'lr_stat': res['lr']['lr_stat'],
            'reject': res['lr']['reject_05'],
            'bic_spline': res['lkv']['bic'] < res['garch']['bic'],
        }
    except Exception:
        return {'rep': rep, 'dgp': dgp_name,
                'alpha': np.nan, 'R2': np.nan,
                'lr_stat': np.nan, 'reject': np.nan, 'bic_spline': np.nan}


def run_monte_carlo(n_workers, mc_reps=MC_REPS, knot_lags=KNOT_LAGS_8):
    rng = np.random.default_rng(42)
    K = len(knot_lags)
    mc_results = {}

    dgps = [
        ('GARCH',   simulate_gjr_garch, MC_GARCH,   1.0),
        ('SEARCH',  simulate_search,    MC_SEARCH,   MC_SEARCH['alpha']),
        ('FIGARCH', simulate_figarch,   MC_FIGARCH,  None),
    ]

    for dgp_name, dgp_func, dgp_params, true_alpha in dgps:
        print(f"\n{'='*60}")
        print(f"  MONTE CARLO: DGP = {dgp_name}")
        print(f"  {mc_reps} reps, T={MC_T}, K={K}")
        print(f"  Params: {dgp_params}")
        print(f"{'='*60}")

        seeds = rng.integers(0, 2**31, size=mc_reps)
        args_list = [
            (i, int(seeds[i]), knot_lags, dgp_name, dgp_func, dgp_params)
            for i in range(mc_reps)
        ]

        t0 = time.time()
        with Pool(processes=n_workers) as pool:
            out = pool.map(_worker_mc, args_list)
        elapsed = time.time() - t0

        df = pd.DataFrame(out)
        df['true_alpha'] = true_alpha
        mc_results[dgp_name] = df
        print(f"  {dgp_name}: {elapsed/60:.1f} min")

    return mc_results


def print_mc_report(rw, mc_results):
    K = len(KNOT_LAGS_8)
    rw.p("=" * 70)
    rw.p("  MONTE CARLO VALIDATION REPORT")
    rw.p(f"  K={K}, df={K-1}, T={MC_T}")
    rw.p("=" * 70)

    for dgp_name, df in mc_results.items():
        rw.p(f"\n{'-'*60}")
        rw.p(f"  DGP: {dgp_name}")
        true_a = df['true_alpha'].iloc[0]
        if pd.notna(true_a):
            rw.p(f"  True alpha = {true_a:.2f}")
        rw.p(f"{'-'*60}")

        valid = df.dropna(subset=['alpha'])
        a = valid['alpha']
        rw.p(f"  Valid: {len(valid)}/{len(df)}")
        rw.p(f"  alpha_hat: mean={a.mean():.3f}  med={a.median():.3f}  sd={a.std():.3f}")
        rw.p(f"     [5%, 95%] = [{a.quantile(0.05):.3f}, {a.quantile(0.95):.3f}]")

        rej = valid['reject'].mean()
        bic = valid['bic_spline'].mean()
        rw.p(f"  LR reject rate 5%: {100*rej:.1f}%")
        rw.p(f"  BIC favors spline: {100*bic:.1f}%")

        if dgp_name == 'GARCH':
            if a.median() > 0.85 and rej < 0.15:
                rw.p(f"\n  VALID: alpha_hat ~ 1.0, rejection ~ nominal.")
            elif a.median() > 0.75:
                rw.p(f"\n  MILD BIAS: alpha_hat median = {a.median():.3f}.")
            else:
                rw.p(f"\n  SEVERE BIAS: alpha_hat median = {a.median():.3f}.")
        elif dgp_name == 'SEARCH':
            bias = a.median() - true_a
            rmse = np.sqrt(((a - true_a)**2).mean())
            rw.p(f"  Bias: {bias:+.3f}  RMSE: {rmse:.3f}")
            rw.p(f"  Power (reject GARCH): {100*rej:.1f}%")
        elif dgp_name == 'FIGARCH':
            rw.p(f"  Power (reject GARCH): {100*rej:.1f}%")


# ==============================================================
#  PART 3: FIGURES
# ==============================================================

def make_figures(all_dfs, mc_results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'font.size': 11, 'axes.labelsize': 12,
            'figure.figsize': (8, 5), 'figure.dpi': 150,
            'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
        })
    except ImportError:
        print("  matplotlib not available — skipping figures")
        return

    figdir = os.path.join(OUTDIR, 'figures')

    # -- Fig: alpha by asset class (K=8) --
    if 'K=8' in all_dfs:
        df8 = all_dfs['K=8']
        fig, ax = plt.subplots(figsize=(10, 5))
        classes = df8.groupby('Class')['SE_alpha'].median().sort_values()
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(classes)))
        for i, (cls, med) in enumerate(classes.items()):
            sub = df8[df8['Class'] == cls]['SE_alpha']
            ax.scatter([cls]*len(sub), sub, alpha=0.5, s=30, color=colors[i])
            ax.scatter(cls, med, s=150, marker='D', color=colors[i],
                      edgecolors='black', zorder=5)
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='GARCH (alpha=1)')
        ax.set_ylabel('Stretched exponential alpha')
        ax.set_title('Shape parameter alpha by asset class (K=8, NLS)')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        fig.savefig(os.path.join(figdir, 'fig_alpha_by_class.pdf'))
        fig.savefig(os.path.join(figdir, 'fig_alpha_by_class.png'))
        plt.close(fig)
        print("  Saved: fig_alpha_by_class")

    # -- Fig: Cross-K alpha stability --
    if len(all_dfs) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        classes = sorted(list(all_dfs.values())[0]['Class'].unique())
        x = np.arange(len(classes))
        width = 0.25
        for i, (label, df) in enumerate(all_dfs.items()):
            meds = [df.loc[df['Class'] == c, 'SE_alpha'].median() for c in classes]
            ax.bar(x + i * width, meds, width, label=label, alpha=0.8)
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_ylabel('Median alpha')
        ax.set_title('alpha stability across knot specifications (NLS)')
        ax.legend()
        fig.savefig(os.path.join(figdir, 'fig_cross_K_alpha.pdf'))
        fig.savefig(os.path.join(figdir, 'fig_cross_K_alpha.png'))
        plt.close(fig)
        print("  Saved: fig_cross_K_alpha")

    # -- Fig: Monte Carlo alpha distributions --
    for dgp_name, df in mc_results.items():
        valid = df['alpha'].dropna()
        if len(valid) < 10:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(valid, bins=40, density=True, alpha=0.7, color='steelblue',
                edgecolor='white')
        true_a = df['true_alpha'].iloc[0]
        if pd.notna(true_a):
            ax.axvline(true_a, color='red', linewidth=2, linestyle='--',
                      label=f'True alpha = {true_a:.2f}')
        ax.axvline(valid.median(), color='black', linewidth=2,
                  label=f'Median alpha_hat = {valid.median():.3f}')
        ax.set_xlabel('Estimated alpha')
        ax.set_ylabel('Density')
        ax.set_title(f'Monte Carlo: DGP = {dgp_name} (N={len(valid)})')
        ax.legend()
        fname = f'fig_mc_{dgp_name.lower()}_alpha_hist'
        fig.savefig(os.path.join(figdir, fname + '.pdf'))
        fig.savefig(os.path.join(figdir, fname + '.png'))
        plt.close(fig)
        print(f"  Saved: {fname}")


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='The Shape of Volatility Memory — Full Analysis')
    parser.add_argument('datafile', nargs='?', default=None)
    parser.add_argument('--mc-only', action='store_true')
    parser.add_argument('--discovery-only', action='store_true')
    parser.add_argument('--mc-reps', type=int, default=MC_REPS)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=OUTDIR)
    args = parser.parse_args()

    mc_reps_run = args.mc_reps
    OUTDIR = args.outdir
    ensure_dirs()

    n_cores = cpu_count()
    n_workers = args.workers or max(1, n_cores - 2)

    print(f"\n{'='*60}")
    print(f"  THE SHAPE OF VOLATILITY MEMORY — PRODUCTION RUN (Final)")
    print(f"  Cores: {n_cores}, Workers: {n_workers}, MC reps: {mc_reps_run}")
    print(f"  Output: {os.path.abspath(OUTDIR)}/")
    print(f"{'='*60}")

    t_total = time.time()
    all_dfs = {}
    mc_results = {}

    # ================ DISCOVERY ================
    if not args.mc_only:
        if args.datafile is None:
            print("ERROR: provide CSV for discovery")
            print("  python run_production_final.py data.csv")
            sys.exit(1)

        assets = load_data(args.datafile)
        print(f"  Loaded {len(assets)} assets")

        for label, knots in [('K=6', KNOT_LAGS_6),
                              ('K=8', KNOT_LAGS_8),
                              ('K=12', KNOT_LAGS_12)]:
            results, elapsed = run_discovery_one_K(assets, knots, n_workers)
            K = len(knots)
            df = results_to_df(results, K)
            path = os.path.join(OUTDIR, 'discovery', f'discovery_{label.replace("=","")}.csv')
            df.to_csv(path, index=False, float_format='%.6f')
            all_dfs[label] = df

        rw = Report(os.path.join(OUTDIR, 'discovery', 'REPORT_DISCOVERY.txt'))
        print_discovery_report(rw, all_dfs)
        rw.save()

    # ================ MONTE CARLO ================
    if not args.discovery_only:
        mc_results = run_monte_carlo(n_workers, mc_reps=mc_reps_run)
        for dgp_name, df in mc_results.items():
            path = os.path.join(OUTDIR, 'montecarlo',
                               f'montecarlo_{dgp_name.lower()}.csv')
            df.to_csv(path, index=False, float_format='%.6f')

        rw_mc = Report(os.path.join(OUTDIR, 'montecarlo', 'REPORT_MONTECARLO.txt'))
        print_mc_report(rw_mc, mc_results)
        rw_mc.save()

    # ================ FIGURES ================
    if all_dfs or mc_results:
        print("\n  Generating figures...")
        make_figures(all_dfs, mc_results)

    t_total = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  COMPLETE: {t_total:.1f} minutes")
    print(f"  Results: {os.path.abspath(OUTDIR)}/")
    print(f"{'='*60}")
