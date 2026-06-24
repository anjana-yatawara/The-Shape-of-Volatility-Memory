"""
LEGACY PIPELINE — 100-ASSET JBES SUBMISSION (SUPERSEDED)
=========================================================
THIS FILE IS FROM THE ORIGINAL 100-ASSET PIPELINE.
Results here (SEARCH wins 70/100 by BIC, median BIC improvement -26.8) are
from the old 100-asset run and are NOT reported in the current paper.

The current paper (JAE target, 2026) does not claim in-sample BIC advantage as a
primary finding. See src/run_battery_500.py and src/verify_paper_numbers.py for the
current pipeline.
=========================================================

THE SHAPE OF VOLATILITY MEMORY — Horse Race (Section 7)
========================================================
Estimate 5 parametric models on all assets, compare by BIC.

Usage:
  python run_horse_race_v2.py shape_of_memory_100.csv --workers 8

Output:
  results/horse_race/
    horse_race_results.csv
    REPORT_HORSE_RACE.txt
    fig_bic_winner_by_class.pdf
"""

import sys, os, time, io, argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from parametric_final import estimate_all_models
from core_final import J_MAX

OUTDIR = 'results/horse_race'
MIN_OBS = 2500
MIN_OBS_CRY = 2000
N_RESTARTS = 5


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


def load_data(fp):
    df = pd.read_csv(fp)
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


def _worker(args):
    tk, r, meta = args
    t0 = time.time()
    try:
        res = estimate_all_models(r, ticker=tk, n_restarts=N_RESTARTS)
        res['meta'] = meta
        elapsed = time.time() - t0
        print(f"  {tk:<8s} BIC winner: {res['best_bic']:<8s} ({elapsed:.0f}s)")
        return tk, res
    except Exception as e:
        print(f"  {tk:<8s} FAILED: {e}")
        return tk, None


def results_to_df(results):
    rows = []
    for tk, res in results.items():
        if res is None:
            continue
        row = {
            'Ticker': tk,
            'Class': res['meta']['class'],
            'Sector': res['meta']['sector'],
            'T': res['T'],
            'BIC_winner': res['best_bic'],
            'LLF_winner': res['best_llf'],
        }
        for mname, mres in res['models'].items():
            row[f'{mname}_LLF'] = mres['llf']
            row[f'{mname}_BIC'] = mres['bic']

            if mname == 'GARCH':
                row['GARCH_beta'] = mres['beta']
            elif mname == 'SEARCH':
                row['SEARCH_alpha'] = mres['alpha']
                row['SEARCH_c'] = mres['c']
            elif mname == 'GMARCH':
                row['GMARCH_delta'] = mres['delta']
                row['GMARCH_lam'] = mres['lam']
                row['GMARCH_H'] = mres['H']
            elif mname == 'FIGARCH':
                row['FIGARCH_d'] = mres.get('d', np.nan)
                row['FIGARCH_phi'] = mres.get('phi', np.nan)
                row['FIGARCH_beta1'] = mres.get('beta1', np.nan)
            elif mname == 'HYGARCH':
                row['HYGARCH_d'] = mres.get('d', np.nan)
                row['HYGARCH_tau'] = mres.get('tau', np.nan)

        rows.append(row)
    return pd.DataFrame(rows)


def print_report(rw, df):
    n = len(df)
    rw.p("=" * 70)
    rw.p("  PARAMETRIC HORSE RACE — REPORT")
    rw.p(f"  {n} assets, 5 models, burn-in=2J, BIC primary metric")
    rw.p("=" * 70)

    # ── Table 1: BIC winner counts ──
    rw.p(f"\n{'─'*60}")
    rw.p(f"  BIC WINNER COUNTS")
    rw.p(f"{'─'*60}")
    counts = df['BIC_winner'].value_counts()
    for m in ['GARCH', 'SEARCH', 'GMARCH', 'FIGARCH', 'HYGARCH']:
        c = counts.get(m, 0)
        rw.p(f"  {m:<10s}: {c:>3d}/{n} ({100*c/n:.0f}%)")

    # ── Table 2: BIC winner by asset class ──
    rw.p(f"\n{'─'*60}")
    rw.p(f"  BIC WINNER BY ASSET CLASS")
    rw.p(f"{'─'*60}")
    rw.p(f"  {'Class':<16s} {'N':>3s} {'GARCH':>7s} {'SEARCH':>7s} "
         f"{'GMARCH':>7s} {'FIGARCH':>8s} {'HYGARCH':>8s}")
    rw.p(f"  {'─'*58}")

    for cls in sorted(df['Class'].unique()):
        sub = df[df['Class'] == cls]
        nc = len(sub)
        wins = sub['BIC_winner'].value_counts()
        vals = []
        for m in ['GARCH', 'SEARCH', 'GMARCH', 'FIGARCH', 'HYGARCH']:
            c = wins.get(m, 0)
            vals.append(f"{c}" if c > 0 else ".")
        rw.p(f"  {cls[:15]:<16s} {nc:>3d} {vals[0]:>7s} {vals[1]:>7s} "
             f"{vals[2]:>7s} {vals[3]:>8s} {vals[4]:>8s}")

    # ── Table 3: Shape parameter estimates by class ──
    rw.p(f"\n{'─'*70}")
    rw.p(f"  SHAPE PARAMETER ESTIMATES BY CLASS (medians)")
    rw.p(f"{'─'*70}")
    rw.p(f"  {'Class':<16s} {'SEARCH α':>9s} {'GMARCH δ':>9s} {'GMARCH H':>9s} "
         f"{'FIGARCH d':>10s}")
    rw.p(f"  {'─'*55}")

    for cls in sorted(df['Class'].unique()):
        sub = df[df['Class'] == cls]
        sa = sub['SEARCH_alpha'].median()
        gd = sub['GMARCH_delta'].median()
        gh = sub['GMARCH_H'].median()
        fd = sub['FIGARCH_d'].median()
        rw.p(f"  {cls[:15]:<16s} {sa:>9.3f} {gd:>9.3f} {gh:>9.3f} {fd:>10.3f}")

    # ── Table 4: Full results ──
    rw.p(f"\n{'═'*100}")
    rw.p(f"  FULL RESULTS")
    rw.p(f"{'═'*100}")
    rw.p(f"  {'Ticker':<8s} {'Class':<14s} {'Winner':<8s} "
         f"{'GARCH':>8s} {'SEARCH':>8s} {'GMARCH':>8s} {'FIGARCH':>8s} {'HYGARCH':>8s}")
    rw.p(f"  {'─'*98}")

    for _, row in df.sort_values(['Class', 'Ticker']).iterrows():
        bics = [row[f'{m}_BIC'] for m in ['GARCH','SEARCH','GMARCH','FIGARCH','HYGARCH']]
        rw.p(f"  {row['Ticker']:<8s} {str(row['Class'])[:13]:<14s} "
             f"{row['BIC_winner']:<8s} "
             + "  ".join(f"{b:>8.0f}" if np.isfinite(b) else f"{'FAIL':>8s}" for b in bics))

    # ── Summary stats ──
    rw.p(f"\n{'─'*60}")
    rw.p(f"  SUMMARY")
    rw.p(f"{'─'*60}")

    # Median BIC improvement over GARCH
    for m in ['SEARCH', 'GMARCH', 'FIGARCH', 'HYGARCH']:
        delta_bic = df[f'{m}_BIC'] - df['GARCH_BIC']
        valid = delta_bic.dropna()
        rw.p(f"  {m} vs GARCH (BIC): median ΔBIC = {valid.median():.1f}, "
             f"wins {(valid < 0).sum()}/{len(valid)}")


def make_figures(df):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # BIC winner by class
    classes = sorted(df['Class'].unique())
    models = ['GARCH', 'SEARCH', 'GMARCH', 'FIGARCH', 'HYGARCH']
    colors = ['#888888', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(classes))
    width = 0.15
    for i, (m, c) in enumerate(zip(models, colors)):
        counts = [len(df[(df['Class'] == cls) & (df['BIC_winner'] == m)])
                  for cls in classes]
        ax.bar(x + i * width, counts, width, label=m, color=c, alpha=0.8)
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Number of assets')
    ax.set_title('BIC winner by asset class')
    ax.legend()
    fig.savefig(os.path.join(OUTDIR, 'fig_bic_winner_by_class.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, 'fig_bic_winner_by_class.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved: fig_bic_winner_by_class")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=OUTDIR)
    args = parser.parse_args()

    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    n_cores = cpu_count()
    n_workers = args.workers or max(1, n_cores - 2)

    print(f"\n{'═'*60}")
    print(f"  PARAMETRIC HORSE RACE")
    print(f"  Cores: {n_cores}, Workers: {n_workers}")
    print(f"{'═'*60}")

    assets = load_data(args.datafile)
    print(f"  Loaded {len(assets)} assets")

    args_list = [
        (tk, data['r'],
         {'name': data['name'], 'class': data['class'], 'sector': data['sector']})
        for tk, data in assets.items()
    ]

    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        out = pool.map(_worker, args_list)
    elapsed = time.time() - t0

    results = {tk: res for tk, res in out if res is not None}
    print(f"\n  {len(results)} assets done, {len(assets)-len(results)} failed, "
          f"{elapsed/60:.1f} min")

    df = results_to_df(results)
    csv_path = os.path.join(OUTDIR, 'horse_race_results.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved: {csv_path}")

    rw = Report(os.path.join(OUTDIR, 'REPORT_HORSE_RACE.txt'))
    print_report(rw, df)
    rw.save()

    make_figures(df)

    print(f"\n{'═'*60}")
    print(f"  COMPLETE: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")
