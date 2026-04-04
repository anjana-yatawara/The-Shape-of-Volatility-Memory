"""
THE SHAPE OF VOLATILITY MEMORY — Tables & Figures Generator
=============================================================
Generates all LaTeX tables and publication-quality figures
from the production run outputs.

Usage (in Spyder):
  %run generate_tables_figures.py

Requires:
  - shape_of_memory_100.csv (raw data, for descriptive stats)
  - results/discovery/discovery_K6.csv
  - results/discovery/discovery_K8.csv
  - results/discovery/discovery_K12.csv

Outputs:
  - tables/tab_descriptive.tex
  - tables/tab_full_K8.tex
  - tables/tab_alpha_by_class.tex  (already in Section 5, for reference)
  - tables/tab_robustness_K.tex    (already in Section 5, for reference)
  - tables/tab_kernel_values.tex   (appendix)
  - figures/fig_kernel_overlay.pdf
  - figures/fig_alpha_by_class.pdf
  - figures/fig_cross_K.pdf
  - figures/fig_mc_garch.pdf
  - figures/fig_mc_search.pdf
  - figures/fig_mc_figarch.pdf
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# ══════════════════════════════════════════════════════════════
#  PATHS — adjust if your layout differs
# ══════════════════════════════════════════════════════════════
DATA_FILE = 'shape_of_memory_100.csv'
DISC_K6  = 'results/discovery/discovery_K6.csv'
DISC_K8  = 'results/discovery/discovery_K8.csv'
DISC_K12 = 'results/discovery/discovery_K12.csv'
MC_GARCH   = 'results/montecarlo/montecarlo_garch.csv'
MC_SEARCH  = 'results/montecarlo/montecarlo_search.csv'
MC_FIGARCH = 'results/montecarlo/montecarlo_figarch.csv'

os.makedirs('tables', exist_ok=True)
os.makedirs('figures', exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Journal style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.0,
})

# Class display order (used everywhere)
CLASS_ORDER = [
    'US Index', 'US Stock', 'Sector ETF', 'International',
    'Commodity', 'Currency', 'Fixed Income', 'Crypto', 'Volatility'
]

CLASS_DISPLAY = {
    'US Index': 'US equity indices',
    'US Stock': 'US individual stocks',
    'Sector ETF': 'US sector ETFs',
    'International': 'International equity',
    'Commodity': 'Commodities',
    'Currency': 'Currencies',
    'Fixed Income': 'Fixed income',
    'Crypto': 'Cryptocurrency',
    'Volatility': 'VIX',
}


# ══════════════════════════════════════════════════════════════
#  TABLE 1: DESCRIPTIVE STATISTICS (Section 4)
# ══════════════════════════════════════════════════════════════
def make_descriptive_table():
    print('Generating descriptive statistics table...')
    if not os.path.exists(DATA_FILE):
        print(f'  WARNING: {DATA_FILE} not found. Skipping.')
        return

    df = pd.read_csv(DATA_FILE)

    # Detect format
    if 'Ticker' in df.columns and 'LogRet' in df.columns:
        # Long format with pre-computed returns
        assets = {}
        for tk in df['Ticker'].unique():
            sub = df[df['Ticker'] == tk].sort_values('Date')
            r = sub['LogRet'].values
            r = r - np.mean(r)
            cls = sub['Class'].iloc[0] if 'Class' in sub.columns else 'Unknown'
            assets[tk] = {'r': r, 'T': len(r), 'class': cls}
    else:
        print('  Wide format detected — need Class mapping. Skipping.')
        return

    rows = []
    for tk, d in assets.items():
        r = d['r']
        rows.append({
            'Ticker': tk,
            'Class': d['class'],
            'T': d['T'],
            'mean_r': np.mean(r),
            'sd_r': np.std(r),
            'skew': skew(r),
            'kurt': kurtosis(r),  # excess kurtosis
        })
    rdf = pd.DataFrame(rows)

    # Aggregate by class
    lines = []
    for cls in CLASS_ORDER:
        sub = rdf[rdf['Class'] == cls]
        if len(sub) == 0:
            continue
        n = len(sub)
        med_T = int(sub['T'].median())
        med_r = sub['mean_r'].median()
        med_sd = sub['sd_r'].median()
        med_sk = sub['skew'].median()
        med_ku = sub['kurt'].median()
        label = CLASS_DISPLAY.get(cls, cls)
        lines.append(
            f'{label:<24s} & {n:>2d} & {med_T:>,d} & {med_r:>6.3f} '
            f'& {med_sd:>5.2f} & {med_sk:>6.2f} & {med_ku:>6.1f} \\\\'
        )

    # All assets
    n = len(rdf)
    lines.append('\\midrule')
    lines.append(
        f'{"All assets":<24s} & {n:>2d} & {int(rdf["T"].median()):>,d} '
        f'& {rdf["mean_r"].median():>6.3f} & {rdf["sd_r"].median():>5.2f} '
        f'& {rdf["skew"].median():>6.2f} & {rdf["kurt"].median():>6.1f} \\\\'
    )

    tex = '\\begin{tabular}{lrrrrrr}\n\\toprule\n'
    tex += 'Asset class & $N$ & $\\bar{T}$ & $\\bar{r}$ (\\%) & $\\bar{\\sigma}$ (\\%) & Skew & Kurt \\\\\n'
    tex += '\\midrule\n'
    tex += '\n'.join(lines) + '\n'
    tex += '\\bottomrule\n\\end{tabular}'

    with open('tables/tab_descriptive.tex', 'w') as f:
        f.write(tex)
    print(f'  Saved: tables/tab_descriptive.tex')


# ══════════════════════════════════════════════════════════════
#  TABLE A1: FULL RESULTS (Appendix)
# ══════════════════════════════════════════════════════════════
def make_full_results_table():
    print('Generating full results table (K=8)...')
    df = pd.read_csv(DISC_K8)
    df = df.sort_values(['Class', 'Ticker'])

    lines = []
    for _, row in df.iterrows():
        hl = str(int(row['Half_Life'])) if row['Half_Life'] < 252 else '$>$252'
        rej = '$\\checkmark$' if row['LR_reject05'] else ''
        lines.append(
            f'{row["Ticker"]:<8s} & {row["Class"]:<16s} & {int(row["T"]):>5d} '
            f'& {row["GARCH_LLF"]:>9.1f} & {row["LKV_LLF"]:>9.1f} '
            f'& {row["LR_stat"]:>7.1f} & {hl:>4s} '
            f'& {row["SE_alpha"]:>5.3f} & {row["SE_R2"]:>6.3f} '
            f'& {row["GARCH_beta"]:>5.3f} & {rej} \\\\'
        )

    tex = '\\begin{longtable}{llrrrrcrrrl}\n'
    tex += '\\caption{Full estimation results ($K = 8$, $\\lambda = 10$). '
    tex += 'LLF is the Gaussian quasi-log-likelihood. LR is the likelihood ratio statistic '
    tex += '($\\chi^2_7$ critical value 14.07). HL is the half-life in trading days. '
    tex += '$\\hat{\\alpha}$ and $R^2$ are from the NLS stretched exponential fit.}\n'
    tex += '\\label{tab:full_results}\\\\\n'
    tex += '\\toprule\n'
    tex += 'Ticker & Class & $T$ & GARCH & Spline & LR & HL & $\\hat{\\alpha}$ & $R^2$ & $\\hat{\\beta}$ & Rej. \\\\\n'
    tex += '\\midrule\n\\endfirsthead\n'
    tex += '\\multicolumn{11}{l}{\\textit{Table~\\ref{tab:full_results} continued}} \\\\\n'
    tex += '\\toprule\n'
    tex += 'Ticker & Class & $T$ & GARCH & Spline & LR & HL & $\\hat{\\alpha}$ & $R^2$ & $\\hat{\\beta}$ & Rej. \\\\\n'
    tex += '\\midrule\n\\endhead\n'
    tex += '\n'.join(lines) + '\n'
    tex += '\\bottomrule\n\\end{longtable}'

    with open('tables/tab_full_K8.tex', 'w') as f:
        f.write(tex)
    print(f'  Saved: tables/tab_full_K8.tex')


# ══════════════════════════════════════════════════════════════
#  TABLE A2: KERNEL VALUES AT KEY LAGS (Appendix)
# ══════════════════════════════════════════════════════════════
def make_kernel_table():
    print('Generating kernel values table (K=8)...')
    df = pd.read_csv(DISC_K8)
    df = df.sort_values(['Class', 'Ticker'])

    g_lags = [1, 2, 5, 10, 21, 63, 252]
    g_cols = [f'g_{j}' for j in g_lags]

    lines = []
    for _, row in df.iterrows():
        vals = []
        for c in g_cols:
            v = row[c]
            if v >= 0.01:
                vals.append(f'{v:.3f}')
            elif v >= 1e-4:
                vals.append(f'{v:.4f}')
            else:
                vals.append(f'{v:.1e}')
        lines.append(
            f'{row["Ticker"]:<8s} & {row["Class"]:<14s} & '
            + ' & '.join(vals) + ' \\\\'
        )

    header_lags = ' & '.join([f'$j={j}$' for j in g_lags])
    tex = '\\begin{longtable}{ll' + 'r' * len(g_lags) + '}\n'
    tex += '\\caption{Normalized kernel values $g(j)/g(1)$ at selected lags ($K = 8$).}\n'
    tex += '\\label{tab:kernel_values}\\\\\n'
    tex += '\\toprule\n'
    tex += f'Ticker & Class & {header_lags} \\\\\n'
    tex += '\\midrule\n\\endfirsthead\n'
    tex += '\\multicolumn{' + str(2+len(g_lags)) + '}{l}{\\textit{Table~\\ref{tab:kernel_values} continued}} \\\\\n'
    tex += '\\toprule\n'
    tex += f'Ticker & Class & {header_lags} \\\\\n'
    tex += '\\midrule\n\\endhead\n'
    tex += '\n'.join(lines) + '\n'
    tex += '\\bottomrule\n\\end{longtable}'

    with open('tables/tab_kernel_values.tex', 'w') as f:
        f.write(tex)
    print(f'  Saved: tables/tab_kernel_values.tex')


# ══════════════════════════════════════════════════════════════
#  FIGURE 1: KERNEL OVERLAY (all 100 assets)
# ══════════════════════════════════════════════════════════════
def fig_kernel_overlay():
    print('Generating kernel overlay figure...')
    df = pd.read_csv(DISC_K8)
    g_cols = [c for c in df.columns if c.startswith('g_')]
    lags = [int(c.split('_')[1]) for c in g_cols]

    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Plot each asset in light gray
    for _, row in df.iterrows():
        vals = [row[c] for c in g_cols]
        ax.plot(lags, vals, color='steelblue', alpha=0.15, linewidth=0.5)

    # Cross-sectional mean
    mean_vals = [df[c].mean() for c in g_cols]
    ax.plot(lags, mean_vals, color='black', linewidth=2.0, label='Cross-sectional mean')

    # GARCH reference (beta = median)
    beta_med = df['GARCH_beta'].median()
    j_ref = np.array(lags)
    garch_ref = beta_med ** (j_ref - 1)
    ax.plot(lags, garch_ref, 'r--', linewidth=1.5,
            label=f'GARCH ($\\beta={beta_med:.2f}$)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Lag $j$ (trading days)')
    ax.set_ylabel('$g(j)/g(1)$')
    ax.set_xlim(1, 260)
    ax.set_ylim(1e-4, 1.5)
    ax.legend(frameon=False)
    fig.savefig('figures/fig_kernel_overlay.pdf')
    fig.savefig('figures/fig_kernel_overlay.png')
    plt.close(fig)
    print('  Saved: figures/fig_kernel_overlay')


# ══════════════════════════════════════════════════════════════
#  FIGURE 2: ALPHA BY ASSET CLASS (publication quality)
# ══════════════════════════════════════════════════════════════
def fig_alpha_by_class():
    print('Generating alpha by class figure...')
    df = pd.read_csv(DISC_K8)

    # Order by median alpha
    medians = df.groupby('Class')['SE_alpha'].median().sort_values()
    ordered_classes = medians.index.tolist()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(ordered_classes)))

    for i, cls in enumerate(ordered_classes):
        sub = df[df['Class'] == cls]['SE_alpha']
        x = np.full(len(sub), i) + np.random.uniform(-0.15, 0.15, len(sub))
        ax.scatter(x, sub, s=20, alpha=0.5, color=cmap[i], edgecolors='none')
        ax.scatter(i, sub.median(), s=100, marker='D', color=cmap[i],
                  edgecolors='black', linewidths=0.8, zorder=5)

    ax.axhline(1.0, color='red', linestyle='--', alpha=0.6, linewidth=0.8)
    ax.text(len(ordered_classes)-0.5, 1.05, 'GARCH ($\\alpha=1$)',
            color='red', fontsize=8, ha='right')

    ax.set_xticks(range(len(ordered_classes)))
    ax.set_xticklabels(ordered_classes, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Shape parameter $\\hat{\\alpha}$')
    ax.set_ylim(-0.05, 3.15)
    fig.savefig('figures/fig_alpha_by_class.pdf')
    fig.savefig('figures/fig_alpha_by_class.png')
    plt.close(fig)
    print('  Saved: figures/fig_alpha_by_class')


# ══════════════════════════════════════════════════════════════
#  FIGURE 3: CROSS-K STABILITY
# ══════════════════════════════════════════════════════════════
def fig_cross_K():
    print('Generating cross-K stability figure...')
    dfs = {}
    for label, path in [('$K=6$', DISC_K6), ('$K=8$', DISC_K8), ('$K=12$', DISC_K12)]:
        if os.path.exists(path):
            dfs[label] = pd.read_csv(path)

    classes = sorted(list(dfs.values())[0]['Class'].unique())

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(classes))
    width = 0.25
    colors = ['#4C72B0', '#55A868', '#C44E52']

    for i, (label, d) in enumerate(dfs.items()):
        meds = [d.loc[d['Class'] == c, 'SE_alpha'].median() for c in classes]
        ax.bar(x + i * width, meds, width, label=label, color=colors[i], alpha=0.85)

    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Median $\\hat{\\alpha}$')
    ax.legend(frameon=False)
    fig.savefig('figures/fig_cross_K.pdf')
    fig.savefig('figures/fig_cross_K.png')
    plt.close(fig)
    print('  Saved: figures/fig_cross_K')


# ══════════════════════════════════════════════════════════════
#  FIGURES 4-6: MONTE CARLO HISTOGRAMS
# ══════════════════════════════════════════════════════════════
def fig_mc_histograms():
    print('Generating MC histogram figures...')
    configs = [
        ('GARCH',   MC_GARCH,   1.0,   'fig_mc_garch'),
        ('SEARCH',  MC_SEARCH,  0.55,  'fig_mc_search'),
        ('FIGARCH', MC_FIGARCH, None,  'fig_mc_figarch'),
    ]

    for dgp_name, path, true_alpha, fname in configs:
        if not os.path.exists(path):
            print(f'  WARNING: {path} not found. Skipping {dgp_name}.')
            continue

        df = pd.read_csv(path)
        valid = df['alpha'].dropna()
        if len(valid) < 10:
            continue

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(valid, bins=40, density=True, alpha=0.7, color='steelblue',
                edgecolor='white', linewidth=0.5)

        if true_alpha is not None:
            ax.axvline(true_alpha, color='red', linewidth=1.5, linestyle='--',
                      label=f'True $\\alpha = {true_alpha:.2f}$')
        ax.axvline(valid.median(), color='black', linewidth=1.5,
                  label=f'Median $\\hat{{\\alpha}} = {valid.median():.3f}$')

        ax.set_xlabel('Estimated $\\hat{\\alpha}$')
        ax.set_ylabel('Density')
        ax.legend(frameon=False, fontsize=9)
        fig.savefig(f'figures/{fname}.pdf')
        fig.savefig(f'figures/{fname}.png')
        plt.close(fig)
        print(f'  Saved: figures/{fname}')


# ══════════════════════════════════════════════════════════════
#  FIGURE 7: VIX KERNEL
# ══════════════════════════════════════════════════════════════
def fig_vix_kernel():
    print('Generating VIX kernel figure...')
    df = pd.read_csv(DISC_K8)
    vix = df[df['Ticker'] == '^VIX']
    if len(vix) == 0:
        print('  VIX not found. Skipping.')
        return

    g_cols = [c for c in df.columns if c.startswith('g_')]
    lags = [int(c.split('_')[1]) for c in g_cols]
    vals = [vix[c].values[0] for c in g_cols]

    beta = vix['GARCH_beta'].values[0]
    j_dense = np.arange(1, 253)
    garch_ref = beta ** (j_dense - 1)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(lags, vals, 'ko-', markersize=5, linewidth=1.5, label='Estimated kernel')
    ax.plot(j_dense, garch_ref, 'r--', linewidth=1.0, alpha=0.7,
            label=f'GARCH ($\\beta={beta:.3f}$)')
    ax.axhline(np.mean(vals[4:]), color='gray', linestyle=':', alpha=0.5,
              label=f'Floor ($g \\approx {np.mean(vals[4:]):.3f}$)')

    ax.set_xlabel('Lag $j$ (trading days)')
    ax.set_ylabel('$g(j)/g(1)$')
    ax.set_xlim(0, 260)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig('figures/fig_vix_kernel.pdf')
    fig.savefig('figures/fig_vix_kernel.png')
    plt.close(fig)
    print('  Saved: figures/fig_vix_kernel')


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('='*60)
    print('  GENERATING ALL TABLES AND FIGURES')
    print('='*60)

    # Tables
    make_descriptive_table()
    make_full_results_table()
    make_kernel_table()

    # Figures
    fig_kernel_overlay()
    fig_alpha_by_class()
    fig_cross_K()
    fig_mc_histograms()
    fig_vix_kernel()

    print('\n' + '='*60)
    print('  DONE. Check tables/ and figures/ directories.')
    print('='*60)
