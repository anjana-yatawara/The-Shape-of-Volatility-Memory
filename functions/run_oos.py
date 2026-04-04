"""
THE SHAPE OF VOLATILITY MEMORY — Multi-Horizon OOS Forecasting
================================================================
Fixed-parameter design: estimate on first 70%, forecast last 30%.
Multi-step forecasts via ARCH(∞) recursion.
Horizons: h = 1, 5, 10, 22, 44, 66, 88, 110, 132, 154, 176

Usage:
  python run_oos_multihorizon.py shape_of_memory_100.csv --workers 32
"""

import sys, os, time, io, argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from core_final import archinf_filter, gaussian_nll, J_MAX
from parametric_final import (
    garch_estimate, search_estimate, figarch_estimate,
    gmarch_estimate, hygarch_estimate,
    search_kernel, figarch_kernel, gmarch_kernel, hygarch_kernel,
)
import warnings
warnings.filterwarnings('ignore')

OUTDIR = 'results/oos'
MIN_OBS = 2500
MIN_OBS_CRY = 2000
OOS_FRAC = 0.30
N_RESTARTS = 5
HORIZONS = [1, 5, 10, 22, 44, 66, 88, 110, 132, 154, 176]
ALL_MODELS = ['GARCH', 'SEARCH', 'GMARCH', 'FIGARCH', 'HYGARCH']
ALT_MODELS = ['SEARCH', 'GMARCH', 'FIGARCH', 'HYGARCH']


# ══════════════════════════════════════════════════════════════
#  MULTI-STEP ARCH(∞) FORECAST
# ══════════════════════════════════════════════════════════════

def multistep_forecasts(omega, a, gamma, g, r, R, horizons, J=J_MAX):
    """
    Compute E_t[σ²_{t+h}] for each forecast origin t = R, ..., T-h_max-1
    and each horizon h in `horizons`.

    Uses the ARCH(∞) recursion:
      E_t[σ²_{t+h}] = ω + Σ_{j≥h} g(j) w_{t+h-j}
                       + (a+γ/2) Σ_{j=1}^{h-1} g(j) E_t[σ²_{t+h-j}]

    Returns dict: {h: array of forecasts for t = R, ..., T-h}
    """
    T = len(r)
    r2 = r ** 2
    neg = (r < 0).astype(float)
    w = (a + gamma * neg) * r2   # observed news impact
    P_half = a + gamma / 2.0
    h_max = max(horizons)

    # Precompute h-step forecasts iteratively for h = 1, ..., h_max
    # F[t, h] = E_t[σ²_{t+h}] for t = 0, ..., T-1
    # We only need origins t = R-1, ..., T-h_max-1 but compute all for simplicity
    # Store as F_prev[h] = array over t

    # h=1: just the one-step filter
    sigma2 = archinf_filter(r, omega, a, gamma, g)

    # For memory efficiency, store only the horizons we need + recent history
    # But h_max = 176 and we need all h = 1..176 for the recursion
    # Store F as a 2D array: F[h_idx, t] for h = 1, ..., h_max
    # This is h_max × T which is 176 × 6600 ≈ 1.2M floats = 9.2 MB. Fine.

    F = np.zeros((h_max, T))
    F[0, :] = sigma2  # h=1

    g_arr = np.array(g[:J])  # length J

    for h in range(2, h_max + 1):
        # Observed component: Σ_{j=h}^{J} g(j) w_{t+h-j}
        # For each origin t, this sums g(j) * w(t+h-j) for j = h..J
        # Equivalently: sum g(j) * w(t + h - j) = conv(w, g_reversed)[t+h-1]
        # but shifted. Let's compute directly.

        # g(j) for j = h..J (0-indexed: g_arr[h-1:])
        # w(t+h-j) for j = h..J means w(t), w(t-1), ..., w(t-J+h)
        # This is a convolution of w with g_arr[h-1:]

        g_tail = g_arr[h-1:]  # g(h), g(h+1), ..., g(J)
        L = len(g_tail)
        if L > 0:
            obs = np.convolve(w, g_tail, mode='full')[:T]
            # obs[t] = Σ_{k=0}^{L-1} w[t-k] * g_tail[k]
            # = Σ_{k=0}^{L-1} w[t-k] * g(h+k)
            # = Σ_{j=h}^{J} g(j) * w[t-(j-h)] = Σ_{j=h}^{J} g(j) * w[t+h-j]
            # Correct!
        else:
            obs = np.zeros(T)

        # Forecast component: (a+γ/2) Σ_{j=1}^{h-1} g(j) F[h-j-1, :]
        fcast = np.zeros(T)
        for j in range(1, min(h, J + 1)):
            fcast += g_arr[j-1] * F[h - j - 1, :]
        fcast *= P_half

        F[h-1, :] = omega + obs + fcast

    # Extract forecasts at desired horizons
    result = {}
    for h in horizons:
        # Forecast origins: t = R, ..., T-h-1
        # F[h-1, t] = E_t[σ²_{t+h}]
        if R + h <= T:
            result[h] = F[h-1, R:T-h+1] if T - h + 1 > R else F[h-1, R:R]
        else:
            result[h] = np.array([])

    return result


# ══════════════════════════════════════════════════════════════
#  LOSS + DM TEST
# ══════════════════════════════════════════════════════════════

def qlike_vec(r2, sigma2):
    """QLIKE loss vector. Requires r2 > 0 and sigma2 > 0."""
    valid = (r2 > 1e-20) & (sigma2 > 1e-20)
    loss = np.full(len(r2), np.nan)
    ratio = r2[valid] / sigma2[valid]
    loss[valid] = ratio - np.log(ratio) - 1.0
    return loss, valid


def newey_west_se(d, bandwidth=None):
    T = len(d)
    if bandwidth is None:
        bandwidth = int(np.floor(T ** (1/3)))
    d_dm = d - np.mean(d)
    gamma_0 = np.mean(d_dm ** 2)
    gamma_sum = 0.0
    for k in range(1, bandwidth + 1):
        w = 1.0 - k / (bandwidth + 1)
        gamma_k = np.mean(d_dm[k:] * d_dm[:-k])
        gamma_sum += 2 * w * gamma_k
    var_d = (gamma_0 + gamma_sum) / T
    return np.sqrt(max(var_d, 1e-20))


def dm_test(loss_a, loss_b):
    """DM > 0 means model B is better."""
    d = loss_a - loss_b
    d_mean = np.mean(d)
    se = newey_west_se(d)
    dm_stat = d_mean / se if se > 1e-15 else 0.0
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return {'dm_stat': dm_stat, 'p_value': p_value, 'mean_diff': d_mean}


# ══════════════════════════════════════════════════════════════
#  ONE-ASSET PIPELINE
# ══════════════════════════════════════════════════════════════

def oos_one_asset(r, meta, n_restarts=5):
    T = len(r)
    R = int(T * (1 - OOS_FRAC))
    r_is = r[:R]

    # ── Estimate models on in-sample ──
    garch   = garch_estimate(r_is, n_restarts=n_restarts, verbose=False)
    search  = search_estimate(r_is, garch_res=garch, n_restarts=n_restarts)
    gmarch  = gmarch_estimate(r_is, garch_res=garch, n_restarts=n_restarts)
    figarch = figarch_estimate(r_is, garch_res=garch, n_restarts=n_restarts)
    hygarch = hygarch_estimate(r_is, garch_res=garch, n_restarts=n_restarts)

    models_est = {
        'GARCH': garch, 'SEARCH': search, 'GMARCH': gmarch,
        'FIGARCH': figarch, 'HYGARCH': hygarch
    }

    # ── Multi-horizon forecasts ──
    forecasts = {}  # {model: {h: forecast_array}}
    for mname, mres in models_est.items():
        om, a, gam = mres['omega'], mres['a'], mres['gamma']
        g = mres['kernel']

        # BBM scaling for FIGARCH/HYGARCH
        if mname == 'FIGARCH' and not np.isnan(mres.get('d', np.nan)):
            _, psi = figarch_kernel(mres['phi'], mres['beta1'], mres['d'])
            if psi[0] > 1e-8:
                a, gam = a * psi[0], gam * psi[0]
        elif mname == 'HYGARCH' and not np.isnan(mres.get('d', np.nan)):
            _, psi = hygarch_kernel(mres['phi'], mres['beta1'],
                                     mres['d'], mres['tau'])
            if psi[0] > 1e-8:
                a, gam = a * psi[0], gam * psi[0]

        forecasts[mname] = multistep_forecasts(om, a, gam, g, r, R, HORIZONS)

    # ── Evaluate at each horizon ──
    results_by_h = {}
    for h in HORIZONS:
        # Realized proxy at horizon h: r²_{t+h}
        r2_h = r[R+h-1:] ** 2  if R + h - 1 < T else np.array([])
        # Trim to match forecast length
        n_fcast = len(forecasts['GARCH'].get(h, []))
        if n_fcast == 0 or len(r2_h) == 0:
            continue
        n_eval = min(n_fcast, len(r2_h))
        r2_h = r2_h[:n_eval]

        # Compute QLIKE for each model
        valid_all = r2_h > 1e-20
        ql_means = {}
        ql_losses = {}
        for mname in ALL_MODELS:
            fcast_h = forecasts[mname][h][:n_eval]
            valid = valid_all & (fcast_h > 1e-20)
            ratio = np.where(valid, r2_h / np.maximum(fcast_h, 1e-20), np.nan)
            loss = np.where(valid, ratio - np.log(np.maximum(ratio, 1e-20)) - 1.0, np.nan)
            ql_losses[mname] = loss[valid]
            ql_means[mname] = np.nanmean(loss)

        best = min(ql_means, key=ql_means.get)

        # DM tests: each alt vs GARCH
        dm_res = {}
        for mname in ALT_MODELS:
            # Use intersection of valid for both
            n_dm = min(len(ql_losses['GARCH']), len(ql_losses[mname]))
            if n_dm > 30:
                dm_res[mname] = dm_test(
                    ql_losses['GARCH'][:n_dm], ql_losses[mname][:n_dm])
            else:
                dm_res[mname] = {'dm_stat': 0, 'p_value': 1, 'mean_diff': 0}

        # SEARCH vs FIGARCH
        n_dm = min(len(ql_losses.get('FIGARCH', [])),
                   len(ql_losses.get('SEARCH', [])))
        if n_dm > 30:
            dm_res['SvF'] = dm_test(
                ql_losses['FIGARCH'][:n_dm], ql_losses['SEARCH'][:n_dm])
        else:
            dm_res['SvF'] = {'dm_stat': 0, 'p_value': 1, 'mean_diff': 0}

        results_by_h[h] = {
            'n_eval': n_eval,
            'ql_means': ql_means,
            'best': best,
            'dm': dm_res,
        }

    return {
        'meta': meta, 'T': T, 'R': R, 'P': T - R,
        'by_horizon': results_by_h,
        'search_alpha': search['alpha'],
        'figarch_d': figarch.get('d', np.nan),
        'garch_beta': garch['beta'],
    }


# ══════════════════════════════════════════════════════════════
#  PARALLEL WORKER
# ══════════════════════════════════════════════════════════════

def _worker(args):
    tk, r, meta = args
    t0 = time.time()
    try:
        res = oos_one_asset(r, meta, n_restarts=N_RESTARTS)
        elapsed = time.time() - t0
        # Print h=1 SEARCH gain
        h1 = res['by_horizon'].get(1, {})
        ql = h1.get('ql_means', {})
        if 'GARCH' in ql and 'SEARCH' in ql and ql['GARCH'] > 0:
            pct = 100 * (ql['GARCH'] - ql['SEARCH']) / ql['GARCH']
            print(f"  {tk:<8s} h=1 best: {h1.get('best','?'):<8s} "
                  f"SEARCH: {pct:+.1f}%  ({elapsed:.0f}s)")
        else:
            print(f"  {tk:<8s} done ({elapsed:.0f}s)")
        return tk, res
    except Exception as e:
        print(f"  {tk:<8s} FAILED: {e}")
        import traceback; traceback.print_exc()
        return tk, None


# ══════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════

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


def print_report(rw, results):
    n = len(results)
    rw.p("=" * 80)
    rw.p("  MULTI-HORIZON OOS FORECASTING REPORT")
    rw.p(f"  {n} assets, 5 models, 70/30 split, QLIKE primary")
    rw.p(f"  Horizons: {HORIZONS}")
    rw.p("=" * 80)

    for h in HORIZONS:
        rw.p(f"\n{'━'*80}")
        rw.p(f"  HORIZON h = {h} days ({h/22:.1f} months)")
        rw.p(f"{'━'*80}")

        # Collect results for this horizon
        winners = []
        ql_by_model = {m: [] for m in ALL_MODELS}
        dm_wins = {m: 0 for m in ALT_MODELS}
        dm_loses = {m: 0 for m in ALT_MODELS}
        class_data = {}

        for tk, res in results.items():
            if res is None:
                continue
            h_res = res['by_horizon'].get(h)
            if h_res is None:
                continue
            winners.append(h_res['best'])
            cls = res['meta']['class']
            for m in ALL_MODELS:
                ql_by_model[m].append(h_res['ql_means'].get(m, np.nan))
            for m in ALT_MODELS:
                dm = h_res['dm'].get(m, {})
                if dm.get('p_value', 1) < 0.05:
                    if dm.get('dm_stat', 0) > 0:
                        dm_wins[m] += 1
                    else:
                        dm_loses[m] += 1
            if cls not in class_data:
                class_data[cls] = {m: [] for m in ALL_MODELS}
                class_data[cls]['_winners'] = []
            for m in ALL_MODELS:
                class_data[cls][m].append(h_res['ql_means'].get(m, np.nan))
            class_data[cls]['_winners'].append(h_res['best'])

        # Winner counts
        from collections import Counter
        wc = Counter(winners)
        rw.p(f"\n  Winner counts:")
        for m in ALL_MODELS:
            rw.p(f"    {m:<10s}: {wc.get(m, 0):>3d}/{len(winners)}")

        # Mean QLIKE
        rw.p(f"\n  Mean QLIKE:")
        for m in ALL_MODELS:
            vals = [v for v in ql_by_model[m] if np.isfinite(v)]
            rw.p(f"    {m:<10s}: {np.mean(vals):.4f}")

        # DM summary
        rw.p(f"\n  DM tests (5% two-sided) vs GARCH:")
        for m in ALT_MODELS:
            rw.p(f"    {m:<10s}: wins {dm_wins[m]}, loses {dm_loses[m]}")

        # QLIKE by class
        rw.p(f"\n  {'Class':<16s} {'N':>3s}", end='')
        for m in ALL_MODELS:
            rw.p(f" {m:>8s}", end='')
        rw.p(f" {'Best':>8s}")
        rw.p(f"  {'─'*70}")
        for cls in sorted(class_data.keys()):
            cd = class_data[cls]
            nc = len(cd['_winners'])
            rw.p(f"  {cls[:15]:<16s} {nc:>3d}", end='')
            means = {}
            for m in ALL_MODELS:
                v = [x for x in cd[m] if np.isfinite(x)]
                mn = np.mean(v) if v else np.nan
                means[m] = mn
                rw.p(f" {mn:>8.4f}", end='')
            best = min(means, key=means.get)
            rw.p(f" {best:>8s}")

    # ── Summary table across horizons ──
    rw.p(f"\n\n{'━'*80}")
    rw.p(f"  SUMMARY ACROSS HORIZONS")
    rw.p(f"{'━'*80}")
    rw.p(f"\n  {'h':>5s} {'days':>5s}", end='')
    for m in ALL_MODELS:
        rw.p(f" {m+' wins':>10s}", end='')
    rw.p(f" {'DM S>G':>7s} {'DM F>G':>7s} {'DM S>F':>7s}")
    rw.p(f"  {'─'*80}")

    for h in HORIZONS:
        winners_h = []
        dm_sg_w, dm_fg_w, dm_sf_w = 0, 0, 0
        for tk, res in results.items():
            if res is None:
                continue
            h_res = res['by_horizon'].get(h)
            if h_res is None:
                continue
            winners_h.append(h_res['best'])
            for label, key in [('sg', 'SEARCH'), ('fg', 'FIGARCH')]:
                dm = h_res['dm'].get(key, {})
                if dm.get('p_value', 1) < 0.05 and dm.get('dm_stat', 0) > 0:
                    if label == 'sg':
                        dm_sg_w += 1
                    else:
                        dm_fg_w += 1
            dm_sf = h_res['dm'].get('SvF', {})
            if dm_sf.get('p_value', 1) < 0.05 and dm_sf.get('dm_stat', 0) > 0:
                dm_sf_w += 1

        wc = Counter(winners_h)
        rw.p(f"  {h:>5d} {h/22:>5.1f}", end='')
        for m in ALL_MODELS:
            rw.p(f" {wc.get(m,0):>10d}", end='')
        rw.p(f" {dm_sg_w:>7d} {dm_fg_w:>7d} {dm_sf_w:>7d}")


def results_to_csv(results, path):
    """Flatten results to CSV: one row per (asset, horizon)."""
    rows = []
    for tk, res in results.items():
        if res is None:
            continue
        for h in HORIZONS:
            h_res = res['by_horizon'].get(h)
            if h_res is None:
                continue
            row = {
                'Ticker': tk, 'Class': res['meta']['class'],
                'T': res['T'], 'R': res['R'], 'h': h,
                'n_eval': h_res['n_eval'],
                'Best': h_res['best'],
            }
            for m in ALL_MODELS:
                row[f'{m}_QLIKE'] = h_res['ql_means'].get(m, np.nan)
            for m in ALT_MODELS:
                dm = h_res['dm'].get(m, {})
                row[f'DM_{m}_stat'] = dm.get('dm_stat', np.nan)
                row[f'DM_{m}_pval'] = dm.get('p_value', np.nan)
            dm_sf = h_res['dm'].get('SvF', {})
            row['DM_SvF_stat'] = dm_sf.get('dm_stat', np.nan)
            row['DM_SvF_pval'] = dm_sf.get('p_value', np.nan)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, float_format='%.6f')


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

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
    print(f"  MULTI-HORIZON OOS FORECASTING")
    print(f"  Horizons: {HORIZONS}")
    print(f"  Models: {ALL_MODELS}")
    print(f"  Workers: {n_workers}")
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
    print(f"\n  {len(results)} done, {len(assets)-len(results)} failed, "
          f"{elapsed/60:.1f} min")

    csv_path = os.path.join(OUTDIR, 'oos_multihorizon.csv')
    results_to_csv(results, csv_path)
    print(f"  Saved: {csv_path}")

    rw = Report(os.path.join(OUTDIR, 'REPORT_OOS_MULTIHORIZON.txt'))
    print_report(rw, results)
    rw.save()
    print(f"  Saved: REPORT_OOS_MULTIHORIZON.txt")

    print(f"\n{'═'*60}")
    print(f"  COMPLETE: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")
