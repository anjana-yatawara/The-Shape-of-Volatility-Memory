"""
PARAMETRIC MODEL ESTIMATION v2
===============================
Imports infrastructure from core_final (vectorized filter, burn-in=2J).
Defines kernel functions + QMLE estimators for:
  SEARCH, GMARCH, FIGARCH, HYGARCH

All models use the SAME archinf_filter and gaussian_nll as the
discovery pipeline, ensuring LLF/BIC values are directly comparable.

Usage:
  from parametric_final import estimate_all_models
  res = estimate_all_models(r, verbose=True)
"""

import numpy as np
from scipy.optimize import minimize
from core_final import (
    archinf_filter, gaussian_nll, garch_estimate,
    J_MAX, REPORT_LAGS,
)
import warnings
warnings.filterwarnings('ignore')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ══════════════════════════════════════════════════════════════
#  KERNEL BUILDERS
# ══════════════════════════════════════════════════════════════

def search_kernel(c, alpha, J=J_MAX):
    """SEARCH: g(j) = exp[-c(j^α - 1)].  g(1) = 1."""
    jj = np.arange(1, J + 1, dtype=float)
    return np.exp(-c * (jj ** alpha - 1.0))


def gmarch_kernel(delta, lam, J=J_MAX):
    """GMARCH: g(j) = j^δ exp[-λ(j-1)].  g(1) = 1."""
    jj = np.arange(1, J + 1, dtype=float)
    return (jj ** delta) * np.exp(-lam * (jj - 1.0))


def figarch_weights(phi, beta1, d, J=J_MAX):
    """FIGARCH(1,d,1) ARCH(∞) weights via BBM recursion.
    Returns raw ψ_1, ..., ψ_J (NOT normalized).
    """
    delta = np.zeros(J + 1)
    delta[0] = 1.0
    for k in range(1, J + 1):
        delta[k] = delta[k - 1] * (k - 1 - d) / k

    psi = np.zeros(J)
    psi[0] = phi - beta1 + d
    for k in range(1, J):
        kk = k + 1
        psi[k] = beta1 * psi[k - 1] + (phi * delta[kk - 1] - delta[kk])

    return np.maximum(psi, 0.0)


def figarch_kernel(phi, beta1, d, J=J_MAX):
    """FIGARCH kernel normalized to g(1) = 1."""
    psi = figarch_weights(phi, beta1, d, J)
    if psi[0] > 1e-10:
        return psi / psi[0], psi
    return np.ones(J), psi


def hygarch_kernel(phi, beta1, d, tau, J=J_MAX):
    """HYGARCH kernel: (1-τ)·GARCH + τ·FIGARCH, normalized to g(1)=1."""
    jj = np.arange(J)
    psi_ga = max(phi - beta1, 1e-8) * (beta1 ** jj)
    psi_fi = figarch_weights(phi, beta1, d, J)
    psi_hy = (1 - tau) * psi_ga + tau * psi_fi
    psi_hy = np.maximum(psi_hy, 0.0)

    if psi_hy[0] > 1e-10:
        g = psi_hy / psi_hy[0]
    else:
        g = np.ones(J)
    return g, psi_hy


# ══════════════════════════════════════════════════════════════
#  SHARED ESTIMATION HELPERS
# ══════════════════════════════════════════════════════════════

def _compute_bic(llf, k, T):
    """BIC with T_eff = T - 2*J_MAX (matching core_final)."""
    T_eff = T - 2 * J_MAX
    return np.log(T_eff) * k - 2 * llf


def _run_restarts(obj_func, inits, method='L-BFGS-B', maxiter=10000):
    """Run optimizer from multiple initial points, return best."""
    best_res, best_fun = None, np.inf
    for x0 in inits:
        try:
            res = minimize(obj_func, x0, method=method,
                           options={'maxiter': maxiter, 'ftol': 1e-12,
                                    'gtol': 1e-8})
            if res.fun < best_fun:
                best_fun, best_res = res.fun, res
        except Exception:
            pass
    return best_res


# ══════════════════════════════════════════════════════════════
#  SEARCH ESTIMATION
# ══════════════════════════════════════════════════════════════

def search_estimate(r, garch_res=None, n_restarts=5, verbose=False):
    """SEARCH: g(j) = exp[-c(j^α - 1)]. 5 params."""
    T = len(r)

    def obj(x):
        omega = np.exp(x[0])
        a     = np.exp(x[1])
        gamma = np.exp(x[2])
        c     = np.exp(x[3])
        alpha = np.exp(x[4])
        g = search_kernel(c, alpha)
        P = (a + gamma / 2) * np.sum(g)
        if P >= 0.999:
            return 1e12 + 1e6 * (P - 0.999) ** 2
        sigma2 = archinf_filter(r, omega, a, gamma, g)
        nll = gaussian_nll(r, sigma2)
        return nll if np.isfinite(nll) else 1e12

    # Build inits from GARCH
    if garch_res is not None:
        beta_g = garch_res['beta']
        c_g = -np.log(max(beta_g, 0.01))
        x0_g = np.array([np.log(max(garch_res['omega'], 1e-8)),
                          np.log(max(garch_res['a'], 1e-8)),
                          np.log(max(garch_res['gamma'], 1e-8)),
                          np.log(max(c_g, 1e-4)),
                          np.log(1.0)])             # α = 1 → GARCH
    else:
        x0_g = np.array([np.log(0.05), np.log(0.03), np.log(0.07),
                          np.log(0.062), np.log(1.0)])

    inits = [x0_g.copy()]                           # #0: GARCH point
    for a_try, c_try in [(0.5, 0.15), (0.3, 0.25), (0.7, 0.08)]:
        x = x0_g.copy()
        x[3], x[4] = np.log(c_try), np.log(a_try)
        inits.append(x)
    for _ in range(max(0, n_restarts - 4)):
        x = x0_g.copy()
        x += np.random.randn(5) * 0.3
        x[4] = np.log(0.3 + np.random.rand() * 0.8)
        inits.append(x)

    best = _run_restarts(obj, inits)
    if best is None:
        raise RuntimeError("SEARCH estimation failed")

    omega = np.exp(best.x[0])
    a     = np.exp(best.x[1])
    gamma = np.exp(best.x[2])
    c     = np.exp(best.x[3])
    alpha = np.exp(best.x[4])
    g = search_kernel(c, alpha)
    llf = -best.fun
    bic = _compute_bic(llf, 5, T)

    if verbose:
        print(f"    SEARCH:  c={c:.4f} α={alpha:.4f} LLF={llf:.1f} BIC={bic:.1f}")

    return {'name': 'SEARCH', 'omega': omega, 'a': a, 'gamma': gamma,
            'c': c, 'alpha': alpha, 'kernel': g,
            'llf': llf, 'bic': bic, 'k': 5}


# ══════════════════════════════════════════════════════════════
#  GMARCH ESTIMATION
# ══════════════════════════════════════════════════════════════

def gmarch_estimate(r, garch_res=None, n_restarts=5, verbose=False):
    """GMARCH: g(j) = j^δ exp[-λ(j-1)]. 5 params."""
    T = len(r)

    def obj(x):
        omega = np.exp(x[0])
        a     = np.exp(x[1])
        gamma = np.exp(x[2])
        delta = x[3]                               # unconstrained
        lam   = np.exp(x[4])
        if delta < -0.5:
            return 1e12 + 1e6 * (delta + 0.5) ** 2
        g = gmarch_kernel(delta, lam)
        P = (a + gamma / 2) * np.sum(g)
        if P >= 0.999:
            return 1e12 + 1e6 * (P - 0.999) ** 2
        sigma2 = archinf_filter(r, omega, a, gamma, g)
        nll = gaussian_nll(r, sigma2)
        return nll if np.isfinite(nll) else 1e12

    if garch_res is not None:
        beta_g = garch_res['beta']
        lam_g = -np.log(max(beta_g, 0.01))
        x0_g = np.array([np.log(max(garch_res['omega'], 1e-8)),
                          np.log(max(garch_res['a'], 1e-8)),
                          np.log(max(garch_res['gamma'], 1e-8)),
                          0.0,                       # δ = 0 → GARCH
                          np.log(max(lam_g, 1e-4))])
    else:
        x0_g = np.array([np.log(0.05), np.log(0.03), np.log(0.07),
                          0.0, np.log(0.062)])

    inits = [x0_g.copy()]                           # #0: GARCH point
    # Positive δ (sub-exponential, matching our discovery)
    for d_try in [0.2, 0.4]:
        x = x0_g.copy()
        x[3] = d_try
        inits.append(x)
    # Negative δ (rough volatility region)
    x = x0_g.copy()
    x[3] = -0.3
    inits.append(x)
    for _ in range(max(0, n_restarts - 4)):
        x = x0_g.copy()
        x[3] = np.random.uniform(-0.3, 0.5)
        x[4] += np.random.randn() * 0.3
        inits.append(x)

    best = _run_restarts(obj, inits)
    if best is None:
        raise RuntimeError("GMARCH estimation failed")

    omega = np.exp(best.x[0])
    a     = np.exp(best.x[1])
    gamma = np.exp(best.x[2])
    delta = best.x[3]
    lam   = np.exp(best.x[4])
    g = gmarch_kernel(delta, lam)
    llf = -best.fun
    bic = _compute_bic(llf, 5, T)

    if verbose:
        print(f"    GMARCH:  δ={delta:.4f} λ={lam:.4f} H={(delta+1)/2:.3f} "
              f"LLF={llf:.1f} BIC={bic:.1f}")

    return {'name': 'GMARCH', 'omega': omega, 'a': a, 'gamma': gamma,
            'delta': delta, 'lam': lam, 'H': (delta + 1) / 2,
            'kernel': g, 'llf': llf, 'bic': bic, 'k': 5}


# ══════════════════════════════════════════════════════════════
#  FIGARCH ESTIMATION
# ══════════════════════════════════════════════════════════════

def figarch_estimate(r, garch_res=None, n_restarts=5, verbose=False):
    """FIGARCH(1,d,1): recursive kernel. 6 params."""
    T = len(r)

    def obj(x):
        omega = np.exp(x[0])
        a     = np.exp(x[1])
        gamma = np.exp(x[2])
        phi   = sigmoid(x[3])
        beta1 = sigmoid(x[4])
        d     = sigmoid(x[5])

        if phi <= beta1:
            return 1e12 + 1e6 * (beta1 - phi + 0.01) ** 2

        g, psi = figarch_kernel(phi, beta1, d)
        # For FIGARCH, news impact scaled by ψ₁
        a_eff = a * max(psi[0], 1e-8)
        gamma_eff = gamma * max(psi[0], 1e-8)
        P = (a_eff + gamma_eff / 2) * np.sum(g)
        if P >= 0.999:
            return 1e12 + 1e6 * (P - 0.999) ** 2

        sigma2 = archinf_filter(r, omega, a_eff, gamma_eff, g)
        nll = gaussian_nll(r, sigma2)

        n_neg = np.sum(psi < -1e-8)
        if n_neg > 0:
            nll += 1e4 * np.sum(np.minimum(psi, 0) ** 2)

        return nll if np.isfinite(nll) else 1e12

    if garch_res is not None:
        beta_g = garch_res['beta']
        phi_init = min(garch_res['a'] + garch_res['gamma'] / 2 + beta_g, 0.999)
        x0_g = np.array([
            np.log(max(garch_res['omega'], 1e-8)),
            np.log(max(garch_res['a'], 1e-8)),
            np.log(max(garch_res['gamma'], 1e-8)),
            np.log(phi_init / (1 - phi_init)),
            np.log(beta_g / (1 - beta_g)),
            -3.0,                                   # d ≈ 0.05 (near GARCH)
        ])
    else:
        x0_g = np.array([np.log(0.05), np.log(0.03), np.log(0.07),
                          1.5, 2.0, -3.0])

    inits = [x0_g.copy()]
    for d_logit in [-1.5, -0.5, 0.0]:              # d ≈ 0.18, 0.38, 0.50
        x = x0_g.copy()
        x[5] = d_logit
        inits.append(x)
    for _ in range(max(0, n_restarts - 4)):
        x = x0_g.copy()
        x += np.random.randn(6) * 0.3
        inits.append(x)

    best = _run_restarts(obj, inits)
    if best is None:
        return {'name': 'FIGARCH', 'llf': -1e12, 'bic': 1e12, 'k': 6,
                'omega': np.nan, 'a': np.nan, 'gamma': np.nan,
                'phi': np.nan, 'beta1': np.nan, 'd': np.nan,
                'kernel': np.ones(J_MAX)}

    omega = np.exp(best.x[0])
    a     = np.exp(best.x[1])
    gamma = np.exp(best.x[2])
    phi   = sigmoid(best.x[3])
    beta1 = sigmoid(best.x[4])
    d     = sigmoid(best.x[5])
    g, psi = figarch_kernel(phi, beta1, d)
    llf = -best.fun
    bic = _compute_bic(llf, 6, T)

    if verbose:
        print(f"    FIGARCH: d={d:.4f} φ={phi:.4f} β₁={beta1:.4f} "
              f"LLF={llf:.1f} BIC={bic:.1f}")

    return {'name': 'FIGARCH', 'omega': omega, 'a': a, 'gamma': gamma,
            'phi': phi, 'beta1': beta1, 'd': d,
            'kernel': g, 'llf': llf, 'bic': bic, 'k': 6}


# ══════════════════════════════════════════════════════════════
#  HYGARCH ESTIMATION
# ══════════════════════════════════════════════════════════════

def hygarch_estimate(r, garch_res=None, n_restarts=5, verbose=False):
    """HYGARCH(1,d,1): (1-τ)GARCH + τ·FIGARCH. 7 params."""
    T = len(r)

    def obj(x):
        omega = np.exp(x[0])
        a     = np.exp(x[1])
        gamma = np.exp(x[2])
        phi   = sigmoid(x[3])
        beta1 = sigmoid(x[4])
        d     = sigmoid(x[5])
        tau   = sigmoid(x[6])

        if phi <= beta1:
            return 1e12 + 1e6 * (beta1 - phi + 0.01) ** 2

        g, psi_hy = hygarch_kernel(phi, beta1, d, tau)
        a_eff = a * max(psi_hy[0], 1e-8)
        gamma_eff = gamma * max(psi_hy[0], 1e-8)
        P = (a_eff + gamma_eff / 2) * np.sum(g)
        if P >= 0.999:
            return 1e12 + 1e6 * (P - 0.999) ** 2

        sigma2 = archinf_filter(r, omega, a_eff, gamma_eff, g)
        nll = gaussian_nll(r, sigma2)

        n_neg = np.sum(psi_hy < -1e-8)
        if n_neg > 0:
            nll += 1e4 * np.sum(np.minimum(psi_hy, 0) ** 2)

        return nll if np.isfinite(nll) else 1e12

    if garch_res is not None:
        beta_g = garch_res['beta']
        phi_init = min(garch_res['a'] + garch_res['gamma'] / 2 + beta_g, 0.999)
        x0_g = np.array([
            np.log(max(garch_res['omega'], 1e-8)),
            np.log(max(garch_res['a'], 1e-8)),
            np.log(max(garch_res['gamma'], 1e-8)),
            np.log(phi_init / (1 - phi_init)),
            np.log(beta_g / (1 - beta_g)),
            -3.0,                                   # d ≈ 0.05
            -3.0,                                   # τ ≈ 0.05
        ])
    else:
        x0_g = np.array([np.log(0.05), np.log(0.03), np.log(0.07),
                          1.5, 2.0, -3.0, -3.0])

    inits = [x0_g.copy()]
    for d_l, t_l in [(-1.5, -1.0), (-0.5, 0.0), (0.0, 0.5)]:
        x = x0_g.copy()
        x[5], x[6] = d_l, t_l
        inits.append(x)
    for _ in range(max(0, n_restarts - 4)):
        x = x0_g.copy()
        x += np.random.randn(7) * 0.3
        inits.append(x)

    best = _run_restarts(obj, inits)
    if best is None:
        return {'name': 'HYGARCH', 'llf': -1e12, 'bic': 1e12, 'k': 7,
                'omega': np.nan, 'a': np.nan, 'gamma': np.nan,
                'phi': np.nan, 'beta1': np.nan, 'd': np.nan, 'tau': np.nan,
                'kernel': np.ones(J_MAX)}

    omega = np.exp(best.x[0])
    a     = np.exp(best.x[1])
    gamma = np.exp(best.x[2])
    phi   = sigmoid(best.x[3])
    beta1 = sigmoid(best.x[4])
    d     = sigmoid(best.x[5])
    tau   = sigmoid(best.x[6])
    g, _ = hygarch_kernel(phi, beta1, d, tau)
    llf = -best.fun
    bic = _compute_bic(llf, 7, T)

    if verbose:
        print(f"    HYGARCH: d={d:.4f} τ={tau:.4f} φ={phi:.4f} β₁={beta1:.4f} "
              f"LLF={llf:.1f} BIC={bic:.1f}")

    return {'name': 'HYGARCH', 'omega': omega, 'a': a, 'gamma': gamma,
            'phi': phi, 'beta1': beta1, 'd': d, 'tau': tau,
            'kernel': g, 'llf': llf, 'bic': bic, 'k': 7}


# ══════════════════════════════════════════════════════════════
#  ESTIMATE ALL 5 MODELS FOR ONE ASSET
# ══════════════════════════════════════════════════════════════

def estimate_all_models(r, ticker='', n_restarts=5, verbose=False):
    """Estimate GARCH + 4 alternatives. Returns comparison dict."""
    garch   = garch_estimate(r, n_restarts=n_restarts, verbose=False)
    search  = search_estimate(r, garch_res=garch, n_restarts=n_restarts, verbose=verbose)
    gmarch  = gmarch_estimate(r, garch_res=garch, n_restarts=n_restarts, verbose=verbose)
    fig     = figarch_estimate(r, garch_res=garch, n_restarts=n_restarts, verbose=verbose)
    hyg     = hygarch_estimate(r, garch_res=garch, n_restarts=n_restarts, verbose=verbose)

    models = {'GARCH': garch, 'SEARCH': search, 'GMARCH': gmarch,
              'FIGARCH': fig, 'HYGARCH': hyg}

    bics = {m: models[m]['bic'] for m in models}
    best_bic = min(bics, key=bics.get)

    llfs = {m: models[m]['llf'] for m in models}
    best_llf = max(llfs, key=llfs.get)

    return {
        'ticker': ticker,
        'models': models,
        'bics': bics,
        'llfs': llfs,
        'best_bic': best_bic,
        'best_llf': best_llf,
        'T': len(r),
    }
