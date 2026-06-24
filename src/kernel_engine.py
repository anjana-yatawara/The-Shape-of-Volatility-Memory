"""
kernel_engine.py  --  ARCH(infinity) memory-kernel estimation, redo (Paper 1).

Model (GJR-type news impact, separable memory kernel):
    r_t = sigma_t eps_t,  E eps=0, E eps^2=1
    sigma^2_t = omega + sum_{j=1..J} g(j) * w_{t-j}
    w_t = c_pos * r_t^2 * 1{r_t>=0} + c_neg * r_t^2 * 1{r_t<0}     (both coeffs >=0)
    leverage gamma = c_neg - c_pos ;  symmetric persistence P = (c_pos+c_neg)/2 * sum_j g(j)

Kernel families (all normalized g(1)=1, so the *scale* lives in (c_pos,c_neg)):
    geometric (GARCH):   g(j) = beta^{j-1}                         [1 kernel param]
    spline (sieve):      g(j) = exp( S(log j) ), S natural cubic spline, S(log1)=0
                                  -> IDENTIFICATION: anchor S(0)=0 fixes the level,
                                     the K-1 free knot values carry the shape.
    searche (stretched): g(j) = exp(-c (j^alpha - 1)),  c>0, alpha>0  [2 params]
    figarch:             BBM fractional-difference psi-weights, normalized [3 params]

Design notes / fixes vs prior code:
  * Pre-sample news filled with the in-sample mean of w (backcast) -- NO full-sample
    variance leak into sigma^2.
  * Burn-in = J (documented), not 2J.
  * Robust QMLE sandwich vcov  V = J^{-1} I J^{-1} / T  (numerical score/hessian).
  * (c_pos,c_neg) >=0 via softplus; omega>0 via exp.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_fprime, approx_hess

# ---------- numerics ----------
def softplus(x):  return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
def inv_softplus(y):  # y>0 -> x with softplus(x)=y
    y = np.maximum(y, 1e-12);  return np.log(np.expm1(np.minimum(y, 30.0))) if y < 30 else y

# ---------- kernels (all return g[0..J-1] = g(1..J), g(1)=1) ----------
def kernel_geometric(beta, J):
    j = np.arange(J)
    return beta ** j

def kernel_spline(theta_free, knot_loglags, J):
    """theta_free: length K-1 values at knots 2..K; knot at j=1 anchored to 0."""
    log_knots = knot_loglags
    theta = np.empty(len(log_knots)); theta[0] = 0.0; theta[1:] = theta_free
    cs = CubicSpline(log_knots, theta, bc_type='natural')
    logj = np.log(np.arange(1, J + 1))
    s = cs(logj)
    s = np.clip(s, -50.0, 5.0)
    return np.exp(s)

ALPHA_MAX = 3.0          # SEARCH stretching exponent is not separately identified above this:
ALPHA_MIN = 0.02         # for alpha>~3 the kernel is a near-delta (memoryless) and the
                         # likelihood is flat in alpha (see fit_model bounds). Capping here
                         # removes the optimizer-overflow degeneracy (j**alpha -> inf) that
                         # produced absurd fits (e.g. alpha_hat=185).
def kernel_searche(c, alpha, J):
    j = np.arange(1, J + 1, dtype=float)
    # Guard the j**alpha overflow that yields degenerate near-delta kernels for large alpha.
    a = np.minimum(alpha, 50.0)
    arg = np.clip(c * (np.power(j, a) - 1.0), -700.0, 700.0)
    return np.exp(-arg)

def kernel_tfarch(d, lam, J):
    """Tempered-fractional kernel g(j) = j^{d-1} exp(-lam (j-1)), g(1)=1.
       lam=0, d in (0,1) -> pure power-law j^{d-1} (NON-summable, true long memory).
       lam>0             -> exponentially tempered -> SUMMABLE.  (== GMARCH, delta=d-1)
       Summability test: H0 lam=0 vs H1 lam>0."""
    j = np.arange(1, J + 1, dtype=float)
    return (j ** (d - 1.0)) * np.exp(-lam * (j - 1.0))

def kernel_tempfigarch(d, phi, beta1, lam, J):
    """Tempered FIGARCH: BBM FIGARCH weights * exp(-lam(j-1)), normalized g(1)=1.
       lam=0 -> pure FIGARCH (NON-summable); lam>0 -> tempered (SUMMABLE), HYGARCH-type.
       Shares FIGARCH's short-lag (BBM) richness; isolates the summability tempering."""
    g = kernel_figarch(d, phi, beta1, J)
    if g is None:
        return None
    j = np.arange(1, J + 1, dtype=float)
    return g * np.exp(-lam * (j - 1.0))

def kernel_figarch(d, phi, beta1, J):
    """BBM ARCH(inf) weights for FIGARCH(1,d,1); normalized to g(1)=1.
       lambda_1 = d - beta1 + phi ; delta_k fractional-diff weights of (1-L)^d."""
    # fractional difference weights delta_k, k=0..J (delta_0=1)
    delta = np.empty(J + 2); delta[0] = 1.0
    for k in range(1, J + 2):
        delta[k] = delta[k - 1] * (k - 1 - d) / k
    psi = np.zeros(J + 1)  # psi[1..J]
    psi[1] = phi - beta1 + d
    for k in range(2, J + 1):
        psi[k] = beta1 * psi[k - 1] + (phi * delta[k - 1] - delta[k])
    g = psi[1:J + 1].copy()
    if g[0] <= 0 or not np.all(np.isfinite(g)):
        return None
    g = g / g[0]
    g = np.clip(g, 0.0, None)
    return g

# ---------- filter ----------
def archinf_sigma2(r, omega, c_pos, c_neg, g, presample_w=None):
    """Vectorized ARCH(inf) filter. r: (T,) demeaned returns. g: g(1..J)."""
    T = len(r); J = len(g)
    w = np.where(r >= 0.0, c_pos, c_neg) * (r * r)
    if presample_w is None:
        presample_w = w.mean()
    Wp = np.concatenate([np.full(J, presample_w), w])           # length T+J
    # sigma2[t] = omega + sum_{j=1..J} g(j) w[t-j],  w[t-j] = Wp[J + t - j]
    # contrib[t] = sum_{i=0..J-1} g[i] * Wp[(J+t-1) - i] = conv(Wp,g)[J-1+t]
    conv = np.convolve(Wp, g)                                   # length (T+J)+J-1
    contrib = conv[J - 1: J - 1 + T]
    sigma2 = omega + contrib
    return np.maximum(sigma2, 1e-10)

def _neg_qll_core(r, sigma2, burn):
    ll = np.log(sigma2[burn:]) + (r[burn:] ** 2) / sigma2[burn:]
    return 0.5 * np.sum(ll)

# ---------- packing ----------
def _unpack_shared(p):
    omega = np.exp(np.clip(p[0], -30, 30))
    c_pos = softplus(p[1]); c_neg = softplus(p[2])
    return omega, c_pos, c_neg

def _stationarity_pen(c_pos, c_neg, g, cap=0.999):
    P = 0.5 * (c_pos + c_neg) * g.sum()
    return (1e4 * (P - cap) ** 2) if P >= cap else 0.0, P

# ---------- objective builders ----------
def make_obj_spline(r, knot_loglags, J, lam, burn, presample_w):
    K = len(knot_loglags)
    def nll(p):
        omega, c_pos, c_neg = _unpack_shared(p)
        theta_free = p[3:3 + (K - 1)]
        g = kernel_spline(theta_free, knot_loglags, J)
        sigma2 = archinf_sigma2(r, omega, c_pos, c_neg, g, presample_w)
        val = _neg_qll_core(r, sigma2, burn)
        pen_s, _ = _stationarity_pen(c_pos, c_neg, g)
        # ridge penalty on 2nd differences of knot values (smoothness); lam>=0
        theta = np.concatenate([[0.0], theta_free])
        if lam > 0 and len(theta) >= 3:
            d2 = np.diff(theta, 2)
            pen_r = lam * np.sum(d2 ** 2)
        else:
            pen_r = 0.0
        return val + pen_s + pen_r
    return nll, K

def make_obj_geometric(r, J, burn, presample_w):
    def nll(p):
        omega, c_pos, c_neg = _unpack_shared(p)
        beta = 1.0 / (1.0 + np.exp(-p[3]))         # logistic in (0,1)
        g = kernel_geometric(beta, J)
        sigma2 = archinf_sigma2(r, omega, c_pos, c_neg, g, presample_w)
        val = _neg_qll_core(r, sigma2, burn)
        pen_s, _ = _stationarity_pen(c_pos, c_neg, g)
        return val + pen_s
    return nll

def make_obj_searche(r, J, burn, presample_w):
    def nll(p):
        omega, c_pos, c_neg = _unpack_shared(p)
        c = softplus(p[3]); alpha = softplus(p[4])
        g = kernel_searche(c, alpha, J)
        sigma2 = archinf_sigma2(r, omega, c_pos, c_neg, g, presample_w)
        val = _neg_qll_core(r, sigma2, burn)
        pen_s, _ = _stationarity_pen(c_pos, c_neg, g)
        return val + pen_s
    return nll

def make_obj_tfarch(r, J, burn, presample_w, fix_lam0=False):
    def nll(p):
        omega, c_pos, c_neg = _unpack_shared(p)
        d = p[3]                                   # free real (long memory d in (0,1))
        lam = 0.0 if fix_lam0 else softplus(p[4])
        g = kernel_tfarch(d, lam, J)
        if not np.all(np.isfinite(g)): return 1e10
        sigma2 = archinf_sigma2(r, omega, c_pos, c_neg, g, presample_w)
        val = _neg_qll_core(r, sigma2, burn)
        pen_s, _ = _stationarity_pen(c_pos, c_neg, g)
        return val + pen_s
    return nll

def make_obj_figarch(r, J, burn, presample_w):
    def nll(p):
        omega, c_pos, c_neg = _unpack_shared(p)
        d = 1.0 / (1.0 + np.exp(-p[3]))            # d in (0,1)
        phi = 1.0 / (1.0 + np.exp(-p[4]))          # phi in (0,1)
        beta1 = 1.0 / (1.0 + np.exp(-p[5]))        # beta1 in (0,1)
        g = kernel_figarch(d, phi, beta1, J)
        if g is None:
            return 1e10
        sigma2 = archinf_sigma2(r, omega, c_pos, c_neg, g, presample_w)
        val = _neg_qll_core(r, sigma2, burn)
        pen_s, _ = _stationarity_pen(c_pos, c_neg, g)
        return val + pen_s
    return nll

def make_obj_tempfigarch(r, J, burn, presample_w):
    def nll(p):
        omega, c_pos, c_neg = _unpack_shared(p)
        d = 1.0 / (1.0 + np.exp(-p[3])); phi = 1.0 / (1.0 + np.exp(-p[4]))
        beta1 = 1.0 / (1.0 + np.exp(-p[5])); lam = softplus(p[6])
        g = kernel_tempfigarch(d, phi, beta1, lam, J)
        if g is None or not np.all(np.isfinite(g)):
            return 1e10
        sigma2 = archinf_sigma2(r, omega, c_pos, c_neg, g, presample_w)
        val = _neg_qll_core(r, sigma2, burn)
        pen_s, _ = _stationarity_pen(c_pos, c_neg, g)
        return val + pen_s
    return nll

# ---------- fit driver ----------
def _garch_init(r):
    v = np.var(r)
    return np.array([np.log(0.05 * v), inv_softplus(0.03), inv_softplus(0.09)])

def fit_model(r, model, knot_loglags=None, J=252, lam=0.0, n_restarts=4, seed=0, p0_override=None):
    rng = np.random.default_rng(seed)
    burn = J
    presample_w = (np.where(r >= 0, 0.06, 0.06) * r * r).mean()  # neutral backcast scale
    base = _garch_init(r)
    if model == 'geometric':
        nll = make_obj_geometric(r, J, burn, presample_w); p0 = np.concatenate([base, [2.7]])  # beta~0.94
    elif model == 'searche':
        nll = make_obj_searche(r, J, burn, presample_w); p0 = np.concatenate([base, [inv_softplus(0.06), inv_softplus(1.0)]])
    elif model == 'figarch':
        nll = make_obj_figarch(r, J, burn, presample_w); p0 = np.concatenate([base, [0.0, 0.3, 0.3]])
    elif model == 'tfarch':
        nll = make_obj_tfarch(r, J, burn, presample_w); p0 = np.concatenate([base, [0.5, inv_softplus(0.05)]])
    elif model == 'tfarch_lam0':
        nll = make_obj_tfarch(r, J, burn, presample_w, fix_lam0=True); p0 = np.concatenate([base, [0.3, 0.0]])
    elif model == 'tempfigarch':
        nll = make_obj_tempfigarch(r, J, burn, presample_w); p0 = np.concatenate([base, [0.0, 0.3, 0.3, inv_softplus(0.02)]])
    elif model == 'spline':
        nll, K = make_obj_spline(r, knot_loglags, J, lam, burn, presample_w)
        beta0 = 0.94
        # init spline knots near geometric kernel: log g(k)=(k-1)log beta0
        knots = np.exp(knot_loglags)
        theta0 = (knots[1:] - 1.0) * np.log(beta0)
        p0 = np.concatenate([base, theta0])
    else:
        raise ValueError(model)

    if p0_override is not None:
        p0 = np.asarray(p0_override, float)

    # Box constraints: cap the SEARCH stretching exponent alpha=softplus(p[4]) to
    # [ALPHA_MIN, ALPHA_MAX] so the optimizer cannot wander into the flat, unidentified
    # super-exponential region (which overflowed j**alpha and produced alpha_hat>>1).
    bounds = None
    if model == 'searche':
        bounds = [(None, None)] * len(p0)
        bounds[4] = (inv_softplus(ALPHA_MIN), inv_softplus(ALPHA_MAX))

    best = None
    for it in range(n_restarts):
        pp = p0 if it == 0 else p0 + rng.normal(0, 0.4, size=p0.shape)
        if bounds is not None:
            lo, hi = bounds[4]
            pp[4] = min(max(pp[4], lo), hi)
        try:
            res = minimize(nll, pp, method='L-BFGS-B', bounds=bounds,
                           options=dict(maxiter=2000, ftol=1e-11, gtol=1e-7))
            if res.success or np.isfinite(res.fun):
                if best is None or res.fun < best.fun:
                    best = res
        except Exception:
            continue
    if best is None:
        return None
    return _finalize(r, model, best, knot_loglags, J, burn, presample_w, lam)

def _finalize(r, model, res, knot_loglags, J, burn, presample_w, lam):
    p = res.x
    omega, c_pos, c_neg = _unpack_shared(p)
    if model == 'geometric':
        beta = 1/(1+np.exp(-p[3])); g = kernel_geometric(beta, J); kp = dict(beta=beta)
    elif model == 'searche':
        c = softplus(p[3]); alpha = softplus(p[4]); g = kernel_searche(c, alpha, J); kp = dict(c=c, alpha=alpha)
    elif model == 'figarch':
        d=1/(1+np.exp(-p[3])); phi=1/(1+np.exp(-p[4])); b1=1/(1+np.exp(-p[5]))
        g = kernel_figarch(d, phi, b1, J); kp = dict(d=d, phi=phi, beta1=b1)
    elif model in ('tfarch', 'tfarch_lam0'):
        d = p[3]; lam = 0.0 if model == 'tfarch_lam0' else softplus(p[4])
        g = kernel_tfarch(d, lam, J); kp = dict(d=d, lam=lam)
    elif model == 'tempfigarch':
        d=1/(1+np.exp(-p[3])); phi=1/(1+np.exp(-p[4])); b1=1/(1+np.exp(-p[5])); lam=softplus(p[6])
        g = kernel_tempfigarch(d, phi, b1, lam, J); kp = dict(d=d, phi=phi, beta1=b1, lam=lam)
    elif model == 'spline':
        K = len(knot_loglags); g = kernel_spline(p[3:3+K-1], knot_loglags, J); kp = dict(theta_free=p[3:3+K-1])
    sigma2 = archinf_sigma2(r, omega, c_pos, c_neg, g, presample_w)
    llf = -_neg_qll_core(r, sigma2, burn)
    k_params = len(p)
    Teff = len(r) - burn
    out = dict(model=model, params=p, omega=omega, c_pos=c_pos, c_neg=c_neg,
               gamma=c_neg - c_pos, g=g, sigma2=sigma2, llf=llf, k=k_params,
               Teff=Teff, P=0.5*(c_pos+c_neg)*g.sum(), sum_g=g.sum(),
               aic=2*k_params - 2*llf, bic=np.log(Teff)*k_params - 2*llf, **kp)
    return out

# ---------- robust sandwich vcov (for joint inference, incl. alpha & functionals) ----------
def sandwich_vcov(r, model, p, knot_loglags=None, J=252, burn=None, presample_w=None, lam=0.0):
    """Robust QMLE sandwich  V = A^{-1} B A^{-1},  A = full Hessian of negll, B = sum of
    per-obs score outer products. Efficient: per-obs scores via 2*dim filter evals."""
    if burn is None: burn = J
    if presample_w is None: presample_w = (0.06 * r * r).mean()
    T = len(r); dim = len(p)
    def per_obs(p_):
        omega, c_pos, c_neg = _unpack_shared(p_)
        g = _g_from_params(model, p_, knot_loglags, J)
        if g is None or not np.all(np.isfinite(g)): return None
        s2 = archinf_sigma2(r, omega, c_pos, c_neg, g, presample_w)
        return 0.5 * (np.log(s2[burn:]) + r[burn:] ** 2 / s2[burn:])
    h = 1e-4 * np.maximum(np.abs(p), 1.0)
    n = T - burn
    scores = np.zeros((n, dim))
    for k in range(dim):
        pp = p.copy(); pp[k] += h[k]; fp = per_obs(pp)
        pm = p.copy(); pm[k] -= h[k]; fm = per_obs(pm)
        if fp is None or fm is None: return None
        scores[:, k] = (fp - fm) / (2 * h[k])           # d(negll_t)/dp_k
    B = scores.T @ scores
    def full(p_):
        v = per_obs(p_); return float(np.sum(v)) if v is not None else 1e10
    H = approx_hess(p, full)
    try:
        Ainv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Ainv = np.linalg.pinv(H)
    return Ainv @ B @ Ainv

def _g_from_params(model, p, knot_loglags, J):
    if model == 'geometric':
        return kernel_geometric(1/(1+np.exp(-p[3])), J)
    if model == 'searche':
        return kernel_searche(softplus(p[3]), softplus(p[4]), J)
    if model == 'figarch':
        return kernel_figarch(1/(1+np.exp(-p[3])), 1/(1+np.exp(-p[4])), 1/(1+np.exp(-p[5])), J)
    if model == 'tfarch':
        return kernel_tfarch(p[3], softplus(p[4]), J)
    if model == 'tfarch_lam0':
        return kernel_tfarch(p[3], 0.0, J)
    if model == 'tempfigarch':
        return kernel_tempfigarch(1/(1+np.exp(-p[3])), 1/(1+np.exp(-p[4])), 1/(1+np.exp(-p[5])), softplus(p[6]), J)
    if model == 'spline':
        K = len(knot_loglags); return kernel_spline(p[3:3+K-1], knot_loglags, J)

# ---------- functionals ----------
def half_life(g):
    below = np.where(g <= 0.5)[0]
    return int(below[0] + 1) if len(below) else len(g)

DEFAULT_KNOTS_8 = np.log(np.array([1, 2, 5, 10, 21, 63, 126, 252], dtype=float))

# ---------- simulator (for Monte Carlo + bootstrap) ----------
def simulate_archinf(g, omega, c_pos, c_neg, T, rng, burn=600, eps=None, df=None):
    J = len(g); n = T + burn
    if eps is None:
        eps = rng.standard_t(df, n) / np.sqrt(df/(df-2)) if df else rng.standard_normal(n)
    P = 0.5 * (c_pos + c_neg) * g.sum()
    sig2_bar = omega / max(1e-6, 1.0 - P)
    wbar = 0.5 * (c_pos + c_neg) * sig2_bar
    w = np.full(n + J, wbar)            # w[J:] are the realized news; w[:J] presample
    r = np.zeros(n)
    gv = g[::-1]
    for t in range(n):
        contrib = np.dot(gv, w[t:t + J])       # sum_{j=1..J} g(j) w[t+J-j]
        s2 = omega + contrib
        s2 = max(s2, 1e-10)
        rt = np.sqrt(s2) * eps[t]
        r[t] = rt
        w[t + J] = (c_pos if rt >= 0 else c_neg) * rt * rt
    return r[burn:]
