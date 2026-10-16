import numpy as np
from numba import njit, types
from numba.typed import Dict



###############################################################################
# Distance-binned Ising prior for cis-eQTL fine-mapping, fit by collapsed tempered Gibbs + stochastic EM.
#
# Likelihood (per gene, SBayesRC-style low-rank transform, as in eb_coloc mcvc_tempered.py; single trait):
#   w = Q beta + eps,   eps ~ N(0, resid_var_g * I_k)
# where w = Lambda^{-1/2} U' b are the transformed marginal eQTL effects, Q = Lambda^{1/2} U' (k x n_snps).
#
# Prior on the causal indicators gamma of the SNPs of one gene (genes independent; SNPs within +/- 25 kb of the TSS):
#   p(gamma) propto exp( sum_i h[tss_bin(i)] gamma_i + sum_{i<j} J[pair_bin(i,j)] gamma_i gamma_j )
#   logit p(gamma_i = 1 | gamma_-i) = h[tss_bin(i)] + F_i,   F_i = sum_{j != i} J[pair_bin(i,j)] gamma_j
# tss_bin: |SNP - TSS| in TSS_BIN_EDGES bins; pair_bin: |pos_i - pos_j| in PAIR_BIN_EDGES bins (every pair of a gene has
# a bin, since the window is 50 kb wide). J = 0 is the independent TSS-distance-binned spike-and-slab.
#   beta_i | gamma_i = 1 ~ N(0, slab_var),  beta_i | gamma_i = 0 = 0
#
# Sampling: COLLAPSED tempered Gibbs (Zanella & Roberts 2019) over the gamma's of each gene: the effects are integrated
# out analytically, so the state is gamma alone. (The non-collapsed sampler of mcvc_tempered.py, which keeps beta in the
# state, gets trapped in over-fitted states of SNPs in strong LD with compensating effects: switching one off with the
# others' effects held fixed is never accepted. That gave badly over-called PIPs in simulations with strong eQTLs.)
# With S the active set, s = |S|, G = Q'Q, c = Q'w, lam = resid_var / slab_var and M = G_SS + lam I:
#   log p(w | S) = const - 0.5 (log|M| - s log lam) + c_S' M^{-1} c_S / (2 resid_var)
# and the log Bayes factor of SNP j being active vs not, given the other active SNPs, is
#   j not in S: a = M^{-1} G_Sj, m = G_jj + lam - G_Sj' a, r = c_j - a' c_S:    -0.5 log(m / lam) + r^2 / (2 resid_var m)
#   j in S:     m = 1 / (M^{-1})_jj, mu = M^{-1} c_S:                            -0.5 log(m / lam) + mu_j^2 m / (2 resid_var)
# (for S = {} this is the Wakefield ABF). Each sub-step picks SNP j with probability proportional to
# g_j(current) / P_j(current), g_j = (1 - eps) P_j + eps / 2, draws gamma_j from g_j, and after a flip M^{-1} and all
# conditionals are recomputed (O(n_snps s^2)). Expectations use 1 / Z importance weights and Rao-Blackwellized P_j.
#
# Learning (h, J): stochastic EM with the pseudo-likelihood M-step of MODEL_SPEC.md 4.1. Per gene and sweep, the 1 / Z
# importance-weighted Rao-Blackwellized pseudo-likelihood rows (target P_i = p(gamma_i = 1 | gamma_-i, w), features
# [onehot(tss_bin_i), n_i], n_ib = number of causal j != i in pair bin b) are aggregated over identical feature rows into
# exact sufficient statistics (sum of weights, sum of weight * P), normalized per gene, pooled over the last window_sweeps
# sweeps, and (h, J) are set to the maximizer of the penalized pseudo-likelihood
#   sum_r [ S_wP_r eta_r - S_w_r softplus(eta_r) ] - lambda_J / 2 |J|^2 - lambda_smooth / 2 sum_b (J_b - J_{b+1})^2
# Slab and residual variances are drawn from their scaled-inverse-chi-square conditionals given importance-weighted
# expected sufficient statistics (exact posterior moments of beta given gamma), as in mcvc_tempered.py.
###############################################################################

TSS_BIN_EDGES = np.array([500, 1000, 2500, 5000, 10000, 15000])                 # 7 bins of |SNP - TSS| (bp)
PAIR_BIN_EDGES = np.array([200, 500, 1000, 2000, 5000, 10000, 20000, 35000])     # 9 bins of |pos_i - pos_j| (bp)
KEY_BASE = 16  # pseudo-likelihood rows are keyed by n_ib in base 16 (at most 15 causal SNPs per gene and bin)
MIN_COND_PROB = 1e-250  # floor on P_j(current class) in the selection weights (keeps their sum finite)


def tss_bins(tss_distance):
    return np.searchsorted(TSS_BIN_EDGES, np.abs(tss_distance), side='right').astype(np.int64)


def pair_bins(positions):
    d = np.abs(positions[:, None] - positions[None, :])
    b = np.searchsorted(PAIR_BIN_EDGES, d, side='right').astype(np.int8)
    np.fill_diagonal(b, -1)
    return b


@njit
def _seed_numba(seed):
    np.random.seed(seed)


@njit
def _refresh_active(Gm, c, active, s, lam, Minv, mu):
    # M^{-1} = (G_SS + lam I)^{-1} and mu = M^{-1} c_S for the active set active[:s]
    if s == 0:
        return
    M = np.empty((s, s))
    for a in range(s):
        for b in range(s):
            M[a, b] = Gm[active[a], active[b]]
        M[a, a] += lam
    Mi = np.linalg.inv(M)
    for a in range(s):
        acc = 0.0
        for b in range(s):
            Minv[a, b] = Mi[a, b]
            acc += Mi[a, b] * c[active[b]]
        mu[a] = acc


@njit
def _all_conditionals(Gm, c, gamma, pos_in_active, active, s, Minv, mu, lam, resid_var, h_snp, F, P, p_sel,
                      one_minus_eps, eps_over_2):
    # Collapsed full conditionals p(gamma_j = 1 | gamma_-j, w) of every SNP -> P, and selection weights -> p_sel
    n_snps = Gm.shape[0]
    log_lam = np.log(lam)
    a = np.empty(s)
    for j in range(n_snps):
        if gamma[j] == 1:
            idx = pos_in_active[j]
            m = 1.0 / Minv[idx, idx]
            log_bf = -0.5 * (np.log(m) - log_lam) + mu[idx] * mu[idx] * m / (2.0 * resid_var)
        else:
            m = Gm[j, j] + lam
            r = c[j]
            for u in range(s):
                acc = 0.0
                for v in range(s):
                    acc += Minv[u, v] * Gm[active[v], j]
                a[u] = acc
            for u in range(s):
                m -= Gm[active[u], j] * a[u]
                r -= a[u] * c[active[u]]
            log_bf = -0.5 * (np.log(m) - log_lam) + r * r / (2.0 * resid_var * m)
        logit = h_snp[j] + F[j] + log_bf
        # P(on) and P(off) computed separately so that the probability of the current class keeps its precision
        if logit >= 0:
            e = np.exp(-logit)
            p_on = 1.0 / (1.0 + e)
            p_off = e / (1.0 + e)
        else:
            e = np.exp(logit)
            p_on = e / (1.0 + e)
            p_off = 1.0 / (1.0 + e)
        P[j] = p_on
        pc = p_on if gamma[j] == 1 else p_off
        if pc < MIN_COND_PROB:
            pc = MIN_COND_PROB
        p_sel[j] = one_minus_eps + eps_over_2 / pc


@njit
def _tgs_sweep_gene(Gm, c, ww, n_components, tss_bin, pair_bin, gamma, h, J, slab_var, resid_var,
                    n_sub_steps, epsilon, pl_stride, acc_P, acc_beta, stats, pl_sum_w, pl_sum_wp, pow_base, n_tss_bins,
                    store_gene, gene_key_offset, gene_pl_w, gene_pl_wp):
    # One collapsed tempered-Gibbs sweep (n_sub_steps sub-steps) over the SNPs of one gene; gamma is updated in place.
    # Outputs (1 / Z - weighted, NOT normalized): acc_P (Rao-Blackwellized PIPs), acc_beta (posterior means of beta given
    # gamma); stats: [0] flips, [1] sum of weights, [2] ESS of the weights, [3] weighted E[RSS | gamma],
    # [4] weighted E[sum beta^2 | gamma], [5] weighted number of active SNPs. The pseudo-likelihood rows of every
    # pl_stride-th sub-state are aggregated by key and added to pl_sum_w / pl_sum_wp normalized by this gene's recorded weight;
    # with store_gene they are also added to gene_pl_w / gene_pl_wp under gene_key_offset + key (for the gene bootstrap).
    n_snps = Gm.shape[0]
    eps_over_2 = epsilon / 2.0
    one_minus_eps = 1.0 - epsilon
    lam = resid_var / slab_var

    h_snp = np.zeros(n_snps)
    for j in range(n_snps):
        h_snp[j] = h[tss_bin[j]]
    active = np.zeros(n_snps, dtype=np.int64)
    pos_in_active = np.full(n_snps, -1, dtype=np.int64)
    s = 0
    for j in range(n_snps):
        if gamma[j] == 1:
            active[s] = j
            pos_in_active[j] = s
            s += 1
    F = np.zeros(n_snps)
    for cc in range(s):
        k = active[cc]
        for a in range(n_snps):
            if a != k:
                F[a] += J[pair_bin[a, k]]
    Minv = np.zeros((n_snps, n_snps))
    mu = np.zeros(n_snps)
    _refresh_active(Gm, c, active, s, lam, Minv, mu)
    P = np.zeros(n_snps)
    p_sel = np.zeros(n_snps)
    _all_conditionals(Gm, c, gamma, pos_in_active, active, s, Minv, mu, lam, resid_var, h_snp, F, P, p_sel, one_minus_eps, eps_over_2)

    for j in range(n_snps):
        acc_P[j] = 0.0
        acc_beta[j] = 0.0
    sum_w = 0.0
    sum_w_sq = 0.0
    acc_rss = 0.0
    acc_sumsq = 0.0
    acc_nact = 0.0
    n_changes = 0
    loc_w = Dict.empty(key_type=types.int64, value_type=types.float64)
    loc_wp = Dict.empty(key_type=types.int64, value_type=types.float64)
    pl_total_w = 0.0

    for step in range(n_sub_steps):
        tot = 0.0
        for j in range(n_snps):
            tot += p_sel[j]
        w = n_snps / tot
        sum_w += w
        sum_w_sq += w * w
        for j in range(n_snps):
            acc_P[j] += w * P[j]
        # Exact posterior moments of beta_S given gamma: mean mu, covariance resid_var * M^{-1}
        sumsq = 0.0
        tr_Minv = 0.0
        cmu = 0.0
        quad = 0.0
        tr_GMinv = 0.0
        for u in range(s):
            acc_beta[active[u]] += w * mu[u]
            sumsq += mu[u] * mu[u]
            tr_Minv += Minv[u, u]
            cmu += c[active[u]] * mu[u]
            for v in range(s):
                quad += mu[u] * Gm[active[u], active[v]] * mu[v]
                tr_GMinv += Gm[active[u], active[v]] * Minv[v, u]
        acc_sumsq += w * (sumsq + resid_var * tr_Minv)
        acc_nact += w * s
        acc_rss += w * (ww - 2.0 * cmu + quad + resid_var * tr_GMinv)

        # Pseudo-likelihood rows of this sub-state, keyed by (tss bin, causal-neighbour counts per pair bin)
        if step % pl_stride == 0 and s < KEY_BASE:
            pl_total_w += w
            for a in range(n_snps):
                code = 0
                for cc in range(s):
                    k = active[cc]
                    if k != a:
                        code += pow_base[pair_bin[a, k]]
                key = tss_bin[a] + n_tss_bins * code
                if key in loc_w:
                    loc_w[key] += w
                    loc_wp[key] += w * P[a]
                else:
                    loc_w[key] = w
                    loc_wp[key] = w * P[a]

        # Informed choice of the SNP to update
        u = np.random.random() * tot
        acc = 0.0
        j = n_snps - 1
        for a in range(n_snps):
            acc += p_sel[a]
            if u < acc:
                j = a
                break

        # Tempered draw of gamma_j from g_j = (1 - eps) P_j + eps / 2
        g_new = 1 if np.random.random() < one_minus_eps * P[j] + eps_over_2 else 0
        if g_new == gamma[j]:
            continue
        n_changes += 1
        gamma[j] = g_new
        if g_new == 1:
            active[s] = j
            pos_in_active[j] = s
            s += 1
            sign = 1.0
        else:
            idx = pos_in_active[j]
            last = active[s - 1]
            active[idx] = last
            pos_in_active[last] = idx
            pos_in_active[j] = -1
            s -= 1
            sign = -1.0
        for a in range(n_snps):
            if a != j:
                F[a] += sign * J[pair_bin[a, j]]
        _refresh_active(Gm, c, active, s, lam, Minv, mu)
        _all_conditionals(Gm, c, gamma, pos_in_active, active, s, Minv, mu, lam, resid_var, h_snp, F, P, p_sel, one_minus_eps, eps_over_2)

    # Normalize this gene's pseudo-likelihood rows by its recorded weight (each gene contributes n_snps rows of unit weight)
    if pl_total_w > 0:
        for key in loc_w:
            v = loc_w[key] / pl_total_w
            vp = loc_wp[key] / pl_total_w
            if key in pl_sum_w:
                pl_sum_w[key] += v
                pl_sum_wp[key] += vp
            else:
                pl_sum_w[key] = v
                pl_sum_wp[key] = vp
            if store_gene:
                gkey = gene_key_offset + key
                if gkey in gene_pl_w:
                    gene_pl_w[gkey] += v
                    gene_pl_wp[gkey] += vp
                else:
                    gene_pl_w[gkey] = v
                    gene_pl_wp[gkey] = vp

    stats[0] = n_changes
    stats[1] = sum_w
    stats[2] = sum_w * sum_w / sum_w_sq
    stats[3] = acc_rss
    stats[4] = acc_sumsq
    stats[5] = acc_nact


@njit
def _dict_to_arrays(dw, dwp):
    n = len(dw)
    keys = np.zeros(n, dtype=np.int64)
    sw = np.zeros(n)
    swp = np.zeros(n)
    i = 0
    for k in dw:
        keys[i] = k
        sw[i] = dw[k]
        swp[i] = dwp[k]
        i += 1
    return keys, sw, swp


def _new_pl_dicts():
    return (Dict.empty(key_type=types.int64, value_type=types.float64),
            Dict.empty(key_type=types.int64, value_type=types.float64))


def _sample_scaled_inv_chi_square(df, scale, n_obs, sum_sq):
    shape = 0.5 * (df + n_obs)
    rate = 0.5 * (df * scale + sum_sq)
    return 1.0 / np.random.gamma(shape, 1.0 / rate)


def fit_pseudo_likelihood(X, S_w, S_wp, n_tss_bins, theta_init, lambda_J=1.0, lambda_smooth=1.0, lambda_h=1e-4, n_newton=50, tol=1e-8):
    # Maximize the penalized pseudo-likelihood over theta = [h (n_tss_bins), J (n_pair_bins)] by damped Newton.
    # X: n_rows x n_params feature rows; S_w, S_wp: summed weights and weight * target per row
    n_params = X.shape[1]
    n_pair = n_params - n_tss_bins
    Pen = np.zeros((n_params, n_params))
    Pen[:n_tss_bins, :n_tss_bins] = lambda_h * np.eye(n_tss_bins)
    Pen[n_tss_bins:, n_tss_bins:] += lambda_J * np.eye(n_pair)
    D = np.diff(np.eye(n_pair), axis=0)
    Pen[n_tss_bins:, n_tss_bins:] += lambda_smooth * D.T @ D

    def objective(theta):
        eta = X @ theta
        return np.sum(S_wp * eta - S_w * np.logaddexp(0.0, eta)) - 0.5 * theta @ Pen @ theta

    theta = theta_init.copy()
    obj = objective(theta)
    for _ in range(n_newton):
        eta = X @ theta
        mu = 1.0 / (1.0 + np.exp(-eta))
        grad = X.T @ (S_wp - S_w * mu) - Pen @ theta
        H = (X * (S_w * mu * (1.0 - mu))[:, None]).T @ X + Pen
        step = np.linalg.solve(H, grad)
        t = 1.0
        while True:
            theta_new = theta + t * step
            obj_new = objective(theta_new)
            if obj_new >= obj - 1e-12 or t < 1e-6:
                break
            t *= 0.5
        converged = abs(obj_new - obj) < tol * (1.0 + abs(obj))
        theta, obj = theta_new, obj_new
        if converged:
            break
    return theta


class IsingEQTLTempered(object):
    def __init__(self):
        pass

    def fit(self, Q_list, w_list, tss_distance_list, position_list,
            n_iters=600, burn_in=300, learn_J=True, h_update_start=5, J_update_start=50,
            slab_var_init=0.01, slab_var_df=1e-16, resid_var_df=1e-16, h_init=None, J_init=None,
            lambda_J=1.0, lambda_smooth=1.0, tgs_epsilon=0.5, tgs_sub_steps_per_snp=1.0, pl_stride=1,
            window_sweeps=10, resid_var_bounds=None, update_slab_var=True, update_prior=True, seed=None, verbose=True, verbose_every=10):
        # Q_list: per gene k x n_snps Q = Lambda^{1/2} U'; w_list: per gene length-k w = Lambda^{-1/2} U' b
        # tss_distance_list / position_list: per gene SNP - TSS (bp) and SNP position (bp), in the order of Q's columns
        # learn_J: False fixes J = 0 (the independent TSS-binned baseline)
        # update_slab_var / update_prior: False holds the slab variance / (h, J) at their initial values (for calibration checks)
        # resid_var_bounds: None (free per-gene residual variance, as in mcvc_tempered.py) or (lo, hi): every draw is clipped to
        #   [lo, hi], and lo == hi fixes it. With standardized eQTL sumstats the residual variance of w is (1 - cis h2) / N
        if seed is not None:
            np.random.seed(seed)
            _seed_numba(seed)
        self.n_tss_bins = len(TSS_BIN_EDGES) + 1
        self.n_pair_bins = len(PAIR_BIN_EDGES) + 1
        self.pow_base = np.asarray([KEY_BASE ** b for b in range(self.n_pair_bins)], dtype=np.int64)
        self.epsilon = float(tgs_epsilon)
        self.n_genes = len(Q_list)

        self.Gm, self.c, self.ww, self.tss_bin, self.pair_bin = [], [], np.zeros(self.n_genes), [], []
        self.n_snps, self.n_components, self.gamma = [], [], []
        self.resid_var = np.zeros(self.n_genes)
        self.resid_var_scale = np.zeros(self.n_genes)
        for g in range(self.n_genes):
            Q = np.asarray(Q_list[g], dtype=np.float64)
            w = np.asarray(w_list[g], dtype=np.float64)
            k, p = Q.shape
            self.Gm.append(np.ascontiguousarray(Q.T @ Q))
            self.c.append(np.ascontiguousarray(Q.T @ w))
            self.ww[g] = w @ w
            self.tss_bin.append(tss_bins(np.asarray(tss_distance_list[g])))
            self.pair_bin.append(pair_bins(np.asarray(position_list[g], dtype=np.int64)))
            self.n_snps.append(p)
            self.n_components.append(k)
            self.gamma.append(np.zeros(p, dtype=np.int64))
            self.resid_var[g] = np.mean(w ** 2)
            if resid_var_bounds is not None:
                self.resid_var[g] = np.clip(self.resid_var[g], resid_var_bounds[0], resid_var_bounds[1])
            self.resid_var_scale[g] = self.resid_var[g]
        self.slab_var = float(slab_var_init)
        self.slab_var_scale = float(slab_var_init)
        if h_init is None:
            h_init = np.full(self.n_tss_bins, np.log(1.0 / np.mean(self.n_snps)))
        self.h = np.asarray(h_init, dtype=np.float64).copy()
        self.J = np.zeros(self.n_pair_bins) if J_init is None else np.asarray(J_init, dtype=np.float64).copy()

        W = window_sweeps
        buf_pl = [None] * W
        buf_w = np.zeros((W, self.n_genes))
        buf_rss = np.zeros((W, self.n_genes))
        buf_sumsq = np.zeros((W, self.n_genes))
        buf_nact = np.zeros((W, self.n_genes))
        acc_P = [np.zeros(p) for p in self.n_snps]
        acc_beta = [np.zeros(p) for p in self.n_snps]
        pip_sum = [np.zeros(p) for p in self.n_snps]
        beta_sum = [np.zeros(p) for p in self.n_snps]
        weight_sum = np.zeros(self.n_genes)
        n_post = n_iters - burn_in
        self.sampled_h = np.zeros((n_post, self.n_tss_bins))
        self.sampled_J = np.zeros((n_post, self.n_pair_bins))
        self.sampled_slab_var = np.zeros(n_post)
        self.trace = []
        stats = np.zeros(6)
        theta = np.concatenate([self.h, self.J])
        self.learn_J = learn_J
        self.lambda_J = lambda_J
        self.lambda_smooth = lambda_smooth
        # Per-gene pseudo-likelihood rows summed over the retained sweeps, keyed by gene * key_space + key (gene bootstrap)
        self.key_space = self.n_tss_bins * KEY_BASE ** self.n_pair_bins
        gene_pl_w, gene_pl_wp = _new_pl_dicts()

        for itera in range(n_iters):
            slot = itera % W
            pl_w, pl_wp = _new_pl_dicts()
            changes, ess = 0.0, 0.0
            for g in np.random.permutation(self.n_genes):
                n_sub = max(1, int(round(tgs_sub_steps_per_snp * self.n_snps[g])))
                _tgs_sweep_gene(self.Gm[g], self.c[g], self.ww[g], self.n_components[g], self.tss_bin[g], self.pair_bin[g],
                                self.gamma[g], self.h, self.J, self.slab_var, self.resid_var[g],
                                n_sub, self.epsilon, pl_stride, acc_P[g], acc_beta[g], stats, pl_w, pl_wp,
                                self.pow_base, self.n_tss_bins, itera >= burn_in, g * self.key_space, gene_pl_w, gene_pl_wp)
                changes += stats[0]
                ess += stats[2]
                buf_w[slot, g] = stats[1]
                buf_rss[slot, g] = stats[3]
                buf_sumsq[slot, g] = stats[4]
                buf_nact[slot, g] = stats[5]
            buf_pl[slot] = _dict_to_arrays(pl_w, pl_wp)
            n_filled = min(itera + 1, W)
            total_w = np.sum(buf_w[:n_filled], axis=0)

            # Variances: draws given importance-weighted expected sufficient statistics pooled over the window
            exp_rss = np.sum(buf_rss[:n_filled], axis=0) / total_w
            exp_sumsq = np.sum(np.sum(buf_sumsq[:n_filled], axis=0) / total_w)
            exp_nact = np.sum(np.sum(buf_nact[:n_filled], axis=0) / total_w)
            if itera >= h_update_start:
                if update_slab_var and exp_nact > 0:
                    self.slab_var = _sample_scaled_inv_chi_square(slab_var_df, self.slab_var_scale, exp_nact, exp_sumsq)
                for g in range(self.n_genes):
                    self.resid_var[g] = _sample_scaled_inv_chi_square(resid_var_df, self.resid_var_scale[g], self.n_components[g], exp_rss[g])
                if resid_var_bounds is not None:
                    np.clip(self.resid_var, resid_var_bounds[0], resid_var_bounds[1], out=self.resid_var)

            # (h, J): pseudo-likelihood M-step on the window-pooled rows
            if update_prior and itera >= h_update_start:
                X, S_w, S_wp = self._pooled_pl_rows(buf_pl[:n_filled])
                if learn_J and itera >= J_update_start:
                    theta = fit_pseudo_likelihood(X, S_w, S_wp, self.n_tss_bins, theta, lambda_J=lambda_J, lambda_smooth=lambda_smooth)
                else:
                    th = fit_pseudo_likelihood(X[:, :self.n_tss_bins], S_w, S_wp, self.n_tss_bins, theta[:self.n_tss_bins], lambda_J=lambda_J, lambda_smooth=lambda_smooth)
                    theta = np.concatenate([th, np.zeros(self.n_pair_bins)])
                self.h = theta[:self.n_tss_bins].copy()
                self.J = theta[self.n_tss_bins:].copy()

            if itera >= burn_in:
                s = itera - burn_in
                self.sampled_h[s] = self.h
                self.sampled_J[s] = self.J
                self.sampled_slab_var[s] = self.slab_var
                for g in range(self.n_genes):
                    pip_sum[g] += acc_P[g]
                    beta_sum[g] += acc_beta[g]
                    weight_sum[g] += buf_w[slot, g]

            n_causal = np.sum(np.sum(buf_nact[slot] / buf_w[slot]))
            self.trace.append(np.concatenate([[itera, self.slab_var, n_causal], self.h, self.J]))
            if verbose and (itera % verbose_every == 0 or itera == n_iters - 1):
                print('iter %d: slab_var=%.3e mean_resid_var=%.3e E[n_causal]=%.1f changes/gene=%.1f ess=%.1f\n   h=%s\n   J=%s' % (
                    itera, self.slab_var, np.mean(self.resid_var), n_causal, changes / self.n_genes, ess / self.n_genes,
                    np.array2string(self.h, precision=2), np.array2string(self.J, precision=2)), flush=True)

        self.pip = [pip_sum[g] / weight_sum[g] for g in range(self.n_genes)]
        self.beta_posterior_mean = [beta_sum[g] / weight_sum[g] for g in range(self.n_genes)]
        self.h_mean = np.mean(self.sampled_h, axis=0)
        self.J_mean = np.mean(self.sampled_J, axis=0)
        self.trace = np.asarray(self.trace)
        gkeys, gsw, gswp = _dict_to_arrays(gene_pl_w, gene_pl_wp)
        self.gene_pl_gene = gkeys // self.key_space
        self.gene_pl_key = gkeys % self.key_space
        self.gene_pl_sw = gsw / n_post
        self.gene_pl_swp = gswp / n_post
        return self

    def bootstrap(self, n_bootstrap=200, seed=0):
        # Gene bootstrap of (h, J): resample genes with replacement and refit the penalized pseudo-likelihood on their pseudo-
        # likelihood rows averaged over the retained sweeps (the posterior is held fixed, so only the M-step is refit).
        # Returns n_bootstrap x (n_tss_bins + n_pair_bins)
        rng = np.random.default_rng(seed)
        uniq, inv = np.unique(self.gene_pl_key, return_inverse=True)
        X = self._decode_keys(uniq)
        theta0 = np.concatenate([self.h_mean, self.J_mean])
        out = np.zeros((n_bootstrap, len(theta0)))
        for b in range(n_bootstrap):
            mult = np.bincount(rng.integers(0, self.n_genes, self.n_genes), minlength=self.n_genes)[self.gene_pl_gene]
            S_w = np.bincount(inv, weights=self.gene_pl_sw * mult, minlength=len(uniq))
            S_wp = np.bincount(inv, weights=self.gene_pl_swp * mult, minlength=len(uniq))
            if self.learn_J:
                out[b] = fit_pseudo_likelihood(X, S_w, S_wp, self.n_tss_bins, theta0, lambda_J=self.lambda_J, lambda_smooth=self.lambda_smooth)
            else:
                out[b, :self.n_tss_bins] = fit_pseudo_likelihood(X[:, :self.n_tss_bins], S_w, S_wp, self.n_tss_bins, theta0[:self.n_tss_bins])
        return out

    def _pooled_pl_rows(self, bufs):
        # Merge the per-sweep aggregated rows (each sweep normalized per gene) and average over the window; decode the keys
        keys = np.concatenate([b[0] for b in bufs])
        sw = np.concatenate([b[1] for b in bufs])
        swp = np.concatenate([b[2] for b in bufs])
        uniq, inv = np.unique(keys, return_inverse=True)
        S_w = np.bincount(inv, weights=sw) / len(bufs)
        S_wp = np.bincount(inv, weights=swp) / len(bufs)
        return self._decode_keys(uniq), S_w, S_wp

    def _decode_keys(self, keys):
        # Pseudo-likelihood row key -> feature row [onehot(tss bin), causal-neighbour counts per pair bin]
        tb = keys % self.n_tss_bins
        code = keys // self.n_tss_bins
        X = np.zeros((len(keys), self.n_tss_bins + self.n_pair_bins))
        X[np.arange(len(keys)), tb] = 1.0
        for b in range(self.n_pair_bins):
            X[:, self.n_tss_bins + b] = (code // self.pow_base[b]) % KEY_BASE
        return X
