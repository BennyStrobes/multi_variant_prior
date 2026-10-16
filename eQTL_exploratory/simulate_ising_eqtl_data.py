import numpy as np
from numba import njit
from ising_eqtl_tempered import tss_bins, pair_bins, TSS_BIN_EDGES, PAIR_BIN_EDGES



###############################################################################
# Small-scale simulation for the distance-binned Ising eQTL prior (no real genotypes needed).
#
# Per gene:
#   - SNP positions uniform in TSS +/- 25 kb (Poisson number of SNPs);
#   - genotypes: 2N haplotypes, each a Li-Stephens-style mosaic of n_founders founder haplotypes, switching founder with
#     probability 1 - exp(-d / switch_length_bp) between adjacent SNPs, with a small copying error. This gives LD that is
#     high within a few hundred bp and decays over ~switch_length_bp. SNPs with MAF < min_maf are dropped;
#   - causal indicators drawn from the Ising prior (true h, J) by Gibbs sampling on the prior;
#   - effects beta_j ~ N(0, slab_var) on the standardized-genotype scale; y = X beta + eps, var(y) = 1;
#   - marginal effects b = X'y / N on standardized x and y, in-sample LD R = X'X / N, and the low-rank eigen transform of
#     eb_coloc (99.5% of the trace): Q = Lambda^{1/2} U', w = Lambda^{-1/2} U' b.
###############################################################################

N_TSS_BINS = len(TSS_BIN_EDGES) + 1
N_PAIR_BINS = len(PAIR_BIN_EDGES) + 1


def eigen_decompose_ld(ld_mat, variance_explained_threshold=0.995):
    # (Copied from eb_coloc preprocess_colocalization_data.py)
    eigenvalues, eigenvectors = np.linalg.eigh(ld_mat)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    zero_tolerance = eigenvalues[0] * ld_mat.shape[0] * np.finfo(eigenvalues.dtype).eps
    eigenvalues[eigenvalues < zero_tolerance] = 0.0
    cumulative_variance_explained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    n_components = int(np.searchsorted(cumulative_variance_explained, variance_explained_threshold)) + 1
    n_components = min(n_components, int(np.sum(eigenvalues > 0.0)))
    eigenvalues = eigenvalues[:n_components]
    U = eigenvectors[:, :n_components]
    Q = np.sqrt(eigenvalues)[:, np.newaxis] * U.T
    lambda_inv_sqrt_Ut = (1.0 / np.sqrt(eigenvalues))[:, np.newaxis] * U.T
    return eigenvalues, Q, lambda_inv_sqrt_Ut


@njit
def _mosaic_haplotypes(founders, switch_prob, n_haps, error_rate):
    n_founders, n_snps = founders.shape
    H = np.zeros((n_haps, n_snps), dtype=np.int8)
    for hh in range(n_haps):
        f = np.random.randint(n_founders)
        for j in range(n_snps):
            if j > 0 and np.random.random() < switch_prob[j]:
                f = np.random.randint(n_founders)
            a = founders[f, j]
            if np.random.random() < error_rate:
                a = 1 - a
            H[hh, j] = a
    return H


@njit
def _sample_ising_prior(h_snp, J, pair_bin, n_sweeps):
    # Random-scan Gibbs on p(gamma) propto exp(sum h gamma + sum_{i<j} J[bin] gamma_i gamma_j)
    p = h_snp.shape[0]
    gamma = np.zeros(p, dtype=np.int64)
    F = np.zeros(p)
    for _ in range(n_sweeps * p):
        i = np.random.randint(p)
        logit = h_snp[i] + F[i]
        g_new = 1 if np.random.random() < 1.0 / (1.0 + np.exp(-logit)) else 0
        if g_new != gamma[i]:
            sign = 1.0 if g_new == 1 else -1.0
            for a in range(p):
                if a != i:
                    F[a] += sign * J[pair_bin[a, i]]
            gamma[i] = g_new
    return gamma


@njit
def _seed(seed):
    np.random.seed(seed)


def coalescent_founders(rng, n_founders, n_snps):
    # Founder alleles from one Kingman genealogy: each SNP is a mutation on a branch chosen proportional to branch length, so
    # the founders carrying it form a clade. SNPs on the same branch are in perfect LD among founders and nested clades in
    # partial LD; the mosaic copying then breaks LD down with distance.
    clades = [frozenset([i]) for i in range(n_founders)]
    birth = {c: 0.0 for c in clades}
    branches, lengths = [], []
    t = 0.0
    active = list(clades)
    while len(active) > 1:
        k = len(active)
        t += rng.exponential(1.0 / (k * (k - 1) / 2.0))
        i, j = sorted(rng.choice(k, size=2, replace=False))
        a, b = active[i], active[j]
        for c in (a, b):
            branches.append(c)
            lengths.append(t - birth[c])
        merged = a | b
        birth[merged] = t
        active = [c for idx, c in enumerate(active) if idx not in (i, j)] + [merged]
    lengths = np.asarray(lengths)
    picks = rng.choice(len(branches), size=n_snps, p=lengths / lengths.sum())
    founders = np.zeros((n_founders, n_snps), dtype=np.int8)
    for s, bidx in enumerate(picks):
        founders[list(branches[bidx]), s] = 1
    return founders


def simulate_gene(rng, h_true, J_true, n_samples, mean_n_snps, slab_var, cis_window=25000, n_founders=20,
                  switch_length_bp=15000.0, error_rate=0.005, min_maf=0.01, min_snps=25):
    while True:
        n_snps = rng.poisson(mean_n_snps)
        tss_dist = np.sort(rng.integers(-cis_window, cis_window + 1, size=n_snps))
        tss_dist = np.unique(tss_dist)
        founders = coalescent_founders(rng, n_founders, len(tss_dist))
        switch_prob = np.zeros(len(tss_dist))
        switch_prob[1:] = 1.0 - np.exp(-np.diff(tss_dist) / switch_length_bp)
        H = _mosaic_haplotypes(founders, switch_prob, 2 * n_samples, error_rate)
        G = (H[:n_samples] + H[n_samples:]).astype(np.float64)
        af = G.mean(axis=0) / 2.0
        keep = np.minimum(af, 1 - af) >= min_maf
        if keep.sum() >= min_snps:
            break
    G = G[:, keep]
    tss_dist = tss_dist[keep]
    X = (G - G.mean(axis=0)) / G.std(axis=0)

    pb = pair_bins(tss_dist)
    h_snp = h_true[tss_bins(tss_dist)]
    gamma = _sample_ising_prior(h_snp, J_true, pb, 200)
    beta = np.zeros(len(tss_dist))
    beta[gamma == 1] = rng.normal(0.0, np.sqrt(slab_var), size=int(gamma.sum()))
    genetic = X @ beta
    var_g = np.var(genetic)
    if var_g > 0.9:
        genetic *= np.sqrt(0.9 / var_g)
        beta *= np.sqrt(0.9 / var_g)
        var_g = 0.9
    y = genetic + rng.normal(0.0, np.sqrt(1.0 - var_g), size=n_samples)
    y = (y - y.mean()) / y.std()
    b = X.T @ y / n_samples
    R = X.T @ X / n_samples
    _, Q, lam_inv_sqrt_Ut = eigen_decompose_ld(R)
    w = lam_inv_sqrt_Ut @ b
    return dict(Q=Q, w=w, tss_distance=tss_dist, position=tss_dist.astype(np.int64), gamma=gamma, beta=beta,
                z=b * np.sqrt(n_samples), R=R)


def simulate(n_genes, h_true, J_true, n_samples=800, mean_n_snps=120, slab_var=0.02, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    _seed(seed)
    return [simulate_gene(rng, np.asarray(h_true, float), np.asarray(J_true, float), n_samples, mean_n_snps, slab_var, **kwargs)
            for _ in range(n_genes)]
