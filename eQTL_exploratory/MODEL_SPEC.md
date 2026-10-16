# Attention-coupled causal prior for genome-wide fine-mapping — model spec

Status: design spec, nothing implemented yet. This file is the source of truth for the model;
code in this directory should follow it, and the spec should be updated if the design changes.

## 1. Goal

Genome-wide spike-and-slab fine-mapping in which the prior probability that a variant is causal
depends on whether *nearby* variants are causal. Which variants influence each other ("attend to"
each other) is learned from functional features via an attention-style bilinear score, with
parameters shared across the whole genome.

Key design principles:

- **LD enters only through the likelihood. Biological co-causality enters only through the prior.**
  Attention inputs must never include genotypes, LD, or z-scores.
- The prior is a coherent joint distribution (pairwise Ising / MRF) — hence the coupling matrix
  `J` must be symmetric and there is no softmax.
- With `J ≡ 0` the model reduces exactly to an independent functional-prior spike-and-slab
  (the baseline). Always initialise at `J = 0`.
- Inference uses an existing genome-wide spike-and-slab + tempered Gibbs sampler
  (Zanella & Roberts-style tempered Gibbs). **Not SuSiE.** The model only changes the prior
  log-odds fed into the sampler's full conditionals.

## 2. Notation and shapes

| Symbol | Shape | Meaning |
|---|---|---|
| `M` | scalar | number of SNPs genome-wide (~1e7) |
| `a_i` | `(F,)` | feature vector for SNP i (annotations, conservation, frozen sequence-model embeddings) |
| `pos_i` | scalar | genomic position (only used within-chromosome, as differences) |
| `N(i)` | set | neighbours of i: SNPs on same chromosome with `|pos_i - pos_j| < w` (e.g. `w = 50kb`), `j != i`. Defined by distance only, never LD |
| `c_ij` | `(P,)` | pair features: same cCRE/peak, same TAD, Hi-C contact, distance bins, ... |
| `e_i` | `(D,)` | SNP embedding |
| `q_i` | `(d,)` | query/key projection (shared, since J is symmetric) |
| `h_i` | scalar | unary prior log-odds |
| `J_ij` | scalar, sparse `M x M` | pairwise coupling, symmetric, zero outside `N(i)` |
| `gamma_i` | {0,1} | causal indicator |
| `beta_i` | scalar | effect size |
| `z` | `(M,)` | GWAS z-scores |
| `R` | block-diagonal | LD matrix (blockwise) |
| `n` | scalar | GWAS sample size |

## 3. Generative model

### 3.1 Embeddings (contextual)

```
e_i = f_theta( {a_j, pos_j - pos_i : j in window(i)} )_i
```

`f_theta` is a small transformer encoder over the local window (features only), or an MLP on
`a_i` as the simplest version. Context matters: it lets `h_i` absorb pair-specific
"neighbour-of-j" effects (see §6), and lets coupling depend on the surrounding regulatory context.

### 3.2 Unary prior log-odds

```
h_i = w^T e_i + b
```

### 3.3 Pairwise coupling (the attention part)

```
q_i     = W_q e_i                                    # (d,)
ell_ij  = softplus( v^T (e_i + e_j) ) + ell_min      # feature-dependent length scale, symmetric
kappa_ij = exp( -|pos_i - pos_j| / ell_ij ) * 1[ j in N(i) ]

J_ij = sum_{heads k=1..H} [ q_i^(k)T Omega^(k) q_j^(k) / sqrt(d) ] * kappa_ij^(k)
       + u^T c_ij * 1[ j in N(i) ]
J_ii = 0
```

- `Omega^(k)` is **symmetric** (parameterise as `(A + A^T)/2`); it may be indefinite, so `J`
  can be positive (co-causality / clustering) or negative (redundancy / competition).
- **No softmax.** `J` enters the log-odds directly, so distance must act *multiplicatively*
  (`kappa`) to drive coupling to 0; an additive distance bias (ALiBi-style) would be wrong here.
- Multiple heads (H = 2–4) let separate mechanisms (shared element, motif cooperativity,
  enhancer–promoter loops) be learned and interpreted separately.
- `u^T c_ij` handles inherently pairwise facts (same element, Hi-C contact) that a bilinear
  form of per-SNP features represents poorly. Treat it as essential.
- Optional bound on magnitude for sampler mixing: `J_ij <- J_max * tanh(J_ij / J_max)`.

### 3.4 Prior on causal indicators (Ising / MRF)

```
p(gamma | theta) = (1/Z(theta)) * exp( sum_i h_i gamma_i + sum_{i<j} J_ij gamma_i gamma_j )
```

Equivalent conditional form (what the sampler uses):

```
logit p(gamma_i = 1 | gamma_{-i}) = h_i + sum_{j in N(i)} J_ij gamma_j
```

`exp(J_ij)` is the prior odds ratio of co-causality:
`p(1,1) p(0,0) / (p(1,0) p(0,1))`. The pairwise binary table has 3 free parameters
(`h_i`, `h_j`, `J_ij`), so this is fully general at the pair level.

### 3.5 Effect sizes (spike and slab)

```
beta_i | gamma_i ~ gamma_i * N(0, sigma2_beta_i) + (1 - gamma_i) * delta_0
sigma2_beta_i = sigma2_beta                  # v1
sigma2_beta_i = exp(r^T e_i)                 # optional: annotation-dependent slab
```

### 3.6 Likelihood (summary statistics, RSS)

```
z | beta ~ N( sqrt(n) R beta , R )
```

(Individual-level alternative: `y = X beta + eps`, `eps ~ N(0, sigma2_e I)`.)

### 3.7 Parameters

- Shared, learned genome-wide: `theta = {f_theta, w, b, W_q, Omega^(k), v, ell_min, u, sigma2_beta (or r)}`.
- Latent: `gamma`, `beta`.

## 4. Learning (Monte Carlo / stochastic-approximation EM)

Objective: maximise `log p(z | theta)`. The M-step only involves the prior:
`theta <- argmax E_{p(gamma | z, theta_old)}[ log p(gamma | theta) ]`.
`log Z(theta)` is intractable, so replace `log p(gamma | theta)` with the **pseudo-likelihood**.

### 4.1 M-step loss

```
eta_i^(t) = h_i + sum_{j in N(i)} J_ij gamma_j^(t)

L(theta) = - sum_t w^(t) sum_i [ pi_i^(t) * eta_i^(t) - softplus(eta_i^(t)) ]
           + lambda_J * ||J||^2  (+ weight decay on theta)
```

- `gamma^(t)`: posterior samples from the tempered Gibbs sampler.
- `w^(t)`: tempered-Gibbs importance weights.
- `pi_i^(t) = p(gamma_i = 1 | gamma_{-i}^(t), z)`: Rao–Blackwellised target (the sampler already
  computes all full conditionals); use instead of the raw `gamma_i^(t)`.
- This is a per-SNP logistic regression of "is SNP i causal" on its features + attention-weighted
  causal neighbours. Differentiable end to end.

### 4.2 Algorithm

```
init: J = 0; fit h with J = 0 (or from an existing functional prior); burn in sampler
repeat until converged:
  1. SAMPLE:  run S tempered-Gibbs sweeps genome-wide under current (h, J), LD blocks in parallel.
              Keep chain state (never restart). Store thinned samples sparsely:
              causal indices, pi_i for SNPs with pi_i > eps, weight w^(t). Push to rolling buffer.
  2. LEARN:   K SGD steps on L(theta) using minibatches of (SNP, sample) pairs from the buffer.
  3. REFRESH: recompute h_i for all SNPs; invalidate cached J rows (recomputed lazily).
  4. ANNEAL:  Robbins–Monro step sizes; buffer forgets old samples.
also update sigma2_beta (or r) from beta samples in the M-step.
```

### 4.3 Sampler integration

The only change to the existing sampler is its prior log-odds:

```
prior_logit_i = h_i + F_i,   where   F_i = sum_{j in N(i)} J_ij gamma_j
on flip of SNP k (gamma_k: 0 -> 1 or 1 -> 0):
    for i in N(k):  F_i += (+/-) J_ik
```

`O(|N(k)|)` per flip. Row `k` of `J` is computed lazily and cached until the next REFRESH.
No partition function is needed for sampling.

### 4.4 Scaling notes

- `gamma` is sparse (≲1e4 causal of ~1e7). Computing `eta_i` only needs `J_ij` for causal
  neighbours `j` in that sample — usually 0–3 pairs. Never materialise dense attention.
- Most SNPs have `pi_i ≈ 0`: keep all SNPs with non-trivial `pi_i`, uniformly subsample the rest
  with inverse-probability weights (~1e5 terms per sample instead of ~1e7).
- Heavy features (sequence-model embeddings) are precomputed once and frozen.

## 5. Validation (required before trusting learned J)

1. **Null simulation (most important):** simulate on real LD with `J = 0`; learned `J` must be ≈ 0.
   Spurious short-range `J` signals misspecification (LD reference mismatch, wrong slab, etc.).
   Rationale: under a correct model `E_z[p(gamma | z)] = p(gamma)`, so LD-induced posterior
   anticorrelation averages out across loci; misspecification breaks this.
2. **Planted coupling:** simulate with known `J` (e.g. clustering within cCREs); check recovery
   of `J`, PIP calibration, and power vs. the `J = 0` baseline.
3. **Real data:** learned high-`J` pairs should be enriched for shared elements / 3D contacts
   held out from the features.
4. **Ablations:** `J = 0`; single head without `c_ij`; without contextual `f_theta`.

## 6. Design rationale (decisions already made — don't re-litigate without reason)

- **Ising with symmetric `J`, not softmax attention:** softmax row-normalises (asymmetric `J`,
  conditionals not from any joint, fixed total influence per SNP).
- **Single bilinear head(s), not a transformer over `gamma`:** a transformer that takes
  causal states as input gives a dependency network (BERT-like conditionals) with no coherent
  joint; the energy-based fix is expensive genome-wide. A transformer is used only to build
  embeddings from *features* (§3.1).
- **Separate multiplicative `kappa`, not position folded into `q`:** concatenating position into
  `q` makes distance additive; distance must gate coupling to 0. RoPE is a possible later upgrade
  (inside the hard window, multiplied by a decay envelope).
- **Pair-specific "only one causal" effects** (`alpha_ij gamma_i`) don't depend on `gamma_j` and
  are absorbed into `h_i` — which is why `h_i` uses contextual embeddings.
- **Small model:** the signal for `J` comes only from loci with ≥2 causal variants after LD;
  keep few heads, low `d`, L2 on `J`.

## 7. Extensions (not v1)

- **Saturating / higher-order coupling:** energy
  `E(gamma) = sum_i gamma_i [ h_i + 0.5 * g( sum_j J_ij gamma_j ) ]` with saturating `g`.
- **Effect-size coupling:** when both causal,
  `(beta_i, beta_j) ~ N(0, sigma2 [[1, rho_ij], [rho_ij, 1]])`,
  `rho_ij = tanh(score(e_i, e_j))` — concordant effects within an element. Easier to learn than
  causal-status coupling (each causal pair contributes an observed sign).
- **eQTL version:** indicators per gene–SNP pair `gamma_ig`; `h_ig` uses gene features and TSS
  distance; `J` shared across genes (optional gene-specific scale).
- **Multi-context:** share `J` across tissues/contexts, let `h` vary by context; could connect to
  the latent-factor structure in `causal_eqtl_lf_model/`.

## 8. Open questions

- Which genome-wide tempered-Gibbs implementation/package to build on, and its interface for
  per-SNP prior log-odds.
- Feature set for `a_i` and pair features `c_ij`.
- Window size `w`, number of heads `H`, projection dim `d`.
