import argparse
import time
import numpy as np
from simulate_ising_eqtl_data import simulate, N_TSS_BINS, N_PAIR_BINS
from ising_eqtl_tempered import IsingEQTLTempered, TSS_BIN_EDGES, PAIR_BIN_EDGES


# Simulate genes under a planted (or null) distance-binned Ising prior, fit the model with J learned and the J = 0 baseline,
# and report h / J recovery and PIP calibration and power.

H_TRUE = np.array([-2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5])
J_PRESETS = {
    'null': np.zeros(N_PAIR_BINS),
    'planted': np.array([1.0, 0.8, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--J', choices=list(J_PRESETS), default='planted')
    parser.add_argument('--n_genes', type=int, default=1000)
    parser.add_argument('--n_samples', type=int, default=800)
    parser.add_argument('--mean_n_snps', type=int, default=120)
    parser.add_argument('--slab_var', type=float, default=0.02)
    parser.add_argument('--n_iters', type=int, default=400)
    parser.add_argument('--burn_in', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_baseline', action='store_true')
    parser.add_argument('--resid_var', choices=['free', 'bounded', 'fixed'], default='free',
                        help='per-gene residual variance: free (mcvc_tempered.py), bounded to [(1 - h2_max) / N, 1 / N], or fixed at 1 / N')
    parser.add_argument('--h2_max', type=float, default=0.5)
    parser.add_argument('--out', default=None, help='output .npz')
    return parser.parse_args()


def bin_labels(edges):
    lo = np.concatenate([[0], edges])
    hi = np.concatenate([edges, [np.inf]])
    return ['%g-%g' % (a, b) if np.isfinite(b) else '>%g' % a for a, b in zip(lo, hi)]


def pip_report(name, model, data):
    pip = np.concatenate(model.pip)
    truth = np.concatenate([d['gamma'] for d in data])
    print('%s: sum PIP = %.0f (true causal %d)' % (name, pip.sum(), truth.sum()))
    for lo, hi in [(0.1, 0.5), (0.5, 0.9), (0.9, 1.01)]:
        m = (pip >= lo) & (pip < hi)
        print('   PIP in [%.1f, %.1f): n=%5d  mean PIP=%.3f  frac causal=%.3f' % (lo, min(hi, 1), m.sum(), pip[m].mean() if m.any() else np.nan, truth[m].mean() if m.any() else np.nan))
    print('   causal SNPs with PIP>0.5: %d / %d' % (np.sum(truth[pip > 0.5]), truth.sum()))
    return pip


def main():
    args = parse_args()
    J_true = J_PRESETS[args.J]
    t0 = time.time()
    data = simulate(args.n_genes, H_TRUE, J_true, n_samples=args.n_samples, mean_n_snps=args.mean_n_snps, slab_var=args.slab_var, seed=args.seed)
    n_causal = np.array([d['gamma'].sum() for d in data])
    print('simulated %d genes in %.0fs: mean SNPs/gene %.0f, mean causal/gene %.2f, genes with >=2 causal %d, mean |z|max %.1f' % (
        args.n_genes, time.time() - t0, np.mean([len(d['gamma']) for d in data]), n_causal.mean(), np.sum(n_causal >= 2),
        np.mean([np.max(np.abs(d['z'])) for d in data])), flush=True)

    N = args.n_samples
    bounds = {'free': None, 'bounded': ((1.0 - args.h2_max) / N, 1.0 / N), 'fixed': (1.0 / N, 1.0 / N)}[args.resid_var]
    print('residual variance: %s %s' % (args.resid_var, bounds))
    inputs = ([d['Q'] for d in data], [d['w'] for d in data], [d['tss_distance'] for d in data], [d['position'] for d in data])
    t0 = time.time()
    model = IsingEQTLTempered().fit(*inputs, n_iters=args.n_iters, burn_in=args.burn_in, seed=args.seed, learn_J=True, resid_var_bounds=bounds)
    print('Ising fit: %.0fs' % (time.time() - t0))

    J_sd = model.sampled_J.std(axis=0)
    print('\nJ by pair distance (bp):   true    est   (sd over retained sweeps)')
    for lab, t, e, s in zip(bin_labels(PAIR_BIN_EDGES), J_true, model.J_mean, J_sd):
        print('  %-12s %6.2f %6.2f  (%.2f)' % (lab, t, e, s))
    print('h by TSS distance (bp):    true    est')
    for lab, t, e in zip(bin_labels(TSS_BIN_EDGES), H_TRUE, model.h_mean):
        print('  %-12s %6.2f %6.2f' % (lab, t, e))
    print('slab var: true %.3g est %.3g\n' % (args.slab_var, model.sampled_slab_var.mean()))
    pip_ising = pip_report('Ising (J learned)', model, data)

    result = dict(J_true=J_true, h_true=H_TRUE, J_est=model.J_mean, h_est=model.h_mean, sampled_J=model.sampled_J,
                  sampled_h=model.sampled_h, trace=model.trace, pip_ising=pip_ising,
                  gamma=np.concatenate([d['gamma'] for d in data]), resid_var=model.resid_var, slab_var_est=model.sampled_slab_var.mean())
    if not args.no_baseline:
        t0 = time.time()
        base = IsingEQTLTempered().fit(*inputs, n_iters=args.n_iters, burn_in=args.burn_in, seed=args.seed, learn_J=False, verbose=False, resid_var_bounds=bounds)
        print('\nbaseline fit: %.0fs; h est: %s' % (time.time() - t0, np.array2string(base.h_mean, precision=2)))
        result['pip_baseline'] = pip_report('Baseline (J = 0)', base, data)
        result['h_est_baseline'] = base.h_mean
    if args.out is not None:
        np.savez(args.out, **result)


if __name__ == '__main__':
    main()
