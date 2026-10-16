import argparse
import sys
import time
import numpy as np
sys.dont_write_bytecode = True
from ising_eqtl_tempered import IsingEQTLTempered, TSS_BIN_EDGES, PAIR_BIN_EDGES


# Fit the distance-binned Ising eQTL model (ising_eqtl_tempered.py) to the per-gene inputs written by preprocess_eqtl_data.py,
# or the J = 0 baseline (--no_J: TSS-distance-binned prior only). Writes per-gene PIPs and posterior mean effects, the learned
# (h, J) with their spread over the retained sweeps, and the per-sweep trace.
def parse_args():
    parser = argparse.ArgumentParser(description='Fit the distance-binned Ising eQTL prior to one tissue')
    parser.add_argument('input_summary_file', help='<stem>_input_summary.txt written by preprocess_eqtl_data.py')
    parser.add_argument('output_stem', help='Output stem')
    parser.add_argument('--no_J', action='store_true', help='Fit the J = 0 baseline (independent TSS-distance-binned prior)')
    parser.add_argument('--n_iters', type=int, default=600)
    parser.add_argument('--burn_in', type=int, default=300)
    parser.add_argument('--J_update_start', type=int, default=50, help='Sweep at which J starts being learned (h is learned from sweep 5)')
    parser.add_argument('--lambda_J', type=float, default=1.0, help='L2 penalty on J')
    parser.add_argument('--lambda_smooth', type=float, default=1.0, help='L2 penalty on differences of J between adjacent distance bins')
    parser.add_argument('--max_genes', type=int, default=0, help='Use only the first max_genes genes (0: all; for testing)')
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()


def bin_labels(edges):
    lo = np.concatenate([[0], edges])
    hi = np.concatenate([edges, [np.inf]])
    return ['%d_%d' % (a, b) if np.isfinite(b) else '%d_inf' % a for a, b in zip(lo, hi)]


def load_input_summary(input_summary_file, max_genes):
    gene_names, Q_list, w_list, tss_list = [], [], [], []
    f = open(input_summary_file, 'r')
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            column = {name: index for index, name in enumerate(data)}
            continue
        gene_names.append(data[column['Gene_Name']])
        tss_list.append(np.load(data[column['TSS_Distance_File']]))
        Q_list.append(np.load(data[column['LD_Q_File']]))
        w_list.append(np.load(data[column['Eigen_Sumstats_File']]))
        if max_genes > 0 and len(gene_names) >= max_genes:
            break
    f.close()
    if len(set(gene_names)) != len(gene_names):
        raise ValueError('Duplicate gene names in ' + input_summary_file)
    return gene_names, Q_list, w_list, tss_list


args = parse_args()
gene_names, Q_list, w_list, tss_list = load_input_summary(args.input_summary_file, args.max_genes)
n_snps = np.array([Q.shape[1] for Q in Q_list])
print('%d genes, %d variants (median %d per gene) from %s' % (len(gene_names), n_snps.sum(), np.median(n_snps), args.input_summary_file), flush=True)

t0 = time.time()
model = IsingEQTLTempered()
# Positions only enter through pairwise differences, so the TSS distances serve as positions within a gene
model.fit(Q_list, w_list, tss_list, tss_list, n_iters=args.n_iters, burn_in=args.burn_in, learn_J=not args.no_J,
          J_update_start=args.J_update_start, lambda_J=args.lambda_J, lambda_smooth=args.lambda_smooth, seed=args.seed)
print('fit: %.0f s' % (time.time() - t0), flush=True)

# Per-gene PIPs and posterior mean effects (';'-separated, in the order of the gene's variant ids file)
t = open(args.output_stem + '_posterior_summary.txt', 'w')
t.write('gene_name\tPIP\tbeta_posterior_mean\n')
for g, gene_name in enumerate(gene_names):
    t.write(gene_name + '\t' + ';'.join('%.6g' % v for v in model.pip[g]) + '\t' + ';'.join('%.6g' % v for v in model.beta_posterior_mean[g]) + '\n')
t.close()

# Hyper-parameters: mean and 95% range over the retained sweeps (stochastic-EM iterates). The range reflects Monte Carlo
# fluctuation of the fit only, NOT uncertainty in (h, J): in null simulations J varied far more between replicates
names = ['h_tss_' + lab for lab in bin_labels(TSS_BIN_EDGES)] + ['J_pair_' + lab for lab in bin_labels(PAIR_BIN_EDGES)] + ['slab_var']
samples = np.column_stack([model.sampled_h, model.sampled_J, model.sampled_slab_var])
t = open(args.output_stem + '_hyperparameter_summary.txt', 'w')
t.write('hyperparameter\testimate\tsweep_95_lb\tsweep_95_ub\n')
for i, name in enumerate(names):
    vals = [np.mean(samples[:, i]), np.quantile(samples[:, i], 0.025), np.quantile(samples[:, i], 0.975)]
    t.write(name + '\t' + '\t'.join('%.6g' % v for v in vals) + '\n')
t.close()
np.save(args.output_stem + '_hyperparameter_samples.npy', samples)
np.save(args.output_stem + '_trace.npy', model.trace)
print('results: ' + args.output_stem + '_*', flush=True)
