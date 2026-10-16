import argparse
import numpy as np
import pdb
import pyarrow.parquet as pq
import pgenlib as pg


###############################################################################
# Preprocess GTEx v10 cis-eQTL summary statistics of one tissue into the per-gene inputs of the distance-binned Ising eQTL
# model (ising_eqtl_tempered.py / run_ising_eqtl.py). Adapted from eb_coloc multi_causal_variant/real_data/
# preprocess_colocalization_data.py (same GTEx files, genotype panel, allele orientation, z -> standardized effect conversion
# and SBayesRC low-rank LD transform), with three differences: eQTL only (no GWAS matching), a +/- 25 kb cis window, and the
# GTEx statistics are read one chromosome at a time with vectorized filtering on the TSS distance.
#
# Genes: protein-coding (gene annotation file), window not overlapping --exclude_region (default: the MHC, chr6:25-35 Mb, hg38),
# at least min_variants_per_gene variants after matching to the genotype panel. Variants: every variant with a GTEx statistic for
# the gene within cis_window bp of its TSS (GTEx tss_distance), present in the GTEx genotype panel (either allele orientation)
# with no missing dosages and non-zero variance.
#
# Per gene (output_stem = <output_dir><tissue>_cis_window_<cis_window>):
#   <output_stem>_gene_<gene>_variant_ids.txt   variant order of everything else
#   <output_stem>_gene_<gene>_tss_distance.npy  variant position - TSS (bp)
#   <output_stem>_gene_<gene>_ld_Q.npy          Q = Lambda^{1/2} U' (k x n_snps), LD truncated to 99.5% of its trace
#   <output_stem>_gene_<gene>_eigen_sumstats.npy w = Lambda^{-1/2} U' b, b the standardized marginal eQTL effects
#   <output_stem>_gene_<gene>_sumstats.npy      n_snps x 2: b, se(b) (untransformed; for other fine-mapping methods)
# plus <output_stem>_input_summary.txt (one line per gene), _gene_summary.txt and _excluded_genes.txt.
###############################################################################


def extract_protein_coding_genes(gene_annotation_file):
    # (copied from eb_coloc preprocess_colocalization_data.py)
    protein_coding_genes = {}
    f = open(gene_annotation_file, 'r')
    for line in f:
        line = line.strip()
        data = line.split('\t')
        ens_id = data[8].split(';')[0].split('"')[1]
        if ens_id.startswith('ENSG') == False:
            print('assumption erororo')
            pdb.set_trace()
        protein_coding_genes[ens_id.split('.')[0]] = 0
    f.close()
    return protein_coding_genes


def load_eqtl_data_for_chromosome(gtex_sumstats_file, cis_window, protein_coding_genes):
    # Per protein-coding gene of one chromosome, the variants within cis_window bp of the TSS with their z-scores and TSS
    # distances. Columns as read by eb_coloc: 0 gene id, 1 variant id, 2 tss_distance, -2 slope, -1 slope_se
    pf = pq.ParquetFile(gtex_sumstats_file)
    gene_ids, variant_ids, tss_dists, zeds = [], [], [], []
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        n_cols = table.num_columns
        tss_dist = table.column(2).to_numpy().astype(np.float64)
        keep = np.abs(tss_dist) <= cis_window
        if not np.any(keep):
            continue
        idx = np.flatnonzero(keep)
        gene_ids.append(np.asarray(table.column(0).take(idx).to_pylist()))
        variant_ids.append(np.asarray(table.column(1).take(idx).to_pylist()))
        tss_dists.append(tss_dist[idx])
        slope = table.column(n_cols - 2).take(idx).to_numpy(zero_copy_only=False).astype(np.float64)
        slope_se = table.column(n_cols - 1).take(idx).to_numpy(zero_copy_only=False).astype(np.float64)
        zeds.append(slope / slope_se)
    if len(gene_ids) == 0:
        return {}
    gene_ids = np.concatenate(gene_ids)
    variant_ids = np.concatenate(variant_ids)
    tss_dists = np.concatenate(tss_dists)
    zeds = np.concatenate(zeds)

    mapping = {}
    order = np.argsort(gene_ids, kind='stable')
    uniq, starts = np.unique(gene_ids[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    for gene_id, a, b in zip(uniq, starts, ends):
        if gene_id.split('.')[0] not in protein_coding_genes:
            continue
        ii = order[a:b]
        valid = np.isfinite(zeds[ii])
        ii = ii[valid]
        if len(ii) == 0:
            continue
        # Sort the gene's variants by position
        ii = ii[np.argsort(tss_dists[ii], kind='stable')]
        mapping[gene_id] = {
            'variant_ids': variant_ids[ii],
            'tss_distance': tss_dists[ii].astype(np.int64),
            'z': zeds[ii],
            # Gene TSS: GTEx tss_distance is variant position minus TSS
            'tss': int(variant_ids[ii[0]].split('_')[1]) - int(tss_dists[ii[0]]),
        }
    return mapping


def load_in_genotype(gtex_genotype_data_dir, chrom_num):
    # (copied from eb_coloc preprocess_colocalization_data.py)
    stem = gtex_genotype_data_dir + 'gtex_v9_eqtl_chr' + str(chrom_num)
    pgen = pg.PgenReader((stem + '.pgen').encode())
    n_samples = pgen.get_raw_sample_ct()
    n_variants = pgen.get_variant_ct()
    pvar = np.loadtxt(stem + '.pvar', dtype=str, delimiter='\t')
    psam = np.loadtxt(stem + '.psam', dtype=str, delimiter='\t')
    if pvar.shape[0] != n_variants or psam.shape[0] != n_samples:
        print('assumption error: pvar/psam dimensions do not match pgen')
        pdb.set_trace()
    return pgen, pvar, psam


def convert_genotype_variant_id_to_gtex_format(genotype_variant_id):
    # Genotype file ids are chr1:13550:G:A; convert to GTEx format chr1_13550_G_A_b38
    info = genotype_variant_id.split(':')
    if len(info) != 4 or info[0].startswith('chr') == False:
        print('assumption error: unexpected variant id format in genotype file: ' + genotype_variant_id)
        pdb.set_trace()
    return info[0] + '_' + info[1] + '_' + info[2] + '_' + info[3] + '_b38'


def create_mapping_from_genotype_variant_id_to_index(genotype_variant_ids):
    mapping = {}
    for index, genotype_variant_id in enumerate(genotype_variant_ids):
        variant_id = convert_genotype_variant_id_to_gtex_format(genotype_variant_id)
        if variant_id in mapping:
            print('assumption error: duplicate variant id in genotype file: ' + variant_id)
            pdb.set_trace()
        mapping[variant_id] = index
    return mapping


def flip_variant_id_alleles(variant_id):
    info = variant_id.split('_')
    if len(info) != 5:
        print('assumption error: unexpected variant id format: ' + variant_id)
        pdb.set_trace()
    return info[0] + '_' + info[1] + '_' + info[3] + '_' + info[2] + '_' + info[4]


def extract_ld_for_gene(pgen, genotype_variant_id_to_index, gene_variant_ids):
    # (copied from eb_coloc preprocess_colocalization_data.py) LD over the gene's variants, dosages oriented to the alt allele of
    # the GTEx variant id; variants absent from the panel, with missing dosages or monomorphic are dropped (keep == False)
    n_gene_variants = len(gene_variant_ids)
    n_samples = pgen.get_raw_sample_ct()
    G = np.zeros((n_gene_variants, n_samples), dtype=np.float32)
    keep = np.zeros(n_gene_variants, dtype=bool)
    buf = np.empty(n_samples, dtype=np.float32)
    for j, variant_id in enumerate(gene_variant_ids):
        if variant_id in genotype_variant_id_to_index:
            genotype_index = genotype_variant_id_to_index[variant_id]
            flip = False
        else:
            flipped_variant_id = flip_variant_id_alleles(variant_id)
            if flipped_variant_id in genotype_variant_id_to_index:
                genotype_index = genotype_variant_id_to_index[flipped_variant_id]
                flip = True
            else:
                continue
        pgen.read_dosages(int(genotype_index), buf)
        if np.any(buf < 0):
            continue
        if flip:
            G[j, :] = 2.0 - buf
        else:
            G[j, :] = buf
        keep[j] = True
    G = G[keep, :]
    polymorphic = G.std(axis=1) > 0
    if np.sum(polymorphic == False) > 0:
        keep_indices = np.flatnonzero(keep)
        keep[keep_indices[polymorphic == False]] = False
        G = G[polymorphic, :]
    if G.shape[0] < 2:
        return None, keep
    ld_mat = np.corrcoef(G)
    return ld_mat, keep


def eigen_decompose_ld(ld_mat, variance_explained_threshold):
    # (copied from eb_coloc preprocess_colocalization_data.py) R = U Lambda U' truncated to variance_explained_threshold of the
    # trace; returns Lambda, Q = Lambda^{1/2} U' and Lambda^{-1/2} U'
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


def convert_zed_to_standardized_effects(zed, sample_size):
    # (copied from eb_coloc) beta = z / sqrt(N - 2 + z^2), se = 1 / sqrt(N - 2 + z^2): standardized genotype and trait (sdY = 1)
    zed = np.asarray(zed, dtype=np.float64)
    denom = np.sqrt(sample_size - 2.0 + (zed ** 2))
    return zed / denom, 1.0 / denom


def parse_genomic_region(region_string):
    if region_string is None or region_string.strip() == '' or region_string.strip().lower() == 'none':
        return None
    chrom, span = region_string.strip().split(':')
    start, end = span.split('-')
    return chrom, int(start), int(end)


def gene_window_overlaps_region(chrom_num, window_start, window_end, region):
    if region is None:
        return False
    return 'chr' + str(chrom_num) == region[0] and window_start < region[2] and window_end > region[1]


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess GTEx eQTL summary statistics of one tissue for the distance-binned Ising eQTL model')
    parser.add_argument('--gtex_sumstats_dir', required=True, help='Directory with <tissue>.v10.allpairs.chr<c>.parquet')
    parser.add_argument('--tissue_name', required=True)
    parser.add_argument('--eqtl_ss', type=int, required=True, help='eQTL sample size of the tissue')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gene_annotation_file', required=True, help='GTF of protein-coding genes (as in eb_coloc)')
    parser.add_argument('--gtex_genotype_data_dir', required=True, help='Directory with gtex_v9_eqtl_chr<c>.pgen/.pvar/.psam')
    parser.add_argument('--cis_window', type=int, default=25000, help='bp on each side of the TSS')
    parser.add_argument('--min_variants_per_gene', type=int, default=25)
    parser.add_argument('--ld_eigen_variance_explained', type=float, default=0.995)
    parser.add_argument('--exclude_region', default='chr6:25000000-35000000', help="hg38 region whose genes are dropped (default: MHC); 'none' to disable")
    parser.add_argument('--chromosomes', default='1-22', help="e.g. '1-22' or '22' (for testing)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if '-' in args.chromosomes:
        lo, hi = args.chromosomes.split('-')
        chromosomes = list(range(int(lo), int(hi) + 1))
    else:
        chromosomes = [int(c) for c in args.chromosomes.split(',')]
    exclude_region = parse_genomic_region(args.exclude_region)
    protein_coding_genes = extract_protein_coding_genes(args.gene_annotation_file)
    print(str(len(protein_coding_genes)) + ' protein-coding genes in ' + args.gene_annotation_file, flush=True)

    output_stem = args.output_dir + args.tissue_name + '_cis_window_' + str(args.cis_window)
    t_input = open(output_stem + '_input_summary.txt', 'w')
    t_gene = open(output_stem + '_gene_summary.txt', 'w')
    t_excl = open(output_stem + '_excluded_genes.txt', 'w')
    t_input.write('Gene_Name\tGene_Chrom\tGene_TSS\tVariant_IDs_File\tTSS_Distance_File\tLD_Q_File\tEigen_Sumstats_File\tSumstats_File\n')
    t_gene.write('Gene_Name\tGene_Chrom\tGene_TSS\tNum_SNPs\tNum_LD_Eigen_Components\tMax_Abs_Z\n')
    t_excl.write('Gene_Name\tGene_Chrom\tGene_TSS\tReason\n')
    n_written = 0
    n_snps_total = 0

    for chrom_num in chromosomes:
        gtex_file = args.gtex_sumstats_dir + args.tissue_name + '.v10.allpairs.chr' + str(chrom_num) + '.parquet'
        gene_data = load_eqtl_data_for_chromosome(gtex_file, args.cis_window, protein_coding_genes)
        pgen, pvar, psam = load_in_genotype(args.gtex_genotype_data_dir, chrom_num)
        genotype_variant_id_to_index = create_mapping_from_genotype_variant_id_to_index(pvar[:, 2])
        n_chrom = 0
        for gene_name in sorted(gene_data, key=lambda gname: gene_data[gname]['tss']):
            d = gene_data[gene_name]
            tss = d['tss']
            if gene_window_overlaps_region(chrom_num, tss - args.cis_window, tss + args.cis_window + 1, exclude_region):
                t_excl.write(gene_name + '\t' + str(chrom_num) + '\t' + str(tss) + '\toverlaps_excluded_region_' + args.exclude_region + '\n')
                continue
            ld_mat, keep = extract_ld_for_gene(pgen, genotype_variant_id_to_index, d['variant_ids'])
            if ld_mat is None or int(np.sum(keep)) < args.min_variants_per_gene:
                t_excl.write(gene_name + '\t' + str(chrom_num) + '\t' + str(tss) + '\tfewer_than_' + str(args.min_variants_per_gene) + '_variants\n')
                continue
            variant_ids = d['variant_ids'][keep]
            tss_distance = d['tss_distance'][keep]
            z = d['z'][keep]
            if ld_mat.shape[0] != len(variant_ids):
                print('assumption error: LD matrix dimension does not match number of variants')
                pdb.set_trace()

            ld_eigenvalues, ld_Q, ld_lambda_inv_sqrt_Ut = eigen_decompose_ld(ld_mat, args.ld_eigen_variance_explained)
            beta_hat, beta_se = convert_zed_to_standardized_effects(z, args.eqtl_ss)
            w = ld_lambda_inv_sqrt_Ut @ beta_hat

            gene_stem = output_stem + '_gene_' + gene_name
            np.savetxt(gene_stem + '_variant_ids.txt', variant_ids, fmt='%s')
            np.save(gene_stem + '_tss_distance.npy', tss_distance)
            np.save(gene_stem + '_ld_Q.npy', ld_Q)
            np.save(gene_stem + '_eigen_sumstats.npy', w)
            np.save(gene_stem + '_sumstats.npy', np.column_stack([beta_hat, beta_se]))
            t_input.write('\t'.join([gene_name, str(chrom_num), str(tss), gene_stem + '_variant_ids.txt', gene_stem + '_tss_distance.npy',
                                     gene_stem + '_ld_Q.npy', gene_stem + '_eigen_sumstats.npy', gene_stem + '_sumstats.npy']) + '\n')
            t_gene.write('\t'.join([gene_name, str(chrom_num), str(tss), str(len(variant_ids)), str(len(ld_eigenvalues)), '%.3f' % np.max(np.abs(z))]) + '\n')
            n_written += 1
            n_chrom += 1
            n_snps_total += len(variant_ids)
        print('chr' + str(chrom_num) + ': ' + str(n_chrom) + ' genes written', flush=True)

    t_input.close()
    t_gene.close()
    t_excl.close()
    print(str(n_written) + ' genes written (' + str(n_snps_total) + ' variants; cis window +/- ' + str(args.cis_window) + ' bp)')
    print('Input summary file: ' + output_stem + '_input_summary.txt')
