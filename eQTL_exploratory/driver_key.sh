##################
# Input data (same files as eb_coloc multi_causal_variant/real_data/driver_key.sh)
##################
# GTEx v10 summary statistics (<tissue>.v10.allpairs.chr<c>.parquet)
gtex_sumstats_dir="/lab-share/CHIP-Strober-e2/Public/GTEx/eqtl_sumstats/"

# Genotype data for GTEx (gtex_v9_eqtl_chr<c>.pgen/.pvar/.psam), used for in-sample LD
gtex_genotype_data_dir="/lab-share/CHIP-Strober-e2/Public/ben/process_gtex_genotype_data/processed_genotype/"

# Gene annotation file (protein-coding genes)
gene_annotation_file="/lab-share/CHIP-Strober-e2/Public/gene_annotation_files/gencode.v39.gtex.protein_coding.genes.gtf"


##################
# Output data
##################
output_root="/lab-share/CHIP-Strober-e2/Public/ben/multi_variant_prior/eQTL_exploratory/"

# Separate folder within for each tissue (and cis window)
preprocessed_data=${output_root}"preprocessed_data/"

# Model fits
ising_results=${output_root}"ising_results/"


##################
# Model settings
##################
# Variants within +/- cis_window bp of the TSS (every pair of a gene's variants then falls in one of the 9 pair-distance bins of
# ising_eqtl_tempered.py, which cover 0-50 kb)
cis_window="25000"
# Genes with fewer variants (after matching to the genotype panel) are dropped
min_variants_per_gene="25"
# Stochastic-EM sweeps and burn-in (h is learned from sweep 5, J from sweep 50; the retained sweeps give the PIPs and the
# reported (h, J))
n_iters="1200"
burn_in="600"

tissue_name="Whole_Blood"
eqtl_ss="803"


##################
# Step 1: preprocess one tissue: GTEx statistics of protein-coding genes within +/- cis_window of the TSS, in-sample LD from the
# GTEx genotype panel, SBayesRC low-rank transform (preprocess_eqtl_data.py). MHC genes dropped
##################
if false; then
sbatch run_preprocess_eqtl_data.sh $gtex_sumstats_dir $tissue_name $eqtl_ss $preprocessed_data $gene_annotation_file $gtex_genotype_data_dir $cis_window $min_variants_per_gene
fi


##################
# Step 2 (after step 1 has finished): fit the distance-binned Ising prior (J learned) and the J = 0 baseline as separate jobs on the
# same preprocessed data. Results: ${ising_results}<tissue>_cis_window_<w>_<ising|baseline>_n_iters_<n>_* with per-gene PIPs
# (_posterior_summary.txt), (h, J) with their range over the retained sweeps (_hyperparameter_summary.txt; Monte Carlo fluctuation only,
# not uncertainty in J) and the trace
##################
if false; then
sbatch run_ising_eqtl.sh $tissue_name $preprocessed_data $ising_results $cis_window ising $n_iters $burn_in
sbatch run_ising_eqtl.sh $tissue_name $preprocessed_data $ising_results $cis_window baseline $n_iters $burn_in
fi
