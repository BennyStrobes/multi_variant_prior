#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-24:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                     # Partition to run in
#SBATCH --mem=30G                          # Memory total in MiB (for all cores): one chromosome of GTEx allpairs statistics at a time




gtex_sumstats_dir="${1}"
tissue_name="${2}"
eqtl_ss="${3}"
preprocessed_data="${4}"
gene_annotation_file="${5}"
gtex_genotype_data_dir="${6}"
cis_window="${7}"
min_variants_per_gene="${8}"

source ~/.bash_profile
conda activate plink_env

# Same stem as preprocess_eqtl_data.py: <tissue>_cis_window_<cis_window>
run_stem="${tissue_name}_cis_window_${cis_window}"
tissue_preprocessed_dir="${preprocessed_data}data_${run_stem}/"
mkdir -p "${tissue_preprocessed_dir}"

echo "Preprocessing ${tissue_name} (cis window +/- ${cis_window} bp, at least ${min_variants_per_gene} variants per gene)..."
python preprocess_eqtl_data.py \
    --gtex_sumstats_dir "${gtex_sumstats_dir}" \
    --tissue_name "${tissue_name}" \
    --eqtl_ss "${eqtl_ss}" \
    --output_dir "${tissue_preprocessed_dir}" \
    --gene_annotation_file "${gene_annotation_file}" \
    --gtex_genotype_data_dir "${gtex_genotype_data_dir}" \
    --cis_window "${cis_window}" \
    --min_variants_per_gene "${min_variants_per_gene}"
if [ $? -ne 0 ]; then
    echo "Preprocessing failed for ${run_stem}"
    exit 1
fi
