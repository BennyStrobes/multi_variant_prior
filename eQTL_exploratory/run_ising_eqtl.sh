#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-72:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                     # Partition to run in
#SBATCH --mem=20G                          # Memory total in MiB (for all cores): every gene's Q, LD Gram matrix Q'Q and pair-bin matrix in memory




tissue_name="${1}"
preprocessed_data="${2}"
ising_results="${3}"
cis_window="${4}"
# ising: distance-binned Ising prior (J learned); baseline: J = 0 (TSS-distance-binned prior only)
model_type="${5}"
n_iters="${6}"
burn_in="${7}"

source ~/.bash_profile
conda activate plink_env

# Same stem and preprocessed data as run_preprocess_eqtl_data.sh (which must have finished for this tissue)
run_stem="${tissue_name}_cis_window_${cis_window}"
input_summary_file="${preprocessed_data}data_${run_stem}/${run_stem}_input_summary.txt"
if [ ! -f "${input_summary_file}" ]; then
    echo "Missing ${input_summary_file}; run run_preprocess_eqtl_data.sh for ${tissue_name} first"
    exit 1
fi

mkdir -p "${ising_results}"
output_stem="${ising_results}${run_stem}_${model_type}_n_iters_${n_iters}"

if [ "${model_type}" == "ising" ]; then
    model_flag=""
elif [ "${model_type}" == "baseline" ]; then
    model_flag="--no_J"
else
    echo "model_type must be ising or baseline, not ${model_type}"
    exit 1
fi

echo "Fitting ${model_type} on ${tissue_name} (${n_iters} sweeps, ${burn_in} burn-in)..."
python run_ising_eqtl.py ${input_summary_file} ${output_stem} ${model_flag} --n_iters ${n_iters} --burn_in ${burn_in}
