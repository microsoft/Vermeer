#!/bin/bash

# replace path to your pups environment
conda activate /n/netscratch/chenf2011_lab/sandeep/pip_envs/pups_v2

val_split=${1:-val1}

echo "Benchmarking PUPS on ${val_split} split"

# replace paths to your datadirs
python benchmark_pups.py \
    --val_dir /n/netscratch/chenf2011_lab/sandeep/data/hpa_dataset/hpa_preprocessed_split_256_code_flip_ten_crop_rotate_esm_embed_mean_pool \
    --val_split "${val_split}" \
    --raw_hpa_dir /n/netscratch/chenf2011_lab/sandeep/data/hpa_dataset/humanproteinatlas_temp \
    --pups_dir /n/home08/skambhampati/ar-microscopy-gen/code/CA_LlamaGen/benchmarking/PUPS \
    --output_dir "/n/netscratch/chenf2011_lab/sandeep/evaluation_results_benchmarks/PUPS/${val_split}" \
    --num_workers 4 \
    --crop_batch_size 64 \
    --skip_eval
