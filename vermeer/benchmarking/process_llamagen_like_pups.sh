#!/bin/bash

set -e

# --- Environment Setup ---

# replace with path your environment
conda activate /n/home08/skambhampati/.local/share/mamba/envs/ar_microscopy_gen_2
which python
python --version
export PYTHONPATH="/n/home08/skambhampati/ar-microscopy-gen/code/CA_LlamaGen:$PYTHONPATH"  # change to your path


# replace with path to your input_dir
python process_llamagen_like_pups.py \
    --input_dir /n/netscratch/chenf2011_lab/sandeep/evaluation_results_fov_val2/386-hpa_split_size_256-GPT-L \
    --num_workers 4