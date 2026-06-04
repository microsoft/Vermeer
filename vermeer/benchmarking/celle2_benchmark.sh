#!/bin/bash

# pip environment
#replace with path to environment for running celle2
VENV_PATH="/n/netscratch/chenf2011_lab/sandeep/pip_envs/celle2_v2"
if [ ! -x "${VENV_PATH}/bin/python" ]; then
    echo "ERROR: Virtualenv python not found at ${VENV_PATH}/bin/python"
    exit 1
fi

source "${VENV_PATH}/bin/activate"
hash -r
PYTHON_BIN="${VENV_PATH}/bin/python"

echo "Using python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

val_split=${1:-val1}

echo "Benchmarking CELL-E2 on ${val_split} split"

# replace paths to your data / output dir
"${PYTHON_BIN}" benchmark_celle2.py \
    --val_data_dir /n/netscratch/chenf2011_lab/sandeep/data/hpa_dataset/hpa_preprocessed_split_256/concatenated_arrays \
    --val_split "${val_split}" \
    --output_dir "/n/netscratch/chenf2011_lab/sandeep/evaluation_results_benchmarks/CELL-E2_HPA_480/${val_split}" \
    --batch_size 16 \
    --seed 42 \
    --skip_eval
