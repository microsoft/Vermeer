#!/bin/bash

# ==============================================================================
# Model-agnostic evaluation pipeline.
#
# Runs FID, MSE, SubCell localization, and intra-nuclear proportion metrics
# on pre-generated images. Works with output from any model.
#
# Usage:
#   sbatch benchmarking/eval_model.sh /path/to/images_dir [OPTIONS]
#
# The images_dir must contain:
#   images_true/   true_{basename}_{channel}.png
#   images_gen/    gen_{basename}_{channel}.png
#
# Examples:
#   # Basic protein-only evaluation
#   sbatch benchmarking/eval_model.sh /path/to/output
#
#   # All-channels evaluation
#   sbatch benchmarking/eval_model.sh /path/to/output --generation_mode all_channels
#
#   # Skip specific evals
#   sbatch benchmarking/eval_model.sh /path/to/output --skip_subcell --skip_intra_nuclear
#
#   # Custom output directory
#   sbatch benchmarking/eval_model.sh /path/to/output --output_dir /path/to/results
# ==============================================================================

set -e

# --- Environment Setup ---
# replace with path to your environment
conda activate /n/home08/skambhampati/.local/share/mamba/envs/ar_microscopy_gen_2

# --- Parse Arguments ---
if [ -z "$1" ]; then
    echo "ERROR: images_dir is required as the first argument"
    echo "Usage: sbatch benchmarking/eval_model.sh /path/to/images_dir [OPTIONS]"
    exit 1
fi

IMAGES_DIR="$1"
shift  # Remove images_dir from args, pass the rest through

echo "============================================================"
echo "Model-Agnostic Evaluation Pipeline"
echo "============================================================"
echo "Images dir: $IMAGES_DIR"
echo "Extra args: $@"
echo "============================================================"

python eval_model.py \
    --images_dir "$IMAGES_DIR" \
    "$@"

echo "============================================================"
echo "Evaluation complete."
echo "============================================================"
