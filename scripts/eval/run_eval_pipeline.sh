#!/bin/bash

# ==============================================================================
# Evaluation Pipeline: Generate -> ProteinValLoss -> SubCell -> FID -> MSE -> IntraNuclearProp
#
# Usage:
#   sbatch scripts/eval/run_eval_pipeline.sh [VAL_SPLIT]
#
# Arguments:
#   VAL_SPLIT   Validation split to evaluate (default: val1)
#
# Optional environment variable overrides:
#   DECODE_BATCH_SIZE, TEMPERATURE, TOP_K, TOP_P, N_PREFIX_CHANNELS, IMAGE_SIZE, NUM_SAMPLES
#
# GENERATION_MODE (set in script):
#   protein_only   (default) Generate protein channel conditioned on landmark stains.
#                  Runs: generate -> protein val loss -> subcell eval -> FID (protein only) -> MSE (protein only) -> intra-nuclear proportion
#   all_channels   Generate all channels from ESM embedding only.
#                  Runs: generate -> protein val loss -> FID (all channels) -> MSE (all channels). SubCell eval is skipped.
#
# Example:
#   sbatch scripts/eval/run_eval_pipeline.sh val1
# ==============================================================================

set -e  # Exit on error

# --- Environment Setup ---
module load python
eval "$(conda shell.bash hook)"
conda activate /n/home08/skambhampati/.local/share/mamba/envs/ar_microscopy_gen_2  # change to your path

# --- Parse Arguments ---
val_split=${1:-val1} # double check this

MODEL_PREFIX="386-hpa_split_size_256-GPT-L"
CHECKPOINT_NUMBER="0075000"
DECODE_BATCH_SIZE=256
GPT_MODEL="GPT-L"
MODEL_TYPE="ca_esm_embed_mean_pool"
N_MAX_CHANNELS=4


# --- Default values ---

MODEL_CHECKPOINT="/n/netscratch/chenf2011_lab/sandeep/CA_LlamaGen_local_output/${MODEL_PREFIX}/checkpoints/${CHECKPOINT_NUMBER}.pt"  # change to your path
if [ "$MODEL_TYPE" = "ca_esm_embed_full" ]; then
    VAL_DIR="/n/netscratch/chenf2011_lab/sandeep/data/hpa_dataset/hpa_preprocessed_split_256_code_flip_ten_crop_rotate_esm_embed_full"  # change to your path
else
    VAL_DIR="/n/netscratch/chenf2011_lab/sandeep/data/hpa_dataset/hpa_preprocessed_split_256_code_flip_ten_crop_rotate_esm_embed_mean_pool"  # change to your path
fi
VAL_SPLIT="${1:-val1}"

GENERATION_MODE="protein_only"
OUTPUT_DIR="/n/netscratch/chenf2011_lab/sandeep/evaluation_results_fov_${VAL_SPLIT}/${MODEL_PREFIX}"  # change to your path
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-64}"  # 64 on a100
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-2000}"
TOP_P="${TOP_P:-1.0}"
N_PREFIX_CHANNELS="${N_PREFIX_CHANNELS:-3}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
NO_COMPILE="${NO_COMPILE:-false}"

echo "============================================================"
echo "Evaluation Pipeline"
echo "============================================================"
echo "Model checkpoint:  $MODEL_CHECKPOINT"
echo "Validation dir:    $VAL_DIR"
echo "Output dir:        $OUTPUT_DIR"
echo "Val split:         $VAL_SPLIT"
echo "GPT model:         $GPT_MODEL"
echo "Model type:        $MODEL_TYPE"
echo "Generation mode:   $GENERATION_MODE"
echo "Batch size:        $BATCH_SIZE"
echo "Decode batch size: $DECODE_BATCH_SIZE"
echo "Temperature:       $TEMPERATURE"
echo "Top-k:             $TOP_K"
echo "============================================================"

# Build optional num_samples argument
NUM_SAMPLES_ARG=""
if [ -n "$NUM_SAMPLES" ]; then
    NUM_SAMPLES_ARG="--num_samples $NUM_SAMPLES"
fi

NO_COMPILE_ARG=""
if [ "$NO_COMPILE" = "true" ]; then
    NO_COMPILE_ARG="--no-compile"
fi

# --- Step 1: Generate Images ---
echo ""
echo "============================================================"
echo "Step 1/5: Generating images..."
echo "============================================================"

python vermeer/evaluations/ca_esm/generate_images.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --val_split "$VAL_SPLIT" \
    --gpt_model "$GPT_MODEL" \
    --model_type "$MODEL_TYPE" \
    --generation_mode "$GENERATION_MODE" \
    --batch_size "$BATCH_SIZE" \
    --decode_batch_size "$DECODE_BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --n_prefix_channels "$N_PREFIX_CHANNELS" \
    --n_max_channels $N_MAX_CHANNELS \
    --image_size "$IMAGE_SIZE" \
    $NUM_SAMPLES_ARG \
    $NO_COMPILE_ARG

echo "Step 1 complete."

# --- Step 2: Protein Validation Loss ---
echo ""
echo "============================================================"
echo "Step 2/6: Computing protein-channel validation CE loss..."
echo "============================================================"

python vermeer/evaluations/ca_esm/evaluate_protein_val_loss.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --code_path "$VAL_DIR" \
    --val_splits "$VAL_SPLIT" \
    --gpt_model "$GPT_MODEL" \
    --gpt_type "$MODEL_TYPE" \
    --image_size "$IMAGE_SIZE" \
    --n_channels 4 \
    --n_max_channels $N_MAX_CHANNELS \
    --batch_size "$BATCH_SIZE" \
    --output_json "${OUTPUT_DIR}/protein_val_loss_results.json" \
    $NO_COMPILE_ARG

echo "Step 2 complete."

# --- Step 3: SubCell Localization Evaluation (protein_only mode only) ---
if [ "$GENERATION_MODE" = "protein_only" ]; then
    echo ""
    echo "============================================================"
    echo "Step 3/6: Running SubCell evaluation..."
    echo "============================================================"

    python vermeer/evaluations/ca_esm/evaluate_subcell.py \
        --images_dir "$OUTPUT_DIR" \
        --output_json "${OUTPUT_DIR}/subcell_results.json"

    echo "Step 3 complete."
else
    echo ""
    echo "============================================================"
    echo "Step 3/6: Skipping SubCell evaluation (all_channels mode)"
    echo "============================================================"
fi

# --- Step 4: FID Evaluation ---
# In protein_only mode, evaluate_fid.py auto-detects and only runs on protein channel.
# In all_channels mode, it runs on all channels.
echo ""
echo "============================================================"
echo "Step 4/6: Computing FID scores..."
echo "============================================================"

python vermeer/evaluations/ca_esm/evaluate_fid.py \
    --images_dir "$OUTPUT_DIR" \
    --output_json "${OUTPUT_DIR}/fid_results.json" \
    --batch_size 32

echo "Step 4 complete."

# --- Step 5: MSE Evaluation ---
# In protein_only mode, evaluate_mse.py auto-detects and only runs on protein channel.
# In all_channels mode, it runs on all channels.
echo ""
echo "============================================================"
echo "Step 5/6: Computing MSE scores..."
echo "============================================================"

python vermeer/evaluations/ca_esm/evaluate_mse.py \
    --images_dir "$OUTPUT_DIR" \
    --output_json "${OUTPUT_DIR}/mse_results.json"

echo "Step 5 complete."

# --- Step 6: Intra-Nuclear Proportion Evaluation (protein_only mode only) ---
if [ "$GENERATION_MODE" = "protein_only" ]; then
    echo ""
    echo "============================================================"
    echo "Step 6/6: Computing intra-nuclear proportion metrics..."
    echo "============================================================"

    python vermeer/evaluations/ca_esm/evaluate_intra_nuclear_prop.py \
        --images_dir "$OUTPUT_DIR" \
        --output_json "${OUTPUT_DIR}/intra_nuclear_prop_results.json"

    echo "Step 6 complete."
else
    echo ""
    echo "============================================================"
    echo "Step 6/6: Skipping intra-nuclear proportion (all_channels mode)"
    echo "============================================================"
fi

# --- Merge Results ---
echo ""
echo "============================================================"
echo "Merging results..."
echo "============================================================"

python -c "
import json, sys
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
generation_mode = '${GENERATION_MODE}'
combined = {}

files_to_merge = ['generation_metadata.json', 'protein_val_loss_results.json', 'fid_results.json', 'mse_results.json']
if generation_mode == 'protein_only':
    files_to_merge.extend(['subcell_results.json', 'intra_nuclear_prop_results.json'])

for name in files_to_merge:
    path = output_dir / name
    if path.exists():
        with open(path) as f:
            combined[name.replace('.json', '')] = json.load(f)
    else:
        print(f'WARNING: {path} not found', file=sys.stderr)

with open(output_dir / 'eval_results.json', 'w') as f:
    json.dump(combined, f, indent=2)

print(f'Combined results saved to: {output_dir}/eval_results.json')
"

# --- Summary ---
echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
echo "Output files:"
echo "  Images (true):      ${OUTPUT_DIR}/images_true/"
echo "  Images (generated): ${OUTPUT_DIR}/images_gen/"
echo "  Generation meta:    ${OUTPUT_DIR}/generation_metadata.json"
echo "  Protein val loss:   ${OUTPUT_DIR}/protein_val_loss_results.json"
echo "  SubCell results:    ${OUTPUT_DIR}/subcell_results.json"
echo "  FID results:        ${OUTPUT_DIR}/fid_results.json"
echo "  MSE results:        ${OUTPUT_DIR}/mse_results.json"
echo "  IntraNuc results:   ${OUTPUT_DIR}/intra_nuclear_prop_results.json"
echo "  Combined results:   ${OUTPUT_DIR}/eval_results.json"
echo "============================================================"
