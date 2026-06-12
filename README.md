# Vermeer

Autoregressive generative modeling of microscopy predicts protein localization.

Vermeer extends autoregressive generative modeling of natural images to multi-channel microscopy images. We train Vermeer on the Human Protein Atlas, using protein language model features of protein sequence as conditioning information.

Given the first three reference channels and a protein's ESM-C embedding, the model generates the corresponding fluorescent protein channel. `tutorial.ipynb` demonstrates this virtual staining task. To run this notebook, first download the example data and example model checkpoints (contains tokenizer checkpoint and vermeer-L checkpoint) from [vermeer_models_and_examples.zip](https://drive.google.com/file/d/1_sPZiUnn8yZFAsOopRxFH5_nzk-Tgiiw/view?usp=sharing). The provided yaml file `vermeer.yaml` can be used to create a conda/mamba environment to run this notebook.

![Unseen protein generation with attention](assets/unseen_protein_generation_w_attention.gif)

## Model Weights

Vermeer model checkpoints can be downloaded from HuggingFace:

| Model | Params | Checkpoint |
| --- | --- | --- |
| `vermeer_B` | 112M | [vermeer_B.ckpt](https://huggingface.co/microsoft/vermeer-B/blob/main/vermeer_B.ckpt) |
| `vermeer_L` | 344M | [vermeer_L.ckpt](https://huggingface.co/microsoft/vermeer-L/blob/main/vermeer_L.ckpt) |
| `vermeer_L_ap` | 344M | [vermeer_L_AP.ckpt](https://huggingface.co/microsoft/vermeer-L-AP/blob/main/vermeer_L_AP.ckpt) |
| `vermeer_XL` | 777M | [vermeer_XL.ckpt](https://huggingface.co/microsoft/vermeer-XL/blob/main/vermeer_XL.ckpt) |
| `vermeer_XL_CA` | 777M | [vermeer_XL_CA.ckpt](https://huggingface.co/microsoft/vermeer-XL-CA/blob/main/vermeer_XL_CA.ckpt) |

The pretrained tokenizer can be downloaded from the LlamaGen HuggingFace: [vq_ds16_c2i.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt)

Scripts for preparing the dataset load ESMC-600M through the `esmc` package, which automatically downloads the weights if they are not already cached. The weights can also be downloaded manually from Biohub HuggingFace: [esmc-600m.safetensors](https://huggingface.co/biohub/ESMC-600M/blob/main/model.safetensors)

## Installation

First clone the repository:

```bash
git clone https://github.com/microsoft/Vermeer.git
cd Vermeer
```

Then create and activate a conda/mamba environment using the provided `vermeer.yaml` file:

```bash
mamba env create -f vermeer.yaml
mamba activate vermeer
```

Finally, from the repository root, install Vermeer as an editable package:

```bash
pip install -e .
```

## Training/Fine-tuning Vermeer

To train a new model, first download the raw microscopy images from the Human Protein Atlas using the scripts `scripts/prepare_data/download_images_parallel.py` and `scripts/prepare_data/hpa_stratified_preprocessing_final.py`:

```bash
cd scripts/prepare_data
python download_images_parallel.py --output-dir <output_dir>
# specify the paths in the config file at the beginning of this script
python hpa_stratified_preprocessing_final.py --image_size 256
```

Then compute the ESM-C embeddings:

```bash
# From the repository root
# Update paths to those set in the config file in scripts/prepare_data/hpa_stratified_preprocessing_final.py
python vermeer/dataset/prepare_protein_prefix.py \
    --input_dir <input_dir> \
    --h5_filename protein_prefix.h5 \
    --metadata_dir <output_metadata_dir> \
    --device cuda
```
and tokenize all of the images:

```bash
# Update paths to those set in the config file in scripts/prepare_data/hpa_stratified_preprocessing_final.py and for vq-ckpt if necessary.
python vermeer/autoregressive/train/extract_codes_ca.py \
    --data-path <input_data_path> \
    --code-path <output_code_path> \
    --vq-ckpt vq_ds16_c2i.pt \
    --ten-crop \
    --rotate \
    --debug \
    --label-type esm_embed_mean_pool \
    --label-file protein_prefix.h5 \
    --n-channels 4 \
    --num-workers 4 \
    --image-size 256
```

Before training, download the checkpoint needed for your workflow:

- **Re-training Vermeer:** download a pretrained LlamaGen checkpoint from [LlamaGen](https://github.com/foundationvision/llamagen), such as [`c2i_L_256.pt`](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_L_256.pt) and pass it with `--pretrained-gpt_ckpt`
- **Fine-tuning Vermeer on new data:** download a Vermeer model checkpoint and pass it with `--gpt_ckpt`.

Then run the training script `vermeer/autoregressive/train/train_ca.py`:

```bash
# Update paths for results-dir (where model checkpoints are saved), code-path (input codes), and for --pretrained-gpt-ckpt / --gpt-ckpt if necessary. 
torchrun \
    --nnodes=1 --nproc_per_node=2 --node_rank=0 \
    --master_port=12334 \
    vermeer/autoregressive/train/train_ca.py \
    --results-dir <output_dir> \
    --val-dirs val1,val2,cell_line_holdouts \
    --code-path <code_path> \
    --image-size 256 \
    --gpt-model GPT-L \
    --num-workers 8 \
    --ckpt-every 5000 \
    --lr 1e-4 \
    --epochs 150 \
    --experiment-name "hpa_split_size_256" \
    --global-batch-size 96 \
    --val-every 1000 \
    --gpt-type ca_esm_embed_mean_pool \
    --pretrained-gpt-ckpt c2i_L_256.pt \
    --lr-schedule lin \
    --warmup-epochs 10
```

Evaluation can be run using the evaluation script, `scripts/eval/run_eval_pipeline.sh` and adjusting the necessary path names to point to the correct locations of data and model checkpoints. This wrapper calls several scripts that compute different evaluation  metrics (FID, mean-squared error, etc.). For a more detailed description of the various evaluation metrics, please refer to the preprint. 

## Acknowledgements

Built on [LlamaGen](https://github.com/FoundationVision/LlamaGen). Protein representations from [ESM-C](https://github.com/facebookresearch/esm). Microscopy data from the [Human Protein Atlas](https://www.proteinatlas.org/).
