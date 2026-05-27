# moPPIt: De Novo Generation of Motif-Specific Peptide Binders with Protein Language Models



![image/png](https://cdn-uploads.huggingface.co/production/uploads/64cd5b3f0494187a9e8b7c69/YxTTgvVen6xmZdoH9AoEO.png)

Motif-specific targeting of protein-protein interactions (PPIs) is crucial for developing highly selective therapeutics, yet remains a significant challenge in drug discovery. The ability to precisely target specific motifs or epitopes within these proteins is essential for modulating their function while minimizing off-target effects, but current methods struggle to achieve this specificity without structural information. In this work, we introduce a motif-specific PPI targeting algorithm, moPPIt, for de novo generation of motif-specific peptide binders using only protein sequence information. At the core of moPPIt is BindEvaluator, a transformer-based model that interpolates protein language model embeddings via a series of multi-headed self-attention blocks, with a key focus on local interaction changes. Trained on over 510,000 PPI-hotspot triplets from the PPIRef dataset, BindEvaluator accurately predicts binding hotspots between two proteins with a test AUC > 0.94, improving to AUC > 0.96 when fine-tuned on peptide-protein pairs. By combining BindEvaluator with our PepMLM peptide generator and genetic algorithm-based optimization, moPPIt generates peptides that bind specifically to user-defined motifs on target proteins.

---
**Model Checkpoints can be found in the Huggingface repo**: [Link](https://huggingface.co/ChatterjeeLab/moPPIt)

**Colab Notebook for Binding Site Prediction and Motif-Specific Binder Generation**: [Link](https://colab.research.google.com/drive/1SL3H_vI1y6qccce3vLOo0W2EpxIF4Xik?usp=sharing)

**Colab Notebook for Hugging Face MOG-DFM Generation**: [Link](https://colab.research.google.com/drive/16n8PIwKwAiG-oDLm171BWvv-lQH0dHMg?usp=sharing)

**Colab Notebook for PeptiDerive**: [Link](https://colab.research.google.com/drive/1aCODZ-WRwhxr-u8nEB6ZrdrhIOTz7-UF?usp=sharing)

---

# 0. Complete Local Setup

moPPIt is installable as a Python package. The BindEvaluator architecture names used by the checkpoints are preserved (`esm_model`, `repeated_module`, final attention/FFN layers, and output projection), so existing `.ckpt` weights can still be loaded for prediction and binder design as long as the model hyperparameters match the checkpoint. The published ChatterjeeLab/moPPIt checkpoints use the default `published` preset, so you should not need to pass architecture flags such as `--n-layers` for normal use.

There are three kinds of model assets to be aware of:

| Asset | Used by | How it is obtained | Recommended location |
| --- | --- | --- | --- |
| BindEvaluator checkpoint, usually `finetuned_BindEvaluator.ckpt` | `moppit-predict`, `moppit-generate`, motif/specificity scoring inside MOG-DFM | Download from the Hugging Face moPPIt repository or use your existing checkpoint | Recommended: `~/model_weights/moppit/finetuned_BindEvaluator.ckpt` with `MOPPIT_MODEL_DIR=~/model_weights/moppit`; exact path override: `MOPPIT_BINDEVALUATOR_CKPT` |
| Protein language models `facebook/esm2_t33_650M_UR50D` and `ChatterjeeLab/PepMLM-650M` | ESM embedding, prediction, PepMLM/GA binder design | Downloaded automatically by `transformers` on first use | Hugging Face cache, optionally controlled with `HF_HOME` |
| MOG-DFM solver, classifier assets, and `PeptiVerse/` | `moppit-mog-dfm` multi-objective binder design | Hugging Face moPPIt Git LFS assets plus a local PeptiVerse checkout/release | Recommended: `~/model_weights/moppit/moPPIt` with `MOPPIT_HF_ROOT=~/model_weights/moppit/moPPIt`; repo-local `moPPIt/` also works |

The local package exposes the same user-facing functionality as the Hugging Face version when the Hugging Face weights and PeptiVerse assets are present, but those large assets are not bundled in this repository.

## 0.1 Create the Environment

For Blackwell / CUDA 13+ machines, install a PyTorch build that matches the remote driver and CUDA runtime before installing moPPIt. This repository does not compile custom CUDA extensions; GPU compatibility comes from the installed PyTorch wheel or container. On remote machines, the most reliable path is usually a current NVIDIA/PyTorch container or the command from the official PyTorch install selector for your CUDA runtime.

```bash
conda env create -f environment.yml
conda activate moppit
python -m pip install --upgrade pip

# Install the PyTorch wheel or container-compatible build for your CUDA runtime.
# Use the official PyTorch selector for the exact CUDA 13.x command available for your release.
# If your remote image already provides a compatible torch build, skip this line.
python -m pip install torch torchvision torchaudio --index-url <PYTORCH_CUDA_13_WHEEL_INDEX>

# Install normal prediction, PepMLM, and GA binder design commands.
python -m pip install -e .
```

Install optional extras for the workflows you plan to run:

```bash
# Training and fine-tuning utilities.
python -m pip install -e ".[train]"

# Hugging Face MOG-DFM launcher dependencies.
python -m pip install -e ".[mogdfm]"

# Development tools.
python -m pip install -e ".[dev]"
```

`moppit-peptiderive` requires PyRosetta, which is not installed by pip from this package. Install PyRosetta separately under its own license before using PeptiDerive.

Verify the GPU build before running checkpoint prediction or generation:

```bash
python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
```

Verify that the console commands are installed:

```bash
moppit-predict --help
moppit-generate --help
pepmlm-generate --help
moppit-mog-dfm --help
```

## 0.2 Download and Place BindEvaluator Weights

For binding-site prediction and the default PepMLM/GA binder designer, you need a BindEvaluator checkpoint. The fine-tuned peptide-protein checkpoint is the usual choice. You do not need to keep weights in the repository; the recommended layout is an external model directory:

```bash
mkdir -p ~/model_weights/moppit
# Put your checkpoint here:
# ~/model_weights/moppit/finetuned_BindEvaluator.ckpt

# Add this to your shell profile or job script.
export MOPPIT_MODEL_DIR=~/model_weights/moppit
```

moPPIt searches for checkpoints in this order:

1. The explicit `--model` / `-sm` argument.
2. The `MOPPIT_BINDEVALUATOR_CKPT` environment variable.
3. The explicit `--model-dir` argument.
4. The `MOPPIT_MODEL_DIR` or `MOPPIT_MODEL_WEIGHTS_DIR` environment variable.
5. `MOPPIT_HF_ROOT/classifier_ckpt/`, if `MOPPIT_HF_ROOT` is set.
6. `~/model_weights/moppit/finetuned_BindEvaluator.ckpt`.
7. `~/model_weights/moppit/pretrained_BindEvaluator.ckpt`.
8. Repo-local fallbacks: `model_path/*.ckpt`, `classifier_ckpt/finetuned_BindEvaluator.ckpt`, and `moPPIt/classifier_ckpt/finetuned_BindEvaluator.ckpt`.

When searching a directory, moPPIt first looks for `finetuned_BindEvaluator.ckpt` and then `pretrained_BindEvaluator.ckpt`. If neither exists and the directory contains exactly one `.ckpt` file, that file is used. If there are multiple `.ckpt` files with non-standard names, pass `--model /path/to/file.ckpt` or rename the desired file to one of the standard names.

Use an exact file path if you want complete control:

```bash
export MOPPIT_BINDEVALUATOR_CKPT=/absolute/path/to/finetuned_BindEvaluator.ckpt
```

Or use `--model-dir` for a one-off command without setting an environment variable:

```bash
moppit-predict \
  --model-dir ~/model_weights/moppit \
  --target TARGET_PROTEIN_SEQUENCE \
  --binder BINDER_SEQUENCE
```

To download the Hugging Face moPPIt assets outside the repository, use Git LFS:

```bash
mkdir -p ~/model_weights/moppit
git lfs install
git clone https://huggingface.co/ChatterjeeLab/moPPIt ~/model_weights/moppit/moPPIt
cd ~/model_weights/moppit/moPPIt
git lfs pull
cd ..

# Let moPPIt commands discover the external HF clone and its BindEvaluator checkpoint.
export MOPPIT_HF_ROOT=~/model_weights/moppit/moPPIt
```

You can either let `MOPPIT_HF_ROOT` point at the external clone or copy/symlink the fine-tuned checkpoint into your model directory:

```bash
ln -sf ~/model_weights/moppit/moPPIt/classifier_ckpt/finetuned_BindEvaluator.ckpt \
  ~/model_weights/moppit/finetuned_BindEvaluator.ckpt
```

Validate that the checkpoint is a real weight file, not a Git LFS pointer:

```bash
python - <<'PY'
from moppit.bindevaluator import resolve_checkpoint_path, validate_checkpoint_path

print(validate_checkpoint_path(resolve_checkpoint_path()))
PY
```

If a checkpoint is still a Git LFS pointer, `moppit-predict` and `moppit-generate` fail early with a clear message instead of a PyTorch deserialization traceback.

Some current `transformers` versions print an ESM load report about `lm_head.*` keys being unexpected and `pooler.*` keys being missing when `facebook/esm2_t33_650M_UR50D` is loaded as an encoder. That report is expected for this workflow. Older BindEvaluator checkpoints may also contain the frozen ESM key `esm_model.embeddings.position_embeddings.weight`; moPPIt ignores that obsolete key only when the installed `EsmModel` no longer has it, while keeping strict loading for the BindEvaluator model head.

## 0.3 Pre-Download Language Models on a Cluster

`transformers` downloads ESM2 and PepMLM automatically on first use. On clusters or offline jobs, it is better to stage them in advance. Set `HF_HOME` to a persistent cache if your compute node home directory is small:

```bash
export HF_HOME=/path/to/persistent/huggingface-cache

python - <<'PY'
from transformers import AutoModelForMaskedLM, AutoTokenizer, EsmModel

AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
AutoTokenizer.from_pretrained("ChatterjeeLab/PepMLM-650M")
AutoModelForMaskedLM.from_pretrained("ChatterjeeLab/PepMLM-650M")
print("language models cached")
PY
```

These language models are separate from the BindEvaluator `.ckpt` weights. You still need `finetuned_BindEvaluator.ckpt` for prediction and BindEvaluator-guided design.

## 0.4 Optional MOG-DFM / Hugging Face Feature-Parity Assets

Use `moppit-mog-dfm` for the newer Hugging Face Multi-Objective-Guided Discrete Flow Matching workflow. The checked-in Hugging Face script implements hemolysis, non-fouling, solubility, permeability, half-life, affinity, motif, and specificity objectives, and the launcher passes through upstream flags such as `--offtarget` for compatibility. This workflow requires the Hugging Face clone and a local PeptiVerse checkout/release because the upstream script imports `./PeptiVerse/inference.py` and `./PeptiVerse/best_models.txt` at startup.

Expected layout:

```text
~/model_weights/moppit/
  moPPIt/
    moppit.py
    ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt
    classifier_ckpt/finetuned_BindEvaluator.ckpt
    classifier_ckpt/best_model_half_life.pth
    classifier_ckpt/best_model_nonfouling.json
    classifier_ckpt/best_model_solubility.json
    classifier_ckpt/binding_affinity_pooled.pt
    classifier_ckpt/binding_affinity_unpooled.pt
    classifier_ckpt/wt_affinity.pt
    classifier_ckpt/wt_halflife.pt
    classifier_ckpt/wt_hemolysis.json
    classifier_ckpt/wt_nonfouling.pt
    PeptiVerse/
      inference.py
      best_models.txt
      training_classifiers/...
```

Set it up with:

```bash
python -m pip install -e ".[mogdfm]"

mkdir -p ~/model_weights/moppit
git lfs install
git clone https://huggingface.co/ChatterjeeLab/moPPIt ~/model_weights/moppit/moPPIt
cd ~/model_weights/moppit/moPPIt
git lfs pull
cd ..

export MOPPIT_HF_ROOT=~/model_weights/moppit/moPPIt

# Place or clone PeptiVerse here so these files exist:
# ~/model_weights/moppit/moPPIt/PeptiVerse/inference.py
# ~/model_weights/moppit/moPPIt/PeptiVerse/best_models.txt
```

Validate the MOG-DFM setup without generating peptides:

```bash
moppit-mog-dfm --dry-run \
  --output_file samples.csv \
  --length 10 \
  --objectives Hemolysis Motif Specificity \
  --target_protein MHVPSGAQLGLRPDLLARRRLKRCPSRWLCLSAAWSFVQVFSEPDGFTVIFSGLGNNAGGTMHWNDTRPAHFRILKVVLREAVAECLMDSYSLDVHGGRRTAAG
```

If an asset is still a Git LFS pointer or `PeptiVerse/` is missing, the command reports all missing prerequisites before attempting to import the HF script. The preflight also checks bundled Hugging Face classifier assets under `moPPIt/classifier_ckpt/` when they are present, so stale LFS pointer files are caught before generation starts.

You can keep the Hugging Face clone somewhere else by passing `--hf-root /path/to/moPPIt` or setting:

```bash
export MOPPIT_HF_ROOT=/path/to/moPPIt
```

## 0.5 First Commands After Setup

These examples assume you set `MOPPIT_MODEL_DIR=~/model_weights/moppit` and, for MOG-DFM, `MOPPIT_HF_ROOT=~/model_weights/moppit/moPPIt`. Add `--model-dir ~/model_weights/moppit` or `--model /absolute/path/to/file.ckpt` to any prediction or GA generation command if you prefer not to use environment variables.

Predict binding residues for a known target/binder pair:

```bash
moppit-predict \
  --target TARGET_PROTEIN_SEQUENCE \
  --binder BINDER_SEQUENCE \
  --motifs 18,23,59-61 \
  --output predicted_sites.json
```

Design motif-specific binders with the default PepMLM plus genetic-algorithm workflow:

```bash
moppit-generate \
  --protein-seq TARGET_PROTEIN_SEQUENCE \
  --peptide-length 11 \
  --motif 18,23,59-61 \
  --num-binders 50 \
  --num-display 10 \
  --output generated_binders.csv
```

Design binders with the Hugging Face MOG-DFM multi-objective workflow:

```bash
moppit-mog-dfm \
  --output_file samples.csv \
  --length 10 \
  --n_batches 600 \
  --weights 1 1 1 4 4 2 \
  --motifs '16-31,62-79' \
  --motif_penalty \
  --objectives Hemolysis Non-Fouling Half-Life Affinity Motif Specificity \
  --target_protein TARGET_PROTEIN_SEQUENCE
```

Use 0-based protein residue indices for `moppit-predict`, `moppit-generate`, and `moppit-mog-dfm` unless you explicitly pass `--motif-index-base 1` to `moppit-predict` for outside annotations.

## 0.6 Hugging Face Implementation Notes

The Hugging Face repository is newer than the original GitHub code. The BindEvaluator architecture and module weights are unchanged, so this package keeps the same checkpoint-compatible predictor and adds the safe inference improvements from the Hugging Face code: published 8-layer defaults, `weights_only=False` checkpoint loading, frozen/eval inference loading, motif scoring helpers, and support for the Hugging Face checkpoint path.

The Hugging Face generator is a different research pipeline: Multi-Objective-Guided Discrete Flow Matching with PeptiVerse-guided objectives. It is exposed as `moppit-mog-dfm` so the packaged repo has the same generation feature surface while keeping the lighter PepMLM plus genetic-algorithm generator as `moppit-generate`.

# 1. Dataset Preparation

Pre-training dataset: `dataset/pretrain_dataset.csv`

Fine-tuning dataset: `dataset/finetune_dataset.csv`

To accelerate training and fine-tuning, datasets need to be processed  into HuggingFace Dataset in advance.

Before pre-training, run:
```
python dataset/pretrain_preprocessing.py -dataset_pth dataset/pretrain_dataset.csv -output_dir dataset
```

Before fine-tuning, run:
```
python dataset/pretrain_preprocessing.py -dataset_pth dataset/finetune_dataset.csv -output_dir dataset
```

The processed datasets will be saved in `output_dir` 

# 2. Model Training and Fine-tuning

To train BindEvaluator with dilated CNN modules, run `scripts/train.sh`

To fine-tune the pre-trained BindEvaluator, run `scripts/finetune.sh`

To test the performance of BindEvaluator, run `scripts/test.sh`

`scripts/finetune.sh` and `scripts/test.sh` default to checkpoints under `${MOPPIT_MODEL_DIR:-$HOME/model_weights/moppit}`. Override the exact pretrained checkpoint for fine-tuning with `MOPPIT_PRETRAINED_BINDEVALUATOR_CKPT`, or override the exact fine-tuned checkpoint for testing/prediction with `MOPPIT_BINDEVALUATOR_CKPT`.

Ensure you adjust the hyper-parameters according to your specific requirements.

# 3. Binding site prediction

Protein-protein interaction binding sites can be predicted using the pre-trained BindEvaluator, usually named `pretrained_BindEvaluator.ckpt` in your model directory.

Peptide-protein interaction binding sites can be predicted using the fine-tuned BindEvaluator, usually named `finetuned_BindEvaluator.ckpt` in your model directory.

We provide an example script to use BindEvaluator to predict binding sites (`scripts/predict.sh`). After installation, use the `moppit-predict` console command. The legacy `python predict_motifs.py` wrapper is still available from the repository checkout.

NOTE: amino acid indices start from 0 on a protein sequence

The published ChatterjeeLab/moPPIt BindEvaluator checkpoints use the default `published` architecture preset (`n_layers=8`, `d_model=128`, `d_hidden=128`, `n_head=8`, `d_inner=64`), so these values do not need to be provided for normal prediction. Use `--checkpoint-preset legacy` or the explicit architecture override flags only for non-published checkpoints.

``` txt
usage: moppit-predict [--model MODEL_PATH] [--model-dir MODEL_DIR] --target TARGET --binder BINDER
                      [--threshold THRESHOLD] [--output OUTPUT.json]
                      [--device auto|cpu|cuda|cuda:0]
                      [--checkpoint-preset published|legacy]
                      [--ground-truth MOTIF] [--motifs MOTIF]
                      [architecture overrides]

arguments:
  --model, -sm         Exact path to the BindEvaluator model weights. Overrides checkpoint directory discovery.
  --model-dir          Directory containing BindEvaluator checkpoints. If omitted, moPPIt checks MOPPIT_MODEL_DIR, MOPPIT_MODEL_WEIGHTS_DIR, ~/model_weights/moppit, MOPPIT_HF_ROOT/classifier_ckpt, and repo-local fallback paths.
  --target, -target    Target protein sequence
  --binder, -binder    Binder sequence
  --ground-truth, -gt  Ground truth binding motifs if known, for example 18,23,59-61. Brackets are optional.
  --motifs, -motifs    Motif residues to score, using the same motif syntax.
  --motif-index-base   Index base for --motifs and --ground-truth. Defaults to 0. Use 1 for 1-based motif annotations.
  --threshold          Binding-site probability threshold, default 0.5
  --output             Optional JSON file containing predicted residues and per-residue scores
  --print-scores       Print per-residue prediction probabilities
  --device             Torch device, default auto
  --checkpoint-preset  Architecture preset, default published
  --n-layers, --d-model, --d-hidden, --n-head, --d-inner   Optional overrides for unusual checkpoints
```

Example:

```bash
moppit-predict \
  --target IVEGSDAEIGMSPWQVMLFRKSPQELLCGASLISDRWVLTAAHCLLYPPWDKNFTENDLLVRIGKHSRTRYERNIEKISMLEKIYIHPRYNWRENLDRDIALMKLKKPVAFSDYIHPVCLPDRETAASLLQAGYKGRVTGWGNLKETGQPSVLQVVNLPIVERPVCKDSTRIRITDNMFCAGYKPDEGKRGDACEGDSGGPFVMKSPFNNRWYQMGIVSWGEGCDRDGKYGFYTHVFRLKKWIQKVIDQFGE \
  --binder GYEEIPEEYLQ \
  --motifs 18,23,59,67,68,69,70,76,77 \
  --output predicted_sites.json
```

# 4. Motif-Specific Binder Generation

We provide an example script to use moPPIt for generating motif-specific binders based on a target sequence (`scripts/generation.sh`). After installation, use the `moppit-generate` console command. The legacy `python moppit.py` wrapper is still available from the repository checkout.
``` txt
usage: moppit-generate [--model MODEL_PATH] [--model-dir MODEL_DIR] --protein-seq PROTEIN --peptide-length LENGTH --motif MOTIF
                       [--top-k TOP_K] [--num-binders NUM_BINDERS]
                       [--num-display NUM_DISPLAY] [--max-iterations MAX_ITERATIONS]
                       [--threshold THRESHOLD] [--output OUTPUT.csv]
                       [--device auto|cpu|cuda|cuda:0]
                       [--checkpoint-preset published|legacy]
                       [architecture overrides]

arguments:
  --model, -sm         Exact path to the BindEvaluator model weights. Overrides checkpoint directory discovery.
  --model-dir          Directory containing BindEvaluator checkpoints. If omitted, moPPIt checks MOPPIT_MODEL_DIR, MOPPIT_MODEL_WEIGHTS_DIR, ~/model_weights/moppit, MOPPIT_HF_ROOT/classifier_ckpt, and repo-local fallback paths.
  --protein-seq        Target protein sequence. The old --protein_seq spelling is also accepted.
  --peptide-length     The length for generated binders. The old --peptide_length spelling is also accepted.
  --motif              Binding motifs with 0-based indices, for example 18,23,59-61. Brackets are optional.
  --top-k              Sampling argument for each position used in PepMLM
  --num-binders        The size of the candidate pool in the genetic algorithm
  --num-display        The number of top binders to display and write after each generation
  --max-iterations     Maximum no-improvement iterations
  --threshold          Binding-site probability threshold used by the motif score, default 0.5
  --output             Optional CSV file containing the final displayed binders, scores, and pseudo-perplexities
  --device             Torch device, default auto
  --checkpoint-preset  Architecture preset, default published
  --n-layers, --d-model, --d-hidden, --n-head, --d-inner   Optional overrides for unusual checkpoints
```

Example:

```bash
moppit-generate \
  --protein-seq IVEGSDAEIGMSPWQVMLFRKSPQELLCGASLISDRWVLTAAHCLLYPPWDKNFTENDLLVRIGKHSRTRYERNIEKISMLEKIYIHPRYNWRENLDRDIALMKLKKPVAFSDYIHPVCLPDRETAASLLQAGYKGRVTGWGNLKETGQPSVLQVVNLPIVERPVCKDSTRIRITDNMFCAGYKPDEGKRGDACEGDSGGPFVMKSPFNNRWYQMGIVSWGEGCDRDGKYGFYTHVFRLKKWIQKVIDQFGE \
  --peptide-length 11 \
  --motif 18,23,59,67,68,69,70,76,77 \
  --num-binders 50 \
  --num-display 10 \
  --output generated_binders.csv
```

`--motif` uses 0-based protein residue indices. This matches the original local generator and the Hugging Face MOG-DFM motif parser used by `moppit-mog-dfm`.

The standalone PepMLM helper also accepts the same hyphenated style and can write CSV output:

```bash
pepmlm-generate \
  --sequence IVEGSDAEIGMSPWQVMLFRKSPQELLCGASLISDRWVLTAAHCLLYPPWDKNFTENDLLVRIGKHSRTRYERNIEKISMLEKIYIHPRYNWRENLDRDIALMKLKKPVAFSDYIHPVCLPDRETAASLLQAGYKGRVTGWGNLKETGQPSVLQVVNLPIVERPVCKDSTRIRITDNMFCAGYKPDEGKRGDACEGDSGGPFVMKSPFNNRWYQMGIVSWGEGCDRDGKYGFYTHVFRLKKWIQKVIDQFGE \
  --peptide-length 11 \
  --top-k 3 \
  --num-binders 50 \
  --output pepmlm_binders.csv
```

## 4.1 Multi-Objective Flow-Matching Generation

For parity with the newer Hugging Face implementation, use `moppit-mog-dfm` to run the MOG-DFM workflow from the local `moPPIt/` clone. It accepts the Hugging Face arguments directly, including `--fixed_positions`, `--cyclic`, `--starting_sequence`, and `--offtarget`. The documented `--motif_penalty` spelling is normalized to the `Specificity` objective.

```bash
moppit-mog-dfm \
  --hf-root moPPIt \
  --output_file './samples.csv' \
  --length 10 \
  --n_batches 600 \
  --weights 1 1 1 4 4 2 \
  --motifs '16-31,62-79' \
  --motif_penalty \
  --objectives Hemolysis Non-Fouling Half-Life Affinity Motif Specificity \
  --target_protein MHVPSGAQLGLRPDLLARRRLKRCPSRWLCLSAAWSFVQVFSEPDGFTVIFSGLGNNAGGTMHWNDTRPAHFRILKVVLREAVAECLMDSYSLDVHGGRRTAAG
```

The legacy Hugging Face-style `python -u moo.py ...` command also works from this repository checkout and forwards to `moppit-mog-dfm`.

MOG-DFM motif and fixed-position ranges are also interpreted as 0-based indices by the Hugging Face generation code. The prediction command can additionally accept 1-based motif lists with `--motif-index-base 1` when comparing against outside annotations.


# 5. PeptiDerive

We provide the Python script to run PeptiDerive locally. 

`pyrosetta` needs to be installed in the conda environment before running this script. ([Installation Guideline](https://www.pyrosetta.org/downloads#h.c0px19b8kvuw))

NOTE: In PeptiDerive results, amino acid indices start from 1 on protein sequences.
``` txt
usage: moppit-peptiderive --pdb PDB_PATH [--binder_chain]

arguments:
  --pdb             The path to the binder-target protein complex structure
  --binder_chain    Whether the binder is chain A or chain B in the protein complex structure
```

---
The newer Hugging Face model card declares `apache-2.0`; verify the current upstream terms for the exact checkpoint or asset you use.

## Repository Authors

[Tong Chen](mailto:tong.chen2@duke.edu), Visiting Student at Duke University <br>
[Pranam Chatterjee](mailto:pranam.chatterjee@duke.edu), Assistant Professor at Duke University 

Reach out to us with any questions!
