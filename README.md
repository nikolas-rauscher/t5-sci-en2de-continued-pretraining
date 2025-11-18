<div align="center">

# T5 Scientific English-to-German Continued Pretraining

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>

</div>

## Description

This repository contains the code and configurations for developing a German scientific language model through cross-lingual transfer from English. The approach combines:

1. **Domain Adaptation**: Continued pretraining of T5-base on a curated English scientific corpus (Unpaywall/Scilons)
2. **Cross-Lingual Transfer**: Using WECHSEL methodology to transfer learned knowledge to German

The domain-adapted English model (EN-T5-Sci) achieves +4.0 percentage points improvement on Global-MMLU compared to the base T5 model. The transferred German model (DE-T5-Sci-Transfer-15k) outperforms monolingual German baselines on scientific benchmarks.

## Released Models

| Model | Description | HuggingFace |
|-------|-------------|-------------|
| **EN-T5-Sci** | English T5-base with 487k steps continued pretraining on scientific text | [rausch/en-t5-sci-continued-pretraining-487k](https://huggingface.co/rausch/en-t5-sci-continued-pretraining-487k) |
| **DE-T5-Sci-Transfer-Init** | German model after WECHSEL transfer (before alignment) | [rausch/de-t5-sci-transfer-init](https://huggingface.co/rausch/de-t5-sci-transfer-init) |
| **DE-T5-Sci-Transfer-15k** | Final German model via WECHSEL + 15k alignment steps | [rausch/de-t5-sci-transfer-15k](https://huggingface.co/rausch/de-t5-sci-transfer-15k) |
| **DE-T5-Base-15k** | German baseline with 15k steps continued pretraining | [rausch/de-t5-base-continued-15k](https://huggingface.co/rausch/de-t5-base-continued-15k) |

**Datasets**

| Dataset | Description | HuggingFace |
|---------|-------------|-------------|
| Scientific Corpus (Cleaned) | Cleaned English scientific corpus from Unpaywall | [rausch/scientific_corpus_cleaned](https://huggingface.co/datasets/rausch/scientific_corpus_cleaned) |
| Sliding Windows | Preprocessed 512-token windows for training | [rausch/scientific_sliding_windows_en512](https://huggingface.co/datasets/rausch/scientific_sliding_windows_en512) |

## Key Results

### Zero-Shot Global-MMLU Evaluation

**English Models (EN split)**

| Model | Average | STEM | Humanities | Social Sci. | Other |
|-------|---------|------|------------|-------------|-------|
| EN-T5-Base | 22.9% | 21.3% | 24.1% | 21.7% | 23.9% |
| EN-T5-Sci | **26.9%** | **28.5%** | 24.2% | **31.1%** | 25.1% |
| Improvement | +4.0 pp | +7.2 pp | +0.1 pp | +9.4 pp | +1.2 pp |

**German Models (DE split)**

| Model | Average | STEM | Humanities | Social Sci. | Other |
|-------|---------|------|------------|-------------|-------|
| DE-T5-Base | 22.9% | 21.3% | 24.0% | 21.8% | 23.9% |
| DE-T5-Sci-Transfer-15k | **27.0%** | **28.5%** | **25.4%** | **30.6%** | 24.4% |
| Improvement | +4.1 pp | +7.2 pp | +1.4 pp | +8.8 pp | +0.5 pp |

The domain-adapted models show the strongest improvements in STEM (+7.2 pp) and Social Sciences (+9.4 pp EN, +8.8 pp DE).

### Training Configuration

**English Pretraining**
- **Steps**: 487k steps on ~261M sliding windows (512 tokens, 50% overlap)
- **Optimizer**: Adafactor with inverse square-root schedule
- **Learning Rate**: 0.001 peak with 20k warmup steps
- **Gradient Clipping**: 0.5
- **Batch Size**: 384 effective (48 per GPU × 4 GPUs × 2 accumulation)
- **Hardware**: 4× NVIDIA H100 (80GB VRAM)

**German Alignment Training** (post-WECHSEL transfer)
- **Steps**: 15k steps on German scientific data
- **Learning Rate**: 0.001 with 1.5k warmup steps
- **Gradient Clipping**: 1.0
- **Batch Size**: 48 (single GPU)

## Project Structure

```
├── configs/                    # Hydra configuration files
│   ├── experiment/            # Training experiment configs
│   │   ├── t5_continued_pretraining_*.yaml
│   │   ├── german_*.yaml
│   │   └── archive/           # Older experiment configs
│   ├── data/                  # Data configs
│   ├── model/                 # Model configs
│   └── trainer/               # Trainer configs
│
├── src/                       # Source code
│   ├── data/                  # Data loading (T5DataModule, collators)
│   ├── dataprep/              # Data preprocessing
│   │   ├── pipelines/         # Main preprocessing scripts
│   │   │   ├── clean_data.py  # Text cleaning pipeline
│   │   │   └── run_sliding_windows.py
│   │   ├── citation_cleaner*.py
│   │   ├── language_cleaner.py
│   │   └── text_normalizer.py
│   ├── models/                # Model implementations
│   ├── train.py               # Training entry point
│   ├── eval_pipeline.py       # Evaluation pipeline
│   └── utils/                 # Utility functions
│
├── scripts/                   # Utility scripts
│   ├── convert_models.py      # Model conversion utilities
│   └── ...
│
├── jobs/                      # SLURM job scripts
│   ├── training/              # Training jobs by experiment
│   │   ├── adafactor/
│   │   ├── lr_001_OPTIMIZED_clean_restart_v2/
│   │   ├── scifive/
│   │   └── ...
│   ├── eval/                  # Evaluation jobs
│   ├── stats/                 # Statistics jobs
│   ├── data_processing/       # Data processing jobs
│   └── misc/                  # Miscellaneous jobs
│
├── evaluation/                # Evaluation code
│
├── evaluation_results/        # Evaluation outputs
│   ├── scientific_crosslingual_transfer_eval_full_15k/  # Final results
│   ├── finetuning_results/    # Fine-tuning results
│   └── archive/               # Older evaluation results
│
├── cross_lingual_transfer/    # Cross-lingual transfer code
│
├── notebooks/                 # Jupyter notebooks for analysis
│
└── tests/                     # Unit tests
```

## Installation

### Conda (Recommended)

```bash
# Clone project
git clone https://github.com/nikolas-rauscher/t5-sci-en2de-continued-pretraining
cd t5-sci-en2de-continued-pretraining

# Create conda environment
conda env create -f environment.yaml -n t5-sci
conda activate t5-sci
```

### Pip

```bash
# Clone project
git clone https://github.com/nikolas-rauscher/t5-sci-en2de-continued-pretraining
cd t5-sci-en2de-continued-pretraining

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch (adjust for your CUDA version)
# https://pytorch.org/get-started/

# Install requirements
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

The preprocessing pipeline uses DataTrove for efficient processing of the scientific corpus:

```bash
# Run data cleaning via SLURM job
sbatch jobs/data_processing/run_clean_data_h100.sh

# Or run locally
python src/dataprep/pipelines/clean_data.py

# Create sliding windows for training
python src/dataprep/pipelines/run_sliding_windows.py
```

### Continued Pretraining

Train the English scientific model:

```bash
# Run training via SLURM job (with smart resume)
sbatch jobs/training/lr_001_OPTIMIZED_clean_restart_v2/smart_resume_t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart_v2.sh

# Or run locally
python src/train.py experiment=t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart_v2
```

### Cross-Lingual Transfer (WECHSEL)

Transfer the trained English model to German:

```bash
# Apply WECHSEL transfer
python cross_lingual_transfer/scripts/wechsel_transfer.py \
    --source_model path/to/en_t5_sci \
    --target_tokenizer GermanT5/t5-efficient-gc4-german-base-nl36

# Prepare German data
python cross_lingual_transfer/scripts/prepare_german_sliding_windows.py

# Continue pretraining on German data
python src/train.py experiment=german_continued_pretraining
```

### Evaluation

Evaluate models on Global-MMLU using the [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
# Run evaluation via SLURM job
sbatch jobs/eval/flexible_eval_24h.sh scientific_crosslingual_transfer_eval_full_15k

# Or run locally
python src/eval_pipeline.py experiment=scientific_crosslingual_transfer_eval_full_15k

# Analyze results
python evaluation/analysis_scripts/analyze_evaluation_results.py
```

For task-specific fine-tuning evaluation (NER, relation extraction, etc.), use the [scilons-eval](https://github.com/scilons/scilons-eval) pipeline.

## Data

### Corpus Statistics

- **Source**: Unpaywall database (~10M scientific articles)
- **Size**: ~230 GB (English portion after deduplication)
- **Processing**: GROBID for PDF-to-text conversion

### Preprocessing Pipeline

The cleaning pipeline (implemented with DataTrove) systematically removes artifacts while preserving scientific content:

**Citation and Identifier Removal** (184M instances removed)
- Numeric brackets: `[1]`, `[1-3]` (128M removed)
- Author-year: `(Smith, 2020)` (28M removed)
- Semicolon-separated lists: `Miller ; Davis ; Brown`
- URLs, DOIs, ISBNs, arXiv IDs

**Structural Cleaning**
- Figure/table captions and fragments (33M lines removed)
- Headers, footers, page markers
- Duplicate lines within lookback window

**Language Filtering**
- FastText English confidence threshold ≥ 0.75

**Normalization**
- Unicode harmonization (preserve scientific symbols: ≤, ≥, ×, Greek letters)
- Whitespace and newline standardization

**Results**
- Documents: 10,030,761 → 10,027,111 (-0.04%)
- Mean document length: 28,627 → 22,319 chars (-22%)
- Mean perplexity: 569.5 → 538.9 (-5.4%)

## Acknowledgments

- Based on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)
- Scientific corpus from the [DFKI Scilons](https://huggingface.co/scilons) project
- Cross-lingual transfer using [WECHSEL](https://github.com/CPJKU/wechsel)

## License

This project is licensed under the MIT License.
