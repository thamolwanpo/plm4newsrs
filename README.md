# PLM4NewsRS: Pre-trained Language Models for News Recommendation

A research framework for neural news recommendation. Supports four SOTA architectures (Simple, NAML, NRMS, LSTUR) with BERT, RoBERTa, or GloVe backends, with tools to study models under clean and poisoned training data conditions.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Highlights

- **Four Model Architectures**: Simple (two-tower), NAML (multi-view), NRMS (self-attention), LSTUR (long/short-term user modeling)
- **Multiple PLM Backends**: BERT, RoBERTa, or GloVe embeddings
- **Flexible Training**: Fine-tune or freeze PLM layers
- **Poisoning Support**: Train on clean or poisoned data to study misinformation contamination
- **Rich Evaluation**: AUC, MRR, NDCG, misinformation exposure metrics, and dedicated analyzers
- **PyTorch Lightning**: Checkpointing, early stopping, reproducible seeds

---

## Table of Contents

- [Installation](#installation)
- [Data Format](#data-format)
- [Quick Start](#quick-start)
- [Model Architectures](#model-architectures)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended, 8GB+ VRAM for BERT/RoBERTa)

```bash
# Clone the repository
git clone https://github.com/yourusername/plm4newsrs.git
cd plm4newsrs

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

---

## Data Format

The framework expects pairwise CSV files per split:

```
{base_dir}/
└── {experiment_name}/
    ├── models/
    │   ├── train_clean.csv
    │   ├── val_clean.csv
    │   ├── train_poisoned.csv   # optional
    │   └── val_poisoned.csv     # optional
    ├── benchmarks/
    │   ├── benchmark_real_only.csv
    │   ├── benchmark_mixed.csv
    │   └── benchmark_honeypot.csv
    └── news_items.csv           # full-text lookup by item_id
```

**Pairwise CSV columns:**
```csv
user_id,history_ids,candidate_id,candidate_title,label,candidate_is_fake
user_1,"[1,2,3]",news_123,"Breaking News Title",1,0
user_1,"[1,2,3]",news_456,"Misleading Story",0,1
```

- `history_ids`: JSON list of previously read item IDs (looked up in `news_items.csv`)
- `label`: 1 = clicked (positive), 0 = not clicked (negative)
- `candidate_is_fake`: 1 = misinformation, 0 = real news

Benchmark files use the same format, evaluated with a strict 1 positive + 99 negatives per group.

For LSTUR, a `user_to_idx_map.json` mapping user IDs to integer indices is also required.

---

## Quick Start

### Train a model
```bash
# Simple model with BERT (fine-tune) — trains clean + poisoned splits
python scripts/train.py --config configs/experiments/simple/bert_finetune.yaml --train-all

# NRMS with BERT
python scripts/train.py --config configs/experiments/nrms/nrms_bert_finetune.yaml --train-all

# NAML with GloVe
python scripts/train.py --config configs/experiments/naml/naml_glove.yaml --train-all

# Train only the clean model type
python scripts/train.py --config configs/experiments/nrms/nrms_bert_finetune.yaml --model-type clean
```

### Evaluate a trained model
```bash
python scripts/evaluate.py --config configs/experiments/nrms/nrms_bert_finetune.yaml

python scripts/evaluate.py --config configs/experiments/nrms/nrms_bert_finetune.yaml \
    --models clean poisoned \
    --analyses exposure failure
```

### Use Makefile shortcuts
```bash
make train-simple-bert-ft      # Simple + BERT fine-tune (all types)
make train-nrms-all            # All NRMS variants
make train-naml-all            # All NAML variants
make evaluate CONFIG=configs/experiments/nrms/nrms_bert_finetune.yaml
```

---

## Model Architectures

| Model | Paper | Key Innovation | Supported PLMs |
|-------|-------|----------------|---------------|
| **Simple** | — | Two-tower with additive attention | BERT, RoBERTa, GloVe |
| **NAML** | IJCAI'19 | Multi-view: title + body + category CNNs | BERT, RoBERTa, GloVe |
| **NRMS** | EMNLP'19 | Multi-head self-attention for news & user | BERT, RoBERTa, GloVe |
| **LSTUR** | — | Long/short-term user interest with masking | GloVe |

### Simple: Two-Tower with Attention

```
History Articles → PLM → Multi-Head Attention → Additive Attention → User Vector
                                                                           ↓  dot product
Candidates       → PLM → Multi-Head Attention → Additive Attention → News Vector
```

**When to use:** Baseline experiments, limited compute, fast iteration.

### NAML: Neural News Recommendation with Attentive Multi-View Learning

```
Title    → CNN  ─┐
Body     → CNN  ─┤→ View-Level Attention → News Vector
Category → Emb  ─┘
```

**When to use:** Rich metadata available (categories, body text), interpretable view weights needed.

**Reference:**
```bibtex
@inproceedings{wu2019naml,
  title={Neural News Recommendation with Attentive Multi-View Learning},
  author={Wu, Chuhan and Wu, Fangzhao and Ge, Suyu and Qi, Tao and Huang, Yongfeng and Xie, Xing},
  booktitle={IJCAI},
  pages={3863--3869},
  year={2019}
}
```

### NRMS: Neural News Recommendation with Multi-Head Self-Attention

```
News Encoder:  Tokens → Multi-Head Self-Attention → Additive Attention → News Vector
User Encoder:  Clicked News Vectors → Multi-Head Self-Attention → Additive Attention → User Vector
```

**When to use:** Complex interaction patterns, long histories, best performance.

**Reference:**
```bibtex
@inproceedings{wu2019nrms,
  title={Neural News Recommendation with Multi-Head Self-Attention},
  author={Wu, Fangzhao and Qiao, Ying and Chen, Jiun-Hung and Wu, Chuhan and Qi, Tao and Lian, Jianxun and Liu, Danyang and Xie, Xing and Gao, Jianfeng and Wu, Winnie and Zhou, Ming},
  booktitle={EMNLP-IJCNLP},
  pages={6389--6394},
  year={2019}
}
```

### LSTUR: Long- and Short-Term User Representation

Maintains a persistent long-term user embedding alongside a session-based user encoder. Applies random masking during training to balance the two representations.

**When to use:** User identity is stable across sessions; long-term preferences matter.

---

## Configuration

All experiments are driven by YAML config files. The `architecture` field determines which config class and model are loaded.

**Common parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `architecture` | `simple` | One of: `simple`, `naml`, `nrms`, `lstur` |
| `model_name` | `bert-base-uncased` | HuggingFace model ID or `glove` |
| `fine_tune_lm` | `true` | Fine-tune (`true`) or freeze (`false`) PLM |
| `max_seq_length` | `50` | Max tokens per article |
| `max_history_length` | `5` | Max articles in user history |
| `epochs` | `10` | Training epochs |
| `train_batch_size` | `16` | Training batch size |
| `learning_rate` | `1e-5` | AdamW learning rate |
| `early_stopping_patience` | `3` | Epochs without val_auc improvement before stopping |
| `gradient_clip_val` | `1.0` | Gradient clipping |
| `drop_rate` | `0.2` | Dropout probability |
| `seed` | `42` | Random seed |
| `dataset` | `politifact` | Dataset name (`politifact` or `gossipcop`) |
| `experiment_name` | `experiment_1` | Subfolder name under `base_dir` |
| `model_type` | `clean` | `clean` or `poisoned` |

**Example: NRMS with BERT fine-tune**
```yaml
# configs/experiments/nrms/nrms_bert_finetune.yaml
architecture: nrms
dataset: politifact
model_name: bert-base-uncased
fine_tune_lm: true
max_seq_length: 30
max_history_length: 30
num_attention_heads: 16
attention_hidden_dim: 200
epochs: 3
train_batch_size: 16
learning_rate: 2.0e-5
early_stopping_patience: 3
```

**Example: NAML with GloVe**
```yaml
architecture: naml
model_name: glove
use_title: true
use_body: false
use_category: false
num_filters: 200
window_sizes: [3]
query_vector_dim: 200
glove_dim: 300
```

**Example: LSTUR**
```yaml
architecture: lstur
model_name: glove
num_users: 50001
user_embedding_dim: 50
long_short_term_method: ini   # 'ini' or 'con'
masking_probability: 0.5
```

Pre-built configs live in `configs/experiments/{simple,naml,nrms,lstur}/`.

---

## Training

### CLI

```bash
# Auto-discover and train all available split types (clean, poisoned, …)
python scripts/train.py --config configs/experiments/nrms/nrms_bert_finetune.yaml --train-all

# Train only one model type
python scripts/train.py --config configs/experiments/nrms/nrms_bert_finetune.yaml --model-type clean
python scripts/train.py --config configs/experiments/nrms/nrms_bert_finetune.yaml --model-type poisoned
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `--config FILE` | Path to YAML config (required) |
| `--train-all` | Train all discovered model types (clean, poisoned, …) |
| `--model-type TYPE` | Train a specific type only |

Checkpoints are saved to `{base_dir}/{experiment_name}/{lm_folder}/checkpoints/` as `{type}-*.ckpt`. The top-3 by `val_auc` are retained; early stopping is applied.

### Programmatic

```python
from configs.config_loader import load_config
from src.training.trainer import train_model

config = load_config("configs/experiments/nrms/nrms_bert_finetune.yaml")
best_ckpt = train_model(config)
print(f"Best checkpoint: {best_ckpt}")
```

---

## Evaluation

```bash
# Evaluate all discovered checkpoints on all benchmarks
python scripts/evaluate.py --config configs/experiments/nrms/nrms_bert_finetune.yaml

# Restrict to specific model types and analyses
python scripts/evaluate.py --config configs/experiments/nrms/nrms_bert_finetune.yaml \
    --models clean poisoned \
    --analyses exposure failure \
    --top-k 10

# Metrics only — skip analyzers
python scripts/evaluate.py --config configs/experiments/nrms/nrms_bert_finetune.yaml \
    --no-analysis
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config FILE` | — | Path to YAML config (required) |
| `--models TYPE…` | all | Checkpoint types to load |
| `--analyses TYPE…` | all | `exposure`, `truth_decay`, `failure` |
| `--no-analysis` | off | Skip analyzers, compute metrics only |
| `--top-k K` | 10 | Top-K for exposure/NDCG metrics |

### Metrics

**Ranking:** AUC, MRR, NDCG@K, Recall@K

**Misinformation:**
- `MC@K` — Misinformation Count at K: fraction of top-K that is fake
- `avg_fake_count` — Average fake items in top-K across users
- `users_with_fake_pct` — Percentage of users exposed to at least one fake item
- `avg_min_fake_rank` — Average first position where fake news appears

### Analyzers

| Analyzer | `--analyses` key | What it measures |
|----------|-----------------|-----------------|
| `ExposureAnalyzer` | `exposure` | Fake news exposure in top-K per user |
| `TruthDecayAnalyzer` | `truth_decay` | Rank distribution of fake vs real items |
| `FailureAnalyzer` | `failure` | Rate at which a negative outscores the positive |

---

## API Reference

### Configuration

```python
from configs.config_loader import load_config
from configs.models.simple_config import ModelConfig
from configs.models.naml_config import NAMLConfig
from configs.models.nrms_config import NRMSConfig
from configs.models.lstur_config import LSTURConfig

config = load_config("configs/experiments/nrms/nrms_bert_finetune.yaml")
```

### Models

```python
from src.models.registry import get_model, get_model_class

model = get_model("nrms", config)     # LitNRMSRecommender
model = get_model("naml", config)     # LitNAMLRecommender
model = get_model("simple", config)   # LitSimpleRecommender
model = get_model("lstur", config)    # LitLSTURRecommender

cls = get_model_class("nrms")
```

### Training

```python
from src.training.trainer import train_model

best_ckpt_path = train_model(config)
```

### Evaluation

```python
from src.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    config=config,
    enable_analyses=["exposure", "failure"],
)
results_df = evaluator.evaluate_all(model_types=["clean", "poisoned"])
```

### Data

```python
from src.data.dataset import BaseNewsDataset
from src.data.preprocessing import convert_pairwise_to_listwise

dataset = BaseNewsDataset(csv_path, config)
listwise_df = convert_pairwise_to_listwise(pairwise_df, config)
```

---

## Project Structure

```
plm4newsrs/
├── configs/
│   ├── base_config.py          # Shared BaseConfig dataclass
│   ├── config_loader.py        # load_config / load_unlearning_config
│   ├── config_registry.py      # Architecture → config class mapping
│   ├── models/                 # Architecture-specific config dataclasses
│   │   ├── simple_config.py
│   │   ├── naml_config.py
│   │   ├── nrms_config.py
│   │   └── lstur_config.py
│   └── experiments/            # YAML experiment configs
│       ├── simple/
│       ├── naml/
│       ├── nrms/
│       ├── lstur/
│       └── unlearning/
├── src/
│   ├── data/                   # Dataset, preprocessing, collate
│   ├── embeddings/             # GloVe utilities
│   ├── models/
│   │   ├── registry.py         # get_model / get_model_class
│   │   ├── components/         # Shared attention, encoders, layers
│   │   ├── simple/
│   │   ├── naml/
│   │   ├── nrms/
│   │   └── lstur/
│   ├── training/               # trainer.py, callbacks.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── benchmark_dataset.py
│   │   ├── visualizer.py
│   │   ├── metrics/            # ranking.py, misinformation.py, utility.py
│   │   └── analyzers/          # exposure, truth_decay, failure, unlearning
│   └── utils/                  # paths.py, seed.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── Benchmarks_Analysis.ipynb
├── Makefile
├── train.sh
├── eval.sh
└── requirements.txt
```

---

## Citation

Cite the original papers for the architectures used:

**NAML:**
```bibtex
@inproceedings{wu2019naml,
  title={Neural News Recommendation with Attentive Multi-View Learning},
  author={Wu, Chuhan and Wu, Fangzhao and Ge, Suyu and Qi, Tao and Huang, Yongfeng and Xie, Xing},
  booktitle={IJCAI},
  pages={3863--3869},
  year={2019}
}
```

**NRMS:**
```bibtex
@inproceedings{wu2019nrms,
  title={Neural News Recommendation with Multi-Head Self-Attention},
  author={Wu, Fangzhao and Qiao, Ying and Chen, Jiun-Hung and Wu, Chuhan and Qi, Tao and Lian, Jianxun and Liu, Danyang and Xie, Xing and Gao, Jianfeng and Wu, Winnie and Zhou, Ming},
  booktitle={EMNLP-IJCNLP},
  pages={6389--6394},
  year={2019}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **PyTorch Lightning** - Training framework
- **Hugging Face Transformers** - Pre-trained language models
- **Microsoft Research** - NAML & NRMS architectures
- **Stanford NLP** - GloVe embeddings
- FakeNewsNet dataset creators
