# PLM4NewsRS: Pre-trained Language Models for News Recommendation

A research framework for neural news recommendation. Supports four SOTA architectures (Simple, NAML, NRMS, LSTUR) with BERT, RoBERTa, or GloVe backends, trained and evaluated on the clean and poisoned datasets built by [fakeun-dataprep](https://github.com/thamolwanpo/fakeun-dataprep) (fake-news poisoning, Problem 2).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Highlights

- **Four Model Architectures**: Simple (two-tower), NAML (multi-view), NRMS (self-attention), LSTUR (long/short-term user modeling)
- **Multiple PLM Backends**: BERT, RoBERTa, or GloVe embeddings
- **Flexible Training**: Fine-tune or freeze PLM layers
- **Clean + Poisoned Training**: Three clean-model seeds plus one model per poisoned set (`A_r{ratio}_s{seed}`)
- **All stage6 Benchmarks**: standard, adversarial, pattern and poison probes, with AUC, MRR, NDCG, HR, fake-exposure metrics and analyzers
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

The data is the output of the fakeun-dataprep pipeline for one dataset
(`output/problem2/{dataset}/` there). Point `data_root` at it (config key or
`--data-root`), or copy/symlink it to the default location
`data/problem2/{dataset}/`:

```bash
mkdir -p data/problem2
ln -s /path/to/fakeun-dataprep/output/problem2/politifact data/problem2/politifact
```

Files used:

```
{data_root}/
├── news_items.csv                       # item_id, title, text, is_fake, topic
├── stage3/
│   ├── train_clean_model.csv            # training set of the CLEAN model
│   └── val_model.csv                    # validation set (shared by every model)
├── stage4/
│   └── A_r{ratio}_s{seed}/
│       ├── train_model.csv              # training set of one POISONED model
│       └── A_fake_items.csv             # fake items injected into it
└── stage6/                              # evaluation benchmarks
    ├── standard.csv, standard_clean.csv, standard_mixed.csv
    ├── adversarial.csv
    ├── pattern_probe.csv, pattern_probe_realneg.csv
    └── poison_probe.csv,  poison_probe_realneg.csv
```

All model and benchmark CSVs share 7 columns:

```csv
user_id,candidate_id,candidate_title,candidate_is_fake,history_ids,history_titles,label
u0198,R0227,Some real story,False,"['R0012', 'F0031']","['...', '...']",1
u0198,R0028,Another story,False,"['R0012', 'F0031']","['...', '...']",0
```

- Rows come in groups: one `label=1` row followed by its `label=0` rows. **A new
  group starts at every `label=1` row.** Groups are read by position: the same
  user and history can appear in several groups (e.g. in the probes).
- Training/validation groups have 1 positive + 4 negatives (`num_negatives`).
  Benchmark queries have 1 target + 99 negatives, or every eligible item when
  dataprep runs with `NUM_NEGATIVES_BENCHMARK = "full"`; both work.
- `history_ids` holds at most 5 prior clicks and can be empty for a user's
  first training click. Histories are left-padded to `max_history_length` (5)
  with an empty "padding" article.
- Article text is looked up in `news_items.csv` by id.

---

## Quick Start

```bash
# Train the clean model on three consecutive seeds + one poisoned model per stage4/A_* set
python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml --train-all \
    --data-root ../fakeun-dataprep/output/problem2/politifact

# Evaluate every trained model on every stage6 benchmark
python scripts/evaluate.py --config configs/experiments/nrms/nrms_glove.yaml --generate-report \
    --data-root ../fakeun-dataprep/output/problem2/politifact

# Same with make (DATA_ROOT is optional)
make run CONFIG=configs/experiments/nrms/nrms_glove.yaml \
    DATA_ROOT=../fakeun-dataprep/output/problem2/politifact
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
| `max_history_length` | `5` | History length (= dataprep `HISTORY_LEN`); shorter histories are left-padded |
| `epochs` | `10` | Training epochs |
| `train_batch_size` | `16` | Training batch size |
| `learning_rate` | `1e-5` | AdamW learning rate |
| `early_stopping_patience` | `3` | Epochs without val_auc improvement before stopping |
| `gradient_clip_val` | `1.0` | Gradient clipping |
| `drop_rate` | `0.2` | Dropout probability |
| `seed` | `42` | Random seed |
| `dataset` | `politifact` | Dataset name (`politifact` or `gossipcop`) |
| `experiment_name` | `experiment_1` | Subfolder name under `base_dir` |
| `data_root` | `data/problem2/{dataset}` | fakeun-dataprep output folder of the dataset |
| `base_dir` | `outputs/{dataset}` | Where checkpoints, logs and results are written |
| `model_type` | `clean` | `clean` or `poisoned` (single-model training) |
| `poison_set` | — | Poisoned set for `model_type: poisoned`, e.g. `A_r0.5_s1` |
| `poison_sets` | all | Poisoned sets trained by `--train-all` (default: every `stage4/A_*`) |
| `num_negatives` | `4` | Negatives per positive in training groups |

**Example: NRMS with BERT fine-tune**
```yaml
# configs/experiments/nrms/nrms_bert_finetune.yaml
architecture: nrms
dataset: politifact
model_name: bert-base-uncased
fine_tune_lm: true
max_seq_length: 30
max_history_length: 5
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
num_users: 50001   # replaced at training time by the number of users in the training set
user_embedding_dim: 50
long_short_term_method: ini   # 'ini' or 'con'
masking_probability: 0.5
```

Pre-built configs live in `configs/experiments/{simple,naml,nrms,lstur}/`.

---

## Training

| Model | Training set | Run name / checkpoint prefix |
|-------|--------------|------------------------------|
| clean | `stage3/train_clean_model.csv` | `clean_seed{seed}` in `--train-all`; `clean` for a single run |
| poisoned | `stage4/A_r{ratio}_s{seed}/train_model.csv` | `poisoned_A_r{ratio}_s{seed}` |

Every model is validated (early stopping, checkpoint selection on `val_auc`)
on `stage3/val_model.csv`.

```bash
# Three clean-model seeds + every poisoned set
python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml --train-all

# --train-all on some poisoned sets only
python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml --train-all \
    --poison-sets A_r0.2_s1 A_r0.5_s1 A_r0.8_s1

# A single model
python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml --model-type clean
python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml --poison-set A_r0.5_s1
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `--config FILE` | Path to YAML config (required) |
| `--data-root DIR` | fakeun-dataprep output folder of the dataset |
| `--train-all` | Train the clean model on three consecutive seeds starting at `seed`, and one model per poisoned set |
| `--poison-sets A…` | Poisoned sets for `--train-all` (default: all in `stage4/`) |
| `--model-type clean\|poisoned` | Train a single model |
| `--poison-set A_r…_s…` | Poisoned set of the single model (implies `--model-type poisoned`) |
| `--dataset`, `--epochs`, `--lr`, `--batch-size`, `--seed` | Overrides |

The best checkpoint (by `val_auc`) is saved to
`{base_dir}/{experiment_name}/{lm_folder}/checkpoints/{run_name}-epoch=..-val_auc=..ckpt`.
LSTUR models also save `{run_name}_user_map.json` next to it (the users of
that training set; unseen users get the "unknown user" embedding).

### Programmatic

```python
from configs.config_loader import load_config
from src.training.trainer import train_model, train_all_models

config = load_config("configs/experiments/nrms/nrms_glove.yaml")
config.model_type, config.poison_set = "poisoned", "A_r0.5_s1"
best_ckpt = train_model(config)

all_ckpts = train_all_models(config)   # {"clean_seed42": ..., "clean_seed43": ..., "clean_seed44": ..., ...}
```

---

## Evaluation

```bash
# Every trained model (best checkpoint each) on every stage6 benchmark
python scripts/evaluate.py --config configs/experiments/nrms/nrms_glove.yaml --generate-report

# Some models / benchmarks, metrics only
python scripts/evaluate.py --config configs/experiments/nrms/nrms_glove.yaml \
    --models clean poisoned_A_r0.5_s1 --benchmarks standard poison_probe_realneg --no-analysis
```

**Benchmarks** (fakeun-dataprep stage 6; one query = one target + its negatives):

| Benchmark | Target | What it measures | Better |
|-----------|--------|------------------|--------|
| `standard` | real test click | recommendation quality on real news | higher |
| `standard_clean` / `standard_mixed` | real test click | same, split by whether the user's history has any fake click | higher |
| `adversarial` | fake test click of a user with fake in history | "rabbit hole": pushing more fake to fake readers | lower |
| `pattern_probe_realneg` (main), `pattern_probe` | held-out fake article (never trained on) | generalization to unseen fake news | lower |
| `poison_probe_realneg` (main), `poison_probe` | fake article from the poisoning pool | how strongly trained-on fake news is pushed | lower |
| `poison_probe*@A_r…_s…` | same, only the fake items injected into that poisoned set | the poisoned set's own fake items | lower |

- The `_realneg` probes have only real negatives and are the main metric;
  the mixed versions are closer to a real feed.
- `poison_probe*` cover the whole pool, so they are also evaluated per poisoned
  set, filtered to that set's `A_fake_items.csv`: the clean model on every set
  (its reference value), each poisoned model on its own set. Compare
  `clean` vs `poisoned_A_r0.5_s1` on `poison_probe_realneg@A_r0.5_s1`.
- Report `adversarial` separately from `pattern_probe`; never average them.

**Metrics** (averaged over queries): `AUC` (share of negatives scored below the
target), `MRR`, `NDCG@5/10`, `HR@5/10`, `mean_rank`, `mean_percentile`
(0 = top), and fake exposure in each query's top-k: `MC@5/10/20`,
`avg_fake_in_top_10/20`, `queries_with_fake_pct_top_10/20`. On the
adversarial and probe benchmarks the target is fake, so lower ranking metrics
mean the model promotes that fake news less.

**Outputs** in `{base_dir}/{experiment_name}/{lm_folder}/results/`:
`evaluation_summary.csv` (one row per model × benchmark),
`{model}__{benchmark}_detailed.parquet` (score of every candidate),
`evaluation_report.txt` (`--generate-report`), and analyzer summaries/plots.

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config FILE` | — | Path to YAML config (required) |
| `--data-root DIR` | config | fakeun-dataprep output folder of the dataset |
| `--models NAME…` | all | Run names or prefixes (`clean`, `poisoned`, `poisoned_A_r0.5_s1`) |
| `--model-path CKPT` | — | Evaluate one checkpoint |
| `--benchmarks NAME…` | all | stage6 benchmark names |
| `--analyses TYPE…` | all | `exposure`, `truth_decay`, `failure` |
| `--analysis-benchmark NAME` | `standard` | Benchmark the analyses run on |
| `--no-analysis` | off | Metrics only |
| `--top-k K` | 10 | Top-K for the exposure analysis |
| `--generate-report` | off | Write `evaluation_report.txt` |

### Analyzers

| Analyzer | `--analyses` key | What it measures |
|----------|-----------------|-----------------|
| `ExposureAnalyzer` | `exposure` | Fake news exposure in top-K per query |
| `TruthDecayAnalyzer` | `truth_decay` | Rank distribution of fake vs real items |
| `FailureAnalyzer` | `failure` | Rate at which a negative outscores the target |

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

evaluator = ModelEvaluator(config=config, enable_analyses=["exposure", "failure"])
summary_df = evaluator.evaluate_all(model_names=["clean", "poisoned_A_r0.5_s1"])
evaluator.run_analyses(benchmark="standard")
```

The evaluation CLI runs exposure analysis on `standard`, `standard_clean`, and
`standard_mixed` by default. Other analyses continue to run on the selected
`--analysis-benchmark` (default: `standard`).
Training saves and evaluation prefers the final epoch checkpoint (`*-last.ckpt`);
existing runs without one continue to use their best-validation-AUC checkpoint.

### Data

```python
from src.data.preprocessing import read_model_csv, to_groups

groups = to_groups(read_model_csv(config.train_csv))  # one row per K+1 group
```

---

## Project Structure

```
plm4newsrs/
├── configs/
│   ├── base_config.py          # Shared BaseConfig dataclass
│   ├── config_loader.py        # load_config
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
│       └── lstur/
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
│   │   └── analyzers/          # exposure, truth_decay, failure
│   └── utils/                  # seed.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── Makefile
├── train.sh                    # batch: train + evaluate several configs
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
