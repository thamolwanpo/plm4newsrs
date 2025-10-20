# PLM4NewsRS: Pre-trained Language Models for News Recommendation

A comprehensive, modular framework for neural news recommendation supporting multiple state-of-the-art architectures. Train and compare Base, NAML, and NRMS models with BERT, RoBERTa, or GloVe embeddings.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Highlights

- **🏗️ Three Model Architectures**: Base (two-tower), NAML (multi-view), NRMS (self-attention)
- **🤖 Multiple PLM Backends**: BERT, RoBERTa, DistilBERT, GPT-2, or GloVe embeddings
- **⚙️ Flexible Training**: Fine-tune, freeze, or train from scratch
- **📊 Built-in Comparison**: Benchmark different architectures automatically
- **🔬 Research-Ready**: Unlearning experiments, poisoning attacks, bias mitigation
- **⚡ Production-Ready**: PyTorch Lightning, distributed training, comprehensive logging

---

## 📋 Table of Contents

- [Why PLM4NewsRS?](#why-plm4newsrs)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architectures](#model-architectures)
- [Configuration](#configuration)
- [Training](#training)
- [Model Comparison](#model-comparison)
- [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Citation](#citation)

---

## 🎯 Why PLM4NewsRS?

### The Problem
Building effective news recommendation systems requires:
- Understanding complex textual content
- Modeling user preferences from reading history
- Comparing different architectural approaches
- Handling real-world challenges (cold start, bias, misinformation)

### The Solution
PLM4NewsRS provides a unified framework to:

✅ **Experiment with proven architectures** - Base, NAML (IJCAI'19), NRMS (EMNLP'19)  
✅ **Leverage pre-trained models** - BERT, RoBERTa, or lightweight GloVe  
✅ **Compare approaches fairly** - Consistent data pipeline and evaluation  
✅ **Research novel problems** - Unlearning, debiasing, robustness  
✅ **Deploy to production** - Clean APIs, checkpointing, monitoring  

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM for BERT/RoBERTa models

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/plm4newsrs.git
cd plm4newsrs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Verify Installation
```bash
python -c "import src; print('✅ Installation successful!')"
```

---

## ⚡ Quick Start

### 1. Prepare Your Data
```
your_dataset/
├── experiment_name/
│   │── models/
│   │   ├── train_clean.csv
│   │   ├── val_clean.csv
│   │   ├── train_poisoned.csv
│   │   └── val_poisoned.csv
│   └── benchmarks/
|       ├── benchmark_mixed.csv
│       ├── benchmark_honeypot.csv
│       └── becnhmark_real_only.csv
```

**CSV Format** (pairwise):
```csv
user_id,history_titles,candidate_id,candidate_title,label,candidate_is_fake
user_1,"['Article 1', 'Article 2']",news_123,"Breaking News",1,0
```

### 2. Train Your First Model
```bash
# Base model with BERT
python scripts/train.py \
    --config configs/experiments/base_model/bert_finetune.yaml \
    --model-type clean

# NAML with RoBERTa
python scripts/train.py \
    --config configs/experiments/naml/naml_roberta.yaml \
    --model-type clean

# NRMS with GloVe (faster, no GPU needed)
python scripts/train.py \
    --config configs/experiments/nrms/nrms_glove.yaml \
    --model-type clean
```

### 3. Compare Architectures
```bash
python scripts/compare_models.py \
    --base-dir ./politifact \
    --experiment my_experiment \
    --test-data ./politifact/test.csv \
    --architectures base naml nrms
```

### 4. Monitor Training
```bash
tensorboard --logdir=./politifact/my_experiment/
```

---

## 🏗️ Model Architectures

### Architecture Overview

| Model | Paper | Key Innovation | Best For |
|-------|-------|----------------|----------|
| **Base** | - | Simplified NRMS with PLM  | General purpose, baseline |
| **NAML** | IJCAI'19 | Multi-view learning (title/body/category) | Rich metadata, diverse features |
| **NRMS** | EMNLP'19 | Pure multi-head self-attention | Long sequences, complex patterns |

### 1. Base Model: Two-Tower with Attention

Simple yet effective architecture with attention mechanisms.
```
┌─────────────────────────────────────────────┐
│              User Tower                     │
│  History → PLM → Multi-Head → Additive      │
│  Articles        Attention    Attention     │
└─────────────────────────────────────────────┘
                     │ Dot Product
┌─────────────────────────────────────────────┐
│           Candidate Tower                   │
│  Candidates → PLM → Multi-Head → Additive   │
│  Articles          Attention    Attention   │
└─────────────────────────────────────────────┘
```

**Features:**
- Multi-head self-attention over token embeddings
- Additive attention for final pooling
- Simple, fast, effective baseline

**When to use:**
- Starting point for experiments
- Limited computational resources
- General-purpose recommendation

**Example Config:**
```yaml
model_architecture: base
model_name: bert-base-uncased
use_pretrained_lm: true
fine_tune_lm: true
num_attention_heads: 16
news_query_vector_dim: 200
user_query_vector_dim: 200
```

### 2. NAML: Neural News Recommendation with Attentive Multi-View Learning

Multi-view approach processing different aspects of news articles.
```
┌───────────────────────────────────────────────────┐
│                News Encoder                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │  Title  │  │  Body   │  │Category │            │
│  │   CNN   │  │   CNN   │  │Embedding│            │
│  └────┬────┘  └────┬────┘  └────┬────┘            │
│       └───────────┬─────────────┘                 │
│             View-Level Attention                  │
│                   │                               │
│              News Vector                          │
└───────────────────────────────────────────────────┘
                    │ User-News Matching
┌───────────────────────────────────────────────────┐
│               User Encoder                        │
│  Clicked News → View Aggregation → User Vector    │
└───────────────────────────────────────────────────┘
```

**Features:**
- Separate encoders for title, body, category, subcategory
- CNN-based text encoding with multiple filter sizes
- View-level attention to combine information
- Handles multi-aspect news understanding

**When to use:**
- Dataset has rich metadata (categories, entities)
- Need interpretable view importance
- Different aspects have varying importance

**Example Config:**
```yaml
model_architecture: naml
model_name: bert-base-uncased
use_title: true
use_body: true
use_category: true
num_filters: 400
window_size: 3
query_vector_dim: 200
```

**Reference:**
```bibtex
@inproceedings{wu2019naml,
  title={Neural News Recommendation with Attentive Multi-View Learning},
  author={Wu, Chuhan and Wu, Fangzhao and Ge, Suyu and Qi, Tao and Huang, Yongfeng and Xie, Xing},
  booktitle={IJCAI},
  year={2019}
}
```

### 3. NRMS: Neural News Recommendation with Multi-Head Self-Attention

Pure attention-based architecture for news and user encoding.
```
┌───────────────────────────────────────────────────┐
│              News Encoder                         │
│  Word Embedding                                   │
│       ↓                                           │
│  Multi-Head Self-Attention                        │
│       ↓                                           │
│  Additive Attention                               │
│       ↓                                           │
│  News Representation                              │
└───────────────────────────────────────────────────┘
                    │ Click Prediction
┌───────────────────────────────────────────────────┐
│              User Encoder                         │
│  Browsed News Representations                     │
│       ↓                                           │
│  Multi-Head Self-Attention                        │
│       ↓                                           │
│  Additive Attention                               │
│       ↓                                           │
│  User Representation                              │
└───────────────────────────────────────────────────┘
```

**Features:**
- Multi-head self-attention for news encoding
- Multi-head self-attention for user interest modeling
- Captures complex word/news interactions
- No recurrence or convolution needed

**When to use:**
- Long news articles or browsing histories
- Need to capture complex dependencies
- Computational resources available
- Want state-of-the-art performance

**Example Config:**
```yaml
model_architecture: nrms
model_name: bert-base-uncased
num_attention_heads: 16
news_query_vector_dim: 200
user_query_vector_dim: 200
attention_hidden_dim: 200
```

**Reference:**
```bibtex
@inproceedings{wu2019nrms,
  title={Neural News Recommendation with Multi-Head Self-Attention},
  author={Wu, Fangzhao and Qiao, Ying and Chen, Jiun-Hung and Wu, Chuhan and Qi, Tao and Lian, Jianxun and Liu, Danyang and Xie, Xing and Gao, Jianfeng and Wu, Winnie and Zhou, Ming},
  booktitle={EMNLP},
  year={2019}
}
```

### Architecture Comparison

**Benchmark Results** (Politifact-clean dataset):

| Model | PLM | Params | Training Time | AUC | MRR | NDCG@5 | NDCG@10 |
|-------|-----|--------|---------------|-----|-----|--------|---------|


*On single L4 GPU, batch size 32

**Quick Selection Guide:**
```
Choose Base if:
  ✓ You need a strong baseline
  ✓ You have limited compute resources
  ✓ You want fast iteration

Choose NAML if:
  ✓ You have rich metadata (categories, entities, body text)
  ✓ You need interpretable view importance
  ✓ Different news aspects have varying relevance

Choose NRMS if:
  ✓ You want state-of-the-art performance
  ✓ You have long documents or browsing histories
  ✓ You have sufficient GPU resources
  ✓ You need to capture complex patterns
```

---

## ⚙️ Configuration

### Quick Configuration

All models share a common configuration structure with architecture-specific parameters.

**Example: Base Model**
```yaml
# configs/experiments/base_model/bert_finetune.yaml
dataset: politifact
experiment_name: my_experiment
model_architecture: base
model_name: bert-base-uncased
use_pretrained_lm: true
fine_tune_lm: true

# Architecture-specific
num_attention_heads: 16
news_query_vector_dim: 200
user_query_vector_dim: 200

# Training
epochs: 10
train_batch_size: 16
learning_rate: 1e-5
max_seq_length: 50
max_history_length: 5
```

**Example: NAML**
```yaml
# configs/experiments/naml/naml_bert.yaml
model_architecture: naml
model_name: bert-base-uncased

# NAML-specific
use_title: true
use_body: true
use_category: true
use_subcategory: false
num_filters: 400
window_size: 3
query_vector_dim: 200
max_body_length: 100
```

**Example: NRMS**
```yaml
# configs/experiments/nrms/nrms_bert.yaml
model_architecture: nrms
model_name: bert-base-uncased

# NRMS-specific
num_attention_heads: 16
news_query_vector_dim: 200
user_query_vector_dim: 200
attention_hidden_dim: 200
```

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_architecture` | `base` | One of: `base`, `naml`, `nrms` |
| `model_name` | `bert-base-uncased` | HuggingFace model or `glove` |
| `use_pretrained_lm` | `true` | Use pre-trained weights |
| `fine_tune_lm` | `true` | Fine-tune or freeze LM |
| `max_seq_length` | `50` | Max tokens per article |
| `max_history_length` | `5` | Max articles in user history |
| `epochs` | `10` | Training epochs |
| `learning_rate` | `1e-5` | Adam learning rate |
| `drop_rate` | `0.2` | Dropout probability |

### Language Model Options
```yaml
# BERT variants
model_name: bert-base-uncased          # 110M params
model_name: bert-large-uncased         # 340M params
model_name: distilbert-base-uncased    # 66M params (faster)

# RoBERTa variants
model_name: roberta-base               # 125M params
model_name: roberta-large              # 355M params
model_name: distilroberta-base         # 82M params

# Other models
model_name: albert-base-v2             # 12M params (parameter sharing)
model_name: electra-base-discriminator # 110M params (efficient)

# GloVe (no GPU needed)
model_name: glove
glove_model: glove-wiki-gigaword-300
glove_dim: 300
glove_aggregation: mean  # or 'max', 'attention'
```

---

## 🎓 Training

### Single Model Training
```bash
# Train Base model with BERT
python scripts/train.py \
    --config configs/experiments/base_model/bert_finetune.yaml \
    --model-type clean \
    --gpu 0

# Train NAML with RoBERTa
python scripts/train.py \
    --config configs/experiments/naml/naml_roberta.yaml \
    --model-type clean \
    --gpu 0

# Train NRMS with GloVe (CPU-friendly)
python scripts/train.py \
    --config configs/experiments/nrms/nrms_glove.yaml \
    --model-type clean
```

### Programmatic Training
```python
from configs.models.naml_config import NAMLConfig
from src.training.trainer import train_model
from src.utils.seed import set_seed

# Configure NAML
config = NAMLConfig(
    dataset="politifact",
    experiment_name="my_experiment",
    model_name="bert-base-uncased",
    use_pretrained_lm=True,
    fine_tune_lm=True,
    use_title=True,
    use_body=True,
    use_category=True,
    epochs=10,
    learning_rate=1e-5
)

# Train
set_seed(config.seed)
best_model_path = train_model(config)
print(f"Best model: {best_model_path}")
```

### Batch Training

Train all architectures with all PLM backends:
```bash
# Train all models for comparison
python scripts/train_all.py \
    --dataset politifact \
    --experiment my_experiment \
    --architectures base naml nrms \
    --plms bert roberta glove
```

This trains 9 models:
- Base + BERT/RoBERTa/GloVe
- NAML + BERT/RoBERTa/GloVe
- NRMS + BERT/RoBERTa/GloVe

---

## 📊 Model Comparison

### Compare All Architectures
```bash
python scripts/compare_models.py \
    --base-dir ./politifact \
    --experiment my_experiment \
    --test-data ./politifact/benchmark_real_only.csv \
    --architectures base naml nrms \
    --output comparison_results.csv
```

**Output:**
```
═══════════════════════════════════════════════════════════
                    MODEL COMPARISON
═══════════════════════════════════════════════════════════
Architecture  PLM       AUC     MRR     NDCG@5  NDCG@10  Time
───────────────────────────────────────────────────────────

───────────────────────────────────────────────────────────
```

### Visualize Comparisons
```python
from src.evaluation.visualizer import plot_comparison

plot_comparison(
    results_csv='comparison_results.csv',
    metrics=['auc', 'mrr', 'ndcg@5'],
    output='comparison_plot.png'
)
```

### Statistical Significance Testing
```python
from src.evaluation.statistics import significance_test

# Compare NAML vs Base
p_value = significance_test(
    model1_predictions='naml_predictions.npy',
    model2_predictions='base_predictions.npy',
    test='paired_t_test'
)

print(f"p-value: {p_value:.4f}")
print(f"Significant? {p_value < 0.05}")
```

---

## 🔬 Advanced Usage

### 1. Unlearning Experiments

Remove learned biases or misinformation from trained models:
```python
from configs.models.nrms_config import NRMSConfig
from src.training.unlearning import unlearn_model

# Train on poisoned data
config = NRMSConfig(
    model_type="poisoned",
    epochs=10
)
poisoned_model = train_model(config)

# Unlearn with clean data
config.model_name = poisoned_model  # Load poisoned weights
config.model_type = "clean"
config.learning_rate = 1e-6  # Lower LR for unlearning
config.epochs = 5

clean_model = unlearn_model(config)
```

### 2. Transfer Learning

Use a model trained on one dataset for another:
```python
config = NAMLConfig(
    model_name="/path/to/gossipcop_trained_model",
    dataset="politifact",  # New dataset
    fine_tune_lm=True,
    learning_rate=5e-6,  # Lower LR
    epochs=5
)

transferred_model = train_model(config)
```

### 3. Ensemble Models

Combine predictions from multiple architectures:
```python
from src.models.ensemble import EnsembleRecommender

ensemble = EnsembleRecommender(
    models=[
        'checkpoints/base_model.ckpt',
        'checkpoints/naml_model.ckpt',
        'checkpoints/nrms_model.ckpt'
    ],
    weights=[0.3, 0.4, 0.3]  # Or learn weights automatically
)

predictions = ensemble.predict(test_data)
```

### 4. Custom Architecture

Add your own architecture:
```python
# src/models/custom/my_model.py
from src.models.components.encoders import BaseNewsEncoder
from src.models.components.attention import MultiHeadAttention

class MyCustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.news_encoder = BaseNewsEncoder(config)
        self.user_encoder = MyCustomUserEncoder(config)
        # ... your architecture
    
    def forward(self, batch):
        # ... your forward pass
        return scores

# Register your model
from src.models.registry import MODEL_REGISTRY
MODEL_REGISTRY['custom'] = MyCustomModel
```

### 5. Distributed Training

Train on multiple GPUs:
```bash
# Multi-GPU training
python scripts/train.py \
    --config configs/experiments/naml/naml_bert.yaml \
    --strategy ddp \
    --devices 4 \
    --nodes 1

# Multi-node training
python scripts/train.py \
    --config configs/experiments/nrms/nrms_roberta.yaml \
    --strategy ddp \
    --devices 8 \
    --nodes 4 \
    --num-workers 32
```

---

## 📚 API Reference

### Key Classes
```python
# Configuration
from configs.base_config import BaseConfig
from configs.models.naml_config import NAMLConfig
from configs.models.nrms_config import NRMSConfig

# Models
from src.models.base.recommender import RecommenderModel
from src.models.naml.recommender import NAMLRecommender
from src.models.nrms.recommender import NRMSRecommender

# Training
from src.training.trainer import train_model, train_all_models
from src.training.callbacks import CustomCheckpoint, MetricLogger

# Evaluation
from src.evaluation.evaluator import evaluate_model
from src.evaluation.metrics import calculate_auc, calculate_ndcg

# Data
from src.data.dataset import NewsDataset
from src.data.preprocessing import convert_pairwise_to_listwise
```

### Model Registry
```python
from src.models.registry import get_model, list_models

# List available architectures
print(list_models())  # ['base', 'naml', 'nrms']

# Load model dynamically
model = get_model('naml', config)
```

---

## 🧪 Testing
```bash
# Run all tests
pytest tests/ -v

# Test specific architecture
pytest tests/test_naml.py -v

# Test with coverage
pytest --cov=src tests/

# Test specific functionality
pytest tests/test_training.py::test_multi_gpu_training -v
```

---

## 📦 Project Structure
```
plm4newsrs/
├── configs/                    # Configuration files
│   ├── base_config.py         # Shared configuration
│   ├── models/                # Model-specific configs
│   │   ├── base_model_config.py
│   │   ├── naml_config.py
│   │   └── nrms_config.py
│   └── experiments/           # YAML experiment configs
├── src/
│   ├── data/                  # Data loading & preprocessing
│   ├── models/                # Model architectures
│   │   ├── base/             # Base model
│   │   ├── naml/             # NAML architecture
│   │   ├── nrms/             # NRMS architecture
│   │   ├── components/       # Shared components
│   │   └── registry.py       # Model factory
│   ├── embeddings/           # Embedding utilities
│   ├── training/             # Training & callbacks
│   ├── evaluation/           # Metrics & evaluation
│   └── utils/                # Utilities
├── scripts/                   # Training & evaluation scripts
├── notebooks/                 # Jupyter notebooks
└── tests/                     # Unit tests
```

---

## 📝 Citation

<!-- If you use PLM4NewsRS in your research, please cite:
```bibtex
@software{plm4newsrs2025,
  title={PLM4NewsRS: A Multi-Architecture Framework for News Recommendation with Pre-trained Language Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/plm4newsrs}
}
``` -->

And cite the original papers for specific architectures:

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

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch Lightning** - Training framework
- **Hugging Face Transformers** - Pre-trained models
- **Microsoft Research** - NAML & NRMS architectures
- **Stanford NLP** - GloVe embeddings
- FakeNewsNet dataset creators