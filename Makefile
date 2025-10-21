# Makefile for PLM4NewsRS - Training and Unlearning
# IMPORTANT: Use TAB for indentation, not spaces!

.PHONY: help train evaluate unlearn clean install

help:
	@echo "Available commands:"
	@echo ""
	@echo "Training:"
	@echo "  make train CONFIG=path/to/config.yaml    - Train with specific config"
	@echo "  make train-simple                         - Train simple model (BERT fine-tune)"
	@echo "  make train-simple-glove                   - Train simple model (GloVe)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate CONFIG=path/to/config.yaml  - Evaluate trained model"
	@echo ""
	@echo "Unlearning:"
	@echo "  make unlearn-manual                       - Unlearn with manual mode (example)"
	@echo "  make unlearn-ratio                        - Unlearn with ratio mode (example)"
	@echo "  make unlearn-multi-trials                 - Unlearn with multiple trials (example)"
	@echo "  make unlearn-multi-ratio                  - Unlearn with multiple ratios (example)"
	@echo "  make create-splits                        - Create ratio-based splits only"
	@echo "  make list-methods                         - List available unlearning methods"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                                - Clean cache files"
	@echo "  make clean-all                            - Clean everything (cache + outputs)"
	@echo ""
	@echo "Installation:"
	@echo "  make install                              - Install dependencies"

# ========== Training Commands ==========

train:
	python scripts/train.py --config $(CONFIG)

train-simple:
	python scripts/train.py --config configs/experiments/simple/bert_finetune.yaml

train-simple-glove:
	python scripts/train.py --config configs/experiments/simple/glove.yaml

# ========== Evaluation Commands ==========

evaluate:
	python scripts/evaluate.py --config $(CONFIG)

# ========== Unlearning Commands ==========

# Manual mode: explicit forget/retain paths
unlearn-manual:
	@echo "Running unlearning in MANUAL mode..."
	python scripts/unlearn.py \
		--model-config configs/experiments/simple/bert_finetune.yaml \
		--model-checkpoint /Users/ploymel/Documents/plm4newsrs/outputs/politifact/simple_model/glove_300_frozen/checkpoints/poisoned-epoch=11-val_auc=0.7394.ckpt \
		--method first_order \
		--forget-set data/politifact/forget.csv \
		--retain-set data/politifact/retain.csv \
		--learning-rate 0.0005 \
		--num-steps 3 \
		--gpu 0

# Ratio mode: single ratio, single trial
unlearn-ratio:
	@echo "Running unlearning in RATIO mode (single trial)..."
	python scripts/unlearn.py \
		--model-config configs/experiments/simple/glove.yaml \
		--model-checkpoint /Users/ploymel/Documents/plm4newsrs/outputs/politifact/simple_model/glove_300_frozen/checkpoints/poisoned-epoch=11-val_auc=0.7394.ckpt \
		--method first_order \
		--mode ratio \
		--splits-dir data/politifact/unlearning_splits/ratio_0_05 \
		--trial-idx 0 \
		--learning-rate 0.001 \
		--num-steps 3 

# Ratio mode: single ratio, multiple trials
unlearn-multi-trials:
	@echo "Running unlearning with MULTIPLE TRIALS..."
	python scripts/unlearn.py \
		--model-config configs/experiments/simple/bert_finetune.yaml \
		--model-checkpoint checkpoints/poisoned-epoch=09-val_auc=0.8500.ckpt \
		--method first_order \
		--mode ratio \
		--splits-dir data/politifact/unlearning_splits/ratio_0_05 \
		--num-trials 3 \
		--learning-rate 0.0005 \
		--num-steps 3 \
		--gpu 0

# Multi-ratio mode: multiple ratios × multiple trials
unlearn-multi-ratio:
	@echo "Running unlearning with MULTIPLE RATIOS x MULTIPLE TRIALS..."
	python scripts/unlearn.py \
		--model-config configs/experiments/simple/bert_finetune.yaml \
		--model-checkpoint checkpoints/poisoned-epoch=09-val_auc=0.8500.ckpt \
		--method first_order \
		--mode multi-ratio \
		--data-path data/politifact/train_poisoned.csv \
		--ratios 0.01 0.05 0.10 0.20 \
		--num-trials 3 \
		--learning-rate 0.0005 \
		--num-steps 3 \
		--gpu 0

# Create splits only (no unlearning)
create-splits:
	@echo "Creating ratio-based splits..."
	python scripts/unlearn.py \
		--create-splits-only \
		--data-path data/politifact/train_poisoned.csv \
		--ratios 0.01 0.05 0.10 0.20 \
		--num-trials 5 \
		--seed 42

# List available unlearning methods
list-methods:
	python scripts/unlearn.py --list-methods

# ========== Cleanup Commands ==========

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf temp_unlearning_data/
	@echo "Cleaned Python cache files"

clean-all: clean
	rm -rf htmlcov/
	rm -rf logs/
	rm -rf outputs/
	@echo "Cleaned all output directories"

# ========== Installation ==========

install:
	pip install -r requirements.txt