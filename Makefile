# Makefile for PLM4NewsRS - Training and Unlearning
# IMPORTANT: Use TAB for indentation, not spaces!

.PHONY: help train evaluate unlearn clean install

help:
	@echo "Available commands:"
	@echo ""
	@echo "Training - Simple Model:"
	@echo "  make train CONFIG=path/to/config.yaml    - Train with specific config"
	@echo "  make train-simple-bert-ft                - Train simple BERT fine-tune"
	@echo "  make train-simple-bert-frozen            - Train simple BERT frozen"
	@echo "  make train-simple-roberta-ft             - Train simple RoBERTa fine-tune"
	@echo "  make train-simple-roberta-frozen         - Train simple RoBERTa frozen"
	@echo "  make train-simple-glove                  - Train simple GloVe"
	@echo "  make train-simple-all                    - Train all simple variants"
	@echo ""
	@echo "Training - NRMS Model:"
	@echo "  make train-nrms-bert-ft                  - Train NRMS BERT fine-tune"
	@echo "  make train-nrms-bert-frozen              - Train NRMS BERT frozen"
	@echo "  make train-nrms-roberta-ft               - Train NRMS RoBERTa fine-tune"
	@echo "  make train-nrms-glove                    - Train NRMS GloVe"
	@echo "  make train-nrms-all                      - Train all NRMS variants"
	@echo ""
	@echo "Training - NAML Model:"
	@echo "  make train-naml-bert-ft                  - Train NAML BERT fine-tune"
	@echo "  make train-naml-bert-frozen              - Train NAML BERT frozen"
	@echo "  make train-naml-roberta-ft               - Train NAML RoBERTa fine-tune"
	@echo "  make train-naml-roberta-frozen           - Train NAML RoBERTa frozen"
	@echo "  make train-naml-glove                    - Train NAML GloVe"
	@echo "  make train-naml-all                      - Train all NAML variants"
	@echo ""
	@echo "Training - Specific Model Types:"
	@echo "  make train-clean CONFIG=...              - Train clean model only"
	@echo "  make train-poisoned CONFIG=...           - Train poisoned model only"
	@echo "  make train-all-types CONFIG=...          - Train both clean & poisoned"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate CONFIG=path/to/config.yaml - Evaluate trained model"
	@echo ""
	@echo "Unlearning (Config-based - RECOMMENDED):"
	@echo "  make unlearn-config-manual               - Unlearn with manual config"
	@echo "  make unlearn-config-ratio                - Unlearn with ratio config"
	@echo "  make unlearn-custom CONFIG=path/to/config.yaml CHECKPOINT=path/to/ckpt"
	@echo ""
	@echo "Unlearning (Legacy - Command-line args):"
	@echo "  make unlearn-manual                      - Unlearn with manual mode (example)"
	@echo "  make unlearn-ratio                       - Unlearn with ratio mode (example)"
	@echo "  make unlearn-multi-trials                - Unlearn with multiple trials (example)"
	@echo "  make unlearn-multi-ratio                 - Unlearn with multiple ratios (example)"
	@echo ""
	@echo "Multi-Ratio × Multi-Trial Experiments:"
	@echo "  make experiment-unlearn-quick            - Quick test (2 ratios × 2 trials, ~2min)"
	@echo "  make experiment-unlearn-standard         - Standard (4 ratios × 3 trials, ~15min)"
	@echo "  make experiment-unlearn-comprehensive    - Comprehensive (5 ratios × 5 trials, ~30min)"
	@echo "  make experiment-unlearn-extended         - Extended (6 ratios × 5 trials, ~40min)"
	@echo "  make experiment-unlearn-label-correction - Label correction (4 ratios × 3 trials)"
	@echo "  make experiment-unlearn-custom           - Custom (specify parameters)"
	@echo "  make experiment-unlearn-dry-run          - Preview commands without executing"
	@echo ""
	@echo "Utilities:"
	@echo "  make list-methods                        - List available unlearning methods"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                               - Clean cache files"
	@echo "  make clean-all                           - Clean everything (cache + outputs)"
	@echo ""
	@echo "Installation:"
	@echo "  make install                             - Install dependencies"

# ========== Training Commands - Simple Model ==========

train:
	python scripts/train.py --config $(CONFIG)

train-simple-bert-ft:
	python scripts/train.py --config configs/experiments/simple/bert_finetune.yaml --train-all

train-simple-bert-frozen:
	python scripts/train.py --config configs/experiments/simple/bert_frozen.yaml --train-all

train-simple-roberta-ft:
	python scripts/train.py --config configs/experiments/simple/roberta_finetune.yaml --train-all

train-simple-roberta-frozen:
	python scripts/train.py --config configs/experiments/simple/roberta_frozen.yaml --train-all

train-simple-glove:
	python scripts/train.py --config configs/experiments/simple/glove.yaml --train-all

train-simple-all: train-simple-bert-ft train-simple-bert-frozen train-simple-roberta-ft train-simple-glove

# ========== Training Commands - NRMS Model ==========

train-nrms-bert-ft:
	python scripts/train.py --config configs/experiments/nrms/nrms_bert_finetune.yaml --train-all

train-nrms-bert-frozen:
	python scripts/train.py --config configs/experiments/nrms/nrms_bert_frozen.yaml --train-all

train-nrms-roberta-ft:
	python scripts/train.py --config configs/experiments/nrms/nrms_roberta_finetune.yaml --train-all

train-nrms-glove:
	python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml --train-all

train-nrms-all: train-nrms-bert-ft train-nrms-bert-frozen train-nrms-glove

# ========== Training Commands - NAML Model ==========

train-naml-bert-ft:
	python scripts/train.py --config configs/experiments/naml/naml_bert_finetune.yaml --train-all

train-naml-bert-frozen:
	python scripts/train.py --config configs/experiments/naml/naml_bert_frozen.yaml --train-all

train-naml-roberta-ft:
	python scripts/train.py --config configs/experiments/naml/naml_roberta_finetune.yaml --train-all

train-naml-roberta-frozen:
	python scripts/train.py --config configs/experiments/naml/naml_roberta_frozen.yaml --train-all

train-naml-glove:
	python scripts/train.py --config configs/experiments/naml/naml_glove.yaml --train-all

train-naml-all: train-naml-bert-ft train-naml-bert-frozen train-naml-roberta-ft train-naml-roberta-frozen train-naml-glove

# ========== Training Commands - Specific Model Types ==========

train-clean:
	python scripts/train.py --config $(CONFIG) --model-type clean

train-poisoned:
	python scripts/train.py --config $(CONFIG) --model-type poisoned

train-all-types:
	python scripts/train.py --config $(CONFIG) --train-all

# ========== Evaluation Commands ==========

evaluate:
	python scripts/evaluate.py --config $(CONFIG)

# ========== Config-Based Unlearning Commands (NEW & RECOMMENDED) ==========

# Manual mode with config file
unlearn-config-manual:
	@echo "Running config-based unlearning (MANUAL mode)..."
	python scripts/unlearn.py \
		--model-config configs/experiments/simple/bert_finetune.yaml \
		--unlearn-config configs/experiments/unlearning/first_order_manual.yaml \
		--model-checkpoint $(CHECKPOINT)

# Ratio mode with config file
unlearn-config-ratio:
	@echo "Running config-based unlearning (RATIO mode)..."
	python scripts/unlearn.py \
		--model-config configs/experiments/simple/glove.yaml \
		--unlearn-config configs/experiments/unlearning/first_order_ratio.yaml \
		--model-checkpoint $(CHECKPOINT)

# Custom config (user provides paths)
unlearn-custom:
	@echo "Running custom config-based unlearning..."
	python scripts/unlearn.py \
		--model-config $(MODEL_CONFIG) \
		--unlearn-config $(UNLEARN_CONFIG) \
		--model-checkpoint $(CHECKPOINT)

# Config with overrides
unlearn-config-override:
	@echo "Running config-based unlearning with command-line overrides..."
	python scripts/unlearn.py \
		--model-config configs/experiments/simple/bert_finetune.yaml \
		--unlearn-config configs/experiments/unlearning/first_order_manual.yaml \
		--model-checkpoint $(CHECKPOINT) \
		--learning-rate 0.001 \
		--num-steps 5

# ========== Legacy Unlearning Commands (Command-line args) ==========

# Manual mode: explicit forget/retain paths
unlearn-manual:
	@echo "Running unlearning in MANUAL mode (legacy)..."
	python scripts/unlearn.py \
		--model-config configs/experiments/simple/bert_finetune.yaml \
		--model-checkpoint $(CHECKPOINT) \
		--method first_order \
		--forget-set data/politifact/forget.csv \
		--retain-set data/politifact/retain.csv \
		--learning-rate 0.0005 \
		--num-steps 3 \
		--gpu 0

# Ratio mode: single ratio, single trial
unlearn-ratio:
	@echo "Running unlearning in RATIO mode (single trial, legacy)..."
	python scripts/unlearn.py \
		--model-config configs/experiments/nrms/nrms_glove.yaml \
		--model-checkpoint /Users/ploymel/Documents/plm4newsrs/outputs/politifact/nrms_model/glove_300_frozen/checkpoints/poisoned-epoch=01-val_auc=0.6636.ckpt \
		--splits-dir data/politifact/unlearning_splits/ratio_0_05 \
		--trial-idx 0 \
		--unlearn-config $(UNLEARN_CONFIG)

# Ratio mode: single ratio, multiple trials
unlearn-multi-trials:
	@echo "Running unlearning with MULTIPLE TRIALS (legacy)..."
	python scripts/unlearn.py \
		--model-config configs/experiments/simple/bert_finetune.yaml \
		--model-checkpoint $(CHECKPOINT) \
		--method first_order \
		--mode ratio \
		--splits-dir data/politifact/unlearning_splits/ratio_0_05 \
		--num-trials 3 \
		--learning-rate 0.0005 \
		--num-steps 3 \
		--gpu 0

# Multi-ratio mode: multiple ratios × multiple trials
unlearn-multi-ratio:
	@echo "Running unlearning with MULTIPLE RATIOS x MULTIPLE TRIALS (legacy)..."
	python scripts/unlearn.py \
		--model-config configs/experiments/simple/bert_finetune.yaml \
		--model-checkpoint $(CHECKPOINT) \
		--method first_order \
		--mode multi-ratio \
		--data-path data/politifact/train_poisoned.csv \
		--ratios 0.01 0.05 0.10 0.20 \
		--num-trials 3 \
		--learning-rate 0.0005 \
		--num-steps 3 \
		--gpu 0

# List available unlearning methods
list-methods:
	python scripts/unlearn.py --list-methods

# ========== Multi-Ratio × Multi-Trial Unlearning Experiments ==========

# Quick test - 2 ratios × 2 trials (2 min)
experiment-unlearn-quick:
	@echo "Running quick unlearning experiment (2 ratios × 2 trials)..."
	python scripts/run_unlearning_experiments.py \
		--model-configs configs/experiments/nrms/nrms_glove.yaml \
		--checkpoints /Users/ploymel/Documents/plm4newsrs/outputs/politifact/nrms_model/glove_300_frozen/checkpoints/poisoned-epoch=01-val_auc=0.6636.ckpt \
		--data-path data/politifact/train_poisoned.csv \
		--ratios 0.05 0.10 \
		--num-trials 2 \
		--quick-test \
		--num-steps 3 \
		--use-label-correction \
		--learning-rate 0.01 \
		--save-summary

# Standard - 4 ratios × 3 trials (15 min)
experiment-unlearn-standard:
	@echo "Running standard unlearning experiment (4 ratios × 3 trials)..."
	python scripts/run_unlearning_experiments.py \
		--model-configs configs/experiments/simple/bert_finetune.yaml \
		--checkpoints outputs/simple_bert_finetune/checkpoints/poisoned.ckpt \
		--data-path data/politifact/train_poisoned.csv \
		--ratios 0.01 0.05 0.10 0.20 \
		--num-trials 3 \
		--save-summary

# Comprehensive - 5 ratios × 5 trials (30 min)
experiment-unlearn-comprehensive:
	@echo "Running comprehensive unlearning experiment (5 ratios × 5 trials)..."
	python scripts/run_unlearning_experiments.py \
		--model-configs configs/experiments/simple/bert_finetune.yaml \
		--checkpoints outputs/simple_bert_finetune/checkpoints/poisoned.ckpt \
		--data-path data/politifact/train_poisoned.csv \
		--ratios 0.01 0.05 0.10 0.20 0.50 \
		--num-trials 5 \
		--save-summary

# Custom experiment with parameters
experiment-unlearn-custom:
	@echo "Running custom unlearning experiment..."
	@echo "Usage: make experiment-unlearn-custom MODEL_CONFIG=... CHECKPOINT=... DATA_PATH=... RATIOS='0.05 0.10' TRIALS=3"
	python scripts/run_unlearning_experiments.py \
		--model-configs $(MODEL_CONFIG) \
		--checkpoints $(CHECKPOINT) \
		--data-path $(DATA_PATH) \
		--ratios $(RATIOS) \
		--num-trials $(TRIALS) \
		--save-summary

# Dry run - preview commands without executing
experiment-unlearn-dry-run:
	@echo "Dry run - printing commands without executing..."
	python scripts/run_unlearning_experiments.py \
		--model-configs configs/experiments/nrms/nrms_glove.yaml \
		--checkpoints /Users/ploymel/Documents/plm4newsrs/outputs/politifact/nrms_model/glove_300_frozen/checkpoints/poisoned-epoch=01-val_auc=0.6636.ckpt \
		--data-path data/politifact/train_poisoned.csv \
		--ratios 0.05 0.10 \
		--num-trials 2 \
		--dry-run

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