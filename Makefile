# Makefile for training news recommendation models
# IMPORTANT: Use TAB for indentation, not spaces!

.PHONY: help train train-simple train-naml train-nrms train-all train-sweep clean test

help:
	@echo "Available commands:"
	@echo ""
	@echo "Training:"
	@echo "  make train CONFIG=path/to/config.yaml  - Train with specific config"
	@echo "  make train-simple                         - Train simple model (BERT fine-tune)"
	@echo "  make train-simple-frozen                  - Train simple model (BERT frozen)"
	@echo "  make train-simple-glove                   - Train simple model (GloVe)"
	@echo "  make train-naml                         - Train NAML (BERT)"
	@echo "  make train-naml-glove                   - Train NAML (GloVe)"
	@echo "  make train-nrms                         - Train NRMS (BERT)"
	@echo "  make train-nrms-glove                   - Train NRMS (GloVe)"
	@echo "  make train-all                          - Train all architectures"
	@echo "  make train-sweep                        - Run hyperparameter sweep"
	@echo ""
	@echo "Testing:"
	@echo "  make test                               - Run all tests"
	@echo "  make test-fast                          - Run fast tests only"
	@echo "  make test-cov                           - Run tests with coverage"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                              - Clean cache files"
	@echo "  make clean-all                          - Clean everything (cache + outputs)"

# ========== Training Commands ==========

train:
	python scripts/train.py --config $(CONFIG)

train-simple:
	python scripts/train.py --config configs/experiments/simple/bert_finetune.yaml

train-simple-frozen:
	python scripts/train.py --config configs/experiments/simple/bert_frozen.yaml

train-simple-glove:
	python scripts/train.py --config configs/experiments/simple/glove.yaml

train-naml:
	python scripts/train.py --config configs/experiments/naml/naml_bert.yaml

train-naml-glove:
	python scripts/train.py --config configs/experiments/naml/naml_glove.yaml

train-nrms:
	python scripts/train.py --config configs/experiments/nrms/nrms_bert.yaml

train-nrms-glove:
	python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml

# ========== Testing Commands ==========

test:
	pytest

test-fast:
	pytest -m "not slow"

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

# ========== Cleanup Commands ==========

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	@echo "Cleaned Python cache files"

clean-all: clean
	rm -rf htmlcov/
	rm -rf logs/
	rm -rf checkpoints/
	rm -rf results/
	@echo "Cleaned all output directories"

# ========== Installation ==========

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-test.txt

# ========== Quick Examples ==========

example-bert:
	@echo "Training simple model with BERT fine-tuning..."
	python scripts/train.py \
		--config configs/experiments/simple/bert_finetune.yaml \
		--epochs 2

example-glove:
	@echo "Training simple model with GloVe..."
	python scripts/train.py \
		--config configs/experiments/simple/glove.yaml \
		--epochs 2

example-comparison:
	@echo "Comparing all architectures..."
	python scripts/train_all_architectures.py --epochs 2