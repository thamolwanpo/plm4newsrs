# Makefile for PLM4NewsRS - training and evaluation on fakeun-dataprep data
# IMPORTANT: Use TAB for indentation, not spaces!
#
# Variables:
#   CONFIG     path to an experiment YAML (configs/experiments/...)
#   DATA_ROOT  fakeun-dataprep output folder, e.g. ../fakeun-dataprep/output/problem2/politifact
#              (default: data_root in the config, i.e. data/problem2/{dataset})
#   POISON_SET poisoned training set, e.g. A_r0.5_s1

DATA_ROOT_ARG = $(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

.PHONY: help train train-clean train-poisoned evaluate run clean clean-all install \
	train-simple-all train-nrms-all train-naml-all train-lstur-all train-lightgcn-all

help:
	@echo "Training (clean = stage3/train_clean_model.csv, poisoned = stage4/A_*/train_model.csv):"
	@echo "  make train CONFIG=...                         - Clean model + one poisoned model per A set"
	@echo "  make train-clean CONFIG=...                   - Clean model only"
	@echo "  make train-poisoned CONFIG=... POISON_SET=A_r0.5_s1"
	@echo "  make train-simple-all | train-nrms-all | train-naml-all | train-lstur-all | train-lightgcn-all"
	@echo ""
	@echo "Evaluation (all stage6 benchmarks):"
	@echo "  make evaluate CONFIG=...                      - Evaluate every trained model"
	@echo "  make run CONFIG=...                           - train + evaluate"
	@echo ""
	@echo "Add DATA_ROOT=path/to/output/problem2/<dataset> to any target to override the data folder."
	@echo ""
	@echo "Cleanup / install:"
	@echo "  make clean | clean-all | install"

# ========== Training ==========

train:
	python scripts/train.py --config $(CONFIG) --train-all $(DATA_ROOT_ARG)

train-clean:
	python scripts/train.py --config $(CONFIG) --model-type clean $(DATA_ROOT_ARG)

train-poisoned:
	python scripts/train.py --config $(CONFIG) --model-type poisoned --poison-set $(POISON_SET) $(DATA_ROOT_ARG)

train-simple-all:
	$(MAKE) train CONFIG=configs/experiments/simple/glove.yaml
	$(MAKE) train CONFIG=configs/experiments/simple/bert_finetune.yaml
	$(MAKE) train CONFIG=configs/experiments/simple/bert_frozen.yaml
	$(MAKE) train CONFIG=configs/experiments/simple/roberta_finetune.yaml
	$(MAKE) train CONFIG=configs/experiments/simple/roberta_frozen.yaml

train-nrms-all:
	$(MAKE) train CONFIG=configs/experiments/nrms/nrms_glove.yaml
	$(MAKE) train CONFIG=configs/experiments/nrms/nrms_bert_finetune.yaml
	$(MAKE) train CONFIG=configs/experiments/nrms/nrms_bert_frozen.yaml

train-naml-all:
	$(MAKE) train CONFIG=configs/experiments/naml/naml_glove.yaml
	$(MAKE) train CONFIG=configs/experiments/naml/naml_bert_finetune.yaml
	$(MAKE) train CONFIG=configs/experiments/naml/naml_bert_frozen.yaml

train-lstur-all:
	$(MAKE) train CONFIG=configs/experiments/lstur/lstur_glove.yaml
	$(MAKE) train CONFIG=configs/experiments/lstur/lstur_bert_finetune.yaml
	$(MAKE) train CONFIG=configs/experiments/lstur/lstur_bert_frozen.yaml

train-lightgcn-all:
	$(MAKE) train CONFIG=configs/experiments/lightgcn/lightgcn_bert_frozen_politifact.yaml
	$(MAKE) train CONFIG=configs/experiments/lightgcn/lightgcn_bert_finetune_politifact.yaml

# ========== Evaluation ==========

evaluate:
	python scripts/evaluate.py --config $(CONFIG) --generate-report $(DATA_ROOT_ARG)

run: train evaluate

# ========== Cleanup ==========

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
	rm -rf outputs/
	@echo "Cleaned all output directories"

# ========== Installation ==========

install:
	pip install -r requirements.txt
