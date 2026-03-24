.PHONY: train eval test lint prepare-data visualize install

install:
	pip install -e ".[dev]"

train:
	python scripts/train.py --config configs/ashy_small.yaml

train-debug:
	python scripts/train.py --config configs/ashy_small_debug.yaml

train-baseline:
	python scripts/train.py --config configs/ashy_small_no_cer.yaml

eval:
	python scripts/eval.py --checkpoint $(CHECKPOINT) --battery cer

prepare-data:
	python scripts/prepare_data.py --dataset openwebtext --out-dir data/openwebtext

visualize:
	python scripts/visualize_cer.py --checkpoint $(CHECKPOINT) --text "$(TEXT)"

test:
	pytest tests/ -v

lint:
	ruff check ash/ tests/ scripts/
