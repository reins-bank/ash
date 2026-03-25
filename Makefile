.PHONY: train eval test lint prepare-data visualize install pipeline-list pipeline-fetch pipeline-clean pipeline-run modal-train modal-upload-data

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

pipeline-list:
	python scripts/pipeline.py list

pipeline-fetch:
	python scripts/pipeline.py fetch --all --dry-run

pipeline-clean:
	python scripts/pipeline.py clean --all --dry-run

pipeline-run:
	python scripts/pipeline.py run --source $(SOURCE)

lint:
	ruff check ash/ tests/ scripts/

modal-train:
	python scripts/train.py --config configs/ashy_small_modal.yaml

modal-upload-data:
	python -m ash.infra.modal_data --data-dir data/ --volume ash-data
