.PHONY: install test lint prepare-data train train-debug train-baseline eval visualize \
       ready-list ready train-model \
       pipeline-list pipeline-fetch pipeline-clean pipeline-run \
       modal-train modal-upload-data modal-status modal-checkpoints modal-download modal-cancel

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:
	pip install -e ".[dev]"

install-modal:
	pip install -e ".[dev,modal]"

install-s3:
	pip install -e ".[dev,modal,s3]"

# ---------------------------------------------------------------------------
# Prepare & Train (model-name workflow)
# ---------------------------------------------------------------------------

ready-list:
	python scripts/ready.py --list

# Usage: make ready MODEL=ashy-small
#        make ready MODEL=ashy-small-wikitext
#        make ready MODEL=debug
ready:
	python scripts/ready.py $(MODEL)

# Usage: make train-model MODEL=ashy-small
train-model:
	python scripts/ready.py $(MODEL) --train

# Prepare then train in one shot:
#   make ready-train MODEL=ashy-small
ready-train:
	python scripts/ready.py $(MODEL) --train

# ---------------------------------------------------------------------------
# Direct training (config-level, for dev/debug)
# ---------------------------------------------------------------------------

train:
	python scripts/train.py --config configs/ashy_small.yaml

train-debug:
	python scripts/train.py --config configs/ashy_small_debug.yaml

train-baseline:
	python scripts/train.py --config configs/ashy_small_no_cer.yaml

# ---------------------------------------------------------------------------
# Data (legacy prepare_data.py for simple datasets)
# ---------------------------------------------------------------------------

prepare-data:
	python scripts/prepare_data.py --dataset openwebtext --out-dir data/openwebtext

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

pipeline-list:
	python scripts/pipeline.py list

pipeline-fetch:
	python scripts/pipeline.py fetch --all --dry-run

pipeline-clean:
	python scripts/pipeline.py clean --all --dry-run

pipeline-run:
	python scripts/pipeline.py run --source $(SOURCE)

# ---------------------------------------------------------------------------
# Modal (cloud GPU)
# ---------------------------------------------------------------------------

modal-train:
	python scripts/train.py --config configs/ashy_small_modal.yaml

modal-upload-data:
	python -m ash.infra.modal_data --data-dir data/ --volume ash-data

modal-status:
	python -m ash.infra.modal_status status

modal-checkpoints:
	python -m ash.infra.modal_status checkpoints

modal-download:
	python -m ash.infra.modal_status download $(CHECKPOINT) --out-dir checkpoints/

modal-cancel:
	python -m ash.infra.modal_status cancel

# ---------------------------------------------------------------------------
# Eval & Visualize
# ---------------------------------------------------------------------------

eval:
	python scripts/eval.py --checkpoint $(CHECKPOINT) --battery cer

visualize:
	python scripts/visualize_cer.py --checkpoint $(CHECKPOINT) --text "$(TEXT)"

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

test:
	pytest tests/ -v

lint:
	ruff check ash/ tests/ scripts/
