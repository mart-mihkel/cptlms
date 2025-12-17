TB_LOG_DIR = out
DOCUMENT_VIEWER = zathura

sync:
	uv sync
	uv pip install --editable .

marimo:
	uv run marimo edit notebooks

test:
	@uv run pytest

types:
	@uv run ty check

lint:
	@uv run ruff check --fix

format:
	@uv run ruff format

check: format lint types test

tensorboard:
	uv run tensorboard --logdir $(TB_LOG_DIR)

typst:
	typst watch typesetting/main.typ --open $(DOCUMENT_VIEWER)

rsync:
	rsync -r --exclude-from '.gitignore' . $(REMOTE):git/cptlms
