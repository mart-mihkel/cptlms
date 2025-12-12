DOCUMENT_VIEWER = zathura

.PHONY: sync
sync:
	uv sync
	uv pip install --editable .

.PHONY: marimo
marimo:
	uv run marimo edit notebooks

.PHONY: test
test:
	@uv run pytest

.PHONY: types
types:
	@uv run ty check

.PHONY: lint
lint:
	@uv run ruff check --fix

.PHONY: format
format:
	@uv run ruff format

.PHONY: check
check: format lint types test

.PHONY: typst
typst:
	typst watch typesetting/main.typ --open $(DOCUMENT_VIEWER)

.PHONY: rsync
rsync:
	rsync -rv --exclude-from '.gitignore' . $(REMOTE):git/cptlms
