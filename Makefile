DOCUMENT_VIEWER = "zathura"

.PHONY: sync
sync:
	uv sync

.PHONY: marimo
marimo:
	uv run marimo edit

.PHONY: test
test:
	uv run pytest

.PHONY: types
types:
	uv run ty check

.PHONY: lint
lint:
	uv run ruff check

.PHONY: format
format:
	uv run ruff format

.PHONY: check
check: test types lint format

.PHONY: typst
typst:
	typst watch typesetting/main.typ --open $(DOCUMENT_VIEWER)
