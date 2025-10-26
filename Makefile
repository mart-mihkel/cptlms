.PHONY: setup
setup:
	uv sync

.PHONY: edit
edit:
	uv run marimo edit

.PHONY: format
format:
	uv run ruff format

.PHONY: lint
lint:
	uv run ruff check
