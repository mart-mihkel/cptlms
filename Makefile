.PHONY: setup
setup:
	uv sync

.PHONY: marimo
marimo: setup
	uv run marimo edit

.PHONY: format
format: setup
	uv run ruff format

.PHONY: lint
lint: setup
	uv run ruff check

.PHONY: check
check: setup
	uv run ty check

.PHONY: watch
watch:
	typst watch thesis/main.typ --open zathura
