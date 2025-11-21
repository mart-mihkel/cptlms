.PHONY: sync
sync:
	uv sync

.PHONY: marimo
marimo: sync
	uv run marimo edit

.PHONY: test
test: sync
	uv run pytest

.PHONY: format
format: sync
	uv run ruff format

.PHONY: format-check
format-check: sync
	uv run ruff format --check

.PHONY: lint
lint: sync
	uv run ruff check

.PHONY: types
types: sync
	uv run ty check

.PHONY: check
check: test format-check lint types

.PHONY: typst
typst:
	typst compile typesetting/main.typ

.PHONY: typst-watch
typst-watch:
	typst watch typesetting/main.typ --open zathura
