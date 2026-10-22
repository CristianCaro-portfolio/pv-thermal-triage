.PHONY: install data run smoke test lint figures

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev]"

data:
	python -m pvtriage download

run:
	python -m pvtriage run

smoke:
	python -m pvtriage smoke

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

figures:
	python -m pvtriage architecture
