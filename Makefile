.PHONY: help install check build docker-build docker-run clean

PYTHON ?= python3
VENV ?= .venv
BIN = $(VENV)/bin

help:
	@echo "Memora Framework - Command Menu"
	@echo "  make install       Install dependencies and Memora in editable mode"
	@echo "  make check         Execute deployment health check smoke test"
	@echo "  make build         Build wheel (.whl) and source dist (.tar.gz) in dist/"
	@echo "  make docker-build  Build production Docker image"
	@echo "  make docker-run    Run Memora container hardware check"
	@echo "  make clean         Clean build artifacts and temporary files"

install:
	$(BIN)/pip install --upgrade pip setuptools wheel
	$(BIN)/pip install -e .

check:
	$(BIN)/python scripts/deploy_check.py

build: clean
	$(BIN)/python -m build
	$(BIN)/twine check dist/*

docker-build:
	docker build -t memora:latest .

docker-run:
	docker run --rm memora:latest hardware

clean:
	rm -rf dist build *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

