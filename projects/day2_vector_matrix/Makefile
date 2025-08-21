.PHONY: help test clean install demo benchmark all

help:
	@echo "🔢 Day 2: Vector & Matrix Operations"
	@echo "======================================"
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test      - Run all tests"
	@echo "  make demo      - Run MNIST mini demo"
	@echo "  make benchmark - Compare scalar vs vector performance"
	@echo "  make clean     - Clean cache files"
	@echo "  make all       - Install, test, and run demos"

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	pip install -e ../..

test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

demo:
	@echo "🎯 Running MNIST mini demo..."
	python 50_eval/mnist_mini.py

benchmark:
	@echo "📊 Running performance benchmark..."
	python 50_eval/benchmark.py

clean:
	@echo "🧹 Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

all: install test demo benchmark
	@echo "✅ All tasks completed!"