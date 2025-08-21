.PHONY: help test clean install visualize translate all

help:
	@echo "🎯 Day 3: Attention Mechanism"
	@echo "======================================"
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test      - Run all tests"
	@echo "  make visualize - Run attention visualizer"
	@echo "  make translate - Run translation demo"
	@echo "  make clean     - Clean cache files"
	@echo "  make all       - Install, test, and run demos"

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	pip install -e ../..

test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

visualize:
	@echo "👁️ Running attention visualizer..."
	python 50_eval/attention_visualizer.py

translate:
	@echo "🌐 Running translation demo..."
	python 50_eval/translation_toy.py

clean:
	@echo "🧹 Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

all: install test visualize translate
	@echo "✅ All tasks completed!"