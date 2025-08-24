# Day 4: Transformer Makefile

.PHONY: test demo clean help

help:
	@echo "Available commands:"
	@echo "  make test    - Run all tests"
	@echo "  make demo    - Run transformer demo"
	@echo "  make clean   - Clean cache files"

test:
	python tests/test_transformer.py

demo:
	python 50_eval/transformer_demo.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete