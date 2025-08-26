# Day 5: Tokenizer and Training Makefile

.PHONY: test train generate notebook clean help

help:
	@echo "Available commands:"
	@echo "  make test      - Run all tests"
	@echo "  make train     - Train mini language model"
	@echo "  make generate  - Run text generation demo"
	@echo "  make notebook  - Open Jupyter notebook"
	@echo "  make clean     - Clean cache files"

test:
	python tests/test_tokenizer_training.py

train:
	python 50_eval/train_mini_lm.py

generate:
	python 50_eval/generate_text.py

notebook:
	cd notebooks && jupyter notebook tokenizer_training_tutorial.ipynb

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -f training_curves.png