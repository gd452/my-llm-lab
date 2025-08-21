setup:
	pip install -U pip && pip install -r requirements.txt

test:
	python -m pytest tests/ -v

test-skip:
	python -m pytest tests/ -v -k "not skip"

xor:
	python 50_eval/xor_demo.py

notebook:
	jupyter notebook notebooks/nn_tutorial.ipynb

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

help:
	@echo "Available commands:"
	@echo "  make setup    - Install dependencies"
	@echo "  make test     - Run all tests"
	@echo "  make xor      - Run XOR demo"
	@echo "  make notebook - Open tutorial notebook"
	@echo "  make clean    - Clean cache files"