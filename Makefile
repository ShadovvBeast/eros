# E.R.O.S Test and Development Commands

.PHONY: help test test-unit test-integration coverage coverage-html lint format install clean

help:  ## Show this help message
	@echo "E.R.O.S Development Commands:"
	@echo "=============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

test:  ## Run all tests with coverage
	python run_tests.py

test-unit:  ## Run only unit tests
	python run_tests.py --unit

test-integration:  ## Run only integration tests  
	python run_tests.py --integration

test-fast:  ## Run tests quickly with minimal output
	python run_tests.py --fast

coverage:  ## Generate coverage report
	python -m pytest tests --cov=src --cov-report=term-missing

coverage-html:  ## Generate HTML coverage report
	python run_tests.py --html
	@echo "Coverage report available at htmlcov/index.html"

lint:  ## Run linting checks
	flake8 src tests --max-line-length=100 --ignore=E203,W503
	black --check src tests
	isort --check-only src tests
	mypy src --ignore-missing-imports

format:  ## Format code with black and isort
	black src tests
	isort src tests

clean:  ## Clean up generated files
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

readiness:  ## Check system readiness
	python utils/system_readiness_check.py

gui:  ## Launch E.R.O.S GUI
	python main.py gui

demo:  ## Run demo scenarios
	python demos/simple_agent_test.py